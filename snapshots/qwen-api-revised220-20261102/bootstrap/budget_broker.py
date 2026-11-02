"""Single-writer budget RPC; workers never flock the durable NFS ledger.

Only this broker instantiates BudgetLedger. It holds a separate, nonblocking
exclusive lifetime lock on the durable filesystem, then serializes ledger calls
with an in-process RLock. Its Unix socket must be on private node-local storage.
Generation is performed by clients AFTER reserve returns, outside every broker
lock. A restarted broker marks every old active reservation unknown before it
listens. No inference, secrets, scheduling, native scoring, or automatic retries.

The lifetime ownership lock depends on coherent filesystem flock semantics. A
deployment must validate the competing-broker test on its durable filesystem;
metadata saying a PID is dead never authorizes breaking that lock. All writers
must use this broker, not BudgetLedger directly. Price estimates are not invoices.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import socketserver
import stat
import struct
import sys
import threading
import time
import uuid

from budget_ledger import (AccountingError, AdmissionBlocked, BudgetLedger,
                           BudgetUnavailable, LedgerCorrupt, LedgerError, canonical)

MAX_RPC_BYTES = 1_048_576
RPC_TIMEOUT_SECONDS = 10
PROTOCOL_VERSION = 1


class BrokerError(RuntimeError):
    pass


class BrokerOwnershipError(BrokerError):
    pass


class BrokerUnavailable(BrokerError):
    """No send permission: pause unscored work; do not retry generation."""


def _strict_json(data):
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON value")
    return json.loads(data, object_pairs_hook=no_duplicates, parse_constant=invalid_constant)


def _mount_type(path):
    """Find the containing Linux mount; macOS tests use private local /tmp."""
    if not sys.platform.startswith("linux"):
        return None
    best = None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields = left.split()
        mount = fields[4]
        for before, after in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
            mount = mount.replace(before, after)
        parent = Path(mount)
        if path == parent or parent in path.parents:
            if best is None or len(mount) > best[0]:
                best = (len(mount), right.split()[0])
    if best is None:
        raise BrokerError("Cannot establish socket filesystem type")
    return best[1]


def _private_socket_dir(path, *, create):
    path = Path(path)
    if not path.is_absolute():
        raise BrokerError("Socket directory must be absolute")
    if path.is_symlink():
        raise BrokerError("Socket directory cannot be a symlink")
    if create:
        path.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise BrokerError("Socket directory must be owned by this user and mode 0700")
    resolved = path.resolve(strict=True)
    fs = _mount_type(resolved)
    if sys.platform.startswith("linux"):
        # Fail closed on unknown/network filesystems; add a type only after review.
        if fs not in {"tmpfs", "ramfs", "ext2", "ext3", "ext4", "xfs", "btrfs", "zfs", "overlay"}:
            raise BrokerError("Unix socket requires a verified local filesystem, got " + str(fs))
    else:
        # This branch is for local macOS CPU tests, not Lighthouse deployment.
        local_tmp = Path("/tmp").resolve()
        if resolved != local_tmp and local_tmp not in resolved.parents:
            raise BrokerError("Non-Linux socket tests must use a private directory under /tmp")
    return resolved


def _read_line(sock):
    data = bytearray()
    while len(data) <= MAX_RPC_BYTES:
        chunk = sock.recv(min(65536, MAX_RPC_BYTES + 1 - len(data)))
        if not chunk:
            raise BrokerUnavailable("RPC closed before a complete response")
        data.extend(chunk)
        if b"\n" in chunk:
            break
    if len(data) > MAX_RPC_BYTES or not data.endswith(b"\n") or data.count(b"\n") != 1:
        raise BrokerError("Malformed or oversized RPC record")
    return _strict_json(bytes(data))


def _send_line(sock, value):
    data = (canonical(value) + "\n").encode()
    if len(data) > MAX_RPC_BYTES:
        raise BrokerError("RPC result exceeds transport limit")
    sock.sendall(data)


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 64  # Cover simultaneous nine-worker reserve/settle bursts.


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        broker = self.server.broker
        self.request.settimeout(RPC_TIMEOUT_SECONDS)
        reserved_id = None
        try:
            if hasattr(socket, "SO_PEERCRED"):
                _, peer_uid, _ = struct.unpack("3i", self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if peer_uid != os.getuid():
                    raise BrokerError("Peer UID is not the broker owner")
            request = _read_line(self.request)
            result = broker._dispatch(request)
            if request["op"] == "reserve":
                reserved_id = request["kwargs"]["request_id"]
            _send_line(self.request, {"version": PROTOCOL_VERSION, "ok": True, "result": result})
        except Exception as exc:
            # A lost reservation reply is not proof that a client could not send.
            if reserved_id is not None:
                broker._uncertain_delivery(reserved_id)
            try:
                _send_line(self.request, {"version": PROTOCOL_VERSION, "ok": False,
                                          "error": {"type": type(exc).__name__, "message": str(exc)}})
            except Exception:
                pass


class BudgetBroker:
    def __init__(self, durable_path, socket_dir, *, cap_usd="2000"):
        self.durable_path = Path(durable_path).absolute()
        self.socket_dir = Path(socket_dir)
        self.cap_usd = cap_usd
        self.owner_path = self.durable_path.with_name(self.durable_path.name + ".broker-owner.lock")
        self._owner_fd = None
        self._owner_nonce = str(uuid.uuid4())
        self._tx = threading.RLock()
        self._server = self._thread = self._ledger = None
        self._socket_identity = None
        self._closing = False
        self.socket_path = None

    def _claim_owner(self):
        self.durable_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.owner_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise BrokerOwnershipError("Another broker owns this durable ledger") from exc
            record = {"host": socket.gethostname(), "pid": os.getpid(), "nonce": self._owner_nonce,
                      "started_unix": time.time(), "scope": "single_writer_budget_broker"}
            os.ftruncate(fd, 0)
            os.write(fd, (canonical(record) + "\n").encode())
            os.fsync(fd)
            self._owner_fd = fd
        except Exception:
            os.close(fd)
            raise

    def _recover_unsettled(self, reason):
        # The lifetime ownership lock and _tx enclose these existing ledger
        # internals. No worker imports/locks the durable ledger in broker mode.
        with self._ledger._lock():
            state = self._ledger._read()
            active = [rid for rid, request in state["requests"].items() if request["state"] == "active"]
            for rid in active:
                state = self._ledger._append(state, {"kind": "unknown", "request_id": rid,
                                                    "reason": reason, "raw_usage": None,
                                                    "recorded_unix": time.time()})
        return len(active)

    def start(self):
        if self._owner_fd is not None or self._server is not None:
            raise BrokerError("Broker has already started")
        self._claim_owner()
        try:
            directory = _private_socket_dir(self.socket_dir, create=True)
            self.socket_path = directory / "b.sock"
            if len(os.fsencode(self.socket_path)) >= 104:
                raise BrokerError("Private Unix socket path is too long")
            if os.path.lexists(self.socket_path):
                raise BrokerError("Socket path already exists; will not unlink another endpoint")
            with self._tx:
                self._ledger = BudgetLedger(self.durable_path, cap_usd=self.cap_usd, single_writer_cache=True)
                recovered = self._recover_unsettled("broker_restarted_without_terminal_usage")
            self._server = _Server(str(self.socket_path), _Handler)
            self._server.broker = self
            info = self.socket_path.lstat()
            self._socket_identity = (info.st_dev, info.st_ino)
            os.chmod(self.socket_path, 0o600)
            self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05},
                                            name="terra-budget-broker", daemon=True)
            self._thread.start()
            return {"socket_path": str(self.socket_path), "durable_path": str(self.durable_path),
                    "recovered_unknown_requests": recovered}
        except Exception:
            self.stop()
            raise

    def _dispatch(self, request):
        if (not isinstance(request, dict) or set(request) != {"version", "op", "kwargs"}
                or request["version"] != PROTOCOL_VERSION or not isinstance(request["kwargs"], dict)):
            raise BrokerError("Invalid RPC envelope")
        op = request["op"]
        if op not in {"reserve", "settle", "mark_unknown", "status", "pause_admission"}:
            raise BrokerError("Unknown RPC operation")
        with self._tx:
            if self._closing or self._owner_fd is None:
                raise BrokerUnavailable("Broker is closing; do not send generation requests")
            return getattr(self._ledger, op)(**request["kwargs"])

    def authorize_resume_with_unknown_exposure(self, authorization, *, approval_record_path):
        """Trusted owner-only operator action; deliberately absent from RPC.

        Call only after independently checking explicit human consent against
        the saved approval artifact. No worker/client can invoke this via the
        budget socket; this does not send a request or retry an episode.
        """
        with self._tx:
            if self._closing or self._owner_fd is None or self._ledger is None:
                raise BrokerUnavailable('An active exclusive broker owner is required')
            self._verify_human_approval_artifact(authorization, approval_record_path)
            return self._ledger.authorize_resume_with_unknown_exposure(authorization)

    def cache_integrity_diagnostics(self):
        """Read bounded owner-only evidence without changing financial status.

        Deliberately not in the worker RPC allowlist. The supervisor may archive
        this alongside allocation cleanup evidence, without touching the ledger.
        """
        with self._tx:
            if self._ledger is None:
                return None
            return self._ledger.cache_integrity_diagnostics()

    @staticmethod
    def _verify_human_approval_artifact(authorization, approval_record_path):
        """Verify preserved bytes and exact scope; consent still comes from user.

        A hash verifies artifact integrity, not human identity. The trusted
        owner must record actual user consent; generated workers cannot use
        this method or manufacture an authorization through the RPC endpoint.
        """
        path = Path(approval_record_path)
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
                     | getattr(os, 'O_NONBLOCK', 0))
        with os.fdopen(fd, 'rb') as file:
            info = os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_size > 65536:
                raise BrokerError('Approval evidence must be an owned bounded regular file')
            data = file.read(65537)
            if len(data) > 65536:
                raise BrokerError('Approval evidence exceeds the bounded record limit')
        if hashlib.sha256(data).hexdigest() != authorization.get('approval_record_sha256'):
            raise BrokerError('Actual human approval artifact differs from the bound digest')
        record = _strict_json(data)
        scope_keys = {'schema_version', 'authority', 'decision', 'approval_id', 'ledger_id',
                      'expected_head_sha256', 'request_ids', 'expected_pause_reason', 'cap_usd', 'reason'}
        scope = {key: authorization.get(key) for key in scope_keys}
        if (not isinstance(record, dict) or record.get('human_approval_recorded') is not True
                or record.get('authorization_scope') != scope
                or not isinstance(record.get('human_message'), str)
                or not 1 <= len(record['human_message']) <= 4096):
            raise BrokerError('Preserved approval record does not bind this exact human-approved scope')

    def _uncertain_delivery(self, request_id):
        with self._tx:
            if self._closing or self._owner_fd is None:
                return
            self._ledger.mark_unknown(request_id, reason="broker_reservation_reply_delivery_uncertain")

    def stop(self):
        with self._tx:
            self._closing = True
        if self._server is not None:
            if self._thread is not None and self._thread.is_alive():
                self._server.shutdown()
                self._thread.join(timeout=5)
            self._server.server_close()
            self._server = None
        try:
            with self._tx:
                if self._ledger is not None and self._owner_fd is not None:
                    self._recover_unsettled("broker_stopped_with_unsettled_requests")
        finally:
            if self.socket_path is not None and self._socket_identity is not None:
                try:
                    info = self.socket_path.lstat()
                    if (info.st_dev, info.st_ino) == self._socket_identity and stat.S_ISSOCK(info.st_mode):
                        self.socket_path.unlink()
                except FileNotFoundError:
                    pass
                self._socket_identity = None
            if self._owner_fd is not None:
                fd, self._owner_fd = self._owner_fd, None
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


_REMOTE_ERRORS = {cls.__name__: cls for cls in (
    AccountingError, AdmissionBlocked, BudgetUnavailable, LedgerCorrupt, LedgerError,
    BrokerError, BrokerOwnershipError, BrokerUnavailable, ValueError, TypeError)}


class BudgetClient:
    """One connection per metadata RPC. No network generation, no retries."""
    def __init__(self, socket_path, *, timeout=RPC_TIMEOUT_SECONDS):
        self.socket_path = Path(socket_path)
        self.timeout = timeout

    def _call(self, op, kwargs):
        _private_socket_dir(self.socket_path.parent, create=False)
        info = self.socket_path.lstat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
            raise BrokerUnavailable("Budget endpoint must be an owned mode-0600 Unix socket")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect(str(self.socket_path))
                _send_line(sock, {"version": PROTOCOL_VERSION, "op": op, "kwargs": kwargs})
                response = _read_line(sock)
        except (OSError, TimeoutError) as exc:
            raise BrokerUnavailable("Budget RPC interrupted; no retry or generation permission") from exc
        if not isinstance(response, dict) or response.get("version") != PROTOCOL_VERSION:
            raise BrokerUnavailable("Invalid broker response")
        if response.get("ok") is True and set(response) == {"version", "ok", "result"}:
            return response["result"]
        error = response.get("error", {})
        error_type = _REMOTE_ERRORS.get(error.get("type"), BrokerError)
        raise error_type(error.get("message", "Unknown budget RPC failure"))

    def reserve(self, request_id, **kwargs):
        return self._call("reserve", {"request_id": request_id, **kwargs})

    def settle(self, request_id, **kwargs):
        return self._call("settle", {"request_id": request_id, **kwargs})

    def mark_unknown(self, request_id, **kwargs):
        return self._call("mark_unknown", {"request_id": request_id, **kwargs})

    def status(self):
        return self._call("status", {})

    def pause_admission(self, reason):
        return self._call('pause_admission', {'reason': reason})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--durable-path", required=True)
    parser.add_argument("--socket-dir", required=True)
    parser.add_argument("--cap-usd", default="2000")
    args = parser.parse_args()
    broker = BudgetBroker(args.durable_path, args.socket_dir, cap_usd=args.cap_usd)
    stop_event = threading.Event()
    def request_stop(signum, frame):
        stop_event.set()
    old_handlers = {signum: signal.signal(signum, request_stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        print(json.dumps(broker.start()), flush=True)
        stop_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            broker.stop()
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    main()
