"""Private, sequential file relay between a credential-free simulator and Astra.

The trusted launcher starts model_driver.py --config ... --relay-dir ... outside
the simulator. Simulator processes only construct RelayModelClient. Generated
programs never receive this client or its directory. Files are retained as an
audit trail. A response timeout NEVER retries an inference.

This is process/credential separation, not an OS security sandbox against other
processes running as the same user. ASPIRE's restricted program API still applies.
"""

from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import time
import uuid

from model_driver import DriverStopped, ProgramProposal


_REQUEST_KEYS = frozenset({"episode_id", "task_instruction", "public_observation",
                           "previous_result", "api_documentation"})


def _private_directory(path):
    path = Path(path).absolute()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise DriverStopped("relay directory must be owned by this user with mode 700")
    return path


def _write_json(path, value):
    path = Path(path)
    data = (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode()
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise DriverStopped("relay output already exists; refusing overwrite/retry")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise DriverStopped("relay files must be private regular files")
        if info.st_size > 150_000_000:
            raise DriverStopped("relay message exceeds size limit")
        return json.load(handle)


class RelayModelClient:
    """Credential-free trusted-runner helper; one request outstanding per suite."""

    def __init__(self, relay_dir, *, timeout_s=480):
        self.path = Path(relay_dir).absolute()
        self.timeout_s = float(timeout_s)
        self.stopped = False
        if not 1 <= self.timeout_s <= 3600:
            raise ValueError("relay timeout must be between 1 and 3600 seconds")

    def wait_ready(self):
        started = time.monotonic()
        while not (self.path / "ready.json").exists():
            if (self.path / "stopped.json").exists() or time.monotonic() - started > self.timeout_s:
                self.stopped = True
                raise DriverStopped("API controller is unavailable; do not retry a measured suite")
            time.sleep(0.1)
        _private_directory(self.path)
        if (self.path / "stopped.json").exists():
            self.stopped = True
            raise DriverStopped("API controller has stopped")
        return _read_json(self.path / "ready.json")

    def _request(self, payload):
        if self.stopped:
            raise DriverStopped("relay client has stopped")
        self.wait_ready()
        request_dir = _private_directory(self.path / "requests")
        response_dir = _private_directory(self.path / "responses")
        if any(not (response_dir / p.name).exists() for p in request_dir.glob("*.json")):
            self.stopped = True
            raise DriverStopped("another relay request is outstanding; no parallel or retried calls")
        request_id = uuid.uuid4().hex
        _write_json(request_dir / (request_id + ".json"), {"request_id": request_id, "payload": payload})
        started = time.monotonic()
        response_path = response_dir / (request_id + ".json")
        while not response_path.exists():
            if (self.path / "stopped.json").exists() or time.monotonic() - started > self.timeout_s:
                self.stopped = True
                raise DriverStopped("relay response unavailable/unknown; stop and never resend")
            time.sleep(0.1)
        response = _read_json(response_path)
        if response.get("request_id") != request_id:
            self.stopped = True
            raise DriverStopped("relay response identifier mismatch")
        if response.get("ok") is not True:
            self.stopped = True
            raise DriverStopped("API controller stopped this suite; do not retry")
        return response

    def next_program(self, episode_id, task_instruction, public_observation,
                     previous_result=None, api_documentation=None):
        response = self._request({
            "episode_id": episode_id, "task_instruction": task_instruction,
            "public_observation": public_observation, "previous_result": previous_result,
            "api_documentation": api_documentation,
        })
        return ProgramProposal(**response["proposal"])

    def stop(self):
        response = self._request({"command": "stop"})
        self.stopped = True
        return response


def serve_relay(driver, relay_dir, *, idle_timeout_s=1800):
    """Run in main thread: the reviewed API client's alarm deadline requires it."""
    path = _private_directory(relay_dir)
    requests = _private_directory(path / "requests")
    responses = _private_directory(path / "responses")
    lock_fd = os.open(path / "controller.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DriverStopped("another API controller owns this relay") from None
        if any(requests.glob("*.json")) or (path / "ready.json").exists():
            raise DriverStopped("relay was already started; never replay old requests")
        _write_json(path / "ready.json", {
            "ready": True, "episode_ids": list(driver.client.episode_ids),
            "max_iterations": driver.max_iterations, "budget": driver.client.snapshot(),
        })
        last_activity = time.monotonic()
        while True:
            pending = [p for p in requests.glob("*.json") if not (responses / p.name).exists()]
            if not pending:
                if time.monotonic() - last_activity > idle_timeout_s:
                    _write_json(path / "stopped.json", {"reason": "controller_idle_timeout", "no_retry": True})
                    return
                time.sleep(0.1)
                continue
            if len(pending) != 1:
                _write_json(path / "stopped.json", {"reason": "parallel_requests_forbidden", "no_retry": True})
                return
            request_path = pending[0]
            request_id = request_path.stem
            stop = False
            try:
                if not re.fullmatch(r"[0-9a-f]{32}", request_id):
                    raise ValueError("invalid relay request filename")
                envelope = _read_json(request_path)
                if set(envelope) != {"request_id", "payload"} or envelope["request_id"] != request_id:
                    raise ValueError("invalid relay request envelope")
                payload = envelope["payload"]
                if payload == {"command": "stop"}:
                    response = {"ok": True, "stopped": True, "budget": driver.client.snapshot()}
                    stop = True
                else:
                    if not isinstance(payload, dict) or set(payload) - _REQUEST_KEYS:
                        raise ValueError("invalid driver request")
                    proposal = driver.next_program(**payload)
                    response = {"ok": True, "proposal": asdict(proposal)}
            except BaseException as exc:
                response = {"ok": False, "error_type": type(exc).__name__, "stop_suite": True, "no_retry": True}
                stop = True
            response["request_id"] = request_id
            _write_json(responses / request_path.name, response)
            last_activity = time.monotonic()
            if stop:
                _write_json(path / "stopped.json", {"reason": "controller_stopped", "no_retry": True})
                return
    finally:
        os.close(lock_fd)
