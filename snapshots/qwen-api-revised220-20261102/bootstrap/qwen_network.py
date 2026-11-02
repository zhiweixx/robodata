"""Private userspace Tailscale lifecycle and fixed-destination HTTP transport.

Only an already-enrolled state is used. Enrollment credentials are never read by
this evaluation component. Each allocation owns the enrollment lock and a private
node-local socket; no host VPN, routes, DNS, or shared driver is changed.
"""
from __future__ import annotations

import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import threading
import time

PRIVATE = Path('/home/ROBODATA_USER/.config/professor-api-pilot')
HOST = 'qwen-api.example.internal'
PORT = 8000
MODEL = 'qwen38-27b'
BINARY_SHA256 = {
    'tailscale': 'f27bd0ad0a9a04c23ae02aa8a5897cf7f99e4e5ae153c0e5f7000b122418c830',
    'tailscaled': 'afc601d3c5be812105064eba11b64afa8928e983815bc47db218f25f3749c0c0',
}


def require_private(path, directory=False):
    info = Path(path).lstat()
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    mode = 0o700 if directory else 0o600
    if not kind(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != mode:
        raise ValueError('private path must be nonsymlink, user-owned and mode0700/0600')


class PipeSocket:
    def __init__(self, stream):
        self.stream = stream

    def makefile(self, mode):
        if mode != 'rb':
            raise ValueError('response stream must be read-only')
        return self.stream


class NCConnection:
    """HTTPConnection-shaped adapter. The caller owns response-byte archival."""
    sock = None

    def __init__(self, network, *, stage=lambda _: None, popen=subprocess.Popen):
        self.network, self.stage, self.popen = network, stage, popen
        self.proc = None
        self._lock = threading.Lock()
        self._closed = False
        self.reaped = False
        self.returncode = None
        self.response = None

    def connect(self):
        self.network.assert_running()
        self.stage('tailscale_nc_start')
        with self._lock:
            if self._closed or self.proc is not None:
                raise RuntimeError('connection closed or already connected')
            self.proc = self.popen([
                str(self.network.private / 'bin/tailscale'),
                '--socket=' + str(self.network.socket), 'nc', HOST, str(PORT)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                start_new_session=False)

    def request(self, method, path, body=None, headers=None):
        if (method, path) not in (('GET', '/v1/models'), ('POST', '/v1/chat/completions')):
            raise ValueError('fixed professor destination/path required')
        if self.proc is None or self._closed:
            raise RuntimeError('connection is not open')
        body = b'' if body is None else body
        if not isinstance(body, bytes):
            raise TypeError('request bytes required')
        safe = dict(headers or {})
        if any(k.lower() in ('authorization', 'proxy-authorization') for k in safe):
            raise ValueError('provider transport must not receive credentials')
        safe.update({'Host': f'{HOST}:{PORT}', 'Content-Length': str(len(body)), 'Connection': 'close'})
        for name, value in safe.items():
            if any(c in name + str(value) for c in '\r\n'):
                raise ValueError('invalid HTTP header')
        packet = (f'{method} {path} HTTP/1.1\r\n' + ''.join(
            f'{k}: {v}\r\n' for k, v in safe.items()) + '\r\n').encode('latin1') + body
        # Keep stdin open: tailscale nc exits on stdin EOF before the HTTP reply.
        self.proc.stdin.write(packet)
        self.proc.stdin.flush()

    def getresponse(self):
        if self.proc is None or self._closed:
            raise RuntimeError('connection is not open')
        self.response = http.client.HTTPResponse(PipeSocket(self.proc.stdout))
        self.response.begin()
        return self.response

    def close(self):
        with self._lock:
            self._closed = True
            proc = self.proc
        if proc is None:
            self.reaped = True
            return
        # Stop the child first so blocked pipe readers/writers can exit.
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        self.returncode, self.reaped = proc.returncode, proc.poll() is not None
        for stream in (proc.stdin, proc.stdout):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass


class PrivateTailnet:
    def __init__(self, runtime_dir, *, private=PRIVATE):
        self.private, self.runtime = Path(private), Path(runtime_dir)
        self.socket = self.runtime / 'tailscaled.sock'
        self.daemon = self.lock = None
        self.proof = None
        self.runtime_owned = False

    def verify_files(self):
        require_private(self.private, directory=True)
        require_private(self.private / 'tailscaled.state')
        hashes = {}
        for name, expected in BINARY_SHA256.items():
            path = self.private / 'bin' / name
            info = path.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) != 0o700):
                raise ValueError('private Tailscale executable ownership/mode invalid')
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashes[name] != expected:
                raise ValueError('pinned Tailscale executable changed')
        return hashes

    def start(self, *, stop_requested=lambda: False):
        hashes = self.verify_files()
        path = self.private / 'state.lock'
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        self.lock = os.fdopen(fd, 'a+')
        try:
            require_private(path)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.runtime.mkdir(mode=0o700, parents=False, exist_ok=False)
            self.runtime_owned = True
            require_private(self.runtime, directory=True)
            log = self.private / 'tailscaled-official-qwen.log'
            logfd = os.open(log, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            require_private(log)
            with os.fdopen(logfd, 'ab') as stream:
                self.daemon = subprocess.Popen([
                    str(self.private / 'bin/tailscaled'), '--tun=userspace-networking',
                    '--state=' + str(self.private / 'tailscaled.state'),
                    '--socket=' + str(self.socket), '--port=0'],
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if stop_requested() or self.daemon.poll() is not None:
                    raise RuntimeError('private Tailscale startup stopped')
                if self.socket.exists():
                    status = subprocess.run([str(self.private / 'bin/tailscale'),
                        '--socket=' + str(self.socket), 'status', '--json'],
                        capture_output=True, timeout=5)
                    try:
                        state = json.loads(status.stdout).get('BackendState')
                    except (ValueError, AttributeError):
                        raise RuntimeError('private Tailscale status malformed') from None
                    if state == 'NeedsLogin':
                        raise RuntimeError('private enrollment needs authorized restoration')
                    if state == 'Running':
                        break
                time.sleep(.25)
            else:
                raise RuntimeError('private Tailscale startup timed out')
            self.assert_running()
            started = time.monotonic()
            connection = NCConnection(self)
            # Metadata-only deadline; generation requests retain native policy deadlines.
            timer = threading.Timer(30, connection.close)
            timer.daemon = True
            timer.start()
            try:
                connection.connect()
                connection.request('GET', '/v1/models')
                response = connection.getresponse()
                raw = response.read(65537)
                if response.status != 200 or len(raw) > 65536:
                    raise RuntimeError('professor model metadata unavailable')
                value = json.loads(raw)
                ids = [x.get('id') for x in value.get('data', []) if isinstance(x, dict)]
                if MODEL not in ids:
                    raise RuntimeError('exact professor model alias unavailable')
            finally:
                timer.cancel()
                connection.close()
            self.proof = {'transport': 'private_rootless_tailscale_nc',
                'model_alias': MODEL, 'models_http_status': 200, 'exact_model_available': True,
                'models_probe_wall_seconds': time.monotonic() - started,
                'tailscale_binaries_sha256': hashes, 'backend_state': 'Running',
                'existing_enrollment': True, 'enrollment_key_read': False,
                'provider_api_key_read': False, 'system_vpn_dns_routes_changed': False,
                'models_probe_child_reaped': connection.reaped}
            return dict(self.proof)
        except BaseException:
            self.close()
            raise

    def assert_running(self):
        if self.daemon is None or self.daemon.poll() is not None:
            raise RuntimeError('private Tailscale daemon is not running')
        info = self.socket.lstat()
        require_private(self.runtime, directory=True)
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
            raise RuntimeError('private Tailscale socket invalid')

    def close(self):
        reaped = self.daemon is None
        if self.daemon is not None:
            if self.daemon.poll() is None:
                self.daemon.terminate()
            try:
                self.daemon.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.daemon.kill()
                self.daemon.wait(timeout=5)
            reaped = self.daemon.poll() is not None
        if self.runtime_owned and self.runtime.exists():
            shutil.rmtree(self.runtime)
        if self.lock is not None:
            self.lock.close()
            self.lock = None
        return {'daemon_reaped': reaped, 'private_socket_removed': not self.runtime.exists(),
                'state_lock_released': self.lock is None, 'network_cleanup_verified':
                reaped and not self.runtime.exists() and self.lock is None}


class ExistingTailnet:
    """Controller view of the supervisor-owned private socket; no state/key reads."""
    def __init__(self, socket_path):
        self.socket = Path(socket_path)
        self.private = PRIVATE
        self.assert_running()

    def assert_running(self):
        if not self.socket.is_absolute():
            raise ValueError('absolute private Unix socket required')
        require_private(self.socket.parent, directory=True)
        info = self.socket.lstat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError('supervisor-owned Tailscale socket missing')
