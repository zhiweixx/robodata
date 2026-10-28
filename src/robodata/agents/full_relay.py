"""Private sequential local-inference relay, with no absolute request deadline.

One relay/controller belongs to one simulator worker. A known policy end only
ends that episode; infrastructure errors stop the worker without inventing a
score. Requests, responses, and model streams are retained for audit. There are
no credentials, remote services, automatic inference retries, or pilot timers.
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

from robodata.agents.full_model_driver import (InferenceInfrastructureError, ModelEpisodeFailure,
                               ProgramProposal)

_REQUEST_KEYS = frozenset({"episode_id", "task_instruction", "public_observation",
                           "previous_result", "api_documentation"})


def _private_directory(path):
    path = Path(path).absolute()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise InferenceInfrastructureError("relay_permissions", "relay directory must be private and user-owned")
    return path


def _write_json(path, value):
    path = Path(path)
    data = (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode()
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise InferenceInfrastructureError("relay_replay", "relay output exists; refusing overwrite/retry")
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
            raise InferenceInfrastructureError("relay_permissions", "relay files must be private regular files")
        return json.load(handle)


def _controller_alive(ready):
    pid = ready.get("controller_pid")
    if type(pid) is not int or pid <= 0:
        raise InferenceInfrastructureError("relay_protocol", "relay has no controller process identity")
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class FullRelayClient:
    """Trusted simulator helper; a pending request is never resent or timed out."""
    def __init__(self, relay_dir):
        self.path = Path(relay_dir).absolute()
        self.stopped = False

    def wait_ready(self):
        while not (self.path / "ready.json").exists():
            if (self.path / "stopped.json").exists():
                self.stopped = True
                raise InferenceInfrastructureError("controller_unavailable", "local inference controller stopped")
            time.sleep(.1)
        _private_directory(self.path)
        ready = _read_json(self.path / "ready.json")
        if (self.path / "stopped.json").exists() or not _controller_alive(ready):
            self.stopped = True
            raise InferenceInfrastructureError("controller_unavailable", "local inference controller stopped")
        return ready

    def _request(self, payload):
        if self.stopped:
            raise InferenceInfrastructureError("controller_stopped", "relay client has stopped")
        ready = self.wait_ready()
        inbox = _private_directory(self.path / "requests")
        responses = _private_directory(self.path / "responses")
        if any(not (responses / p.name).exists() for p in inbox.glob("*.json")):
            self.stopped = True
            raise InferenceInfrastructureError("relay_concurrency", "another request is pending; do not resend")
        request_id = uuid.uuid4().hex
        _write_json(inbox / (request_id + ".json"), {"request_id": request_id, "payload": payload})
        response_path = responses / (request_id + ".json")
        while not response_path.exists():
            if (self.path / "stopped.json").exists() or not _controller_alive(ready):
                self.stopped = True
                raise InferenceInfrastructureError("relay_interrupted", "request outcome is unknown; never resend")
            time.sleep(.1)
        response = _read_json(response_path)
        if response.get("request_id") != request_id:
            self.stopped = True
            raise InferenceInfrastructureError("relay_protocol", "response request identifier mismatch")
        if response.get("ok") is not True:
            if response.get("episode_terminal") is True and not response.get("infrastructure_error"):
                raise ModelEpisodeFailure(response["code"], response["message"])
            self.stopped = True
            raise InferenceInfrastructureError(response.get("code", "relay_failure"),
                                                response.get("message", "controller interrupted"))
        return response

    def next_program(self, episode_id, task_instruction, public_observation,
                     previous_result=None, api_documentation=None):
        response = self._request({
            "episode_id": episode_id, "task_instruction": task_instruction,
            "public_observation": public_observation, "previous_result": previous_result,
            "api_documentation": api_documentation,
        })
        return ProgramProposal(**response["proposal"])

    def finish_episode(self, episode_id, reason):
        return self._request({"command": "finish_episode", "episode_id": episode_id, "reason": reason})

    def abandon_episode(self, episode_id, reason):
        return self._request({"command": "abandon_episode", "episode_id": episode_id, "reason": reason})

    def stop(self):
        response = self._request({"command": "stop"})
        self.stopped = True
        return response


def serve_full_relay(driver, relay_dir):
    path = _private_directory(relay_dir)
    inbox = _private_directory(path / "requests")
    archive = _private_directory(path / "processed_requests")
    responses = _private_directory(path / "responses")
    lock_fd = os.open(path / "controller.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise InferenceInfrastructureError("relay_controller_exists", "another controller owns this relay") from None
        if any(inbox.glob("*.json")) or any(archive.glob("*.json")) or (path / "ready.json").exists():
            raise InferenceInfrastructureError("relay_replay", "relay already started; never replay requests")
        _write_json(path / "ready.json", {
            "ready": True, "controller_pid": os.getpid(), "episode_ids": list(driver.episode_ids),
            "max_iterations": None, "absolute_request_timeout_s": None, "budget": driver.snapshot(),
        })
        while True:
            pending = list(inbox.glob("*.json"))
            if not pending:
                time.sleep(.1)
                continue
            if len(pending) != 1:
                failure = InferenceInfrastructureError("relay_concurrency", "parallel requests to one worker forbidden")
                _write_json(path / "stopped.json", {"reason": failure.code, "no_retry": True})
                raise failure
            request_path = pending[0]
            request_id = request_path.stem
            stop = False
            fatal = None
            try:
                if not re.fullmatch(r"[0-9a-f]{32}", request_id):
                    raise ValueError("invalid relay filename")
                envelope = _read_json(request_path)
                if set(envelope) != {"request_id", "payload"} or envelope["request_id"] != request_id:
                    raise ValueError("invalid relay request envelope")
                payload = envelope["payload"]
                if payload == {"command": "stop"}:
                    response = {"ok": True, "stopped": True, "budget": driver.snapshot()}
                    stop = True
                elif (isinstance(payload, dict) and set(payload) == {"command", "episode_id", "reason"}
                      and payload["command"] in {"finish_episode", "abandon_episode"}):
                    method = getattr(driver, payload["command"])
                    response = {"ok": True, "budget": method(payload["episode_id"], payload["reason"])}
                else:
                    if not isinstance(payload, dict) or set(payload) - _REQUEST_KEYS:
                        raise ValueError("invalid model request")
                    response = {"ok": True, "proposal": asdict(driver.next_program(**payload))}
            except ModelEpisodeFailure as exc:
                response = {"ok": False, "episode_terminal": True, "code": exc.code,
                            "message": str(exc), "no_retry": True}
            except BaseException as exc:
                fatal = exc if isinstance(exc, InferenceInfrastructureError) else InferenceInfrastructureError(
                    "relay_failure", f"{type(exc).__name__}: {exc}")
                response = {"ok": False, "infrastructure_error": True, "code": fatal.code,
                            "message": str(fatal), "no_retry": True}
                stop = True
            response["request_id"] = request_id
            # Move out of the watched inbox before publishing the response. The
            # retained archive is never rescanned at every token/turn.
            os.rename(request_path, archive / request_path.name)
            _write_json(responses / request_path.name, response)
            if stop:
                _write_json(path / "stopped.json", {"reason": fatal.code if fatal else "requested_stop", "no_retry": True})
                if fatal:
                    raise fatal
                return
    finally:
        os.close(lock_fd)
