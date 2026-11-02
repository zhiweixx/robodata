"""Host-only Qwen controller for the restored professor endpoint.

The shared revised harness supplies prompts, images, proposal schema and public
history. This adapter changes transport only: no retries, output/turn/time cap,
financial control, or simulator access to the private Tailscale state/socket.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
from urllib.parse import urlsplit

import full_model_driver as base
from task_prompt_contract import TASK_CONTRACT_VERSION

MODEL = "qwen38-27b"
BASE_URL = "http://qwen-api.example.internal:8000/v1"
NATIVE_CONTEXT = 262144
PROTOCOL = "qwen_professor_robodojo_program_v2"
DEFAULT_KEY_FILE = None
InferenceInfrastructureError = base.InferenceInfrastructureError
ModelEpisodeFailure = base.ModelEpisodeFailure
MAX_CONTEXT_ERROR_BYTES = 64 * 1024
_INPUT_CONTEXT_ERROR = re.compile(
    r"The input \(([1-9][0-9]*) tokens\) is longer than the model's context length "
    r"\(([1-9][0-9]*) tokens\)\.")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_endpoint(base_url, model=MODEL):
    if base_url.rstrip("/") != BASE_URL or model != MODEL:
        raise ValueError("only the restored professor Qwen endpoint/model is permitted")


def _write_json(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(base.canonical(value) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def observational_usage(value):
    """Invalid counters stay unknown; original bytes remain in the raw stream."""
    try:
        base.canonical(value)
    except (ValueError, TypeError, OverflowError, RecursionError):
        return None
    return value


def classify_context_rejection(http_status, body):
    """Only an explicit structured input-context rejection ends one episode."""
    if http_status != 400 or not isinstance(body, bytes) or len(body) > MAX_CONTEXT_ERROR_BYTES:
        return None
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate error field")
            value[key] = item
        return value
    def reject(_):
        raise ValueError("nonfinite error field")
    try:
        value = json.loads(body.decode("utf-8"), object_pairs_hook=unique, parse_constant=reject)
        if (not isinstance(value, dict)
                or set(value) != {"object", "message", "type", "param", "code"}
                or value["object"] != "error" or value["type"] != "BadRequestError"
                or type(value["code"]) is not int or value["code"] != 400
                or value["param"] is not None or not isinstance(value["message"], str)):
            return None
        match = _INPUT_CONTEXT_ERROR.fullmatch(value["message"])
        if match is None:
            return None
        count, limit = map(int, match.groups())
        if count <= limit or limit != NATIVE_CONTEXT:
            return None
    except (ValueError, UnicodeError, RecursionError):
        return None
    return {"http_status":400, "provider_error_type":"BadRequestError",
            "provider_error_code":400, "server_input_tokens":count,
            "server_context_length":limit, "raw_body_sha256":hashlib.sha256(body).hexdigest()}


def read_sse(response, archive, progress=None):
    """Read a completed ChatCompletions stream; token usage is observational."""
    content, reasoning, tool_calls = [], [], []
    usage = finish_reason = response_id = model = None
    done = False
    chunks = 0
    last = {}
    event_lines = []
    pending_content, pending_reasoning = [], []
    started = last_flush = time.monotonic()
    first_token = None
    def emit(force=False):
        nonlocal last_flush
        if progress and (force or time.monotonic() - last_flush >= 5):
            progress({"chunks_received":chunks, "content_delta":"".join(pending_content),
                "reasoning_delta":"".join(pending_reasoning), "finish_reason":finish_reason})
            pending_content.clear(); pending_reasoning.clear(); last_flush = time.monotonic()
    def event(data):
        nonlocal done, chunks, response_id, model, usage, finish_reason, last, first_token
        if data == "[DONE]":
            done = True
            return
        packet = json.loads(data)
        if not isinstance(packet, dict) or packet.get("error"):
            raise InferenceInfrastructureError("stream_error", "inference emitted an error event")
        chunks += 1
        if packet.get("model") is not None:
            if packet["model"] != MODEL:
                raise InferenceInfrastructureError("model_mismatch", "stream model differs from requested checkpoint")
            model = packet["model"]
        if packet.get("id") is not None:
            if response_id is not None and packet["id"] != response_id:
                raise InferenceInfrastructureError("stream_protocol", "response ID changed within stream")
            response_id = packet["id"]
        if packet.get("usage") is not None:
            usage = observational_usage(packet["usage"])
        last = packet
        choices = packet.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise InferenceInfrastructureError("stream_protocol", "one streamed choice is required")
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                raise InferenceInfrastructureError("stream_protocol", "unexpected streamed choice index")
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                raise InferenceInfrastructureError("stream_protocol", "invalid stream delta")
            if delta.get("reasoning") is not None and delta.get("reasoning_content") is not None:
                raise InferenceInfrastructureError("stream_protocol", "ambiguous reasoning fields")
            for key, target, pending in (("content",content,pending_content),
                    ("reasoning",reasoning,pending_reasoning), ("reasoning_content",reasoning,pending_reasoning)):
                value = delta.get(key)
                if value is not None:
                    if not isinstance(value, str):
                        raise InferenceInfrastructureError("stream_protocol", "non-text generation delta")
                    target.append(value); pending.append(value)
                    if value and first_token is None:
                        first_token = time.monotonic() - started
            if delta.get("tool_calls"):
                tool_calls.extend(delta["tool_calls"])
            if delta.get("function_call"):
                tool_calls.append(delta["function_call"])
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
        emit()
    try:
        for raw in response:
            archive.write(raw)
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line:
                if event_lines:
                    event("\n".join(event_lines)); event_lines.clear()
                    if done:
                        break
            elif line.startswith("data:"):
                event_lines.append(line[5:].lstrip(" "))
        if event_lines and not done:
            event("\n".join(event_lines))
    finally:
        emit(force=True)
    if not done or finish_reason is None or model != MODEL or not isinstance(response_id,str) or not response_id:
        raise InferenceInfrastructureError("incomplete_stream", "stream ended without terminal status/model; no resend")
    return {"id":response_id, "model":model, "object":"chat.completion",
        "choices":[{"index":0, "finish_reason":finish_reason, "message":{
            "role":"assistant", "content":"".join(content), "reasoning":"".join(reasoning),
            "tool_calls":tool_calls or None}}], "usage":usage, "stream_chunks":chunks,
        "system_fingerprint":last.get("system_fingerprint"), "stream_done":True,
        "first_token_after_headers_seconds":first_token}


class InternalHTTPTransport:
    def __init__(self, base_url, *, api_key_file=None, archive_dir, network_socket):
        validate_endpoint(base_url)
        from qwen_network import ExistingTailnet
        self.base_url = base_url
        self.network = ExistingTailnet(network_socket)
        self.archive = Path(archive_dir)
        self.archive.mkdir(parents=True, exist_ok=True)
        endpoint = urlsplit(base_url)
        self.host, self.port = endpoint.hostname, endpoint.port
        self.counter = 0

    def __call__(self, path, payload, progress=None):
        if path != "/v1/chat/completions" or payload.get("model") != MODEL:
            raise ValueError("unsupported inference request")
        if any(k in payload for k in ("max_tokens", "max_completion_tokens", "max_output_tokens")):
            raise ValueError("no independent client output cap is authorized")
        self.counter += 1
        prefix = self.archive / f"request_{self.counter:04d}"
        body = base.canonical(payload).encode()
        started = time.monotonic()
        receipt = {"request_number":self.counter, "request_sha256":hashlib.sha256(body).hexdigest(),
            "sent_model":MODEL, "base_url":self.base_url, "started_utc":utc_now(),
            "status":"interrupted", "automatic_retries":0}
        response = connection = None
        fd = os.open(str(prefix)+".sse", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb") as archive:
                from qwen_network import NCConnection
                connection = NCConnection(self.network)
                connection.connect()
                connection.request("POST", path, body=body, headers={
                    "Content-Type":"application/json", "Accept":"text/event-stream", "Connection":"close"})
                response = connection.getresponse()
                receipt["http_status"] = response.status
                receipt["headers_wall_seconds"] = time.monotonic()-started
                _write_json(str(prefix)+".headers.json", {**receipt, "headers":[
                    [k,v] for k,v in response.getheaders() if k.lower() in
                    {"content-type","content-length","transfer-encoding","connection","date","server"}]})
                try:
                    if response.status != 200:
                        error_body = bytearray()
                        while chunk := response.read(64*1024):
                            archive.write(chunk)
                            remaining = MAX_CONTEXT_ERROR_BYTES+1-len(error_body)
                            if remaining > 0:
                                error_body.extend(chunk[:remaining])
                        evidence = classify_context_rejection(response.status, bytes(error_body))
                        if evidence is not None:
                            failure = ModelEpisodeFailure("native_context_exhausted", "full public history exceeded native context")
                            failure.context_rejection = evidence
                            raise failure
                        raise InferenceInfrastructureError("remote_http_error", f"inference returned HTTP {response.status}; no retry")
                    result = read_sse(response, archive, progress)
                    receipt.update(status="completed", model=result["model"],
                        finish_reason=result["choices"][0]["finish_reason"], stream_done=True,
                        stream_chunks=result["stream_chunks"], provider_usage=result.get("usage"),
                        response_id=result.get("id"))
                    first = result.get("first_token_after_headers_seconds")
                    receipt["time_to_first_token_seconds"] = None if first is None else receipt["headers_wall_seconds"]+first
                    return result
                finally:
                    archive.flush(); os.fsync(archive.fileno())
        except (OSError, http.client.HTTPException, UnicodeError, json.JSONDecodeError) as exc:
            receipt["error_type"] = type(exc).__name__
            raise InferenceInfrastructureError("remote_transport_interrupted", "inference interrupted; raw stream retained; no resend") from None
        except BaseException as exc:
            receipt["error_type"] = type(exc).__name__
            receipt["error_code"] = getattr(exc, "code", None)
            raise
        finally:
            if response is not None:
                response.close()
            if connection is not None:
                connection.close()
            receipt["wall_seconds"] = time.monotonic()-started
            _write_json(str(prefix)+".receipt.json", receipt)


class RemoteDriver(base.FullQwenDriver):
    def __init__(self, usage_jsonl, episode_ids, *, episode_seeds, base_url=BASE_URL,
            model=MODEL, api_key_file=DEFAULT_KEY_FILE, expected_context_length=NATIVE_CONTEXT,
            api_documentation=None, billing_mode="user_managed", serving_report_path=None,
            allocation_id=None, worker_id=None, network_socket=None, _transport=None):
        validate_endpoint(base_url, model)
        if expected_context_length != NATIVE_CONTEXT or billing_mode != "user_managed":
            raise ValueError("pinned context and nonfinancial controller mode required")
        ids = tuple(episode_ids)
        if (not ids or len(set(ids)) != len(ids)
                or any(not isinstance(e,str) or not re.fullmatch(r"[A-Za-z0-9_-]+",e) for e in ids)
                or set(episode_seeds) != set(ids) or any(type(v) is not int or v<0 for v in episode_seeds.values())):
            raise ValueError("distinct authorized IDs and matching policy seeds required")
        self.episode_ids, self.episode_seeds = ids, dict(episode_seeds)
        self.model, self.expected_context_length = model, expected_context_length
        self.api_documentation = api_documentation
        self.path = Path(usage_jsonl).absolute(); self.path.parent.mkdir(parents=True, exist_ok=True)
        self._transport = _transport or InternalHTTPTransport(base_url, api_key_file=api_key_file,
            archive_dir=self.path.parent/"raw_inference", network_socket=network_socket)
        self.history, self.task_instructions, self.api_documents = {}, {}, {}
        self.finished, self.policy_ended, self.failed = set(), set(), {}
        self.calls = dict.fromkeys(ids,0); self.lifetime_calls = dict.fromkeys(ids,0); self.attempts = dict.fromkeys(ids,1)
        self.completed_calls = self.total_prompt_tokens = self.total_completion_tokens = 0
        self.valid_responses = self.unknown_usage_responses = 0
        self.total_generation_wall_s = self.total_tokenize_wall_s = 0.0
        self.infrastructure_stopped = False; self.max_iterations = None; self.client = self
        self._call_lock = threading.Lock()
        fd = os.open(self.path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); os.close(fd)
        self.config = {"protocol":PROTOCOL, "proposal_schema":base.PROPOSAL_SCHEMA_VERSION,
            "task_contract":TASK_CONTRACT_VERSION, "purpose":"revised_exact220_professor_qwen",
            "model":model, "provider":"professor_api", "episode_ids":list(ids), "episode_seeds":self.episode_seeds,
            "base_url":base_url, "max_iterations":None, "max_input_tokens":None,
            "independent_max_output_tokens":None, "expected_context_length":NATIVE_CONTEXT,
            "generation_length_rule":"no client output cap; expected context262144; remote limit unverified",
            "history_policy":"all public messages and images; no rolling truncation; prior reasoning omitted",
            "socket_idle_timeout_s":None, "absolute_generation_timeout_s":None,
            "reasoning_effort":"medium", "chat_template_kwargs":base.THINKING, "sampling":base.SAMPLING,
            "sampling_seed_rule":"episode policy seed", "billing_mode":"user_managed",
            "spending_control_enabled":False, "cost_monitoring_enabled":False,
            "transport":"private_rootless_tailscale_nc", "automatic_retries":0,
            "serving_report_path":serving_report_path,
            "allocation_id":allocation_id, "worker_id":worker_id,
            "driver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "base_driver_sha256":hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()}
        self._append("suite_initialized", config=self.config)

    def snapshot(self, episode_id=None):
        result = super().snapshot(episode_id)
        result.pop("api_cost_usd",None)
        unknown = sum(self.lifetime_calls.values())-self.completed_calls
        result.update(context_length=NATIVE_CONTEXT, provider_reported_completion_tokens=self.total_completion_tokens,
            completion_token_semantics="Provider counters; reasoning inclusion is unverified",
            valid_responses=self.valid_responses, requests_without_complete_usage=unknown,
            token_accounting_complete=unknown==0, token_totals_scope="valid reported usage only; unknown is not zero")
        return result

    def _next(self, episode_id, task_instruction, public_observation, previous_result, api_documentation):
        if self.infrastructure_stopped:
            raise InferenceInfrastructureError("controller_stopped", "worker controller previously stopped")
        if episode_id not in self.calls or episode_id in self.finished or episode_id in self.policy_ended:
            raise ValueError("episode outside authorized live scope; no implicit replay")
        count = self.calls[episode_id]
        if not isinstance(task_instruction,str) or not task_instruction.strip():
            raise ValueError("public task instruction required")
        if self.task_instructions.setdefault(episode_id,task_instruction) != task_instruction:
            raise ValueError("task instruction changed during episode")
        docs = api_documentation or self.api_documentation
        if not isinstance(docs,str) or not docs.strip() or self.api_documents.setdefault(episode_id,docs) != docs:
            raise ValueError("stable public API documentation required")
        if bool(count) != (previous_result is not None):
            raise ValueError("feedback required exactly after initial program")
        feedback = None if previous_result is None else base.freeze_public(previous_result)
        current = [{"type":"text", "text":json.dumps({"task_instruction":task_instruction,
            "iteration":count+1, "previous_execution":feedback, "episode_continues_without_reset":True},sort_keys=True)}]
        current += base.observation_content(public_observation)
        messages = [{"role":"system", "content":base.SYSTEM_PREFIX+docs}]
        messages += self.history.setdefault(episode_id,[])+[{"role":"user","content":current}]
        seed = self.episode_seeds[episode_id]
        payload,_ = base.build_chat_requests(messages,seed=seed,model=self.model)
        self._append("program_input",episode_id=episode_id,iteration=count+1,
            request=base.redact(payload),request_sha256=base.sha(payload),
            instructions_sha256=hashlib.sha256((base.SYSTEM_PREFIX+docs).encode()).hexdigest())
        self.calls[episode_id]+=1; self.lifetime_calls[episode_id]+=1
        self._append("generation_started",episode_id=episode_id,iteration=count+1,
            counted_input_tokens=None,remaining_native_context=None,policy_seed=seed)
        started=time.monotonic()
        try:
            response=self._transport("/v1/chat/completions",payload,
                progress=lambda delta:self._append("generation_stream",episode_id=episode_id,iteration=count+1,**delta))
        except BaseException as exc:
            elapsed=time.monotonic()-started; self.total_generation_wall_s+=elapsed
            self._append("generation_context_rejected" if isinstance(exc,ModelEpisodeFailure) else "generation_transport_failure",
                episode_id=episode_id,iteration=count+1,generation_wall_s=elapsed,
                code=getattr(exc,"code",None),error_type=type(exc).__name__,provider_usage=None,
                token_usage_unknown=True,no_retry=True,**getattr(exc,"context_rejection",{}))
            raise
        elapsed=time.monotonic()-started; self.total_generation_wall_s+=elapsed; self.valid_responses+=1
        response={**response,"usage":observational_usage(response.get("usage"))}
        self._append("raw_response",episode_id=episode_id,iteration=count+1,generation_wall_s=elapsed,response=base.redact(response))
        usage=response.get("usage")
        known=(isinstance(usage,dict) and all(type(usage.get(k)) is int and usage[k]>=0 for k in ("prompt_tokens","completion_tokens")))
        measured={"input_tokens":None,"output_tokens":None,"generation_wall_s":elapsed,"input_count_wall_s":0.0,
            "preflight_input_tokens":None,"remaining_native_context":None,"token_usage_unknown":not known,
            "response_id":response.get("id")}
        if known:
            prompt,completion=usage["prompt_tokens"],usage["completion_tokens"]
            self.total_prompt_tokens+=prompt; self.total_completion_tokens+=completion; self.completed_calls+=1
            measured.update(usage,input_tokens=prompt,output_tokens=completion,raw_usage=usage,
                output_includes_reasoning=None,token_count_source="server_reported",token_count_approximate=False)
            self._append("generation_usage",episode_id=episode_id,iteration=count+1,usage=measured)
        else:
            self.unknown_usage_responses+=1
            self._append("generation_usage_unknown",episode_id=episode_id,iteration=count+1,
                reason="missing_or_invalid_provider_usage",raw_usage=usage,token_usage_unknown=True)
        if response.get("model") != self.model:
            raise InferenceInfrastructureError("model_mismatch","response differs from requested model")
        choices=response.get("choices")
        if not isinstance(choices,list) or len(choices)!=1:
            raise InferenceInfrastructureError("invalid_choices","expected one model choice")
        finish=choices[0].get("finish_reason")
        if finish=="length":
            raise InferenceInfrastructureError("provider_output_truncated","output was truncated; preserve attempt without score or retry")
        if finish!="stop":
            raise ModelEpisodeFailure("model_response_not_completed","model response ended without completed program")
        message=choices[0].get("message",{})
        if message.get("tool_calls") or message.get("function_call"):
            raise ModelEpisodeFailure("invalid_model_program","native tool call outside code-as-policy protocol")
        raw=message.get("content")
        try:
            parsed=base.parse_program_proposal(raw)
        except (ValueError,TypeError):
            raise ModelEpisodeFailure("invalid_model_program","model content not valid code-as-policy JSON") from None
        self.history[episode_id]+=[{"role":"user","content":current},{"role":"assistant","content":raw}]
        if parsed["give_up"]:
            self.policy_ended.add(episode_id); self.history.pop(episode_id,None)
        return base.ProgramProposal(**parsed,iteration=count+1,response_id=response.get("id"),
            usage=measured,budget=self.snapshot(episode_id))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--relay-dir",type=Path,required=True)
    args=parser.parse_args()
    driver=RemoteDriver(**json.loads(args.config.read_text()))
    from full_relay import serve_full_relay
    serve_full_relay(driver,args.relay_dir)


if __name__ == "__main__":
    main()
