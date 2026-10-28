"""Local Qwen policy with only the model's finite context as a token boundary.

There is no model-turn, input-token, or independent output-token pilot cap.
Every request can generate through the remaining native context. Full public
history is retained (no hidden rolling window); context exhaustion is an
explicit episode outcome. There is no network deadline by default: a shared
server may legitimately queue a request while another uses its native context.
The driver never executes generated programs or reads keys.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from urllib import error, parse, request

from robodata.agents.model_driver import FORBIDDEN_DATA_KEYS, OUTPUT_FORMAT, PUBLIC_OBSERVATION_KEYS, ProgramProposal, SYSTEM_PREFIX
from robodata.agents.local_model_driver import redact

MODEL = "Qwen/Qwen3.8-27B"
NATIVE_CONTEXT_LENGTH = 262144
THINKING = {"enable_thinking": True, "preserve_thinking": False, "reasoning_effort": "medium"}
SAMPLING = {"temperature": 1.0, "top_p": .95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0}


class ModelEpisodeFailure(RuntimeError):
    """Known policy failure: end this episode, not other authorized episodes."""
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


class InferenceInfrastructureError(RuntimeError):
    """An interrupted/invalid infrastructure attempt must not become a score."""
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def canonical(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def integer(value, name):
    if type(value) is not int or value < 0:
        raise InferenceInfrastructureError("invalid_accounting", name + " must be a nonnegative integer")
    return value


def freeze_public(value):
    """Schema/provenance validation without a pilot text-size or token cap."""
    frozen = json.loads(json.dumps(value, allow_nan=False))
    def check(item):
        if isinstance(item, dict):
            if any(key.lower() in FORBIDDEN_DATA_KEYS for key in item):
                raise ValueError("privileged/evaluator fields are forbidden in policy input")
            for nested in item.values():
                check(nested)
        elif isinstance(item, list):
            for nested in item:
                check(nested)
    check(frozen)
    return frozen


def observation_content(observation):
    if not isinstance(observation, dict) or set(observation) - PUBLIC_OBSERVATION_KEYS:
        raise ValueError("unknown public observation fields")
    fields = {key: value for key, value in observation.items() if key != "images"}
    content = [{"type": "text", "text": json.dumps(freeze_public(fields), sort_keys=True)}]
    images = observation.get("images", [])
    if not isinstance(images, list):
        raise ValueError("images must be a list")
    for item in images:
        if not isinstance(item, dict) or set(item) != {"camera", "data_url"}:
            raise ValueError("image must contain camera and data_url only")
        camera, url = item["camera"], item["data_url"]
        if not isinstance(camera, str) or not camera or not isinstance(url, str):
            raise ValueError("invalid public camera image")
        if not re.fullmatch(r"data:image/(?:png|jpeg);base64,[A-Za-z0-9+/=]+", url):
            raise ValueError("camera images must be inline PNG or JPEG")
        content.extend([{"type": "text", "text": "Public camera: " + camera},
                        {"type": "image_url", "image_url": {"url": url, "detail": "high"}}])
    return content


def build_chat_requests(messages, *, seed, model=MODEL, remaining_context=None):
    """The only generation length bound is the finite serving context."""
    fmt = OUTPUT_FORMAT["format"]
    generation = {
        "model": model, "messages": messages, "stream": True,
        "stream_options": {"include_usage": True}, "reasoning_effort": "medium",
        "add_generation_prompt": True, "continue_final_message": False,
        "add_special_tokens": False, "chat_template_kwargs": dict(THINKING),
        "response_format": {"type": "json_schema", "json_schema": {
            "name": fmt["name"], "strict": fmt["strict"], "schema": fmt["schema"]}},
        "seed": seed, **SAMPLING,
    }
    if remaining_context is not None:
        if type(remaining_context) is not int or remaining_context <= 0:
            raise ValueError("remaining native context must be positive")
        generation["max_tokens"] = remaining_context
    tokenize = {"model": model, "messages": messages,
                "chat_template_kwargs": dict(THINKING), "add_generation_prompt": True,
                "continue_final_message": False, "add_special_tokens": False,
                "return_token_strs": False}
    return generation, tokenize


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise InferenceInfrastructureError("redirect", "local inference redirect rejected")


class LoopbackStreamingHTTP:
    """Unauthenticated loopback only; no proxies/retries or total generation timer."""
    def __init__(self, base_url, *, socket_idle_timeout_s=None):
        parsed = parse.urlsplit(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path.rstrip("/") != "/v1"):
            raise ValueError("base_url must be an unauthenticated loopback http URL ending /v1")
        if socket_idle_timeout_s is not None and (
                type(socket_idle_timeout_s) not in (int, float)
                or not math.isfinite(socket_idle_timeout_s) or socket_idle_timeout_s <= 0):
            raise ValueError("socket idle timeout must be None or finite and positive")
        self.origin = parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        self.idle_timeout = None if socket_idle_timeout_s is None else float(socket_idle_timeout_s)
        self.opener = request.build_opener(request.ProxyHandler({}), NoRedirect())

    def __call__(self, path, payload, progress=None):
        if path not in {"/tokenize", "/v1/chat/completions"}:
            raise ValueError("unsupported local inference endpoint")
        req = request.Request(self.origin + path, data=canonical(payload).encode(), method="POST",
                              headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=self.idle_timeout) as response:
                if path == "/tokenize":
                    result = json.load(response)
                    if not isinstance(result, dict):
                        raise InferenceInfrastructureError("invalid_tokenize_response", "tokenizer returned non-object")
                    return result
                return self.read_sse(response, progress)
        except error.HTTPError as exc:
            raise InferenceInfrastructureError("http_error", f"local inference returned HTTP {exc.code}") from None
        except (error.URLError, TimeoutError, OSError) as exc:
            raise InferenceInfrastructureError("transport_interrupted", "local inference transport interrupted; do not resend") from None
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise InferenceInfrastructureError("invalid_response", "local inference returned malformed data") from None

    @staticmethod
    def read_sse(response, progress=None):
        content, reasoning, tool_calls = [], [], []
        last = {}
        usage = None
        finish_reason = None
        response_id = model = None
        done = False
        chunks = 0
        buffered_content, buffered_reasoning = [], []
        last_flush = time.monotonic()
        event_lines = []

        def emit_progress(force=False):
            nonlocal last_flush
            if progress and (force or time.monotonic() - last_flush >= 5):
                progress({"chunks_received": chunks, "content_delta": "".join(buffered_content),
                          "reasoning_delta": "".join(buffered_reasoning), "finish_reason": finish_reason})
                buffered_content.clear(); buffered_reasoning.clear()
                last_flush = time.monotonic()

        def event(data):
            nonlocal done, chunks, response_id, model, usage, finish_reason, last
            if data == "[DONE]":
                done = True
                return
            packet = json.loads(data)
            if not isinstance(packet, dict) or packet.get("error"):
                raise InferenceInfrastructureError("stream_error", "local server emitted an error event")
            chunks += 1
            response_id = packet.get("id", response_id)
            model = packet.get("model", model)
            last = packet
            if packet.get("usage") is not None:
                usage = packet["usage"]
            choices = packet.get("choices", [])
            if not isinstance(choices, list) or len(choices) > 1:
                raise InferenceInfrastructureError("stream_protocol", "expected one streamed choice")
            for choice in choices:
                if choice.get("index", 0) != 0:
                    raise InferenceInfrastructureError("stream_protocol", "unexpected streamed choice index")
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    raise InferenceInfrastructureError("stream_protocol", "invalid stream delta")
                if delta.get("reasoning") is not None and delta.get("reasoning_content") is not None:
                    raise InferenceInfrastructureError("stream_protocol", "ambiguous reasoning delta fields")
                for key, target, buffered in (("content", content, buffered_content),
                                              ("reasoning", reasoning, buffered_reasoning),
                                              ("reasoning_content", reasoning, buffered_reasoning)):
                    value = delta.get(key)
                    if value is not None:
                        if not isinstance(value, str):
                            raise InferenceInfrastructureError("stream_protocol", "non-text model delta")
                        target.append(value); buffered.append(value)
                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])
                if delta.get("function_call"):
                    tool_calls.append(delta["function_call"])
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
            emit_progress()

        try:
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line:
                    if event_lines:
                        event("\n".join(event_lines)); event_lines.clear()
                        if done:
                            break
                elif line.startswith("data:"):
                    event_lines.append(line[5:].lstrip(" "))
                # SSE comments/event names are not model data.
            if event_lines and not done:
                event("\n".join(event_lines))
        finally:
            emit_progress(force=True)
        if not done or finish_reason is None or usage is None:
            raise InferenceInfrastructureError("incomplete_stream", "stream ended without final status and usage")
        return {"id": response_id, "model": model, "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": finish_reason,
                             "message": {"role": "assistant", "content": "".join(content),
                                         "reasoning": "".join(reasoning), "tool_calls": tool_calls or None}}],
                "usage": usage, "stream_chunks": chunks,
                "system_fingerprint": last.get("system_fingerprint")}


class FullQwenDriver:
    """One sequential worker controller; independent controllers share vLLM."""
    def __init__(self, usage_jsonl, episode_ids, *, episode_seeds,
                 base_url="http://127.0.0.1:8000/v1", model=MODEL,
                 expected_context_length=NATIVE_CONTEXT_LENGTH,
                 socket_idle_timeout_s=None, api_documentation=None, _transport=None):
        ids = tuple(episode_ids)
        if (not ids or len(set(ids)) != len(ids)
                or any(not isinstance(e, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", e) for e in ids)):
            raise ValueError("explicit distinct authorized episode IDs required")
        if set(episode_seeds) != set(ids) or any(type(v) is not int or v < 0 for v in episode_seeds.values()):
            raise ValueError("episode_seeds must map every authorized ID to its policy seed")
        if model != MODEL or expected_context_length != NATIVE_CONTEXT_LENGTH:
            raise ValueError("full evaluation requires pinned Qwen3.8-27B native262144 context")
        self.episode_ids, self.episode_seeds = ids, dict(episode_seeds)
        self.model, self.expected_context_length = model, expected_context_length
        self.api_documentation = api_documentation
        self._transport = _transport or LoopbackStreamingHTTP(base_url, socket_idle_timeout_s=socket_idle_timeout_s)
        # Validate the URL even when tests inject a transport.
        LoopbackStreamingHTTP(base_url, socket_idle_timeout_s=socket_idle_timeout_s)
        self.history, self.task_instructions, self.api_documents = {}, {}, {}
        self.finished, self.policy_ended, self.failed = set(), set(), {}
        self.calls = dict.fromkeys(ids, 0)
        self.lifetime_calls = dict.fromkeys(ids, 0)
        self.attempts = dict.fromkeys(ids, 1)
        self.completed_calls = self.total_prompt_tokens = self.total_completion_tokens = 0
        self.total_generation_wall_s = self.total_tokenize_wall_s = 0.0
        self.infrastructure_stopped = False
        self.max_iterations = None
        self.client = self
        self._call_lock = threading.Lock()
        self.path = Path(usage_jsonl).absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        self.config = {
            "purpose": "aspire_robodojo_full_local_qwen", "model": model,
            "episode_ids": list(ids), "episode_seeds": self.episode_seeds,
            "base_url": base_url, "max_iterations": None, "max_input_tokens": None,
            "independent_max_output_tokens": None, "expected_context_length": expected_context_length,
            "generation_length_rule": "native_context_length minus exact multimodal prompt tokens",
            "history_policy": "all public messages and images; no rolling truncation; prior reasoning omitted",
            "socket_idle_timeout_s": socket_idle_timeout_s, "absolute_generation_timeout_s": None,
            "reasoning_effort": "medium", "chat_template_kwargs": THINKING, "sampling": SAMPLING,
            "sampling_seed_rule": "episode policy seed", "remote_api_calls": False,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        self._append("suite_initialized", config=self.config)

    def _append(self, event, **fields):
        episode_id = fields.get("episode_id")
        if episode_id in self.attempts:
            fields.setdefault("attempt", self.attempts[episode_id])
        row = {"event": event, "utc": datetime.now(timezone.utc).isoformat(), **fields}
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(canonical(row) + "\n"); handle.flush(); os.fsync(handle.fileno())

    def snapshot(self, episode_id=None):
        # Per-turn receipts stay constant-size even when a worker owns thousands
        # of candidate layouts. Full manifest/state appears at start and stop.
        scope = self.calls if episode_id is None else {episode_id: self.calls[episode_id]}
        attempts = self.attempts if episode_id is None else {episode_id: self.attempts[episode_id]}
        lifetime = self.lifetime_calls if episode_id is None else {episode_id: self.lifetime_calls[episode_id]}
        failures = self.failed if episode_id is None else {
            key: value for key, value in self.failed.items() if key == episode_id}
        finished = self.finished if episode_id is None else self.finished.intersection({episode_id})
        return {"model": self.model, "calls": dict(scope), "attempts": dict(attempts),
                "lifetime_calls": dict(lifetime), "completed_calls": self.completed_calls,
                "aborted": self.infrastructure_stopped, "episode_failures": dict(failures),
                "finished_episode_ids": sorted(finished), "finished_episode_count": len(self.finished),
                "max_calls_per_episode": None,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens_including_reasoning": self.total_completion_tokens,
                "generation_wall_s": self.total_generation_wall_s,
                "tokenize_wall_s": self.total_tokenize_wall_s, "api_cost_usd": 0}

    def finish_episode(self, episode_id, reason):
        if episode_id not in self.calls:
            raise ValueError("episode outside authorized manifest")
        if not isinstance(reason, str) or not reason:
            raise ValueError("episode end reason required")
        if episode_id in self.finished:
            return self.snapshot(episode_id)
        self.history.pop(episode_id, None)
        self.finished.add(episode_id)
        self._append("episode_closed", episode_id=episode_id, reason=reason,
                     calls=self.calls[episode_id], model_failure=self.failed.get(episode_id))
        return self.snapshot(episode_id)

    def abandon_episode(self, episode_id, reason):
        """Explicit native unscored interruption; never reopen a scored end.

        The native evaluator owns the decision to retry an unscored unstable
        layout. This operation does not retry anything itself, and preserves all
        prior request/token records. A stopped inference controller needs a fresh
        process/ledger instead of this operation.
        """
        if episode_id not in self.calls or self.infrastructure_stopped:
            raise InferenceInfrastructureError("invalid_abandon", "cannot abandon unknown/stopped episode")
        if reason not in {"native_unstable", "infrastructure_interrupted"}:
            raise InferenceInfrastructureError("invalid_abandon", "abandon requires an unscored native interruption")
        if episode_id in self.finished:
            raise InferenceInfrastructureError("invalid_abandon", "a committed scored episode cannot be reopened")
        self._append("unscored_attempt_abandoned", episode_id=episode_id, reason=reason,
                     attempt=self.attempts[episode_id], calls=self.calls[episode_id],
                     native_evaluator_owns_retry=True)
        self.history.pop(episode_id, None)
        self.policy_ended.discard(episode_id)
        self.failed.pop(episode_id, None)
        self.calls[episode_id] = 0
        self.attempts[episode_id] += 1
        return self.snapshot(episode_id)

    def next_program(self, episode_id, task_instruction, public_observation,
                     previous_result=None, api_documentation=None):
        if not self._call_lock.acquire(blocking=False):
            raise InferenceInfrastructureError("worker_concurrency", "one controller cannot generate two requests concurrently")
        try:
            return self._next(episode_id, task_instruction, public_observation, previous_result, api_documentation)
        except ModelEpisodeFailure as exc:
            self.failed[episode_id] = exc.code
            self.policy_ended.add(episode_id)
            self.history.pop(episode_id, None)
            self._append("model_episode_failed", episode_id=episode_id, code=exc.code,
                         message=str(exc), no_retry=True)
            raise
        except BaseException as exc:
            self.infrastructure_stopped = True
            self._append("infrastructure_stopped", episode_id=episode_id,
                         code=getattr(exc, "code", "invalid_controller_input"),
                         error_type=type(exc).__name__, message=str(exc), no_retry=True)
            if isinstance(exc, InferenceInfrastructureError):
                raise
            raise InferenceInfrastructureError("invalid_controller_input", str(exc)) from exc
        finally:
            self._call_lock.release()

    def _next(self, episode_id, task_instruction, public_observation, previous_result, api_documentation):
        if self.infrastructure_stopped:
            raise InferenceInfrastructureError("controller_stopped", "worker infrastructure previously stopped")
        if episode_id not in self.calls:
            raise ValueError("episode outside authorized manifest")
        if episode_id in self.finished or episode_id in self.policy_ended:
            raise ValueError("episode policy ended/score committed; no implicit replay")
        count = self.calls[episode_id]
        if not isinstance(task_instruction, str) or not task_instruction.strip():
            raise ValueError("nonempty public task instruction required")
        if self.task_instructions.setdefault(episode_id, task_instruction) != task_instruction:
            raise ValueError("task instruction changed during episode")
        docs = api_documentation or self.api_documentation
        if not isinstance(docs, str) or not docs.strip():
            raise ValueError("deployed API documentation required")
        if self.api_documents.setdefault(episode_id, docs) != docs:
            raise ValueError("API documentation changed during episode")
        if bool(count) != (previous_result is not None):
            raise ValueError("feedback required exactly after initial program")
        feedback = None if previous_result is None else freeze_public(previous_result)
        current = [{"type": "text", "text": json.dumps({
            "task_instruction": task_instruction, "iteration": count + 1,
            "previous_execution": feedback, "episode_continues_without_reset": True,
        }, sort_keys=True)}] + observation_content(public_observation)
        messages = [{"role": "system", "content": SYSTEM_PREFIX + docs}]
        messages += self.history.setdefault(episode_id, []) + [{"role": "user", "content": current}]
        seed = self.episode_seeds[episode_id]
        _, tokenize = build_chat_requests(messages, seed=seed)
        started = time.monotonic()
        counted = self._transport("/tokenize", tokenize)
        count_seconds = time.monotonic() - started
        self.total_tokenize_wall_s += count_seconds
        prompt_count = integer(counted.get("count"), "tokenize.count")
        context_length = integer(counted.get("max_model_len"), "tokenize.max_model_len")
        self._append("input_token_count", episode_id=episode_id, iteration=count + 1,
                     count=prompt_count, max_model_len=context_length, wall_s=count_seconds,
                     request_sha256=sha(tokenize), approximate=False)
        if context_length != self.expected_context_length:
            raise InferenceInfrastructureError("context_configuration_mismatch", "server is not using pinned native model context")
        remaining = context_length - prompt_count
        if remaining <= 0:
            raise ModelEpisodeFailure("native_context_exhausted", "full public history filled the native model context")
        payload, _ = build_chat_requests(messages, seed=seed, remaining_context=remaining)
        self._append("program_input", episode_id=episode_id, iteration=count + 1,
                     request=redact(payload), request_sha256=sha(payload),
                     instructions_sha256=hashlib.sha256((SYSTEM_PREFIX + docs).encode()).hexdigest())
        self.calls[episode_id] += 1
        self.lifetime_calls[episode_id] += 1
        self._append("generation_started", episode_id=episode_id, iteration=count + 1,
                     counted_input_tokens=prompt_count, remaining_native_context=remaining, policy_seed=seed)
        started = time.monotonic()
        try:
            response = self._transport("/v1/chat/completions", payload,
                progress=lambda delta: self._append("generation_stream", episode_id=episode_id,
                    iteration=count + 1, **delta))
        except BaseException as exc:
            elapsed = time.monotonic() - started
            self.total_generation_wall_s += elapsed
            self._append("generation_transport_failure", episode_id=episode_id, iteration=count + 1,
                         generation_wall_s=elapsed, error_type=type(exc).__name__,
                         token_usage_unknown=True, no_retry=True)
            raise
        elapsed = time.monotonic() - started
        self.total_generation_wall_s += elapsed
        self._append("raw_response", episode_id=episode_id, iteration=count + 1,
                     generation_wall_s=elapsed, response=redact(response))
        if response.get("model") != self.model:
            raise InferenceInfrastructureError("model_mismatch", "response model differs from requested model")
        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise InferenceInfrastructureError("missing_usage", "server did not report complete token usage")
        actual_prompt = integer(usage.get("prompt_tokens"), "usage.prompt_tokens")
        completion = integer(usage.get("completion_tokens"), "usage.completion_tokens")
        self.total_prompt_tokens += actual_prompt
        self.total_completion_tokens += completion
        self.completed_calls += 1
        measured = {**usage, "input_tokens": actual_prompt, "output_tokens": completion,
                    "generation_wall_s": elapsed, "input_count_wall_s": count_seconds,
                    "preflight_input_tokens": prompt_count,
                    "preflight_matches_generation_usage": actual_prompt == prompt_count,
                    "remaining_native_context": remaining, "output_includes_reasoning": True,
                    "token_count_source": "server_reported", "token_count_approximate": False,
                    "api_cost_usd": 0}
        self._append("generation_usage", episode_id=episode_id, iteration=count + 1, usage=measured)
        if actual_prompt != prompt_count or completion > remaining:
            raise InferenceInfrastructureError("accounting_mismatch", "token accounting differs from native context/preflight")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise InferenceInfrastructureError("invalid_choices", "server returned other than one choice")
        finish = choices[0].get("finish_reason")
        if finish == "length":
            if completion != remaining:
                raise InferenceInfrastructureError("unexpected_generation_cap",
                    "server truncated generation before the requested native context boundary")
            raise ModelEpisodeFailure("native_context_exhausted", "generation consumed the remaining native context without a completed answer")
        if finish != "stop":
            raise ModelEpisodeFailure("model_response_not_completed", "model response ended without a completed program")
        message = choices[0].get("message", {})
        if message.get("tool_calls") or message.get("function_call"):
            raise ModelEpisodeFailure("invalid_model_program", "native tool calls are outside code-as-policy protocol")
        raw = message.get("content")
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict) or set(parsed) != {"code", "summary", "done"}:
                raise ValueError("wrong schema")
            if not isinstance(parsed["code"], str) or not isinstance(parsed["summary"], str) or type(parsed["done"]) is not bool:
                raise ValueError("wrong types")
            if (parsed["done"] and parsed["code"].strip()) or (not parsed["done"] and not parsed["code"].strip()):
                raise ValueError("done/code mismatch")
            if "```" in parsed["code"]:
                raise ValueError("markdown code")
        except (ValueError, TypeError):
            raise ModelEpisodeFailure("invalid_model_program", "model final content was not a valid code-as-policy JSON object") from None
        self.history[episode_id] += [{"role": "user", "content": current}, {"role": "assistant", "content": raw}]
        if parsed["done"]:
            self.policy_ended.add(episode_id)
            self.history.pop(episode_id, None)
        return ProgramProposal(**parsed, iteration=count + 1, response_id=response.get("id"),
                               usage=measured, budget=self.snapshot(episode_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--relay-dir", type=Path)
    args = parser.parse_args()
    driver = FullQwenDriver(**json.loads(args.config.read_text()))
    if args.relay_dir:
        from robodata.agents.full_relay import serve_full_relay
        serve_full_relay(driver, args.relay_dir)
        return
    print(canonical({"ready": True, "budget": driver.snapshot()}), flush=True)
    for line in sys.stdin:
        value = json.loads(line)
        try:
            if value.get("command") == "stop":
                print(canonical({"stopped": True, "budget": driver.snapshot()}), flush=True)
                return
            if value.get("command") == "abandon_episode":
                result = {"budget": driver.abandon_episode(value["episode_id"], value["reason"])}
            elif value.get("command") == "finish_episode":
                result = {"budget": driver.finish_episode(value["episode_id"], value["reason"])}
            else:
                result = {"proposal": asdict(driver.next_program(**value))}
            print(canonical({"ok": True, **result}), flush=True)
        except ModelEpisodeFailure as exc:
            print(canonical({"ok": False, "episode_terminal": True, "code": exc.code, "message": str(exc)}), flush=True)
        except InferenceInfrastructureError as exc:
            print(canonical({"ok": False, "infrastructure_error": True, "code": exc.code}), flush=True)
            raise


if __name__ == "__main__":
    # full_relay imports this module's exception classes. Register this same
    # module before main imports the relay, rather than loading a second copy
    # whose ModelEpisodeFailure would have a different class identity.
    sys.modules["robodata.agents.full_model_driver"] = sys.modules[__name__]
    main()
