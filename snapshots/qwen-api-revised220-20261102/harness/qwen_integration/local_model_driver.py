"""Local-only Qwen Chat Completions driver for the unchanged ASPIRE relay.

No API key is read, no Astra client is instantiated, and requests are restricted
to a loopback vLLM server. Exact public prompt content comes from model_driver;
only Responses API content wrappers are translated to Chat Completions wrappers.
Historical reasoning is not fed back, matching the Astra pilot's final-only
assistant history. Per-turn thinking is enabled with official Qwen medium effort.

References, inspected 2026-10-19:
https://huggingface.co/Qwen/Qwen3.8-27B
https://recipes.vllm.ai/Qwen/Qwen3.8-27B
https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/serve/tokenize/serving.py
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

from model_driver import (
    DriverStopped, OUTPUT_FORMAT, ProgramProposal, SYSTEM_PREFIX, parse_program_proposal,
    _bounded_json, observation_content,
)


MODEL = "Qwen/Qwen3.8-27B"
EPISODES = ("general_pickup_s0_l0", "stack_bowls_s0_l0", "align_blocks_s0_l0")
THINKING = {"enable_thinking": True, "preserve_thinking": False,
            "reasoning_effort": "medium"}
SAMPLING = {"seed": 0, "temperature": 1.0, "top_p": .95, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
MAX_RESPONSE_BYTES = 20_000_000


def _canonical(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=True,
                      sort_keys=True, separators=(",", ":"))


def _sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def chat_content(responses_content):
    """Lossless translation of text and inline-image content, preserving order."""
    converted = []
    for item in responses_content:
        if item["type"] == "input_text":
            converted.append({"type": "text", "text": item["text"]})
        elif item["type"] == "input_image":
            converted.append({"type": "image_url", "image_url": {
                "url": item["image_url"], "detail": item.get("detail", "high")}})
        else:
            raise ValueError("unsupported public input content")
    return converted


def build_chat_requests(messages, *, model=MODEL, max_output_tokens=8192):
    """Shared exact request settings for isolated server smoke and pilot calls."""
    fmt = OUTPUT_FORMAT["format"]
    generation = {
        "model": model, "messages": messages, "stream": False,
        "max_tokens": max_output_tokens, "reasoning_effort": "medium",
        "add_generation_prompt": True, "continue_final_message": False,
        "add_special_tokens": False,
        # vLLM0.32 rejects an empty tools array. Omission preserves the Astra
        # request's no-native-tools semantics without changing prompt content.
        "chat_template_kwargs": dict(THINKING),
        "response_format": {"type": "json_schema", "json_schema": {
            "name": fmt["name"], "strict": fmt["strict"], "schema": fmt["schema"]}},
        **SAMPLING,
    }
    tokenize = {"model": model, "messages": messages,
                "chat_template_kwargs": dict(THINKING),
                "add_generation_prompt": True, "continue_final_message": False,
                "add_special_tokens": False,
                "return_token_strs": False}
    return generation, tokenize


def redact(value):
    """Keep audit text; replace inline image bytes with their deterministic hash.

    The simulator retains the exact public_input_NN.json including images. This
    ledger avoids duplicating those bytes across every historical request.
    """
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key.lower() in {"api_key", "authorization", "access_token", "secret", "password"}:
                result[key] = "[REDACTED]"
            else:
                result[key] = redact(item)
        return result
    if isinstance(value, list):
        return [redact(x) for x in value]
    if isinstance(value, str) and value.startswith("data:image/"):
        return {"inline_image_redacted": True,
                "sha256_data_url": hashlib.sha256(value.encode()).hexdigest(),
                "data_url_characters": len(value), "mime": value.split(";", 1)[0][5:]}
    return value


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DriverStopped("local model server redirect rejected")


class LoopbackHTTP:
    """No credentials, proxy, redirect, retry, or non-loopback destination."""
    def __init__(self, base_url, timeout_s, tokenize_timeout_s=60):
        url = parse.urlsplit(base_url)
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}
                or url.username or url.password or url.query or url.fragment
                or url.path.rstrip("/") != "/v1"):
            raise ValueError("base_url must be an unauthenticated loopback http://host:port/v1")
        for name, value, upper in (("generation timeout", timeout_s, 400),
                                   ("tokenization timeout", tokenize_timeout_s, 60)):
            if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= upper:
                raise ValueError(f"{name} must be within1..{upper} seconds")
        self.origin = parse.urlunsplit((url.scheme, url.netloc, "", "", ""))
        self.timeout_s = float(timeout_s)
        self.tokenize_timeout_s = float(tokenize_timeout_s)
        self.opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())

    def __call__(self, path, payload):
        if path not in {"/tokenize", "/v1/chat/completions"}:
            raise ValueError("unsupported local inference endpoint")
        body = _canonical(payload).encode()
        req = request.Request(self.origin + path, data=body, method="POST",
                              headers={"Content-Type": "application/json"})
        try:
            timeout_s = self.tokenize_timeout_s if path == "/tokenize" else self.timeout_s
            with self.opener.open(req, timeout=timeout_s) as response:
                content = response.read(MAX_RESPONSE_BYTES + 1)
        except error.HTTPError as exc:
            # Keep status without reflecting server-generated text into errors.
            raise DriverStopped(f"local inference returned HTTP {exc.code}; no retry") from None
        except (error.URLError, TimeoutError, OSError):
            raise DriverStopped("local inference transport failed; no retry") from None
        if len(content) > MAX_RESPONSE_BYTES:
            raise DriverStopped("local inference response exceeds audit size cap")
        result = json.loads(content)
        if not isinstance(result, dict):
            raise DriverStopped("local inference returned a non-object JSON response")
        return result


class LocalQwenDriver:
    """Drop-in next_program/client.snapshot interface, with fresh-suite auditing."""
    def __init__(self, usage_jsonl, episode_ids=EPISODES, *,
                 base_url="http://127.0.0.1:8000/v1", model=MODEL,
                 api_documentation=None, max_iterations=8, max_input_tokens=128_000,
                 max_output_tokens=8192, request_timeout_s=400, tokenize_timeout_s=60,
                 sampling_seed=0,
                 _transport=None):
        self.episode_ids = tuple(episode_ids)
        if len(self.episode_ids) != 3 or set(self.episode_ids) != set(EPISODES):
            raise ValueError("local pilot requires the same three seed0/layout0 episode IDs")
        if model != MODEL:
            raise ValueError(f"this pilot is frozen to {MODEL}")
        for name, value, upper in (("max_iterations", max_iterations, 8),
                                   ("max_input_tokens", max_input_tokens, 128_000),
                                   ("max_output_tokens", max_output_tokens, 8192)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be within1..{upper}")
        if not isinstance(request_timeout_s, (int, float)) or not math.isfinite(request_timeout_s) or not 1 <= request_timeout_s <= 400:
            raise ValueError("request timeout must be within1..400 seconds")
        if type(sampling_seed) is not int or sampling_seed != 0:
            raise ValueError("pilot sampling_seed is fixed to0")
        if api_documentation is not None and (not isinstance(api_documentation, str) or not api_documentation.strip()):
            raise ValueError("API documentation must be nonempty")
        # Validate URL even for a CPU test's injected fake transport.
        local_http = LoopbackHTTP(base_url, float(request_timeout_s), tokenize_timeout_s)
        self._transport = _transport or local_http
        self.model = model
        self.max_iterations = max_iterations
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.api_documentation = api_documentation
        self.history = {e: [] for e in self.episode_ids}
        self.task_instructions = {}
        self.episode_api_documents = {}
        self.finished = set()
        self.stopped = False
        self.calls = dict.fromkeys(self.episode_ids, 0)
        self.completed_calls = 0
        self.total_prompt_tokens = self.total_completion_tokens = 0
        self.total_generation_wall_s = self.total_tokenize_wall_s = 0.0
        self._call_lock = threading.Lock()
        self.client = self  # Existing serve_relay uses driver.client.snapshot().
        self.path = Path(usage_jsonl).absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Never resume or overwrite a measured suite, even after an unknown result.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        self.config = {
            "purpose": "aspire_robodojo_local_qwen_pilot", "model": model,
            "episode_ids": list(self.episode_ids), "base_url": base_url,
            "max_iterations": max_iterations, "max_input_tokens": max_input_tokens,
            "max_output_tokens": max_output_tokens, "request_timeout_s": request_timeout_s,
            "tokenize_timeout_s": tokenize_timeout_s,
            "request_timeout_scope": "HTTP timeout: count<=60s plus generation<=400s; relay480s, host1200s",
            "reasoning_effort": "medium", "chat_template_kwargs": THINKING,
            "sampling": dict(SAMPLING),
            "input_cap_enforcement": "mandatory vLLM /tokenize with multimodal preprocessing",
            "historical_reasoning": "omitted to match Astra final-JSON-only history",
            "output_cap_includes_reasoning": True, "remote_api_calls": False,
            "native_tools": "disabled; tools omitted because vLLM0.32 rejects an empty array",
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "shared_prompt_source_sha256": hashlib.sha256(Path(sys.modules[ProgramProposal.__module__].__file__).read_bytes()).hexdigest(),
        }
        self._append("suite_initialized", config=self.config)

    def _append(self, event, **fields):
        record = {"event": event, "utc": datetime.now(timezone.utc).isoformat(), **fields}
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(_canonical(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def snapshot(self):
        return {"model": self.model, "calls": dict(self.calls),
                "completed_calls": self.completed_calls, "aborted": self.stopped,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens_including_reasoning": self.total_completion_tokens,
                "generation_wall_s": self.total_generation_wall_s,
                "tokenize_wall_s": self.total_tokenize_wall_s,
                "max_calls_per_episode": self.max_iterations,
                "api_cost_usd": 0, "accounting": "local inference; electricity/GPU opportunity cost not estimated"}

    def next_program(self, episode_id, task_instruction, public_observation,
                     previous_result=None, api_documentation=None):
        if not self._call_lock.acquire(blocking=False):
            raise DriverStopped("concurrent model generation is forbidden")
        try:
            return self._next_program(episode_id, task_instruction, public_observation,
                                      previous_result, api_documentation)
        except BaseException as exc:
            self.stopped = True
            self._append("driver_stopped", error_type=type(exc).__name__,
                         reason=str(exc)[:1000], no_retry=True)
            raise
        finally:
            self._call_lock.release()

    def _next_program(self, episode_id, task_instruction, public_observation,
                      previous_result, api_documentation):
        if self.stopped or episode_id in self.finished:
            raise DriverStopped("driver or episode has stopped")
        if episode_id not in self.history:
            raise ValueError("episode outside the frozen pilot")
        count = self.calls[episode_id]
        if count >= self.max_iterations:
            self.finished.add(episode_id)
            raise DriverStopped("episode code-generation iteration limit reached")
        if not isinstance(task_instruction, str) or not task_instruction.strip() or len(task_instruction) > 20_000:
            raise ValueError("invalid public task instruction")
        if self.task_instructions.setdefault(episode_id, task_instruction) != task_instruction:
            raise ValueError("cannot change task inside measured episode")
        documentation = api_documentation or self.api_documentation
        if not isinstance(documentation, str) or not documentation.strip():
            raise ValueError("deployed API documentation is required")
        if self.episode_api_documents.setdefault(episode_id, documentation) != documentation:
            raise ValueError("API documentation changed inside episode")
        if bool(count) != (previous_result is not None):
            raise ValueError("feedback is required exactly after the first program")
        feedback = None if previous_result is None else _bounded_json(previous_result)
        # Match model_driver.py's current-turn text byte for byte.
        current = [{"type": "input_text", "text": json.dumps({
            "task_instruction": task_instruction, "iteration": count + 1,
            "iterations_remaining_including_this": self.max_iterations - count,
            "previous_execution": feedback, "episode_continues_without_reset": True,
        }, sort_keys=True)}] + observation_content(public_observation)
        messages = [{"role": "system", "content": SYSTEM_PREFIX + documentation}]
        messages += self.history[episode_id] + [{"role": "user", "content": chat_content(current)}]
        payload, tokenize = build_chat_requests(messages, model=self.model,
                                                max_output_tokens=self.max_output_tokens)
        self._append("program_input", episode_id=episode_id, iteration=count + 1,
                     request=redact(payload), request_sha256=_sha(payload),
                     instructions_sha256=hashlib.sha256((SYSTEM_PREFIX + documentation).encode()).hexdigest())
        t0 = time.monotonic()
        token_result = self._transport("/tokenize", tokenize)
        tokenize_wall_s = time.monotonic() - t0
        self.total_tokenize_wall_s += tokenize_wall_s
        prompt_count = _integer(token_result.get("count"), "tokenize.count")
        model_len = _integer(token_result.get("max_model_len"), "tokenize.max_model_len")
        self._append("input_token_count", episode_id=episode_id, iteration=count + 1,
                     count=prompt_count, max_model_len=model_len, wall_s=tokenize_wall_s,
                     source="vllm_multimodal_preprocessed_tokenize", approximate=False,
                     request_sha256=_sha(tokenize))
        if prompt_count > self.max_input_tokens:
            raise DriverStopped("input token cap reached; no generation requested")
        if prompt_count + self.max_output_tokens > model_len:
            raise DriverStopped("server context cannot reserve full output budget; no truncation")
        self.calls[episode_id] += 1
        self._append("generation_started", episode_id=episode_id, iteration=count + 1,
                     counted_input_tokens=prompt_count)
        t0 = time.monotonic()
        try:
            response = self._transport("/v1/chat/completions", payload)
        except BaseException as exc:
            elapsed = time.monotonic() - t0
            self.total_generation_wall_s += elapsed
            self._append("generation_transport_failure", episode_id=episode_id,
                         iteration=count + 1, generation_wall_s=elapsed,
                         error_type=type(exc).__name__, token_usage_unknown=True, no_retry=True)
            raise
        generation_wall_s = time.monotonic() - t0
        self.total_generation_wall_s += generation_wall_s
        self._append("raw_response", episode_id=episode_id, iteration=count + 1,
                     generation_wall_s=generation_wall_s, response=redact(response))
        if response.get("model") != self.model:
            raise DriverStopped("response model differs from frozen served model name")
        usage = response.get("usage")
        if not isinstance(usage, dict):
            raise DriverStopped("server did not report token usage")
        actual_prompt = _integer(usage.get("prompt_tokens"), "usage.prompt_tokens")
        completion = _integer(usage.get("completion_tokens"), "usage.completion_tokens")
        self.total_prompt_tokens += actual_prompt
        self.total_completion_tokens += completion
        count_matches = actual_prompt == prompt_count
        measured_usage = {**usage, "input_tokens": actual_prompt,
                          "output_tokens": completion, "generation_wall_s": generation_wall_s,
                          "input_count_wall_s": tokenize_wall_s,
                          "preflight_input_tokens": prompt_count,
                          "preflight_matches_generation_usage": count_matches,
                          "token_count_source": "server_reported", "token_count_approximate": False,
                          "output_includes_reasoning": True, "api_cost_usd": 0}
        self._append("generation_usage", episode_id=episode_id, iteration=count + 1,
                     usage=measured_usage)
        if not count_matches or actual_prompt > self.max_input_tokens or completion > self.max_output_tokens:
            raise DriverStopped("server token accounting differs from enforced preflight or caps")
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or choices[0].get("finish_reason") != "stop":
            raise DriverStopped("generation incomplete or not a single stopped response")
        message = choices[0].get("message", {})
        if message.get("tool_calls") or message.get("function_call"):
            raise DriverStopped("native tool calls are not part of this code-as-policy protocol")
        raw = message.get("content")
        if not isinstance(raw, str):
            raise DriverStopped("model final content is missing; check reasoning parser")
        try:
            parsed = parse_program_proposal(raw)
        except (ValueError, TypeError):
            raise DriverStopped("invalid final JSON/program; no repair or retry") from None
        # Save raw reasoning in audit response, but never add it to history.
        self.history[episode_id] += [{"role": "user", "content": chat_content(current)},
                                    {"role": "assistant", "content": raw}]
        self.completed_calls += 1
        if parsed["give_up"]:
            self.finished.add(episode_id)
        return ProgramProposal(**parsed, iteration=count + 1, response_id=response.get("id"),
                               usage=measured_usage, budget=self.snapshot())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--relay-dir", type=Path)
    parser.add_argument("--relay-idle-timeout-s", type=float, default=1800)
    args = parser.parse_args()
    driver = LocalQwenDriver(**json.loads(args.config.read_text()))
    if args.relay_dir:
        from model_relay import serve_relay
        serve_relay(driver, args.relay_dir, idle_timeout_s=args.relay_idle_timeout_s)
        return
    print(json.dumps({"ready": True, "budget": driver.snapshot()}), flush=True)
    for line in sys.stdin:
        try:
            value = json.loads(line)
            if value == {"command": "stop"}:
                print(json.dumps({"stopped": True, "budget": driver.snapshot()}), flush=True)
                return
            print(json.dumps({"proposal": asdict(driver.next_program(**value))}), flush=True)
        except Exception as exc:
            print(json.dumps({"stopped": True, "error_type": type(exc).__name__}), flush=True)
            return


if __name__ == "__main__":
    main()
