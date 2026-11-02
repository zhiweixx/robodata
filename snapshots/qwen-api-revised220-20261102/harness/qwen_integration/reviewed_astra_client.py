"""Fail-closed, serial Responses client for the four-episode Astra pilot.

No credentials or SDK are touched at import/construction. Runtime uses the
official input-token-count endpoint before every inference. Prices verified
2026-10-14: https://developers.openai.com/api/docs/models/gpt-6-astra
Count contract: https://developers.openai.com/api/docs/guides/token-counting
Cache formula: https://developers.openai.com/api/docs/guides/prompt-caching

The durable JSONL is also the budget ledger. Keep the SAME file for the entire
suite; deleting/replacing it or launching another suite bypasses this guard.
Provider prices/counts/limits must honor their documented contract. The guard
covers this client's Standard token charges, not taxes, unrelated account use,
or external tool charges (hosted tools are disabled). No invoice is inferred
from a timeout: its entire reservation stays charged and the suite stops.
"""

import base64
import contextlib
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import ssl
import stat
import threading
import time
import uuid


MODEL = "gpt-6-astra"
BASE_URL = "https://api.openai.com/v1"
MAX_INPUT_TOKENS = 128000
MAX_OUTPUT_TOKENS = 4096
MAX_CALLS_PER_EPISODE = 100
REQUEST_DEADLINE_S = 180
EPISODE_BUDGET = Decimal("24")
SUITE_BUDGET = Decimal("96")
# Assume every input token is a cache write, the highest Standard input rate.
REQUEST_RESERVATION = Decimal("1.8048")
DEFAULT_EPISODE_IDS = (
    "safelibero_spatial_I_t0_e0_s0", "safelibero_object_I_t0_e0_s0",
    "safelibero_goal_I_t0_e0_s0", "safelibero_long_I_t0_e0_s0",
)


class BudgetExceeded(RuntimeError):
    """End this episode without another inference; other slots may continue."""


class SuiteAborted(RuntimeError):
    """Stop all episodes. Do not retry with another client or ledger."""


class _DeadlineExceeded(BaseException):
    pass


def _integer(value):
    if type(value) is not int or value < 0:
        raise ValueError("invalid token count")
    return value


def price_usage(usage):
    """Return (Decimal USD, breakdown); output_tokens already includes reasoning.

    Missing cache-write detail is charged conservatively as cache writes,
    never silently interpreted as zero. Raw usage is separately preserved.
    """
    inp = _integer(usage["input_tokens"])
    out = _integer(usage["output_tokens"])
    details = usage.get("input_tokens_details") or {}
    cached = _integer(details.get("cached_tokens", 0))
    write_known = "cache_write_tokens" in details
    written = _integer(details["cache_write_tokens"]) if write_known else inp - cached
    if written < 0 or cached + written > inp:
        raise ValueError("inconsistent cache token counts")
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    if reasoning is not None and _integer(reasoning) > out:
        raise ValueError("inconsistent reasoning token counts")
    if "total_tokens" in usage and _integer(usage["total_tokens"]) != inp + out:
        raise ValueError("inconsistent total token count")
    ordinary = inp - cached - written
    long_context = inp > 272000
    input_factor = Decimal(2 if long_context else 1)
    output_factor = Decimal("1.5" if long_context else "1")
    cost = ((ordinary * Decimal("10") + cached * Decimal("1") +
             written * Decimal("12.5")) * input_factor +
            out * Decimal("50") * output_factor) / Decimal(1000000)
    return cost, {
        "input_tokens": inp, "ordinary_input_tokens": ordinary,
        "cached_input_tokens": cached, "cache_write_tokens": written,
        "cache_write_count_reported": write_known, "output_tokens": out,
        "reasoning_tokens": reasoning, "long_context": long_context,
        "cost_is_conservative": not write_known,
    }


def extract_text(response):
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    return "".join(part.get("text", "")
                   for item in response.get("output", [])
                   if item.get("type") == "message"
                   for part in item.get("content", [])
                   if part.get("type") == "output_text")


def _redact(value, key=None):
    if isinstance(value, dict):
        return {_redact(str(k), key): _redact(v, key) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, key) for v in value]
    if isinstance(value, str):
        if key:
            value = value.replace(key, "[REDACTED]")
        # A cluster CONNECT proxy can contain userinfo; never emit it in URLs.
        value = re.sub(r"(?i)\b(https?://)[^/\s]*@", r"\1[REDACTED]@", value)
        return re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", value)
    return value


def _read_key():
    # Only called from create(), never from a module import or constructor.
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        path = os.environ.get("OPENAI_API_KEY_FILE")
        if not path:
            raise SuiteAborted("API credential is not configured")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise SuiteAborted("API credential file must be owned by this user and private")
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                fd = None
                key = handle.read(16385).strip()
        finally:
            if fd is not None:
                os.close(fd)
    if not key or len(key) > 16384 or any(c.isspace() for c in key):
        raise SuiteAborted("API credential is invalid")
    return key


def _make_sdk(key):
    import openai
    # SDK 3.x uses HTTPX2. Select the SDK's public compatible client rather
    # than importing an independently installed, potentially wrong transport.
    # The legacy export supports older SDKs without HTTPX2.
    client_type = getattr(openai, "DefaultHttpx2Client", None)
    if client_type is None:
        client_type = getattr(openai, "DefaultHttpxClient", None)
    if client_type is None:
        raise SuiteAborted("SDK has no supported synchronous HTTP client export")
    # trust_env=False intentionally ignores SSL_CERT_FILE as well as proxies.
    # Select the administrator-provided CA bundle explicitly, without weakening
    # certificate or hostname validation. Invalid bundles fail before a request.
    tls_context = ssl.create_default_context(cafile=os.environ.get("SAFEHARNESS_CA_FILE") or None)
    # Use only an explicitly approved CONNECT proxy. Do not inherit unrelated
    # environment proxies, redirects, API gateways, or provider overrides.
    transport_options = dict(trust_env=False, follow_redirects=False,
                             timeout=REQUEST_DEADLINE_S, verify=tls_context)
    proxy = os.environ.get("SAFEHARNESS_PROXY", "").strip()
    if proxy:
        transport_options["proxy"] = proxy
    transport = client_type(**transport_options)
    try:
        return openai.OpenAI(api_key=key, base_url=BASE_URL, max_retries=0,
                             timeout=REQUEST_DEADLINE_S, http_client=transport)
    except BaseException:
        transport.close()
        raise


@contextlib.contextmanager
def _deadline():
    if threading.current_thread() is not threading.main_thread():
        raise SuiteAborted("Astra client requires the main thread for a hard deadline")
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise SuiteAborted("An existing process alarm prevents the request deadline")
    old = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise _DeadlineExceeded()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, REQUEST_DEADLINE_S)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _as_dict(value):
    if isinstance(value, dict):
        return value
    return value.model_dump(mode="json")


def _validate_input(value):
    """Only immutable inline context; no mutable hosted conversations or URLs."""
    if isinstance(value, str):
        return
    if not isinstance(value, list):
        raise ValueError("input must be a string or inline item list")
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("invalid input item")
        kind = item.get("type", "message")
        if kind in ("function_call", "function_call_output", "reasoning"):
            # Restrict nested inputs too; tool output must be plain text.
            if kind == "function_call_output" and not isinstance(item.get("output"), str):
                raise ValueError("function output must be text")
            continue
        if kind != "message" or item.get("role") not in ("system", "developer", "user", "assistant"):
            raise ValueError("unsupported input item")
        content = item.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ValueError("invalid message content")
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("invalid content part")
            if part.get("type") in ("input_text", "output_text") and isinstance(part.get("text"), str):
                continue
            if part.get("type") == "input_image":
                url = part.get("image_url", "")
                match = re.fullmatch(r"data:image/(?:png|jpeg);base64,([A-Za-z0-9+/=]+)", url)
                if not match or "file_id" in part:
                    raise ValueError("images must be immutable inline PNG/JPEG data URLs")
                base64.b64decode(match.group(1), validate=True)
                continue
            raise ValueError("only text and inline PNG/JPEG inputs are allowed")


class AstraClient:
    def __init__(self, usage_jsonl, episode_ids=DEFAULT_EPISODE_IDS, *, _sdk_factory=None):
        self.path = Path(usage_jsonl).absolute()
        self.episode_ids = tuple(episode_ids)
        if len(set(self.episode_ids)) != 4 or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", e) for e in self.episode_ids):
            raise ValueError("exactly four distinct episode IDs are required")
        self._sdk_factory = _sdk_factory or _make_sdk
        self._poisoned = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            records = self._read()
            if not records:
                self._append("suite_initialized", config=self._config())
            elif records[0].get("config") != self._config():
                raise SuiteAborted("Budget ledger configuration does not match this suite")

    def _config(self):
        return {"episode_ids": list(self.episode_ids), "model": MODEL,
                "reasoning_effort": "medium", "service_tier": "default",
                "max_input_tokens": MAX_INPUT_TOKENS, "max_output_tokens": MAX_OUTPUT_TOKENS,
                "max_calls_per_episode": MAX_CALLS_PER_EPISODE,
                "episode_budget_usd": str(EPISODE_BUDGET), "suite_budget_usd": str(SUITE_BUDGET),
                "request_reservation_usd": str(REQUEST_RESERVATION),
                "prices_per_million_usd": {"input": "10", "cached": "1", "cache_write": "12.5", "output": "50"}}

    @contextlib.contextmanager
    def _locked(self):
        fd = os.open(str(self.path) + ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SuiteAborted("Another process is using this suite ledger") from None
            yield
        finally:
            os.close(fd)

    def _read(self):
        if not self.path.exists():
            return []
        if self.path.is_symlink():
            raise SuiteAborted("Budget ledger cannot be a symlink")
        try:
            data = self.path.read_bytes()
            if data and not data.endswith(b"\n"):
                raise ValueError("partial record")
            return [json.loads(line) for line in data.splitlines()]
        except Exception:
            raise SuiteAborted("Budget ledger is incomplete or invalid; do not reset it") from None

    def _append(self, event, key=None, **fields):
        record = dict(schema_version=1, event=event,
                      utc=datetime.now(timezone.utc).isoformat(), monotonic_s=time.monotonic(), **fields)
        encoded = (json.dumps(_redact(record, key), separators=(",", ":"), allow_nan=False) + "\n").encode()
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "ab", buffering=0) as handle:
                remaining = memoryview(encoded)
                while remaining:
                    written = handle.write(remaining)
                    if not written:
                        raise OSError("short journal write")
                    remaining = remaining[written:]
                os.fsync(handle.fileno())
            dirfd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        except Exception:
            self._poisoned = True
            raise SuiteAborted("Durable budget logging failed; do not make another request") from None

    def _state(self):
        records = self._read()
        if not records or records[0].get("config") != self._config():
            raise SuiteAborted("Suite ledger was replaced or removed")
        starts, settled, counts_open = {}, {}, set()
        aborted = self._poisoned
        for rec in records[1:]:
            event, attempt = rec["event"], rec.get("attempt_id")
            if event == "token_count_started":
                counts_open.add(attempt)
            elif event in ("token_count_received", "token_count_error"):
                counts_open.discard(attempt)
            elif event == "request_started":
                if attempt in starts:
                    raise SuiteAborted("Duplicate request in ledger")
                starts[attempt] = rec
            elif event == "request_settled":
                if attempt not in starts or attempt in settled:
                    raise SuiteAborted("Invalid settlement in ledger")
                amount = Decimal(rec["charged_usd"])
                if not amount.is_finite() or amount < 0 or amount > REQUEST_RESERVATION:
                    raise SuiteAborted("Invalid budget amount in ledger")
                settled[attempt] = amount
            elif event == "suite_aborted":
                aborted = True
        amounts = {e: Decimal(0) for e in self.episode_ids}
        calls = {e: 0 for e in self.episode_ids}
        for attempt, rec in starts.items():
            episode = rec["episode_id"]
            amounts[episode] += settled.get(attempt, REQUEST_RESERVATION)
            calls[episode] += 1
        return {"episode_charged_usd": amounts, "suite_charged_usd": sum(amounts.values(), Decimal(0)),
                "calls": calls, "unsettled_attempts": list(set(starts) - set(settled)),
                "aborted": aborted or bool(counts_open)}

    def snapshot(self):
        with self._locked():
            state = self._state()
            return json.loads(json.dumps(state, default=str))

    def _abort(self, reason, episode_id, attempt_id, key=None, **fields):
        self._poisoned = True
        self._append("suite_aborted", key=key, reason=reason, episode_id=episode_id,
                     attempt_id=attempt_id, **fields)
        raise SuiteAborted("Astra suite aborted: " + reason) from None

    def create(self, episode_id, *, input, instructions=None, tools=None, text=None, tool_choice=None):
        if episode_id not in self.episode_ids:
            raise ValueError("episode is not in the authorized suite")
        payload = {"model": MODEL, "input": input, "reasoning": {"effort": "medium"}, "truncation": "disabled"}
        for name, value in (("instructions", instructions), ("tools", tools), ("text", text), ("tool_choice", tool_choice)):
            if value is not None:
                payload[name] = value
        # Freeze caller-owned data so count and create see identical context.
        payload = json.loads(json.dumps(payload, allow_nan=False))
        _validate_input(payload["input"])
        if any(t.get("type") != "function" for t in payload.get("tools", [])):
            raise ValueError("hosted or billable tools are not permitted")
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        attempt = uuid.uuid4().hex
        with self._locked():
            state = self._state()
            if state["aborted"] or state["unsettled_attempts"]:
                raise SuiteAborted("Suite is aborted or has unknown billing; do not resume")
            if state["calls"][episode_id] >= MAX_CALLS_PER_EPISODE:
                raise BudgetExceeded("Episode model-call limit reached")
            if state["episode_charged_usd"][episode_id] + REQUEST_RESERVATION > EPISODE_BUDGET:
                raise BudgetExceeded("Episode cannot reserve the next request within $24")
            if state["suite_charged_usd"] + REQUEST_RESERVATION > SUITE_BUDGET:
                raise BudgetExceeded("Suite cannot reserve the next request within $96")
            sdk, key = None, None
            try:
                key = _read_key()
                sdk = self._sdk_factory(key)
                self._append("token_count_started", episode_id=episode_id, attempt_id=attempt,
                             endpoint=BASE_URL + "/responses/input_tokens", payload_sha256=digest)
                started = time.monotonic()
                try:
                    with _deadline():
                        counted_obj = sdk.responses.input_tokens.count(**payload)
                        counted = _as_dict(counted_obj)
                    n_input = _integer(counted["input_tokens"])
                    self._append("token_count_received", key=key, episode_id=episode_id, attempt_id=attempt,
                                 input_tokens=n_input, latency_s=time.monotonic() - started,
                                 request_id=getattr(counted_obj, "_request_id", None))
                except BaseException as error:
                    self._append("token_count_error", key=key, episode_id=episode_id, attempt_id=attempt,
                                 error_type=type(error).__name__, latency_s=time.monotonic() - started,
                                 http_status=getattr(error, "status_code", None),
                                 request_id=getattr(error, "request_id", None))
                    self._abort("input token counting failed", episode_id, attempt)
                if n_input > MAX_INPUT_TOKENS:
                    self._append("request_rejected", episode_id=episode_id, attempt_id=attempt,
                                 reason="input_token_cap", counted_input_tokens=n_input)
                    raise BudgetExceeded("Episode input exceeds 128,000 tokens; inference was not sent")
                self._append("request_started", episode_id=episode_id, attempt_id=attempt,
                             model=MODEL, reasoning_effort="medium", service_tier="default",
                             endpoint=BASE_URL + "/responses", counted_input_tokens=n_input,
                             max_output_tokens=MAX_OUTPUT_TOKENS, payload_sha256=digest,
                             reserved_usd=str(REQUEST_RESERVATION),
                             episode_charged_with_reservation_usd=str(state["episode_charged_usd"][episode_id] + REQUEST_RESERVATION),
                             suite_charged_with_reservation_usd=str(state["suite_charged_usd"] + REQUEST_RESERVATION))
                started = time.monotonic()
                try:
                    with _deadline():
                        response_obj = sdk.responses.create(**payload, max_output_tokens=MAX_OUTPUT_TOKENS,
                                                           service_tier="default", store=False,
                                                           stream=False, background=False)
                        response = _as_dict(response_obj)
                    # Persist returned usage BEFORE interpreting model output or releasing money.
                    self._append("response_received", key=key, episode_id=episode_id, attempt_id=attempt,
                                 response_id=response.get("id"), request_id=getattr(response_obj, "_request_id", None),
                                 model=response.get("model"), status=response.get("status"),
                                 service_tier=response.get("service_tier"), usage_raw=response.get("usage"),
                                 incomplete_details=response.get("incomplete_details"),
                                 latency_s=time.monotonic() - started)
                except BaseException as error:
                    self._append("request_error", key=key, episode_id=episode_id, attempt_id=attempt,
                                 error_type=type(error).__name__, latency_s=time.monotonic() - started,
                                 http_status=getattr(error, "status_code", None),
                                 request_id=getattr(error, "request_id", None),
                                 billing_unknown=True, retained_reservation_usd=str(REQUEST_RESERVATION))
                    self._abort("request failed with unknown billing", episode_id, attempt)
                try:
                    cost, breakdown = price_usage(response["usage"])
                    if response.get("model") != MODEL or response.get("service_tier") != "default":
                        raise ValueError("unverified model or service tier")
                    if breakdown["input_tokens"] > MAX_INPUT_TOKENS or breakdown["output_tokens"] > MAX_OUTPUT_TOKENS:
                        raise ValueError("provider token limit mismatch")
                    if cost > REQUEST_RESERVATION:
                        raise ValueError("provider cost exceeds reserved bound")
                    if response.get("status") not in ("completed", "incomplete", "failed", "cancelled"):
                        raise ValueError("nonterminal response")
                except Exception:
                    self._abort("response billing could not be validated", episode_id, attempt,
                                retained_reservation_usd=str(REQUEST_RESERVATION))
                self._append("request_settled", episode_id=episode_id, attempt_id=attempt,
                             charged_usd=str(cost), estimated_cost_usd=str(cost), token_breakdown=breakdown,
                             episode_charged_usd=str(state["episode_charged_usd"][episode_id] + cost),
                             suite_charged_usd=str(state["suite_charged_usd"] + cost))
                if response.get("status") in ("failed", "cancelled"):
                    self._abort("model response failed or was cancelled", episode_id, attempt)
                return response
            except (BudgetExceeded, SuiteAborted):
                raise
            except BaseException:
                self._abort("client setup or execution failed", episode_id, attempt)
            finally:
                if sdk is not None:
                    try:
                        sdk.close()
                    except Exception:
                        pass
