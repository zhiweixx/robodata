"""Fail-closed accounting for a single-node Terra RoboDojo campaign.

This module makes no network requests and never scores an episode. Call reserve()
before sending an API request, leave its short file-lock transaction, then perform
network work. Call settle() with raw provider usage even when output is invalid.
Unknown billing remains reserved and stops admission unless a separately preserved
human decision explicitly acknowledges specific existing unknown reservations.
Acknowledgment does not settle or release any unknown liability. An exhausted budget pauses
the campaign; callers must not convert it to a native model failure.

Use one durable campaign path across resumes. flock must be tested on the actual
filesystem with independent processes before deployment. Turbo/NFS durability and
locking are not established by local tests. All concurrent writers must be on one
host in one allocation (or use a single budget broker); never run two allocations
against this ledger. A node-local-only ledger is not durable across node loss.

Amounts estimate documented Standard API list prices, excluding taxes, unrelated
project activity, and unannounced/provider-specific adjustments. They are not an
invoice guarantee. Reserve the full native model maxima, not a client token cap.
"""
from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import uuid


MODEL = "gpt-6.1-terra"
NATIVE_CONTEXT = 1_050_000
NATIVE_OUTPUT = 128_000
LONG_INPUT_THRESHOLD = 272_000
MILLION = Decimal(1_000_000)
NANO_USD = Decimal(1_000_000_000)
PRICING = {
    "model": MODEL, "service_tier": "default", "currency": "USD",
    "input_per_million": "1.00", "cached_per_million": "0.05",
    "cache_write_30m_per_million": "1.25",
    "cache_write_12h_per_million": "2.50", "output_per_million": "5.00",
    "long_input_threshold_exclusive": LONG_INPUT_THRESHOLD,
    "long_input_multiplier": "2", "long_output_multiplier": "1.5",
    "native_context_tokens": NATIVE_CONTEXT, "native_max_output_tokens": NATIVE_OUTPUT,
    "source": "https://developers.openai.com/api/docs/models/gpt-6.1-terra",
    "verified_date": "2026-10-30",
}
WORST_REQUEST_USD = (
    Decimal(NATIVE_CONTEXT) * Decimal("2.50") * 2
    + Decimal(NATIVE_OUTPUT) * Decimal("5.00") * Decimal("1.5")
) / MILLION  # 6.21; accounting reservation only, never sent as a token limit.


class LedgerError(RuntimeError):
    pass


class LedgerCorrupt(LedgerError):
    pass


class BudgetUnavailable(LedgerError):
    """Pause admission and preserve unfinished episodes as unscored."""


class AdmissionBlocked(LedgerError):
    """An uncertain charge or unsafe owner transition needs reconciliation."""


class AccountingError(LedgerError):
    """Provider usage cannot be safely settled; its reservation remains held."""


def canonical(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True,
                      separators=(",", ":"))


def _money(value):
    if isinstance(value, (bool, float)):
        raise ValueError("Money must be a Decimal, integer or decimal string")
    try:
        result = Decimal(value)
    except Exception as exc:
        raise ValueError("Invalid monetary amount") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("Money must be finite and nonnegative")
    return result


def _text(value, field):
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(field + " must be a nonempty string of at most 256 characters")
    return value


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise AccountingError(name + " must be a nonnegative integer")
    return value


def _nano(value):
    amount = _money(value) * NANO_USD
    if amount != amount.to_integral_value():
        raise AccountingError("Price is not representable in nanodollars")
    return int(amount)


def price_usage(usage, *, cache_ttl="30m"):
    """Return a conservative Standard-price settlement, keeping raw usage separate.

    cache_ttl must match the actual request; do not pass per-block overrides or
    other price-affecting modes. Missing cache details use the largest possible
    write charge for that request. Reasoning is a subset of output, not added.
    """
    if cache_ttl not in {"30m", "12h"}:
        raise AccountingError("Unsupported cache TTL")
    if not isinstance(usage, dict):
        raise AccountingError("Raw usage must be an object")
    inp = _integer(usage.get("input_tokens"), "input_tokens")
    out = _integer(usage.get("output_tokens"), "output_tokens")
    if inp > NATIVE_CONTEXT or out > NATIVE_OUTPUT:
        raise AccountingError("Provider usage exceeds documented native limits")
    if "total_tokens" in usage and _integer(usage["total_tokens"], "total_tokens") != inp + out:
        raise AccountingError("total_tokens does not equal input plus output")
    details = usage.get("input_tokens_details")
    if details is None:
        details = {}
    if not isinstance(details, dict):
        raise AccountingError("input_tokens_details must be an object")
    conservative = []
    if "cached_tokens" not in details:
        cached = 0
        conservative.append("cached_tokens_missing_assume_no_cache_discount")
    else:
        cached = _integer(details["cached_tokens"], "cached_tokens")
    if cached > inp:
        raise AccountingError("cached_tokens exceeds input_tokens")
    if "cache_write_tokens" not in details:
        writes = inp - cached
        conservative.append("cache_write_tokens_missing_price_all_noncached_as_writes")
    else:
        writes = _integer(details["cache_write_tokens"], "cache_write_tokens")
    if cached + writes > inp:
        raise AccountingError("Cached plus written tokens exceeds input_tokens")
    output_details = usage.get("output_tokens_details")
    if output_details is None:
        output_details = {}
    if not isinstance(output_details, dict):
        raise AccountingError("output_tokens_details must be an object")
    reasoning = output_details.get("reasoning_tokens")
    if reasoning is not None and _integer(reasoning, "reasoning_tokens") > out:
        raise AccountingError("Reasoning tokens exceeds total output_tokens")
    ordinary = inp - cached - writes
    long_context = inp > LONG_INPUT_THRESHOLD
    input_factor = Decimal(2 if long_context else 1)
    output_factor = Decimal("1.5") if long_context else Decimal(1)
    write_rate = Decimal("1.25" if cache_ttl == "30m" else "2.50")
    input_usd = (Decimal(ordinary) + Decimal(cached) * Decimal("0.05")
                 + Decimal(writes) * write_rate) * input_factor / MILLION
    output_usd = Decimal(out) * 5 * output_factor / MILLION
    return {
        "estimated_standard_list_price_usd": str(input_usd + output_usd),
        "input_estimate_usd": str(input_usd), "output_estimate_usd": str(output_usd),
        "input_tokens": inp, "output_tokens_including_reasoning": out,
        "reasoning_tokens_subset": reasoning, "cached_tokens": cached,
        "cache_write_tokens_priced": writes, "ordinary_input_tokens_priced": ordinary,
        "cache_ttl": cache_ttl, "long_context_full_request": long_context,
        "conservative_assumptions": conservative,
        "accounting_status": "conservative_standard_usage" if conservative else "priced_standard_usage",
        "invoice_verified": False,
    }


class BudgetLedger:
    """A fsynced, hash-chained append log with locks only around accounting."""

    def __init__(self, path, *, cap_usd="2000", host=None, single_writer_cache=False):
        self.path = Path(path)
        self.cap = _money(cap_usd)
        if not self.cap:
            raise ValueError("Campaign cap must be positive")
        self.host = _text(host or socket.gethostname(), "host")
        # Only the lifetime-lock-owning broker enables this. Auditors always
        # replay every byte. A changed file under exclusive ownership fails.
        self._cache_enabled = bool(single_writer_cache)
        self._cache_state = self._cache_signature = None
        self._cache_digest = None
        self._cache_refresh_count = 0
        self._cache_refresh_events = []
        self._cache_failed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        with self._lock():
            if not self.path.exists():
                self._create()
            self._read()

    @contextmanager
    def _lock(self):
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.lock_path, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _sync_dir(self):
        fd = os.open(self.path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _record(self, event, seq, previous):
        body = {"seq": seq, "previous_sha256": previous, "event": event}
        return {**body, "sha256": hashlib.sha256(canonical(body).encode()).hexdigest()}

    def _create(self):
        event = {"kind": "created", "ledger_id": str(uuid.uuid4()), "created_unix": time.time(),
                 "cap_usd": str(self.cap), "pricing": PRICING,
                 "reservation_usd": str(WORST_REQUEST_USD), "format_version": 1}
        row = self._record(event, 0, None)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "wb") as file:
            file.write((canonical(row) + "\n").encode())
            file.flush()
            os.fsync(file.fileno())
        self._sync_dir()

    def _read(self):
        cache = getattr(self, '_cache_enabled', False)
        if cache and self._cache_failed:
            raise LedgerCorrupt("Single-writer cache invalidated; preserve journal and reconcile")
        if cache and self._cache_state is not None:
            try:
                self._cache_signature = self._validated_signature(
                    self._cache_signature, self._cache_digest, reason='cached_read')
            except BaseException:
                self._cache_failed = True
                raise
            return self._cache_state
        fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as file:
            before = self._signature(os.fstat(file.fileno()))
            raw = file.read()
            after = self._signature(os.fstat(file.fileno()))
        if before != after or after != self._signature(self.path.stat()):
            raise LedgerCorrupt("Journal changed while being verified")
        if not raw or not raw.endswith(b"\n"):
            raise LedgerCorrupt("Empty or partially written ledger; preserve and reconcile")
        state = {"requests": {}, "seq": -1, "head": None, "halt_reason": None,
                 "pause_reason": None, "settled_nano": 0, "held_nano": 0,
                 "active_ids": set(), "unknown_ids": set(),
                 "acknowledged_unknown_ids": set(), "unknown_resume_authorizations": {}}
        try:
            for seq, line in enumerate(raw.splitlines()):
                row = json.loads(line)
                if set(row) != {"seq", "previous_sha256", "event", "sha256"}:
                    raise ValueError("Unexpected record keys")
                if row["seq"] != seq or row["previous_sha256"] != state["head"]:
                    raise ValueError("Broken sequence or predecessor")
                if self._record(row["event"], seq, state["head"]) != row:
                    raise ValueError("Broken record hash")
                self._apply(state, row["event"], first=(seq == 0))
                state["seq"], state["head"] = seq, row["sha256"]
        except (ValueError, KeyError, TypeError, AccountingError, ArithmeticError) as exc:
            raise LedgerCorrupt("Invalid ledger: " + str(exc)) from exc
        if cache:
            self._cache_state, self._cache_signature = state, after
            self._cache_digest = hashlib.sha256(raw)
        return state

    @staticmethod
    def _signature(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def _validated_signature(self, expected, expected_digest, *, reason):
        """Accept timestamp-only drift only after proving identical bytes.

        File timestamps are change hints, not the financial source of truth.
        Inode/size changes and any content mismatch still fail closed. This
        rare full byte check does not replay or accept an external transaction.
        Normal unchanged metadata keeps the constant-time cached path.
        """
        observed = self._signature(self.path.lstat())
        if observed[:3] != expected[:3]:
            raise LedgerCorrupt('Durable journal identity or size changed outside its exclusive writer')
        if observed == expected:
            return observed
        for attempt in range(1, 4):
            fd = os.open(self.path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            actual_digest = hashlib.sha256()
            with os.fdopen(fd, 'rb') as file:
                opened = self._signature(os.fstat(file.fileno()))
                if opened[:3] != expected[:3]:
                    raise LedgerCorrupt('Durable journal identity or size changed during metadata verification')
                while True:
                    block = file.read(1024 * 1024)
                    if not block:
                        break
                    actual_digest.update(block)
                finished = self._signature(os.fstat(file.fileno()))
            final = self._signature(self.path.lstat())
            if finished[:3] != expected[:3] or final[:3] != expected[:3]:
                raise LedgerCorrupt('Durable journal identity or size changed during metadata verification')
            if expected_digest is None or actual_digest.digest() != expected_digest.digest():
                raise LedgerCorrupt('Durable journal content changed outside its exclusive writer')
            # A digest of old bytes must not bless a newer timestamp following
            # a concurrent same-size write. Require one stable full-stat window;
            # harmless metadata churn can revalidate at most three times.
            if opened == finished == final:
                break
        else:
            raise LedgerCorrupt('Durable journal metadata did not stabilize during byte verification')
        self._cache_refresh_count += 1
        self._cache_refresh_events.append({
            'kind': 'unchanged_journal_metadata_refreshed', 'reason': reason,
            'recorded_unix': time.time(), 'verified_bytes': final[2],
            'full_read_attempts': attempt,
            'journal_sha256': actual_digest.hexdigest(),
            'previous_mtime_ns': expected[3], 'previous_ctime_ns': expected[4],
            'observed_mtime_ns': observed[3], 'observed_ctime_ns': observed[4],
            'refreshed_mtime_ns': final[3], 'refreshed_ctime_ns': final[4]})
        del self._cache_refresh_events[:-16]
        return final

    def cache_integrity_diagnostics(self):
        """Owner diagnostics only; no journal writes or financial status change."""
        return {'single_writer_cache_enabled': bool(self._cache_enabled),
                'cache_failed': bool(self._cache_failed),
                'metadata_refresh_count': self._cache_refresh_count,
                'recent_metadata_refreshes': [dict(e) for e in self._cache_refresh_events]}

    def _apply(self, state, event, *, first=False):
        kind = event["kind"]
        if first:
            if (kind != "created" or event["format_version"] != 1 or event["pricing"] != PRICING
                    or _money(event["cap_usd"]) != self.cap
                    or _money(event["reservation_usd"]) != WORST_REQUEST_USD):
                raise ValueError("Ledger configuration differs from approved policy")
            state["ledger_id"] = event["ledger_id"]
            return
        if kind == "reserved":
            rid = event["request_id"]
            if state["halt_reason"]:
                raise ValueError("Reservation admitted after an accounting conflict")
            if state["pause_reason"]:
                raise ValueError("Reservation admitted after financial admission paused")
            if rid in state["requests"] or _money(event["reservation_usd"]) != WORST_REQUEST_USD:
                raise ValueError("Invalid duplicate or amount in reservation")
            if event["cache_ttl"] not in {"30m", "12h"}:
                raise ValueError("Invalid reservation cache policy")
            if self._totals(state)["exposure"] + WORST_REQUEST_USD > self.cap:
                raise ValueError("Reservation exceeds campaign cap")
            if state['unknown_ids'] - state['acknowledged_unknown_ids']:
                raise ValueError("Reservation admitted with unacknowledged unresolved usage")
            for pending_id in state['active_ids']:
                pending = state['requests'][pending_id]
                if (
                    pending["host"] != event["host"] or pending["allocation_id"] != event["allocation_id"]
                ):
                    raise ValueError("Concurrent allocation or host transition")
            state["requests"][rid] = {**event, "state": "active"}
            state['held_nano'] += _nano(event['reservation_usd'])
            state['active_ids'].add(rid)
        elif kind == "unknown":
            request = state["requests"][event["request_id"]]
            if request["state"] == "settled":
                raise ValueError("Settled request cannot become unknown")
            request.update(state="unknown", unknown_event=event)
            state['active_ids'].discard(event['request_id'])
            state['unknown_ids'].add(event['request_id'])
            # A later unknown event requires a fresh human decision even for
            # an identifier previously acknowledged. Never acknowledge futures.
            state['acknowledged_unknown_ids'].discard(event['request_id'])
        elif kind == "settled":
            request = state["requests"][event["request_id"]]
            if request["state"] == "settled":
                raise ValueError("Duplicate settlement record")
            if event["model"] != MODEL or event["service_tier"] != "default":
                raise ValueError("Unpriced model or service tier")
            expected = price_usage(event["raw_usage"], cache_ttl=request["cache_ttl"])
            if expected != event["price"]:
                raise ValueError("Recorded price does not match raw usage")
            amount = _money(expected["estimated_standard_list_price_usd"])
            if amount > _money(request["reservation_usd"]):
                raise ValueError("Cost exceeds reservation")
            request.update(state="settled", settlement=event)
            state['held_nano'] -= _nano(request['reservation_usd'])
            state['settled_nano'] += _nano(amount)
            state['active_ids'].discard(event['request_id'])
            state['unknown_ids'].discard(event['request_id'])
            state['acknowledged_unknown_ids'].discard(event['request_id'])
        elif kind == "unknown_exposure_resume_authorized":
            authorization = event["authorization"]
            self._validate_unknown_resume(state, authorization)
            aid = authorization["approval_id"]
            state['acknowledged_unknown_ids'].update(authorization['request_ids'])
            state['unknown_resume_authorizations'][aid] = authorization
            state['pause_reason'] = None
            # No settlement, token counters, reservation amount or request state
            # changes. The entire worst-case unknown liability remains held.
        elif kind == "accounting_conflict":
            if event["request_id"] not in state["requests"]:
                raise ValueError("Accounting conflict refers to unknown request")
            state["halt_reason"] = event["reason"]
        elif kind == "admission_paused":
            if state['pause_reason'] is not None:
                raise ValueError("Duplicate admission pause event")
            state['pause_reason'] = _text(event['reason'], 'reason')
        else:
            raise ValueError("Unknown ledger event")

    def _append(self, state, event):
        # Validate the prospective transition before writing a durable event.
        # Copy only mutable indexes and the affected request; settled historical
        # records are immutable. Avoid serializing the entire history per call.
        trial = dict(state)
        trial['requests'] = dict(state['requests'])
        rid = event.get('request_id')
        if rid in trial['requests']:
            trial['requests'][rid] = dict(trial['requests'][rid])
        trial['active_ids'] = set(state['active_ids'])
        trial['unknown_ids'] = set(state['unknown_ids'])
        trial['acknowledged_unknown_ids'] = set(state['acknowledged_unknown_ids'])
        trial['unknown_resume_authorizations'] = dict(state['unknown_resume_authorizations'])
        self._apply(trial, event)
        row = self._record(event, state["seq"] + 1, state["head"])
        encoded = (canonical(row) + "\n").encode()
        cache = getattr(self, '_cache_enabled', False)
        if cache:
            try:
                if self._cache_failed or self._cache_state is not state:
                    raise LedgerCorrupt("Exclusive writer lost its verified journal state")
                self._cache_signature = self._validated_signature(
                    self._cache_signature, self._cache_digest, reason='before_append')
            except BaseException:
                self._cache_failed = True
                raise
        next_digest = self._cache_digest.copy() if cache else None
        if cache:
            next_digest.update(encoded)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0))
        try:
            with os.fdopen(fd, "wb") as file:
                if cache and self._signature(os.fstat(file.fileno()))[:3] != self._cache_signature[:3]:
                    raise LedgerCorrupt('Journal identity or size changed before durable append')
                file.write(encoded)
                file.flush()
                os.fsync(file.fileno())
                after = self._signature(os.fstat(file.fileno()))
            if cache:
                if (after[:2] != self._cache_signature[:2]
                        or after[2] != self._cache_signature[2] + len(encoded)):
                    raise LedgerCorrupt("Journal changed during durable append")
                after = self._validated_signature(after, next_digest, reason='after_append')
        except BaseException:
            if cache:
                self._cache_failed = True
            raise
        trial["seq"], trial["head"] = row["seq"], row["sha256"]
        if cache:
            self._cache_state, self._cache_signature = trial, after
            self._cache_digest = next_digest
        return trial

    @staticmethod
    def _totals(state):
        settled = Decimal(state['settled_nano']) / NANO_USD
        held = Decimal(state['held_nano']) / NANO_USD
        return {"settled": settled, "held": held, "exposure": settled + held}

    def status(self):
        with self._lock():
            state = self._read()
        return self._status(state)

    def _status(self, state):
        totals = self._totals(state)
        counts = {'active': len(state['active_ids']), 'unknown': len(state['unknown_ids'])}
        counts['settled'] = len(state['requests']) - counts['active'] - counts['unknown']
        unacknowledged = state['unknown_ids'] - state['acknowledged_unknown_ids']
        result = {"ledger_id": state["ledger_id"], "cap_usd": str(self.cap),
                "settled_estimate_usd": str(totals["settled"]), "held_reservations_usd": str(totals["held"]),
                "accounted_exposure_usd": str(totals["exposure"]),
                "unreserved_usd": str(self.cap - totals["exposure"]), "request_counts": counts,
                "admission_blocked_unknown": bool(unacknowledged),
                "admission_blocked_integrity": bool(state["halt_reason"]),
                "halt_reason": state["halt_reason"],
                "admission_paused": bool(state['pause_reason']),
                "admission_pause_reason": state['pause_reason'],
                "new_full_native_request_fits": not unacknowledged and not state["halt_reason"] and not state['pause_reason'] and totals["exposure"] + WORST_REQUEST_USD <= self.cap,
                "head_sha256": state["head"], "invoice_verified": False}
        # Preserve exact historical status shape for pre-authorization ledgers.
        if state['unknown_resume_authorizations']:
            result['acknowledged_unknown_requests'] = len(state['unknown_ids'] & state['acknowledged_unknown_ids'])
            result['unacknowledged_unknown_requests'] = len(unacknowledged)
        return result

    def reserve(self, request_id, *, payload_sha256, allocation_id, worker_id, cache_ttl="30m"):
        request_id = _text(request_id, "request_id")
        allocation_id = _text(allocation_id, "allocation_id")
        worker_id = _text(worker_id, "worker_id")
        if (not isinstance(payload_sha256, str) or len(payload_sha256) != 64
                or any(c not in "0123456789abcdef" for c in payload_sha256)):
            raise ValueError("payload_sha256 must be a lowercase SHA256 digest")
        if cache_ttl not in {"30m", "12h"}:
            raise ValueError("Unsupported cache TTL")
        with self._lock():
            state = self._read()
            if state["halt_reason"]:
                raise AdmissionBlocked("Accounting conflict requires reconciliation")
            if state['pause_reason']:
                raise BudgetUnavailable("Financial admission paused: " + state['pause_reason'])
            # A second reserve is deliberately not permission to send twice.
            if request_id in state["requests"]:
                raise AdmissionBlocked("Request ID already reserved; never resend it implicitly")
            if state['unknown_ids'] - state['acknowledged_unknown_ids']:
                raise AdmissionBlocked("Unknown provider charges require reconciliation or explicit human acknowledgment")
            for pending_id in state['active_ids']:
                request = state['requests'][pending_id]
                if (
                    request["host"] != self.host or request["allocation_id"] != allocation_id
                ):
                    raise AdmissionBlocked("Unsettled requests belong to another host/allocation")
            if self._totals(state)["exposure"] + WORST_REQUEST_USD > self.cap:
                self._append(state, {'kind': 'admission_paused', 'reason': 'insufficient_unreserved_budget',
                                     'recorded_unix': time.time()})
                raise BudgetUnavailable("Full native request does not fit; pause, do not score a failure")
            event = {"kind": "reserved", "request_id": request_id, "payload_sha256": payload_sha256,
                     "allocation_id": allocation_id, "worker_id": worker_id, "host": self.host,
                     "reservation_usd": str(WORST_REQUEST_USD), "cache_ttl": cache_ttl,
                     "created_unix": time.time()}
            self._append(state, event)
            return {**event, "may_send_once": True}

    def _validate_unknown_resume(self, state, authorization):
        """Validate an explicit operator record; this does not obtain consent.

        A trusted human operator must first preserve the actual human approval
        whose artifact digest is recorded here. Runtime/model clients have no
        RPC for this operation. The ledger verifies the financial scope and
        replay binding, not the human identity outside the computer.
        """
        required = {'schema_version', 'authority', 'decision', 'approval_id',
                    'approval_record_sha256', 'approval_reference', 'ledger_id',
                    'expected_head_sha256', 'request_ids', 'expected_pause_reason',
                    'cap_usd', 'reason'}
        if not isinstance(authorization, dict) or set(authorization) != required:
            raise ValueError('Explicit human-authorization record has invalid fields')
        if (authorization['schema_version'] != 1 or type(authorization['schema_version']) is not int
                or authorization['authority'] != 'human_user'
                or authorization['decision'] != 'resume_retaining_unknown_reservations'):
            raise ValueError('Human approval to retain unknown exposure is required')
        for key in ('approval_id', 'approval_reference', 'reason', 'ledger_id'):
            _text(authorization[key], key)
        for key in ('approval_record_sha256', 'expected_head_sha256'):
            value = authorization[key]
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('Authorization digest is invalid')
        if (authorization['ledger_id'] != state['ledger_id']
                or authorization['expected_head_sha256'] != state['head']
                or authorization['approval_id'] in state['unknown_resume_authorizations']):
            raise ValueError('Authorization belongs to a different ledger/head or was already used')
        if _money(authorization['cap_usd']) != self.cap:
            raise ValueError('Human acknowledgment cannot change the existing cap')
        expected = sorted(state['unknown_ids'] - state['acknowledged_unknown_ids'])
        ids = authorization['request_ids']
        if (not isinstance(ids, list) or not ids or ids != expected
                or any(not isinstance(rid, str) for rid in ids)):
            raise ValueError('Authorization must name exactly the current unacknowledged unknown requests')
        if (state['active_ids'] or state['halt_reason']
                or state['pause_reason'] != 'supervisor_financial_pause_drain'
                or authorization['expected_pause_reason'] != state['pause_reason']):
            raise ValueError('All requests must be drained without an integrity conflict before acknowledgment')
        if self._totals(state)['exposure'] + WORST_REQUEST_USD > self.cap:
            raise ValueError('Full retained exposure leaves insufficient budget for another native request')

    def authorize_resume_with_unknown_exposure(self, authorization):
        """Owner-only explicit human decision. Never call automatically.

        Atomically acknowledges only existing unknown IDs and clears the known
        drain pause. Unknown balances stay fully held; new unknowns block again.
        The caller must independently verify actual human consent and its saved
        approval artifact. This method must not be exposed to worker RPC.
        """
        authorization = json.loads(canonical(authorization))
        with self._lock():
            state = self._read()
            self._validate_unknown_resume(state, authorization)
            state = self._append(state, {
                'kind': 'unknown_exposure_resume_authorized',
                'authorization': authorization, 'recorded_unix': time.time()})
            return {'authorization_recorded': True,
                    'approval_id': authorization['approval_id'],
                    'acknowledged_request_ids': list(authorization['request_ids']),
                    'unknown_exposure_was_settled': False,
                    'status': self._status(state)}

    def pause_admission(self, reason):
        reason = _text(reason, 'reason')
        with self._lock():
            state = self._read()
            if state['pause_reason'] is None:
                state = self._append(state, {'kind': 'admission_paused', 'reason': reason,
                                             'recorded_unix': time.time()})
            return {'admission_paused': True, 'reason': state['pause_reason']}

    def mark_unknown(self, request_id, *, reason, raw_usage=None):
        reason = _text(reason, "reason")
        raw_usage = json.loads(canonical(raw_usage))
        with self._lock():
            state = self._read()
            request = state["requests"].get(request_id)
            if request is None:
                raise LedgerError("Unknown request ID")
            if request["state"] == "settled":
                raise LedgerError("Cannot discard a settled charge")
            event = {"kind": "unknown", "request_id": request_id, "reason": reason,
                     "raw_usage": raw_usage, "recorded_unix": time.time()}
            self._append(state, event)
        return {"state": "unknown", "held_usd": request["reservation_usd"]}

    def settle(self, request_id, *, usage, response_id, model=MODEL, service_tier="default"):
        try:
            raw_usage = json.loads(canonical(usage))
        except (ValueError, TypeError):
            # Non-JSON values have no raw wire representation. Preserve a marker.
            raw_usage = {"invalid_non_json_usage": True}
        error = None
        with self._lock():
            state = self._read()
            request = state["requests"].get(request_id)
            if request is None:
                raise LedgerError("Settlement has no prior reservation")
            if request["state"] == "settled":
                existing = request["settlement"]
                if (existing["raw_usage"] == raw_usage and existing["response_id"] == response_id
                        and existing["model"] == model and existing["service_tier"] == service_tier):
                    return {"state": "settled", "idempotent": True, **existing["price"]}
                self._append(state, {"kind": "accounting_conflict", "request_id": request_id,
                                    "reason": "conflicting_duplicate_settlement", "raw_usage": raw_usage,
                                    "response_id": response_id, "model": model,
                                    "service_tier": service_tier, "recorded_unix": time.time()})
                raise AccountingError("Conflicting duplicate settlement; preserve both evidence records")
            try:
                try:
                    _text(response_id, "response_id")
                except ValueError as exc:
                    raise AccountingError("Missing or invalid response ID") from exc
                if model != MODEL or service_tier != "default":
                    raise AccountingError("Unpriced model or service tier")
                price = price_usage(raw_usage, cache_ttl=request["cache_ttl"])
                if _money(price["estimated_standard_list_price_usd"]) > _money(request["reservation_usd"]):
                    raise AccountingError("Cost exceeds reservation")
            except AccountingError as exc:
                error = exc
                self._append(state, {"kind": "unknown", "request_id": request_id,
                                    "reason": str(exc), "raw_usage": raw_usage,
                                    "response_id": response_id, "model": model,
                                    "service_tier": service_tier, "recorded_unix": time.time()})
            if error is None:
                self._append(state, {"kind": "settled", "request_id": request_id,
                                    "raw_usage": raw_usage, "price": price, "response_id": response_id,
                                    "model": model, "service_tier": service_tier, "recorded_unix": time.time()})
        if error is not None:
            raise error
        return {"state": "settled", "idempotent": False, **price}


def read_only_snapshot(path, *, cap_usd="2000"):
    """Full verified replay without creating locks, directories, or files."""
    ledger = object.__new__(BudgetLedger)
    ledger.path = Path(path)
    ledger.cap = _money(cap_usd)
    ledger._cache_enabled = False
    state = ledger._read()
    return {'status': ledger._status(state), 'requests': state['requests'],
            'ledger_id': state['ledger_id'], 'head_sha256': state['head'],
            'record_count': state['seq'] + 1,
            'unknown_resume_authorizations': state['unknown_resume_authorizations']}
