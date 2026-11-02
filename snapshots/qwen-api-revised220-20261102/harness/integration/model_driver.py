"""Budgeted Astra code generation, deliberately separate from robot execution.

This module never imports or executes ASPIRE-generated Python. The caller passes
only public sensor observations and public tool receipts. Run ``--config FILE``
as a dedicated API-controller process and exchange JSON lines over stdin/stdout;
do not pass API credentials into the simulator/program-execution process.

The reviewed SafeLIBERO client supplies its durable reservation ledger, exact
token preflight, zero retries, and stop-on-unknown-billing behavior unchanged.
An isolated module instance adapts its original four-slot suite configuration.
Prices/schema checked against official OpenAI documentation on 2026-10-19:
https://developers.openai.com/api/docs/models/gpt-6-astra
https://developers.openai.com/api/docs/guides/structured-outputs
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Any
import uuid


DEFAULT_CLIENT_SOURCE = Path(__file__).resolve().parents[2] / "safeharness_pilot" / "astra_client.py"
PUBLIC_OBSERVATION_KEYS = frozenset({
    "robot_state", "camera_calibration", "images", "elapsed_sim_s", "remaining_actions",
})
FORBIDDEN_DATA_KEYS = frozenset({
    "ground_truth", "groundtruth", "gt_pose", "gt_poses", "object_pose", "object_poses",
    "task_success", "success_oracle", "reward", "oracle", "evaluator", "privileged",
})
OUTPUT_FORMAT = {
    "format": {
        "type": "json_schema", "name": "aspire_program", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "code": {"type": "string"},
                "summary": {"type": "string"},
                "done": {"type": "boolean"},
            },
            "required": ["code", "summary", "done"],
        },
    },
}

SYSTEM_PREFIX = """You are writing a bounded Python robot-control program for the genuine ASPIRE runtime on RoboDojo simulation. Use only the API documented below and the provided public RGB/RGBD observations, calibration, robot proprioception and tool feedback. No hidden object poses, evaluator rewards, success oracle, environment internals, files, network, shell or simulator reset are available.

Each program executes on the SAME current episode. Another iteration continues from the resulting state; it never starts over. Use bounded loops and short movement segments. Assign a JSON-compatible `result` describing your observation or next intent. Import only math if needed. Do not write imports for the provided API types/functions.

Action receipts report command execution, not object contact or task success. A closed-gripper command does not establish a grasp. Use observations and actual robot-state fields to assess progress. Do not invent force or contact measurements. Pose position is the gripper grasp-center in world metres, quaternion_wxyz is the native flange orientation. Preserve the observed orientation when testing a small move; do not assume a canonical grasp rotation without supporting observations.

Return exactly the requested JSON object. `code` is executable Python without markdown fences; `summary` is a brief description, not private reasoning. Set `done=true` only to request ending the episode based on your observations (it is not an evaluator label). If done=true, code must be empty. Never use code to reset, restart or access the evaluator.

Available runtime API:
"""


class DriverStopped(RuntimeError):
    """The caller must finish the suite without resending this request."""


@dataclass(frozen=True)
class ProgramProposal:
    code: str
    summary: str
    done: bool
    iteration: int
    response_id: str | None
    usage: dict[str, Any]
    budget: dict[str, Any]


def _bounded_json(value: Any, *, max_bytes: int = 200_000) -> Any:
    """Freeze public input and reject accidentally supplied evaluator fields."""
    serialized = json.dumps(value, allow_nan=False)
    if len(serialized.encode()) > max_bytes:
        raise ValueError("public text/feedback exceeds the configured size cap")
    frozen = json.loads(serialized)

    def check(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if key.lower() in FORBIDDEN_DATA_KEYS:
                    raise ValueError("privileged/evaluator field is forbidden in model input")
                check(nested)
        elif isinstance(item, list):
            for nested in item:
                check(nested)
    check(frozen)
    return frozen


def observation_content(observation: dict[str, Any]) -> list[dict[str, Any]]:
    """Build immutable API content from an explicit public observation schema.

    `images` is a list of {camera, data_url}; each data URL must be inline PNG or
    JPEG. Raw depth arrays stay in the execution worker; models may receive an
    explicitly rendered depth image or public tool-derived depth measurement.
    This structural check is not an information-flow proof: the adapter remains
    responsible for sourcing all supplied values from public APIs.
    """
    if not isinstance(observation, dict) or set(observation) - PUBLIC_OBSERVATION_KEYS:
        raise ValueError("unknown public observation fields")
    text_fields = {k: v for k, v in observation.items() if k != "images"}
    content = [{"type": "input_text", "text": json.dumps(_bounded_json(text_fields), sort_keys=True)}]
    images = observation.get("images", [])
    if not isinstance(images, list) or len(images) > 8:
        raise ValueError("at most eight public camera images are allowed")
    for item in images:
        if not isinstance(item, dict) or set(item) != {"camera", "data_url"}:
            raise ValueError("image must contain only camera and data_url")
        camera, url = item["camera"], item["data_url"]
        if not isinstance(camera, str) or len(camera) > 100:
            raise ValueError("invalid camera label")
        if not isinstance(url, str) or len(url) > 16_000_000:
            raise ValueError("image exceeds size cap")
        if not re.fullmatch(r"data:image/(?:png|jpeg);base64,[A-Za-z0-9+/=]+", url):
            raise ValueError("only inline PNG/JPEG camera images are allowed")
        content.extend([
            {"type": "input_text", "text": "Public camera: " + camera},
            {"type": "input_image", "image_url": url, "detail": "high"},
        ])
    return content


def _load_budget_client(source: Path, *, suite_budget: Decimal,
                        episode_budget: Decimal, max_iterations: int,
                        max_input_tokens: int, max_output_tokens: int):
    """Adapt configuration without mutating the original module or pilot files."""
    source = source.resolve(strict=True)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("_aspire_budget_" + uuid.uuid4().hex, source)
    if spec is None or spec.loader is None:
        raise DriverStopped("Cannot load the reviewed budget client")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SUITE_BUDGET = suite_budget
    module.EPISODE_BUDGET = episode_budget
    module.MAX_CALLS_PER_EPISODE = max_iterations
    module.MAX_INPUT_TOKENS = max_input_tokens
    module.MAX_OUTPUT_TOKENS = max_output_tokens
    module.REQUEST_RESERVATION = (Decimal(max_input_tokens) * Decimal("12.5") +
                                  Decimal(max_output_tokens) * Decimal("50")) / Decimal(1_000_000)

    class PilotClient(module.AstraClient):
        def __init__(self, usage_jsonl, episode_ids, *, _sdk_factory=None):
            # The only replaced behavior is original exactly-four-slot validation.
            self.path = Path(usage_jsonl).absolute()
            self.episode_ids = tuple(episode_ids)
            if not 1 <= len(self.episode_ids) <= 3 or len(set(self.episode_ids)) != len(self.episode_ids):
                raise ValueError("one to three distinct episode IDs are required")
            if any(not isinstance(e, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", e) for e in self.episode_ids):
                raise ValueError("invalid episode identifier")
            self._sdk_factory = _sdk_factory or module._make_sdk
            self._poisoned = False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._locked():
                records = self._read()
                if not records:
                    self._append("suite_initialized", config=self._config())
                elif records[0].get("config") != self._config():
                    raise module.SuiteAborted("Budget ledger configuration does not match this suite")

        def _config(self):
            config = super()._config()
            config.update(purpose="aspire_robodojo_pilot", client_source_sha256=source_sha256,
                          driver_schema_version=1)
            return config

    return module, PilotClient


class BudgetedAstraDriver:
    """One controller for an entire authorized suite; no automatic restart/retry.

    Inputs to next_program:
      episode_id: exact authorized ID; task_instruction: public task text;
      public_observation: robot_state/calibration/images/time/action budget;
      previous_result: public execution receipt of the immediately prior program.
    Output: ProgramProposal. The caller executes `.code` through upstream ASPIRE,
    then supplies its public receipt and refreshed observation on the next call.
    """

    def __init__(self, usage_jsonl, episode_ids, api_documentation: str | None = None, *,
                 suite_budget_usd="20", episode_budget_usd=None, max_iterations=4,
                 max_input_tokens=128_000, max_output_tokens=8192,
                 client_source=DEFAULT_CLIENT_SOURCE, _sdk_factory=None):
        suite_budget = Decimal(str(suite_budget_usd))
        episode_budget = Decimal(str(episode_budget_usd or suite_budget_usd))
        if not suite_budget.is_finite() or not Decimal(0) < suite_budget <= Decimal(50):
            raise ValueError("suite budget must be positive and at most $50")
        if not episode_budget.is_finite() or not Decimal(0) < episode_budget <= suite_budget:
            raise ValueError("episode budget must be within the suite budget")
        if type(max_iterations) is not int or not 1 <= max_iterations <= 20:
            raise ValueError("one to twenty bounded code iterations are required")
        if type(max_input_tokens) is not int or not 1 <= max_input_tokens <= 128_000:
            raise ValueError("input token cap must be at most 128,000")
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 16_384:
            raise ValueError("output token cap must be at most 16,384")
        if api_documentation is not None and (not isinstance(api_documentation, str) or not api_documentation.strip()):
            raise ValueError("deployed ASPIRE API documentation must be nonempty")
        self._module, cls = _load_budget_client(
            Path(client_source), suite_budget=suite_budget, episode_budget=episode_budget,
            max_iterations=max_iterations, max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        )
        self.client = cls(usage_jsonl, episode_ids, _sdk_factory=_sdk_factory)
        state = self.client.snapshot()
        if state["aborted"] or state["unsettled_attempts"] or any(state["calls"].values()):
            raise DriverStopped("A previously started measured suite cannot be resumed or reset")
        self.api_documentation = api_documentation
        self.max_iterations = max_iterations
        self.history = {e: [] for e in self.client.episode_ids}
        self.task_instructions: dict[str, str] = {}
        self.episode_api_documents: dict[str, str] = {}
        self.finished: set[str] = set()
        self.stopped = False

    def next_program(self, episode_id: str, task_instruction: str,
                     public_observation: dict[str, Any], previous_result=None,
                     api_documentation: str | None = None) -> ProgramProposal:
        if self.stopped or episode_id in self.finished:
            raise DriverStopped("Driver or episode has already stopped")
        if episode_id not in self.history:
            raise ValueError("episode is not in the authorized suite")
        count = self.client.snapshot()["calls"][episode_id]
        if count >= self.max_iterations:
            self.finished.add(episode_id)
            raise DriverStopped("Episode code-generation iteration limit reached")
        if not isinstance(task_instruction, str) or not task_instruction.strip() or len(task_instruction) > 20_000:
            raise ValueError("invalid public task instruction")
        previous_instruction = self.task_instructions.setdefault(episode_id, task_instruction)
        if previous_instruction != task_instruction:
            raise ValueError("cannot change the task inside a measured episode")
        documentation = api_documentation or self.api_documentation
        if not isinstance(documentation, str) or not documentation.strip():
            raise ValueError("actual deployed ASPIRE API documentation is required before inference")
        if self.episode_api_documents.setdefault(episode_id, documentation) != documentation:
            raise ValueError("runtime API documentation cannot change inside an episode")
        if count and previous_result is None:
            raise ValueError("execution feedback is required before another program")
        if not count and previous_result is not None:
            raise ValueError("first iteration cannot have prior execution feedback")
        feedback = None if previous_result is None else _bounded_json(previous_result)
        current = [{"type": "input_text", "text": json.dumps({
            "task_instruction": task_instruction,
            "iteration": count + 1,
            "iterations_remaining_including_this": self.max_iterations - count,
            "previous_execution": feedback,
            "episode_continues_without_reset": True,
        }, sort_keys=True)}] + observation_content(public_observation)
        messages = self.history[episode_id] + [{"role": "user", "content": current}]
        with self.client._locked():
            self.client._append("program_input", episode_id=episode_id, iteration=count + 1,
                                input=messages,
                                api_documentation=documentation if count == 0 else None,
                                instructions_sha256=hashlib.sha256((SYSTEM_PREFIX + documentation).encode()).hexdigest())
        try:
            response = self.client.create(
                episode_id, input=messages, instructions=SYSTEM_PREFIX + documentation,
                text=OUTPUT_FORMAT, tools=[],
            )
        except self._module.BudgetExceeded as exc:
            self.finished.add(episode_id)
            raise DriverStopped("Episode cannot reserve another call within its token/call/dollar caps") from exc
        except BaseException:
            self.stopped = True
            raise
        # Usage has already been durably settled. Never execute partial JSON/code.
        raw = self._module.extract_text(response)
        with self.client._locked():
            self.client._append("program_output", episode_id=episode_id, iteration=count + 1,
                                response_id=response.get("id"), output_text=raw)
        try:
            if response.get("status") != "completed":
                raise ValueError("incomplete model output")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict) or set(parsed) != {"code", "summary", "done"}:
                raise ValueError("invalid output shape")
            if not isinstance(parsed["code"], str) or not isinstance(parsed["summary"], str) or type(parsed["done"]) is not bool:
                raise ValueError("invalid output fields")
            if parsed["done"] and parsed["code"].strip():
                raise ValueError("done responses cannot also execute code")
            if not parsed["done"] and not parsed["code"].strip():
                raise ValueError("empty program")
            if "```" in parsed["code"]:
                raise ValueError("markdown is not executable code")
        except (ValueError, TypeError):
            self.stopped = True
            with self.client._locked():
                self.client._append("driver_stopped", episode_id=episode_id, reason="invalid_or_incomplete_program")
            raise DriverStopped("Model output is incomplete or invalid; no automatic retry") from None
        self.history[episode_id] = messages + [{"role": "assistant", "content": raw}]
        if parsed["done"]:
            self.finished.add(episode_id)
        return ProgramProposal(**parsed, iteration=count + 1, response_id=response.get("id"),
                               usage=response.get("usage", {}), budget=self.client.snapshot())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--relay-dir", type=Path)
    parser.add_argument("--relay-idle-timeout-s", type=float, default=1800)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    driver = BudgetedAstraDriver(**config)
    if args.relay_dir is not None:
        from model_relay import serve_relay
        serve_relay(driver, args.relay_dir, idle_timeout_s=args.relay_idle_timeout_s)
        return
    print(json.dumps({"ready": True, "budget": driver.client.snapshot()}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request == {"command": "stop"}:
                print(json.dumps({"stopped": True, "budget": driver.client.snapshot()}), flush=True)
                return
            allowed = {"episode_id", "task_instruction", "public_observation", "previous_result", "api_documentation"}
            if not isinstance(request, dict) or set(request) - allowed:
                raise ValueError("invalid model-driver request")
            proposal = driver.next_program(**request)
            print(json.dumps({"ok": True, "proposal": asdict(proposal)}), flush=True)
        except BaseException as exc:
            # Never echo SDK exceptions, payloads, credential values or paths.
            print(json.dumps({"ok": False, "error_type": type(exc).__name__,
                              "stop_suite": True, "no_retry": True}), flush=True)
            return


if __name__ == "__main__":
    main()
