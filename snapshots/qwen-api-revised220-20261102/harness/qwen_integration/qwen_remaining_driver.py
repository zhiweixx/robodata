"""Allow only the untouched alignment ID; delegate policy generation unchanged."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from local_model_driver import LocalQwenDriver
from model_driver import DriverStopped
from model_relay import serve_relay
from qwen_remaining_audit import EPISODE, require, verify_evidence


class AlignmentOnlyDriver:
    def __init__(self, driver):
        self._driver = driver
        self.client = self
        self.episode_ids = (EPISODE,)
        self.max_iterations = driver.max_iterations

    def snapshot(self):
        return {**self._driver.snapshot(), "allowed_episode_ids": [EPISODE],
                "excluded_attempted_episode_ids": ["general_pickup_s0_l0", "stack_bowls_s0_l0"],
                "scope": "one untouched episode; prior usage remains in original ledger"}

    def next_program(self, episode_id, task_instruction, public_observation,
                     previous_result=None, api_documentation=None):
        if episode_id != EPISODE:
            raise DriverStopped("continuation forbids every attempted or unauthorized episode")
        return self._driver.next_program(episode_id=episode_id, task_instruction=task_instruction,
            public_observation=public_observation, previous_result=previous_result,
            api_documentation=api_documentation)


def validate_controller_config(config, audit):
    verify_evidence(audit)
    require("claim" in audit, "continuation has no permanent claim")
    claim = json.loads(Path(audit["claim"]["path"]).read_text())
    output = Path(claim["continuation_output"])
    require(Path(config["usage_jsonl"]).resolve() == output / "usage.jsonl",
            "controller ledger is outside the claimed continuation")
    expected = {**audit["controller_config"], "usage_jsonl": config["usage_jsonl"],
                "base_url": config["base_url"]}
    require(config == expected, "continuation changed policy settings")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--relay-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    audit = json.loads(args.audit.read_text())
    validate_controller_config(config, audit)
    driver = LocalQwenDriver(**config)
    driver._append("continuation_scope", prior_run_id=audit["prior_run_id"],
                   allowed_episode_ids=[EPISODE], prior_calls=audit["prior_calls"], claim=audit["claim"])
    serve_relay(AlignmentOnlyDriver(driver), args.relay_dir, idle_timeout_s=1800)


if __name__ == "__main__":
    main()
