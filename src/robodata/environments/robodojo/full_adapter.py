"""Full-task metadata and compact progress around the unchanged robot bridge."""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import time

from cap_harness.contracts import TaskContext
from robodata.environments.robodojo.robodojo_adapter import RoboDojoAdapter


@lru_cache(maxsize=1)
def task_records():
    """Use the pinned benchmark's own inventory, without importing Isaac tasks."""
    root = Path(os.environ["ROBODOJO_ROOT"])
    path = root / "scripts/internal/task_inventory.py"
    spec = importlib.util.spec_from_file_location("full_eval_task_inventory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = [r for r in module.build_inventory()["tasks"] if r["runnable"]]
    return {r["name"]: {**r, "task_id": i} for i, r in enumerate(records)}


class FullRoboDojoAdapter(RoboDojoAdapter):
    """Same public robot actions/sensing as the pilot, for every native task."""

    def __init__(self, task_env, *, progress_path=None):
        self._progress_path = Path(progress_path) if progress_path else None
        super().__init__(task_env, recorder=None)

    def _set_context(self, raw):
        instruction = raw.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("RoboDojo did not supply authoritative task language")
        record = task_records()[self._env.task_name]
        self._task_context = TaskContext(
            suite="RoboDojo", task_id=record["task_id"],
            task_name=self._env.task_name, language=instruction,
            family=record["dimension"],
            metadata={"robot": "dual_arx_x5", "control_hz": self.control_frequency,
                      "episode_max_steps": int(self._env.step_lim),
                      "episode_steps_used": int(self._env.take_action_cnt[0]),
                      "episode_steps_remaining": max(0, int(self._env.step_lim-self._env.take_action_cnt[0])),
                      "end_effector_reference": "native flange orientation; grasp-point origin +0.1501m flange X"})

    def step(self, action):
        result = super().step(action)
        if self._progress_path is not None:
            payload = {"phase": "executing_program", "unix_time": time.time(),
                       "environment_steps": int(self._env.take_action_cnt[0]),
                       "step_limit": int(self._env.step_lim)}
            temporary = self._progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload))
            temporary.replace(self._progress_path)
        return result
