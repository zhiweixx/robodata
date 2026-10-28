"""Shared public-observation and program contracts; no paid API or pilot runner."""
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class ProgramProposal:
    code: str
    summary: str
    done: bool
    iteration: int
    response_id: str | None
    usage: dict[str, Any]
    budget: dict[str, Any]

