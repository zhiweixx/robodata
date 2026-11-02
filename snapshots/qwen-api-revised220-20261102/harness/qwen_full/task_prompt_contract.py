"""Pinned official public task requirements for the ASPIRE RoboDojo adapter.

This module reads documentation only. It never queries native reward, object
state, episode outcomes, a skill library, or an external service.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

TASK_CONTRACT_VERSION = "robodojo_public_task_contract_v2"
UPSTREAM_COMMIT = "3c0bc68383a75bf3fa2d7bfc5ab1d1241c7e171e"
RECIPE_DIR = Path(__file__).resolve().parent / "official_task_recipes"
RUNTIME_ADAPTATION_VERSION = "aspire_recipe_interface_clarifications_v1"
WAIT_RECIPES = frozenset({"imitate_sorting_sequence", "make_kong", "play_tic_tac_toe"})
HOLD_PROGRAM = '''state = get_robot_state()
hold = RobotAction({"right": ArmCommand(mode="joint_position", target=state.joint_positions["right"], embodiment="robodojo")})
result = step(hold)'''


def runtime_clarification(task_name: str):
    """Explain only actual interface differences from the verbatim source recipe."""
    name = task_name.removesuffix("_random")
    sections, adaptations = [], []
    if name == "imitate_sorting_sequence":
        adaptations.append("no_presampled_watch_frames")
        sections.append(
            "This ASPIRE run does NOT supply a pre-recorded demonstration or "
            "pre-sampled watch frames before the first model turn. The source "
            "recipe's statement that such frames are already in the conversation "
            "does not apply here. You must watch the demonstration as it unfolds: "
            "keep the policy arm still, advance native time with explicit hold "
            "actions, and inspect fresh public observations frequently enough "
            "to record each object placement in order. Base the remembered order "
            "only on frames you actually received; do not infer a completed "
            "sequence from a single initial image. No automatic 24-second wait "
            "or demonstration-frame injection occurs."
        )
    if name in WAIT_RECIPES:
        adaptations.append("aspire_joint_position_hold")
        sections.append(
            "The recipe's instruction to name a dimension at its current value "
            "describes the official harness's tool syntax, not this ASPIRE API. "
            "Here an explicit one-tick joint-position hold can be written as:\n\n"
            + HOLD_PROGRAM + "\n\n"
            "An accepted step consumes exactly one native control tick from "
            "the existing task budget. The omitted arm and gripper targets keep "
            "their previously commanded values. For consecutive hold steps, "
            "capture the hold target once and reuse it within the program; "
            "do not repeatedly reset the target to drifting measured joints. "
            "This preserves setpoints, not a guarantee that physics is motionless. "
            "Inspect the receipt and fresh observations, and end the program "
            "to return those observations to the model as needed. Calling "
            "get_observation alone does not advance simulation time."
        )
    if not sections:
        return "", []
    return ("\n\nASPIRE RUNTIME CLARIFICATION — this section takes precedence over "
            "source-recipe statements about supplied history or tool syntax:\n"
            + "\n\n".join(sections)), adaptations


@lru_cache(maxsize=1)
def recipe_provenance():
    provenance = json.loads((RECIPE_DIR / "PROVENANCE.json").read_text())
    if (provenance.get("commit") != UPSTREAM_COMMIT
            or provenance.get("recipe_count") != 42
            or len(provenance.get("files", {})) != 42):
        raise ValueError("official task recipe provenance differs from the pinned contract")
    return provenance


def task_recipe(task_name: str):
    """Use the official base recipe for random variants; never resolve a path."""
    if not isinstance(task_name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", task_name) is None:
        raise ValueError("invalid task name for public recipe lookup")
    filename = task_name.removesuffix("_random") + ".md"
    expected = recipe_provenance()["files"].get(filename)
    if expected is None:
        return None
    data = (RECIPE_DIR / filename).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("official task recipe bytes differ from the pinned contract")
    return filename, data.decode("utf-8"), expected


def task_instruction_with_contract(task_name: str, instruction: str):
    """Preserve native task language and append the official default public notes."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("RoboDojo did not supply authoritative task language")
    recipe = task_recipe(task_name)
    content = instruction
    # Match the official default _goal_content behavior, without optional
    # bimanual experiments, demonstration actions, ICL, or outcome tuning.
    if task_name == "press_by_number":
        content += (
            " After completing each red button, press the blue confirmation "
            "button immediately; do not finish both reds before pressing blue."
        )
    metadata = {"version": TASK_CONTRACT_VERSION,
                "upstream_repository": recipe_provenance()["repository"],
                "upstream_commit": UPSTREAM_COMMIT,
                "native_instruction": instruction,
                "recipe_name": None, "recipe_sha256": None,
                "runtime_adaptation_version": RUNTIME_ADAPTATION_VERSION,
                "runtime_adaptations": [], "rendered_recipe_sha256": None}
    if recipe is not None:
        filename, text, digest = recipe
        clarification, adaptations = runtime_clarification(task_name)
        rendered = text.rstrip() + clarification
        content += "\n\nTASK RECIPE:\n" + rendered
        metadata.update(recipe_name=filename, recipe_sha256=digest,
                        runtime_adaptations=adaptations,
                        rendered_recipe_sha256=hashlib.sha256(rendered.encode()).hexdigest())
    return content, metadata
