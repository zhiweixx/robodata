"""CPU-only public prompt and proposal contract regression tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from model_driver import OUTPUT_FORMAT, SYSTEM_PREFIX, parse_program_proposal
import task_prompt_contract as contract


class TaskPromptContractTests(unittest.TestCase):
    def test_all_42_recipe_bytes_and_license_are_pinned(self):
        provenance = contract.recipe_provenance()
        self.assertEqual(len(provenance["files"]), 42)
        self.assertEqual({p.name for p in contract.RECIPE_DIR.glob("*.md")}, set(provenance["files"]))
        for name, digest in provenance["files"].items():
            with self.subTest(recipe=name):
                self.assertEqual(hashlib.sha256((contract.RECIPE_DIR / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(hashlib.sha256((contract.RECIPE_DIR / "LICENSE").read_bytes()).hexdigest(), provenance["license_sha256"])

    def test_random_variant_reuses_exact_base_recipe(self):
        self.assertEqual(contract.task_recipe("stack_bowls"), contract.task_recipe("stack_bowls_random"))
        self.assertEqual(contract.task_recipe("play_Xylophone")[0], "play_Xylophone.md")

    def test_all_54_native_configurations_have_the_42_public_recipes(self):
        manifest = json.loads(Path(__file__).with_name("full_manifest.json").read_text())
        names = {group["task"] for group in manifest["groups"]}
        self.assertEqual(len(names), 54)
        recipes = set()
        for name in sorted(names):
            with self.subTest(task=name):
                recipe = contract.task_recipe(name)
                self.assertIsNotNone(recipe)
                recipes.add(recipe[0])
                _, metadata = contract.task_instruction_with_contract(name, "Native task instruction")
                self.assertEqual(metadata["recipe_name"], recipe[0])
        self.assertEqual(len(recipes), 42)

    def test_imitation_supplied_history_assumption_is_explicitly_overridden(self):
        prompt, metadata = contract.task_instruction_with_contract("imitate_sorting_sequence", "Watch then copy.")
        original = contract.task_recipe("imitate_sorting_sequence")[1].rstrip()
        self.assertIn(original, prompt)
        marker = "ASPIRE RUNTIME CLARIFICATION"
        self.assertGreater(prompt.index(marker), prompt.index(original))
        clarification = prompt.split(marker, 1)[1]
        self.assertIn("does NOT supply", clarification)
        self.assertIn("does not apply here", clarification)
        self.assertIn("only on frames you actually received", clarification)
        self.assertIn("No automatic 24-second wait", clarification)
        self.assertEqual(metadata["runtime_adaptations"], ["no_presampled_watch_frames", "aspire_joint_position_hold"])
        rendered = prompt.split("TASK RECIPE:\n", 1)[1]
        self.assertEqual(metadata["rendered_recipe_sha256"], hashlib.sha256(rendered.encode()).hexdigest())
        self.assertEqual(metadata["runtime_adaptation_version"], contract.RUNTIME_ADAPTATION_VERSION)

    def test_only_recipes_with_official_noop_syntax_get_hold_clarification(self):
        for task in ("imitate_sorting_sequence", "make_kong", "play_tic_tac_toe"):
            with self.subTest(task=task):
                prompt, metadata = contract.task_instruction_with_contract(task, "Native task instruction")
                self.assertIn(contract.HOLD_PROGRAM, prompt)
                self.assertIn("exactly one native control tick", prompt)
                self.assertIn("get_observation alone does not advance", prompt)
                self.assertIn("aspire_joint_position_hold", metadata["runtime_adaptations"])
        prompt, metadata = contract.task_instruction_with_contract("general_pickup", "Native task instruction")
        self.assertNotIn("ASPIRE RUNTIME CLARIFICATION", prompt)
        self.assertEqual(metadata["runtime_adaptations"], [])

    def test_documented_hold_runs_through_aspire_executor_for_one_native_tick(self):
        import numpy as np
        from aspire_glue import build_api, execute_program
        from robodojo_adapter import RoboDojoAdapter
        from verify_adapter_offline import Env
        env = Env()
        env.robot_manager.robot_key[0].data.joint_pos[0] = .2
        env.robot_manager.robot_key[1].data.joint_pos[0] = -.3
        adapter = RoboDojoAdapter(env)
        api = build_api(adapter)
        with tempfile.TemporaryDirectory() as directory:
            program = Path(directory) / "hold.py"
            program.write_text(contract.HOLD_PROGRAM)
            outcome = execute_program(api, program)
        self.assertTrue(outcome.ok, str(outcome.error))
        self.assertTrue(outcome.result.ok)
        self.assertEqual(env.take_action_cnt, [1])
        self.assertEqual(env.step_lim, 200)
        np.testing.assert_allclose(env.last_action["left_arm_joint_state"], .2)
        np.testing.assert_allclose(env.last_action["right_arm_joint_state"], -.3)
        self.assertEqual(env.physical_gripper, {"left": 1., "right": 1.})

    def test_recipe_lookup_rejects_path_and_alias_injection(self):
        for name in ("../align_blocks", "/align_blocks", "align_blocks.md", "a\\b", "", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                contract.task_recipe(name)
        self.assertIsNone(contract.task_recipe("unknown_task"))

    def test_native_instruction_is_preserved_and_public_recipe_appended(self):
        raw = "Pick up the mint green scissors by 10 cm."
        prompt, metadata = contract.task_instruction_with_contract("general_pickup", raw)
        self.assertTrue(prompt.startswith(raw + "\n\nTASK RECIPE:\n"))
        self.assertEqual(metadata["native_instruction"], raw)
        self.assertEqual(metadata["version"], contract.TASK_CONTRACT_VERSION)
        self.assertNotIn("bimanual press-lift", prompt)
        self.assertEqual(prompt.split("TASK RECIPE:\n", 1)[1], contract.task_recipe("general_pickup")[1].rstrip())

    def test_press_confirmation_matches_official_default(self):
        prompt, _ = contract.task_instruction_with_contract("press_by_number", "Press the requested number.")
        self.assertIn("press the blue confirmation button immediately", prompt)
        self.assertIn("do not finish both reds before pressing blue", prompt)

    def test_recipe_tampering_fails_closed(self):
        original = Path.read_bytes
        def altered(path):
            data = original(path)
            return data + b"changed" if path.name == "align_blocks.md" else data
        with patch.object(Path, "read_bytes", altered), self.assertRaisesRegex(ValueError, "bytes differ"):
            contract.task_recipe("align_blocks")

    def test_prompt_explains_native_completion_and_both_arms_home(self):
        self.assertIn("BOTH arms", SYSTEM_PREFIX)
        self.assertIn("0.15 m and 20 degrees", SYSTEM_PREFIX)
        self.assertIn("no model-declared completion action", SYSTEM_PREFIX)
        self.assertNotIn("done=true", SYSTEM_PREFIX)

    def test_schema_removes_done_and_requires_failure_only_terminal(self):
        schema = OUTPUT_FORMAT["format"]["schema"]
        self.assertEqual(set(schema["properties"]), {"code", "summary", "give_up"})
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(parse_program_proposal('{"code":"result=1","summary":"observe","give_up":false}')["code"], "result=1")
        self.assertTrue(parse_program_proposal('{"code":"","summary":"target is unreachable","give_up":true}')["give_up"])

    def test_legacy_completion_and_invalid_terminal_contract_rejected(self):
        for proposal in (
            {"code": "", "summary": "complete", "done": True},
            {"code": "result=1", "summary": "continue", "done": False},
            {"code": "result=1", "summary": "unable", "give_up": True},
            {"code": "", "summary": " ", "give_up": True},
            {"code": "", "summary": "complete", "give_up": False},
            {"code": "result=1", "summary": "continue", "give_up": 0},
        ):
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                parse_program_proposal(json.dumps(proposal))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_program_proposal('{"code":"result=1","summary":"a","give_up":false,"give_up":true}')


if __name__ == "__main__":
    unittest.main()
