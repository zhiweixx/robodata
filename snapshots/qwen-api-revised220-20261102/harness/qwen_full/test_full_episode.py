"""CPU integration tests: real ASPIRE executor, fake public robot and local model."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from full_adapter import FullRoboDojoAdapter, task_records
from full_episode import run_full_episode, public_feedback
from full_model_driver import ModelEpisodeFailure, InferenceInfrastructureError
from verify_adapter_offline import Env
from test_native_stdout import NoisyEnv

ROOT = Path(__file__).resolve().parents[2] / "audit/RoboDojo"
HOLD = '''state = get_robot_state()
held = step(RobotAction({"right": ArmCommand(mode="joint_position", target=state.joint_positions["right"], embodiment="robodojo")}))
result = {"hold_ok": held.ok}
'''


class Client:
    def __init__(self, turns=10, failure=None):
        self.turns, self.failure, self.calls, self.abandoned = turns, failure, [], []
    def next_program(self, **request):
        self.calls.append(request)
        if self.failure:
            raise self.failure
        index = len(self.calls)
        return NS(code="" if index > self.turns else HOLD,
                  give_up=index > self.turns, summary="hold" if index <= self.turns else "cannot complete task",
                  iteration=index,
                  usage={}, budget={})
    def abandon_episode(self, episode_id, reason):
        self.abandoned.append((episode_id, reason))


class FullEpisodeTests(unittest.TestCase):
    def run_case(self, client, env=None, root=None):
        env = env or Env()
        env.eval_seed = 2
        env.env_seeds = [67]
        env.save_dir = str(root / "native")
        variables = {"ASPIRE_OUTPUT_DIR": str(root), "ASPIRE_RELAY_DIR": str(root / "relay"),
                     "ROBODOJO_ROOT": str(ROOT), "ASPIRE_MAX_ITERATIONS": "8"}
        with patch.dict(os.environ, variables, clear=True):
            result = run_full_episode(env, client_factory=lambda path: client)
        return result, env

    def test_more_than_eight_programs_and_public_only_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = Client(turns=10)
            report, env = self.run_case(client, root=root)
            self.assertEqual(len(client.calls), 11)
            self.assertEqual(env.take_action_cnt, [10])
            self.assertEqual(report["episode_id"], "general_pickup_s2_l67")
            self.assertEqual(report["total_programs_executed"], 10)
            self.assertEqual(report["termination_reason"], "model_give_up")
            self.assertFalse(report["official"]["official_success"])
            self.assertTrue(env.end_flag[0])
            self.assertEqual(client.abandoned, [])
            self.assertEqual(report["model_failure"]["code"], "model_give_up")
            self.assertEqual(report["proposal_schema"], "aspire_program_give_up_v2")
            self.assertEqual(report["task_prompt_contract"]["version"], "robodojo_public_task_contract_v2")
            for request in client.calls:
                public = json.dumps(request)
                self.assertNotIn('"official"', public)
                self.assertNotIn('"reward"', public)
                self.assertEqual(len(request["public_observation"]["images"]), 3)
                self.assertNotIn("300s Python watchdog", request["api_documentation"])
            self.assertEqual(len(list(root.rglob("program_*.py"))), 10)
            self.assertFalse(list(root.rglob("*.npz")))

    def test_apparent_completion_does_not_replace_native_action_budget(self):
        class CompleteSummaryClient(Client):
            def next_program(self, **request):
                proposal = super().next_program(**request)
                proposal.summary = "The task looks complete"
                return proposal
        with tempfile.TemporaryDirectory() as directory:
            env = Env()
            env.step_lim = 4
            client = CompleteSummaryClient(turns=100)
            report, env = self.run_case(client, env=env, root=Path(directory))
            self.assertEqual(env.take_action_cnt, [4])
            self.assertEqual(len(client.calls), 4)
            self.assertEqual(report["termination_reason"], "environment_ended")
            self.assertEqual(report["total_programs_executed"], 4)
            self.assertFalse(report["official"]["official_success"])

    def test_native_success_ends_without_extra_homing_or_model_call(self):
        class SuccessfulEnv(Env):
            def __init__(self):
                super().__init__()
                self.reward_manager = NS(get_reward=lambda: [1. if self.take_action_cnt[0] >= 3 else 0.])
            def is_episode_end(self):
                if self.take_action_cnt[0] >= 3:
                    self.end_flag[0] = True
                return self.end_flag[0]
        with tempfile.TemporaryDirectory() as directory:
            client = Client(turns=100)
            report, env = self.run_case(client, env=SuccessfulEnv(), root=Path(directory))
            self.assertEqual(env.take_action_cnt, [3])
            self.assertEqual(len(client.calls), 3)
            self.assertTrue(report["official"]["official_success"])
            self.assertEqual(report["termination_reason"], "environment_ended")

    def test_give_up_concedes_without_executing_a_program(self):
        with tempfile.TemporaryDirectory() as directory:
            report, env = self.run_case(Client(turns=0), root=Path(directory))
            self.assertEqual(env.take_action_cnt, [0])
            self.assertEqual(report["total_programs_executed"], 0)
            self.assertEqual(report["termination_reason"], "model_give_up")
            self.assertFalse(report["official"]["official_success"])

    def test_legacy_done_relay_cannot_declare_completion(self):
        class LegacyClient(Client):
            def next_program(self, **request):
                return NS(code="", done=True, summary="complete", iteration=1, usage={}, budget={})
        with tempfile.TemporaryDirectory() as directory:
            env = Env()
            client = LegacyClient()
            with self.assertRaisesRegex(ValueError, "legacy done"):
                self.run_case(client, env=env, root=Path(directory))
            self.assertEqual(env.take_action_cnt, [0])
            self.assertFalse(env.end_flag[0])
            self.assertEqual(client.abandoned, [("general_pickup_s2_l67", "infrastructure_interrupted")])

    def test_known_context_failure_finalizes_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            client = Client(failure=ModelEpisodeFailure("native_context_exhausted", "no remaining context"))
            report, env = self.run_case(client, root=Path(directory))
            self.assertTrue(report["ok"])
            self.assertEqual(report["termination_reason"], "native_context_exhausted")
            self.assertTrue(env.end_flag[0])
            self.assertFalse(env.success[0])
            self.assertEqual(len(client.calls), 1)
            self.assertEqual(client.abandoned, [])

    def test_infrastructure_error_does_not_mark_native_episode_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = Env()
            client = Client(failure=InferenceInfrastructureError("lost_server", "server stopped"))
            with self.assertRaises(InferenceInfrastructureError):
                self.run_case(client, env=env, root=root)
            self.assertFalse(env.end_flag[0])
            self.assertTrue(env.success[0])  # still native layout-valid, not a policy score
            self.assertEqual(client.abandoned, [("general_pickup_s2_l67", "infrastructure_interrupted")])
            report = json.loads(next(root.rglob("episode_report.json")).read_text())
            self.assertEqual(report["status"], "interrupted")
            self.assertNotIn("official", report)

    def test_full_feedback_is_not_pilot_truncated(self):
        calls = [NS(tool="read", ok=True, duration_s=0, failure_code=None,
                    error_message=None, inputs_summary={}, outputs_summary={},
                    frame=None, provider=None, result_type=None) for _ in range(65)]
        feedback = public_feedback(NS(ok=True, result={"text": "x"*13000},
                                      stdout="y"*5000, error=None, calls=calls))
        self.assertEqual(len(feedback["tool_calls"]), 65)
        self.assertEqual(len(feedback["stdout"]), 5000)
        self.assertEqual(len(feedback["result"]["text"]), 13000)

    def test_native_progress_is_archived_but_not_replayed_to_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = Client(turns=2)
            report, env = self.run_case(client, env=NoisyEnv(), root=root)
            self.assertEqual(env.take_action_cnt, [2])
            for request in client.calls[1:]:
                feedback = request["previous_result"]
                self.assertEqual(feedback["stdout"], "native warning: tracking error 0.02 rad\n")
                self.assertNotIn("env0 step:", json.dumps(feedback))
                self.assertTrue(feedback["result"]["hold_ok"])
            for index, entry in enumerate(report["iterations"][:2]):
                path = Path(report["artifact_dir"]) / entry["native_stdout_log"]
                self.assertEqual(path.read_bytes().decode(),
                    f"env0 step: \x1b[92m{index + 1} / 200\x1b[0m\r"
                    "native warning: tracking error 0.02 rad\n")
            # Budget metadata remains available without a progress-bar transcript.
            self.assertEqual(client.calls[-1]["public_observation"]["remaining_actions"], 198)

    def test_all_native_tasks_have_correct_public_family(self):
        with patch.dict(os.environ, {"ROBODOJO_ROOT": str(ROOT)}):
            records = task_records()
            self.assertEqual(len(records), 54)
            self.assertEqual(records["match_and_pick_from_conveyor"]["dimension"], "memory")
            self.assertEqual(records["fold_clothes_random"]["dimension"], "generalization")
            self.assertEqual(records["fasten_screws"]["dimension"], "precision")
            env = Env()
            env.task_name = "make_kong"
            adapter = FullRoboDojoAdapter(env)
            self.assertEqual(adapter.get_task_context().family, "long-horizon")
            self.assertEqual(adapter.get_task_context().metadata["episode_steps_remaining"], 200)


if __name__ == "__main__":
    unittest.main()
