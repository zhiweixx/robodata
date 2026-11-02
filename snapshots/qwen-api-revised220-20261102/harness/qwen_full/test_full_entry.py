"""Native lifecycle boundary tests, including partial-commit interruption."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch

import full_entry
from full_episode import FullEvaluationInterrupted


class NativeError(Exception):
    pass


class EntryTests(unittest.TestCase):
    def exercise(self, directory, fail_during_save=False, missing_video=False, hard_exit=False,
                 mode="evaluation", smoke_ok=True):
        root = Path(directory)
        native_output = root / "native"
        native_output.mkdir()
        env = NS(success_nums=0, fail_nums=0, unstable_nums=0, total_score=0,
                 eval_result={"details": {}, "score": 0}, eval_num=1,
                 task_name="general_pickup", eval_seed=0, run_id="run",
                 save_dir=str(native_output), env_seeds=[0])
        events = []
        def checkpoint():
            events.append(("checkpoint", env.success_nums, env.fail_nums, deepcopy_json(env.eval_result)))
        env.persist_resume_manifest = checkpoint
        env.reset = lambda **kwargs: None
        def run_eval():
            env.eval_one_episode()
            env.eval_result["details"][0] = {"layout_id": 0, "success": False, "score": 0}
            if fail_during_save:
                raise OSError("video storage interrupted")
            env.fail_nums += 1
            env.eval_result["eval_time"] = 1
            (native_output / "_result.json").write_text(json.dumps(env.eval_result))
            for camera in ("head", "left", "right"):
                if missing_video and camera == "right":
                    continue
                (native_output / f"episode_0000000_{camera}_fail.mp4").write_bytes(b"MP4-test")
            checkpoint()
        env.run_eval = run_eval
        native = ModuleType("src.eval_client.main")
        native.PhysXFatalError, native.PhysXBrokenError = NativeError, NativeError
        native.config_calls = 0
        def process_config(config, *, task_name):
            native.config_calls += 1
            return config, 1
        native.process_config = process_config
        def create(config, *args, **kwargs):
            env.eval_num = config.eval_cfg.eval_num
            return env
        native.create_eval_env = create
        def app_close():
            if hard_exit:
                # The callback sees on-disk evidence before terminating Python;
                # no full_entry finally or post-main code can execute after it.
                evidence = json.loads((root / "group_report.json").read_text())
                (root / "close_callback_evidence.json").write_text(json.dumps(evidence))
                os._exit(0)
        native.simulation_app = NS(close=app_close)
        def main():
            config = NS(camera=NS(annotator={}), eval_cfg=NS(eval_num=50))
            config, count = native.process_config(config, task_name="general_pickup")
            assert count == 1 and native.config_calls == 1
            made = native.create_eval_env(config, None)
            made.reset(seed=[0])
            try:
                made.run_eval()
            except Exception:
                events.append(("native_generic_skip",))
            native.simulation_app.close()
        native.main = main
        source, client_pkg, eval_module = ModuleType("src"), ModuleType("src.eval_client"), ModuleType("eval_env")
        source.eval_client = client_pkg
        client_pkg.main, client_pkg.eval_env = native, eval_module
        omega = ModuleType("omegaconf")
        def update(config, key, value, **kwargs):
            if key == "eval_cfg.eval_num":
                config.eval_cfg.eval_num = value
        omega.OmegaConf = NS(update=update)
        clutter = ModuleType("utils.cluttered_generator")
        clutter.UnStableError = NativeError
        torch = ModuleType("torch")
        torch.cuda = NS(set_device=lambda index: None)
        modules = {"src": source, "src.eval_client": client_pkg,
                   "src.eval_client.main": native, "src.eval_client.eval_env": eval_module,
                   "omegaconf": omega, "utils.cluttered_generator": clutter, "torch": torch}
        class Client:
            def __init__(self, path):
                pass
            def finish_episode(self, episode_id, reason):
                events.append(("finish", episode_id))
            def abandon_episode(self, episode_id, reason):
                events.append(("abandon", episode_id))
        def episode(env, client_factory):
            artifact = root / "episode"
            artifact.mkdir()
            return {"episode_id": "general_pickup_s0_l0", "layout_id": 0,
                    "artifact_dir": str(artifact), "termination_reason": "model_done", "ok": True}
        def smoke(env, output):
            report = {"ok": smoke_ok, "native_planning_ok": True, "model_requests": 0}
            (output / "smoke_report.json").write_text(json.dumps(report))
            return report
        variables = {"ASPIRE_OUTPUT_DIR": str(root), "ASPIRE_RELAY_DIR": str(root / "relay"),
                     "ASPIRE_SIM_GPU_UUID": "test", "ASPIRE_VULKAN_INDEX": "3",
                     "ASPIRE_FULL_MODE": mode, "EVAL_NUM": "1" if mode == "interface_smoke" else "native"}
        with patch.dict(sys.modules, modules), patch.dict(os.environ, variables, clear=True), \
             patch("full_gpu_guard.verify_simulator_gpu", return_value={"verified": True}), \
             patch("full_relay.FullRelayClient", Client), patch.object(full_entry, "run_full_episode", episode), \
             patch.object(full_entry, "_interface_smoke", smoke):
            if fail_during_save:
                with self.assertRaises(FullEvaluationInterrupted):
                    full_entry.main()
            elif missing_video or (mode == "interface_smoke" and not smoke_ok):
                with self.assertRaises(SystemExit):
                    full_entry.main()
            else:
                full_entry.main()
        return env, events, json.loads((root / "group_report.json").read_text())

    def test_relay_finish_only_after_native_score_is_durable(self):
        with tempfile.TemporaryDirectory() as directory:
            env, events, report = self.exercise(directory)
            self.assertTrue(report["complete"])
            self.assertEqual(report["scored"], 1)
            self.assertEqual(report["video_count"], 3)
            self.assertEqual(events[0][0], "checkpoint")
            self.assertEqual(events[1][0], "finish")

    def test_effective_target_matches_native_25_50_and_explicit_smoke_1(self):
        for native_count in (25, 50):
            for override in (None, "", "native", "NATIVE"):
                with self.subTest(native_count=native_count, override=override):
                    self.assertEqual(full_entry.effective_eval_count(
                        native_count, mode="evaluation", override=override), native_count)
            self.assertEqual(full_entry.effective_eval_count(
                native_count, mode="interface_smoke", override="1"), 1)
        self.assertEqual(full_entry.effective_eval_count(
            25, mode="interface_smoke", override="50"), 25)
        with self.assertRaises(ValueError):
            full_entry.effective_eval_count(50, mode="evaluation", override="1")

    def test_smoke_completion_requires_actual_successful_interface_report(self):
        for smoke_ok in (False, True):
            with self.subTest(smoke_ok=smoke_ok), tempfile.TemporaryDirectory() as directory:
                env, events, report = self.exercise(directory, mode="interface_smoke", smoke_ok=smoke_ok)
                self.assertEqual(report["complete"], smoke_ok)
                self.assertEqual(report["interface_smoke_ok"], smoke_ok)
                self.assertEqual(report["interface_smoke"]["ok"], smoke_ok)
                self.assertEqual(report["interface_smoke"]["model_requests"], 0)

    def test_video_write_failure_rolls_back_partial_details_and_never_skips(self):
        with tempfile.TemporaryDirectory() as directory:
            env, events, report = self.exercise(directory, fail_during_save=True)
            self.assertFalse(report["complete"])
            self.assertEqual(report["scored"], 0)
            self.assertEqual(env.eval_result["details"], {})
            self.assertNotIn("native_generic_skip", [event[0] for event in events])
            self.assertNotIn("finish", [event[0] for event in events])
            self.assertIn("abandon", [event[0] for event in events])
            self.assertTrue(all(event[3]["details"] == {} for event in events if event[0] == "checkpoint"))

    def test_incomplete_video_evidence_restores_scored_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            env, events, report = self.exercise(directory, missing_video=True)
            self.assertFalse(report["complete"])
            self.assertTrue(report["scores_complete"])
            self.assertEqual(report["scored"], 1)
            self.assertFalse(report["videos_validated"])
            self.assertEqual(events[-1][0], "checkpoint")
            self.assertEqual(events[-1][2], 1)
            self.assertEqual(len(events[-1][3]["details"]), 1)

    @unittest.skipUnless(hasattr(os, "fork"), "requires POSIX child process")
    def test_native_hard_exit_has_report_before_close_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            pid = os.fork()
            if pid == 0:
                try:
                    self.exercise(directory, hard_exit=True)
                except BaseException:
                    os._exit(1)
                os._exit(2)
            _, status = os.waitpid(pid, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
            root = Path(directory)
            report = json.loads((root / "group_report.json").read_text())
            evidence = json.loads((root / "close_callback_evidence.json").read_text())
            self.assertEqual(report, evidence)
            self.assertTrue(report["complete"])
            self.assertEqual(report["report_stage"], "before_simulation_app_close")
            progress = json.loads((root / "group_progress.json").read_text())
            self.assertTrue(progress["complete"])
            self.assertEqual(progress["report_stage"], "native_score_committed")


def deepcopy_json(value):
    return json.loads(json.dumps(value))


if __name__ == "__main__":
    unittest.main()
