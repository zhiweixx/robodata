"""CPU regressions for native progress vs agent stdout; no GPU or model calls."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from robodata.execution.aspire_glue import build_api, execute_program
from robodata.execution.native_stdout import archive_native_stdout, route_native_progress
from robodata.environments.robodojo.robodojo_adapter import RoboDojoAdapter
from tests.runtime.fixtures import Env


class NoisyEnv(Env):
    def take_action(self, action):
        super().take_action(action)
        print(f"env0 step: \x1b[92m{self.take_action_cnt[0]} / {self.step_lim}\x1b[0m", end="\r")
        print("native warning: tracking error 0.02 rad")


HOLD = '''state = get_robot_state()
held = step(RobotAction({"right": ArmCommand(mode="joint_position", target=state.joint_positions["right"], embodiment="robodojo")}))
result = {"hold_ok": held.ok}
'''


class NativeStdoutTests(unittest.TestCase):
    def test_actual_executor_preserves_agent_print_and_native_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = NoisyEnv()
            program = root / "program.py"
            # Identical-looking agent text must survive; no global regex filter.
            program.write_text('print("env0 step: 7 / 8", end="\\r")\n' + HOLD +
                               'print("finished", result["hold_ok"])\n')
            with archive_native_stdout(root / "native.log"):
                result = execute_program(build_api(RoboDojoAdapter(env)), program)
            self.assertTrue(result.ok, result.error)
            self.assertEqual(result.stdout,
                "env0 step: 7 / 8\rnative warning: tracking error 0.02 rad\nfinished True\n")
            raw = (root / "native.log").read_bytes().decode()
            self.assertEqual(raw, "env0 step: \x1b[92m1 / 200\x1b[0m\r"
                                 "native warning: tracking error 0.02 rad\n")
            self.assertEqual(env.take_action_cnt, [1])
            self.assertEqual([c.tool for c in result.calls], ["get_robot_state", "step"])

    def test_chunked_progress_and_other_stdout_are_exactly_preserved_in_archive(self):
        raw = ("before\nenv0 step: \x1b[92m12 / 800\x1b[0m\r"
               "env1 step: 2 / 3\r\n"
               "diagnostic mentions env0 step: 9 / 10\r"
               "env0 step: unknown\n" + "x" * 10000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.log"
            public = io.StringIO()
            with redirect_stdout(public), archive_native_stdout(path), route_native_progress():
                for char in raw:
                    sys.stdout.write(char)
            self.assertEqual(path.read_bytes().decode(), raw)
            self.assertEqual(public.getvalue(), "before\n"
                "diagnostic mentions env0 step: 9 / 10\renv0 step: unknown\n" + "x" * 10000)

    def test_exception_keeps_diagnostics_and_raw_progress_and_restores_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.log"
            public = io.StringIO()
            with redirect_stdout(public):
                with self.assertRaisesRegex(RuntimeError, "native failed"):
                    with archive_native_stdout(path), route_native_progress():
                        print("env0 step: 1 / 10", end="\r")
                        print("controller diagnostic", end="")
                        raise RuntimeError("native failed")
                self.assertIs(sys.stdout, public)
                print("agent resumed")
            self.assertEqual(public.getvalue(), "controller diagnosticagent resumed\n")
            self.assertEqual(path.read_bytes(), b"env0 step: 1 / 10\rcontroller diagnostic")

    def test_without_archive_only_progress_is_routed_to_process_log(self):
        public, process = io.StringIO(), io.StringIO()
        with patch.object(sys, "__stdout__", process), redirect_stdout(public):
            with route_native_progress():
                print("env0 step: 1 / 3", end="\r")
                print("preserve this warning")
            print("agent")
        self.assertEqual(process.getvalue(), "env0 step: 1 / 3\r")
        self.assertEqual(public.getvalue(), "preserve this warning\nagent\n")

    def test_archive_cannot_overwrite_prior_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.log"
            path.write_text("previous run")
            with self.assertRaises(FileExistsError):
                with archive_native_stdout(path):
                    self.fail("existing archive must not be overwritten")
            self.assertEqual(path.read_text(), "previous run")


if __name__ == "__main__":
    unittest.main()
