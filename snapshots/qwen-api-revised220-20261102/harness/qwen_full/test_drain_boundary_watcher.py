import json
import os
from pathlib import Path
import signal
import socket
import tempfile
import unittest
from unittest.mock import patch

import drain_boundary_watcher as drain


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class Tests(unittest.TestCase):
    def proc(self, root, pid, *, ppid=1, pgid=None, uid=1000, argv=None, start=123):
        path = root / str(pid)
        path.mkdir(parents=True, exist_ok=True)
        fields = ["S", str(ppid), str(pgid or pid)] + ["0"]*16 + [str(start)]
        (path / "stat").write_text(f"{pid} (a process name) " + " ".join(fields))
        (path / "status").write_text(f"Uid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
        (path / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv or ["python"]))

    def fixture(self, root):
        procroot = root / "proc"
        self.proc(procroot, 60001, argv=["python", "full_supervisor.py", "--campaign", "campaign"])
        worker = {"name": "worker0", "task": "general_pickup", "eval_seed": 0,
                  "simulator_pid": 60002, "controller_pid": 60003, "relay_dir": "/tmp/relay0"}
        self.proc(procroot, 60002, ppid=60001,
                  argv=["starter-suid"])
        self.proc(procroot, 60005, ppid=60002, pgid=60002,
                  argv=["bash", "python.sh", "full_entry.py", "--task_name", "general_pickup", "--seed", "0"])
        self.proc(procroot, 60004, ppid=60005, pgid=60002,
                  argv=["python", "full_entry.py", "--task_name", "general_pickup", "--seed", "0"])
        self.proc(procroot, 60003, ppid=60001,
                  argv=["python", "full_model_driver.py", "--relay-dir", "/tmp/relay0"])
        config = {"campaign": "campaign", "supervisor_pid": 60001}
        return drain.ProcInspector(procroot, uid=1000), config, worker

    def test_exact_uid_parent_role_and_start_time_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proc, config, worker = self.fixture(root)
            first = drain.validate_identity(proc, config, worker)
            self.assertEqual(first["simulator"]["pgid"], 60002)
            self.proc(proc.root, 60002, ppid=60001, start=999,
                      argv=["bash", "full_inside.sh", "general_pickup"])
            with self.assertRaisesRegex(RuntimeError, "reused"):
                drain.validate_identity(proc, config, worker, first)
            self.proc(proc.root, 60002, ppid=60001, uid=0,
                      argv=["bash", "full_inside.sh", "general_pickup"])
            with self.assertRaisesRegex(RuntimeError, "UID"):
                drain.validate_identity(proc, config, worker)

    def test_missing_commit_or_video_never_authorizes_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group, native = root / "group", root / "native/run/_result.json"
            worker = {"task": "general_pickup", "eval_seed": 0, "group_dir": str(group)}
            initial = {"scored": 1, "layout_ids": [0]}
            progress = {"task": "general_pickup", "eval_seed": 0, "scored": 2,
                        "native_result_path": str(native), "native_run_id": "run",
                        "details": {"0": {"layout_id": 0, "success": False},
                                    "1": {"layout_id": 1, "success": False}}}
            save(group / "group_progress.json", progress)
            save(native, {"eval_time": 2})
            checkpoint = native.parent.parent / "_resume_run.json"
            save(checkpoint, {"success_nums": 0, "fail_nums": 2, "completed_layout_ids": [0, 1]})
            episode = group / "episodes/general_pickup_s0_l1/attempt_001/episode_report.json"
            save(episode.parent.parent / "latest_attempt.json", {"report": str(episode)})
            save(episode, {"episode_id": "general_pickup_s0_l1", "native_score_committed": False})
            self.assertIsNone(drain.committed_boundary({"simroot": str(root)}, worker, initial))
            save(episode, {"episode_id": "general_pickup_s0_l1", "native_score_committed": True})
            self.assertIsNone(drain.committed_boundary({"simroot": str(root)}, worker, initial))
            for camera in ("head", "left_wrist", "right_wrist"):
                (native.parent / f"episode_0000001_cam_{camera}_fail.mp4").write_bytes(b"video")
            boundary = drain.committed_boundary({"simroot": str(root)}, worker, initial)
            self.assertEqual(boundary["scored"], 2)
            self.assertEqual(boundary["new_committed"][0]["layout_id"], 1)

    def test_default_dry_run_sends_no_signals(self):
        config = {"host": socket.gethostname(), "supervisor_pid": 101,
                  "workers": [{"name": "a", "simulator_pid": 102, "controller_pid": 103},
                              {"name": "b", "simulator_pid": 104, "controller_pid": 105}]}
        sent = []
        with patch.object(drain, "validate_identity", return_value={}), \
             patch.object(drain, "snapshot", return_value={"scored": 1, "layout_ids": [0]}), \
             patch.object(drain, "work_evidence", return_value={}):
            result = drain.run(config, execute=False, output=None, killpg=lambda *args: sent.append(args))
        self.assertEqual(result["mode"], "dry_run")
        self.assertEqual(sent, [])

    def test_execute_freezes_simulator_then_controller_and_never_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"host": socket.gethostname(), "supervisor_pid": 101,
                      "workers": [{"name": "a", "simulator_pid": 102, "controller_pid": 103},
                                  {"name": "b", "simulator_pid": 104, "controller_pid": 105}]}
            sent = []
            class FakeProc:
                def members(self, pgid):
                    return [{"pid": pgid, "state": "T"}]
            def identity(proc, config, worker, baseline=None):
                return {"simulator": {"pgid": worker["simulator_pid"]},
                        "controller": {"pgid": worker["controller_pid"]}}
            with patch.object(drain, "validate_identity", side_effect=identity), \
                 patch.object(drain, "snapshot", return_value={"scored": 1, "layout_ids": [0]}), \
                 patch.object(drain, "work_evidence", return_value={}), \
                 patch.object(drain, "committed_boundary", return_value={"scored": 2}), \
                 patch.object(drain.time, "sleep"):
                result = drain.run(config, execute=True, output=Path(directory) / "drain.json",
                                   proc=FakeProc(), killpg=lambda *args: sent.append(args))
            self.assertTrue(result["complete"])
            self.assertEqual(sent, [(102, signal.SIGSTOP), (103, signal.SIGSTOP),
                                    (104, signal.SIGSTOP), (105, signal.SIGSTOP)])


if __name__ == "__main__":
    unittest.main()
