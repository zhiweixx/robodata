"""Offline checks for GPU isolation, credential stripping, and hard cleanup."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from unittest import mock

from qwen_server_suite import (
    allocated_devices, resolve_gpu_uuids, sanitized_environment, server_command,
    stop_process_group, cpu_partition, with_cpu_partition,
    restore_simulator_proxies, simulator_proxy_names,
)


class SupervisorChecks(unittest.TestCase):
    def test_cpu_partition_default_keeps_launch_unchanged(self):
        with mock.patch("qwen_server_suite.os.sched_getaffinity", create=True) as affinity:
            partition = cpu_partition(False)
        affinity.assert_not_called()
        self.assertFalse(partition["enabled"])
        for role in ("server", "smoke", "suite"):
            self.assertEqual(with_cpu_partition(["/python", "script.py"], partition, role),
                             ["/python", "script.py"])

    def test_cpu_partition_uses_disjoint_allocated_cpus_and_reports_commands(self):
        # Noncontiguous, unsorted allocation: never assume the CPUs are 0..15.
        cpus = list(range(32, 66, 2))
        with mock.patch("qwen_server_suite.os.sched_getaffinity", return_value=set(cpus), create=True) as affinity, \
                mock.patch("qwen_server_suite.shutil.which", return_value="/usr/bin/taskset"):
            partition = cpu_partition(True)
        affinity.assert_called_once_with(0)
        self.assertEqual(partition["allocated_cpus"], cpus)
        self.assertEqual(partition["server_cpus"], cpus[:8])
        self.assertEqual(partition["suite_cpus"], cpus[8:])
        self.assertEqual(partition["smoke_cpus"], partition["suite_cpus"])
        self.assertFalse(set(partition["server_cpus"]) & set(partition["suite_cpus"]))
        self.assertEqual(set(partition["server_cpus"]) | set(partition["suite_cpus"]), set(cpus))
        for role in ("server", "smoke", "suite"):
            command = with_cpu_partition(["/python", role + ".py"], partition, role)
            self.assertEqual(command[:2], ["/usr/bin/taskset", "--cpu-list"])
            self.assertEqual([int(cpu) for cpu in command[2].split(",")], partition[role + "_cpus"])
            self.assertEqual(command[3:], ["/python", role + ".py"])
        self.assertEqual(json.loads(json.dumps(partition)), partition)

    def test_cpu_partition_rejects_insufficient_allocation_or_missing_taskset(self):
        with mock.patch("qwen_server_suite.os.sched_getaffinity", return_value=set(range(15)), create=True):
            with self.assertRaisesRegex(RuntimeError, "at least 16"):
                cpu_partition(True)
        with mock.patch("qwen_server_suite.os.sched_getaffinity", return_value=set(range(16)), create=True), \
                mock.patch("qwen_server_suite.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "taskset executable"):
                cpu_partition(True)

    def test_reject_ambiguous_allocation(self):
        for value in (None, "", "0,1,2", "0,1,2,2", "-1,0,1,2", "0, 1,2,3"):
            with self.assertRaises(RuntimeError):
                allocated_devices(value)

    def test_uuid_mapping_is_not_assumed_from_ordinal(self):
        table = "index, uuid, name, driver_version, memory.total\n" + "\n".join(
            f"{i}, GPU-test-{i}, NVIDIA L40S, 595.71.05, 46068 MiB" for i in range(4))
        rows = resolve_gpu_uuids(table, ["2", "GPU-test-0", "1", "3"])
        self.assertEqual([row["uuid"] for row in rows],
                         ["GPU-test-2", "GPU-test-0", "GPU-test-1", "GPU-test-3"])
        with self.assertRaises(RuntimeError):
            resolve_gpu_uuids(table, ["0", "GPU-test-0", "1", "2"])

    def test_offline_environment_drops_forwarded_credentials_and_proxies(self):
        names = ["HF_TOKEN", "OPENAI_API_KEY", "HUGGING_FACE_HUB_TOKEN", "https_proxy",
                 "SINGULARITYENV_OPENAI_API_KEY", "APPTAINERENV_HF_TOKEN", "AWS_ACCESS_TOKEN"]
        source = {name: "must-not-survive" for name in names}
        source.update(CUDA_VISIBLE_DEVICES="0,1,2,3", SLURM_JOB_ID="123")
        env = sanitized_environment(source, Path("/runtime"))
        self.assertTrue(all(name not in env for name in names))
        self.assertEqual(env["HF_TOKEN_PATH"], "/dev/null")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0,1,2,3")

    def test_simulator_proxy_restore_is_allowlisted_and_does_not_mutate_server_env(self):
        proxy_names = [prefix + name for prefix in ("", "SINGULARITYENV_", "APPTAINERENV_")
                       for base in ("HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "ALL_PROXY", "NO_PROXY")
                       for name in (base, base.lower())]
        source = {name: "private-proxy-value-" + name for name in proxy_names}
        source.update(OPENAI_API_KEY="secret", APPTAINERENV_HF_TOKEN="secret",
                      SINGULARITYENV_OPENAI_API_KEY="secret", HTTP_PROXY_API_KEY="secret",
                      UNRELATED="not-a-proxy", CUDA_VISIBLE_DEVICES="0,1,2,3")
        source_before = dict(source)
        server_env = sanitized_environment(source, Path("/runtime"))
        server_before = dict(server_env)
        suite_env = restore_simulator_proxies(server_env, source)
        self.assertEqual(source, source_before)
        self.assertEqual(server_env, server_before)
        self.assertEqual(simulator_proxy_names(source), sorted(proxy_names))
        for name in proxy_names:
            self.assertEqual(suite_env[name], source[name])
            if name not in ("NO_PROXY", "no_proxy"):
                self.assertNotIn(name, server_env)
        self.assertEqual(server_env["NO_PROXY"], "127.0.0.1,localhost,::1")
        for name in ("OPENAI_API_KEY", "APPTAINERENV_HF_TOKEN",
                     "SINGULARITYENV_OPENAI_API_KEY", "HTTP_PROXY_API_KEY"):
            self.assertNotIn(name, suite_env)
        self.assertEqual(suite_env["CUDA_VISIBLE_DEVICES"], "0,1,2,3")
        audit = {"restored_variable_names": simulator_proxy_names(source), "values_recorded": False}
        self.assertNotIn("private-proxy-value", json.dumps(audit))
        self.assertEqual(restore_simulator_proxies(server_env, {}), server_env)

    def test_server_budget_preserves_image_history(self):
        args = server_command(Path("/python"), Path("/model"), 34001)
        self.assertEqual(args[args.index("--tensor-parallel-size") + 1], "2")
        self.assertGreaterEqual(int(args[args.index("--max-model-len") + 1]), 128000 + 8192)
        options = json.loads(args[args.index("--limit-mm-per-prompt") + 1])
        self.assertEqual(options["image"], {"count": 24, "width": 640, "height": 480})
        self.assertNotIn("--mm-processor-kwargs", args)

    def test_cleanup_escalates_for_a_server_ignoring_term(self):
        process = subprocess.Popen(
            [sys.executable, "-u", "-c",
             "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
             "print('ready',flush=True); time.sleep(30)"],
            stdout=subprocess.PIPE, text=True, start_new_session=True)
        try:
            self.assertEqual(process.stdout.readline().strip(), "ready")
            result = stop_process_group(process, grace=.2)
            self.assertTrue(result["term_sent"])
            self.assertTrue(result["kill_sent"])
            self.assertEqual(result["returncode"], -signal.SIGKILL)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()


if __name__ == "__main__":
    unittest.main()
