"""External Linux watcher: freeze existing workers only after durable score commit.

Default is read-only inspection. --execute sends SIGSTOP to the two explicitly
authorized child process groups, simulator first and controller second. It never
changes model/simulator code, requests, observations, actions or checkpoints.
An episode reset can race a poll; any resulting unscored work is recorded. The
operator owns subsequent supervisor shutdown and SIGCONT of frozen groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class ProcInspector:
    def __init__(self, proc_root=Path("/proc"), uid=None):
        self.root = Path(proc_root)
        self.uid = os.getuid() if uid is None else uid

    def get(self, pid):
        pid = int(pid)
        folder = self.root / str(pid)
        raw = (folder / "stat").read_text()
        fields = raw[raw.rfind(")")+2:].split()
        status = (folder / "status").read_text().splitlines()
        uids = next(line.split()[1:] for line in status if line.startswith("Uid:"))
        if any(int(uid) != self.uid for uid in uids):
            raise RuntimeError(f"PID {pid} does not have the current user's real/effective/saved/fs UID")
        if fields[0] in ("Z", "X"):
            raise ProcessLookupError(f"PID {pid} is no longer a live process")
        argv = [part.decode() for part in (folder / "cmdline").read_bytes().split(b"\0") if part]
        return {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]),
                "state": fields[0], "start_ticks": int(fields[19]), "argv": argv}

    def descendant(self, pid, ancestor):
        seen = set()
        while pid > 1 and pid not in seen:
            seen.add(pid)
            parent = self.get(pid)["ppid"]
            if parent == ancestor:
                return True
            pid = parent
        return False

    def members(self, pgid):
        result = []
        for path in self.root.iterdir():
            if not path.name.isdigit():
                continue
            try:
                raw = (path / "stat").read_text()
                fields = raw[raw.rfind(")")+2:].split()
                if int(fields[2]) == pgid:
                    result.append(self.get(int(path.name)))
            except (FileNotFoundError, ProcessLookupError):
                continue
        if not result:
            raise RuntimeError(f"empty process group {pgid}")
        return result


def option(argv, name):
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            return argv[index+1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def validate_identity(proc, config, worker, baseline=None):
    supervisor = proc.get(config["supervisor_pid"])
    if not any(Path(a).name == "full_supervisor.py" for a in supervisor["argv"]):
        raise RuntimeError("specified supervisor PID is not full_supervisor.py")
    if option(supervisor["argv"], "--campaign") != config["campaign"]:
        raise RuntimeError("supervisor campaign differs from drain configuration")
    result = {"supervisor": supervisor}
    for role, key in (("simulator", "simulator_pid"), ("controller", "controller_pid")):
        record = proc.get(worker[key])
        if record["pgid"] != record["pid"] or record["pgid"] == os.getpgrp():
            raise RuntimeError(f"{role} must lead its own separate process group")
        if not proc.descendant(record["pid"], supervisor["pid"]):
            raise RuntimeError(f"{role} is not a child of the specified supervisor")
        argv = record["argv"]
        members = proc.members(record["pgid"])
        if role == "controller":
            if not any(Path(a).name == "full_model_driver.py" for a in argv):
                raise RuntimeError("controller PID does not run full_model_driver.py")
            if option(argv, "--relay-dir") != worker["relay_dir"]:
                raise RuntimeError("controller relay directory mismatch")
        else:
            # Apptainer can replace the session leader with starter-suid. The
            # actual Isaac Python must still belong to this exact owned PGID,
            # descend from its leader and run the authorized task + eval seed.
            matched = []
            for member in members:
                arguments = member["argv"]
                if (arguments and Path(arguments[0]).name.startswith("python")
                        and any(Path(a).name == "full_entry.py" for a in arguments)
                        and option(arguments, "--task_name") == worker["task"]
                        and option(arguments, "--seed") == str(worker["eval_seed"])
                        and (member["pid"] == record["pid"] or proc.descendant(member["pid"], record["pid"]))):
                    matched.append(member)
            if len(matched) != 1:
                raise RuntimeError("simulator PGID must contain exactly one authorized task/seed full_entry.py")
            expected_python = worker.get("simulator_python_pid")
            if expected_python is not None and matched[0]["pid"] != expected_python:
                raise RuntimeError("simulator Python PID differs from explicit authorization")
            record["simulator_python_pid"] = matched[0]["pid"]
            record["simulator_python_start_ticks"] = matched[0]["start_ticks"]
        record["members"] = [{k: m[k] for k in ("pid", "ppid", "pgid", "state", "start_ticks")}
                             for m in members]
        result[role] = record
    if baseline:
        for role in result:
            if result[role]["start_ticks"] != baseline[role]["start_ticks"]:
                raise RuntimeError(f"{role} PID was reused; refusing signals")
        if result["simulator"]["simulator_python_start_ticks"] != baseline["simulator"]["simulator_python_start_ticks"]:
            raise RuntimeError("simulator Python PID was reused; refusing signals")
    return result


def host_path(path, simroot):
    path = Path(path)
    if path.is_relative_to("/work"):
        return Path(simroot) / path.relative_to("/work")
    if not path.is_absolute():
        return Path(simroot) / "RoboDojo" / path
    return path


def snapshot(worker):
    progress = read_json(worker["group_dir"] + "/group_progress.json")
    if progress["task"] != worker["task"] or progress["eval_seed"] != worker["eval_seed"]:
        raise RuntimeError("group identity mismatch")
    return {"scored": int(progress["scored"]),
            "layout_ids": sorted(int(v["layout_id"]) for v in progress["details"].values())}


def committed_boundary(config, worker, initial):
    group = Path(worker["group_dir"])
    progress = read_json(group / "group_progress.json")
    if int(progress["scored"]) <= initial["scored"]:
        return None
    if progress["task"] != worker["task"] or progress["eval_seed"] != worker["eval_seed"]:
        raise RuntimeError("group identity changed")
    result_path = host_path(progress["native_result_path"], config["simroot"])
    checkpoint_path = result_path.parent.parent / ("_resume_" + progress["native_run_id"] + ".json")
    checkpoint = read_json(checkpoint_path)
    native = read_json(result_path)
    if checkpoint["success_nums"] + checkpoint["fail_nums"] < progress["scored"]:
        return None
    if native["eval_time"] < progress["scored"]:
        return None
    new_rows = [(index, row) for index, row in progress["details"].items()
                if int(row["layout_id"]) not in initial["layout_ids"]]
    if not new_rows:
        raise RuntimeError("score count increased without a new layout")
    verified = []
    for index, row in new_rows:
        layout_id = int(row["layout_id"])
        if layout_id not in checkpoint["completed_layout_ids"]:
            return None
        episode_id = f"{worker['task']}_s{worker['eval_seed']}_l{layout_id}"
        latest = read_json(group / "episodes" / episode_id / "latest_attempt.json")
        episode_path = host_path(latest["report"], config["simroot"])
        episode = read_json(episode_path)
        if episode.get("native_score_committed") is not True:
            return None
        if episode["episode_id"] != episode_id:
            raise RuntimeError("episode report identity mismatch")
        videos = []
        for camera in ("head", "left_wrist", "right_wrist"):
            tag = "success" if row["success"] else "fail"
            path = result_path.parent / f"episode_{int(index):07d}_cam_{camera}_{tag}.mp4"
            if not path.is_file() or path.stat().st_size == 0:
                return None
            videos.append(str(path))
        verified.append({"episode_id": episode_id, "layout_id": layout_id,
                         "episode_report": str(episode_path), "videos": videos})
    return {"scored": progress["scored"], "new_committed": verified,
            "checkpoint": str(checkpoint_path), "native_result": str(result_path),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}


def work_evidence(worker):
    group = Path(worker["group_dir"])
    attempts = []
    for path in (group / "episodes").glob("*/attempt_*/execution_state.json"):
        attempts.append({"path": str(path), **read_json(path)})
    usage = Path(worker["usage_jsonl"])
    return {"execution_states": attempts, "usage_jsonl": str(usage),
            "usage_bytes": usage.stat().st_size if usage.exists() else None,
            "pending_relay_requests": sorted(str(p) for p in (Path(worker["relay_dir"]) / "requests").glob("*.json"))}


def run(config, *, execute, output, poll_seconds=.1, proc=None, killpg=os.killpg):
    if socket.gethostname().split(".")[0] != config["host"].split(".")[0]:
        raise RuntimeError("watcher must run on the configured compute node")
    if len(config["workers"]) != 2:
        raise ValueError("this drain requires exactly two explicitly identified workers")
    pids = [w[k] for w in config["workers"] for k in ("simulator_pid", "controller_pid")]
    if len(set(pids + [config["supervisor_pid"]])) != 5:
        raise ValueError("worker and supervisor PIDs must all be distinct")
    proc = proc or ProcInspector()
    report = {"mode": "execute" if execute else "dry_run", "host": socket.gethostname(),
              "supervisor_pid": config["supervisor_pid"], "started_unix": time.time(),
              "complete": False, "workers": {},
              "scope": "SIGSTOP after durable new native score; possible next-layout unscored reset is retained"}
    for worker in config["workers"]:
        report["workers"][worker["name"]] = {
            "identity": validate_identity(proc, config, worker), "initial": snapshot(worker),
            "state": "watching", "initial_work": work_evidence(worker)}
    if not execute:
        return report
    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing drain evidence file")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(json.dumps({"event": "watcher_ready", "output": str(output)}), flush=True)
    try:
        while not all(row["state"] == "frozen" for row in report["workers"].values()):
            for worker in config["workers"]:
                state = report["workers"][worker["name"]]
                if state["state"] == "frozen":
                    continue
                validate_identity(proc, config, worker, state["identity"])
                try:
                    boundary = committed_boundary(config, worker, state["initial"])
                except (FileNotFoundError, json.JSONDecodeError):
                    boundary = None
                if boundary is None:
                    continue
                identity = validate_identity(proc, config, worker, state["identity"])
                state["boundary"] = boundary
                state["state"] = "freezing"
                atomic_json(output, report)
                for role in ("simulator", "controller"):
                    pgid = identity[role]["pgid"]
                    killpg(pgid, signal.SIGSTOP)
                    state[role + "_stop_sent"] = pgid
                    atomic_json(output, report)
                # Verify STOP took effect before claiming a boundary freeze.
                until = time.monotonic() + 5
                while True:
                    stopped = {role: proc.members(identity[role]["pgid"])
                               for role in ("simulator", "controller")}
                    if all(p["state"] in ("T", "t") for group in stopped.values() for p in group):
                        break
                    if time.monotonic() >= until:
                        raise RuntimeError("SIGSTOP not verified; inspect partial drain evidence")
                    time.sleep(.02)
                state.update(state="frozen", frozen_unix=time.time(), work_at_freeze=work_evidence(worker),
                             stopped_members={role: [p["pid"] for p in group] for role, group in stopped.items()})
                atomic_json(output, report)
                print(json.dumps({"event": "worker_frozen", "worker": worker["name"],
                                  "scored": boundary["scored"]}), flush=True)
            time.sleep(poll_seconds)
        report.update(complete=True, finished_unix=time.time(),
                      next_action="Operator must coordinate supervisor shutdown and SIGCONT; watcher sends no TERM/KILL/CONT")
        atomic_json(output, report)
        return report
    except BaseException as exc:
        report.update(error=f"{type(exc).__name__}: {exc}", partial_freeze_must_be_inspected=True)
        atomic_json(output, report)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="send SIGSTOP after verified boundaries")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.execute and args.output is None:
        parser.error("--execute requires a fresh --output evidence path")
    os.umask(0o077)
    result = run(read_json(args.config), execute=args.execute, output=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
