"""Compare copied local Qwen pilot artifacts with the recorded Astra pilot.

This is offline analysis only: no simulator imports, credentials, network or
model calls. Missing/incomplete evidence is reported as unknown, never failure.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parent
TASKS = ("general_pickup", "stack_bowls", "align_blocks")
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
LIMITS = dict(zip(TASKS, (200, 800, 200)))


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def read_ledger(path, warnings):
    if not path.is_file():
        warnings.append(f"Missing ledger: {path}")
        return
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                warnings.append(f"Incomplete/invalid JSONL line {number}: {path}")


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sum_present(values):
    values = list(values)
    return sum(values) if values and all(number(x) for x in values) else None


def first_astra_observation(taskdir, episode_id, warnings):
    direct = read_json(taskdir / "public_input_00.json")
    if direct is not None:
        return direct, str(taskdir / "public_input_00.json")
    ledger = taskdir.parent / "usage.jsonl"
    for event in read_ledger(ledger, warnings):
        if (event.get("event") != "program_input" or event.get("episode_id") != episode_id
                or event.get("iteration") != 1):
            continue
        content = event["input"][0]["content"]
        context = json.loads(content[0]["text"])
        observation = json.loads(content[1]["text"])
        images = []
        for index, part in enumerate(content):
            if part.get("type") == "input_image":
                label = content[index-1]["text"]
                if not label.startswith("Public camera: "):
                    raise ValueError("Unexpected historical camera label")
                images.append({"camera": label.removeprefix("Public camera: "),
                               "data_url": part["image_url"]})
        observation["images"] = images
        return {"task_instruction": context["task_instruction"], "public_observation": observation,
                "previous_result": context.get("previous_execution")}, str(ledger)
    return None, None


def numeric_comparison(left, right, tolerance=1e-5):
    """Report field-level differences without mistaking render noise for mismatch."""
    structural, textual, differences = [], [], []

    def visit(a, b, path):
        if number(a) and number(b):
            differences.append((path, abs(a-b)))
        elif isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                structural.append({"path": path, "astra_only": sorted(set(a)-set(b)),
                                   "qwen_only": sorted(set(b)-set(a))})
            for key in sorted(set(a) & set(b)):
                visit(a[key], b[key], path + "." + key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                structural.append({"path": path, "lengths": [len(a), len(b)]})
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}[{index}]")
        elif a != b:
            textual.append({"path": path, "astra": a, "qwen": b})

    visit(left, right, "root")
    return {"structure_equal": not structural, "structural_differences": structural,
            "nonnumeric_differences": textual, "numeric_fields_compared": len(differences),
            "max_absolute_numeric_difference": max((d for _, d in differences), default=None),
            "absolute_tolerance": tolerance,
            "numeric_within_tolerance": bool(differences) and all(d <= tolerance for _, d in differences),
            "largest_numeric_differences": [{"path": p, "absolute_difference": d}
                                             for p, d in sorted(differences, key=lambda x: -x[1])[:12]]}


def image_info(image):
    header, encoded = image["data_url"].split(",", 1)
    raw = base64.b64decode(encoded, validate=True)
    shape = None
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR":
        shape = list(struct.unpack(">II", raw[16:24]))
    return {"camera": image["camera"], "mime": header.split(";")[0].removeprefix("data:"),
            "image_bytes_sha256": hashlib.sha256(raw).hexdigest(), "size_wh": shape}


def observation_parity(astra, qwen, astra_source, qwen_source):
    if astra is None or qwen is None:
        return {"available": False, "astra_source": astra_source, "qwen_source": qwen_source}
    a, q = astra["public_observation"], qwen["public_observation"]
    ai, qi = [image_info(x) for x in a["images"]], [image_info(x) for x in q["images"]]
    all_views = all([x["camera"] for x in images] == list(CAMERAS) for images in (ai, qi))
    camera_checks = []
    for camera in CAMERAS:
        x = next((v for v in ai if v["camera"] == camera), None)
        y = next((v for v in qi if v["camera"] == camera), None)
        ac, qc = a.get("camera_calibration", {}).get(camera), q.get("camera_calibration", {}).get(camera)
        camera_checks.append({"camera": camera, "astra_image": x, "qwen_image": y,
            "both_640x480": bool(x and y and x["size_wh"] == y["size_wh"] == [640,480]),
            "exact_image_bytes_equal": bool(x and y and x["image_bytes_sha256"] == y["image_bytes_sha256"]),
            "both_offer_metric_depth_and_calibration": all(isinstance(c, dict)
                and all(k in c for k in ("shape", "K", "world_from_camera", "depth")) for c in (ac,qc)),
            "calibration_comparison": numeric_comparison(ac,qc)})
    fields = {"robot_state", "camera_calibration", "images", "elapsed_sim_s", "remaining_actions"}
    return {"available": True, "astra_source": astra_source, "qwen_source": qwen_source,
        "task_instruction_equal": astra["task_instruction"] == qwen["task_instruction"],
        "astra_instruction": astra["task_instruction"], "qwen_instruction": qwen["task_instruction"],
        "public_field_set_equal": set(a) == set(q) == fields,
        "all_three_ordered_camera_views": all_views,
        "all_three_modalities_present": all_views and all(x["both_640x480"]
            and x["both_offer_metric_depth_and_calibration"] for x in camera_checks)
            and all(isinstance(x.get("robot_state"),dict) for x in (a,q)),
        "modalities": ["three RGB views", "metric depth accessible through code plus calibration", "robot proprioception"],
        "robot_state_comparison": numeric_comparison(a.get("robot_state"),q.get("robot_state")),
        "initial_elapsed_sim_s": {"astra": a.get("elapsed_sim_s"), "qwen": q.get("elapsed_sim_s")},
        "initial_remaining_actions": {"astra": a.get("remaining_actions"), "qwen": q.get("remaining_actions")},
        "cameras": camera_checks,
        "interpretation": "Image hashes compare encoded PNG bytes, not physical scene equivalence. Rendering noise or PNG encoding can differ. Depth availability is checked from the public contract; raw depth arrays are not present in these model-input snapshots and are not compared."}


def qwen_usage(path, warnings):
    by_episode = defaultdict(lambda: {"generation_attempts": 0, "completed_usage_records": 0,
        "prompt_tokens": 0, "completion_tokens_including_reasoning": 0,
        "reasoning_tokens_reported": 0, "reasoning_records": 0,
        "generation_wall_s": 0.0, "tokenize_wall_s": 0.0, "unknown_usage_attempts": 0,
        "response_outcomes": [], "output_budget_exhausted": False})
    seen, config = set(), None
    for row in read_ledger(path, warnings):
        event, ep = row.get("event"), row.get("episode_id")
        if event == "suite_initialized":
            config = row.get("config")
        if not ep:
            continue
        key = (event,ep,row.get("iteration"))
        if event in {"generation_started","generation_usage","raw_response","generation_transport_failure","input_token_count"}:
            if key in seen:
                warnings.append(f"Duplicate ledger record: {key}; not double-counted")
                continue
            seen.add(key)
        item = by_episode[ep]
        if event == "generation_started":
            item["generation_attempts"] += 1
        elif event == "input_token_count":
            item["tokenize_wall_s"] += row["wall_s"]
        elif event in ("raw_response","generation_transport_failure"):
            item["generation_wall_s"] += row["generation_wall_s"]
            if event == "raw_response":
                response = row["response"]
                choices = response.get("choices") or []
                choice = choices[0] if len(choices) == 1 else {}
                raw_usage = response.get("usage") or {}
                completion = raw_usage.get("completion_tokens")
                reasoning = (raw_usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                cap = (config or {}).get("max_output_tokens",8192)
                exhausted = choice.get("finish_reason") == "length" and completion == cap
                item["output_budget_exhausted"] |= exhausted
                item["response_outcomes"].append({"iteration":row.get("iteration"),
                    "finish_reason":choice.get("finish_reason"),"completion_tokens":completion,
                    "reasoning_tokens":reasoning,"configured_output_cap":cap,
                    "final_content_characters":len(choice.get("message",{}).get("content") or ""),
                    "output_budget_exhausted":exhausted})
        elif event == "generation_usage":
            usage = row["usage"]
            item["completed_usage_records"] += 1
            item["prompt_tokens"] += usage["prompt_tokens"]
            item["completion_tokens_including_reasoning"] += usage["completion_tokens"]
            reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if reasoning is not None:
                item["reasoning_tokens_reported"] += reasoning
                item["reasoning_records"] += 1
    for item in by_episode.values():
        item["unknown_usage_attempts"] = item["generation_attempts"]-item["completed_usage_records"]
        if item["reasoning_records"] != item["completed_usage_records"] or not item["completed_usage_records"]:
            item["reasoning_tokens_reported"] = None
    keys = ("generation_attempts","completed_usage_records","prompt_tokens",
            "completion_tokens_including_reasoning","generation_wall_s","tokenize_wall_s","unknown_usage_attempts")
    totals = {k: sum(v[k] for v in by_episode.values()) for k in keys}
    totals["reasoning_tokens_reported"] = sum_present(v["reasoning_tokens_reported"] for v in by_episode.values())
    totals["ledger_available"] = path.is_file()
    return {"configuration": config, "episodes": dict(by_episode), "totals": totals,
            "accounting_note": "Completion tokens already include reasoning. Do not add reasoning tokens again. Tokenization and generation durations are separate sequential phases, counted once from their own ledger events. Missing usage is not estimated."}


def load_run(run_dir, server_dir, warnings):
    launcher = read_json(run_dir/"launcher_summary.json") or {}
    usage = qwen_usage(run_dir/"usage.jsonl", warnings)
    recorded_attempts=launcher.get("attempts",[])
    attempts = {a["task"]:a for a in recorded_attempts}
    if len(attempts)!=len(recorded_attempts):
        raise ValueError(f"Repeated task attempts in launcher evidence: {run_dir}")
    # A started process counts as an attempted task even if it never produced a
    # report or generation. A configured-but-never-started task does not.
    attempted = set(attempts)
    if (run_dir/"attempts.jsonl").is_file():
        started=set()
        for event in read_ledger(run_dir/"attempts.jsonl",warnings):
            if event.get("event")=="episode_process_start":
                if event["task"] in started:
                    raise ValueError(f"Repeated task process start in run: {event['task']}")
                started.add(event["task"])
                attempted.add(event["task"])
    for task in TASKS:
        if ((run_dir/task/"episode_report.json").exists()
                or (run_dir/task/"public_input_00.json").exists()
                or task+"_s0_l0" in usage["episodes"]):
            attempted.add(task)
    return {"run_dir":run_dir,"server_dir":server_dir,"launcher":launcher,
            "experiment":read_json(run_dir/"experiment_config.json") or {},
            "server":read_json(server_dir/"server_suite.json") or {},
            "smoke":read_json(server_dir/"smoke/model_smoke.json") or {},
            "snapshot":read_json(run_dir/"source_snapshot.json") or {},
            "usage":usage,"attempts":attempts,"attempted_tasks":attempted}


def merge_usage(runs):
    merged={"configuration":runs[0]["usage"]["configuration"],"episodes":{},
            "configurations_by_run":{},"accounting_note":runs[0]["usage"]["accounting_note"]}
    for run in runs:
        merged["configurations_by_run"][str(run["run_dir"])]=run["usage"]["configuration"]
        for ep,item in run["usage"]["episodes"].items():
            if ep in merged["episodes"]:
                raise ValueError(f"Duplicate episode ledger across runs; refusing to merge retries: {ep}")
            merged["episodes"][ep]={**item,"source_ledger":str(run["run_dir"]/"usage.jsonl")}
    keys=("generation_attempts","completed_usage_records","prompt_tokens",
          "completion_tokens_including_reasoning","generation_wall_s","tokenize_wall_s","unknown_usage_attempts")
    merged["totals"]={k:sum(run["usage"]["totals"][k] for run in runs) for k in keys}
    merged["totals"]["reasoning_tokens_reported"]=sum_present(
        v["reasoning_tokens_reported"] for v in merged["episodes"].values())
    merged["totals"]["ledger_available"]=all(run["usage"]["totals"]["ledger_available"] for run in runs)
    return merged


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def build_comparison(run_dir, server_dir, astra_summary, continuation_dir=None, continuation_server_dir=None):
    warnings = []
    astra = read_json(astra_summary)
    if astra is None:
        raise FileNotFoundError(astra_summary)
    if (continuation_dir is None) != (continuation_server_dir is None):
        raise ValueError("continuation-dir and continuation-server-dir must be supplied together")
    runs=[load_run(run_dir,server_dir,warnings)]
    if continuation_dir is not None:
        if continuation_dir == run_dir:
            raise ValueError("continuation must be a distinct copied run directory")
        runs.append(load_run(continuation_dir,continuation_server_dir,warnings))
    task_sources={}
    for run in runs:
        for task in run["attempted_tasks"]:
            if task not in TASKS:
                raise ValueError(f"Unexpected attempted task: {task}")
            if task in task_sources:
                raise ValueError(f"Task attempted in both runs; refusing to merge retries: {task}")
            task_sources[task]=run
    usage = merge_usage(runs)
    experiment,server=runs[0]["experiment"],runs[0]["server"]
    astra_rows = {r["task"]:r for r in astra["results"]}
    rows = []
    for task in TASKS:
        ar = astra_rows[task]
        source=task_sources.get(task,runs[0])
        task_run_dir=source["run_dir"]
        report_path = task_run_dir/task/"episode_report.json"
        report = read_json(report_path)
        episode = task+"_s0_l0"
        episode_usage = usage["episodes"].get(episode)
        recorded_proposals = len(report.get("iterations",[])) if report else None
        budget_exhausted=bool(episode_usage and episode_usage["output_budget_exhausted"])
        qr = {"report_available": report is not None, "report": str(report_path),
            "source_run_dir":str(task_run_dir),"source_server_dir":str(source["server_dir"]),
            "attempted":task in task_sources,
            "outcome_class":"model_output_budget_exhausted" if budget_exhausted else
                "official_task_result" if report and report.get("ok") is True else
                "incomplete_or_unknown_failure" if task in task_sources else "not_attempted",
            "protocol_complete": report.get("ok") if report else None,
            "official_success": report.get("official",{}).get("official_success") if report else None,
            "model_calls": episode_usage["generation_attempts"] if episode_usage else recorded_proposals,
            "model_calls_source": "generation_started ledger" if episode_usage else "report proposals only; ledger unavailable",
            "recorded_model_proposals": recorded_proposals,
            "control_steps_used": report.get("official",{}).get("environment_steps") if report else None,
            "control_step_limit": report.get("official",{}).get("step_limit") if report else None,
            "episode_loop_seconds": report.get("wall_seconds") if report else None,
            "process_seconds_including_startup_shutdown": source["attempts"].get(task,{}).get("wall_seconds"),
            "termination_reason": report.get("termination_reason") if report else None,
            "error": report.get("error") if report else None,
            "ledger_usage": episode_usage}
        checks = {"episode_identity": bool(report and report.get("episode_id")==episode
            and report.get("task")==task and report.get("seed")==0 and report.get("layout_id")==0),
            "model_identity": bool(report and report.get("model")=="Qwen/Qwen3.8-27B"),
            "native_action_budget": bool(report and qr["control_step_limit"]==LIMITS[task]
                and type(qr["control_steps_used"]) is int and 0<=qr["control_steps_used"]<=LIMITS[task]),
            "model_round_budget": bool(report and 1<=qr["model_calls"]<=8),
            "ledger_call_count": bool(report and qr["ledger_usage"] and
                qr["ledger_usage"]["generation_attempts"]==recorded_proposals)}
        ap = Path(ar["report"])
        if not ap.is_file():
            ap = astra_summary.parent/ap.parent.parent.name/task/"episode_report.json"
        initial_a, source_a = first_astra_observation(ap.parent,episode,warnings)
        qi = task_run_dir/task/"public_input_00.json"
        parity = observation_parity(initial_a,read_json(qi),source_a,str(qi) if qi.is_file() else None)
        api_hashes={"astra":file_hash(ap.parent/"public_api.txt"),"qwen":file_hash(task_run_dir/task/"public_api.txt")}
        api_parity={**api_hashes,"equal":bool(api_hashes["astra"] and api_hashes["astra"]==api_hashes["qwen"])}
        fixed_snapshot=read_json(astra_summary.parent/"astra3_remaining_27940636/source_snapshot.json") or {}
        critical=("robodojo_adapter.py","native_retime.py","aspire_glue.py","deployment_docs.py","model_driver.py")
        source_parity={n:{"qwen":source["snapshot"].get(n),
            "fixed_astra":fixed_snapshot.get("files",{}).get("integration/"+n)} for n in critical}
        source_match=all(v["qwen"] is not None and v["qwen"]==v["fixed_astra"] for v in source_parity.values())
        comparable = task != "general_pickup"
        rows.append({"task":task,"seed":0,"layout_id":0,"astra":ar,"qwen":qr,
            "qwen_consistency_checks":checks,"initial_observation_parity":parity,
            "api_documentation_parity":api_parity,"fixed_adapter_source_parity":source_parity,
            "fixed_adapter_sources_match":source_match,
            "astra_capability_measurement_valid":comparable,
            "paired_eligible":comparable and qr["protocol_complete"] is True and all(checks.values())
                and parity.get("available") is True and parity.get("task_instruction_equal") is True
                and parity.get("public_field_set_equal") is True and parity.get("all_three_modalities_present") is True
                and api_parity["equal"] and source_match,
            "comparison_note":"Astra旧适配器拒绝任务相关运动；排除模型能力胜负比较，不重跑。" if not comparable else
                "可对照同布局下的这一次结果；仍需结合协议、初始观察差异和轨迹审计，不能推出总体成功率。"})
    manifest = experiment.get("model_manifest",{})
    def runtime_sum(key):
        return sum_present(run["server"].get("timings_seconds",{}).get(key) for run in runs)
    return {"schema_version":2,"created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources":{"qwen_run_dir":str(run_dir),"qwen_server_dir":str(server_dir),"astra_summary":str(astra_summary),
            "continuation_run_dir":str(continuation_dir) if continuation_dir else None,
            "continuation_server_dir":str(continuation_server_dir) if continuation_server_dir else None},
        "run_completion_history":[{"run_dir":str(r["run_dir"]),"launcher_complete":r["launcher"].get("complete"),
            "server_suite_complete":r["server"].get("complete"),"attempted_tasks":sorted(r["attempted_tasks"])} for r in runs],
        "qwen_launcher_complete":all(r["launcher"].get("complete") is True for r in runs),
        "qwen_server_suite_complete":all(r["server"].get("complete") is True for r in runs),
        "all_authorized_tasks_attempted_once":set(task_sources)==set(TASKS),
        "duplicate_task_attempts":False,
        "tasks":rows,"qwen_usage":usage,"astra_totals":astra["totals"],
        "qwen_runtime":{"episode_loop_seconds_sum":sum_present(r["qwen"]["episode_loop_seconds"] for r in rows),
            "simulator_process_seconds_sum":sum_present(r["qwen"]["process_seconds_including_startup_shutdown"] for r in rows),
            "recorded_task_reports":sum(r["qwen"]["report_available"] for r in rows),
            "observed_episode_loop_seconds_sum":sum_present(r["qwen"]["episode_loop_seconds"] for r in rows if r["qwen"]["report_available"]),
            "observed_process_seconds_sum":sum_present(r["qwen"]["process_seconds_including_startup_shutdown"] for r in rows if r["qwen"]["report_available"]),
            "suite_launcher_seconds":sum_present(r["launcher"].get("wall_seconds") for r in runs),
            "server_load_and_ready_seconds":runtime_sum("server_startup"),
            "isolated_smoke_seconds":runtime_sum("smoke"),"server_supervisor_total_seconds":runtime_sum("total"),
            "server_cleanup_seconds":runtime_sum("cleanup"),"preparation_download_seconds":manifest.get("download_wall_seconds"),
            "installation_seconds":None,"smoke_model_generation_seconds":sum_present(r["smoke"].get("generation_wall_s") for r in runs),
            "smoke_tokenize_seconds":sum_present(r["smoke"].get("tokenize_wall_s") for r in runs),
            "runs":[{"run_dir":str(r["run_dir"]),"server_dir":str(r["server_dir"]),
                "timings_seconds":r["server"].get("timings_seconds",{}),"smoke_usage":r["smoke"].get("usage")} for r in runs],
            "note":"任务loop包含模型等待与执行，进程时间另含Isaac启动/退出。服务器加载与接口smoke单列；smoke不算任务调用。下载/准备来自独立manifest，可能包含访问检查；安装时间未记录，不估计。"},
        "qwen_model":{"name":server.get("model") or experiment.get("model"),
            "revision":server.get("model_revision") or manifest.get("revision"),
            "package_versions":server.get("package_versions"),"server_devices":server.get("server_devices"),
            "simulator_device":server.get("simulator_device")},
        "protocol_differences":{"request_timeout_s":{"astra":180,"qwen":experiment.get("request_timeout_s")},
            "process_timeout_s":{"astra":1200,"qwen":experiment.get("process_timeout_s")},
            "sam3":experiment.get("sam3"),"contact_graspnet":experiment.get("contact_graspnet")},
        "interpretation_zh":["general_pickup 的Astra结果因旧适配器错误无效，不计入两模型能力对照。",
            "仅stack_bowls和align_blocks具备Astra配对基线。明确的输出token预算耗尽属于预算内未解出任务，但须与正常结束的物理失败、未知基础设施异常分列，不合成常规配对成功率。",
            "初始图像哈希不同可能来自渲染噪声或PNG编码；须结合标定、机器人状态和实际画面，不能只靠哈希认定布局不同。",
            "两模型视觉编码器、分词器和采样实现不同；同一输入内容与数值token上限不代表相同计算量。",
            "固定重定时和夹爪等待消耗原生动作预算；结果反映模型与当前桥接配置共同表现。",
            "这是三个固定任务各一次的小测试，不外推RoboDojo总体成功率、排名或全量成本。",
            "本地Qwen未调用付费云API时外部推理API费用为0；GPU占用、电费与机会成本并非零，未在此估算。"],
        "warnings":warnings}


def render_markdown(data):
    def fmt(v):
        return "—" if v is None else f"{v:.2f}" if isinstance(v,float) else str(v)
    def success(v, invalid=False):
        return "无效（适配器故障）" if invalid else "成功" if v is True else "失败" if v is False else "未知"
    lines=["本报告读取本地保存的真实报告与日志。Astra 的 general_pickup 遇到旧适配器故障，仅其余两个任务可作配对观察。", "",
        "| 任务 | 模型 | 结果／终止原因 | 调用数 | 动作步数 | 回合 loop 秒 | 完整进程秒 |",
        "|---|---|---|---:|---:|---:|---:|"]
    for row in data["tasks"]:
        for model,key in [("Astra","astra"),("Qwen3.8-27B","qwen")]:
            v=row[key]; label=success(v.get("official_success"),key=="astra" and not row["astra_capability_measurement_valid"])
            if key=="qwen" and v["protocol_complete"] is not True and v["report_available"]:
                label="输出预算耗尽（任务未解出）" if v["outcome_class"]=="model_output_budget_exhausted" else "未完成（原因待审计）"
            if key=="qwen" and v["outcome_class"]=="not_attempted":
                label="尚未尝试"
            lines.append(f"| {row['task']} | {model} | {label} | {fmt(v.get('model_calls'))} | {fmt(v.get('control_steps_used'))}/{fmt(v.get('control_step_limit'))} | {fmt(v.get('episode_loop_seconds'))} | {fmt(v.get('process_seconds_including_startup_shutdown'))} |")
    u=data["qwen_usage"]["totals"]; t=data["qwen_runtime"]
    lines += ["",f"Qwen任务日志：发起 {u['generation_attempts']} 次生成，已取得用量 {u['completed_usage_records']} 次；输入 {u['prompt_tokens']} tokens，输出 {u['completion_tokens_including_reasoning']} tokens（已含thinking，不重复相加）。生成累计 {fmt(u['generation_wall_s'])} 秒，token计数累计 {fmt(u['tokenize_wall_s'])} 秒。"]
    if u["unknown_usage_attempts"] or not u["ledger_available"]:
        lines.append(f"用量记录不完整：{u['unknown_usage_attempts']} 次已发起生成缺少完整用量；缺失部分未估算。")
    for row in data["tasks"]:
        if row["qwen"]["outcome_class"]=="model_output_budget_exhausted":
            evidence=next(x for x in row["qwen"]["ledger_usage"]["response_outcomes"] if x["output_budget_exhausted"])
            lines += ["",f"{row['task']} 第{evidence['iteration']}次生成以 finish_reason=length 结束，达到{evidence['completion_tokens']} token输出上限，其中thinking={fmt(evidence['reasoning_tokens'])}，最终内容长度={evidence['final_content_characters']}。这是已知的模型输出预算耗尽，任务在该预算内未解出；protocol_complete仍为false，不能伪装成正常完成的物理回合，也不是未知基础设施故障。未重试该任务。"]
    lines += ["",f"服务器加载至就绪 {fmt(t['server_load_and_ready_seconds'])} 秒；独立接口smoke {fmt(t['isolated_smoke_seconds'])} 秒；已记录{t['recorded_task_reports']}个任务的loop合计 {fmt(t['observed_episode_loop_seconds_sum'])} 秒，模拟器完整进程合计 {fmt(t['observed_process_seconds_sum'])} 秒。上述嵌套时间不能全部相加。下载/准备 {fmt(t['preparation_download_seconds'])} 秒另列，安装耗时未记录。smoke不计入任务调用数。", ""]
    for row in data["tasks"]:
        p=row["initial_observation_parity"]
        if not p["available"]:
            lines.append(f"- {row['task']}：缺少初始观察，无法核对输入一致性。")
        else:
            same=sum(c["exact_image_bytes_equal"] for c in p["cameras"])
            state=p["robot_state_comparison"]
            delta=state['max_absolute_numeric_difference']
            delta_text="未知" if delta is None else f"{delta:.3g}"
            lines.append(f"- {row['task']}：三路RGB、深度接口/标定和机器人状态{'齐全' if p['all_three_modalities_present'] else '存在缺项'}；初始机器人数值最大差 {delta_text}，在1e-5阈值内={state['numeric_within_tolerance']}；PNG字节完全相同 {same}/3。")
            lines.append(f"  API文档与Astra一致={row['api_documentation_parity']['equal']}；固定适配器/规划/提示源码哈希一致={row['fixed_adapter_sources_match']}。")
    if data["sources"]["continuation_run_dir"]:
        lines += ["", "合并来源（不同任务只接入一次，不覆盖或重试已尝试任务）："]
        for row in data["tasks"]:
            if row["qwen"]["attempted"]:
                lines.append(f"- {row['task']}：[原始报告]({row['qwen']['report']})")
    lines += ["", "PNG哈希差异可能来自渲染噪声或编码差异，不单独认定布局不同。原始深度未直接写入模型输入，因此这里只核对深度接口可用性，不能证明深度数组完全一致。", "",
        f"Qwen请求/进程时限为 {fmt(data['protocol_differences']['request_timeout_s']['qwen'])}/{fmt(data['protocol_differences']['process_timeout_s']['qwen'])} 秒，Astra为180/1200秒；任务动作、轮数和token预算保持原合同。SAM3与Contact-GraspNet配置分别为 {data['protocol_differences']['sam3']}、{data['protocol_differences']['contact_graspnet']}。", "",
        "本地Qwen外部推理API费用为0的前提是没有调用云API；GPU时间、电费与机会成本另计。本测试不外推基准总体成功率、排名或全量费用。"]
    if data["warnings"]:
        lines += ["", "证据缺项："]+["- "+x for x in data["warnings"]]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--server-dir",type=Path,required=True)
    parser.add_argument("--continuation-dir",type=Path)
    parser.add_argument("--continuation-server-dir",type=Path)
    parser.add_argument("--astra-summary",type=Path,default=ROOT/"results/pilot_summary.json")
    args=parser.parse_args()
    run=args.run_dir.resolve(strict=True); server=args.server_dir.resolve(strict=True)
    if not run.is_dir() or not server.is_dir():
        raise ValueError("run-dir and server-dir must be copied local artifact directories")
    if (args.continuation_dir is None)!=(args.continuation_server_dir is None):
        parser.error("--continuation-dir and --continuation-server-dir are required together")
    continuation=args.continuation_dir.resolve(strict=True) if args.continuation_dir else None
    continuation_server=args.continuation_server_dir.resolve(strict=True) if args.continuation_server_dir else None
    if continuation and (not continuation.is_dir() or not continuation_server.is_dir()):
        raise ValueError("continuation paths must be copied local artifact directories")
    data=build_comparison(run,server,args.astra_summary.resolve(strict=True),continuation,continuation_server)
    (run/"comparison.json").write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    (run/"comparison.md").write_text(render_markdown(data))
    print(json.dumps({"json":str(run/"comparison.json"),"markdown":str(run/"comparison.md"),
                      "launcher_complete":data["qwen_launcher_complete"],"warnings":len(data["warnings"])},ensure_ascii=False))


if __name__ == "__main__":
    main()
