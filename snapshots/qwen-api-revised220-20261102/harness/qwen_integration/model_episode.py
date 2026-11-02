"""One episode, multiple bounded Astra programs; credentials stay in API worker."""
from __future__ import annotations
import base64
import io
import json
import os
from pathlib import Path
import signal
import time
from collections.abc import Mapping
import numpy as np
from PIL import Image
from cap_harness.artifacts import RunRecorder
from aspire_glue import build_api, execute_program, implementation_provenance, render_api_docs
from robodojo_adapter import RoboDojoAdapter
from deployment_docs import deployed_api_documentation
from model_driver import PROPOSAL_SCHEMA_VERSION


def execute_bounded_program(api, program, recorder, *, timeout_s=300):
    """Install a best-effort watchdog in addition to the host process deadline.

    Upstream tool exception handling can convert TimeoutError into a failed tool
    receipt, allowing generated code to continue. Native C/CUDA calls can also
    delay delivery. The host launcher's recorded process deadline is the hard cap.
    """
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError("an existing process alarm prevents a bounded program deadline")
    previous_handler=signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError(f"generated program exceeded {timeout_s}s wall limit")
    signal.signal(signal.SIGALRM,expired)
    signal.setitimer(signal.ITIMER_REAL,float(timeout_s))
    try:
        return execute_program(api,program,recorder)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous_handler)


def plain(value):
    if isinstance(value,Mapping):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [plain(v) for v in value]
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if value is None or isinstance(value,(str,bool,int,float)):
        return value
    return str(value)


def public_observation(adapter):
    observation=adapter.get_observation()
    state=observation.robot_state
    image_parts=[]
    calibration={}
    for name,camera in observation.cameras.items():
        buffer=io.BytesIO()
        Image.fromarray(camera.rgb).save(buffer,format="PNG",compress_level=1)
        image_parts.append({"camera":name,"data_url":"data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode()})
        calibration[name]={"shape":list(camera.rgb.shape),"K":camera.intrinsics.tolist(),
            "world_from_camera":camera.camera_pose.as_matrix().tolist(),
            "depth":"metric optical depth, accessible in code via get_observation().cameras[name].depth_m; invalid rays are0"}
    poses={a:{"position":p.position.tolist(),"quaternion_wxyz":p.quaternion_wxyz.tolist(),"frame":p.frame}
           for a,p in state.end_effector_poses.items()}
    return {"robot_state":{"embodiment":state.embodiment,"base_frame":state.base_frame,
                "joint_positions":plain(state.joint_positions),"joint_names":plain(state.joint_names),
                "end_effector_poses":poses,"gripper_positions":plain(state.gripper_positions),
                "controller":adapter.get_controller_metadata()},
            "camera_calibration":calibration,"images":image_parts,
            "elapsed_sim_s":adapter.current_time_s,
            "remaining_actions":int(adapter.native_env.step_lim-adapter.native_env.take_action_cnt[0])}


def public_feedback(outcome):
    # Trace receipts and generated program's own public result only. No scorer data.
    result=plain(outcome.result)
    encoded=json.dumps(result,allow_nan=False)
    if len(encoded)>12000:
        result={"truncated_program_result":encoded[:12000]}
    return {"program_ok":bool(outcome.ok),"result":result,"stdout":outcome.stdout[-4000:],
        "error":None if outcome.error is None else str(outcome.error),
        "tool_calls":[{"tool":c.tool,"ok":c.ok,"duration_s":c.duration_s,
            "failure_code":c.failure_code,"error_message":c.error_message,
            "inputs_summary":plain(c.inputs_summary),"outputs_summary":plain(c.outputs_summary),
            "frame":c.frame,"provider":c.provider,"result_type":c.result_type}
            for c in outcome.calls[-60:]]}


def run_model_episode(task_env, *, client_factory=None):
    # The generated-code process must never inherit authentication material.
    forbidden=[k for k in os.environ if k in {"OPENAI_API_KEY","OPENAI_API_KEY_FILE","ARK_API_KEY","ARK_API_KEY_BACKUP","SAFEHARNESS_KEY_FILE"}]
    if forbidden:
        raise RuntimeError("simulator inherited API credential variables: "+",".join(forbidden))
    if client_factory is None:
        from model_relay import RelayModelClient
        client_factory=RelayModelClient
    root=Path(os.environ["ASPIRE_OUTPUT_DIR"]).resolve()
    root.mkdir(parents=True,exist_ok=True)
    episode_id=os.environ["ASPIRE_EPISODE_ID"]
    if task_env.task_name not in {"general_pickup","stack_bowls","align_blocks"}:
        raise ValueError("task is outside the authorized three-task pilot")
    if int(task_env.env_seeds[0]) != 0 or episode_id != f"{task_env.task_name}_s0_l0":
        raise ValueError("pilot task IDs and layout must match frozen seed0/layout0 configuration")
    max_iterations=int(os.environ.get("ASPIRE_MAX_ITERATIONS","8"))
    if not 1<=max_iterations<=8:
        raise ValueError("model pilot permits1..8programiterations per episode")
    recorder=RunRecorder(output_root=root/"artifacts",benchmark="RoboDojo",suite="pilot",
        task_id={"general_pickup":0,"stack_bowls":1,"align_blocks":2}[task_env.task_name],
        task_name=task_env.task_name,seed=0,max_steps=int(task_env.step_lim),
        camera_width=640,camera_height=480,control_frequency_hz=25,
        capture_videos=False,step_video=False,
        providers={"planner":"RoboDojo-native-CuRobo","sam3":None,"graspnet":None})
    adapter=RoboDojoAdapter(task_env,recorder=recorder)
    report={"proposal_schema":PROPOSAL_SCHEMA_VERSION,"episode_id":episode_id,"task":task_env.task_name,"layout_id":int(task_env.env_seeds[0]),
        "seed":0,"model":os.environ.get("ASPIRE_MODEL_NAME","gpt-6-astra"),"ok":False,"artifact_dir":str(recorder.root),
        "implementation_provenance":implementation_provenance(),"iterations":[],
        "scope":"ASPIRE public API/program executor + custom ARX X5 environment/planning bridge; no SAM3 or grasp provider"}
    started=time.monotonic()
    outcome=None
    reason="iteration_limit"
    try:
        report["camera_calibration"]=adapter.calibration_sanity()
        initial=adapter.get_observation()
        recorder.on_reset(initial,{"task":task_env.task_name,"layout_id":int(task_env.env_seeds[0]),"episode_id":episode_id})
        api=build_api(adapter,recorder=recorder)
        api_docs=deployed_api_documentation(api)
        (root/"public_api.txt").write_text(api_docs)
        if os.environ.get("ASPIRE_MODEL_NAME") == "Qwen/Qwen3.8-27B":
            from qwen_gpu_guard import verify_simulator_gpu
            report["gpu_isolation"] = verify_simulator_gpu(root/"simulator.log", os.environ["ASPIRE_SIM_GPU_UUID"])
        client=client_factory(Path(os.environ["ASPIRE_RELAY_DIR"]),timeout_s=480)
        previous=None
        for index in range(max_iterations):
            if task_env.is_episode_end():
                reason="environment_ended";break
            sensed=public_observation(adapter)
            # This public snapshot contains the exact image bytes supplied to Astra.
            request_path=root/f"public_input_{index:02d}.json"
            request_path.write_text(json.dumps({"task_instruction":initial.task_context.language,
                "public_observation":sensed,"previous_result":previous},allow_nan=False))
            proposal=client.next_program(episode_id=episode_id,task_instruction=initial.task_context.language,
                public_observation=sensed,previous_result=previous,api_documentation=api_docs)
            entry={"iteration":proposal.iteration,"summary":proposal.summary,"give_up":proposal.give_up,
                "usage":plain(proposal.usage),"budget":plain(proposal.budget),"source":f"program_{index:02d}.py"}
            report["iterations"].append(entry)
            if proposal.give_up:
                reason="model_give_up";break
            program=root/f"program_{index:02d}.py"
            program.write_text(proposal.code,encoding="utf-8")
            outcome=execute_bounded_program(api,program,recorder)
            previous=public_feedback(outcome)
            entry["execution"]=previous
            entry["environment_steps_after"]=int(task_env.take_action_cnt[0])
            (root/"episode_progress.json").write_text(json.dumps(report,indent=2,default=str))
            if task_env.is_episode_end():
                reason="environment_ended";break
        official=adapter.official_status()
        report.update(ok=True,termination_reason=reason,official=official,
            total_programs_executed=sum(not i["give_up"] for i in report["iterations"]))
        recorder.finalize(program_ok=bool(outcome is not None and outcome.ok),
            task_success=official["official_success"],protocol_success=True,
            termination_reason="task_succeeded" if official["official_success"] else "program_completed",
            program_result={"iterations":report["iterations"],"driver_stop_reason":reason})
        return report
    except BaseException as exc:
        report["error"]=f"{type(exc).__name__}: {exc}"
        recorder.close_incomplete(exc)
        raise
    finally:
        report["wall_seconds"]=time.monotonic()-started
        official=adapter.official_status()
        report.setdefault("official",official)
        (root/"episode_report.json").write_text(json.dumps(report,indent=2,default=str))
        task_env.success[0]=bool(official["official_success"])
        task_env.end_flag[0]=True
        adapter.close()
