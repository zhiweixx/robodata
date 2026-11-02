"""SAM3/CGN interface smoke on the already-exposed general_pickup s0/l0 only.

The native lifecycle, sensor/action implementation and one hold tick are reused.
No model program is requested and no predicted grasp is executed. Candidate
reachability is recorded separately from provider/frame/interface validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit


CGN_COMMIT = "da3dcfb2f53e43b186083ee4a9d1e232f73efc98"
# Read-only inspection of that clean remote commit verified the actual inference
# build_6d_grasp, matching contact indices, and checkpoint postprocessing.
CGN_FILES = {
    "contact_graspnet_pytorch/contact_graspnet.py":
        "17adfc04f5667c4a1801592105bea9c5a7d619977319b3a059240e658f041e80",
    "contact_graspnet_pytorch/contact_grasp_estimator.py":
        "d063d4379f838fdabf82aa970161b048f32519203ef5b980f47600f7b9345356",
    "contact_graspnet_pytorch/utils.py":
        "4acb887871b575043170bf1410c6c73482711af9240df48fab617a2aa22bfd38",
    "checkpoints/contact_graspnet/config.yaml":
        "7ae6cc15c726e8ec2c46b0c799d58ca95caeb9ce9f261aaaf3e467712df76ac5",
}
X5_URDF_SHA256 = "7aaa308a1a5b6a213d914db86ba09ddd6252f25cb498ac9a148bfa6c7166a423"


def checked_smoke_identity(task, seed, layout_ids, *, mode, count):
    if (task != "general_pickup" or int(seed) != 0 or str(layout_ids) != "0"
            or mode != "interface_smoke" or str(count) != "1"):
        raise ValueError("perception smoke requires general_pickup seed 0 layout 0, interface_smoke count 1")


def instruction_query(instruction):
    """Use public language only; no task asset names, pixels or object poses."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("public task instruction is required")
    match = re.search(r"\b(?:pick\s+up|lift|grasp)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\s+by\s+|[.!?]|$)",
                      instruction.strip(), re.IGNORECASE)
    if match and 0 < len(match[1].strip()) <= 120:
        return match[1].strip(), "public_task_instruction"
    return "object", "generic_object_query"


def load_service_config(path):
    from perception_bridge import X5GraspCalibration
    value = json.loads(Path(path).read_text())
    required = {"sam3_url", "graspnet_url", "grasp_calibration", "provenance"}
    if not isinstance(value, dict) or set(value) != required or not isinstance(value["provenance"], dict):
        raise ValueError("explicit perception config must contain URLs, calibration and provenance")
    for field in ("sam3_url", "graspnet_url"):
        endpoint = urlsplit(value[field])
        if (endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
                or endpoint.port is None or endpoint.username or endpoint.password
                or endpoint.query or endpoint.fragment or endpoint.path not in ("", "/")):
            raise ValueError("smoke services must use explicit node-local HTTP ports")
    calibration = X5GraspCalibration.from_mapping(value["grasp_calibration"])
    if (calibration.native_closing_direction != (0., 1., 0.)
            or X5_URDF_SHA256 not in calibration.source_id):
        raise ValueError("smoke calibration must identify the reviewed X5A URDF and native +Y jaw axis")
    return value, calibration


def verify_vendor_source(root):
    root = Path(root).resolve(strict=True)
    actual = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in CGN_FILES}
    if actual != CGN_FILES:
        raise ValueError("Contact-GraspNet source/config differs from reviewed frame convention")
    return {"root": str(root), "reviewed_commit": CGN_COMMIT, "files_sha256": actual,
            "convention": "raw X jaw baseline, raw Z approach; t=contact+thickness/2*X-depth*Z",
            "opening": "extra_opening affects separate opening output, not predicted contact separation"}


def activate_profile(profile):
    profile = Path(profile).resolve(strict=True)
    selected = [str(profile / "qwen_full"), str(profile / "qwen_integration"),
                str(profile / "ASPIRE/src")]
    sys.path[:] = selected + [p for p in sys.path if p not in selected]
    return profile


def run_perception_smoke(env, output, config, calibration):
    import numpy as np
    from PIL import Image
    from aspire_glue import build_api, implementation_provenance
    from cap_harness.contracts import ArmCommand, RobotAction
    from full_adapter import FullRoboDojoAdapter
    from full_episode import FullEvaluationInterrupted, finalize_native_episode, write_json
    from model_episode import plain

    checked_smoke_identity(env.task_name, env.eval_seed, env.env_seeds[0],
                           mode="interface_smoke", count=env.eval_num)
    if int(env.take_action_cnt[0]) != 0:
        raise ValueError("perception smoke must begin before the first action")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    adapter = FullRoboDojoAdapter(env)
    started = time.monotonic()
    report = {"scope": "perception_interface_smoke_only", "model_requests": 0, "ok": False,
              "task": env.task_name, "eval_seed": 0, "layout_id": 0,
              "excluded_from_fresh_evaluation": True, "predicted_grasp_executed": False,
              "task_success_required": False, "provider_provenance": config["provenance"]}
    try:
        api = build_api(adapter, sam3_url=config["sam3_url"], graspnet_url=config["graspnet_url"],
                        grasp_calibration=calibration)
        report["implementation"] = implementation_provenance()
        report["provider_health"] = {
            "sam3": bool(api._segmentation_provider.health()),
            "contact_graspnet": bool(api._grasp_providers["contact-graspnet"].health())}
        if not all(report["provider_health"].values()):
            raise RuntimeError("perception provider health check failed")
        report["camera_calibration"] = adapter.calibration_sanity()
        observation = adapter.get_observation()
        state = observation.robot_state
        camera_name = "cam_head"
        camera = observation.cameras[camera_name]
        query, query_source = instruction_query(observation.task_context.language)
        # A visible baseball in this already-exposed scene is an interface
        # probe when the task's thin scissors yield a legitimate NO_GRASP.
        # This is never an evaluated agent prompt or a task success claim.
        override = os.environ.get("ASPIRE_PERCEPTION_SMOKE_QUERY")
        if override:
            if override != "baseball":
                raise ValueError("only the documented excluded baseball interface probe is supported")
            query, query_source = override, "explicit_visible_object_interface_probe"
        report.update(instruction=observation.task_context.language, query=query,
                      query_source=query_source, camera_name=camera_name,
                      sensor_timestamp_s=observation.timestamp_s)
        for name, view in observation.cameras.items():
            Image.fromarray(view.rgb).save(output / f"{name}_initial.png")
        np.savez_compressed(output / "public_sensor_input.npz", rgb=camera.rgb,
                            depth_m=camera.depth_m, intrinsics=camera.intrinsics,
                            world_from_camera=camera.camera_pose.as_matrix())
        write_json(output / "public_initial_observation.json", {
            "instruction": observation.task_context.language,
            "joint_positions": plain(state.joint_positions),
            "end_effector_poses": {arm: {"position": pose.position.tolist(),
                "quaternion_wxyz": pose.quaternion_wxyz.tolist(), "frame": pose.frame}
                for arm, pose in state.end_effector_poses.items()},
            "controller": adapter.get_controller_metadata(), "sensor_timestamp_s": observation.timestamp_s})

        # Reuse the exact upstream stages against this single public snapshot,
        # so masks and projected depth can be reproduced from archived inputs.
        segmented = api._segment_text(camera_name, query, observation)
        report["segmentation"] = {"ok": segmented.ok, "diagnostics": plain(segmented.diagnostics),
            "error": None if segmented.error is None else str(segmented.error),
            "candidates": [{"score": s.score, "label": s.label, "frame": s.frame,
                "camera_name": s.camera_name, "pixels": int(np.count_nonzero(s.mask)),
                "box_xyxy": None if s.box_xyxy is None else s.box_xyxy.tolist()}
                for s in segmented.segmentations]}
        np.savez_compressed(output / "sam3_masks.npz",
                            **{f"mask_{i}": s.mask for i, s in enumerate(segmented.segmentations)})
        eligible = [(i, s) for i, s in enumerate(segmented.segmentations) if np.any(s.mask)]
        if not segmented.ok or not eligible:
            raise RuntimeError("SAM3 returned no nonempty predicted object mask")
        mask_index, selected = max(eligible, key=lambda pair: pair[1].score)
        report["selected_mask_index"] = mask_index
        Image.fromarray(selected.mask.astype(np.uint8) * 255).save(output / "selected_mask.png")
        overlay = np.array(camera.rgb, copy=True)
        overlay[selected.mask] = (0.5 * overlay[selected.mask] + np.array([0., 127.5, 0.])).astype(np.uint8)
        Image.fromarray(overlay).save(output / "selected_mask_overlay.png")
        segment_cloud = api._project_mask(selected, camera_name=camera_name,
                                         target_frame=state.base_frame, observation=observation)
        scene_cloud = api._project_mask(np.ones(camera.depth_m.shape, dtype=bool), camera_name=camera_name,
                                       target_frame=state.base_frame, observation=observation)
        np.savez_compressed(output / "public_point_clouds.npz", segment=segment_cloud.points,
                            scene=scene_cloud.points)
        report["point_clouds"] = {"frame": segment_cloud.frame, "segment_points": len(segment_cloud.points),
                                  "scene_points": len(scene_cloud.points)}
        grasps = api._grasp_providers["contact-graspnet"].generate_grasps(
            segment_cloud, scene_point_cloud=scene_cloud, max_candidates=5)
        report["grasp_generation"] = {"ok": grasps.ok, "diagnostics": plain(grasps.diagnostics),
            "error": None if grasps.error is None else str(grasps.error), "candidate_count": len(grasps.grasps)}
        candidates = []
        for grasp in grasps.grasps:
            transform = grasp.pose.as_matrix()
            if (grasp.frame != "world" or not np.isfinite(transform).all()
                    or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
                    or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-6)
                    or not np.isfinite(grasp.width_m) or grasp.width_m < 0
                    or grasp.metadata.get("width_is_commanded_opening") is not False
                    or grasp.metadata.get("grasp_calibration_source") != calibration.source_id):
                raise RuntimeError("CGN candidate violates finite calibrated world-frame X5 contract")
            candidates.append({"position": grasp.pose.position.tolist(),
                "quaternion_wxyz": grasp.pose.quaternion_wxyz.tolist(), "transform": transform.tolist(),
                "frame": grasp.frame, "score": grasp.score, "width_m": grasp.width_m,
                "metadata": plain(grasp.metadata)})
        write_json(output / "grasp_candidates.json", candidates)
        if not grasps.ok or not candidates:
            raise RuntimeError("Contact-GraspNet returned no valid converted X5 candidates")
        report["frame_conversion_ok"] = True
        chosen = api.select_grasp(grasps, strategy="top_down") or api.select_grasp(grasps, strategy="highest_score")
        plan = api.plan_motion(chosen.pose, arm="right")
        report["candidate_plan"] = {"ok": plan.ok, "arm": "right", "executed": False,
            "target_position": chosen.pose.position.tolist(),
            "target_quaternion_wxyz": chosen.pose.quaternion_wxyz.tolist(),
            "diagnostics": plain(plan.diagnostics), "error": None if plan.error is None else str(plan.error)}
        if plan.ok:
            trajectory = plan.trajectory
            if (trajectory is None or trajectory.embodiment != "robodojo"
                    or not np.isfinite(trajectory.joint_positions).all()
                    or not np.isclose(trajectory.dt_s, adapter.control_period_s)):
                raise RuntimeError("native candidate planner returned an invalid trajectory")
            np.savez_compressed(output / "candidate_plan.npz", joint_positions=trajectory.joint_positions,
                                dt_s=trajectory.dt_s)
            report["candidate_plan"]["waypoint_count"] = len(trajectory.joint_positions)
        # A typed unreachable candidate is not a provider/interface failure.
        # Independently verify the native planner against the measured start.
        hold_plan = adapter.native_plan_to_joints(state, state.joint_positions["right"], arm="right")
        report["native_planning_ok"] = bool(hold_plan.get("ok"))
        if not report["native_planning_ok"]:
            raise RuntimeError("native hold planning check failed")
        if int(env.take_action_cnt[0]) != 0:
            raise RuntimeError("perception or planning unexpectedly consumed a control tick")
        held = adapter.step(RobotAction({"right": ArmCommand(mode="joint_position",
            target=state.joint_positions["right"], embodiment="robodojo")}))
        if not held.ok or int(env.take_action_cnt[0]) != 1:
            raise RuntimeError("native hold did not advance exactly one tick")
        report.update(ok=True, control_ticks=1, official=finalize_native_episode(env, adapter))
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise FullEvaluationInterrupted(f"perception interface smoke failed: {type(exc).__name__}: {exc}") from exc
    finally:
        report["wall_seconds"] = time.monotonic() - started
        write_json(output / "smoke_report.json", report)
        adapter.close()


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    checked_smoke_identity(args.task_name, args.seed, os.environ.get("EVAL_LAYOUT_IDS"),
                           mode=os.environ.get("ASPIRE_FULL_MODE"), count=os.environ.get("EVAL_NUM"))
    profile = activate_profile(os.environ.get("ASPIRE_PERCEPTION_PROFILE", Path(__file__).resolve().parents[1]))
    config_path = Path(os.environ["ASPIRE_PERCEPTION_CONFIG"]).resolve(strict=True)
    config, calibration = load_service_config(config_path)
    vendor = verify_vendor_source(os.environ.get("ASPIRE_PERCEPTION_VENDOR_ROOT",
                                  profile / "ASPIRE/third_party/contact_graspnet_pytorch"))
    import full_entry
    import full_episode
    import full_adapter
    import aspire_glue
    import perception_bridge
    import cap_harness.contracts
    modules = (full_entry, full_episode, full_adapter, aspire_glue, perception_bridge, cap_harness.contracts)
    sources = {}
    for module in modules:
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(profile):
            raise ValueError("smoke imported code outside its selected source profile")
        sources[module.__name__] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    output = Path(os.environ["ASPIRE_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    full_episode.write_json(output / "perception_smoke_contract.json", {
        "scope": "excluded already-exposed interface smoke; no Qwen or grasp execution",
        "task": "general_pickup", "eval_seed": 0, "layout_id": 0,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "service_config": config, "source_profile": str(profile), "imported_sources": sources,
        "verified_cgn_vendor": vendor})
    full_entry._interface_smoke = lambda env, target: run_perception_smoke(env, target, config, calibration)
    full_entry.main()


if __name__ == "__main__":
    main()
