"""Public robot/sensor contract supplement for the deployed upstream ASPIRE API."""
from robodata.execution.aspire_glue import render_api_docs

PIXEL_TO_WORLD_REFERENCE = '''def pixel_to_world(camera, u, v):
    height, width = camera.depth_m.shape
    u, v = int(u), int(v)
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError("pixel outside image")
    z = float(camera.depth_m[v, u])
    if z <= 0:
        raise ValueError("invalid metric depth at selected pixel")
    K = camera.intrinsics
    du, dv = u - float(K[0, 2]), v - float(K[1, 2])
    a, b, c, d = float(K[0, 0]), float(K[0, 1]), float(K[1, 0]), float(K[1, 1])
    determinant = a*d - b*c
    point_camera = [z*(d*du-b*dv)/determinant, z*(-c*du+a*dv)/determinant, z]
    T = camera.camera_pose.as_matrix()
    return [sum(float(T[i, j])*point_camera[j] for j in range(3)) + float(T[i, 3]) for i in range(3)]
'''


def deployed_api_documentation(api):
    """Document actual constructors, sensor fields, and this robot's tool frame."""
    docs = render_api_docs(api) + '''

DEPLOYMENT CONTRACT
- All functions above are injected directly: call get_robot_state(), not api.get_robot_state().
- Arms are "left" and "right"; "primary" resolves to "right", "secondary" to "left".
- Pose(position, quaternion_wxyz, frame) is an injected constructor. Position is three values
  in metres; quaternion_wxyz is a normalized four-element [w,x,y,z] list; frame is "world".
- Pose.position, Pose.quaternion_wxyz, Pose.frame and Pose.as_matrix() are public.
  A Pose is an object, not a dictionary: use pose.position, not pose["position"].
- RobotState has joint_positions, joint_velocities, joint_names, end_effector_poses and
  gripper_positions mappings keyed by arm. Each arm has six joints. Gripper state is actual
  measured finger position, normalized 0 closed / 1 open. This is not a width in metres.
  Controller metadata's gripper_physical_joint_limits_m are per-joint travel limits in
  metres; a single finger's travel limit is not the total space between the fingers.
  Last commanded targets are separately
  available from robodojo.get_controller_metadata()["last_gripper_commands"].
- Every end-effector position is the grasp-center, 0.1501m along flange +X.
  The quaternion is the native X5 flange orientation; its +X axis points along approach.
  Do not add the 0.1501m offset again. Controller metadata supplies a robot-specific
  straight_down_quaternion_wxyz. This is robot geometry, not an object pose oracle.
- move_to_pose(Pose(xyz, quaternion_wxyz, "world"), arm="right", max_steps=120) uses
  the configured native RoboDojo planner. Omit strategy to keep the configured backend.
  Constructing an unconfigured default MotionStrategy() selects unavailable PyRoki.
- There is no SAM3, Contact-GraspNet or standalone IK service. Do not repeatedly request
  segment_text, segment_points, localize_object, generate_grasps, or solve_ik. Jointly
  planned synchronized motion is also unavailable. Sequential arm motions are available.
- get_observation().cameras maps names (e.g. "cam_head") to CameraObservation objects.
  Each camera exposes rgb[height,width,3], depth_m[height,width], intrinsics[3,3],
  camera_pose (world-from-optical-camera Pose), frame and timestamp_s.
  Arrays are read-only, but indexing, arithmetic and public methods such as tolist() work.
  No NumPy import is needed or allowed. Pixel (u,v) means column u, row v.
  Capture one observation and reuse its camera arrays for multiple pixel measurements.
  get_observation() acquires new rendered images; do not call it inside a pixel loop.
  After motion, capture a fresh observation before measuring the changed scene.
- Programs have the documented constructors and these builtins: abs, all, any, bool,
  dict, enumerate, float, int, len, list, max, min, print, range, reversed, round,
  set, sorted, str, sum, tuple, zip, Exception, ValueError, TypeError. Only math may
  be imported. hasattr, getattr, isinstance, dir and NumPy imports are not available.
- Depth is optical z in metres, not distance along a ray. Invalid depth is zero.
  Depth measures the first visible surface: it is not automatically an object's center,
  underside, grasp pose, or free clearance. Choose pixels from supplied RGB, inspect
  nearby valid samples as needed, and use camera calibration to obtain world positions.
- If using mask_to_point_cloud with a constructed mask, explicitly pass target_frame="world";
  its upstream default "robot_base" is not this deployment's frame.

PUBLIC PIXEL-TO-WORLD REFERENCE (pure sensor geometry; no motion or hidden object state):

''' + PIXEL_TO_WORLD_REFERENCE + '''

Choose u,v yourself from a visible target in a supplied RGB image. Capture once with
observation = get_observation(), then camera = observation.cameras["cam_head"].
For each chosen pixel, call pixel_to_world(camera, u, v) using that same camera object.
This function performs geometry only; it does not acquire another image or move the robot.
Report measurements in result before motion if uncertain; there is no default target pixel.

EXECUTION AND FEEDBACK
- Each program continues the same episode; no reset occurs between programs.
- All actions consume the shared remaining 25Hz control ticks. A 300s Python watchdog
  guards each program; the host enforces a 1200s process deadline including startup
  and model calls. max_steps on move_to_pose limits the planned trajectory length.
- Gripper command 0 closes / 1 opens. Keep measured finger position distinct from commanded state.
  A stable finger position, successful command, or completed trajectory is not proof of a grasp.
  Use a short lift and renewed public images/measurements to assess object motion.
- Native planning checks the configured table and self collisions only. It does not consume
  the RGB-D point cloud as a collision world and does not check arbitrary scene obstacles.
  Direct joint moves and home are not general collision guarantees. go_home restores the
  initial arm joints; use it after manipulation when appropriate, preserving the task state.
- ExecutionResult exposes ok, steps_executed, final_errors, diagnostics, error and
  final_observation. Inspect unsuccessful results and report relevant values in result.
  error.details and diagnostics use unit-labelled measurements where available:
  position_error_m / position_tolerance_m, orientation_error_rad / orientation_tolerance_rad,
  and joint_error_rad / joint_tolerance_rad. final_errors alone has no universal unit.
- result must be JSON-compatible. Use arrays' tolist(), and float() for scalar measurements.
  Result keys should describe public measurements or intent; evaluator-like keys including
  reward, task_success, object_poses, and ground_truth are prohibited.
'''
    perception = getattr(api, "_robodojo_perception", {})
    if perception.get("sam3") or perception.get("contact_graspnet"):
        unavailable = '''- There is no SAM3, Contact-GraspNet or standalone IK service. Do not repeatedly request
  segment_text, segment_points, localize_object, generate_grasps, or solve_ik. Jointly
  planned synchronized motion is also unavailable. Sequential arm motions are available.'''
        if docs.count(unavailable) != 1:
            raise RuntimeError("perception availability documentation changed; review replacement")
        capabilities = []
        if perception.get("sam3"):
            capabilities.append('''- SAM3 is configured for segment_text, segment_points and localize_object.
  These are predictions from public camera images, not ground-truth object masks or poses.
  For localize_object and mask_to_point_cloud, pass target_frame="world" explicitly.
  SegmentationSet.segmentations contains candidates; each Segmentation has .mask and .score.
  LocalizationResult exposes .ok, .segmentation, .point_cloud, .geometry and .error.''')
        else:
            capabilities.append("- SAM3 is unavailable; segment_text, segment_points and localize_object are unavailable.")
        if perception.get("contact_graspnet"):
            capabilities.append('''- Contact-GraspNet is configured for generate_grasps(camera_name, mask).
  It uses observed RGB-D point clouds only. GraspSet.grasps contains GraspCandidate objects.
  Their .pose is already converted to this robot's world-frame grasp-center position and
  native X5 flange orientation (+X approach), directly compatible with move_to_pose.
  Do not apply a Panda rotation, Panda depth offset, or the X5 0.1501m offset again.
  select_grasp(..., strategy="top_down") uses this native +X approach axis.
  Candidate .width_m is reconstructed predicted contact separation, not a normalized
  gripper command or a calibrated required jaw opening. .metadata labels this provenance.
  Candidates are proposals, not guarantees of reachability, collision avoidance or grasp success.''')
        else:
            capabilities.append("- Contact-GraspNet is unavailable; generate_grasps is unavailable.")
        capabilities.append('''- Standalone solve_ik and jointly planned synchronized motion remain unavailable.
  Sequential arm motions use the same native RoboDojo planner and action interface.''')
        docs = docs.replace(unavailable, "\n".join(capabilities))
    return docs
