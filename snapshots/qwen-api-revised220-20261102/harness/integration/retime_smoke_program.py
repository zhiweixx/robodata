# Free interface regression using actual failed pilot targets, not a model run.
import math
state = get_robot_state()
targets = [
    Pose([0.389,-0.025,0.865],[0.5,-0.5,0.5,0.5],"world"),
    Pose([0.389,-0.025,0.850],[0.5,-0.5,0.5,0.5],"world"),
    Pose([0.30042562873929507,-0.17219989027368438,0.9215000307281415],
         [0.7070000591108495,-4.327418927037452e-08,-6.00060701275679e-08,0.7072134871573433],"world"),
]
plans=[]
plan_receipts=[]
for target in targets:
    planned=plan_motion(target,arm="right")
    plan_receipts.append({"ok":planned.ok,"error":str(planned.error),"diagnostics":dict(planned.diagnostics)})
    plans.append(planned)
if not all(p.ok for p in plans):
    raise ValueError("an actual failed pilot target still does not plan: "+str(plan_receipts))
trajectory=plans[0].trajectory
if len(trajectory.joint_positions)>190:
    raise ValueError("overhead regression path does not fit original200step horizon: "+str(len(trajectory.joint_positions)))
motion=execute_trajectory(trajectory)
after=get_robot_state()
actual=after.end_effector_poses["right"]
position_error=math.sqrt(sum((float(actual.position[i])-float(targets[0].position[i]))**2 for i in range(3)))
dot=abs(sum(float(actual.quaternion_wxyz[i])*float(targets[0].quaternion_wxyz[i]) for i in range(4)))
orientation_error=2*math.acos(min(1.,dot))
if not motion.ok or position_error>.005 or orientation_error>.03:
    raise ValueError("overhead execution validation failed: "+str({"error":str(motion.error),"position_error_m":position_error,"orientation_error_rad":orientation_error}))
result={"scope":"free retiming regression; no model call, no attempted task solve",
    "planned_original_failed_targets":len(plans),"plans":plan_receipts,
    "executed_target":targets[0].position.tolist(),"executed_steps":motion.steps_executed,
    "actual_position":actual.position.tolist(),"position_error_m":position_error,
    "orientation_error_rad":orientation_error}
