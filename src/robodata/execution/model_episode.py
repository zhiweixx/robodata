"""Public observation serialization shared with evaluation; no capped pilot loop."""
import base64
import io
from collections.abc import Mapping
import numpy as np
from PIL import Image

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

