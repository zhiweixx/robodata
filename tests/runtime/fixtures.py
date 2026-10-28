"""Synthetic public robot and camera fixtures; no simulator or scorer import."""
from types import SimpleNamespace as NS
import numpy as np

class Camera:
    def get_intrinsics_matrix(self):
        return np.array([[100.,0,16],[0,80.,12],[0,0,1]])
    def get_world_pose(self, camera_axes):
        assert camera_axes == "ros"
        return np.zeros(3), np.array([1.,0,0,0])
    def get_image_coords_from_world_points(self, points):
        uvw = points @ self.get_intrinsics_matrix().T
        return uvw[:,:2] / uvw[:,2:]


class Planner:
    dt = .004
    def plan_path(self, current, target, real_robot_pose):
        desired = current.copy(); desired[0] = target[2] - .9
        return self.plan_joint(current, desired)
    def plan_joint(self, current, target):
        return {"status":"Success", "position":np.linspace(current, target, 11)}


class Env:
    num_envs=1
    task_name="general_pickup"
    step_lim=200
    env_seeds=[0]
    def __init__(self):
        self.take_action_cnt=[0];self.success=[True];self.end_flag=[False]
        self.obs_manager=NS(collect_freq=25.)
        robots=[];entities=[]
        for arm in ['left','right']:
            robots.append(NS(robot_name=arm,arm_joints_name=[f'joint{i}' for i in range(6)],
                arm_joint_indices=list(range(6)),gripper_scale=[0.,.04],gripper_move={'sign':1},
                gripper_joint_indices=[6,7],entity_origin_pose=[0.,0,0,1,0,0,0],ee_link_name='link6'))
            entities.append(NS(data=NS(joint_pos=np.zeros((1,6)),joint_vel=np.zeros((1,6))),
                root_physx_view=NS(get_dof_limits=lambda:np.array([[[-3.,3.]]*6+[[0.,.04]]*2]))))
        self.physical_gripper={'left':1.,'right':1.}
        self.robot_manager=NS(robot_list=robots,robot_key=entities,planner={a:Planner() for a in ['left','right']})
        manager=self.robot_manager
        manager.get_robot_by_arm_name=lambda name: robots[0 if name=='left_arm' else 1]
        manager.get_joint=lambda robot,ids:{0:entities[robots.index(robot)].data.joint_pos[0].copy()}
        manager.get_real_endpose=lambda robot,ids,is_relative=False:{0:np.array([.3,-.2,.9+manager.get_joint(robot,ids)[0][0],1.,0,0,0])}
        manager.get_end_effector_real_val=lambda robot,ids:{0:np.array([self.physical_gripper[robot.robot_name]*.04,0])}
        self.camera_manager=NS(camera_names=[['cam_head','cam_left_wrist','cam_right_wrist']],cameras=[[Camera(),Camera(),Camera()]])
        self.reward_manager=NS(get_reward=lambda:[0.])
    def get_obs(self):
        return {'instruction':'Pick up the mint green scissors by 10 cm.',
            'vision':{name:{'color':np.zeros((24,32,3),np.uint8),'depth':np.ones((24,32))}
                      for name in self.camera_manager.camera_names[0]}}
    def is_episode_end(self):
        if self.take_action_cnt[0]>=self.step_lim:
            self.success[0]=False;self.end_flag[0]=True
        return self.end_flag[0]
    def take_action(self, action):
        self.last_action=action
        for i,a in enumerate(['left','right']):
            self.robot_manager.robot_key[i].data.joint_pos[0]=action[f'{a}_arm_joint_state']
            self.physical_gripper[a]=float(action[f'{a}_ee_joint_state'][0])
        self.take_action_cnt[0]+=1

