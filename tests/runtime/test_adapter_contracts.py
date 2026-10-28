"""Original CPU adapter regressions, independent of the historical pilot runner."""
import unittest
import numpy as np
from cap_harness.contracts import ArmCommand, RobotAction
from robodata.environments.robodojo.robodojo_adapter import RoboDojoAdapter
from tests.runtime.fixtures import Env

class AdapterContractTests(unittest.TestCase):
    def test_calibration_non_square_intrinsics_and_tcp_roundtrip(self):
        adapter=RoboDojoAdapter(Env())
        self.assertEqual(len(adapter.calibration_sanity()),3)
        state=adapter.get_robot_state()
        np.testing.assert_allclose(state.end_effector_poses['right'].position,[.4501,-.2,.9])
        np.testing.assert_allclose(adapter._flange_array(state.end_effector_poses['right'])[:3],[.3,-.2,.9])

    def test_calibration_rejects_nonfinite_projection(self):
        env=Env()
        env.camera_manager.cameras[0][0].get_image_coords_from_world_points=lambda points:np.full((len(points),2),np.nan)
        with self.assertRaisesRegex(ValueError,'calibration mismatch'):
            RoboDojoAdapter(env).calibration_sanity()

    def test_rounded_native_base_quaternion_normalized_at_boundary(self):
        env=Env()
        for robot in env.robot_manager.robot_list:
            robot.entity_origin_pose=[0.,0.,0.,.707,0.,0.,.707]
        adapter=RoboDojoAdapter(env)
        context=adapter.get_planning_context()
        for matrix in context.base_transforms.values():
            np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-12)
        planner=env.robot_manager.planner['right']
        original=planner.plan_path
        captured=[]
        def capture(current,target,real_robot_pose):
            captured.append(real_robot_pose)
            return original(current,target,real_robot_pose)
        planner.plan_path=capture
        state=adapter.get_robot_state()
        result=adapter.native_plan_to_pose(state,state.end_effector_poses['right'],arm='right')
        self.assertTrue(result['ok'])
        np.testing.assert_allclose(captured[0][3:],[2**-.5,0.,0.,2**-.5])
        env.robot_manager.robot_list[0].entity_origin_pose=[0.]*7
        with self.assertRaisesRegex(ValueError,'not near unit length'):
            RoboDojoAdapter(env)

    def test_gripper_waits_for_aperture_and_counts_stall_ticks(self):
        class SlowGripper(Env):
            contact_floor=0.
            def take_action(self,action):
                before=dict(self.physical_gripper)
                super().take_action(action)
                for arm in self.physical_gripper:
                    target=float(action[f'{arm}_ee_joint_state'][0])
                    actual=before[arm]+np.clip(target-before[arm],-.2,.2)
                    self.physical_gripper[arm]=float(max(self.contact_floor,actual))
        env=SlowGripper();adapter=RoboDojoAdapter(env)
        close=adapter.set_gripper(0.,arm='right')
        self.assertTrue(close.ok)
        self.assertEqual(close.steps_executed,5)
        adapter.set_gripper(1.,arm='right')
        env.contact_floor=.35
        start=env.take_action_cnt[0]
        stall=adapter.close_gripper(arm='right')
        self.assertTrue(stall.ok)
        self.assertEqual(stall.diagnostics['completion'],'stable_aperture_stall')
        self.assertEqual(stall.steps_executed,8)
        self.assertEqual(env.take_action_cnt[0]-start,8)
        self.assertAlmostEqual(env.physical_gripper['right'],.35)
        self.assertEqual(adapter._gripper_targets['right'],0.)
        before=env.take_action_cnt[0]
        atomic=adapter.set_grippers({'right':1.,'left':0.})
        self.assertEqual(atomic.steps_executed,1)
        self.assertEqual(env.take_action_cnt[0],before+1)

    def test_x5_preload_range_is_not_physical_aperture_range(self):
        class X5Env(Env):
            def __init__(self):
                super().__init__()
                for robot,entity in zip(self.robot_manager.robot_list,self.robot_manager.robot_key):
                    robot.gripper_scale=[-.01,.044]
                    entity.root_physx_view.get_dof_limits=lambda:np.array([[[-3.,3.]]*6+[[0.,.044]]*2])
                self.q={'left':.044,'right':.044}
                self.robot_manager.get_end_effector_real_val=lambda robot,ids:{0:np.array([self.q[robot.robot_name]]*2)}
            def take_action(self,action):
                super().take_action(action)
                for arm in ['left','right']:
                    native=float(action[f'{arm}_ee_joint_state'][0])
                    self.q[arm]=max(0.,min(.044,-.01+.054*native))
        env=X5Env();adapter=RoboDojoAdapter(env)
        closed=adapter.set_gripper(0.,arm='right')
        self.assertTrue(closed.ok)
        self.assertEqual(closed.steps_executed,1)
        self.assertEqual(env.last_action['right_ee_joint_state'][0],0.)
        self.assertEqual(adapter.get_robot_state().gripper_positions['right'],0.)
        halfway=adapter.set_gripper(.5,arm='right')
        self.assertTrue(halfway.ok)
        self.assertAlmostEqual(env.last_action['right_ee_joint_state'][0],.032/.054)
        self.assertAlmostEqual(env.q['right'],.022)
        self.assertAlmostEqual(adapter.get_robot_state().gripper_positions['right'],.5)

    def test_hold_preserves_closed_command_despite_contact_aperture(self):
        env=Env();adapter=RoboDojoAdapter(env)
        adapter.set_gripper(0.,arm='right')
        env.physical_gripper['right']=.3
        state=adapter.get_robot_state()
        self.assertAlmostEqual(state.gripper_positions['right'],.3)
        adapter.step(RobotAction({'left':ArmCommand(mode='joint_position',target=np.zeros(6),embodiment='robodojo')}))
        self.assertEqual(env.last_action['right_ee_joint_state'][0],0.)
        self.assertEqual(env.take_action_cnt[0],2)
