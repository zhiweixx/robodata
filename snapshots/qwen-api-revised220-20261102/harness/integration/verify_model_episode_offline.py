"""Exercise real ASPIRE execution with fake sensing and a fake paid-model relay.

No network, Isaac, GPU or model calls; this validates runner wiring only.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from aspire_glue import build_api
from model_driver import observation_content, _bounded_json
from model_episode import execute_bounded_program, run_model_episode
from robodojo_adapter import RoboDojoAdapter
from verify_adapter_offline import Env


class Client:
    def __init__(self,*args,**kwargs):
        self.calls=[]
    def next_program(self,**request):
        observation_content(request['public_observation'])
        _bounded_json(request['previous_result'])
        self.calls.append(request)
        if len(self.calls)==1:
            assert request['previous_result'] is None
            code='''state = get_robot_state()
hold = step(RobotAction({"right": ArmCommand(mode="joint_position", target=state.joint_positions["right"], embodiment="robodojo")}))
result = {"hold_executed": hold.ok, "time": get_robot_state().timestamp_s}
'''
            return NS(code=code,done=False,summary='One hold',iteration=1,usage={},budget={})
        assert request['previous_result']['program_ok']
        assert request['public_observation']['remaining_actions']==199
        assert request['api_documentation']==self.calls[0]['api_documentation']
        return NS(code='',done=True,summary='Finish dry run',iteration=2,usage={},budget={})


class Tests(unittest.TestCase):
    def test_one_attempt_continues_and_scoring_stays_private(self):
        env=Env();client=Client()
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,{
            'ASPIRE_OUTPUT_DIR':temp,'ASPIRE_EPISODE_ID':'general_pickup_s0_l0',
            'ASPIRE_RELAY_DIR':temp+'/relay','ASPIRE_MAX_ITERATIONS':'8'},clear=True):
            report=run_model_episode(env,client_factory=lambda *a,**k:client)
            saved=json.loads((Path(temp)/'episode_report.json').read_text())
            self.assertTrue(report['ok'])
            self.assertEqual(report['termination_reason'],'model_done')
            self.assertEqual(report['total_programs_executed'],1)
            self.assertEqual(len(client.calls),2)
            self.assertEqual(env.take_action_cnt,[1])
            self.assertEqual(saved['official']['environment_steps'],1)
            self.assertFalse(saved['official']['official_success'])
            self.assertFalse(env.success[0])
            self.assertTrue(env.end_flag[0])

    def test_pure_python_loop_has_bounded_failure_receipt(self):
        adapter=RoboDojoAdapter(Env())
        api=build_api(adapter)
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'loop.py'
            source.write_text('while True:\n    pass\n')
            outcome=execute_bounded_program(api,source,None,timeout_s=.02)
            self.assertFalse(outcome.ok)
            self.assertIn('TimeoutError',outcome.error.message)
            self.assertEqual(adapter.native_env.take_action_cnt,[0])


if __name__=='__main__':
    unittest.main()
