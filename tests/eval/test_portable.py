"""CPU-only protocol/resume/ownership tests: no model or simulator imported."""
import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from robodata.data.provenance import canonical_sha256,freeze_source
from robodata.eval.robodojo import manifest as m
from robodata.eval.robodojo.runner import (resume_status,native_result_path,resolve_gpus,
    verify_campaign,verify_artifacts,container_command,server_command,validate_agent,run_campaign_group)
from robodata.eval.robodojo.audit import native_details,video_evidence,audit_campaign
from robodata.eval.robodojo.entry import checked_count
ROOT=Path(__file__).resolve().parents[2]
MANIFEST=ROOT/'experiments/manifests/robodojo_subset767.json'

class PortableContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=m.load_manifest(MANIFEST)
    def test_native_semantic_scope(self):
        self.assertEqual(len(self.m['groups']),51)
        self.assertEqual(sum(g['target_episodes'] for g in self.m['groups']),767)
        self.assertEqual(sum(g['candidate_count'] for g in self.m['groups']),2425)
        self.assertEqual(len({g['family'] for g in self.m['groups']}),40)
        self.assertEqual(self.m['migration']['original_manifest_sha256'],m.ORIGINAL_MANIFEST_SHA256)
        self.assertTrue(all(not Path(r['source_path']).is_absolute() for g in self.m['groups'] for r in g['layouts']))
    def test_manifest_semantics_cannot_be_rehashed_into_another_budget(self):
        value=copy.deepcopy(self.m);value['groups'][0]['step_limit']+=1
        value['groups'][0]['source_group_sha256']=m.canonical_sha256(m._native_group(value['groups'][0]))
        with self.assertRaises(ValueError):m.validate_manifest(value)
    def test_duplicate_and_source_escape_rejected(self):
        for field,value in [('candidate_layout_ids',[0,0]),('layouts',[{'source_path':'../../escape'}])]:
            modified=copy.deepcopy(self.m);modified['groups'][0][field]=value
            with self.assertRaises(ValueError):m.validate_manifest(modified)
    def test_whole_group_ownership(self):
        for count in (1,4,16,51):
            assignments=m.plan_shards(self.m,count)
            owned=[g for a in assignments for g in a['group_ids']]
            self.assertEqual(len(owned),51);self.assertEqual(len(set(owned)),51)
            self.assertEqual(set(owned),{g['group_id'] for g in self.m['groups']})
        with self.assertRaises(ValueError):m.plan_shards(self.m,52)
    def test_count_override_requires_official_original(self):
        group=self.m['groups'][0]
        self.assertEqual(checked_count(group['native_target_episodes'],mode='evaluation',override=str(group['target_episodes']),group=group),group['target_episodes'])
        with self.assertRaises(ValueError):checked_count(3,mode='evaluation',override='20',group=group)
    def test_native_failed_scores_never_retried(self):
        group={'task':'fixture','group_id':'fixture_s0','eval_seed':0,'target_episodes':2,'candidate_layout_ids':[0,1,2]}
        with tempfile.TemporaryDirectory() as directory:
            campaign=Path(directory);native=native_result_path(campaign,group);native.parent.mkdir(parents=True)
            value={'eval_time':1,'details':{'0':{'layout_id':0,'success':False,'score':0}}};native.write_text(json.dumps(value))
            checkpoint=native.parent.parent/('_resume_'+native.parent.name+'.json')
            with self.assertRaisesRegex(RuntimeError,'no native checkpoint'):resume_status(campaign,group)
            checkpoint.write_text(json.dumps({'completed_layout_ids':[0],'abandoned_layout_ids':[]}))
            self.assertEqual(resume_status(campaign,group),'pending')
            checkpoint.write_text(json.dumps({'completed_layout_ids':[],'abandoned_layout_ids':[]}))
            with self.assertRaisesRegex(RuntimeError,'mismatch'):resume_status(campaign,group)
            checkpoint.write_text(json.dumps({'completed_layout_ids':[0],'abandoned_layout_ids':[1,2]}))
            with self.assertRaisesRegex(RuntimeError,'exhausted'):resume_status(campaign,group)
            value['eval_time']=2;value['details']['1']={'layout_id':1,'success':False,'score':0};native.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError,'never replay'):resume_status(campaign,group)
            for i in (0,1):
                for camera in ('head','left_wrist','right_wrist'):(native.parent/f'episode_{i:07d}_cam_{camera}_fail.mp4').write_bytes(b'evidence')
            self.assertEqual(resume_status(campaign,group),'complete')
    def test_repeated_native_layout_rejected(self):
        group={'target_episodes':2,'candidate_layout_ids':[0,1]}
        row={'layout_id':0,'success':False,'score':0}
        with self.assertRaises(ValueError):native_details({'eval_time':2,'details':{'0':row,'1':row}},group)
    def test_model_command_has_no_output_or_turn_cap(self):
        command=server_command('/env/python','/weights',1234)
        self.assertEqual(command[command.index('--max-model-len')+1],'262144')
        self.assertEqual(command[command.index('--tensor-parallel-size')+1],'2')
        self.assertNotIn('--max-tokens',command)
    def test_physical_gpu_mapping_and_no_hopper_renderer(self):
        config={'resources':{'model_gpu_indices':[1,2],'simulator_gpu_index':0,'perception_gpu_index':3}}
        rows='\n'.join(f'{i}, GPU-u{i}, NVIDIA L40S, 580, 48000' for i in range(4))
        selected=resolve_gpus(config,{'CUDA_VISIBLE_DEVICES':'2,0,3,1'},output=rows)
        self.assertEqual(selected[0]['uuid'],'GPU-u2')
        with self.assertRaisesRegex(RuntimeError,'unsupported'):resolve_gpus(config,{'CUDA_VISIBLE_DEVICES':'0,1,2,3'},output=rows.replace('NVIDIA L40S','NVIDIA H200'))
    def test_changed_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'w';p.write_bytes(b'wrong')
            with self.assertRaises(ValueError):verify_artifacts([{'path':str(p),'bytes':5,'sha256':'a'*64}])
            with self.assertRaises(ValueError):verify_artifacts([])

class PublishedPlanTests(unittest.TestCase):
    def fixture(self,d):
        campaign=Path(d)/'campaign';campaign.mkdir();source=campaign/'source';source.mkdir();(source/'src/robodata').mkdir(parents=True);(source/'src/robodata/__init__.py').write_text('');(source/'src/sentinel.py').write_text('x=1')
        manifest=m.load_manifest(MANIFEST);config={'paths':{}}
        inventory=freeze_source(source)
        plan={'schema':1,'groups':[g['group_id'] for g in manifest['groups']],'assignments':m.plan_shards(manifest),
            'manifest_sha256':canonical_sha256(manifest),'cluster_config_sha256':canonical_sha256(config),'source_tree_sha256':inventory['tree_sha256']}
        for name,value in [('manifest',manifest),('cluster_config',config),('source_inventory',inventory),('plan',plan)]: (campaign/(name+'.json')).write_text(json.dumps(value))
        return campaign,plan
    def test_status_pure_cpu_and_unstarted_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            campaign,_=self.fixture(d);report=audit_campaign(campaign)
            self.assertEqual(report['scored'],0);self.assertEqual(report['target'],767);self.assertFalse(report['complete'])
    def test_plan_fails_closed_after_config_or_topology_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            campaign,plan=self.fixture(d);verify_campaign(campaign)
            (campaign/'cluster_config.json').write_text('{"paths":{"model":"other"}}')
            with self.assertRaisesRegex(ValueError,'digest'):verify_campaign(campaign)
            (campaign/'cluster_config.json').write_text('{"paths":{}}')
            plan['assignments'][1]['group_ids']=plan['assignments'][0]['group_ids'];(campaign/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'ownership'):verify_campaign(campaign)
    def test_launcher_reexecutes_frozen_source_without_local_gpu_work(self):
        with tempfile.TemporaryDirectory() as d:
            campaign,plan=self.fixture(d)
            with patch('robodata.eval.robodojo.runner.validate_agent'),patch('robodata.eval.robodojo.runner.subprocess.call',return_value=0) as invoke:
                self.assertEqual(run_campaign_group(campaign,plan['groups'][0]),0)
            args=invoke.call_args
            self.assertIn('--frozen',args.args[0]);self.assertIn('robodata.eval.robodojo.runner',args.args[0])
            self.assertEqual(args.kwargs['cwd'],(campaign/'source').resolve())
            self.assertEqual(args.kwargs['env']['PYTHONPATH'],f'{campaign.resolve()}/source/src:{campaign.resolve()}/source/third_party/aspire/src')
    def test_unpublished_plan_and_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            campaign,_=self.fixture(d);(campaign/'source/src/sentinel.py').write_text('x=2')
            with self.assertRaises(ValueError):verify_campaign(campaign)
            (campaign/'plan.json').unlink()
            with self.assertRaises(FileNotFoundError):verify_campaign(campaign)

if __name__=='__main__':unittest.main()
