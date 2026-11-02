"""CPU-only subset count, identity, evidence, and resume-safety checks."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

from subset_entry import checked_subset_count, checked_layout_queue, load_group_contract
from subset_supervisor import SubsetCampaign, subset_experiment_config


MANIFEST_PATH = Path(__file__).with_name('subset_manifest.json')


class SubsetRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST_PATH.read_text())
        self.group = next(g for g in self.manifest['groups'] if g['target_episodes'] == 20)

    def test_native_count_override_counts_valid_episodes_not_candidate_limit(self):
        for group in self.manifest['groups']:
            self.assertEqual(checked_subset_count(group['native_target_episodes'],
                mode='evaluation', override=str(group['target_episodes']), group=group),
                group['target_episodes'])
            # The whole distinct-state pool remains available for replacement;
            # the explicit low-variety exceptions may exhaust that pool.
            self.assertGreaterEqual(len(group['candidate_layout_ids']), group['target_episodes'])
            if group['family'] not in ('cover_blocks', 'play_tic_tac_toe'):
                self.assertGreater(len(group['candidate_layout_ids']), group['target_episodes'])

    def test_unique_queue_preserves_native_ids_and_rejects_other_layouts(self):
        group = {'candidate_layout_ids': [0, 3, 8, 12]}
        self.assertEqual(checked_layout_queue('0,3,8,12', group), [0, 3, 8, 12])
        for other in (None, '', '0,1,2,3', '0,3', '0,3,8,12,13', '3,0,8,12'):
            with self.subTest(other=other), self.assertRaises(ValueError):
                checked_layout_queue(other, group)

    def test_native_seed_manager_filters_without_renumbering_and_resumes(self):
        source = Path(__file__).parents[2] / 'audit/RoboDojo/env/seed_manager/seed_manager.py'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layouts = root / 'Eval_Layout/RoboDojo/arx_x5/0'
            layouts.mkdir(parents=True)
            for suffix in (1, 3, 7, 9, 22):
                (layouts / f'example_{suffix}.json').write_text(json.dumps({'source_suffix': suffix}))
            env_module, globals_module = ModuleType('env'), ModuleType('env.global_configs')
            env_module.__path__ = []
            globals_module.ASSETS_PATH, globals_module.BENCHMARK = str(root), 'RoboDojo'
            utils_module, files_module = ModuleType('utils'), ModuleType('utils.load_file')
            utils_module.__path__ = []
            files_module.__all__ = ['load_json']
            files_module.load_json = lambda path: json.loads(Path(path).read_text())
            modules = {'env': env_module, 'env.global_configs': globals_module,
                       'utils': utils_module, 'utils.load_file': files_module}
            with patch.dict(sys.modules, modules), patch.dict(os.environ, EVAL_LAYOUT_IDS='1,4'):
                spec = importlib.util.spec_from_file_location('subset_native_seed_fixture', source)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                manager = module.SeedManager({'num_envs': 1, 'task_name': 'example',
                                              'config_name': 'arx_x5', 'seed': 0})
                manager.init_eval()
                self.assertEqual(manager.get_seeds(), [1])
                self.assertEqual(manager.get_seed_scene_info(1), {'source_suffix': 3})
                self.assertEqual(manager.get_seeds(), [4])
                self.assertEqual(manager.get_seed_scene_info(4), {'source_suffix': 22})
                self.assertIsNone(manager.get_seeds())
                manager.init_eval(completed_layout_ids=[1])
                self.assertEqual(manager.get_seeds(), [4])
                manager.init_eval(completed_layout_ids=[1], abandoned_layout_ids=[4])
                self.assertIsNone(manager.get_seeds())

    def test_unapproved_count_or_native_drift_cannot_silently_clamp(self):
        for override in (None, 'native', '0', '1', '50', '21'):
            with self.subTest(override=override), self.assertRaises(ValueError):
                checked_subset_count(self.group['native_target_episodes'],
                    mode='evaluation', override=override, group=self.group)
        with self.assertRaises(ValueError):
            checked_subset_count(self.group['native_target_episodes']+1,
                mode='evaluation', override='20', group=self.group)
        with self.assertRaises(ValueError):
            checked_subset_count(self.group['native_target_episodes'],
                mode='interface_smoke', override='20', group=self.group)

    def test_skipped_task_and_unapproved_seed_rejected(self):
        for task, seed in [('general_pickup', 0), ('stack_bowls', 0),
                           ('stack_bowls_random', 0), (self.group['task'], 1)]:
            with self.subTest(task=task, seed=seed), self.assertRaises(ValueError):
                load_group_contract(MANIFEST_PATH, task, seed)
        _, group = load_group_contract(MANIFEST_PATH, self.group['task'], 0)
        self.assertEqual(group, self.group)

    def test_old_full_campaign_namespace_rejected_before_gpu_work(self):
        with self.assertRaises(ValueError):
            SubsetCampaign(SimpleNamespace(campaign='qwen38_full_20261024'))

    def test_experiment_is_explicit_subset_without_policy_caps(self):
        result = subset_experiment_config('qwen38_subset20_test', self.manifest,
                                         {'revision': 'pinned_revision'})
        self.assertFalse(result['is_official_full_evaluation'])
        self.assertEqual((result['task_families'], result['task_configurations'],
                          result['target_scored_episodes']),
                         (40, 51, 767))
        self.assertEqual(result['family_target_exceptions'], {'cover_blocks': 6, 'play_tic_tac_toe': 1})
        self.assertEqual(result['eval_layout_seeds'], [0])
        self.assertEqual(result['native_context_tokens'], 262144)
        self.assertIsNone(result['model_turn_cap'])
        self.assertIsNone(result['independent_output_token_cap'])
        self.assertTrue(result['native_task_action_budgets'])

    def make_campaign(self, root):
        campaign = SubsetCampaign.__new__(SubsetCampaign)
        campaign.campaign = 'qwen38_subset20_test'
        campaign.simroot = root
        campaign.out = root / 'aspire_validate/full_results' / campaign.campaign
        campaign.manifest = {'groups': [copy.deepcopy(self.group)]}
        campaign.completed = {}
        report_path = campaign.out / 'groups' / self.group['group_id'] / 'group_report.json'
        report_path.parent.mkdir(parents=True)
        run_id = campaign.campaign + '_' + self.group['group_id']
        native_dir = root / 'RoboDojo/eval_result' / self.group['task'] / run_id
        native_dir.mkdir(parents=True)
        report = {'complete': False, 'scored': 1, 'target': 20,
                  'native_result_path': str(native_dir / '_result.json'),
                  'native_run_id': run_id, 'mode': 'evaluation',
                  'task': self.group['task'], 'eval_seed': 0}
        return campaign, report_path, native_dir, report

    def test_partial_score_without_native_checkpoint_cannot_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign, path, native, report = self.make_campaign(Path(temporary))
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, 'refusing to replay'):
                campaign.collect_pending_groups()
            checkpoint = native.parent / ('_resume_' + report['native_run_id'] + '.json')
            checkpoint.write_text('{}')
            pending = campaign.collect_pending_groups()
            self.assertEqual(pending.qsize(), 1)
            self.assertEqual(pending.get()['candidate_layout_ids'], self.group['candidate_layout_ids'])

    def test_completed_native_evidence_is_reused_including_scored_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign, path, native, report = self.make_campaign(Path(temporary))
            # One invalid candidate is skipped: score indices stay contiguous,
            # while native layout IDs need not be. All scored failures count.
            layouts = [0] + list(range(2, 21))
            details = {str(i): {'layout_id': layout, 'success': False, 'score': 0.0}
                       for i, layout in enumerate(layouts)}
            (native / '_result.json').write_text(json.dumps({'eval_time': 20, 'details': details}))
            for i in range(20):
                for camera in ('head', 'left_wrist', 'right_wrist'):
                    (native / f'episode_{i:07d}_cam_{camera}_fail.mp4').write_bytes(b'nonempty fixture')
            report.update(complete=True, scored=20)
            path.write_text(json.dumps(report))
            self.assertTrue(campaign.collect_pending_groups().empty())
            saved = campaign.completed[self.group['group_id']]
            self.assertEqual((saved['scored'], saved['success']), (20, 0))

    def test_complete_score_with_missing_evidence_is_not_replayed(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign, path, native, report = self.make_campaign(Path(temporary))
            report.update(scored=20, scores_complete=True, videos_validated=False)
            path.write_text(json.dumps(report))
            (native.parent / ('_resume_' + report['native_run_id'] + '.json')).write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'video evidence'):
                campaign.collect_pending_groups()

    def test_exhausted_unique_pool_cannot_be_declared_complete_below_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign, path, native, report = self.make_campaign(Path(temporary))
            group = campaign.manifest['groups'][0]
            group.update(target_episodes=6, candidate_layout_ids=[0, 1, 2, 7, 12, 14])
            # Five scored states and one native-invalid state exhaust the
            # authorized pool. A mistaken completion flag must not certify six.
            report.update(target=6, scored=5, complete=True)
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, 'incomplete native group'):
                campaign.collect_pending_groups()
            self.assertEqual(campaign.completed, {})


if __name__ == '__main__':
    unittest.main()
