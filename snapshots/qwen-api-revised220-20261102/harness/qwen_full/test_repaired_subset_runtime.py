"""CPU-only frozen rerun/checkpoint/provider tests; never starts a GPU process."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from audit_full_results import expected_result_path
from perception_entry import install_runtime_hooks
from repaired_subset_entry import provider_configuration
from repaired_subset_supervisor import RepairedSubsetCampaign, fresh_campaign_guard, pending_group, server_command
import full_supervisor


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.simroot = Path(self.temp.name)
        self.root = self.simroot / 'aspire_validate'
        self.campaign = self.root / 'full_results/qwen38_subset20_repaired_test'
        self.campaign.mkdir(parents=True)
        self.group = {'group_id': 'task_s0', 'task': 'task', 'eval_seed': 0,
                      'target_episodes': 2, 'candidate_layout_ids': [0, 2, 3]}
        self.native = expected_result_path(self.simroot, self.campaign.name, self.group)
        self.native.parent.mkdir(parents=True)
        self.checkpoint = self.native.parent.parent / ('_resume_' + self.native.parent.name + '.json')

    def scored(self, ids):
        details = {str(i): {'layout_id': layout, 'success': False, 'score': 0.}
                   for i, layout in enumerate(ids)}
        self.native.write_text(json.dumps({'eval_time': len(ids), 'details': details}))

    def saved(self, completed=(), abandoned=()):
        self.checkpoint.write_text(json.dumps({'completed_layout_ids': list(completed),
                                               'abandoned_layout_ids': list(abandoned)}))

    def test_partial_native_scores_without_group_report_require_checkpoint(self):
        self.scored([0])
        with self.assertRaisesRegex(RuntimeError, 'without native checkpoint'):
            pending_group(self.campaign, self.group, self.simroot)
        self.saved([0], [2])
        self.assertIsNone(pending_group(self.campaign, self.group, self.simroot))
        self.assertEqual(json.loads(self.native.read_text())['details']['0']['success'], False)

    def test_untouched_group_is_pending(self):
        self.assertIsNone(pending_group(self.campaign, self.group, self.simroot))

    def test_candidate_exhaustion_is_not_restartable(self):
        self.scored([0])
        self.saved([0], [2, 3])
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            pending_group(self.campaign, self.group, self.simroot)

    def test_checkpoint_mismatch_or_unauthorized_ids_fail_closed(self):
        self.scored([0])
        for completed, abandoned in (([], []), ([0], [99]), ([0], [0]), ([0, 0], [])):
            with self.subTest(completed=completed, abandoned=abandoned):
                self.saved(completed, abandoned)
                with self.assertRaises(RuntimeError):
                    pending_group(self.campaign, self.group, self.simroot)

    def test_scores_complete_without_evidence_never_replayed(self):
        self.scored([0, 3])
        with self.assertRaisesRegex(RuntimeError, 'evidence needs repair'):
            pending_group(self.campaign, self.group, self.simroot)

    def test_completed_failed_scores_return_existing_result_with_three_cameras(self):
        self.scored([0, 3])
        for index in range(2):
            for camera in ('head', 'left_wrist', 'right_wrist'):
                (self.native.parent / f'episode_{index:07d}_cam_{camera}_fail.mp4').write_bytes(b'video')
        directory = self.campaign / 'groups/task_s0'
        directory.mkdir(parents=True)
        report = {'complete': True, 'target': 2, 'scored': 2, 'mode': 'evaluation',
            'task': 'task', 'eval_seed': 0, 'native_run_id': self.native.parent.name,
            'native_result_path': str(self.native)}
        (directory / 'group_progress.json').write_text(json.dumps(report))
        result = pending_group(self.campaign, self.group, self.simroot)
        self.assertEqual((result['scored'], result['success']), (2, 0))

    def test_fresh_campaign_rejects_native_checkpoint_or_prior_groups(self):
        manifest = {'groups': [self.group]}
        fresh_campaign_guard(self.root, self.campaign.name, manifest)
        self.saved([], [])
        with self.assertRaisesRegex(ValueError, 'native scoring/checkpoints'):
            fresh_campaign_guard(self.root, self.campaign.name, manifest)
        self.checkpoint.unlink()
        (self.campaign / 'groups/task_s0').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'existing groups'):
            fresh_campaign_guard(self.root, self.campaign.name, manifest)


class SchedulerTests(unittest.TestCase):
    def test_model_command_only_makes_existing_seed_zero_explicit(self):
        prior = full_supervisor.server_command('/python', '/snapshot', 12345, 3)
        actual = server_command('/python', '/snapshot', 12345, 3)
        self.assertEqual(actual, prior + ['--seed', '0'])

    def test_startup_records_explicit_seed_and_uses_frozen_smoke_with_inference_gpus(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = object.__new__(RepairedSubsetCampaign)
            campaign.runtime = Path(directory)
            campaign.root = Path('/root')
            campaign.code = Path('/root/harness_profiles/frozen/qwen_full')
            campaign.python, campaign.snapshot = Path('/python'), Path('/snapshot')
            campaign.port, campaign.mtp_tokens, campaign.url = 12345, 3, 'http://127.0.0.1:12345/v1'
            campaign.env, campaign.report = {}, {}
            campaign.gpus = [{'uuid': 'gpu' + str(i)} for i in range(4)]
            campaign.track = lambda name, process: process
            campaign.record, campaign.update_inference_phases = Mock(), Mock()
            campaign.wait, campaign.cleanup_child = Mock(return_value=0), Mock()
            smoke = campaign.runtime / 'model_smoke/full_model_smoke.json'
            smoke.parent.mkdir()
            smoke.write_text('{"ok":true}')
            with patch('repaired_subset_supervisor.subprocess.Popen', side_effect=[NS(pid=1), NS(pid=2)]) as launch, \
                    patch('repaired_subset_supervisor.full.wait_ready', return_value={'models': []}), \
                    patch('repaired_subset_supervisor.full.verify_inference_variant', return_value={'variant': 'bf16_mtp3'}):
                campaign.start_server()
            self.assertEqual(launch.call_args_list[0].kwargs['env']['CUDA_VISIBLE_DEVICES'], 'gpu1,gpu2')
            self.assertEqual(launch.call_args_list[0].args[0][-2:], ['--seed', '0'])
            self.assertEqual(launch.call_args_list[1].args[0][1], str(campaign.code / 'full_model_smoke.py'))
            saved = json.loads((campaign.runtime / 'allocation.json').read_text())
            self.assertEqual(saved['server_command'][-2:], ['--seed', '0'])
            self.assertEqual(saved['actual_inference_variant'], 'bf16_mtp3')

    def test_enabled_providers_use_only_simulator_zero_and_single_frozen_service_profile(self):
        campaign = object.__new__(RepairedSubsetCampaign)
        campaign.execution_contract = {'providers': {'sam3': True},
            'hardware': {'simulator_gpus': [0]}, 'source_profile': {'relative_root': 'frozen'},
            'perception_artifacts': {'test': 'frozen-artifact'}}
        campaign.root, campaign.runtime, campaign.source_profile = Path('/root'), Path('/runtime'), Path('/root/frozen')
        campaign.env, campaign.report, campaign.gpus = {}, {}, [{'uuid': str(i)} for i in range(4)]
        campaign.stop, campaign.lock = threading.Event(), threading.Lock()
        campaign.failed, campaign.completed = [], {}
        campaign.manifest = {'group_count': 3}
        campaign.track, campaign.record, campaign.save_progress, campaign.start_server = Mock(), Mock(), Mock(), Mock()
        pending = queue.Queue()
        for i in range(3):
            pending.put({'group_id': str(i)})
        campaign.collect_pending_groups = lambda: pending
        calls = []
        def run_group(group, index):
            calls.append((group['group_id'], index))
            campaign.completed[group['group_id']] = {'scored': 1}
        campaign.run_group = run_group
        with patch('repaired_subset_supervisor.perception_services.start_services', return_value=({}, [])) as services:
            campaign.run()
        self.assertEqual(calls, [('0', 0), ('1', 0), ('2', 0)])
        proxy = services.call_args.args[0]
        self.assertEqual(proxy.manifest['profiles']['perception'], campaign.execution_contract['source_profile'])
        campaign.start_server.assert_called_once()
        self.assertTrue(campaign.report['complete'])

    def test_complete_campaign_does_not_start_model_or_services(self):
        campaign = object.__new__(RepairedSubsetCampaign)
        campaign.collect_pending_groups = lambda: queue.Queue()
        campaign.save_progress, campaign.start_server = Mock(), Mock()
        campaign.report = {}
        with patch('repaired_subset_supervisor.perception_services.start_services') as services:
            campaign.run()
        campaign.start_server.assert_not_called()
        services.assert_not_called()
        self.assertTrue(campaign.report['complete'])


class ProviderEpisodeTests(unittest.TestCase):
    def test_multi_episode_group_records_each_attempt_and_updates_scope_without_changing_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(path, report):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(json.dumps(report))
            calls = []
            def build(adapter, **kwargs):
                calls.append(kwargs)
                return NS(_segmentation_provider=NS(health=lambda: True),
                    _grasp_providers={'contact-graspnet': NS(health=lambda: True)})
            episode_module = NS(write_json=write, build_api=build)
            def episode(env, **kwargs):
                path = root / str(env.env_seeds[0])
                report = {'scope': 'old no SAM3 statement', 'episode_id': str(env.env_seeds[0])}
                episode_module.build_api(NS(_progress_path=path / 'execution_state.json'))
                write(path / 'episode_report.json', report)
                return report
            entry = NS(run_full_episode=episode, FullEvaluationInterrupted=RuntimeError)
            config = {'sam3_url': 'http://127.0.0.1:1', 'graspnet_url': 'http://127.0.0.1:2',
                'grasp_calibration': {'test': 'calibration'}, 'provenance': {'offline_skill_discovery': False}}
            group = {'task': 'task', 'step_limit': 550, 'eval_seed': 0,
                     'target_episodes': 20, 'candidate_layout_ids': [0, 2]}
            install_runtime_hooks(entry, episode_module, arm='perception', group=group,
                config=config, config_sha256='c' * 64,
                calibration_factory=lambda value: NS(as_dict=lambda: value),
                services_resident_for_both_arms=False, report_scope='SAM3 and CGN enabled; no offline learning')
            for layout in [0, 2]:
                env = NS(task_name='task', step_lim=550, eval_num=20, eval_seed=0, env_seeds=[layout])
                result = entry.run_full_episode(env)
                saved = json.loads((root / str(layout) / 'episode_report.json').read_text())
                self.assertEqual(result, saved)
                self.assertEqual(result['scope'], 'SAM3 and CGN enabled; no offline learning')
                self.assertFalse(result['perception_runtime']['services_resident_for_both_arms'])
                self.assertEqual(result['perception_runtime']['provider_config_sha256'], 'c' * 64)
                self.assertEqual(env.step_lim, 550)
            self.assertEqual(len(calls), 2)
            with self.assertRaisesRegex(RuntimeError, 'budget mismatch'):
                entry.run_full_episode(NS(task_name='task', step_lim=550, eval_num=1, eval_seed=0, env_seeds=[0]))

    def test_enabled_configuration_matches_frozen_artifact_and_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'providers.json'
            provenance = {'runtime_manifest': {}, 'runtime_manifest_sha256': 'a' * 64,
                          'service_runner_sha256': 'b' * 64, 'offline_skill_discovery': False}
            config = {'sam3_url': 'http://127.0.0.1:1', 'graspnet_url': 'http://127.0.0.1:2',
                      'grasp_calibration': {'known': True}, 'provenance': provenance}
            contract = {'providers': {'sam3': True}, 'perception_artifacts': provenance,
                        'grasp_calibration': {'known': True}}
            path.write_text(json.dumps(config))
            self.assertEqual(provider_configuration(path, contract), config)
            altered = copy.deepcopy(config)
            altered['provenance']['service_runner_sha256'] = 'c' * 64
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, 'frozen'):
                provider_configuration(path, contract)


if __name__ == '__main__':
    unittest.main()
