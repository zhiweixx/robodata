"""CPU-only checks for actual provider construction and durable attempt evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from perception_entry import checked_count, install_runtime_hooks, read_provider_config, verify_provider_config
from perception_manifest import GRASP_CALIBRATION


class Interrupted(RuntimeError):
    pass


class Provider:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = 0

    def health(self):
        self.calls += 1
        return self.ok


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.group = {'task': 'stack_blocks', 'step_limit': 550, 'eval_seed': 0,
                      'candidate_layout_ids': [21], 'target_episodes': 1,
                      'native_target_episodes': 50}
        self.env = SimpleNamespace(task_name='stack_blocks', step_lim=550, eval_num=1,
                                   eval_seed=0, env_seeds=[21])
        self.config = {'sam3_url': 'http://127.0.0.1:6001', 'graspnet_url': 'http://127.0.0.1:6002',
                       'grasp_calibration': copy.deepcopy(GRASP_CALIBRATION),
                       'provenance': {'runtime_manifest': {'sources': []},
                           'runtime_manifest_sha256': 'b' * 64, 'service_runner_sha256': 'c' * 64,
                           'offline_skill_discovery': False}}

    def tearDown(self):
        self.temp.cleanup()

    def modules(self, arm, *, health=True, interrupt=False, extra_provider=False):
        calls = []
        api = SimpleNamespace(_segmentation_provider=Provider(health) if arm == 'perception' or extra_provider else None,
            _grasp_providers={'contact-graspnet': Provider()} if arm == 'perception' else {})
        def build(adapter, **kwargs):
            calls.append(kwargs)
            return api
        def write(path, value):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(value))
        full_episode = SimpleNamespace(build_api=build, write_json=write)
        def original(env, **kwargs):
            report = {'artifact_dir': str(self.root), 'episode_id': 'stack_blocks_s0_l21'}
            try:
                full_episode.build_api(SimpleNamespace(_progress_path=self.root / 'execution_state.json'), recorder=None)
                self.assertTrue((self.root / 'perception_runtime.json').exists())
                if interrupt:
                    raise Interrupted('inference interrupted')
                return report
            finally:
                write(self.root / 'episode_report.json', report)
        full_entry = SimpleNamespace(run_full_episode=original, FullEvaluationInterrupted=Interrupted)
        factory = lambda value: SimpleNamespace(as_dict=lambda: copy.deepcopy(value))
        install_runtime_hooks(full_entry, full_episode, arm=arm, group=self.group,
            config=self.config, config_sha256='d' * 64, calibration_factory=factory)
        return full_entry, api, calls

    def test_baseline_does_not_receive_provider_urls_or_calibration(self):
        entry, api, calls = self.modules('baseline')
        report = entry.run_full_episode(self.env)
        self.assertEqual(calls, [{'recorder': None}])
        runtime = report['perception_runtime']
        self.assertEqual(runtime['providers'], {'sam3': False, 'contact_graspnet': False})
        self.assertEqual(runtime['provider_health'], {})
        self.assertTrue(runtime['initialization_complete'])
        self.assertEqual(json.loads((self.root / 'episode_report.json').read_text()), report)

    def test_treatment_uses_actual_providers_health_and_explicit_calibration(self):
        entry, api, calls = self.modules('perception')
        report = entry.run_full_episode(self.env)
        self.assertEqual(calls[0]['sam3_url'], self.config['sam3_url'])
        self.assertEqual(calls[0]['graspnet_url'], self.config['graspnet_url'])
        self.assertEqual(calls[0]['grasp_calibration'].as_dict(), GRASP_CALIBRATION)
        self.assertEqual(api._segmentation_provider.calls, 1)
        self.assertEqual(api._grasp_providers['contact-graspnet'].calls, 1)
        runtime = report['perception_runtime']
        self.assertEqual(runtime['providers'], {'sam3': True, 'contact_graspnet': True})
        self.assertEqual(runtime['grasp_calibration'], GRASP_CALIBRATION)
        self.assertFalse(runtime['offline_skill_discovery'])

    def test_provider_health_failure_is_infrastructure_interrupt_with_evidence(self):
        entry, api, calls = self.modules('perception', health=False)
        with self.assertRaisesRegex(Interrupted, 'initialization failed'):
            entry.run_full_episode(self.env)
        runtime = json.loads((self.root / 'perception_runtime.json').read_text())
        self.assertFalse(runtime['initialization_complete'])
        self.assertEqual(runtime['initialization_error_type'], 'ValueError')
        self.assertEqual(json.loads((self.root / 'episode_report.json').read_text())['perception_runtime'], runtime)

    def test_later_interrupt_retains_runtime_in_attempt_report(self):
        entry, api, calls = self.modules('perception', interrupt=True)
        with self.assertRaisesRegex(Interrupted, 'inference interrupted'):
            entry.run_full_episode(self.env)
        runtime = json.loads((self.root / 'episode_report.json').read_text())['perception_runtime']
        self.assertTrue(runtime['initialization_complete'])

    def test_actual_unexpected_baseline_provider_is_rejected(self):
        entry, _, _ = self.modules('baseline', extra_provider=True)
        with self.assertRaisesRegex(Interrupted, 'initialization failed'):
            entry.run_full_episode(self.env)

    def test_changed_state_budget_or_native_count_rejected(self):
        entry, _, calls = self.modules('baseline')
        for field, value in [('step_lim', 551), ('env_seeds', [22]), ('eval_num', 2)]:
            env = copy.deepcopy(self.env)
            setattr(env, field, value)
            with self.assertRaisesRegex(Interrupted, 'budget mismatch'):
                entry.run_full_episode(env)
        self.assertEqual(calls, [])
        self.assertEqual(checked_count(50, mode='evaluation', override='1', group=self.group), 1)
        with self.assertRaises(ValueError):
            checked_count(49, mode='evaluation', override='1', group=self.group)

    def test_config_must_be_local_and_frozen(self):
        path = self.root / 'providers.json'
        path.write_text(json.dumps(self.config))
        self.assertEqual(read_provider_config(path), self.config)
        for value in ['https://example.com', 'http://user:password@127.0.0.1:1', 'http://127.0.0.1:1/?key=x']:
            altered = copy.deepcopy(self.config)
            altered['sam3_url'] = value
            path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                read_provider_config(path)
        manifest = {'perception_artifacts': self.config['provenance'], 'grasp_calibration': GRASP_CALIBRATION}
        verify_provider_config(self.config, manifest)
        altered = copy.deepcopy(self.config)
        altered['grasp_calibration']['native_closing_direction'] = [0, 0, 1]
        with self.assertRaises(ValueError):
            verify_provider_config(altered, manifest)


if __name__ == '__main__':
    unittest.main()
