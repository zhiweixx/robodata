"""Small CPU fixtures for new-only repaired subset evidence; no model/simulator."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_repaired_subset_results as audit
from build_subset_manifest import canonical_sha256
from cgn_singleton_fix import expected_patch_evidence
from paired_manifest import PINNED_INFERENCE, PINNED_MODEL, PINNED_SAMPLING, PINNED_THINKING, sha_file
from repaired_subset_contract import HARDWARE, PROVIDERS, GRASP_CALIBRATION


class RepairedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.name = 'qwen38_subset20_repaired_test'
        self.campaign = self.root / 'full_results' / self.name
        self.campaign.mkdir(parents=True)
        files = {name: 'a' * 64 for name in audit.ALLOCATION_SOURCES | audit.GROUP_SOURCES | {
            'qwen_full/repaired_subset_eval.sbatch', 'qwen_full/full_model_driver.py',
            'qwen_full/perception_service.py', 'qwen_full/cgn_singleton_fix.py'}}
        self.profile = {'relative_root': 'harness_profiles/' + self.name, 'git_commit': 'a' * 40,
                        'files_sha256': files, 'files_manifest_sha256': canonical_sha256(files)}
        self.profile_verification = {'profile_root': str(self.root / self.profile['relative_root']),
            'git_commit': self.profile['git_commit'], 'files_manifest_sha256': self.profile['files_manifest_sha256'],
            'files_count': len(files)}
        weights = {'sam3': {'path': '/weights/sam3.pt', 'bytes': 10, 'sha256': 'b' * 64},
                   'graspnet': {'path': '/weights/contact_graspnet/checkpoints/model.pt',
                               'bytes': 20, 'sha256': 'c' * 64}}
        self.artifacts = {'runtime_manifest': {'files': list(weights.values())},
                          'runtime_manifest_sha256': 'd' * 64, 'service_runner_sha256': files['qwen_full/perception_service.py'],
                          'estimator_patch': expected_patch_evidence(files['qwen_full/cgn_singleton_fix.py'])}
        health = {}
        for kind, checkpoint in weights.items():
            row = {'kind': kind, 'ok': True, 'loaded': True, 'model_loaded': True,
                   'offline_weights': True, 'device': 'cuda:0', 'parameter_devices': ['cuda:0'],
                   'parameter_count': 1, 'random_seed': 0, 'checkpoint_sha256': checkpoint['sha256']}
            row.update({'files': {'sam3.pt': checkpoint}} if kind == 'sam3' else {
                'checkpoint': checkpoint, 'estimator_patch': self.artifacts['estimator_patch']})
            health[kind] = row
        self.provenance = {key: self.artifacts[key] for key in
                           ('runtime_manifest', 'runtime_manifest_sha256', 'service_runner_sha256')}
        self.provenance.update(service_health=health, gpu_uuid='GPU-3', offline_skill_discovery=False)
        self.contract = {'campaign': self.name, 'protocol': 'test-repaired-protocol',
            'dataset_manifest_sha256': 'e' * 64, 'source_profile': self.profile,
            'model': PINNED_MODEL, 'inference': PINNED_INFERENCE, 'sampling': PINNED_SAMPLING,
            'thinking': PINNED_THINKING, 'providers': PROVIDERS, 'hardware': HARDWARE,
            'perception_artifacts': self.artifacts, 'grasp_calibration': GRASP_CALIBRATION}
        self.contract_path = self.campaign / 'execution_manifest.json'
        self.contract_path.write_text(json.dumps(self.contract))
        self.contract_sha = sha_file(self.contract_path)
        self.group = {'group_id': 'task_s0', 'task': 'task', 'eval_seed': 0,
                      'target_episodes': 1, 'step_limit': 200, 'candidate_layout_ids': [0, 1],
                      'episode_ids': ['task_s0_l0', 'task_s0_l1']}
        (self.campaign / 'manifest.json').write_text(json.dumps({'groups': [self.group]}))
        self.allocation = self.campaign / 'allocations' / '123'
        self.ledger = self.allocation / 'scored' / 'worker0' / 'task_s0' / 'usage.jsonl'
        self.ledger.parent.mkdir(parents=True)
        config = {'sam3_url': 'http://127.0.0.1:1', 'graspnet_url': 'http://127.0.0.1:2',
                  'provenance': self.provenance, 'grasp_calibration': GRASP_CALIBRATION}
        config_path = self.allocation / 'providers.json'
        config_path.write_text(json.dumps(config))
        self.config_sha = sha_file(config_path)
        self.report = self.make_allocation()
        (self.allocation / 'allocation.json').write_text(json.dumps(self.report))
        self.group_dir = self.campaign / 'groups' / 'task_s0'
        self.attempt = self.group_dir / 'episodes' / 'task_s0_l0' / 'attempt_001'
        self.attempt.mkdir(parents=True)
        (self.attempt / 'native_stdout_0000.log').write_bytes(b'')
        self.group_record = self.make_group_record()
        (self.group_dir / 'repaired_subset_contract.json').write_text(json.dumps(self.group_record))
        self.episode = {'episode_id': 'task_s0_l0', 'native_score_committed': True,
            'perception_runtime': {'arm': 'perception', 'providers': {'sam3': True, 'contact_graspnet': True},
                'initialization_complete': True, 'offline_skill_discovery': False,
                'services_resident_for_both_arms': False, 'provider_config_sha256': self.config_sha,
                'service_provenance': self.provenance, 'provider_health': {'sam3': True, 'contact_graspnet': True},
                'grasp_calibration': GRASP_CALIBRATION},
            'iterations': [{'done': False, 'native_stdout_log': 'native_stdout_0000.log',
                            'execution': {'tool_calls': []}}, {'done': True}]}
        (self.attempt / 'episode_report.json').write_text(json.dumps(self.episode))
        self.write_ledger()

    def tearDown(self):
        self.temp.cleanup()

    def bootstrap(self, required, container=False):
        prefix = Path('/work/aspire_validate') if container else self.root
        directory = prefix / self.profile['relative_root']
        return {'profile_root': str(directory), 'files_manifest_sha256': self.profile['files_manifest_sha256'],
                'executed_sources': {name: {'path': str(directory / name),
                    'sha256': self.profile['files_sha256'][name]} for name in required}}

    def make_allocation(self):
        return {'execution_contract_sha256': self.contract_sha,
            'manifest_sha256': self.contract['dataset_manifest_sha256'], 'providers': PROVIDERS,
            'source_profile_verification': self.profile_verification,
            'source_sha256': {Path(name).name: digest for name, digest in self.profile['files_sha256'].items()
                             if Path(name).parent == Path('qwen_full') and Path(name).suffix in {'.py', '.sh', '.sbatch'}},
            'batch_script_sha256': self.profile['files_sha256']['qwen_full/repaired_subset_eval.sbatch'],
            'execution_hardware': HARDWARE, 'bootstrap_verification': self.bootstrap(audit.ALLOCATION_SOURCES),
            'model_manifest': {**PINNED_MODEL, 'snapshot_path': '/weights/qwen'},
            'actual_inference_variant': 'bf16_mtp3', 'context_length': 262144,
            'inference_variant_evidence': {'variant': 'bf16_mtp3'},
            'server_command': ['python', '--model', '/weights/qwen', '--dtype', 'bfloat16',
                '--tensor-parallel-size', '2', '--max-model-len', '262144', '--max-num-seqs', '2',
                '--seed', '0', '--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': 3})],
            'perception_services': self.provenance, 'gpus': [{'uuid': f'GPU-{i}'} for i in range(4)]}

    def make_group_record(self):
        bootstrap = self.bootstrap(audit.GROUP_SOURCES, container=True)
        return {'campaign': self.name, 'group_id': self.group['group_id'], 'protocol': self.contract['protocol'],
            'task': self.group['task'], 'eval_seed': 0, 'execution_contract_sha256': self.contract_sha,
            'dataset_manifest_sha256': self.contract['dataset_manifest_sha256'], 'step_limit': 200,
            'target_episodes': 1, 'candidate_layout_ids': [0, 1], 'source_profile': self.profile,
            'providers': PROVIDERS, 'provider_config_sha256': self.config_sha,
            'source_profile_verification': {**self.profile_verification, 'profile_root': bootstrap['profile_root']},
            'bootstrap_verification': bootstrap}

    def write_ledger(self, requests=True):
        config = {**audit.full.UNCAPPED_CONFIG, 'sampling': PINNED_SAMPLING, 'chat_template_kwargs': PINNED_THINKING,
                  'driver_sha256': self.profile['files_sha256']['qwen_full/full_model_driver.py'],
                  'remote_api_calls': False}
        rows = [{'event': 'suite_initialized', 'config': config}]
        if requests:
            rows += [{'event': 'generation_started', 'episode_id': 'task_s0_l0', 'iteration': 0},
                     {'event': 'generation_usage', 'episode_id': 'task_s0_l0', 'iteration': 0,
                      'usage': {'prompt_tokens': 7, 'completion_tokens': 5, 'generation_wall_s': 1.}}]
        self.ledger.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')

    def base_result(self):
        return {'errors': [], 'campaign_complete': False, 'scoring': {'scored': 1, 'success': 0},
                'reused_results': {'scored': 40, 'success': 10},
                'combined_807_summary': {'scored': 41, 'success': 10}}

    def test_bootstrap_rejects_path_and_hash_drift_and_missing_sources(self):
        good = self.bootstrap(audit.GROUP_SOURCES, container=True)
        audit.validate_bootstrap_record(good, self.root, self.contract, audit.GROUP_SOURCES)
        name = next(iter(audit.GROUP_SOURCES))
        for mutation in ('path', 'hash', 'missing', 'root'):
            altered = copy.deepcopy(good)
            if mutation == 'path': altered['executed_sources'][name]['path'] = '/mutable/' + name
            elif mutation == 'hash': altered['executed_sources'][name]['sha256'] = 'f' * 64
            elif mutation == 'missing': altered['executed_sources'].pop(name)
            else: altered['profile_root'] = '/unrelated/profile'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                audit.validate_bootstrap_record(altered, self.root, self.contract, audit.GROUP_SOURCES)

    def test_allocation_rejects_wrong_model_source_hardware_and_patch(self):
        audit.validate_allocation(self.report, self.root, self.contract, self.contract_sha, self.profile_verification)
        changes = [lambda r: r['model_manifest'].update(revision='f' * 40),
                   lambda r: r.update(context_length=32768),
                   lambda r: r.update(source_sha256={}),
                   lambda r: r.update(batch_script_sha256='f' * 64),
                   lambda r: r.update(manifest_sha256='f' * 64),
                   lambda r: r.update(providers={}),
                   lambda r: r.update(execution_hardware={}),
                   lambda r: r['perception_services']['service_health']['graspnet'].pop('estimator_patch')]
        for change in changes:
            altered = copy.deepcopy(self.report)
            change(altered)
            with self.assertRaises(ValueError):
                audit.validate_allocation(altered, self.root, self.contract, self.contract_sha, self.profile_verification)

    def test_group_queue_and_action_budgets_cannot_change(self):
        audit.validate_group_contract(self.group_record, self.group, self.root, self.contract, self.contract_sha, self.name)
        for key, value in [('candidate_layout_ids', [1, 0]), ('step_limit', 201), ('eval_seed', 1)]:
            altered = copy.deepcopy(self.group_record)
            altered[key] = value
            with self.assertRaises(ValueError):
                audit.validate_group_contract(altered, self.group, self.root, self.contract, self.contract_sha, self.name)

    def test_implicit_model_seed_requires_actual_engine_evidence(self):
        report = copy.deepcopy(self.report)
        index = report['server_command'].index('--seed')
        del report['server_command'][index:index + 2]
        with self.assertRaisesRegex(ValueError, 'actual Qwen'):
            audit.validate_allocation(report, self.root, self.contract, self.contract_sha, self.profile_verification)
        report['inference_variant_evidence']['engine_config'] = 'dtype=torch.bfloat16, seed=0, device=cuda'
        audit.validate_allocation(report, self.root, self.contract, self.contract_sha, self.profile_verification)

    def test_ledger_rejects_added_cap_and_changed_sampling_or_driver(self):
        audit.validate_ledger_header(self.ledger, self.contract)
        initial = json.loads(self.ledger.read_text().splitlines()[0])
        for key, value in [('max_iterations', 8), ('sampling', {}), ('driver_sha256', 'f' * 64)]:
            altered = copy.deepcopy(initial)
            altered['config'][key] = value
            self.ledger.write_text(json.dumps(altered) + '\n')
            with self.assertRaises(ValueError):
                audit.validate_ledger_header(self.ledger, self.contract)

    def test_empty_stdout_archive_and_unused_available_providers_are_valid(self):
        audit.validate_episode_runtime(self.episode, self.contract, {self.config_sha: {'provenance': self.provenance}})
        result = audit.audit_program_evidence(self.episode, self.attempt)
        self.assertEqual(result['native_stdout_archives'][0]['bytes'], 0)
        self.assertEqual(result['perception_calls'], [])
        (self.attempt / 'native_stdout_0000.log').unlink()
        with self.assertRaisesRegex(ValueError, 'archive missing'):
            audit.audit_program_evidence(self.episode, self.attempt)

    def test_episode_missing_patch_or_unknown_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'runtime evidence'):
            audit.validate_episode_runtime(self.episode, self.contract, {})
        altered = copy.deepcopy(self.episode)
        altered['perception_runtime']['service_provenance']['service_health']['graspnet'].pop('estimator_patch')
        with self.assertRaisesRegex(ValueError, 'verified allocation'):
            audit.validate_episode_runtime(altered, self.contract, {self.config_sha: {'provenance': self.provenance}})
        altered = copy.deepcopy(self.episode)
        altered['perception_runtime']['service_provenance']['gpu_uuid'] = 'GPU-other'
        with self.assertRaisesRegex(ValueError, 'verified allocation'):
            audit.validate_episode_runtime(altered, self.contract, {self.config_sha: {'provenance': self.provenance}})

    def test_full_wrapper_checks_scored_raw_usage_and_excludes_all_historical_counts(self):
        with patch.object(audit, 'load_contract', return_value=self.contract), \
             patch.object(audit.subset, 'audit_campaign', side_effect=lambda *a, **kw: self.base_result()), \
             patch.object(audit, 'verify_profile', return_value=self.profile_verification), \
             patch.object(audit, 'verify_benchmark_sources', return_value=5), \
             patch.object(audit, 'verify_group_layout_sources'):
            result = audit.audit_campaign(self.campaign, include_details=True)
            self.assertEqual(result['errors'], [])
            self.assertEqual(result['scored_episodes_with_verified_runtime'], 1)
            self.assertEqual(result['scoring'], {'scored': 1, 'success': 0})
            self.assertNotIn('combined_807_summary', result)
            self.assertNotIn('reused_results', result)
            self.assertEqual(result['runtime_episode_evidence'][0]['raw_usage_all_attempts']['prompt_tokens'], 7)
            self.assertFalse(result['campaign_complete'])
            self.write_ledger(requests=False)
            result = audit.audit_campaign(self.campaign)
            self.assertTrue(any('raw model' in error['message'] for error in result['errors']))
            self.assertEqual(result['scored_episodes_with_verified_runtime'], 0)

    def test_progress_does_not_walk_full_evidence_and_flags_reuse_file(self):
        (self.campaign / 'reused_results.json').write_text('{}')
        with patch.object(audit, 'load_contract', return_value=self.contract), \
             patch.object(audit.subset, 'audit_campaign', return_value=self.base_result()), \
             patch.object(audit, 'verify_profile') as verify:
            result = audit.audit_campaign(self.campaign, progress_only=True)
        verify.assert_not_called()
        self.assertFalse(result['runtime_evidence_audited'])
        self.assertFalse(result['campaign_complete'])
        self.assertTrue(any('reuse file' in row['message'] for row in result['errors']))
        self.assertEqual(result['scoring']['scored'], 1)


if __name__ == '__main__':
    unittest.main()
