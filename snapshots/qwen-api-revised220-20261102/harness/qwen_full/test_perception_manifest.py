"""CPU-only state isolation and frozen perception-comparison protocol tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from build_subset_manifest import canonical_sha256
from perception_exposure_inventory import inventory
from audit_perception_results import audit_comparison, provider_evidence, service_provenance
from audit_full_results import expected_result_path
from perception_manifest import (ARMS, TASKS, PROTECTED_PROFILE_FILES, PROTECTED_PROFILE_FILES_V2, REQUIRED_PROFILE_FILES,
    BOOTSTRAP_FILES, LEGACY_PROTOCOL, PROTOCOL, require_frozen_bootstrap,
    ESTIMATOR_PATCH_FILE, estimator_patch_for_profile,
    arm_campaign, build_manifest, validate_manifest, verify_profiles, validate_service_health)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.comparison = 'qwen38_perception_test'
        self.profiles = {}
        for arm in ARMS:
            profile = self.root / 'perception_profiles' / self.comparison / arm
            self.profiles[arm] = profile
            names = set(REQUIRED_PROFILE_FILES) | set(PROTECTED_PROFILE_FILES_V2) | {'qwen_full/perception_service.py'}
            if arm == 'perception':
                names.update(BOOTSTRAP_FILES)
            for name in names:
                target = profile / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('# fixed source fixture\n')
        self.source_path = Path(__file__).with_name('subset_manifest.json')
        self.source = json.loads(self.source_path.read_text())
        self.full = json.loads(Path(__file__).with_name('full_manifest.json').read_text())
        source = next(g for g in self.source['groups'] if g['task'] == TASKS[0])
        # A used equivalent state under a different seed must also be excluded.
        self.exposure = {'schema_version': 1, 'complete_for_scanned_roots': True,
            'scanned_roots': ['/historical/results'], 'unresolved': [],
            'source_full_manifest_canonical_sha256': canonical_sha256(self.full),
            'exposures': [{'task': TASKS[0], 'eval_seed': 2, 'layout_id': 9,
                'semantic_state_sha256': source['layout_content_sha256_by_id']['0'],
                'evidence': ['/historical/results/interrupted/episode_report.json']}]}
        self.exposure_path = self.root / 'exposure.json'
        self.exposure_path.write_text(json.dumps(self.exposure))
        self.runtime_path = self.root / 'runtime_manifest.json'
        self.runtime_path.write_text(json.dumps({'checkpoint_sha256': 'c' * 64, 'environment': {'python': '3.12'}}))

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return build_manifest(self.source_path, self.exposure_path, comparison=self.comparison,
            baseline_profile=self.profiles['baseline'], perception_profile=self.profiles['perception'],
            baseline_commit='a' * 40, perception_commit='b' * 40,
            runtime_manifest_path=self.runtime_path,
            coverage_attestation='All historical result roots inventoried, including earlier pilots and smoke attempts.')

    def test_six_pairs_twelve_unique_campaigns_native_budget(self):
        manifest = self.build()
        self.assertEqual(manifest['pair_count'], 6)
        self.assertEqual(manifest['units'][0]['group']['candidate_layout_ids'], [1])
        names = set()
        for index, unit in enumerate(manifest['units']):
            group = unit['group']
            self.assertEqual(group['group_id'], group['task'] + '_s0')
            self.assertEqual(group['target_episodes'], 1)
            self.assertEqual(group['candidate_count'], 1)
            self.assertEqual(group['step_limit'], {'stack_blocks': 550, 'hang_mugs': 800, 'insert_tubes': 500}[group['task']])
            self.assertEqual(unit['arm_order'], list(ARMS if index % 2 == 0 else reversed(ARMS)))
            for arm in ARMS:
                names.add(arm_campaign(manifest, arm, unit['unit_id']))
        self.assertEqual(len(names), 12)

    def test_rejects_model_budget_order_state_and_offline_skill_drift(self):
        manifest = self.build()
        mutations = [lambda m: m['inference'].update(mtp_tokens=0),
                     lambda m: m['units'][0]['group'].update(step_limit=551),
                     lambda m: m['units'][0].update(arm_order=['perception', 'baseline']),
                     lambda m: m['units'][0]['group'].update(candidate_layout_ids=[0]),
                     lambda m: m.update(offline_skill_discovery=True)]
        for change in mutations:
            altered = copy.deepcopy(manifest)
            change(altered)
            with self.assertRaises(ValueError):
                validate_manifest(altered)

    def test_rejects_incomplete_or_different_source_inventory(self):
        for field, value in [('unresolved', [{'reason': 'unknown attempted state'}]),
                             ('source_full_manifest_canonical_sha256', 'f' * 64)]:
            altered = copy.deepcopy(self.exposure)
            altered[field] = value
            self.exposure_path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                self.build()

    def test_protected_motion_and_model_sources_identical(self):
        path = self.profiles['perception'] / 'qwen_integration/native_retime.py'
        path.write_text('# changed speed\n')
        with self.assertRaisesRegex(ValueError, 'protected'):
            self.build()

    def test_profile_changes_after_freeze_detected(self):
        manifest = self.build()
        self.assertEqual(set(verify_profiles(self.root, manifest)), set(ARMS))
        (self.profiles['perception'] / 'extra.py').write_text('extra=1\n')
        with self.assertRaisesRegex(ValueError, 'pinned'):
            verify_profiles(self.root, manifest)

    def test_new_manifest_pins_shared_treatment_bootstrap_without_requiring_it_in_baseline(self):
        manifest = self.build()
        self.assertEqual(manifest['protocol'], PROTOCOL)
        self.assertEqual(manifest['schema_version'], 2)
        self.assertEqual(manifest['bootstrap']['profile'], 'perception')
        self.assertNotIn('qwen_full/perception_entry.py', manifest['profiles']['baseline']['files_sha256'])
        self.assertEqual(require_frozen_bootstrap(manifest), manifest['bootstrap'])
        for change in (lambda m: m.pop('bootstrap'),
                       lambda m: m['bootstrap']['files_sha256'].update({'qwen_full/perception_entry.py': 'f' * 64}),
                       lambda m: m['bootstrap'].update(profile='baseline')):
            altered = copy.deepcopy(manifest)
            change(altered)
            with self.assertRaisesRegex(ValueError, 'bootstrap'):
                validate_manifest(altered)

    def test_legacy_manifest_readable_but_cannot_be_launched_or_retroactively_upgraded(self):
        manifest = self.build()
        manifest.update(protocol=LEGACY_PROTOCOL, schema_version=1)
        manifest.pop('bootstrap')
        self.assertEqual(validate_manifest(manifest), manifest)
        with self.assertRaisesRegex(ValueError, 'audit only'):
            require_frozen_bootstrap(manifest)
        manifest['bootstrap'] = {}
        with self.assertRaisesRegex(ValueError, 'retrospectively'):
            validate_manifest(manifest)

    def test_progress_keeps_same_task_two_states_in_separate_native_runs(self):
        manifest = self.build()
        out = self.root / 'full_results' / self.comparison
        out.mkdir(parents=True)
        (out / 'manifest.json').write_text(json.dumps(manifest))
        simroot = self.root / 'simulator'
        for index, unit in enumerate(manifest['units'][:2]):
            group = unit['group']
            campaign = arm_campaign(manifest, 'baseline', unit['unit_id'])
            path = expected_result_path(simroot, campaign, group)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'eval_time': 1, 'details': {'0': {
                'layout_id': group['candidate_layout_ids'][0], 'success': bool(index), 'score': float(index)}}}))
        result = audit_comparison(out, simroot=simroot, progress_only=True)
        self.assertEqual(result['scored'], 2)
        self.assertEqual(result['arms']['baseline']['success'], 1)
        self.assertEqual(result['complete_pairs'], 0)
        self.assertFalse(result['complete'])
        self.assertFalse(result['errors'])

    def test_runtime_provider_flags_must_match_actual_arm(self):
        good = {'perception_runtime': {'providers': {'sam3': True, 'contact_graspnet': True},
                                      'offline_skill_discovery': False}}
        self.assertEqual(provider_evidence(good, 'perception'), good['perception_runtime'])
        with self.assertRaisesRegex(ValueError, 'provider'):
            provider_evidence(good, 'baseline')
        with self.assertRaisesRegex(ValueError, 'provider'):
            provider_evidence({}, 'perception')

    def test_patch_expectation_is_derived_from_frozen_helper_and_cannot_be_dropped(self):
        helper = self.profiles['perception'] / ESTIMATOR_PATCH_FILE
        helper.write_text('# fixture patch source\n')
        manifest = self.build()
        expected = estimator_patch_for_profile(manifest['profiles']['perception']['files_sha256'])
        self.assertIsNotNone(expected)
        self.assertEqual(manifest['perception_artifacts']['estimator_patch'], expected)
        for replacement in (None, {}, {**expected, 'patch_module_sha256': 'f' * 64}):
            altered = copy.deepcopy(manifest)
            altered['perception_artifacts']['estimator_patch'] = replacement
            with self.assertRaisesRegex(ValueError, 'estimator patch'):
                validate_manifest(altered)
        manifest['perception_artifacts'].pop('estimator_patch')
        with self.assertRaisesRegex(ValueError, 'estimator patch'):
            validate_manifest(manifest)

    def test_unpatched_profiles_remain_explicitly_unpatched(self):
        manifest = self.build()
        self.assertIsNone(manifest['perception_artifacts']['estimator_patch'])
        manifest['perception_artifacts']['estimator_patch'] = {'id': 'unfrozen correction'}
        with self.assertRaisesRegex(ValueError, 'estimator patch'):
            validate_manifest(manifest)
        # Historical v1 has neither a bootstrap nor a patch declaration.
        manifest['perception_artifacts'].pop('estimator_patch')
        manifest.pop('bootstrap')
        manifest.update(protocol=LEGACY_PROTOCOL, schema_version=1)
        self.assertEqual(validate_manifest(manifest), manifest)


class ExposureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.logs = self.root / 'results'
        self.logs.mkdir()
        layouts = []
        for i in range(4):
            path = self.root / f'layout_{i}.json'
            path.write_text(json.dumps({'Room': {}, 'object': {'position': [i, 0, 0]}}))
            layouts.append({'layout_id': i, 'source_path': str(path)})
        self.manifest = {'groups': [{'task': 'stack_blocks', 'eval_seed': 0, 'layouts': layouts}]}

    def tearDown(self):
        self.temp.cleanup()

    def test_unscored_directory_and_usage_count_planned_queue_does_not(self):
        (self.logs / 'episodes/stack_blocks_s0_l0/attempt_001').mkdir(parents=True)
        (self.logs / 'usage.jsonl').write_text('\n'.join(map(json.dumps, [
            {'event': 'suite_initialized', 'config': {'episode_ids': ['stack_blocks_s0_l3']}},
            {'event': 'generation_started', 'episode_id': 'stack_blocks_s0_l1'}])) + '\n')
        result = inventory(self.manifest, [self.logs])
        self.assertTrue(result['complete_for_scanned_roots'])
        self.assertEqual([r['layout_id'] for r in result['exposures']], [0, 1])

    def test_native_scores_and_abandoned_initializations_count(self):
        directory = self.logs / 'eval_result/RoboDojo/stack_blocks/Policy/arx_x5/0_run/session'
        directory.mkdir(parents=True)
        (directory / '_result.json').write_text(json.dumps({'details': {'0': {'layout_id': 2, 'success': False}}}))
        (directory.parent / '_resume_session.json').write_text(json.dumps({'abandoned_layout_ids': [3]}))
        result = inventory(self.manifest, [self.logs])
        self.assertTrue(result['complete_for_scanned_roots'])
        self.assertEqual([r['layout_id'] for r in result['exposures']], [2, 3])

    def test_unknown_attempt_identity_fails_closed(self):
        (self.logs / 'smoke_report.json').write_text(json.dumps({'task': 'stack_blocks', 'layout_id': 0}))
        result = inventory(self.manifest, [self.logs])
        self.assertFalse(result['complete_for_scanned_roots'])
        self.assertEqual(len(result['unresolved']), 1)

    def test_source_profiles_and_assets_not_interpreted_as_rollouts(self):
        (self.logs / 'perception_profiles/x/stack_blocks_s0_l3').mkdir(parents=True)
        result = inventory(self.manifest, [self.logs])
        self.assertEqual(result['exposures'], [])


class ServiceHealthTests(unittest.TestCase):
    def setUp(self):
        self.files = {'sam3': {'path': '/weights/sam3.pt', 'bytes': 100, 'sha256': 'a' * 64},
                      'graspnet': {'path': '/vendor/checkpoints/contact_graspnet/checkpoints/model.pt',
                                   'bytes': 200, 'sha256': 'b' * 64}}
        self.runtime = {'files': list(self.files.values())}
        self.health = {}
        for kind in ('sam3', 'graspnet'):
            row = {'kind': kind, 'ok': True, 'loaded': True, 'model_loaded': True,
                   'offline_weights': True, 'device': 'cuda:0', 'parameter_devices': ['cuda:0'],
                   'parameter_count': 1000, 'random_seed': 0,
                   'checkpoint_sha256': self.files[kind]['sha256']}
            row.update({'files': {'sam3.pt': self.files[kind]}} if kind == 'sam3'
                       else {'checkpoint': self.files[kind]})
            self.health[kind] = row

    def test_loaded_gpu_checkpoint_evidence(self):
        for kind, row in self.health.items():
            self.assertEqual(validate_service_health(row, self.runtime, kind), row)
            for field, value in [('device', 'cpu'), ('parameter_devices', ['cpu']),
                                 ('loaded', False), ('model_loaded', False), ('parameter_count', 0),
                                 ('checkpoint_sha256', 'f' * 64), ('random_seed', 1)]:
                altered = copy.deepcopy(row)
                altered[field] = value
                with self.assertRaisesRegex(ValueError, 'assigned GPU'):
                    validate_service_health(altered, self.runtime, kind)

    def test_huggingface_logical_name_and_resolved_blob_path(self):
        row = self.files['sam3']
        row.update(file='sam3.pt', path='/weights/blobs/' + row['sha256'])
        validate_service_health(self.health['sam3'], self.runtime, 'sam3')

    def test_missing_duplicate_and_different_checkpoint_rejected(self):
        for runtime in [{'files': []}, {'files': [self.files['sam3'], self.files['sam3']]}]:
            with self.assertRaisesRegex(ValueError, 'exactly one checkpoint'):
                validate_service_health(self.health['sam3'], runtime, 'sam3')
        altered = copy.deepcopy(self.health['graspnet'])
        altered['checkpoint']['bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'path/size/hash'):
            validate_service_health(altered, self.runtime, 'graspnet')

    def test_service_provenance_requires_health_and_gpu_identity(self):
        artifacts = {'runtime_manifest': self.runtime, 'runtime_manifest_sha256': 'c' * 64,
                     'service_runner_sha256': 'd' * 64}
        provenance = {**artifacts, 'offline_skill_discovery': False, 'gpu_uuid': 'GPU-123',
                      'service_health': self.health}
        self.assertEqual(service_provenance(provenance, {'perception_artifacts': artifacts}), provenance)
        for field, value in [('service_health', {}), ('gpu_uuid', ''),
                             ('runtime_manifest_sha256', 'f' * 64)]:
            altered = copy.deepcopy(provenance)
            altered[field] = value
            with self.assertRaises(ValueError):
                service_provenance(altered, {'perception_artifacts': artifacts})

    def test_patch_health_must_match_exact_expected_correction(self):
        expected = estimator_patch_for_profile({ESTIMATOR_PATCH_FILE: 'e' * 64})
        health = copy.deepcopy(self.health['graspnet'])
        with self.assertRaisesRegex(ValueError, 'estimator patch'):
            validate_service_health(health, self.runtime, 'graspnet', estimator_patch=expected)
        health['estimator_patch'] = expected
        self.assertEqual(validate_service_health(health, self.runtime, 'graspnet',
                                                estimator_patch=expected), health)
        for field in expected:
            altered = copy.deepcopy(health)
            altered['estimator_patch'][field] = 'changed'
            with self.assertRaisesRegex(ValueError, 'estimator patch'):
                validate_service_health(altered, self.runtime, 'graspnet', estimator_patch=expected)
        # An old unpatched run must not silently accept newly patched health.
        with self.assertRaisesRegex(ValueError, 'estimator patch'):
            validate_service_health(health, self.runtime, 'graspnet')

    def test_final_audit_rejects_missing_or_altered_patch_health(self):
        expected = estimator_patch_for_profile({ESTIMATOR_PATCH_FILE: 'e' * 64})
        artifacts = {'runtime_manifest': self.runtime, 'runtime_manifest_sha256': 'c' * 64,
                     'service_runner_sha256': 'd' * 64, 'estimator_patch': expected}
        provenance = {'runtime_manifest': self.runtime, 'runtime_manifest_sha256': 'c' * 64,
                      'service_runner_sha256': 'd' * 64, 'offline_skill_discovery': False,
                      'gpu_uuid': 'GPU-123', 'service_health': copy.deepcopy(self.health)}
        with self.assertRaisesRegex(ValueError, 'estimator patch'):
            service_provenance(provenance, {'perception_artifacts': artifacts})
        provenance['service_health']['graspnet']['estimator_patch'] = expected
        self.assertEqual(service_provenance(provenance, {'perception_artifacts': artifacts}), provenance)
        provenance['service_health']['graspnet']['estimator_patch'] = {**expected, 'id': 'another patch'}
        with self.assertRaisesRegex(ValueError, 'estimator patch'):
            service_provenance(provenance, {'perception_artifacts': artifacts})


if __name__ == '__main__':
    unittest.main()
