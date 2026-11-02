"""CPU-only tests for paired state identity, source isolation, resume, and audit."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from audit_paired_results import (audit_comparison, episode_metrics,
                                 allocation_model_evidence, ledger_episode_metrics)
from audit_full_results import expected_result_path
from paired_entry import activate_profile, checked_count
from paired_manifest import (ARMS, REQUIRED_PROFILE_FILES, SELECTION, arm_campaign,
    PINNED_MODEL, build_manifest, canonical_sha256, gpu_assignment, selected_group, sha_file,
    validate_manifest, verify_layouts, verify_profiles)
from paired_supervisor import pending_group


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.simroot = Path(self.temporary.name)
        self.root = self.simroot / 'aspire_validate'
        self.campaign = 'qwen38_paired_test'
        self.profile_paths = {}
        for arm in ARMS:
            profile = self.root / 'paired_profiles' / self.campaign / arm
            self.profile_paths[arm] = profile
            for filename in REQUIRED_PROFILE_FILES:
                path = profile / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# immutable fixture\n')
        self.manifest = build_manifest(Path(__file__).with_name('subset_manifest.json'),
            comparison=self.campaign, baseline_profile=self.profile_paths['baseline'],
            fixed_profile=self.profile_paths['fixed'], baseline_commit='a'*40, fixed_commit='b'*40)
        self.out = self.root / 'full_results' / self.campaign
        self.out.mkdir(parents=True)

    def tearDown(self):
        self.temporary.cleanup()

    def test_twelve_exact_states_per_arm_and_native_budgets(self):
        self.assertEqual(sum(g['target_episodes'] for g in self.manifest['groups']), 12)
        self.assertEqual(self.manifest['total_target_episodes'], 24)
        for group, source in zip(self.manifest['groups'], self.manifest['source_groups']):
            self.assertEqual(group['candidate_layout_ids'], SELECTION[group['task']])
            self.assertEqual(group['step_limit'], source['step_limit'])
            self.assertEqual(group['candidate_count'], group['target_episodes'])
            self.assertEqual(group['eval_seed'], 0)
            self.assertEqual(checked_count(group['native_target_episodes'], mode='evaluation',
                             override=str(group['target_episodes']), group=group), group['target_episodes'])

    def test_scope_model_action_queue_drift_rejected(self):
        for mutate in (lambda m: m['inference'].update(mtp_tokens=0),
                       lambda m: m['groups'][0].update(step_limit=300),
                       lambda m: m['groups'][0].update(candidate_layout_ids=[0,1]),
                       lambda m: m['profiles']['fixed'].update(relative_root='../fixed')):
            altered = copy.deepcopy(self.manifest)
            mutate(altered)
            with self.assertRaises(ValueError):
                validate_manifest(altered)

    def test_profile_hashes_and_extra_source_files_checked(self):
        self.assertEqual(set(verify_profiles(self.root, self.manifest)), set(ARMS))
        extra = self.profile_paths['baseline'] / 'qwen_full/unpinned.py'
        extra.write_text('unexpected = True\n')
        with self.assertRaisesRegex(ValueError, 'differs'):
            verify_profiles(self.root, self.manifest)
        extra.unlink()
        path = self.profile_paths['fixed'] / REQUIRED_PROFILE_FILES[0]
        path.write_text('changed = True\n')
        with self.assertRaisesRegex(ValueError, 'differs'):
            verify_profiles(self.root, self.manifest)

    def test_profile_precedes_mutable_script_directory(self):
        prior = sys.path.copy()
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            selected = activate_profile(self.profile_paths['baseline'])
            self.assertEqual(sys.path[:3], selected)
            self.assertNotIn(str(Path(__file__).resolve().parent), sys.path)
        finally:
            sys.path[:] = prior

    def test_gpu_arm_assignment_alternates(self):
        assignments = [gpu_assignment(i) for i in range(7)]
        self.assertEqual(sum(row['baseline'] == 0 for row in assignments), 4)
        self.assertTrue(all(set(row.values()) == {0, 3} for row in assignments))
        self.assertNotEqual(assignments[0], assignments[1])

    def native_fixture(self, scored=1):
        group = self.manifest['groups'][0]
        arm_dir = self.out.parent / arm_campaign(self.manifest, 'baseline')
        native = expected_result_path(self.simroot, arm_dir.name, group)
        native.parent.mkdir(parents=True)
        details = {str(i): {'layout_id': group['candidate_layout_ids'][i], 'success': False, 'score': 0.}
                   for i in range(scored)}
        native.write_text(json.dumps({'eval_time': scored, 'details': details}))
        return group, arm_dir, native, details

    def test_scored_failures_cannot_replay_without_checkpoint(self):
        group, arm_dir, native, details = self.native_fixture()
        with self.assertRaisesRegex(RuntimeError, 'without native checkpoint'):
            pending_group(arm_dir, group, self.simroot)
        checkpoint = native.parent.parent / ('_resume_' + native.parent.name + '.json')
        checkpoint.write_text(json.dumps({'completed_layout_ids': [0], 'abandoned_layout_ids': []}))
        self.assertIsNone(pending_group(arm_dir, group, self.simroot))
        checkpoint.write_text(json.dumps({'completed_layout_ids': [], 'abandoned_layout_ids': []}))
        with self.assertRaisesRegex(RuntimeError, 'disagree'):
            pending_group(arm_dir, group, self.simroot)

    def test_abandoned_exact_state_is_never_replaced_or_restarted(self):
        group, arm_dir, native, details = self.native_fixture()
        checkpoint = native.parent.parent / ('_resume_' + native.parent.name + '.json')
        checkpoint.write_text(json.dumps({'completed_layout_ids': [0], 'abandoned_layout_ids': [13]}))
        with self.assertRaisesRegex(RuntimeError, 'abandoned'):
            pending_group(arm_dir, group, self.simroot)

    def test_complete_scores_missing_evidence_are_not_replayed(self):
        group, arm_dir, native, details = self.native_fixture(scored=2)
        with self.assertRaisesRegex(RuntimeError, 'completion/video evidence'):
            pending_group(arm_dir, group, self.simroot)

    def test_complete_scored_failures_reused_with_all_camera_evidence(self):
        group, arm_dir, native, details = self.native_fixture(scored=2)
        report_path = arm_dir / 'groups' / group['group_id'] / 'group_report.json'
        report_path.parent.mkdir(parents=True)
        report_path.write_text(json.dumps({'mode': 'evaluation', 'task': group['task'], 'eval_seed': 0,
            'native_run_id': native.parent.name, 'native_result_path': str(native),
            'complete': True, 'target': 2, 'scored': 2}))
        for i in range(2):
            for camera in ('head', 'left_wrist', 'right_wrist'):
                (native.parent / f'episode_{i:07d}_cam_{camera}_fail.mp4').write_bytes(b'evidence')
        result = pending_group(arm_dir, group, self.simroot)
        self.assertEqual((result['scored'], result['success'], result['video_count']), (2, 0, 6))

    def test_missing_pairs_not_implicitly_completed_by_progress_audit(self):
        (self.out / 'manifest.json').write_text(json.dumps(self.manifest))
        self.native_fixture(scored=1)
        result = audit_comparison(self.out, simroot=self.simroot, progress_only=True)
        self.assertEqual(result['scored'], 1)
        self.assertEqual(result['complete_pairs'], 0)
        self.assertFalse(result['complete'])
        self.assertEqual(result['pairs'][0]['baseline']['layout_id'], 0)
        self.assertIsNone(result['pairs'][0]['fixed'])

    def test_feedback_and_tool_metrics_do_not_double_count_timing(self):
        result = episode_metrics({'wall_seconds': 100, 'iterations': [
            {'program_wall_seconds': 20, 'execution': {'program_ok': False, 'error': 'example',
                'tool_calls': [{'tool': 'get_observation', 'ok': True, 'duration_s': 3}]}}]})
        self.assertEqual(result['program_seconds'], 20)
        self.assertEqual(result['tool_seconds']['get_observation'], 3)
        self.assertEqual(result['wall_seconds'], 100)
        self.assertEqual(result['program_errors'], ['example'])

    def test_model_driver_must_be_identical_between_arms(self):
        altered = copy.deepcopy(self.manifest)
        profile = altered['profiles']['fixed']
        profile['files_sha256']['qwen_full/full_model_driver.py'] = 'a'*64
        profile['files_manifest_sha256'] = canonical_sha256(profile['files_sha256'])
        with self.assertRaisesRegex(ValueError, 'byte-identical'):
            validate_manifest(altered)

    def test_terminal_and_unscored_model_usage_counted_without_double_sum(self):
        eid = 'align_blocks_s0_l0'
        rows = [{'event': 'suite_initialized'},
            {'event': 'input_token_count', 'episode_id': eid, 'iteration': 0, 'wall_s': .2},
            {'event': 'generation_started', 'episode_id': eid, 'iteration': 0},
            {'event': 'raw_response', 'episode_id': eid, 'iteration': 0, 'generation_wall_s': 3.5},
            {'event': 'generation_usage', 'episode_id': eid, 'iteration': 0,
             'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'generation_wall_s': 3.4}},
            {'event': 'model_episode_failed', 'episode_id': eid, 'iteration': 0},
            {'event': 'generation_started', 'episode_id': eid, 'attempt': 2, 'iteration': 0},
            {'event': 'generation_transport_failure', 'episode_id': eid, 'attempt': 2,
             'iteration': 0, 'generation_wall_s': 7.},
            {'event': 'unscored_attempt_abandoned', 'episode_id': eid, 'attempt': 2}]
        path = self.out / 'usage.jsonl'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        metrics = ledger_episode_metrics(path, [eid], [])[eid]
        self.assertEqual((metrics['model_calls_started'], metrics['model_calls_with_usage']), (2, 1))
        self.assertEqual(metrics['generation_wall_seconds'], 10.5)
        self.assertEqual(metrics['completion_tokens_including_reasoning'], 20)
        self.assertEqual(metrics['calls_without_final_token_usage'], 1)
        self.assertEqual(metrics['abandoned_attempts'], 1)

    def test_actual_allocation_model_settings_required(self):
        report = {'allocation': '123', 'actual_inference_variant': 'bf16_mtp3',
            'context_length': 262144, 'model_manifest': dict(PINNED_MODEL, snapshot_path='/pinned/model'),
            'server_command': ['python', '--model', '/pinned/model', '--dtype', 'bfloat16',
                '--tensor-parallel-size', '2', '--max-model-len', '262144', '--max-num-seqs', '2',
                '--speculative-config', '{"method":"mtp","num_speculative_tokens":3}'],
            'inference_variant_evidence': {'variant': 'bf16_mtp3'},
            'profile_verification': {arm: {'files_manifest_sha256': self.manifest['profiles'][arm]['files_manifest_sha256']}
                                     for arm in ARMS}}
        path = self.out / 'allocation.json'
        path.write_text(json.dumps(report))
        self.assertEqual(allocation_model_evidence(path, self.manifest)['model_revision'], PINNED_MODEL['revision'])
        report['actual_inference_variant'] = 'bf16'
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'did not verify'):
            allocation_model_evidence(path, self.manifest)


if __name__ == '__main__':
    unittest.main()
