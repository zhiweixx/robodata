"""Offline campaign accounting, process ownership, and graceful-stop tests.

No GPU, inference request, simulator, Slurm command, or remote host is used.
The signal test substitutes a tiny local process for srun.
"""
import copy
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import full_supervisor as supervisor


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        stop_patch = patch.object(supervisor, 'stop_process_group', return_value={'stopped': True})
        stop_patch.start()
        self.addCleanup(stop_patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.simroot = Path(self.tmp.name)
        self.root = self.simroot / 'aspire_validate'
        (self.root / 'qwen_full').mkdir(parents=True)
        self.campaign_id = 'offline_campaign'
        self.group = {'group_id': 'general_pickup_s0', 'task': 'general_pickup',
                      'eval_seed': 0, 'target_episodes': 2, 'candidate_layout_ids': [0, 1, 2, 3],
                      'episode_ids': [f'general_pickup_s0_l{i}' for i in range(4)]}
        self.group2 = {'group_id': 'stack_bowls_s1', 'task': 'stack_bowls',
                       'eval_seed': 1, 'target_episodes': 2, 'candidate_layout_ids': [0, 1, 2, 3],
                       'episode_ids': [f'stack_bowls_s1_l{i}' for i in range(4)]}
        self.manifest = {'group_count': 2, 'total_target_episodes': 4,
                         'groups': [self.group, self.group2]}
        self.manifest_path = self.root / 'input_manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.report_path = self.root / 'full_results' / self.campaign_id / 'groups' / self.group['group_id'] / 'group_report.json'
        self.report_path.parent.mkdir(parents=True)
        self.run_id = self.campaign_id + '_' + self.group['group_id']
        self.native_dir = self.simroot / 'RoboDojo/eval_result/RoboDojo/general_pickup/ASPIRE_Pilot/arx_x5/0_aspire_qwen_full' / self.run_id
        self.native_dir.mkdir(parents=True)
        self.result_path = self.native_dir / '_result.json'
        self.native = {'eval_time': 2, 'details': {
            '0': {'layout_id': 0, 'success': False, 'score': .25},
            '1': {'layout_id': 3, 'success': True, 'score': 1.0}}}
        self.result_path.write_text(json.dumps(self.native))
        container_result = '/work/' + str(self.result_path.relative_to(self.simroot))
        self.report = {'complete': True, 'target': 2, 'scored': 2, 'mode': 'evaluation',
                       'task': self.group['task'], 'eval_seed': 0, 'native_run_id': self.run_id,
                       'native_result_path': container_result}
        self.report_path.write_text(json.dumps(self.report))
        for index, outcome in [(0, 'fail'), (1, 'success')]:
            for camera in ('head', 'left_wrist', 'right_wrist'):
                (self.native_dir / f'episode_{index:07d}_cam_{camera}_{outcome}.mp4').write_bytes(b'video fixture')

    def tearDown(self):
        self.tmp.cleanup()

    def campaign(self, mode='resume', mtp_tokens=None):
        args = NS(root=self.root, campaign=self.campaign_id, manifest=self.manifest_path, mode=mode)
        if mtp_tokens is not None:
            args.mtp_tokens = mtp_tokens
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '5,2,7,6', 'SLURM_JOB_ID': 'offline'}, clear=True):
            return supervisor.Campaign(args)

    def validate(self):
        return supervisor.validate_group(self.report_path, self.group, self.simroot,
                                         expected_campaign=self.campaign_id)

    def test_resume_accepts_only_unique_scored_native_layouts(self):
        result = self.validate()
        self.assertEqual((result['scored'], result['success'], result['score_sum']), (2, 1, 1.25))
        self.assertEqual(result['video_count'], 6)
        self.assertEqual(result['native_result_path'], str(self.result_path))
        for replacement in (0, 4, True):
            with self.subTest(layout_id=replacement):
                native = copy.deepcopy(self.native)
                native['details']['1']['layout_id'] = replacement
                self.result_path.write_text(json.dumps(native))
                with self.assertRaises(RuntimeError):
                    self.validate()

    def test_wrong_task_seed_campaign_or_mode_cannot_resume_as_complete(self):
        for field, wrong in [('task', 'stack_bowls'), ('eval_seed', 1),
                             ('eval_seed', False), ('native_run_id', 'different_campaign'),
                             ('mode', 'interface_smoke')]:
            with self.subTest(field=field, value=wrong):
                self.report_path.write_text(json.dumps(dict(self.report, **{field: wrong})))
                with self.assertRaisesRegex(RuntimeError, 'identity'):
                    self.validate()

    def test_incomplete_count_and_noncanonical_episode_indices_rejected(self):
        self.report_path.write_text(json.dumps(dict(self.report, scored=1)))
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            self.validate()
        self.report_path.write_text(json.dumps(self.report))
        native = copy.deepcopy(self.native)
        native['details']['2'] = native['details'].pop('1')
        self.result_path.write_text(json.dumps(native))
        with self.assertRaisesRegex(RuntimeError, 'indices'):
            self.validate()

    def test_video_count_cannot_hide_missing_wrist_camera(self):
        wrist = self.native_dir / 'episode_0000000_cam_left_wrist_fail.mp4'
        wrist.rename(self.native_dir / 'episode_0000000_cam_head_success.mp4')
        self.assertEqual(len(list(self.native_dir.glob('*.mp4'))), 6)
        with self.assertRaisesRegex(RuntimeError, 'camera video'):
            self.validate()

    def test_zero_length_video_and_wrong_episode_outcome_rejected(self):
        video = self.native_dir / 'episode_0000000_cam_head_fail.mp4'
        video.write_bytes(b'')
        with self.assertRaisesRegex(RuntimeError, 'empty'):
            self.validate()
        video.write_bytes(b'video fixture')
        video.rename(video.with_name('episode_0000000_cam_head_success.mp4'))
        with self.assertRaisesRegex(RuntimeError, 'outcome'):
            self.validate()

    def test_changed_manifest_is_rejected_before_model_or_gpu_work(self):
        campaign = self.campaign()
        (campaign.out / 'manifest.json').write_text(json.dumps(dict(self.manifest, total_target_episodes=5)))
        with patch.object(supervisor.subprocess, 'check_output') as gpu_query:
            with self.assertRaisesRegex(RuntimeError, 'manifest changed'):
                campaign.prepare()
        gpu_query.assert_not_called()

    def test_campaign_lock_prevents_second_launcher_work(self):
        campaign = self.campaign()
        campaign.prepare = Mock()
        with (campaign.out / 'campaign.lock').open('a') as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(supervisor, 'Campaign', return_value=campaign), patch.object(sys, 'argv', [
                    'full_supervisor.py', '--campaign', self.campaign_id, '--manifest', str(self.manifest_path)]):
                with self.assertRaises(BlockingIOError):
                    supervisor.main()
        campaign.prepare.assert_not_called()

    def test_resume_skips_completed_group_and_counts_only_confirmed_results(self):
        campaign = self.campaign()
        campaign.start_server = Mock()
        campaign.record = Mock()
        first = self.validate()
        second = dict(first, group_id=self.group2['group_id'], task=self.group2['task'], eval_seed=1)
        seen = []
        def run(group, gpu_index, smoke=False):
            self.assertFalse(smoke)
            seen.append(group['group_id'])
            with campaign.lock:
                campaign.completed[group['group_id']] = second
        campaign.run_group = run
        with patch.object(supervisor, 'validate_group', return_value=first):
            campaign.run()
        self.assertEqual(seen, [self.group2['group_id']])
        self.assertEqual(len(campaign.completed), 2)
        campaign.save_progress()
        progress = json.loads((campaign.out / 'progress.json').read_text())
        self.assertEqual((progress['groups_completed'], progress['scored'], progress['target']), (2, 4, 4))
        self.assertTrue(progress['complete'])

    def test_partial_progress_is_not_a_complete_campaign(self):
        campaign = self.campaign()
        campaign.completed[self.group['group_id']] = self.validate()
        campaign.save_progress()
        progress = json.loads((campaign.out / 'progress.json').read_text())
        self.assertEqual(progress['scored'], 2)
        self.assertFalse(progress['complete'])

    def test_server_context_and_batch_chunk_are_not_output_caps(self):
        command = supervisor.server_command('/python', '/model', 12345)
        self.assertEqual(command[command.index('--tensor-parallel-size') + 1], '2')
        self.assertEqual(command[command.index('--max-model-len') + 1], '262144')
        self.assertEqual(command[command.index('--max-num-batched-tokens') + 1], '8192')
        self.assertIn('--enable-chunked-prefill', command)
        self.assertNotIn('--max-tokens', command)
        self.assertNotIn('--max-new-tokens', command)

    def test_mtp_is_opt_in_and_default_command_is_identical(self):
        expected = ['/python', '-m', 'vllm.entrypoints.openai.api_server',
            '--model', '/model', '--served-model-name', supervisor.MODEL,
            '--dtype', 'bfloat16', '--tensor-parallel-size', '2',
            '--max-num-seqs', '2', '--max-model-len', '262144',
            '--max-num-batched-tokens', '8192', '--enable-chunked-prefill',
            '--limit-mm-per-prompt', json.dumps({'image': 262144, 'video': 0}),
            '--skip-mm-profiling', '--gpu-memory-utilization', '0.85',
            '--reasoning-parser', 'qwen3', '--host', '127.0.0.1', '--port', '12345']
        self.assertEqual(supervisor.server_command('/python', '/model', 12345), expected)
        self.assertEqual(supervisor.server_command('/python', '/model', 12345, 0), expected)
        for depth in (1, 3):
            command = supervisor.server_command('/python', '/model', 12345, depth)
            self.assertEqual(command[:-2], expected)
            self.assertEqual(command[-2], '--speculative-config')
            self.assertEqual(json.loads(command[-1]), {'method': 'mtp', 'num_speculative_tokens': depth})
        for value in (-1, 2, 4, True, '1', None, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                supervisor.server_command('/python', '/model', 12345, value)

    def test_actual_variant_requires_matching_engine_startup(self):
        log = self.simroot / 'startup.log'
        for depth in (0, 1, 3):
            spec = 'None' if depth == 0 else f"SpeculativeConfig(method='mtp', num_spec_tokens={depth})"
            log.write_text(f"Initializing a V1 LLM engine with config: speculative_config={spec}, tokenizer='model', dtype=torch.bfloat16, quantization=None\n")
            result = supervisor.verify_inference_variant(log, depth)
            self.assertEqual(result['variant'], 'bf16' if depth == 0 else f'bf16_mtp{depth}')
            with self.assertRaisesRegex(RuntimeError, 'differs'):
                supervisor.verify_inference_variant(log, 3 if depth != 3 else 1)

    def test_mtp_phase_index_preserves_historical_baseline_provenance(self):
        campaign = self.campaign(mtp_tokens=3)
        self.assertEqual(campaign.report['requested_inference_variant'], 'bf16_mtp3')
        self.assertIsNone(campaign.report['actual_inference_variant'])
        historical = campaign.out / 'allocations/old_baseline/allocation.json'
        historical.parent.mkdir()
        old = {'allocation': 'old_baseline', 'mode': 'resume', 'complete': False,
               'server_command': supervisor.server_command('/python', '/model', 12345)}
        old_text = json.dumps(old)
        historical.write_text(old_text)
        (campaign.runtime / 'allocation.json').write_text(json.dumps(campaign.report))
        experiment = campaign.out / 'experiment_config.json'
        experiment.write_text(json.dumps({'campaign': campaign.campaign, 'history': 'unchanged'}))
        campaign.update_inference_phases()
        indexed = json.loads(experiment.read_text())
        phases = {phase['allocation']: phase for phase in indexed['inference_phases']}
        self.assertEqual(indexed['history'], 'unchanged')
        self.assertEqual(phases['old_baseline']['variant'], 'bf16')
        self.assertEqual(phases['old_baseline']['variant_evidence'], 'historical_server_command')
        self.assertIsNone(phases['offline']['variant'])
        self.assertEqual(phases['offline']['requested_variant'], 'bf16_mtp3')
        self.assertEqual(historical.read_text(), old_text)

    def test_new_unverified_allocation_is_not_inferred_from_its_command(self):
        report = {'allocation': 'new_mtp', 'mode': 'resume',
                  'requested_inference_variant': 'bf16_mtp3',
                  'actual_inference_variant': None,
                  'server_command': supervisor.server_command('/python', '/model', 12345, 3),
                  'error': {'message': 'engine configuration verification failed'}}
        phase = supervisor.allocation_inference_phase(report, '/new_mtp/allocation.json')
        self.assertIsNone(phase['variant'])
        self.assertEqual(phase['variant_evidence'], 'engine_not_verified')
        self.assertEqual(phase['requested_variant'], 'bf16_mtp3')
        # Only older reports lacking the field may use command-only evidence.
        del report['actual_inference_variant']
        historical = supervisor.allocation_inference_phase(report, '/old_mtp/allocation.json')
        self.assertEqual(historical['variant'], 'bf16_mtp3')
        self.assertEqual(historical['variant_evidence'], 'historical_server_command')

    def test_live_relay_uses_private_node_storage(self):
        with patch.dict(os.environ, {'SLURM_TMPDIR': self.tmp.name}):
            relay = supervisor.create_private_relay('offline', 'test_s0')
        try:
            self.assertEqual(relay.parent, Path(self.tmp.name))
            self.assertEqual(relay.stat().st_mode & 0o077, 0)
            self.assertEqual(relay.stat().st_uid, os.getuid())
        finally:
            relay.rmdir()

    def test_dead_controller_interrupts_even_if_simulator_shutdown_is_still_running(self):
        campaign = self.campaign()
        simulator = NS(poll=Mock(return_value=None))
        controller = NS(poll=Mock(return_value=1))
        with self.assertRaisesRegex(RuntimeError, 'inference controller exited'):
            campaign.wait(simulator, required_processes=(('inference controller', controller),))

    def test_gpu_mapping_reserves_two_server_and_two_simulator_devices(self):
        campaign = self.campaign()
        campaign.gpus = [{'uuid': f'GPU-device-{i}', 'index': physical}
                         for i, physical in enumerate(('5', '2', '7', '6'))]
        campaign.snapshot, campaign.port, campaign.url = Path('/model'), 12345, 'http://127.0.0.1:12345/v1'
        campaign.record = Mock()
        campaign.wait, campaign.cleanup_child = Mock(return_value=0), Mock()
        smoke_dir = campaign.runtime / 'model_smoke'
        smoke_dir.mkdir()
        (smoke_dir / 'full_model_smoke.json').write_text('{"ok": true}')
        fake = NS(pid=123)
        with patch.object(supervisor.subprocess, 'Popen', return_value=fake) as popen, \
                patch.object(supervisor, 'wait_ready', return_value={'data': []}), \
                patch.object(supervisor, 'verify_inference_variant', return_value={'variant': 'bf16'}):
            campaign.start_server()
        self.assertEqual(popen.call_args_list[0].kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-device-1,GPU-device-2')
        for index in (0, 3):
            smoke = campaign.campaign + f'_smoke{index}_{campaign.job}'
            destination = campaign.root / 'full_results' / smoke / 'groups' / self.group['group_id']
            destination.mkdir(parents=True)
            (destination / 'group_report.json').write_text('{"complete": true}')
            with patch.object(supervisor.subprocess, 'Popen', return_value=fake) as popen:
                campaign.run_group(self.group, index, smoke=True)
            self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], f'GPU-device-{index}')
            self.assertEqual(popen.call_args.args[0][-2:], [campaign.gpus[index]['index'], f'GPU-device-{index}'])
            attempt = campaign.runtime / 'smoke' / f'worker{index}' / self.group['group_id']
            self.assertTrue((attempt / 'relay').is_dir())
            location = json.loads((attempt / 'relay_location.json').read_text())
            self.assertFalse(Path(location['live_node_local_path']).exists())

    def test_sbatch_signal_waits_for_step_cleanup_and_preserves_exit_code(self):
        binary = self.simroot / 'bin'
        binary.mkdir()
        marker, ready, argv_path = self.simroot / 'cleaned', self.simroot / 'ready', self.simroot / 'argv.json'
        fake_srun = binary / 'srun'
        fake_srun.write_text('#!' + sys.executable + '\n' +
            'import os, signal, time, json, sys\nfrom pathlib import Path\n'
            'def stop(signum, frame):\n'
            '    time.sleep(.25)\n'
            f'    Path({str(marker)!r}).write_text("cleaned")\n'
            '    raise SystemExit(17)\n'
            'signal.signal(signal.SIGUSR1, stop)\n'
            f'Path({str(argv_path)!r}).write_text(json.dumps(sys.argv))\n'
            f'Path({str(ready)!r}).write_text(str(os.getpid()))\n'
            'while True: time.sleep(.05)\n')
        fake_srun.chmod(0o700)
        source = Path(supervisor.__file__).with_name('full_eval.sbatch').read_text()
        source = source.replace('QWEN_FULL_ROOT=/srv/ROBODATA_RUNTIME/isaac6/aspire_validate',
                                'QWEN_FULL_ROOT=' + shlex.quote(str(self.root)))
        script = self.simroot / 'batch_test.sh'
        script.write_text(source)
        env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ.get('PATH', ''),
                   QWEN_FULL_CAMPAIGN='test', QWEN_FULL_MTP_TOKENS='3')
        process = subprocess.Popen(['bash', str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < deadline and process.poll() is None:
                time.sleep(.01)
            self.assertTrue(ready.exists(), 'fake local step failed to start')
            argv = json.loads(argv_path.read_text())
            self.assertEqual(argv[argv.index('--mtp-tokens')+1], '3')
            os.kill(process.pid, signal.SIGUSR1)
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 17, (stdout, stderr))
            self.assertTrue(marker.exists(), 'batch exited before step cleanup')
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            if ready.exists():
                try:
                    os.kill(int(ready.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == '__main__':
    unittest.main()
