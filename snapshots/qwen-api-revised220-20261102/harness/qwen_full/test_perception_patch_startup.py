"""CPU-only startup provenance checks; service processes and HTTP are mocked."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from perception_manifest import ESTIMATOR_PATCH_FILE, estimator_patch_for_profile
from perception_services import start_services


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PatchStartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        profile_relative = 'perception_profiles/test/perception'
        profile = self.root / profile_relative
        (profile / 'qwen_full').mkdir(parents=True)
        (profile / 'qwen_full/perception_service.py').write_text('# service fixture\n')
        helper = profile / ESTIMATOR_PATCH_FILE
        helper.write_text('# correction fixture\n')
        self.expected = estimator_patch_for_profile({ESTIMATOR_PATCH_FILE: digest(helper)})
        files = []
        for logical in ('sam3.pt', 'contact_graspnet/checkpoints/model.pt'):
            path = self.root / 'weights' / logical
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'CPU checkpoint fixture')
            files.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)})
        runtime_manifest = {'files': files, 'environments': {'sam3': {}, 'graspnet': {}}}
        manifest_path = self.root / 'perception/runtime_manifest.json'
        manifest_path.parent.mkdir()
        manifest_path.write_text(json.dumps(runtime_manifest))
        self.runtime = self.root / 'runtime'
        self.runtime.mkdir()
        self.campaign = NS(root=self.root, runtime=self.runtime, env={},
            gpus=[{'uuid': f'GPU-{index}'} for index in range(4)],
            manifest={'profiles': {'perception': {'relative_root': profile_relative}},
                      'perception_artifacts': {'runtime_manifest': runtime_manifest,
                          'runtime_manifest_sha256': digest(manifest_path),
                          'service_runner_sha256': digest(profile / 'qwen_full/perception_service.py'),
                          'estimator_patch': self.expected}},
            track=lambda kind, process: process, record=Mock(),
            stop=NS(is_set=lambda: False), report={})
        self.health = {}
        for kind, checkpoint in zip(('sam3', 'graspnet'), files):
            row = {'kind': kind, 'ok': True, 'loaded': True, 'model_loaded': True,
                   'offline_weights': True, 'device': 'cuda:0', 'parameter_devices': ['cuda:0'],
                   'parameter_count': 1, 'random_seed': 0,
                   'checkpoint_sha256': checkpoint['sha256']}
            if kind == 'sam3':
                row['files'] = {'sam3.pt': checkpoint}
            else:
                row['checkpoint'] = checkpoint
                row['estimator_patch'] = self.expected
            self.health[kind] = row

    def tearDown(self):
        self.temp.cleanup()

    def test_startup_validates_actual_patch_health_before_recording_ready(self):
        for variation in ('missing', 'tampered', 'matching'):
            with self.subTest(variation=variation):
                health = copy.deepcopy(self.health)
                if variation == 'missing':
                    health['graspnet'].pop('estimator_patch')
                elif variation == 'tampered':
                    health['graspnet']['estimator_patch']['patched_source_sha256'] = 'f' * 64
                opener = Mock()
                opener.open.side_effect = [io.BytesIO(json.dumps(health[kind]).encode())
                                           for kind in ('sam3', 'graspnet')]
                process = NS(pid=42, poll=lambda: None)
                self.campaign.record.reset_mock()
                with patch('perception_services.subprocess.check_output', return_value='{}'), \
                     patch('perception_services.subprocess.Popen', return_value=process), \
                     patch('perception_services.full.unused_loopback_port', return_value=12345), \
                     patch('perception_services.urllib.request.build_opener', return_value=opener):
                    if variation == 'matching':
                        config, processes = start_services(self.campaign)
                        self.assertEqual(len(processes), 2)
                        self.assertEqual(config['provenance']['service_health']['graspnet']['estimator_patch'],
                                         self.expected)
                    else:
                        with self.assertRaisesRegex(ValueError, 'estimator patch'):
                            start_services(self.campaign)
                        self.assertFalse(any(call.args[0] == 'perception_service_ready'
                                             and call.kwargs['kind'] == 'graspnet'
                                             for call in self.campaign.record.call_args_list))

    def test_missing_frozen_patch_declaration_fails_before_spawning(self):
        self.campaign.manifest['perception_artifacts'].pop('estimator_patch')
        with patch('perception_services.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'estimator patch'):
                start_services(self.campaign)
        spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
