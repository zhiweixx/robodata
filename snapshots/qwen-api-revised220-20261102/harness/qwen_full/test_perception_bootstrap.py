"""CPU regression tests for frozen bootstrap execution and saved evidence auditing."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audit_perception_results import allocation_model_evidence
from build_subset_manifest import canonical_sha256
from perception_manifest import (BOOTSTRAP_FILES, BOOTSTRAP_ROLES, HARDWARE,
    LEGACY_PROTOCOL, PINNED_MODEL, PROTOCOL, bootstrap_record, bootstrap_source_hashes,
    sha_file, validate_bootstrap_evidence, verify_bootstrap)


SUPERVISOR_ROLES = ('supervisor', 'manifest', 'full_supervisor', 'paired_supervisor', 'perception_services')


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        relative = 'perception_profiles/qwen38_perception_fixture/perception'
        self.profile = self.root / relative
        files = {}
        for name in BOOTSTRAP_FILES + ('qwen_full/full_model_smoke.py',):
            path = self.profile / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# pinned: ' + name + '\n')
            files[name] = sha_file(path)
        profile = {'relative_root': relative, 'files_sha256': files,
                   'files_manifest_sha256': canonical_sha256(files)}
        self.manifest = {'protocol': PROTOCOL, 'profiles': {'perception': profile,
            'baseline': {'files_manifest_sha256': 'a' * 64}}}
        self.manifest['bootstrap'] = bootstrap_record(self.manifest['profiles'])
        self.executed = {role: self.profile / BOOTSTRAP_ROLES[role] for role in SUPERVISOR_ROLES}

    def tearDown(self):
        self.temp.cleanup()

    def evidence(self, roles=SUPERVISOR_ROLES):
        return verify_bootstrap(self.root, self.manifest,
            {role: self.profile / BOOTSTRAP_ROLES[role] for role in roles})

    def allocation(self):
        pinned = {**PINNED_MODEL, 'snapshot_path': '/weights/pinned-model'}
        return {'allocation': '1234', 'model_manifest': pinned,
            'actual_inference_variant': 'bf16_mtp3', 'context_length': 262144,
            'inference_variant_evidence': {'variant': 'bf16_mtp3'},
            'server_command': ['python', 'vllm', '--dtype', 'bfloat16',
                '--tensor-parallel-size', '2', '--max-model-len', '262144',
                '--max-num-seqs', '2', '--speculative-config',
                json.dumps({'method': 'mtp', 'num_speculative_tokens': 3}),
                '--model', pinned['snapshot_path']],
            'execution_hardware': HARDWARE,
            'profile_verification': {arm: {'files_manifest_sha256': profile['files_manifest_sha256']}
                for arm, profile in self.manifest['profiles'].items()},
            'gpus': [{'uuid': 'GPU-' + str(index)} for index in range(4)],
            'source_sha256': bootstrap_source_hashes(self.manifest),
            'batch_script_sha256': self.manifest['bootstrap']['files_sha256']['qwen_full/perception_eval.sbatch'],
            'bootstrap_verification': self.evidence()}

    def audit_allocation(self, allocation, manifest=None):
        path = self.root / 'allocation.json'
        path.write_text(json.dumps(allocation))
        # Service health and inference identity have independent regression tests;
        # this fixture isolates the new bootstrap integrity checks.
        with patch('audit_perception_results.service_provenance', return_value={'gpu_uuid': 'GPU-3'}):
            return allocation_model_evidence(path, manifest or self.manifest)

    def test_actual_entry_and_shell_must_run_from_designated_frozen_profile(self):
        roles = ('entry', 'inside', 'manifest')
        evidence = self.evidence(roles)
        self.assertTrue(validate_bootstrap_evidence(evidence, self.manifest, roles=roles)['verified'])
        mutable = self.root / 'qwen_full/perception_entry.py'
        mutable.parent.mkdir()
        mutable.write_bytes((self.profile / 'qwen_full/perception_entry.py').read_bytes())
        executed = {role: self.profile / BOOTSTRAP_ROLES[role] for role in roles}
        executed['entry'] = mutable
        with self.assertRaisesRegex(ValueError, 'outside'):
            verify_bootstrap(self.root, self.manifest, executed)

    def test_tampered_frozen_launcher_is_rejected_before_execution(self):
        (self.profile / 'qwen_full/perception_inside.sh').write_text('# modified launcher\n')
        with self.assertRaisesRegex(ValueError, 'pinned bytes'):
            self.evidence()

    def test_valid_allocation_checks_source_inventory_and_bootstrap(self):
        result = self.audit_allocation(self.allocation())
        self.assertTrue(result['bootstrap_freeze']['verified'])
        self.assertIn('full_model_smoke.py', bootstrap_source_hashes(self.manifest))

    def test_missing_and_tampered_source_hashes_now_fail_the_audit(self):
        for change in (lambda r: r.pop('source_sha256'),
                       lambda r: r['source_sha256'].update(perception_entry='0' * 64),
                       lambda r: r['source_sha256'].update({'perception_entry.py': '0' * 64}),
                       lambda r: r['source_sha256'].update({'perception_inside.sh': '1' * 64}),
                       lambda r: r['source_sha256'].update({'full_model_smoke.py': '2' * 64}),
                       lambda r: r.pop('bootstrap_verification'),
                       lambda r: r.update(batch_script_sha256='0' * 64),
                       lambda r: r.pop('batch_script_sha256')):
            allocation = self.allocation()
            change(allocation)
            with self.assertRaisesRegex(ValueError, 'bootstrap'):
                self.audit_allocation(allocation)

    def test_missing_and_changed_executed_paths_or_hashes_fail(self):
        for change in (lambda e: e['executed_sources'].pop('entry'),
                       lambda e: e.update(executed_sources=None),
                       lambda e: e.update(root=None),
                       lambda e: e['executed_sources']['entry'].update(path='/work/aspire_validate/qwen_full/perception_entry.py'),
                       lambda e: e['executed_sources']['inside'].update(sha256='0' * 64),
                       lambda e: e.update(root='/work/aspire_validate/qwen_full')):
            evidence = self.evidence(('entry', 'inside', 'manifest'))
            change(evidence)
            with self.assertRaisesRegex(ValueError, 'bootstrap'):
                validate_bootstrap_evidence(evidence, self.manifest, roles=('entry', 'inside', 'manifest'))

    def test_legacy_audit_explicitly_does_not_claim_bootstrap_verified(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['protocol'] = LEGACY_PROTOCOL
        manifest.pop('bootstrap')
        allocation = self.allocation()
        allocation.pop('bootstrap_verification')
        allocation.pop('batch_script_sha256')
        allocation['source_sha256'] = {'perception_entry.py': '0' * 64, 'perception_inside.sh': '1' * 64}
        result = self.audit_allocation(allocation, manifest)
        self.assertFalse(result['bootstrap_freeze']['verified'])
        self.assertIn('Legacy', result['bootstrap_freeze']['reason'])


if __name__ == '__main__':
    unittest.main()
