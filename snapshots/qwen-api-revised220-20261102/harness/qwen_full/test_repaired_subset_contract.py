"""CPU-only contract regressions: fresh767 identity, unchanged policy, frozen code."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from build_subset_manifest import canonical_sha256
from paired_manifest import REQUIRED_PROFILE_FILES
import repaired_subset_contract as contract


class ContractTests(unittest.TestCase):
    BASELINE_COMMIT = '871bbc0d0e510e9b3a20298b3ddaf3ed6d76d2f9'

    @classmethod
    def setUpClass(cls):
        # This is a historical contract test, not a test of the revised working
        # harness. Read immutable Git objects and verify every pinned digest;
        # copying today's source would silently couple this fixture to v2.
        cls.repository = Path(__file__).resolve().parents[1]
        homebrew_git = Path('/opt/homebrew/bin/git')
        git = str(homebrew_git) if homebrew_git.is_file() else shutil.which('git')
        if git is None:
            raise RuntimeError('Git is required to retrieve the historical source fixture')
        cls.historical_sources = {}
        for name, expected in contract.PINNED_SOURCE_HASHES.items():
            data = subprocess.run(
                [git, 'show', cls.BASELINE_COMMIT + ':' + name],
                cwd=cls.repository, check=True, capture_output=True,
            ).stdout
            if hashlib.sha256(data).hexdigest() != expected:
                raise AssertionError('historical fixture differs from pinned contract: ' + name)
            cls.historical_sources[name] = data

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve() / 'aspire_validate'
        self.campaign = 'qwen38_subset20_repaired_test'
        self.profile = self.root / 'harness_profiles' / self.campaign
        names = set(contract.REQUIRED_BOOTSTRAP) | set(contract.PINNED_SOURCE_HASHES) | set(REQUIRED_PROFILE_FILES)
        for name in names:
            path = self.profile / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name in contract.PINNED_SOURCE_HASHES:
                path.write_bytes(self.historical_sources[name])
            else:
                path.write_text('# source fixture\n')
        runtime = self.root / 'perception/runtime_manifest.json'
        runtime.parent.mkdir()
        runtime.write_text(json.dumps({'files': [], 'environments': {}}))
        for name in contract.WRAPPERS:
            path = self.root.parent / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('# environment wrapper\n')
        output = self.root / 'full_results' / self.campaign
        output.mkdir(parents=True)
        shutil.copyfile(Path(__file__).with_name('subset_manifest.json'), output / 'manifest.json')
        self.value = contract.build_contract(self.root, self.campaign, self.BASELINE_COMMIT)
        (output / 'execution_manifest.json').write_text(json.dumps(self.value))

    def tearDown(self):
        self.temp.cleanup()

    def test_fixed_dataset_and_all_fixes_have_independent_new_identity(self):
        actual = contract.load_contract(self.root, self.campaign)
        self.assertEqual(actual['target_episodes'], 767)
        self.assertEqual(actual['fixes'], list(range(1, 7)))
        self.assertEqual(actual['hardware']['simulator_gpus'], [0])
        self.assertFalse(actual['reuse_historical_scores'])
        self.assertEqual(actual['inference']['mtp_tokens'], 3)
        self.assertEqual(contract.verify_profile(self.root, actual)['profile_root'], str(self.profile))

    def test_policy_budget_provider_and_identity_changes_rejected(self):
        for key, value in [('model', {}), ('sampling', {}), ('thinking', {}), ('inference', {}),
                           ('limits', {}), ('providers', {}), ('hardware', {}),
                           ('target_episodes', 6300), ('reuse_historical_scores', True),
                           ('fixes', [1, 3, 5, 6]), ('dataset_manifest_sha256', '0' * 64),
                           ('campaign', 'qwen38_subset20_20261026')]:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.value); changed[key] = value
                with self.assertRaises(ValueError):
                    contract.validate_contract(changed)

    def test_execution_must_use_correct_path_even_if_bytes_match(self):
        name = 'qwen_full/repaired_subset_supervisor.py'
        evidence = contract.verify_executed_sources(self.root, self.value, {name: self.profile / name})
        self.assertEqual(evidence['executed_sources'][name]['sha256'],
                         self.value['source_profile']['files_sha256'][name])
        outside = self.root / name
        outside.parent.mkdir(); shutil.copyfile(self.profile / name, outside)
        with self.assertRaisesRegex(ValueError, 'outside'):
            contract.verify_executed_sources(self.root, self.value, {name: outside})

    def test_profile_or_wrapper_drift_rejected(self):
        file = self.profile / 'qwen_full/repaired_subset_entry.py'
        old = file.read_bytes(); file.write_text('# drift\n')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            contract.verify_profile(self.root, self.value)
        file.write_bytes(old)
        (self.root.parent / contract.WRAPPERS[0]).write_text('# changed wrapper\n')
        with self.assertRaisesRegex(ValueError, 'wrapper'):
            contract.verify_profile(self.root, self.value)

    def test_manifest_cannot_claim_repair_on_different_source(self):
        changed = copy.deepcopy(self.value)
        files = changed['source_profile']['files_sha256']
        files['ASPIRE/src/cap_harness/api.py'] = '0' * 64
        changed['source_profile']['files_manifest_sha256'] = canonical_sha256(files)
        with self.assertRaisesRegex(ValueError, 'repair'):
            contract.validate_contract(changed)

    def test_revised_working_episode_cannot_be_relabelled_as_historical_repair(self):
        name = 'qwen_full/full_episode.py'
        current = (self.repository / name).read_bytes()
        self.assertNotEqual(hashlib.sha256(current).hexdigest(), contract.PINNED_SOURCE_HASHES[name])
        (self.profile / name).write_bytes(current)
        with self.assertRaisesRegex(ValueError, 'repair'):
            contract.build_contract(self.root, self.campaign, self.BASELINE_COMMIT)

    def test_dataset_changes_are_not_silently_reselected(self):
        path = self.root / 'full_results' / self.campaign / 'manifest.json'
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'candidate queue'):
            contract.load_contract(self.root, self.campaign)


if __name__ == '__main__':
    unittest.main()
