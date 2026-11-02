"""CPU checks for paired execution isolation and artifact drift rejection."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from perception_supervisor import PerceptionCampaign
from perception_services import verify_artifacts


class ExecutionTests(unittest.TestCase):
    def test_resume_preserves_scored_failure_and_alternates_next_pair(self):
        campaign = object.__new__(PerceptionCampaign)
        campaign.manifest = {'units': [
            {'unit_id': 'p000', 'group': {'group_id': 'task_s0'}, 'arm_order': ['baseline', 'perception']},
            {'unit_id': 'p001', 'group': {'group_id': 'task_s0'}, 'arm_order': ['perception', 'baseline']}]}
        campaign.arm_dirs = {(arm, unit['unit_id']): Path('/different') / (arm + unit['unit_id'])
            for unit in campaign.manifest['units'] for arm in unit['arm_order']}
        campaign.simroot, campaign.completed = Path('/sim'), {}
        failure = {'scored': 1, 'success': 0, 'group_id': 'task_s0'}
        def pending(directory, *_):
            return failure if directory.name == 'baselinep000' else None
        with patch('perception_supervisor.pending_group', side_effect=pending):
            work = campaign.collect_pending()
        self.assertEqual([(u['unit_id'], a) for u, a in work],
                         [('p000', 'perception'), ('p001', 'perception'), ('p001', 'baseline')])
        self.assertEqual(campaign.completed['baseline__p000']['success'], 0)
        self.assertEqual(len(set(campaign.arm_dirs.values())), 4)

    def test_changed_weights_rejected_before_service_start(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'model.pt'
            checkpoint.write_bytes(b'known snapshot')
            manifest = Path(temp) / 'runtime.json'
            manifest.write_text(json.dumps({'files': [{'path': str(checkpoint),
                'bytes': checkpoint.stat().st_size,
                'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}]}))
            verify_artifacts(manifest)
            checkpoint.write_bytes(b'changed snapshot')
            with self.assertRaisesRegex(RuntimeError, 'artifact drift'):
                verify_artifacts(manifest)


if __name__ == '__main__':
    unittest.main()
