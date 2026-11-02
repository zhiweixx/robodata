"""Offline continuation accounting: no model request and no credential access."""

from decimal import Decimal
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from run_suite import AUTHORIZED_TASKS, audit_prior_run, claim_continuation, lock_prior_ledger, verify_prior_unchanged


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ledger = self.root / 'usage.jsonl'
        self.episode = 'general_pickup_s0_l0'
        self.rows = [{'event': 'suite_initialized', 'config': {
            'model': 'gpt-6-astra', 'purpose': 'aspire_robodojo_pilot',
            'episode_ids': [task + '_s0_l0' for task in AUTHORIZED_TASKS],
            'suite_budget_usd': '50', 'max_calls_per_episode': 8,
            'request_reservation_usd': '2.0096',
        }}]
        for index in range(2):
            common = {'episode_id': self.episode, 'attempt_id': str(index)}
            self.rows.extend([
                dict(common, event='token_count_started'),
                dict(common, event='token_count_received'),
                dict(common, event='request_started', reserved_usd='2.0096'),
                dict(common, event='response_received', model='gpt-6-astra', service_tier='default',
                     usage_raw={'input_tokens': 1000, 'output_tokens': 1000, 'total_tokens': 2000,
                                'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 1000}}),
                dict(common, event='request_settled', charged_usd='0.0625',
                     episode_charged_usd=str(Decimal('0.0625') * (index + 1)),
                     suite_charged_usd=str(Decimal('0.0625') * (index + 1))),
            ])
        report_dir = self.root / 'general_pickup'
        report_dir.mkdir()
        (report_dir / 'episode_report.json').write_text(json.dumps({
            'ok': True, 'episode_id': self.episode, 'task': 'general_pickup',
            'seed': 0, 'layout_id': 0, 'model': 'gpt-6-astra',
            'iterations': [{'iteration': 1, 'done': False}, {'iteration': 2, 'done': True}],
            'total_programs_executed': 1,
            'official': {'official_success': False, 'environment_steps': 2, 'step_limit': 200},
        }))
        self.write()

    def write(self):
        self.ledger.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))

    def audit(self, tasks=('stack_bowls', 'align_blocks')):
        return audit_prior_run(self.ledger, tasks)

    def test_remaining_budget_and_prior_failure_preserved(self):
        original = self.ledger.read_bytes()
        audit = self.audit()
        self.assertEqual(audit['remaining_budget_usd'], '49.8750')
        self.assertEqual(audit['prior_calls'][self.episode], 2)
        self.assertFalse(audit['prior_episodes'][0]['official_success'])
        self.assertEqual(self.ledger.read_bytes(), original)
        self.assertEqual(audit['prior_ledger_sha256'], hashlib.sha256(original).hexdigest())
        verify_prior_unchanged(audit)

    def test_schema_validation_does_not_make_integration_failure_valid(self):
        (self.root / 'integration_diagnosis.json').write_text(json.dumps({
            'general_pickup_measurement_valid': False,
            'reason': 'Adapter rejected valid Cartesian plans before execution',
        }))
        audit = self.audit()
        self.assertTrue(audit['prior_episodes'][0]['schema_ledger_validated'])
        self.assertFalse(audit['prior_episodes'][0]['measurement_valid'])
        verify_prior_unchanged(audit)

    def test_attempted_task_can_never_be_replayed(self):
        with self.assertRaises(RuntimeError):
            self.audit(AUTHORIZED_TASKS)

    def test_unsettled_request_prohibits_continuation(self):
        self.rows.append({'event': 'request_started', 'attempt_id': 'unknown',
                          'episode_id': self.episode, 'reserved_usd': '2.0096'})
        self.write()
        with self.assertRaisesRegex(RuntimeError, 'unknown billing'):
            self.audit()

    def test_underreported_charge_is_rejected(self):
        self.rows[-1]['charged_usd'] = '0'
        self.write()
        with self.assertRaisesRegex(RuntimeError, 'reconcile'):
            self.audit()

    def test_original_ledger_change_is_rejected(self):
        audit = self.audit()
        self.rows.append({'event': 'program_input'})
        self.write()
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            verify_prior_unchanged(audit)

    def test_read_only_lock_excludes_another_writer(self):
        import fcntl
        import os
        path = Path(str(self.ledger) + '.lock')
        path.touch(mode=0o600)
        fd = lock_prior_ledger(self.ledger)
        other = os.open(path, os.O_RDWR)
        try:
            with self.assertRaises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(other)
            os.close(fd)

    def test_continuation_claim_cannot_be_reused_with_new_run_id(self):
        audit = self.audit()
        original = self.ledger.read_bytes()
        claim = claim_continuation(audit, self.root / 'first_continuation', audit['remaining_tasks'])
        with self.assertRaisesRegex(RuntimeError, 'already claimed'):
            claim_continuation(audit, self.root / 'second_continuation', audit['remaining_tasks'])
        recorded = json.loads(Path(claim['path']).read_text())
        self.assertEqual(recorded['continuation_run_id'], 'first_continuation')
        self.assertEqual(recorded['selected_tasks'], ['stack_bowls', 'align_blocks'])
        self.assertEqual(self.ledger.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
