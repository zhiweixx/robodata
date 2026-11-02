"""Numerical contract tests, not a simulator or task-success measurement."""
import unittest
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from native_retime import retime_native_path


class Tests(unittest.TestCase):
    def test_peak_speed_after_long_dwell_is_bounded_without_global_slowdown(self):
        path=np.zeros((101,6));path[-2:,0]=[.5,1.]
        # The previous total-variation / .045 count plus uniform-index sampling
        # takes25ticks but jumps by1rad at the end, despite average being.04.
        old_count=max(25,int(np.ceil(1/.045)))
        old=np.interp(np.linspace(0,1,old_count+1)[1:],np.linspace(0,1,len(path)),path[:,0])
        self.assertGreater(np.max(np.diff(np.r_[0.,old])),.05)
        sampled,diag=retime_native_path(path,np.zeros(6),native_dt_s=.01,control_dt_s=.04)
        self.assertLessEqual(diag['max_tick_delta_rad'],.045+1e-9)
        self.assertLess(len(sampled),50)
        self.assertGreaterEqual(diag['retimed_duration_s'],1.)
        np.testing.assert_array_equal(sampled[-1],path[-1])

    def test_reversals_and_numerical_start_connector_obey_same_tick_bound(self):
        rng=np.random.default_rng(7)
        path=np.cumsum(rng.normal(0,.025,(300,6)),axis=0)
        path-=path[0];path+=1e-5
        sampled,diag=retime_native_path(path,np.zeros(6),native_dt_s=.004,control_dt_s=.04)
        self.assertLessEqual(diag['max_tick_delta_rad'],.045+1e-9)
        self.assertGreater(diag['numerical_start_connector_s'],0.)
        np.testing.assert_allclose(sampled[-1],path[-1],atol=1e-12)

    def test_discontinuous_native_start_rejected_not_interpolated_away(self):
        path=np.ones((10,6))*.1
        with self.assertRaisesRegex(ValueError,'native path start differs'):
            retime_native_path(path,np.zeros(6),native_dt_s=.004,control_dt_s=.04)

    def test_diagnostic_records_native_paths_and_rejects_wrong_actual_pose(self):
        from verify_adapter_offline import Env
        from retime_diagnostic import run_retime_diagnostic
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'ASPIRE_OUTPUT_DIR':temp}):
            report=run_retime_diagnostic(Env())
            # Fake kinematics only implement Z; allplans exist but wrongXY and
            # orientation must fail the measured endpoint regression.
            self.assertFalse(report['ok'])
            self.assertIn('overhead execution validation failed',report['program_error'])
            self.assertEqual(len(report['native_plan_audits']),3)
            self.assertEqual(len(list(Path(temp).glob('native_plan_*.npz'))),3)
            self.assertEqual(report['model_requests'],0)


if __name__=='__main__':
    unittest.main()
