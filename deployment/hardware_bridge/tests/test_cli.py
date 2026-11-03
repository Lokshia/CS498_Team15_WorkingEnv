"""CLI preflight guarantees in subprocesses that forbid all ROS imports."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
BRIDGE=HERE.parent
sys.path.insert(0,str(BRIDGE)); sys.path.insert(0,str(HERE))
from fixtures import make_config, load_contract


class CliTests(unittest.TestCase):
    def run_cli(self, config, *args):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            # Even accidental ROS import fails before any network could open.
            for name in ('rclpy','booster_interface'):
                (root/(name+'.py')).write_text("raise AssertionError('ROS_IMPORT_ATTEMPT')\n",encoding='utf-8')
            profile=root/'profile.json'; profile.write_text(json.dumps(config),encoding='utf-8')
            env=os.environ.copy(); env['PYTHONPATH']=str(root)
            output=root/'must_not_be_created'
            result=subprocess.run([sys.executable,str(BRIDGE/'run.py'),'--config',str(profile),
                                   '--output',str(output),*args],capture_output=True,text=True,
                                  encoding='utf-8',env=env,timeout=20,cwd=root)
            self.assertNotIn('ROS_IMPORT_ATTEMPT',result.stdout+result.stderr)
            self.assertFalse(output.exists(),'Preflight/rejected launch must not create a session directory')
            return result

    def test_incomplete_site_profile_rejected_before_ros(self):
        profile=make_config(load_contract())
        profile['review']['calibration_verified']=None
        profile['joints'][0]['kp']=None
        result=self.run_cli(profile,'--live')
        self.assertEqual(result.returncode,1)
        self.assertIn('review.calibration_verified',result.stderr)
        self.assertIn('joints[0].kp',result.stderr)
        self.assertNotIn('preflight": "passed',result.stdout)

    def test_simulation_fixture_cannot_enter_live(self):
        result=self.run_cli(make_config(load_contract()),'--live')
        self.assertEqual(result.returncode,1)
        self.assertIn('Simulation fixture cannot be used with --live',result.stderr)

    def test_default_is_offline_preflight_with_all_actor_vectors(self):
        result=self.run_cli(make_config(load_contract()))
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout)
        self.assertEqual(report['preflight'],'passed')
        self.assertEqual(report['test_vectors'],534)
        self.assertFalse(report['hardware_validated'])
        self.assertLess(report['max_abs_error'],2e-5)

    def test_invalid_shadow_duration_rejected_before_ros(self):
        for value in ('nan','0','-1','601'):
            with self.subTest(value=value):
                result=self.run_cli(make_config(load_contract()),'--shadow','--seconds',value)
                self.assertEqual(result.returncode,1)
                self.assertIn('--seconds must be in (0,600]',result.stderr)


if __name__=='__main__': unittest.main()
