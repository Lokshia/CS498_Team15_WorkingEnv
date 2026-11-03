"""Frozen deployment policy envelope is separate from physical joint limits."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

HERE=Path(__file__).resolve().parent
BRIDGE=HERE.parent
sys.path.insert(0,str(BRIDGE)); sys.path.insert(0,str(HERE))
from fixtures import make_config, load_contract
from team15_hardware.core import ControlCore, VALIDATED_POLICY_RANGES
from team15_hardware.ros_transport import StateSnapshot


class PolicyEnvelopeTests(unittest.TestCase):
    def test_envelope_exactly_matches_frozen_canonical_model_audit(self):
        report=BRIDGE.parent/'team15_proprio_package/reports/deploy_0p5kg/report.json'
        audit=json.loads(report.read_text(encoding='utf-8'))['model_audit']
        contract=load_contract()
        self.assertEqual(audit['joint_names'],contract['joint_mapping']['booster_deploy_joint_names'])
        np.testing.assert_array_equal(np.asarray(VALIDATED_POLICY_RANGES),np.asarray(audit['joint_ranges_rad']))

    def policy_target(self, config, action):
        core=ControlCore(config,load_contract())
        snapshot=StateSnapshot(tuple(core.nominal),(0.,)*22,(0.,)*22,(0.,)*3,(0.,)*3,100.,1)
        core.commit_sent(core.prime(snapshot,100.),100.)
        newer=StateSnapshot(tuple(core.nominal),(0.,)*22,(0.,)*22,(0.,)*3,(0.,)*3,100.02,2)
        return core,core.policy_command(action,newer,100.02)

    def test_broad_site_limits_never_expand_validated_actor_envelope(self):
        config=make_config(load_contract())
        for row in config['joints']: row['target_rate_limit_rad_s']=100.
        for value in (-1.,1.):
            core,command=self.policy_target(config,np.full(22,value))
            envelope=np.asarray(VALIDATED_POLICY_RANGES)
            self.assertTrue(np.all(command.model_q>=envelope[:,0]))
            self.assertTrue(np.all(command.model_q<=envelope[:,1]))
            # Prepare/measurement limits still reflect the independent site profile.
            np.testing.assert_array_equal(core.lower,np.full(22,-3.))

    def test_narrower_site_limits_also_bound_actor_and_previous_action(self):
        config=make_config(load_contract())
        for row in config['joints']: row['target_rate_limit_rad_s']=100.
        config['joints'][0]['model_max_rad']=.001
        core,command=self.policy_target(config,np.ones(22))
        self.assertAlmostEqual(command.model_q[0],.001)
        core.commit_sent(command,100.02)
        self.assertAlmostEqual(core.previous_action[0],.001/core.scale[0])


if __name__=='__main__': unittest.main()
