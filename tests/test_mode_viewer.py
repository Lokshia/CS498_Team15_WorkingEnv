"""Mode-switch transactions without a renderer, server, CUDA, or real worlds."""
from pathlib import Path
from threading import RLock
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lifting_lab import mode_viewer as entry


def args_fixture():
    return SimpleNamespace(device='cpu', seed=2027, seconds=20., port=8081,
        mode='empty', mass=.5, empty_checkpoint=Path('empty/policy.pt'),
        box_checkpoint=Path('box/policy.pt'), output=Path('fixture-output'))


class SelectionTests(unittest.TestCase):
    def test_empty_is_exact_zero_even_if_box_control_retains_previous_weight(self):
        selection = entry.Selection('empty', .75)
        self.assertEqual(selection.mass_kg, 0.)
        self.assertEqual(selection.mode, 'empty')

    def test_box_requires_finite_positive_representable_mass(self):
        for value in (0, .0009, -1, 10.1, float('nan'), float('inf'), None, 'bad'):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                entry.Selection('box', value)
        for value in (.001, .5, 10):
            self.assertEqual(entry.Selection('box', value).mass_kg, value)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            entry.Selection('ghost', .5)


class ModeRequestTests(unittest.TestCase):
    def setUp(self):
        self.viewer = entry.ModeViewer.__new__(entry.ModeViewer)
        self.viewer._lifecycle_lock = RLock()
        self.viewer._closing = False
        self.viewer._pending_mass = None
        self.viewer._lockstep = None
        self.viewer._record_thread = None
        self.viewer._record_operations = 0
        self.viewer.requested_selection = None
        for name in ('_weight_input', '_mode_input', '_apply_mass_button', '_record_button'):
            setattr(self.viewer, name, SimpleNamespace(disabled=False))
        self.viewer._mass_notice = SimpleNamespace(content='')

    def test_empty_request_queues_true_zero_and_disables_mode_control(self):
        self.assertTrue(self.viewer.request_selection('empty', 1.))
        self.assertEqual(self.viewer.requested_selection, entry.Selection('empty'))
        self.assertEqual(self.viewer._pending_mass, 0.)
        self.assertTrue(self.viewer._mode_input.disabled)
        self.assertTrue(self.viewer._record_button.disabled)

    def test_pending_request_cannot_be_replaced_by_a_second_mode(self):
        self.assertTrue(self.viewer.request_selection('empty', 1.))
        self.assertFalse(self.viewer.request_selection('box', 2.))
        self.assertEqual(self.viewer.requested_selection, entry.Selection('empty'))
        self.assertEqual(self.viewer._pending_mass, 0.)

    def test_active_recording_blocks_mode_request_without_mutation(self):
        self.viewer._lockstep = object()
        self.assertFalse(self.viewer.request_selection('box', .5))
        self.assertIsNone(self.viewer.requested_selection)
        self.assertIsNone(self.viewer._pending_mass)
        self.assertFalse(self.viewer._mode_input.disabled)
        self.assertIn('结束录像', self.viewer._mass_notice.content)

    def test_recording_finalization_blocks_mode_request(self):
        self.viewer._record_operations = 1
        self.assertFalse(self.viewer.request_selection('empty', .5))
        self.assertIsNone(self.viewer.requested_selection)
        self.assertIsNone(self.viewer._pending_mass)

    def test_invalid_box_mass_does_not_queue_or_disable_controls(self):
        self.assertFalse(self.viewer.request_selection('box', 0))
        self.assertIsNone(self.viewer.requested_selection)
        self.assertIsNone(self.viewer._pending_mass)
        self.assertFalse(self.viewer._mode_input.disabled)


class PreparationTests(unittest.TestCase):
    def environment(self, mode, order):
        dimensions = (72, 76) if mode == 'empty' else (97, 97)
        obs = {'actor': torch.zeros(1, dimensions[0]), 'critic': torch.zeros(1, dimensions[1])}
        env = Mock()
        env.scene.entities = {'robot': object(), 'terrain': object()}
        if mode == 'box':
            env.scene.entities['payload'] = object()
        def reset(**kwargs):
            order.append(('reset', kwargs))
            return obs, {}
        env.reset.side_effect = reset
        return env, obs

    def test_empty_resets_after_policy_and_wrapper_and_checks_the_matching_schema(self):
        args = args_fixture()
        order = []
        env, obs = self.environment('empty', order)
        actor = Mock(return_value=torch.zeros(1, 22))
        def make_policy(*unused):
            order.append('policy')
            return actor
        wrapper = object()
        def make_wrapper(*unused, **kwargs):
            order.append(('wrapper', kwargs))
            return wrapper
        with patch.object(entry.empty_hand_cli, 'read_metadata'), \
             patch.object(entry.empty_hand_cli, 'environment', return_value=env), \
             patch.object(entry.empty_hand_cli, 'policy', side_effect=make_policy), \
             patch.object(entry, 'RslRlVecEnvWrapper', side_effect=make_wrapper):
            world = entry.prepare_selection(args, entry.Selection('empty', .5))
        self.assertEqual(order, ['policy', ('wrapper', {'clip_actions': 1.}), ('reset', {'seed': 2027})])
        self.assertIs(world.env, env)
        self.assertIs(world.wrapper, wrapper)
        self.assertEqual(world.mass['actual_payload_mass_kg'], 0.)
        actor.assert_called_once()
        self.assertIs(actor.call_args.args[0], obs)
        env.close.assert_not_called()

    def test_box_preparation_uses_box_model_and_mass(self):
        args = args_fixture()
        env, _ = self.environment('box', [])
        actor = Mock(return_value=torch.zeros(1, 22))
        mass = {'requested_payload_mass_kg': .5, 'actual_payload_mass_kg': .5}
        with patch.object(entry.cli, '_env', return_value=(env, mass)) as factory, \
             patch.object(entry.cli, '_policy', return_value=actor) as policy, \
             patch.object(entry, 'RslRlVecEnvWrapper', return_value=object()):
            world = entry.prepare_selection(args, entry.Selection('box', .5))
        passed = factory.call_args.args[0]
        self.assertEqual(passed.checkpoint, args.box_checkpoint)
        self.assertEqual(passed.mass, .5)
        self.assertTrue(passed.viewer)
        self.assertEqual(factory.call_args.kwargs, {'seconds': 21., 'auto_reset': False, 'num_envs': 1})
        self.assertIs(policy.call_args.args[0], env)
        self.assertEqual(world.checkpoint, args.box_checkpoint)
        self.assertIs(world.mass, mass)
        env.close.assert_not_called()

    def test_partial_empty_world_closes_on_policy_or_wrapper_failure(self):
        for failing_stage in ('policy', 'wrapper'):
            with self.subTest(stage=failing_stage):
                env, _ = self.environment('empty', [])
                with patch.object(entry.empty_hand_cli, 'read_metadata'), \
                     patch.object(entry.empty_hand_cli, 'environment', return_value=env), \
                     patch.object(entry.empty_hand_cli, 'policy') as policy, \
                     patch.object(entry, 'RslRlVecEnvWrapper') as wrapper:
                    failing = policy if failing_stage == 'policy' else wrapper
                    failing.side_effect = RuntimeError('fixture failure')
                    with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                        entry.prepare_selection(args_fixture(), entry.Selection('empty'))
                env.close.assert_called_once_with()

    def test_wrong_observation_shape_closes_the_created_world(self):
        env, obs = self.environment('empty', [])
        obs['actor'] = torch.zeros(1, 97)
        with patch.object(entry.empty_hand_cli, 'read_metadata'), \
             patch.object(entry.empty_hand_cli, 'environment', return_value=env), \
             patch.object(entry.empty_hand_cli, 'policy', return_value=Mock()), \
             patch.object(entry, 'RslRlVecEnvWrapper'):
            with self.assertRaises(AssertionError):
                entry.prepare_selection(args_fixture(), entry.Selection('empty'))
        env.close.assert_called_once_with()


class RunTransactionTests(unittest.TestCase):
    def run_scenario(self, fail_replacement):
        events, viewers = [], []
        args = args_fixture()
        class FakeWorld:
            def __init__(self, name, selection):
                self.name, self.selection = name, selection
                self.env = SimpleNamespace(device='cpu', step_dt=.02,
                    reset=lambda **kwargs: events.append(('reset', name, kwargs)))
                self.wrapper = name
                self.policy = lambda td: None
                self.mass = {'actual_payload_mass_kg': selection.mass_kg}
                self.close = Mock(side_effect=lambda: events.append(('world.close', name)))
            def snapshot(self, revision):
                return {'revision': revision, 'mode': self.selection.mode, 'paused': True}
        initial = FakeWorld('initial', entry.Selection('empty'))
        replacement = FakeWorld('replacement', entry.Selection('box', .5))
        def factory(_args, selection):
            events.append(('prepare', selection.mode))
            if len([e for e in events if e[0] == 'prepare']) == 1:
                return initial
            if fail_replacement:
                raise RuntimeError('fixture rebuild failure')
            return replacement
        class FakeViewer:
            def __init__(self, wrapper, **kwargs):
                self.wrapper, self.options = wrapper, kwargs
                self.requested_selection = entry.Selection('box', .5) if not viewers else None
                viewers.append(self)
                events.append(('viewer.new', wrapper))
            def pause(self):
                events.append(('viewer.pause', self.wrapper))
            def run(self, **kwargs):
                events.append(('viewer.run', self.wrapper))
                # MassAdjustableViewer.run guarantees worker close before return.
                events.append(('viewer.closed', self.wrapper))
        server = SimpleNamespace(
            gui=SimpleNamespace(reset=lambda: events.append(('gui.reset',))),
            scene=SimpleNamespace(reset=lambda: events.append(('scene.reset',))),
            stop=lambda: events.append(('server.stop',)))
        with patch.object(entry.empty_hand_cli, 'read_metadata'), \
             patch.object(entry.cli, '_validate_checkpoint_support', return_value={'mass': {'mass_range_kg': [.25, 1.]}}), \
             patch.object(entry.viser, 'ViserServer', return_value=server), \
             patch.object(entry, 'runtime_info', return_value={}), \
             patch.object(entry, 'write_json') as writes:
            entry.run(args, world_factory=factory, viewer_class=FakeViewer)
        return events, viewers, initial, replacement, writes

    def test_successful_rebuild_closes_viewer_then_old_world_then_resets_handles(self):
        events, viewers, old, new, writes = self.run_scenario(False)
        self.assertEqual([v.wrapper for v in viewers], ['initial', 'replacement'])
        old.close.assert_called_once_with()
        new.close.assert_called_once_with()
        self.assertLess(events.index(('viewer.closed', 'initial')), events.index(('prepare', 'box')))
        self.assertLess(events.index(('prepare', 'box')), events.index(('world.close', 'initial')))
        self.assertLess(events.index(('world.close', 'initial')), events.index(('gui.reset',)))
        self.assertLess(events.index(('scene.reset',)), events.index(('viewer.new', 'replacement')))
        self.assertIn(('viewer.pause', 'replacement'), events)
        self.assertIn('已应用并暂停', viewers[1].options['notice'])
        self.assertEqual(writes.call_count, 4)

    def test_failed_rebuild_keeps_old_world_resets_seed_and_reopens_paused(self):
        events, viewers, old, new, _ = self.run_scenario(True)
        self.assertEqual([v.wrapper for v in viewers], ['initial', 'initial'])
        old.close.assert_called_once_with()  # Final shutdown only.
        new.close.assert_not_called()
        self.assertIn(('reset', 'initial', {'seed': 2027}), events)
        self.assertEqual(events.count(('viewer.pause', 'initial')), 2)
        self.assertIn('已保留原场景', viewers[1].options['notice'])
        self.assertEqual(viewers[1].options['selection'].mode, 'empty')
        self.assertLess(events.index(('reset', 'initial', {'seed': 2027})), events.index(('gui.reset',)))
        self.assertLess(events.index(('server.stop',)), events.index(('world.close', 'initial')))


if __name__ == '__main__':
    unittest.main()
