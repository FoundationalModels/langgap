"""Run with python -m unittest discover -s langgap/eval/tests."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matched_scenes import (MATCHED_GOAL_TASK_IDS, SceneSnapshot, initialize_episode,
                            load_matched_scenes)
from matched_scene import array_hash


class MatchedScenesTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'matched_demos.hdf5'

    def write_archive(self, count):
        with h5py.File(self.path, 'w') as archive:
            archive.attrs['schema_version'] = 1
            root = archive.create_group('scenes')
            for i in range(count):
                arrays = {'state': np.array([float(i), 1., 2.])}
                scene = SceneSnapshot(f'scene_{i:06d}', arrays, {},
                                      array_hash(arrays, {}), i, i)
                group = root.create_group(scene.scene_id)
                group.attrs['complete'] = True
                scene.save(group.create_group('initial_scene'))
                tasks = group.create_group('tasks')
                for task in MATCHED_GOAL_TASK_IDS:
                    episode = tasks.create_group(str(task))
                    episode.create_dataset('states', data=[arrays['state']])
                    episode.attrs.update(scene_id=scene.scene_id, scene_hash=scene.scene_hash,
                                         bddl_file=f'/other/machine/task{task}.bddl')

    def test_all_150_in_order_and_portable_paths(self):
        self.write_archive(150)
        path, scenes, names = load_matched_scenes(self.path.parent)
        self.assertEqual(path, self.path)
        self.assertEqual([s.init_index for s in scenes], list(range(150)))
        self.assertEqual(names[56], 'task56.bddl')

    def test_wrong_count_rejected(self):
        self.write_archive(1)
        with self.assertRaisesRegex(ValueError, 'Expected 150'):
            load_matched_scenes(self.path)

    def test_incomplete_bundle_rejected(self):
        self.write_archive(1)
        with h5py.File(self.path, 'a') as f:
            f['scenes/scene_000000'].attrs['complete'] = False
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            load_matched_scenes(self.path, 1)

    def test_training_state_mismatch_rejected(self):
        self.write_archive(1)
        with h5py.File(self.path, 'a') as f:
            f['scenes/scene_000000/tasks/56/states'][0, 1] = 999
        with self.assertRaisesRegex(ValueError, 'training initial scene mismatch'):
            load_matched_scenes(self.path, 1)

    def test_corrupt_snapshot_rejected(self):
        self.write_archive(1)
        with h5py.File(self.path, 'a') as f:
            f['scenes/scene_000000/initial_scene/state'][0] = 999
        with self.assertRaisesRegex(ValueError, 'Corrupt scene'):
            load_matched_scenes(self.path, 1)

    def test_matched_restore_has_no_reset_or_settling_afterward(self):
        env, scene = Mock(), Mock()
        obs = initialize_episode(env, None, 149, scene)
        self.assertIs(obs, scene.restore.return_value)
        scene.restore.assert_called_once_with(env.env, verify_images=False)
        env.reset.assert_not_called()
        env.step.assert_not_called()
        env.set_init_state.assert_not_called()

    def test_legacy_reset_precedes_state_restore(self):
        env = Mock()
        env.step.return_value = ('obs', 0, False, {})
        self.assertEqual(initialize_episode(env, ['saved'], 0), 'obs')
        self.assertEqual([c[0] for c in env.mock_calls],
                         ['reset', 'set_init_state'] + ['step'] * 10)


if __name__ == '__main__':
    unittest.main()
