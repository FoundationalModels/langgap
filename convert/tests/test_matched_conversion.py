"""Check the batch conversion preserves the full task/configuration pairing."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert_matched_libero_goal as job


class MatchedConversionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scenes = [SimpleNamespace(scene_id=f'scene_{i:06d}', scene_hash=f'hash{i}')
                       for i in range(150)]
        self.names = {task: job.EXTENDED_TASK_MAP[task]['bddl'] + '.bddl'
                      for task in job.MATCHED_GOAL_TASK_IDS}
        for task in job.MATCHED_GOAL_TASK_IDS:
            (self.root / f'task{task}.hdf5').touch()
        self.archive = self.root / 'matched_demos.hdf5'
        self.archive.touch()
        loader = patch.object(job, 'load_matched_scenes',
                              return_value=(self.archive, self.scenes, self.names))
        self.load = loader.start()
        self.addCleanup(loader.stop)

    def fake_conversion(self, command, check):
        self.assertTrue(check)
        self.assertNotIn('--push', command)
        self.assertEqual(command[command.index('--frame_skip') + 1], '1')
        self.assertNotIn('--trim_zeros', command)
        inputs = command[command.index('--input') + 1:command.index('--bddl')]
        self.assertEqual(len(inputs), 9)
        out = self.root / 'lerobot'
        (out / 'meta').mkdir(parents=True)
        cameras = ('observation.images.image', 'observation.images.image2')
        (out / 'meta/info.json').write_text(json.dumps({
            'total_episodes': 1350, 'total_frames': 2700,
            'features': {key: {'dtype': 'video', 'shape': [256, 256, 3]} for key in cameras}}))
        records = [{'bddl_file': self.names[task], 'scene_id': scene.scene_id,
                    'scene_hash': scene.scene_hash}
                   for task in job.MATCHED_GOAL_TASK_IDS for scene in self.scenes]
        if self.corrupt_pairing:
            records[-1] = records[-2].copy()  # Keep the count, lose a configuration.
        (out / 'meta/matched_episodes.json').write_text(json.dumps({'episodes': records}))
        (out / 'meta/matched_demos.hdf5').touch()
        for camera in cameras:
            path = out / 'videos' / camera / 'chunk-000/file-000.mp4'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'test-video-placeholder')

    def test_all_150_configurations_for_every_task(self):
        self.corrupt_pairing = False
        with patch.object(job.subprocess, 'run', side_effect=self.fake_conversion):
            output = job.convert_run(self.root, 'test/matched')
        self.load.assert_called_once_with(self.root, 150)
        metadata = json.loads((output / 'meta/image_convention.json').read_text())
        self.assertEqual(metadata['raw_simulator_to_dataset'], 'rotate_180')
        self.assertEqual(metadata['episodes_per_task'], 150)

    def test_same_episode_count_with_missing_configuration_is_rejected(self):
        self.corrupt_pairing = True
        with patch.object(job.subprocess, 'run', side_effect=self.fake_conversion):
            with self.assertRaisesRegex(ValueError, 'each matched configuration once per task'):
                job.convert_run(self.root, 'test/matched')


if __name__ == '__main__':
    unittest.main()
