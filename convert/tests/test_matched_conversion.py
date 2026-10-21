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
        self.task_ids = job.MATCHED_GOAL_TASK_IDS
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
        self.assertEqual(len(inputs), len(self.task_ids))
        out = self.root / 'lerobot'
        (out / 'meta').mkdir(parents=True)
        standard = '--standard-libero' in command
        cameras = ('image', 'wrist_image') if standard else ('observation.images.image', 'observation.images.image2')
        (out / 'meta/info.json').write_text(json.dumps({
            'total_episodes': 150 * len(self.task_ids), 'total_frames': 300 * len(self.task_ids),
            'features': {key: {'dtype': 'image' if standard else 'video', 'shape': [256, 256, 3]} for key in cameras}}))
        records = [{'bddl_file': self.names[task], 'scene_id': scene.scene_id,
                    'scene_hash': scene.scene_hash}
                   for task in self.task_ids for scene in self.scenes]
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

    def test_eleven_tasks_use_paired_bddls_and_standard_images(self):
        targets_path = job.LANGGAP.parent / 'dataset/libero_goal_semantic_variants/multi_pick_place_targets.json'
        raw = json.loads(targets_path.read_text())
        targets = {key: entry for key, entry in raw.items() if not key.startswith('_')}
        self.task_ids = tuple(entry['task_id'] for entry in targets.values())
        self.names = {entry['task_id']: Path(entry['bddl_path']).name for entry in targets.values()}
        self.load.return_value = (self.archive, self.scenes, self.names)
        (self.root / 'targets.json').write_text(json.dumps(targets))
        for task in self.task_ids:
            (self.root / f'task{task}.hdf5').touch()
        self.corrupt_pairing = False
        with patch.object(job.subprocess, 'run', side_effect=self.fake_conversion) as run:
            output = job.convert_run(self.root, 'test/standard', standard_libero=True)
        command = run.call_args.args[0]
        self.assertIn('--standard-libero', command)
        for bddl in command[command.index('--bddl') + 1:command.index('--repo_id')]:
            self.assertIn('paired_original_extended', bddl)
        metadata = json.loads((output / 'meta/image_convention.json').read_text())
        self.assertEqual(metadata['cameras'], ['image', 'wrist_image'])

    def test_same_episode_count_with_missing_configuration_is_rejected(self):
        self.corrupt_pairing = True
        with patch.object(job.subprocess, 'run', side_effect=self.fake_conversion):
            with self.assertRaisesRegex(ValueError, 'each matched configuration once per task'):
                job.convert_run(self.root, 'test/matched')


if __name__ == '__main__':
    unittest.main()
