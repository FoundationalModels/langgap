"""Check dataset camera orientation against the LIBERO eval processor.

Run: python -m unittest discover -s langgap/convert/tests -v
"""
from pathlib import Path
import sys
import unittest
import tempfile
import h5py
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

LANGGAP = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(LANGGAP / 'convert'), str(LANGGAP / 'lerobot/src')]
from source_to_lerobot import render_trajectory
import source_to_lerobot as exporter
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def observation(index):
    # Non-square asymmetric patterns make axis swaps and mirrors distinguishable.
    image = (np.arange(5 * 7 * 3).reshape(5, 7, 3) + index).astype(np.uint8)
    return {'agentview_image': image,
            'robot0_eye_in_hand_image': np.roll(image, 2, axis=1),
            'robot0_eef_pos': np.zeros(3),
            'robot0_eef_quat': np.array([0., 0., 0., 1.]),
            'robot0_gripper_qpos': np.zeros(2)}


class ImageOrientationTest(unittest.TestCase):
    def assert_frames_match_eval(self, frames):
        for index, frame in enumerate(frames):
            for camera, key in [('agentview_image', 'observation.images.image'),
                                ('robot0_eye_in_hand_image', 'observation.images.image2')]:
                with self.subTest(frame=index, camera=camera):
                    raw = observation(index)[camera]
                    tensor = torch.from_numpy(raw.copy()).permute(2, 0, 1).unsqueeze(0)
                    expected = LiberoProcessorStep()._process_observation({key: tensor})[key]
                    expected = expected[0].permute(1, 2, 0).numpy()
                    np.testing.assert_array_equal(frame[key], expected)
                    self.assertFalse(np.array_equal(frame[key], raw[::-1]))

    def test_all_frames_and_cameras_match_evaluation(self):
        env = Mock()
        env.set_init_state.side_effect = [observation(i) for i in range(3)]
        actions = np.arange(21, dtype=float).reshape(3, 7)
        frames = render_trajectory(env, np.zeros((3, 2)), actions, 'task')
        self.assert_frames_match_eval(frames)
        for i, frame in enumerate(frames):
            np.testing.assert_array_equal(frame['action'], actions[i].astype(np.float32))

    def test_matched_scene_initial_frame_uses_same_convention(self):
        env, scene = Mock(), Mock()
        scene.restore.return_value = observation(0)
        env.set_init_state.side_effect = [observation(1), observation(2)]
        frames = render_trajectory(env, np.zeros((3, 2)), np.zeros((3, 7)), 'task',
                                   initial_scene=scene)
        self.assert_frames_match_eval(frames)
        env.reset.assert_not_called()

    def test_standard_export_keeps_json_instruction_and_frame_values(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / 'task.hdf5'
            with h5py.File(source, 'w') as file:
                group = file.create_group('data')
                group.attrs['instruction'] = 'Put the Bowl on the Cabinet'
                demo = group.create_group('demo_1')
                demo.create_dataset('states', data=np.zeros((2, 2)))
                demo.create_dataset('actions', data=np.zeros((2, 7)))
            env = Mock()
            env.set_init_state.side_effect = [observation(i) for i in range(2)]
            dataset = Mock()
            with patch.object(exporter, 'create_env', return_value=env), patch.object(
                exporter, 'get_instruction_from_bddl', return_value='Different BDDL Language'
            ):
                count = exporter.process_one_hdf5(source, 'task.bddl', dataset, standard_libero=True)
            self.assertEqual(count, 1)
            self.assertEqual(dataset.add_frame.call_count, 2)
            for index, call in enumerate(dataset.add_frame.call_args_list):
                frame = call.args[0]
                self.assertEqual(set(frame), {'image', 'wrist_image', 'state', 'actions', 'task'})
                self.assertEqual(frame['task'], 'put the bowl on the cabinet')
                np.testing.assert_array_equal(frame['image'], observation(index)['agentview_image'][::-1, ::-1])
                np.testing.assert_array_equal(frame['actions'], np.zeros(7, dtype=np.float32))
            env.close.assert_called_once()

    def test_pi05_model_processing_does_not_flip_stored_images(self):
        # Exercise the actual model method without loading weights. Matching the
        # configured resolution isolates orientation from interpolation differences.
        image = torch.linspace(0, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
        key = 'observation.images.image'
        stub = SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]),
                               config=SimpleNamespace(image_features={key: None},
                                                      image_resolution=(224, 224)))
        images, masks = PI05Policy._preprocess_images(stub, {key: image})
        torch.testing.assert_close(images[0], image * 2 - 1)
        self.assertTrue(masks[0].all())


if __name__ == '__main__':
    unittest.main()
