"""Load accepted training scenes for exhaustive matched-scene evaluation."""
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'collect' / 'missing_scripts'))
from matched_scene import SceneSnapshot

MATCHED_GOAL_TASK_IDS = (54, 55, 56, 57, 77, 78, 79, 80, 81)


def load_matched_scenes(path, expected_count=150, task_ids=MATCHED_GOAL_TASK_IDS):
    """Read accepted bundles only; never substitute sampled or rejected scenes.

    Paths embedded in collection metadata may refer to another machine. Only
    their BDDL basenames are retained, while the archive itself is self-contained.
    """
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / 'matched_demos.hdf5'
    if expected_count < 1:
        raise ValueError('Expected scene count must be positive')
    scenes, bddl_names = [], {}
    with h5py.File(path, 'r') as archive:
        if archive.attrs.get('schema_version') != 1 or 'scenes' not in archive:
            raise ValueError('Expected a version-1 matched_demos.hdf5 archive')
        for key in sorted(archive['scenes']):
            group = archive['scenes'][key]
            if not group.attrs.get('complete', False):
                raise ValueError(f'Incomplete matched bundle: {key}')
            scene = SceneSnapshot.load(group['initial_scene'])
            if scene.scene_id != key:
                raise ValueError(f'Scene ID does not match archive key: {key}')
            # Validate every training task, even when evaluating a task subset.
            for task_id in MATCHED_GOAL_TASK_IDS:
                episode = group[f'tasks/{task_id}']
                if (episode.attrs['scene_id'] != scene.scene_id or
                        episode.attrs['scene_hash'] != scene.scene_hash or
                        not np.array_equal(episode['states'][0], scene.arrays['state'])):
                    raise ValueError(f'{key}/task{task_id}: training initial scene mismatch')
                name = Path(episode.attrs['bddl_file']).name
                if task_id in bddl_names and bddl_names[task_id] != name:
                    raise ValueError(f'Task {task_id} BDDL changes between scenes')
                bddl_names[task_id] = name
            scenes.append(scene)
    if len(scenes) != expected_count:
        raise ValueError(f'Expected {expected_count} accepted scenes, found {len(scenes)} in {path}')
    if len({scene.scene_hash for scene in scenes}) != len(scenes):
        raise ValueError('Archive contains duplicate physical initial configurations')
    if not set(task_ids).issubset(bddl_names):
        raise ValueError('Selected tasks are absent from the matched archive')
    return path, scenes, bddl_names


def initialize_episode(env, init_states, init_state_id, scene=None, verify_images=False):
    """Return the policy's first observation, without advancing matched scenes."""
    if scene is not None:
        return scene.restore(env.env, verify_images=verify_images)
    obs = env.reset()
    if init_states[init_state_id] is not None:
        obs = env.set_init_state(init_states[init_state_id])
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    return obs
