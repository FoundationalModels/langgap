#!/usr/bin/env python3
"""Package a complete matched goal run, including correctly oriented camera data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

LANGGAP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(LANGGAP), str(LANGGAP / 'eval')]
from matched_scenes import load_matched_scenes, MATCHED_GOAL_TASK_IDS
from task_registry import EXTENDED_TASK_MAP


def convert_run(run_dir, repo_id, expected_scenes=150, standard_libero=False):
    run_dir = Path(run_dir).expanduser().resolve()
    output = run_dir / 'lerobot'
    if output.exists():
        raise ValueError(f'Refusing to overwrite existing dataset: {output}')
    archive, scenes, bddl_names = load_matched_scenes(run_dir, expected_scenes)
    inputs, bddls = [], []
    # New runs retain the exact paired BDDL paths in their resolved task list.
    targets_file = run_dir / 'targets.json'
    targets = json.loads(targets_file.read_text()) if targets_file.is_file() else None
    task_ids = tuple(entry['task_id'] for entry in targets.values()) if targets else MATCHED_GOAL_TASK_IDS
    if set(task_ids) != set(bddl_names):
        raise ValueError('Task list differs from the tasks in the matched archive')
    for task_id in task_ids:
        if targets:
            entry = next(entry for entry in targets.values() if entry['task_id'] == task_id)
            bddl = Path(entry['bddl_path'])
            if not bddl.is_absolute():
                bddl = LANGGAP.parent / bddl
            if not bddl.is_file():
                # Resolve paths saved on another collection machine.
                matches = list((LANGGAP / 'data/bddl_files/libero_goal/paired_original_extended').rglob(bddl.name))
                if len(matches) != 1:
                    raise ValueError(f'Cannot uniquely resolve paired BDDL: {bddl}')
                bddl = matches[0]
        else:
            task = EXTENDED_TASK_MAP[task_id]
            bddl = (LANGGAP / 'data/bddl_files' / task['suite'] / 'extended' /
                    task['dim'] / (task['bddl'] + '.bddl'))
        source = run_dir / f'task{task_id}.hdf5'
        if not source.is_file() or not bddl.is_file():
            raise FileNotFoundError(f'Missing task input: {source} or {bddl}')
        if bddl.name != bddl_names[task_id]:
            raise ValueError(f'Task {task_id} does not match the scene archive')
        inputs.append(str(source))
        bddls.append(str(bddl))
    command = [sys.executable, '-u', str(LANGGAP / 'convert/source_to_lerobot.py'),
               '--input', *inputs, '--bddl', *bddls, '--repo_id', repo_id,
               '--output', str(output), '--matched-scenes', str(archive),
               '--frame_skip', '1']
    if standard_libero:
        command.append('--standard-libero')
    # No trimming, frame skipping, or automatic Hub upload.
    subprocess.run(command, check=True)
    info = json.loads((output / 'meta/info.json').read_text())
    expected_episodes = expected_scenes * len(task_ids)
    if info['total_episodes'] != expected_episodes or info['total_frames'] < expected_episodes:
        raise ValueError(f'Converted dataset must contain {expected_episodes} nonempty episodes')
    records = json.loads((output / 'meta/matched_episodes.json').read_text())['episodes']
    actual = Counter((entry['bddl_file'], entry['scene_id'], entry['scene_hash'])
                     for entry in records)
    expected = Counter((bddl_names[task], scene.scene_id, scene.scene_hash)
                       for task in task_ids for scene in scenes)
    if actual != expected:
        raise ValueError('Dataset does not contain each matched configuration once per task')
    if not (output / 'meta/matched_demos.hdf5').is_file():
        raise FileNotFoundError('Converted dataset is missing its bundled scene snapshots')
    cameras = ('image', 'wrist_image') if standard_libero else ('observation.images.image', 'observation.images.image2')
    for camera in cameras:
        feature = info['features'][camera]
        if feature['dtype'] != ('image' if standard_libero else 'video') or feature['shape'] != [256, 256, 3]:
            raise ValueError(f'Invalid training camera feature: {camera}')
        if not standard_libero and not any(path.stat().st_size > 0 for path in (output / 'videos' / camera).rglob('*.mp4')):
            raise FileNotFoundError(f'No encoded training video for {camera}')
    (output / 'meta/image_convention.json').write_text(json.dumps({
        'schema_version': 1,
        'reference': 'HF LIBERO / OpenPI and LeRobot pi0.5',
        'raw_simulator_to_dataset': 'rotate_180',
        'numpy_transform': 'image[::-1, ::-1].copy()',
        'cameras': list(cameras),
        'frame_skip': 1,
        'expected_scenes': expected_scenes,
        'episodes_per_task': expected_scenes,
    }, indent=2) + '\n')
    print(f'Complete: {expected_scenes} shared configurations, {expected_episodes} demos, '
          f'two correctly oriented training cameras per timestep.\nDataset: {output}', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--repo-id', default='local/libero_goal_matched_150',
                        help='Dataset identity for future upload; this script does not upload')
    parser.add_argument('--standard-libero', action='store_true',
                        help='Export OpenPI standard column names and embedded images')
    parser.add_argument('--expected-scenes', type=int, default=150)
    args = parser.parse_args()
    convert_run(args.run_dir, args.repo_id, args.expected_scenes, args.standard_libero)


if __name__ == '__main__':
    main()
