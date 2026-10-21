#!/usr/bin/env python3
"""Collect extended and optional original Goal tasks from identical settled scenes."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import h5py
import numpy as np

from matched_scene import (SceneSnapshot, physical_definition, reset_controller_goals)

LANGGAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LANGGAP))
from task_registry import EXTENDED_TASK_MAP

TASK_IDS = (54, 55, 56, 57, 77, 78, 79, 80, 81)
ORIGINAL_TASKS = {
    10: 'put_the_bowl_on_the_plate',
    11: 'put_the_bowl_on_the_stove',
    12: 'put_the_bowl_on_top_of_the_cabinet',
    14: 'put_the_wine_bottle_on_the_rack',
    17: 'open_the_top_drawer_and_put_the_bowl_inside',
    18: 'open_the_middle_drawer_of_the_cabinet',
    19: 'push_the_plate_to_the_front_of_the_stove',
}
# Extended ID -> representative paired original ID from the BDDL mapping.
ORIGINAL_PAIRS = {54: 11, 55: 10, 56: 17, 57: 18, 77: 11,
                  78: 12, 79: 10, 80: 19, 81: 14}


def task_paths(task_id):
    if task_id in ORIGINAL_TASKS:
        relative = Path('libero_goal/original') / ORIGINAL_TASKS[task_id]
        return (LANGGAP / 'data/bddl_files' / relative.with_suffix('.bddl'),
                LANGGAP / 'data/init_files' / relative.with_suffix('.pruned_init'))
    task = EXTENDED_TASK_MAP[task_id]
    relative = Path(task['suite']) / 'extended' / task['dim'] / task['bddl']
    return (LANGGAP / 'data/bddl_files' / relative.with_suffix('.bddl'),
            LANGGAP / 'data/init_files' / relative.with_suffix('.pruned_init'))


def load_collector(task_id):
    if task_id in ORIGINAL_TASKS:
        path = LANGGAP / 'collect/scripted_original/libero_goal/policies.py'
        spec = importlib.util.spec_from_file_location('matched_original_policies', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        task_name = ORIGINAL_TASKS[task_id]
        def collect(env, max_steps, debug=False, initialize_scene=None):
            return module.collect_one_demo(env, task_name, max_steps, debug, initialize_scene)
        return SimpleNamespace(
            discover_names=lambda env, debug=False: (), collect_one_demo=collect,
            save_hdf5=module.pose.save_hdf5,
            get_instruction_from_bddl=module.pose.get_instruction_from_bddl,
            EXPECTED_MAX_STEPS=700)
    script = Path(__file__).parent / f'scripted_collect_task{task_id}.py'
    if not script.exists():
        script = LANGGAP / 'collect' / script.name
    spec = importlib.util.spec_from_file_location(f'matched_task{task_id}', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_manifest(path, manifest):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(path)


def sample_scene(env, init_states, index, seed, verify_images):
    candidate_seed = (seed + index) % (2**32)
    np.random.seed(candidate_seed)
    env.reset()
    init_index = -1
    if init_states is not None:
        init_index = index % len(init_states)
        state = np.asarray(init_states[init_index], dtype=float)
        if state.shape != env.sim.get_state().flatten().shape:
            raise ValueError('Init bank state shape is incompatible with this model')
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        reset_controller_goals(env)
    for _ in range(10):
        env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
    return SceneSnapshot.capture(env, f'scene_{index:06d}', candidate_seed,
                                 init_index, verify_images)


def export_task_files(run_dir, archive, modules, bddls, instructions, task_ids=TASK_IDS):
    scenes = [archive['scenes'][key] for key in sorted(archive['scenes'])
              if archive['scenes'][key].attrs.get('complete', False)]
    if not scenes:
        return
    poses = [(group['initial_scene/model_body_pos'][:],
              group['initial_scene/model_body_quat'][:]) for group in scenes]
    body_names = json.loads(scenes[0]['initial_scene'].attrs['layout'])['names']['body']
    for task_id in task_ids:
        demos = [(group[f'tasks/{task_id}/states'][:],
                  group[f'tasks/{task_id}/actions'][:]) for group in scenes]
        output = run_dir / f'task{task_id}.hdf5'
        temporary = output.with_suffix('.hdf5.pending')
        env_info = json.dumps({'env_name': 'libero_tabletop_manipulation', 'type': 1,
                              'env_kwargs': {'bddl_file_name': str(bddls[task_id]),
                                             'robots': ['Panda']}})
        modules[task_id].save_hdf5(demos, temporary, env_info, str(bddls[task_id]),
                                  instructions[task_id], model_poses=poses,
                                  model_body_names=body_names)
        with h5py.File(temporary, 'a') as file:
            file['data'].attrs['matched_task_ids'] = json.dumps(task_ids)
            file['data'].attrs['matched_archive'] = str(run_dir / 'matched_demos.hdf5')
            pairs = {str(k): v for k, v in ORIGINAL_PAIRS.items()
                     if k in task_ids and v in task_ids}
            file['data'].attrs['original_pairs'] = json.dumps(pairs)
            file['data'].attrs['task_id'] = task_id
            file['data'].attrs['task_role'] = 'original' if task_id in ORIGINAL_TASKS else 'extended'
            for i, group in enumerate(scenes, 1):
                episode = file[f'data/demo_{i}']
                initial = group['initial_scene']
                group.copy('initial_scene', episode)
                for key in ('scene_id', 'scene_hash'):
                    episode.attrs[key] = initial.attrs[key]
                if 'initial_image' in initial:
                    episode.attrs['initial_image_hash'] = hashlib.sha256(
                        initial['initial_image'][:].tobytes()).hexdigest()
        temporary.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-originals', action='store_true',
                        help='Match all seven paired originals as well (16 demos per scene)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Complete matched scenes requested (nine extended, optionally seven original demos each)')
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='Candidate scenes allowed (default: 4 * episodes)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for new scene candidates')
    parser.add_argument('--output-dir', type=Path,
                        help='New run directory; default: timestamped matched_libero_goal run')
    parser.add_argument('--scene-bank', type=Path,
                        help='Reuse an existing scene_bank.hdf5 without modifying it')
    parser.add_argument('--init-file', type=Path,
                        help='Canonical .pruned_init bank (default: task 54)')
    parser.add_argument('--random-init', action='store_true',
                        help='Sample canonical scenes from BDDL instead of a .pruned_init bank')
    parser.add_argument('--no-verify-images', action='store_true',
                        help='Verify exact model/state equality without rendering initial images')
    parser.add_argument('--videos', action='store_true',
                        help='Render action replay videos for every exported demo')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    task_ids = TASK_IDS + tuple(ORIGINAL_TASKS) if args.include_originals else TASK_IDS
    if args.episodes <= 0 or (args.max_scenes is not None and args.max_scenes <= 0):
        parser.error('--episodes and --max-scenes must be positive')
    if args.max_scenes is not None and args.max_scenes < args.episodes:
        parser.error('--max-scenes must be at least --episodes')
    if args.scene_bank and (args.init_file or args.random_init):
        parser.error('--scene-bank cannot be combined with --init-file or --random-init')
    if args.init_file and args.random_init:
        parser.error('--init-file cannot be combined with --random-init')
    if args.scene_bank and not args.scene_bank.is_file():
        parser.error(f'Scene bank not found: {args.scene_bank}')
    if args.output_dir and args.output_dir.exists():
        parser.error('--output-dir must not already exist')

    from libero.libero.envs import TASK_MAPPING
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    from robosuite import load_controller_config

    bddls = {task: task_paths(task)[0] for task in task_ids}
    parsed = {task: robosuite_parse_problem(str(path)) for task, path in bddls.items()}
    canonical_definition = physical_definition(parsed[54])
    for task in task_ids:
        if physical_definition(parsed[task]) != canonical_definition:
            raise ValueError(f'Task {task} no longer has the canonical physical scene')
    modules = {task: load_collector(task) for task in task_ids}
    instructions = {task: modules[task].get_instruction_from_bddl(str(bddls[task]))
                    for task in task_ids}
    init_path = args.init_file or task_paths(54)[1]
    init_states = None
    if not args.scene_bank and not args.random_init:
        import torch
        if not init_path.is_file():
            parser.error(f'Init bank not found: {init_path}; use --random-init to sample BDDL')
        init_states = np.asarray(torch.load(init_path, weights_only=False, map_location='cpu'))
        if init_states.ndim != 2 or len(init_states) == 0 or not np.isfinite(init_states).all():
            parser.error('Init file must contain a finite, nonempty 2-D state array')

    if args.output_dir:
        run_dir = args.output_dir.resolve()
        run_dir.mkdir(parents=True)
    else:
        parent = LANGGAP / 'data/scripted_runs/matched_libero_goal'
        parent.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=parent))
    print(f'Run directory: {run_dir}', flush=True)
    bank_path = args.scene_bank.resolve() if args.scene_bank else run_dir / 'scene_bank.hdf5'
    manifest = {'schema_version': 1, 'status': 'running', 'task_ids': list(task_ids),
                'scene_bank': str(bank_path), 'seed': args.seed,
                'verify_images': not args.no_verify_images, 'requested_scenes': args.episodes,
                'accepted': [], 'rejected': [],
                'original_pairs': {str(k): v for k, v in ORIGINAL_PAIRS.items()} if args.include_originals else {},
                'bddls': {str(task): str(path) for task, path in bddls.items()}}
    manifest_path = run_dir / 'manifest.json'
    write_manifest(manifest_path, manifest)
    environments = {}
    bank = archive = None
    try:
        bank = h5py.File(bank_path, 'r' if args.scene_bank else 'x')
        if args.scene_bank:
            if bank.attrs.get('physical_definition') != canonical_definition:
                raise ValueError('Scene bank physical definition does not match these tasks')
            keys = sorted(bank['scenes'])
        else:
            bank.attrs.update(schema_version=1, suite='libero_goal',
                              physical_definition=canonical_definition,
                              init_file=str(init_path.resolve()) if init_states is not None else 'BDDL sampling')
            if init_states is not None:
                bank.attrs['init_file_sha256'] = hashlib.sha256(init_path.read_bytes()).hexdigest()
            bank.create_group('scenes')
            keys = None
        archive = h5py.File(run_dir / 'matched_demos.hdf5', 'x')
        archive.attrs.update(schema_version=1, task_ids=json.dumps(task_ids), scene_bank=str(bank_path))
        archive.attrs['original_pairs'] = json.dumps(manifest['original_pairs'])
        archive.create_group('scenes')
        for task in task_ids:
            environments[task] = TASK_MAPPING[parsed[task]['problem_name']](
                bddl_file_name=str(bddls[task]), robots=['Panda'],
                controller_configs=[load_controller_config(default_controller='OSC_POSE')],
                has_renderer=False, has_offscreen_renderer=not args.no_verify_images,
                use_camera_obs=False, ignore_done=True, reward_shaping=True,
                control_freq=20, hard_reset=False)
            environments[task].reset()
        limit = args.max_scenes or 4 * args.episodes
        if keys is not None:
            limit = min(limit, len(keys))
        for index in range(limit):
            if keys is None:
                scene = sample_scene(environments[54], init_states, index, args.seed,
                                     not args.no_verify_images)
                scene.save(bank['scenes'].create_group(scene.scene_id))
                bank.flush()
            else:
                scene = SceneSnapshot.load(bank['scenes'][keys[index]])
            print(f'Candidate {scene.scene_id} ({scene.scene_hash[:12]})', flush=True)
            bundle = {}
            failed = None
            for task in task_ids:
                env = environments[task]
                def initialize(target_env):
                    return scene.restore(target_env, verify_images=not args.no_verify_images)
                # Discover names on the restored model, before policy initialization.
                scene.restore(env, verify_images=False)
                names = modules[task].discover_names(env, debug=False)
                states, actions, success = modules[task].collect_one_demo(
                    env, *names, max_steps=700 if task in ORIGINAL_TASKS else 500,
                    debug=args.debug, initialize_scene=initialize)
                if not states or not np.array_equal(states[0], scene.arrays['state']):
                    raise AssertionError(f'Task {task}: first recorded state differs from scene bank')
                actions = np.asarray(actions)
                if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all() or np.abs(actions).max() > 1:
                    raise AssertionError(f'Task {task}: invalid recorded actions')
                success = bool(success and len(actions) <= modules[task].EXPECTED_MAX_STEPS)
                print(f'  Task {task}: {"SUCCESS" if success else "FAILED"} ({len(actions)} steps)', flush=True)
                if not success:
                    failed = task
                    break
                bundle[task] = (np.asarray(states), actions)
            if failed is not None:
                manifest['rejected'].append({'scene_id': scene.scene_id, 'scene_hash': scene.scene_hash,
                                             'failed_task': failed})
            else:
                group = archive['scenes'].create_group(scene.scene_id)
                group.attrs['complete'] = False
                scene.save(group.create_group('initial_scene'))
                tasks = group.create_group('tasks')
                for task, (states, actions) in bundle.items():
                    episode = tasks.create_group(str(task))
                    episode.create_dataset('states', data=states)
                    episode.create_dataset('actions', data=actions)
                    episode.attrs.update(num_samples=len(actions), instruction=instructions[task],
                                         bddl_file=str(bddls[task]), scene_id=scene.scene_id,
                                         scene_hash=scene.scene_hash)
                group.attrs['complete'] = True
                archive.flush()
                manifest['accepted'].append({'scene_id': scene.scene_id, 'scene_hash': scene.scene_hash})
                print(f'Accepted matched scene {len(manifest["accepted"])}/{args.episodes}', flush=True)
            write_manifest(manifest_path, manifest)
            if len(manifest['accepted']) >= args.episodes:
                break
        export_task_files(run_dir, archive, modules, bddls, instructions, task_ids)
        manifest['status'] = 'complete' if len(manifest['accepted']) == args.episodes else 'partial'
        write_manifest(manifest_path, manifest)
    except BaseException as error:
        manifest['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'error'
        manifest['error'] = str(error)
        write_manifest(manifest_path, manifest)
        raise
    finally:
        for env in environments.values():
            env.close()
        if archive is not None:
            archive.close()
        if bank is not None:
            bank.close()
    if manifest['status'] != 'complete':
        print(f'Only {len(manifest["accepted"])}/{args.episodes} complete scene bundles; see manifest.json', file=sys.stderr)
        return 1
    if args.videos:
        for task in task_ids:
            subprocess.run([sys.executable, str(LANGGAP / 'process/replay_dataset.py'),
                            '--dataset', str(run_dir / f'task{task}.hdf5'),
                            '--bddl', str(bddls[task]), '--episodes', str(args.episodes),
                            '--frame_skip', '1', '--fps', '20',
                            '--output_dir', str(run_dir / 'videos' / f'task{task}')], check=True)
    print(f'Saved {args.episodes} matched scenes × {len(task_ids)} tasks in {run_dir}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
