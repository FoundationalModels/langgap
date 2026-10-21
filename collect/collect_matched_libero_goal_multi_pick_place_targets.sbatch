#!/usr/bin/env bash
#SBATCH --job-name=matched-goal-multi-pick-place
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=36:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=matched-goal-multi-pick-place-%j.out
#SBATCH --error=matched-goal-multi-pick-place-%j.err

# Submit from inside the checkout:
#   sbatch langgap/collect/collect_matched_libero_goal_multi_pick_place_targets.sbatch
# Defaults: 150 complete scenes across the 11 JSON-listed tasks (1,650 demos).
# Collector flags may follow the batch path, e.g. --episodes 5 --max-scenes 20.
# Optional overrides: MATCHED_PROJECT_ROOT, MATCHED_OUTPUT_DIR, MATCHED_EPISODES,
# MATCHED_MAX_SCENES, MATCHED_SEED, MATCHED_VERIFY_IMAGES, MUJOCO_GL.
# This job only collects HDF5 data; it does not convert or upload another dataset.
set -euo pipefail

# Slurm runs a spool copy; locate the checkout from the submission directory.
if [[ -n "${MATCHED_PROJECT_ROOT:-}" ]]; then
    matched_root="$(cd -- "$MATCHED_PROJECT_ROOT" && pwd)"
else
    matched_root="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
    while [[ ! -f "$matched_root/activate_env.sh" ||
             ! -f "$matched_root/langgap/collect/missing_scripts/matched_scene.py" ]]; do
        if [[ "$matched_root" == / ]]; then
            echo 'Cannot locate checkout. Submit from inside it or set MATCHED_PROJECT_ROOT.' >&2
            exit 1
        fi
        matched_root="$(dirname -- "$matched_root")"
    done
fi
export MATCHED_PROJECT_ROOT="$matched_root"
cd -- "$matched_root"

# Tufts GPU nodes use EGL. Exact physical state checks are always enabled.
# EGL pixels can vary for an identical state; enable strict image checks explicitly.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ "$MUJOCO_GL" == egl ]]; then export PYOPENGL_PLATFORM=egl; fi
export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
source "$matched_root/activate_env.sh" langgap

matched_run="$(date -u +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local_$$}"
matched_output="${MATCHED_OUTPUT_DIR:-$matched_root/langgap/data/scripted_runs/matched_libero_goal_multi_pick_place_targets/$matched_run}"
matched_args=(--episodes "${MATCHED_EPISODES:-150}"
              --max-scenes "${MATCHED_MAX_SCENES:-600}"
              --seed "${MATCHED_SEED:-0}"
              --output-dir "$matched_output")
matched_verify_default=1
if [[ "$MUJOCO_GL" == egl ]]; then matched_verify_default=0; fi
if [[ "${MATCHED_VERIFY_IMAGES:-$matched_verify_default}" != 1 ]]; then
    matched_args+=(--no-verify-images)
fi

# Keep the Python collector embedded so this is the only dataset-specific entrypoint.
"$environment_dir/bin/python" -u - "${matched_args[@]}" "$@" <<'MATCHED_COLLECTOR_PY'
"""Collect every JSON-listed task from exactly matched settled LIBERO Goal scenes.

One successful trajectory per task per accepted scene is recorded; semantic
language variants are retained as metadata for relabeling the same behavior.
A scene is accepted only when every listed task succeeds from its snapshot.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import h5py
import numpy as np

PROJECT_ROOT = Path(os.environ['MATCHED_PROJECT_ROOT']).resolve()
LANGGAP = PROJECT_ROOT / 'langgap'
DEFAULT_TARGETS = PROJECT_ROOT / 'dataset/libero_goal_semantic_variants/multi_pick_place_targets.json'
# Reuse the existing policy adapters, scene bank, and standard HDF5 exporter.
sys.path.insert(0, str(LANGGAP / 'collect/missing_scripts'))
from collect_matched_libero_goal import (
    ORIGINAL_TASKS, ORIGINAL_PAIRS, TASK_IDS, export_task_files, load_collector,
    sample_scene, task_paths, write_manifest,
)
from matched_scene import SceneSnapshot, physical_definition


def lowercase_language(value):
    """Normalize language text while preserving variant group/type keys."""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return [lowercase_language(item) for item in value]
    if isinstance(value, dict):
        return {key: lowercase_language(item) for key, item in value.items()}
    return value


def load_targets(path):
    """Read the authoritative task list, resolving BDDL paths from the project root."""
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError('Targets JSON must be an object')
    targets, seen = {}, set()
    for key, entry in raw.items():
        if key.startswith('_'):
            continue
        if not isinstance(entry, dict):
            raise ValueError(f'{key}: expected a task object')
        required = {'task_id', 'task_role', 'bddl_path', 'base_command', 'variants'}
        if required - entry.keys():
            raise ValueError(f'{key}: missing fields {sorted(required - entry.keys())}')
        task_id = entry['task_id']
        if type(task_id) is not int or task_id not in (*TASK_IDS, *ORIGINAL_TASKS):
            raise ValueError(f'{key}: unsupported task_id {task_id!r}')
        if task_id in seen:
            raise ValueError(f'{key}: duplicate task_id {task_id}; exports require unique IDs')
        role = 'original' if task_id in ORIGINAL_TASKS else 'extended'
        if entry['task_role'] != role:
            raise ValueError(f'{key}: expected task_role {role}')
        if not isinstance(entry['base_command'], str) or not entry['base_command'].strip():
            raise ValueError(f'{key}: base_command must be nonempty text')
        if not isinstance(entry['variants'], dict):
            raise ValueError(f'{key}: variants must be an object')
        bddl = Path(entry['bddl_path'])
        if not bddl.is_absolute():
            bddl = PROJECT_ROOT / bddl
        if not bddl.is_file():
            raise ValueError(f'{key}: BDDL not found: {bddl}')
        targets[key] = dict(entry, bddl_path=str(bddl.resolve()),
                            base_command=entry['base_command'].lower(),
                            variants=lowercase_language(entry['variants']))
        seen.add(task_id)
    if not targets:
        raise ValueError('Targets JSON contains no tasks')
    return targets


def annotate_exports(run_dir, targets):
    """Keep JSON identity and both pick/place language groups in each task file."""
    for key, entry in targets.items():
        path = run_dir / f"task{entry['task_id']}.hdf5"
        if not path.exists():
            continue  # No complete bundles were collected.
        with h5py.File(path, 'a') as file:
            data = file['data']
            metadata = dict(task_key=key, task_role=entry['task_role'],
                            base_command=entry['base_command'],
                            variants=json.dumps(entry['variants']),
                            targets_file=str(run_dir / 'targets.json'))
            data.attrs.update(metadata)
            for episode in data.values():
                episode.attrs.update(metadata)


def main():
    parser = argparse.ArgumentParser(
        prog='collect_matched_libero_goal_multi_pick_place_targets.sbatch',
        description=__doc__)
    parser.add_argument('--targets', type=Path, default=DEFAULT_TARGETS,
                        help='Task-list JSON (paths inside it are relative to the checkout root)')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Complete matched scenes requested (one demo per listed task in each scene)')
    parser.add_argument('--max-scenes', type=int, default=None,
                        help='Candidate scenes allowed (default: 4 * episodes)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for new scene candidates')
    parser.add_argument('--output-dir', type=Path,
                        help='New run directory; default: timestamped multi_pick_place_targets run')
    parser.add_argument('--scene-bank', type=Path,
                        help='Reuse an existing scene_bank.hdf5 without modifying it')
    parser.add_argument('--init-file', type=Path,
                        help='Canonical .pruned_init bank (task 54 when listed, otherwise first task)')
    parser.add_argument('--random-init', action='store_true',
                        help='Sample canonical scenes from BDDL instead of a .pruned_init bank')
    parser.add_argument('--no-verify-images', action='store_true',
                        help='Verify exact model/state equality without rendering initial images')
    parser.add_argument('--videos', action='store_true',
                        help='Render action replay videos for every exported demo')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    targets_path = args.targets.resolve()
    try:
        targets = load_targets(targets_path)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    task_ids = tuple(entry['task_id'] for entry in targets.values())
    by_id = {entry['task_id']: (key, entry) for key, entry in targets.items()}
    canonical_task = 54 if 54 in task_ids else task_ids[0]
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

    bddls = {task: Path(by_id[task][1]['bddl_path']) for task in task_ids}
    parsed = {task: robosuite_parse_problem(str(path)) for task, path in bddls.items()}
    canonical_definition = physical_definition(parsed[canonical_task])
    for task in task_ids:
        if physical_definition(parsed[task]) != canonical_definition:
            raise ValueError(f'Task {task} no longer has the canonical physical scene')
    print(f'Validated {len(task_ids)} tasks with one common physical scene: {task_ids}', flush=True)
    modules = {task: load_collector(task) for task in task_ids}
    instructions = {task: by_id[task][1]['base_command'] for task in task_ids}
    init_path = args.init_file or task_paths(canonical_task)[1]
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
        parent = LANGGAP / 'data/scripted_runs/matched_libero_goal_multi_pick_place_targets'
        parent.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=parent))
    print(f'Run directory: {run_dir}', flush=True)
    bank_path = args.scene_bank.resolve() if args.scene_bank else run_dir / 'scene_bank.hdf5'
    manifest = {'schema_version': 1, 'status': 'running', 'task_ids': list(task_ids),
                'scene_bank': str(bank_path), 'seed': args.seed,
                'verify_images': not args.no_verify_images, 'requested_scenes': args.episodes,
                'accepted': [], 'rejected': [],
                'targets_file': str(targets_path),
                'targets_sha256': hashlib.sha256(targets_path.read_bytes()).hexdigest(),
                'targets': targets,
                'original_pairs': {str(k): v for k, v in ORIGINAL_PAIRS.items()
                                   if k in task_ids and v in task_ids},
                'bddls': {str(task): str(path) for task, path in bddls.items()}}
    manifest_path = run_dir / 'manifest.json'
    write_manifest(manifest_path, manifest)
    (run_dir / 'targets.json').write_text(json.dumps(targets, indent=2) + '\n')
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
            print(f'Creating environment for task {task}: {by_id[task][0]}', flush=True)
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
                scene = sample_scene(environments[canonical_task], init_states, index, args.seed,
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
                if len(states) == 0 or not np.array_equal(states[0], scene.arrays['state']):
                    raise AssertionError(f'Task {task}: first recorded state differs from scene bank')
                actions = np.asarray(actions)
                if (actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0
                        or len(states) != len(actions) or not np.isfinite(states).all()
                        or not np.isfinite(actions).all() or np.abs(actions).max() > 1):
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
                                         scene_hash=scene.scene_hash,
                                         task_key=by_id[task][0],
                                         task_role=by_id[task][1]['task_role'],
                                         variants=json.dumps(by_id[task][1]['variants']))
                group.attrs['complete'] = True
                archive.flush()
                manifest['accepted'].append({'scene_id': scene.scene_id, 'scene_hash': scene.scene_hash})
                print(f'Accepted matched scene {len(manifest["accepted"])}/{args.episodes}', flush=True)
            write_manifest(manifest_path, manifest)
            if len(manifest['accepted']) >= args.episodes:
                break
        export_task_files(run_dir, archive, modules, bddls, instructions, task_ids)
        annotate_exports(run_dir, targets)
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
MATCHED_COLLECTOR_PY
