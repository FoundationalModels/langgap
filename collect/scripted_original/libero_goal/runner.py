"""Shared CLI and HDF5 export for original Goal collectors."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import h5py
import numpy as np

from policies import collect_one_demo, pose

LANGGAP = Path(__file__).resolve().parents[3]


def main(task):
    parser = argparse.ArgumentParser(description=f"Collect original LIBERO Goal: {task}")
    parser.add_argument("--bddl", type=Path,
                        default=LANGGAP / "data/bddl_files/libero_goal/original" / f"{task}.bddl")
    parser.add_argument("--target", "--episodes", type=int, default=50)
    parser.add_argument("--num", type=int, default=200, help="Maximum attempts")
    parser.add_argument("--max_steps", "--max-steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if min(args.target, args.num, args.max_steps) <= 0:
        parser.error("Episode, attempt, and step counts must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    bddl = args.bddl.resolve()
    if not bddl.is_file():
        parser.error(f"BDDL not found: {bddl}")
    # Alternate copies, including paired BDDLs, must describe this exact task.
    expected = LANGGAP / "data/bddl_files/libero_goal/original" / f"{task}.bddl"
    if pose.BDDLUtils.robosuite_parse_problem(str(bddl)) != pose.BDDLUtils.robosuite_parse_problem(str(expected)):
        parser.error("--bddl must describe the same task as this collector")
    output = args.output or (LANGGAP / "data/source/scripted_original/libero_goal" /
                             f"{task}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S_%f}.hdf5")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    target, attempts = (1, 1) if args.debug else (args.target, args.num)
    np.random.seed(args.seed)
    problem = pose.BDDLUtils.get_problem_info(str(bddl))["problem_name"]
    kwargs = dict(bddl_file_name=str(bddl), robots=["Panda"],
                  controller_configs=pose.load_controller_config(default_controller="OSC_POSE"),
                  has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                  ignore_done=True, reward_shaping=True, control_freq=20, hard_reset=False)
    env = pose.TASK_MAPPING[problem](**kwargs)
    demos, fixtures, seeds = [], [], []
    try:
        for attempt in range(attempts):
            seed = (args.seed + attempt) % 2**32
            np.random.seed(seed)
            states, actions, success = collect_one_demo(env, task, args.max_steps, args.debug)
            print(f"Attempt {attempt + 1}: {'SUCCESS' if success else 'FAILED'}; "
                  f"{len(actions)} steps; {len(demos) + int(success)}/{target} demos", flush=True)
            if success:
                demos.append((states, actions))
                fixtures.append((env.sim.model.body_pos.copy(), env.sim.model.body_quat.copy()))
                seeds.append(seed)
            if len(demos) >= target:
                break
        if demos:
            names = [env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]
            env_info = json.dumps({"env_name": problem, "type": 1, "env_kwargs": kwargs})
            pose.save_hdf5(demos, output, env_info, str(bddl),
                           pose.get_instruction_from_bddl(str(bddl)),
                           model_poses=fixtures, model_body_names=names)
            with h5py.File(output, "a") as file:
                file["data"].attrs["collector"] = f"scripted_original/libero_goal/{task}"
                file["data"].attrs["requested_episodes"] = target
                file["data"].attrs["complete"] = len(demos) == target
                for index, seed in enumerate(seeds, 1):
                    file[f"data/demo_{index}"].attrs["seed"] = seed
            print(f"Saved {len(demos)} demos to {output}", flush=True)
    finally:
        env.close()
    if len(demos) < target:
        raise SystemExit(f"Collected {len(demos)}/{target}; attempt budget exhausted")
