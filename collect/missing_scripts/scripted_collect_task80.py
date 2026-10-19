#!/usr/bin/env python3
"""
Scripted waypoint policy for automated demo collection.

Task 80: put the black bowl in front of the stove (libero_goal).
Strategy: top-down rim grasp, lift, compensate the grasp offset, place, retreat.
Based on scripted_collect_task43.py (bowl rim grasp) and task 55 (HDF5/CLI).
Action computation follows MimicGen's target_pose_to_action approach.
Output HDF5 format matches the other scripted collection scripts.
"""

import argparse
import json
import re
import sys
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path

import robosuite as suite
from robosuite import load_controller_config
import robosuite.utils.transform_utils as T

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING


# ========== Pose utilities (MimicGen approach) ==========

def make_pose(pos, rot):
    """Create 4x4 pose matrix from position and rotation matrix."""
    pose = np.zeros((4, 4))
    pose[:3, :3] = rot
    pose[:3, 3] = pos
    pose[3, 3] = 1.0
    return pose


def unmake_pose(pose):
    """Extract position and rotation matrix from 4x4 pose."""
    return pose[:3, 3].copy(), pose[:3, :3].copy()


def get_eef_pose(env):
    """Get current EEF 4x4 pose matrix."""
    eef_site = env.robots[0].controller.eef_name
    site_id = env.sim.model.site_name2id(eef_site)
    pos = np.array(env.sim.data.site_xpos[site_id])
    rot = np.array(env.sim.data.site_xmat[site_id].reshape(3, 3))
    return make_pose(pos, rot)


# Fixed gripper rotation for top-down grasp.
# Columns: X=[0,1,0] (fingers close along world +Y, across the bottle neck),
# Y=[1,0,0], Z=[0,0,-1] (down). Matches initial EEF orientation (≈270° approach).
GRASP_ROT = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)


def target_pose_to_action(env, target_pose):
    """
    Convert target 4x4 pose to 6-dim action (MimicGen approach).

    Uses axis-angle representation for rotation delta, compatible with OSC_POSE.
    """
    curr_pose = get_eef_pose(env)
    curr_pos, curr_rot = unmake_pose(curr_pose)
    target_pos, target_rot = unmake_pose(target_pose)

    max_dpos = env.robots[0].controller.output_max[0]
    max_drot = env.robots[0].controller.output_max[3]

    # Position delta
    delta_pos = np.clip((target_pos - curr_pos) / max_dpos, -1., 1.)

    # Rotation delta (axis-angle)
    delta_rot_mat = target_rot @ curr_rot.T
    delta_quat = T.mat2quat(delta_rot_mat)
    delta_rotation = T.quat2axisangle(delta_quat)
    delta_rotation = np.clip(delta_rotation / max_drot, -1., 1.)

    return np.concatenate([delta_pos, delta_rotation])


def get_instruction_from_bddl(bddl_path: str) -> str:
    with open(bddl_path, "r") as f:
        content = f.read()
    match = re.search(r"\(:language ([^)]+)\)", content)
    if match:
        return match.group(1).strip()
    return "Unknown instruction"


def discover_names(env, debug=False):
    names = [env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]
    bowl = next(n for n in names if n and "akita_black_bowl" in n)
    site = "main_table_stove_front_region"
    env.sim.model.site_name2id(site)
    return bowl, site


def get_body_pos(env, body_name):
    body_id = env.sim.model.body_name2id(body_name)
    return np.array(env.sim.data.body_xpos[body_id]).copy()


def get_site_pos(env, site_name):
    site_id = env.sim.model.site_name2id(site_name)
    return np.array(env.sim.data.site_xpos[site_id]).copy()


def compute_action(env, target_pose, gripper):
    """Compute 7-dim action from target pose and gripper state."""
    arm_action = target_pose_to_action(env, target_pose)
    return np.concatenate([arm_action, [gripper]])



def build_grasp_waypoints(bowl_pos, grasp_offset=0.03, bowl_radius=0.048):
    """Pinch the near rim, following the existing bowl collectors' strategy."""
    grasp = bowl_pos + np.array([0., -bowl_radius, grasp_offset])
    lift = grasp + np.array([0., 0., 0.20])
    return [
        (make_pose(lift, GRASP_ROT), -1, 0.01, 0),
        (make_pose(grasp, GRASP_ROT), -1, 0.005, 0),
        (make_pose(grasp, GRASP_ROT), +1, 999, 12),
        (make_pose(lift, GRASP_ROT), +1, 0.01, 0),
    ]


def collect_one_demo(env, bowl_body, target_site, max_steps, debug=False,
                     place_offset=0.01, grasp_offset=0.03, bowl_radius=0.048, initialize_scene=None):
    if initialize_scene is None:
        env.reset()
        for _ in range(10):
            env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
    else:
        # The matched runner restores a settled, verified scene here.
        obs = initialize_scene(env)
    states, actions = [], []
    initial = get_body_pos(env, bowl_body)
    target = get_site_pos(env, target_site)

    def run_waypoints(waypoints, phase):
        for index, (pose, gripper, threshold, hold) in enumerate(waypoints):
            reached = False
            for step in range(150):
                if len(actions) >= max_steps:
                    return False
                states.append(env.sim.get_state().flatten().copy())
                action = compute_action(env, pose, gripper)
                actions.append(action)
                env.step(action)
                distance = np.linalg.norm(get_eef_pose(env)[:3, 3] - pose[:3, 3])
                if (hold and step + 1 >= hold) or (not hold and distance < threshold):
                    reached = True
                    break
            if debug:
                print(f"  {phase} WP{index+1}: steps={step+1}, distance={distance:.4f}, "
                      f"bowl={get_body_pos(env, bowl_body)}")
            if not reached:
                return False
        return True

    if not run_waypoints(build_grasp_waypoints(initial, grasp_offset, bowl_radius), "Grasp"):
        return states, actions, False
    lifted = get_body_pos(env, bowl_body)
    if lifted[2] < initial[2] + 0.10:
        return states, actions, False
    # Measured offset compensation puts the bowl center inside the target zone.
    offset = get_eef_pose(env)[:3, 3] - lifted
    place = target + offset
    place[2] = initial[2] + offset[2] + place_offset
    carry = place + np.array([0., 0., 0.18])
    waypoints = [
        (make_pose(carry, GRASP_ROT), +1, 0.01, 0),
        (make_pose(place, GRASP_ROT), +1, 0.006, 0),
        (make_pose(place, GRASP_ROT), -1, 999, 15),
        (make_pose(carry, GRASP_ROT), -1, 0.01, 0),
        (make_pose(carry, GRASP_ROT), -1, 999, 15),
    ]
    completed = run_waypoints(waypoints, "Place")
    site_id = env.sim.model.site_name2id(target_site)
    half_size = np.abs(env.sim.data.site_xmat[site_id].reshape(3, 3)) @ env.sim.model.site_size[site_id]
    final = get_body_pos(env, bowl_body)
    inside = np.all(np.abs(final[:2] - target[:2]) < half_size[:2])
    upright = env.sim.data.body_xmat[env.sim.model.body_name2id(bowl_body)].reshape(3, 3)[2, 2] > 0.9
    contact = env.check_contact("table_collision", env.get_object("akita_black_bowl_1"))
    success = bool(completed and env._check_success() and inside and upright and contact
                   and abs(final[2] - initial[2]) < 0.015)
    return states, actions, success


def save_hdf5(demos, output_path, env_info, bddl_path, instruction,
              model_poses=None, model_body_names=None):
    """Save collected demos to a new HDF5 file in the standard collection format."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "x") as f:
        grp = f.create_group("data")

        for i, (states, actions) in enumerate(demos):
            ep_grp = grp.create_group(f"demo_{i + 1}")
            ep_grp.create_dataset("states", data=np.array(states))
            ep_grp.create_dataset("actions", data=np.array(actions))
            ep_grp.attrs["num_samples"] = len(actions)
            if model_poses is not None:
                ep_grp.create_dataset("model_body_pos", data=model_poses[i][0])
                ep_grp.create_dataset("model_body_quat", data=model_poses[i][1])

        now = datetime.now()
        grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = "Libero_Tabletop_Manipulation"
        grp.attrs["env_info"] = env_info
        if model_body_names is not None:
            grp.attrs["model_body_names"] = json.dumps(model_body_names)
        grp.attrs["bddl_file"] = bddl_path
        grp.attrs["instruction"] = instruction
        grp.attrs["total"] = len(demos)


EXPECTED_MAX_STEPS = 260  # Includes the measured-offset placement and retreat.

def main():
    parser = argparse.ArgumentParser(
        description="Scripted waypoint demo collection (Task 80: bowl → front of stove)")
    parser.add_argument("--bddl", type=str, required=True,
                        help="BDDL file path")
    parser.add_argument("--num", type=int, default=200,
                        help="Max attempts (upper bound)")
    parser.add_argument("--target", type=int, default=50,
                        help="Stop after this many successes")
    parser.add_argument("--output", type=str, default=None,
                        help="Output HDF5 file path (must not already exist)")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Max steps per demo")
    parser.add_argument("--place_offset", type=float, default=0.01,
                        help="Release clearance above the table (meters)")
    parser.add_argument("--grasp_offset", type=float, default=0.03,
                        help="Height above bowl body for rim grasp (meters)")
    parser.add_argument("--bowl_radius", type=float, default=0.048,
                        help="Bowl rim grasp radius (meters)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 1 demo, print all names/coords")
    args = parser.parse_args()
    if min(args.num, args.target, args.max_steps) <= 0:
        parser.error("--num, --target, and --max_steps must be positive")
    if args.output and Path(args.output).exists():
        parser.error(f"Output already exists: {args.output}")

    # Resolve BDDL path
    bddl_path = Path(args.bddl)
    if not bddl_path.exists():
        bddl_path = Path(__file__).resolve().parents[2] / args.bddl
    if not bddl_path.exists():
        print(f"Error: BDDL file not found: {args.bddl}")
        sys.exit(1)

    bddl_path = str(bddl_path.absolute())
    problem_info = BDDLUtils.get_problem_info(bddl_path)
    problem_name = problem_info["problem_name"]
    instruction = get_instruction_from_bddl(bddl_path)

    num_demos = 1 if args.debug else args.num
    target = 1 if args.debug else args.target

    print("=" * 60)
    print("Scripted Waypoint Demo Collection (Task 80)")
    print("=" * 60)
    print(f"Task: {instruction}")
    print(f"Max attempts: {num_demos}")
    print(f"Target successes: {target}")
    print(f"Max steps/demo: {args.max_steps}")
    print(f"Place offset: {args.place_offset}")
    print(f"Grasp offset: {args.grasp_offset}")
    print(f"Debug: {args.debug}")
    print("=" * 60)

    # Create environment
    print("\nCreating environment...")
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = TASK_MAPPING[problem_name](
        bddl_file_name=bddl_path,
        robots=["Panda"],
        controller_configs=[controller_config],
        has_renderer=False,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )

    # First reset to discover MuJoCo names
    print("Discovering object names...")
    env.reset()
    bowl_body, target_site = discover_names(env, debug=args.debug)
    print(f"  Bowl body: {bowl_body}")
    print(f"  Target site:        {target_site}")

    # Collect demos
    demos = []
    model_poses = []
    model_body_names = [env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]
    successes = 0
    failures = 0

    for i in range(num_demos):
        print(f"\n--- Demo {i + 1}/{num_demos} ---")
        states, actions, success = collect_one_demo(
            env, bowl_body, target_site,
            max_steps=args.max_steps,
            debug=args.debug,
            place_offset=args.place_offset,
            grasp_offset=args.grasp_offset,
            bowl_radius=args.bowl_radius,
        )

        if success:
            if len(actions) <= EXPECTED_MAX_STEPS:
                demos.append((states, actions))
                # Fixture poses are model parameters, absent from flattened states.
                model_poses.append((env.sim.model.body_pos.copy(), env.sim.model.body_quat.copy()))
                successes += 1
                print(f"  SUCCESS ({len(actions)} steps) "
                      f"[{successes}/{i + 1} successful]")
                if successes >= target:
                    print(f"\n  Reached target of {target} successes, stopping early.")
                    break
            else:
                print(f"  DISCARDED ({len(actions)} steps > {EXPECTED_MAX_STEPS})")
        else:
            failures += 1
            print(f"  FAILED ({len(actions)} steps) "
                  f"[{successes}/{i + 1} successful]")

    env.close()

    print(f"\n{'=' * 60}")
    print(f"Collection complete: {successes}/{num_demos} successful")
    print(f"{'=' * 60}")

    if not demos:
        print("No demos to save.")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_dir = Path(__file__).resolve().parents[2] / "data" / "source"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = Path(args.bddl).stem
        output_path = str(output_dir / f"{task_name}_scripted_{timestamp}.hdf5")

    env_info = json.dumps({
        "env_name": problem_name,
        "type": 1,
        "env_kwargs": {
            "bddl_file_name": bddl_path,
            "robots": ["Panda"],
        }
    })

    save_hdf5(demos, output_path, env_info, bddl_path, instruction,
              model_poses=model_poses, model_body_names=model_body_names)
    print(f"Saved {len(demos)} demos to: {output_path}")


if __name__ == "__main__":
    main()
