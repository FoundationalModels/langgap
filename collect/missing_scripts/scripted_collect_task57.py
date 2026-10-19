#!/usr/bin/env python3
"""
Scripted waypoint policy for automated demo collection.

Task 57: open the bottom drawer (libero_goal).
Strategy: insert a closed-finger hook behind the low handle, pull, and retreat.
Based on scripted_collect_task56.py, with a hook adapted to the bottom handle.
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
    """Locate the bottom drawer joint, region, and handle collision geom."""
    model = env.sim.model
    # Use exact suffixes so nested bodies and the lower drawers cannot match.
    joint = next(model.joint_id2name(i) for i in range(model.njnt)
                 if model.joint_id2name(i).endswith("bottom_level"))
    region = next(model.site_id2name(i) for i in range(model.nsite)
                  if model.site_id2name(i).endswith("cabinet_1_bottom_region"))
    joint_id = model.joint_name2id(joint)
    drawer_body = model.jnt_bodyid[joint_id]
    geoms = [i for i in range(model.ngeom)
             if model.geom_bodyid[i] == drawer_body and model.geom_group[i] == 0]
    # The central handle bar is the frontmost collision box in drawer coordinates.
    handle_geom = min(geoms, key=lambda i: model.geom_pos[i, 1])
    if debug:
        print(f"  Drawer joint: {joint}, region: {region}")
        print(f"  Handle geom: {handle_geom}")
    return joint, region, handle_geom


def compute_action(env, target_pose, gripper):
    return np.concatenate([target_pose_to_action(env, target_pose), [gripper]])


def handle_grasp_rotation(handle_axis, outward, drawer_up, tilt_degrees):
    """Orient a closed-finger hook behind the low handle.

    Align the palm's broad axis along the bar, keeping it clear of the table.
    A tilt toward the drawer keeps the palm outside the cabinet face.
    """
    bar = np.asarray(handle_axis, dtype=float).copy()
    bar /= np.linalg.norm(bar)
    normal = np.asarray(outward, dtype=float).copy()
    normal -= bar * np.dot(normal, bar)
    if np.linalg.norm(normal) < 1e-6:
        raise ValueError("Handle bar must be transverse to the drawer slide axis")
    normal /= np.linalg.norm(normal)
    up = np.cross(bar, normal)
    if np.dot(up, drawer_up) < 0:
        up = -up
        bar = -bar
    tilt = np.deg2rad(tilt_degrees)
    approach = -normal * np.sin(tilt) - up * np.cos(tilt)
    return np.column_stack((bar, np.cross(approach, bar), approach))


def collect_one_demo(env, drawer_joint, drawer_region, handle_geom,
                     max_steps, debug=False, handle_tilt=60., initialize_scene=None):
    """Open the bottom drawer through recorded robot actions and retreat."""
    if initialize_scene is None:
        env.reset()
        settle_action = np.array([0., 0., 0., 0., 0., 0., -1.])
        for _ in range(10):
            env.step(settle_action)
    else:
        # The matched runner restores a settled, verified scene here.
        obs = initialize_scene(env)
    states, actions = [], []

    def run_waypoints(waypoints, phase):
        for index, (pose, gripper, threshold, hold) in enumerate(waypoints):
            target, _ = unmake_pose(pose)
            reached = False
            for step in range(150):
                if len(actions) >= max_steps:
                    return False
                states.append(env.sim.get_state().flatten().copy())
                action = compute_action(env, pose, gripper)
                actions.append(action)
                env.step(action)
                current_pose = get_eef_pose(env)
                distance = np.linalg.norm(current_pose[:3, 3] - target)
                rotation_error = pose[:3, :3] @ current_pose[:3, :3].T
                angle = np.arccos(np.clip((np.trace(rotation_error) - 1.) / 2., -1., 1.))
                # Settle the handle orientation before grasping or advancing.
                aligned = angle < np.deg2rad(3) if phase == "Open" else (
                    index != 0 or angle < np.deg2rad(1))
                if aligned and ((hold and step + 1 >= hold) or
                                (not hold and distance < threshold)):
                    reached = True
                    break
            if debug:
                print(f"  {phase} WP{index + 1}: steps={step + 1}, "
                      f"distance={distance:.4f}, rotation_error={np.rad2deg(angle):.1f}deg, eef={get_eef_pose(env)[:3, 3]}, "
                      f"drawer={float(env.sim.data.get_joint_qpos(drawer_joint)):.4f}")
            if not reached:
                return False
        return True

    joint_id = env.sim.model.joint_name2id(drawer_joint)
    outward = -np.array(env.sim.data.xaxis[joint_id]).copy()
    handle = np.array(env.sim.data.geom_xpos[handle_geom]).copy()
    # The long collision-box axis gives the actual handle bar direction.
    geom_rotation = np.array(env.sim.data.geom_xmat[handle_geom]).reshape(3, 3)
    bar_axis = geom_rotation[:, np.argmax(env.sim.model.geom_size[handle_geom])]
    drawer_body = env.sim.model.jnt_bodyid[joint_id]
    drawer_up = np.array(env.sim.data.body_xmat[drawer_body]).reshape(3, 3)[:, 2]
    handle_rot = handle_grasp_rotation(bar_axis, outward, drawer_up, handle_tilt)
    # Equivalent pinch direction with the wrist on the reachable roll branch.
    handle_rot = handle_rot @ np.diag([-1., -1., 1.])
    # Hook into the gap behind the bar with closed fingers. A vertical pinch
    # would put the wide palm through the table at this low handle height.
    hook = handle - outward * 0.015 + drawer_up * 0.012
    above = hook + drawer_up * 0.018
    approach = handle + outward * 0.12 + drawer_up * 0.10
    pregrasp = make_pose(get_eef_pose(env)[:3, 3], handle_rot)
    if not run_waypoints([(pregrasp, +1, 0.006, 0), (pregrasp, +1, 999, 15)], "Pregrasp"):
        return states, actions, False
    opening = [
        (make_pose(approach, handle_rot), +1, 0.01, 0),
        (make_pose(above + outward * 0.12, handle_rot), +1, 0.006, 0),
        (make_pose(above, handle_rot), +1, 0.006, 0),
        (make_pose(hook, handle_rot), +1, 0.006, 0),
        (make_pose(hook, handle_rot), +1, 999, 6),
    ]
    opening += [(make_pose(hook + outward * distance, handle_rot), +1, 0.008, 0)
                for distance in np.linspace(0.025, 0.15, 6)]
    pulled = hook + outward * 0.15
    opening += [
        (make_pose(pulled + drawer_up * 0.10, handle_rot), +1, 0.01, 0),
        (make_pose(pulled + drawer_up * 0.10, handle_rot), -1, 999, 12),
    ]
    if not run_waypoints(opening, "Open"):
        return states, actions, False
    if float(env.sim.data.get_joint_qpos(drawer_joint)) > -0.12:
        if debug:
            print("  Drawer did not open far enough; rejecting attempt.")
        return states, actions, False

    # Hold after retreat, so success is checked with the robot off the handle.
    pose = get_eef_pose(env)
    completed = run_waypoints([(pose, -1, 999, 15)], "Settle")
    success = completed and bool(env._check_success())
    if debug:
        print(f"  Total steps: {len(actions)}, Success: {success}")
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


EXPECTED_MAX_STEPS = 260  # Includes approach, pull, release, retreat, and settling.

def main():
    parser = argparse.ArgumentParser(
        description="Scripted waypoint demo collection (Task 57: open bottom drawer)")
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
    parser.add_argument("--handle_tilt", type=float, default=60.,
                        help="Handle-frame clearance tilt in degrees (0 down, 90 into drawer)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 1 demo, print all names/coords")
    args = parser.parse_args()
    if min(args.num, args.target, args.max_steps) <= 0:
        parser.error("--num, --target, and --max_steps must be positive")
    if not 0. <= args.handle_tilt <= 90.:
        parser.error("--handle_tilt must be between 0 and 90 degrees")
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
    print("Scripted Waypoint Demo Collection (Task 57)")
    print("=" * 60)
    print(f"Task: {instruction}")
    print(f"Max attempts: {num_demos}")
    print(f"Target successes: {target}")
    print(f"Max steps/demo: {args.max_steps}")
    print(f"Handle-frame clearance tilt: {args.handle_tilt} degrees")
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
    drawer_joint, drawer_region, handle_geom = discover_names(env, debug=args.debug)
    print(f"  Bottom drawer joint:  {drawer_joint}")

    # Collect demos
    demos = []
    model_poses = []
    model_body_names = [env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]
    successes = 0

    for i in range(num_demos):
        print(f"\n--- Demo {i + 1}/{num_demos} ---")
        states, actions, success = collect_one_demo(
            env, drawer_joint, drawer_region, handle_geom,
            max_steps=args.max_steps,
            debug=args.debug,
            handle_tilt=args.handle_tilt,
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
            print(f"  FAILED ({len(actions)} steps) "
                  f"[{successes}/{i + 1} successful]")

    env.close()

    print(f"\n{'=' * 60}")
    print(f"Collection complete: {successes}/{i + 1} successful")
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
