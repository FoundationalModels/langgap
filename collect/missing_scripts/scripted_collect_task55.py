#!/usr/bin/env python3
"""
Scripted waypoint policy for automated demo collection.

Task 55: put the wine bottle on the plate (libero_goal).
Strategy: top-down neck grasp, vertical lift, carry, place, release, and retreat.
Based on scripted_collect_task79.py (cream cheese → plate).
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
    """Find MuJoCo body names for wine bottle and plate."""
    body_names = [env.sim.model.body_id2name(i)
                  for i in range(env.sim.model.nbody)]

    if debug:
        print("\n=== MuJoCo Body Names ===")
        for n in sorted(body_names):
            print(f"  {n}")

    # Find wine bottle body
    bottle_body = None
    for n in body_names:
        if "wine_bottle" in n.lower():
            bottle_body = n
            break

    # Find plate body
    plate_body = None
    for n in body_names:
        if "plate" in n.lower():
            plate_body = n
            break

    if bottle_body is None:
        raise RuntimeError(f"Could not find wine bottle body in: {body_names}")
    if plate_body is None:
        raise RuntimeError(f"Could not find plate body in: {body_names}")

    return bottle_body, plate_body


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



def build_waypoints(object_pos, plate_pos, table_height,
                    place_offset=0.015, grasp_offset=0.14):
    """Build top-down neck-grasp waypoints for an upright wine bottle.

    The bottle neck is about 14cm above its body origin. Placement preserves
    its resting height relative to the table, plus the plate's surface height.
    Each waypoint is (target_pose, gripper, distance_threshold, hold_steps).
    """
    grasp = object_pos + np.array([0., 0., grasp_offset])
    lift = grasp + np.array([0., 0., 0.18])
    # Plate body origin is close to its top surface (confirmed from collision
    # geometry). A small clearance allows the released bottle to settle.
    place = np.array([plate_pos[0], plate_pos[1],
                      plate_pos[2] + object_pos[2] - table_height
                      + grasp_offset + place_offset])
    carry = place.copy()
    carry[2] = max(lift[2], place[2] + 0.12)
    return [
        (make_pose(lift, GRASP_ROT), -1, 0.01, 0),
        (make_pose(grasp, GRASP_ROT), -1, 0.004, 0),
        (make_pose(grasp, GRASP_ROT), +1, 999, 12),
        (make_pose(lift, GRASP_ROT), +1, 0.01, 0),
        (make_pose(carry, GRASP_ROT), +1, 0.01, 0),
        (make_pose(place, GRASP_ROT), +1, 0.006, 0),
        (make_pose(place, GRASP_ROT), -1, 999, 15),
        (make_pose(place + np.array([0., 0., 0.14]), GRASP_ROT), -1, 0.01, 0),
        (make_pose(place + np.array([0., 0., 0.14]), GRASP_ROT), -1, 999, 15),
    ]


def collect_one_demo(env, bottle_body, plate_body, max_steps, debug=False,
                     place_offset=0.015, grasp_offset=0.14, initialize_scene=None):
    """Collect a single demo using scripted waypoints. Returns (states, actions, success)."""
    if initialize_scene is None:
        obs = env.reset()

        # Let physics settle (objects may be spawned slightly above surfaces)
        # Do NOT record settling steps — they cause visible pauses in playback
        settle_action = np.zeros(7)
        settle_action[6] = -1.0  # gripper open during settling
        for _ in range(10):
            obs, _, _, _ = env.step(settle_action)
    else:
        # The matched runner restores a settled, verified scene here.
        obs = initialize_scene(env)
    states, actions = [], []

    # Get current object positions after settling
    object_pos = get_body_pos(env, bottle_body)
    plate_pos = get_body_pos(env, plate_body)

    # Get table height
    table_top_pos = get_site_pos(env, "table_top")
    table_height = table_top_pos[2]

    if debug:
        eef_pose = get_eef_pose(env)
        eef_pos, _ = unmake_pose(eef_pose)
        # Also check wine bottle default_site position (object center after settling)
        bottle_site_pos = get_site_pos(env, "wine_bottle_1_default_site")
        print(f"\n  EEF position:         {eef_pos}")
        print(f"  Wine bottle body:    {object_pos}")
        print(f"  Wine bottle site:    {bottle_site_pos}")
        print(f"  Plate pos:            {plate_pos}")
        print(f"  Wine bottle Z (body): {object_pos[2]:.4f}")
        print(f"  Wine bottle Z (site): {bottle_site_pos[2]:.4f}")
        print(f"  Plate Z:              {plate_pos[2]:.4f}")
        print(f"  Table top Z:          {table_height:.4f}")

    waypoints = build_waypoints(object_pos, plate_pos, table_height,
                                place_offset=place_offset, grasp_offset=grasp_offset)

    total_steps = 0  # settling steps are not recorded

    for wp_idx, (target_pose, gripper, dist_thresh, hold_steps) in enumerate(waypoints):
        target_pos, _ = unmake_pose(target_pose)
        if debug:
            print(f"  Waypoint {wp_idx + 1}: target={target_pos}, "
                  f"gripper={'close' if gripper > 0 else 'open'}, "
                  f"hold={hold_steps}")

        steps_at_wp = 0
        held = 0

        while total_steps < max_steps:
            state = env.sim.get_state().flatten()
            action = compute_action(env, target_pose, gripper)
            obs, reward, done, info = env.step(action)

            states.append(state)
            actions.append(action)
            total_steps += 1
            steps_at_wp += 1

            eef_pose = get_eef_pose(env)
            eef_pos, _ = unmake_pose(eef_pose)
            dist = np.linalg.norm(eef_pos - target_pos)

            # For hold waypoints (gripper open/close), just count steps
            if hold_steps > 0:
                held += 1
                if held >= hold_steps:
                    break
            else:
                if dist < dist_thresh:
                    break

            # Safety: don't spend too long on one waypoint
            if steps_at_wp > 150:
                if debug:
                    print(f"    Waypoint {wp_idx + 1} timeout at step {total_steps}, "
                          f"dist={dist:.4f}")
                return states, actions, False

        if debug:
            eef_pose_after = get_eef_pose(env)
            eef_after, _ = unmake_pose(eef_pose_after)
            bottle_after = get_body_pos(env, bottle_body)
            print(f"    WP{wp_idx+1} done: eef={eef_after}, "
                  f"bottle={bottle_after}, steps={steps_at_wp}")

        if total_steps >= max_steps:
            if debug:
                print(f"  Max steps reached at waypoint {wp_idx + 1}")
            break

    # Remove last state to align states/actions (states[i] -> action[i] -> states[i+1])
    if len(states) > len(actions):
        del states[-1]
    assert len(states) == len(actions), \
        f"states ({len(states)}) != actions ({len(actions)})"

    body_id = env.sim.model.body_name2id(bottle_body)
    upright = env.sim.data.body_xmat[body_id].reshape(3, 3)[2, 2] > 0.90
    success = env._check_success() and upright and total_steps < max_steps
    if debug:
        print(f"  Total steps: {total_steps}, Success: {success}")

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


EXPECTED_MAX_STEPS = 160  # Validation: 135–139 steps across 20 randomized rollouts, plus margin.

def main():
    parser = argparse.ArgumentParser(
        description="Scripted waypoint demo collection (Task 55: wine bottle → plate)")
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
    parser.add_argument("--place_offset", type=float, default=0.015,
                        help="Z offset above plate for place (tune this)")
    parser.add_argument("--grasp_offset", type=float, default=0.14,
                        help="Height above bottle body center for neck grasp (meters)")
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
    print("Scripted Waypoint Demo Collection (Task 55)")
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
    bottle_body, plate_body = discover_names(env, debug=args.debug)
    print(f"  Wine bottle body: {bottle_body}")
    print(f"  Plate body:        {plate_body}")

    # Collect demos
    demos = []
    model_poses = []
    model_body_names = [env.sim.model.body_id2name(i) for i in range(env.sim.model.nbody)]
    successes = 0
    failures = 0

    for i in range(num_demos):
        print(f"\n--- Demo {i + 1}/{num_demos} ---")
        states, actions, success = collect_one_demo(
            env, bottle_body, plate_body,
            max_steps=args.max_steps,
            debug=args.debug,
            place_offset=args.place_offset,
            grasp_offset=args.grasp_offset,
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
