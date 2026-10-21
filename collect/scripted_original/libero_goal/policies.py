"""Robot-action policies for the seven originals in the paired Goal BDDLs."""
import importlib.util
from pathlib import Path

import numpy as np

COLLECT = Path(__file__).resolve().parents[2]


def load_helpers(name, relative):
    spec = importlib.util.spec_from_file_location(name, COLLECT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pose = load_helpers("original_pose", "missing_scripts/scripted_collect_task80.py")
drawer = load_helpers("original_drawer", "missing_scripts/scripted_collect_task56.py")
hook_drawer = load_helpers("original_hook_drawer", "missing_scripts/scripted_collect_task57.py")
ROT = pose.GRASP_ROT


class Trajectory:
    def __init__(self, env, max_steps, debug=False):
        self.env, self.max_steps, self.debug = env, max_steps, debug
        self.states, self.actions = [], []

    def move(self, position, grip=-1, rotation=None, tolerance=.01, hold=0, angle_tolerance=.12):
        target = pose.make_pose(position, ROT if rotation is None else rotation)
        for step in range(180):
            if len(self.actions) >= self.max_steps:
                return False
            self.states.append(self.env.sim.get_state().flatten().copy())
            action = pose.compute_action(self.env, target, grip)
            self.actions.append(action)
            self.env.step(action)
            current = pose.get_eef_pose(self.env)
            distance = np.linalg.norm(current[:3, 3] - position)
            error = target[:3, :3] @ current[:3, :3].T
            angle = np.arccos(np.clip((np.trace(error) - 1) / 2, -1, 1))
            if (hold and step + 1 >= hold) or (
                    not hold and distance < tolerance and angle < angle_tolerance):
                return True
        if self.debug:
            print(f"Waypoint timeout: target={position}, distance={distance:.4f}, angle={angle:.3f}")
        return False

    def finish(self, completed):
        current = pose.get_eef_pose(self.env)
        settled = completed and self.move(current[:3, 3], rotation=current[:3, :3], hold=25)
        return self.states, self.actions, bool(settled and self.env._check_success())


def body_pos(env, name):
    return np.array(env.sim.data.body_xpos[env.obj_body_id[name]]).copy()


def site_pos(env, name):
    return np.array(env.sim.data.get_site_xpos(name)).copy()


def open_drawer(t, level):
    env = t.env
    model = env.sim.model
    joint = next(model.joint_id2name(i) for i in range(model.njnt)
                 if model.joint_id2name(i).endswith(f"{level}_level"))
    jid = model.joint_name2id(joint)
    bid = model.jnt_bodyid[jid]
    geoms = [i for i in range(model.ngeom)
             if model.geom_bodyid[i] == bid and model.geom_group[i] == 0]
    gid = min(geoms, key=lambda i: model.geom_pos[i, 1])
    outward = -np.array(env.sim.data.xaxis[jid]).copy()
    handle = np.array(env.sim.data.geom_xpos[gid]).copy()
    matrix = np.array(env.sim.data.geom_xmat[gid]).reshape(3, 3)
    bar = matrix[:, np.argmax(model.geom_size[gid])]
    up = np.array(env.sim.data.body_xmat[bid]).reshape(3, 3)[:, 2]
    rot = drawer.handle_grasp_rotation(bar, outward, up, 60.)
    handle += outward * .009
    ok = t.move(handle + outward * .12, rotation=rot)
    ok = ok and t.move(handle, rotation=rot, tolerance=.008)
    ok = ok and t.move(handle, +1, rot, hold=12)
    for distance in np.linspace(.025, .15, 6):
        ok = ok and t.move(handle + outward * distance, +1, rot)
    pulled = handle + outward * .15
    ok = ok and t.move(pulled, -1, rot, hold=10)
    ok = ok and t.move(pulled + outward * .09, -1, rot)
    ok = ok and t.move(pulled + outward * .09 + up * .15, -1, rot)
    ok = ok and t.move(np.array([-.20, .0, 1.25]))
    return bool(ok and float(env.sim.data.get_joint_qpos(joint)) < -.12), outward


def place_bowl(t, target, destination):
    env = t.env
    initial = body_pos(env, "akita_black_bowl_1")
    rim = -.048
    grasp = initial + np.array([0., rim, .03])
    lift = grasp + np.array([0., 0., .24])
    ok = t.move(lift)
    # Re-read after approach: small objects can finish settling during this move.
    grasp = body_pos(env, "akita_black_bowl_1") + np.array([0., rim, .03])
    ok = ok and t.move(grasp, tolerance=.005)
    ok = ok and t.move(grasp, +1, hold=12)
    ok = ok and t.move(lift, +1)
    lifted = body_pos(env, "akita_black_bowl_1")
    if not ok or lifted[2] < initial[2] + .1:
        return False
    offset = pose.get_eef_pose(env)[:3, 3] - lifted
    place = target + offset
    # Body origin is at the bowl base; preserve its table resting offset.
    clearance = .015 if destination == "staging" else .04
    place[2] += initial[2] - site_pos(env, "table_top")[2] + clearance
    if destination == "drawer":
        place[2] -= .0225  # region center is above the drawer floor
    elif destination == "plate":
        # The plate is beyond the low vertical-grasp workspace. Release above
        # it from a reachable pose and allow the bowl to settle onto the plate.
        place[2] = max(place[2], 1.085)
    carry = place.copy()
    carry[2] = max(lift[2], place[2] + .14)
    ok = t.move(carry, +1)
    ok = ok and t.move(place, +1, tolerance=.015)
    ok = ok and t.move(place, -1, hold=15)
    return ok and t.move(carry)


def push_plate(t):
    env = t.env
    target = site_pos(env, "main_table_stove_front_region")
    # This wrist yaw reaches the low, far edge of the plate without driving
    # the Panda wrist into its joint limit.
    rot = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]) @ ROT
    if not t.move(np.array([-.20, .0, 1.20]), +1, rot):
        return False
    # Closed fingers push the edge horizontally, keeping the plate on the table.
    for _ in range(3):
        plate = body_pos(env, "plate_1")
        delta = target[:2] - plate[:2]
        distance = np.linalg.norm(delta)
        if distance < .012:
            return True
        direction = np.r_[delta / distance, 0.]
        contact = plate - direction * .065
        contact[2] = plate[2] + .012
        above = contact + np.array([0., 0., .13])
        if not (t.move(above, +1, rot) and t.move(contact, +1, rot)):
            return False
        # Include the initial gap between the fingertip and plate rim.
        travel = distance + .035
        for shift in np.linspace(0., travel, max(2, int(travel / .015))):
            if not t.move(contact + direction * shift, +1, rot, tolerance=.006):
                return False
        end = pose.get_eef_pose(env)[:3, 3]
        if not t.move(end + np.array([0., 0., .13]), +1, rot):
            return False
    return True


def place_wine_on_rack(t):
    env = t.env
    initial = body_pos(env, "wine_bottle_1")
    grasp = initial + np.array([0., 0., .14])
    lift = grasp + np.array([0., 0., .23])
    ok = t.move(lift) and t.move(grasp, tolerance=.004)
    ok = ok and t.move(grasp, +1, hold=12) and t.move(lift, +1)
    lifted = body_pos(env, "wine_bottle_1")
    if not ok or lifted[2] < initial[2] + .10:
        return False
    # Lay the bottle across the sloped rack. Track the measured rigid grasp
    # transform while rotating so its center, rather than the wrist, is placed.
    bid = env.obj_body_id["wine_bottle_1"]
    body_rot = np.array(env.sim.data.body_xmat[bid]).reshape(3, 3)
    ee = pose.get_eef_pose(env)
    relative_rot = body_rot.T @ ee[:3, :3]
    relative_pos = body_rot.T @ (ee[:3, 3] - lifted)
    rack_rot = np.array(env.sim.data.get_site_xmat("wine_rack_1_top_region"))
    axis = rack_rot[:, 0]
    up = rack_rot[:, 2]
    desired_body_rot = np.column_stack((up, np.cross(axis, up), axis))
    rot = desired_body_rot @ relative_rot
    center = site_pos(env, "wine_rack_1_top_region") + up * .025
    place = center + desired_body_rot @ relative_pos
    carry = place + np.array([0., 0., .20])
    ok = t.move(carry, +1, rot, tolerance=.025, angle_tolerance=.2)
    ok = ok and t.move(place, +1, rot, tolerance=.02)
    ok = ok and t.move(place, -1, rot, hold=20)
    return ok and t.move(carry, -1, rot)


def collect_one_demo(env, task, max_steps=700, debug=False, initialize_scene=None):
    """Return pre-action states, actions, success; never teleport during a policy."""
    if initialize_scene is None:
        env.reset()
        for _ in range(10):
            env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
    else:
        initialize_scene(env)
    t = Trajectory(env, max_steps, debug)
    if task == "open_the_middle_drawer_of_the_cabinet":
        model = env.sim.model
        joint = next(model.joint_id2name(i) for i in range(model.njnt)
                     if model.joint_id2name(i).endswith("middle_level"))
        bid = model.jnt_bodyid[model.joint_name2id(joint)]
        geoms = [i for i in range(model.ngeom)
                 if model.geom_bodyid[i] == bid and model.geom_group[i] == 0]
        handle = min(geoms, key=lambda i: model.geom_pos[i, 1])
        return hook_drawer.collect_one_demo(
            env, joint, "wooden_cabinet_1_middle_region", handle,
            max_steps=max_steps, debug=debug, initialize_scene=lambda _: None)
    elif task == "open_the_top_drawer_and_put_the_bowl_inside":
        # The opened drawer covers the bowl's starting XY position. Stage the
        # bowl clear of its travel before opening, then pick it up again.
        staging = site_pos(env, "flat_stove_1_cook_region")
        ok = place_bowl(t, staging, "staging")
        opened, outward = open_drawer(t, "top") if ok else (False, np.zeros(3))
        target = site_pos(env, "wooden_cabinet_1_top_region") + outward * .06
        ok = opened and place_bowl(t, target, "drawer")
    elif task == "push_the_plate_to_the_front_of_the_stove":
        ok = push_plate(t)
    elif task == "put_the_wine_bottle_on_the_rack":
        ok = place_wine_on_rack(t)
    else:
        sites = {
            "put_the_bowl_on_the_plate": "plate_1_default_site",
            "put_the_bowl_on_the_stove": "flat_stove_1_cook_region",
            "put_the_bowl_on_top_of_the_cabinet": "wooden_cabinet_1_top_side",
        }
        destination = "plate" if task == "put_the_bowl_on_the_plate" else "surface"
        ok = place_bowl(t, site_pos(env, sites[task]), destination)
    return t.finish(ok)
