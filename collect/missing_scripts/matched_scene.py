"""Exact scene snapshots for matched LIBERO collection (no policy actions)."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np


PHYSICS_FIELDS = ("ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied",
                  "mocap_pos", "mocap_quat", "userdata")
PHYSICAL_BDDL_FIELDS = ("problem_name", "fixtures", "objects", "regions",
                        "scene_properties", "initial_state")


def physical_definition(parsed):
    """Exclude task semantics, retaining every physical initialization field."""
    fields = {key: parsed[key] for key in PHYSICAL_BDDL_FIELDS}
    fields["initial_state"] = sorted(fields["initial_state"])
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def model_layout(env):
    model = env.sim.model
    names = {}
    for kind, count in (("body", model.nbody), ("joint", model.njnt),
                        ("geom", model.ngeom), ("site", model.nsite),
                        ("camera", model.ncam), ("actuator", model.nu)):
        names[kind] = [getattr(model, f"{kind}_id2name")(i) for i in range(count)]
    digest = hashlib.sha256()
    for key in ("body_mass", "body_inertia", "body_ipos", "body_iquat",
                "geom_type", "geom_size", "geom_friction", "geom_rgba",
                "geom_matid", "mat_rgba", "cam_pos", "cam_quat",
                "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange",
                "dof_damping", "jnt_axis", "jnt_range"):
        digest.update(key.encode())
        digest.update(np.asarray(getattr(model, key)).tobytes())
    return {"names": names, "nq": model.nq, "nv": model.nv, "na": model.na,
            "static_hash": digest.hexdigest(), "timestep": model.opt.timestep,
            "gravity": np.asarray(model.opt.gravity).tolist(),
            "iterations": model.opt.iterations, "solver": int(model.opt.solver)}


def render_initial(env, camera="agentview", size=256):
    return env.sim.render(size, size, camera_name=camera)[::-1].copy()


def array_hash(arrays, layout):
    digest = hashlib.sha256(json.dumps(layout, sort_keys=True).encode())
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def current_arrays(env):
    arrays = {"state": env.sim.get_state().flatten().copy(),
              "model_body_pos": env.sim.model.body_pos.copy(),
              "model_body_quat": env.sim.model.body_quat.copy()}
    for field in PHYSICS_FIELDS:
        arrays[field] = np.asarray(getattr(env.sim.data, field)).copy()
    for i, robot in enumerate(env.robots):
        arrays[f"gripper_action_{i}"] = robot.gripper.current_action.copy()
    return arrays


def reset_controller_goals(env):
    for robot in env.robots:
        robot.controller.update_initial_joints(robot.init_qpos)
        robot.controller.reset_goal()


@dataclass
class SceneSnapshot:
    scene_id: str
    arrays: dict
    layout: dict
    scene_hash: str
    seed: int
    init_index: int
    image: object = None
    camera: str = "agentview"
    image_size: int = 256

    @classmethod
    def capture(cls, env, scene_id, seed, init_index, verify_images=True,
                camera="agentview", image_size=256):
        reset_controller_goals(env)
        image = render_initial(env, camera, image_size) if verify_images else None
        arrays = current_arrays(env)
        layout = model_layout(env)
        return cls(scene_id, arrays, layout, array_hash(arrays, layout), seed,
                   init_index, image, camera, image_size)

    def restore(self, env, verify_images=True):
        """Reset bookkeeping, restore model + integration state, then assert equality."""
        env.reset()
        if model_layout(env) != self.layout:
            raise ValueError(f"{self.scene_id}: incompatible physical model or joint layout")
        env.sim.model.body_pos[:] = self.arrays["model_body_pos"]
        env.sim.model.body_quat[:] = self.arrays["model_body_quat"]
        env.sim.set_state_from_flattened(self.arrays["state"])
        env.sim.forward()
        reset_controller_goals(env)
        # mj_forward refreshes derived arrays. Restore integration inputs afterwards.
        for field in PHYSICS_FIELDS:
            getattr(env.sim.data, field)[:] = self.arrays[field]
        for i, robot in enumerate(env.robots):
            robot.gripper.current_action = self.arrays[f"gripper_action_{i}"].copy()
        env._post_process()
        env._update_observables(force=True)
        obs = env._get_observations()
        actual = current_arrays(env)
        if array_hash(actual, self.layout) != self.scene_hash:
            different = [key for key in actual if not np.array_equal(actual[key], self.arrays[key])]
            raise AssertionError(f"{self.scene_id}: restored fields differ: {different}")
        if verify_images:
            if self.image is None:
                raise ValueError("Scene bank has no reference image; use --no-verify-images")
            if not np.array_equal(render_initial(env, self.camera, self.image_size), self.image):
                raise AssertionError(f"{self.scene_id}: initial camera image differs")
        return obs

    def save(self, group):
        group.attrs.update(scene_id=self.scene_id, scene_hash=self.scene_hash,
                           layout=json.dumps(self.layout), seed=self.seed,
                           init_index=self.init_index, camera=self.camera,
                           image_size=self.image_size)
        for key, value in self.arrays.items():
            group.create_dataset(key, data=value)
        if self.image is not None:
            group.create_dataset("initial_image", data=self.image, compression="gzip")

    @classmethod
    def load(cls, group):
        layout = json.loads(group.attrs["layout"])
        arrays = {key: group[key][:] for key in group if key != "initial_image"}
        scene_hash = array_hash(arrays, layout)
        if scene_hash != group.attrs["scene_hash"]:
            raise ValueError(f"Corrupt scene snapshot: {group.name}")
        return cls(str(group.attrs["scene_id"]), arrays, layout, scene_hash,
                   int(group.attrs["seed"]), int(group.attrs["init_index"]),
                   group["initial_image"][:] if "initial_image" in group else None,
                   str(group.attrs["camera"]), int(group.attrs["image_size"]))
