import logging
import math
from typing import Any

import numpy as np

from robocasa_geometry import normalize, orientation_from_direction
from robocasa_tasks import (
    FIXTURE_PRIORITY,
    TASK_FRONT_APPROACH,
    TASK_FRONT_DISTANCE,
    TASK_SKIP_EE,
)

try:
    from robosuite.utils import transform_utils as _T
    from robosuite.utils.ik_utils import IKSolver as _IKSolver, get_nullspace_gains as _get_nullspace_gains
except Exception:  # pragma: no cover - IK not available
    _T = None
    _IKSolver = None
    _get_nullspace_gains = None
try:
    from robocasa.utils import object_utils as _object_utils
except Exception:  # pragma: no cover - object utils unavailable
    _object_utils = None


def prepare_ik_context(env) -> dict | None:
    if _IKSolver is None or _T is None or _get_nullspace_gains is None:
        return None
    try:
        robot = env.robots[0]
    except Exception:
        return None
    if not getattr(robot, "arms", None):
        return None
    arm = robot.arms[0]
    gripper = getattr(robot, "gripper", {}).get(arm)
    eef_site = None
    if gripper is not None:
        eef_site = gripper.important_sites.get("grip_site")
    if not eef_site:
        eef_site = robot.robot_model.eef_name.get(arm) if hasattr(robot.robot_model, "eef_name") else None
    if not eef_site:
        return None
    sim = env.sim
    try:
        site_id = sim.model.site_name2id(eef_site)
    except Exception:
        return None
    model = getattr(sim.model, "_model", getattr(sim.model, "model", sim.model))
    data = getattr(sim.data, "_data", sim.data)
    try:
        current_pos = np.array(data.site_xpos[site_id], dtype=float)
        current_rot = np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)
    except Exception:
        return None
    joint_names = list(getattr(robot.robot_model, "arm_joints", []))
    if not joint_names:
        return None
    nullspace = _get_nullspace_gains(joint_names, {})
    joint_index = {name: idx for idx, name in enumerate(getattr(robot.robot_model, "joints", []))}
    return {
        "env": env,
        "robot": robot,
        "eef_site": eef_site,
        "site_id": site_id,
        "model": model,
        "data": data,
        "current_pos": current_pos,
        "current_rot": current_rot,
        "joint_names": joint_names,
        "nullspace": nullspace,
        "joint_index": joint_index,
    }


def ik_move_to_pose(
    ctx: dict,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    *,
    max_iters: int = 30,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    robot = ctx["robot"]
    model = ctx["model"]
    data = ctx["data"]
    site_id = ctx["site_id"]
    joint_names = ctx["joint_names"]
    nullspace = ctx["nullspace"]
    joint_index = ctx["joint_index"]
    try:
        target_quat_xyzw = _T.mat2quat(target_rot)
    except Exception:
        return False, None, None
    target_quat_wxyz = np.roll(target_quat_xyzw, 1)
    robot_config = {
        "end_effector_sites": [ctx["eef_site"]],
        "joint_names": joint_names,
        "mocap_bodies": [],
        "nullspace_gains": nullspace,
    }
    try:
        ik_solver = _IKSolver(
            model,
            data,
            robot_config,
            damping=5e-2,
            integration_dt=0.05,
            max_dq=2.0,
            input_type="keyboard",
            input_rotation_repr="quat_wxyz",
        )
    except Exception:
        return False, None, None
    target_action = np.concatenate([np.asarray(target_pos, dtype=float), target_quat_wxyz]).astype(float)
    success = False
    final_pos = None
    final_rot = None
    for _ in range(max_iters):
        ik_solver.q0 = data.qpos[ik_solver.dof_ids].copy()
        try:
            q_des = ik_solver.solve(target_action, Kpos=0.8, Kori=0.8)
        except Exception:
            break
        try:
            full_joints = robot._joint_positions.copy()
        except Exception:
            break
        for name, value in zip(joint_names, np.array(q_des, dtype=float)):
            idx = joint_index.get(name)
            if idx is not None:
                full_joints[idx] = value
        try:
            robot.set_robot_joint_positions(full_joints)
            ctx["env"].sim.forward()
        except Exception:
            break
        pos_err = float(np.linalg.norm(data.site_xpos[site_id] - target_pos))
        cur_rot = np.array(data.site_xmat[site_id], dtype=float).reshape(3, 3)
        rot_err = float(np.linalg.norm(cur_rot - target_rot))
        if pos_err < 5e-3 and rot_err < 5e-2:
            success = True
            final_pos = np.array(data.site_xpos[site_id], dtype=float)
            final_rot = cur_rot
            break
    if success:
        try:
            robot.composite_controller.update_state()
        except Exception:
            pass
    return success, final_pos, final_rot


def fixture_focus_points(fixture: Any) -> dict | None:
    if _object_utils is None:
        return None
    try:
        points_rel = np.asarray(fixture.get_ext_sites(all_points=True, relative=True), dtype=float)
    except Exception:
        return None
    if points_rel.size == 0:
        return None
    mins = points_rel.min(axis=0)
    maxs = points_rel.max(axis=0)
    center_rel = 0.5 * (mins + maxs)
    try:
        top_offset = np.array([center_rel[0], center_rel[1], maxs[2]], dtype=float)
        top_center = _object_utils.get_pos_after_rel_offset(fixture, top_offset)
        front_offset = np.array([center_rel[0], maxs[1], center_rel[2]], dtype=float)
        front_center = _object_utils.get_pos_after_rel_offset(fixture, front_offset)
        center = _object_utils.get_pos_after_rel_offset(fixture, center_rel)
    except Exception:
        return None
    front_normal = np.array([math.cos(fixture.rot), math.sin(fixture.rot), 0.0], dtype=float)
    if np.linalg.norm(front_normal[:2]) < 1e-6:
        front_normal = np.array([0.0, 1.0, 0.0], dtype=float)
    front_normal = normalize(front_normal)
    return {
        "top_center": top_center,
        "front_center": front_center,
        "center": center,
        "front_normal": front_normal,
        "height": float(maxs[2] - mins[2]),
    }


def set_pose_from_fixture(env, fixture, task_name: str, ctx: dict) -> tuple[bool, dict]:
    focus = fixture_focus_points(fixture)
    if focus is None:
        return False, {}
    approach = "front" if task_name in TASK_FRONT_APPROACH else "top"
    info: dict[str, np.ndarray | str | float] = {"approach": approach}
    if approach == "front":
        distance = TASK_FRONT_DISTANCE.get(task_name, 0.25)
        height = max(0.10, focus["height"] * 0.25)
        focus_point = focus["front_center"]
        front_normal = focus["front_normal"]
        target_pos = np.asarray(focus_point, dtype=float) - front_normal * distance
        target_pos[2] = max(target_pos[2], focus_point[2] + height)
        approach_dir = focus_point - target_pos
    else:
        height = max(0.25, focus["height"] * 0.5)
        focus_point = focus["top_center"]
        target_pos = np.asarray(focus_point, dtype=float) + np.array([0.0, 0.0, height], dtype=float)
        approach_dir = focus_point - target_pos
    if np.linalg.norm(approach_dir) < 1e-6:
        approach_dir = np.array([0.0, 0.0, -1.0], dtype=float)
    target_rot = orientation_from_direction(approach_dir, up_hint=np.array([0.0, 0.0, 1.0], dtype=float))
    success, final_pos, final_rot = ik_move_to_pose(ctx, target_pos, target_rot)
    if not success or final_pos is None or final_rot is None:
        return False, {}
    info["focus_point"] = np.asarray(focus_point, dtype=float)
    info["target_pos"] = np.asarray(target_pos, dtype=float)
    info["final_pos"] = final_pos
    info["final_rot"] = final_rot
    return True, info


def initialize_eef_for_task(env, task_name: str) -> dict | None:
    if task_name in TASK_SKIP_EE:
        return None
    ctx = prepare_ik_context(env)
    if ctx is None:
        fallback_raise_and_pitch(env)
        return None
    fixture = None
    fixture_refs = getattr(env, "fixture_refs", {}) or {}
    for key in FIXTURE_PRIORITY:
        fixture = fixture_refs.get(key)
        if fixture is not None:
            break
    if fixture is None:
        init_ref = getattr(env, "init_robot_base_pos", None)
        if init_ref is not None:
            try:
                fixture = env.get_fixture(init_ref)
            except Exception:
                fixture = None
    if fixture is None:
        set_initial_ee_pose(env, height_offset=0.20, downward_deg=45.0)
        return None
    success, info = set_pose_from_fixture(env, fixture, task_name, ctx)
    if not success:
        set_initial_ee_pose(env, height_offset=0.20, downward_deg=45.0)
        return None
    info["fixture"] = fixture
    return info


def set_initial_ee_pose(
    env,
    *,
    height_offset: float = 0.50,
    downward_deg: float = 45.0,
    max_iters: int = 30,
) -> None:
    """Lift the EE and pitch it downward by a fixed angle using IK (with heuristic fallback)."""
    ctx = prepare_ik_context(env)
    if ctx is None:
        fallback_raise_and_pitch(env)
        return
    if height_offset <= 0.0:
        return
    current_pos = ctx["current_pos"]
    current_rot = ctx["current_rot"]
    target_pos = current_pos + np.array([0.0, 0.0, float(height_offset)], dtype=float)
    down_axis = np.array([0.0, 0.0, -1.0], dtype=float)
    cur_z = current_rot[:, 2]
    dot_val = float(np.clip(np.dot(down_axis, cur_z), -1.0, 1.0))
    delta = math.acos(dot_val)
    down_rad = math.radians(max(0.0, float(downward_deg)))
    if delta < 1e-3:
        target_z = down_axis
    else:
        axis = np.cross(down_axis, cur_z)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            axis = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            axis /= axis_norm
        rotate_angle = min(delta, down_rad)
        cos_t = math.cos(rotate_angle)
        sin_t = math.sin(rotate_angle)
        target_z = (
            down_axis * cos_t
            + np.cross(axis, down_axis) * sin_t
            + axis * np.dot(axis, down_axis) * (1.0 - cos_t)
        )
    target_x = current_rot[:, 0] - np.dot(current_rot[:, 0], target_z) * target_z
    if np.linalg.norm(target_x) < 1e-6:
        target_x = np.array([0.0, 1.0, 0.0], dtype=float)
    target_x = normalize(target_x)
    target_y = normalize(np.cross(target_z, target_x))
    target_z = normalize(target_z)
    target_rot = np.column_stack([target_x, target_y, target_z])
    success, final_pos, final_rot = ik_move_to_pose(ctx, target_pos, target_rot, max_iters=max_iters)
    if not success or final_pos is None or final_rot is None:
        fallback_raise_and_pitch(env)
        return
    z_axis = final_rot[:, 2]
    down = np.array([0.0, 0.0, -1.0], dtype=float)
    try:
        angle = math.degrees(math.acos(np.clip(np.dot(normalize(z_axis), down), -1.0, 1.0)))
        logging.info(
            "EE initialized at %s | downward angle %.1f° (target %.1f°)",
            final_pos.round(3).tolist(),
            angle,
            downward_deg,
        )
    except Exception:
        logging.info("EE initialized at %s", final_pos.round(3).tolist())


def fallback_raise_and_pitch(env) -> None:
    """Fallback heuristic using hard-coded joint offsets if IK is unavailable."""
    try:
        robot = env.robots[0]
    except Exception:
        return
    try:
        base_q = np.array(robot._joint_positions, dtype=float)
    except Exception:
        return
    if base_q.size < 7:
        return
    tweak = base_q.copy()
    tweak[:7] = np.array([0.0, -0.9, 0.0, -2.0, 0.0, 1.7, -1.0], dtype=float)
    try:
        robot.set_robot_joint_positions(tweak)
        env.sim.forward()
    except Exception:
        return
    logging.debug("Applied fallback EE pose adjustment (heuristic joint preset).")
    try:
        robot.composite_controller.update_state()
    except Exception:
        pass
