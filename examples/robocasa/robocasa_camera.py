import numpy as np

from robocasa_geometry import mat_to_quat_wxyz, normalize, quat_wxyz_to_mat


def lookat_quat_wxyz(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Construct a MuJoCo camera quaternion (wxyz) that looks from `eye` towards `target`."""
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    eye_vec = np.asarray(eye, dtype=float)
    target_vec = np.asarray(target, dtype=float)
    up_vec = normalize(np.asarray(up, dtype=float))

    forward = normalize(target_vec - eye_vec)
    if abs(float(np.dot(forward, up_vec))) > 0.98:
        up_vec = normalize(np.array([0.0, 1.0, 0.0], dtype=float))
    right = normalize(np.cross(forward, up_vec))
    true_up = normalize(np.cross(right, forward))
    rotation = np.stack([right, true_up, -forward], axis=1)
    return mat_to_quat_wxyz(rotation)


def set_camera_pose(env, camera_name: str, cam_pos_world: np.ndarray, lookat: np.ndarray) -> bool:
    """Override a MuJoCo camera pose on the environment model."""
    try:
        cam_id = env.sim.model.camera_name2id(camera_name)
    except Exception:
        return False

    cam_quat_world = lookat_quat_wxyz(cam_pos_world, lookat, up=np.array([0.0, 0.0, 1.0]))
    cam_body_id = int(env.sim.model.cam_bodyid[cam_id]) if hasattr(env.sim.model, "cam_bodyid") else -1

    if cam_body_id >= 0:
        body_pos = env.sim.data.xpos[cam_body_id].copy()
        body_xmat = env.sim.data.xmat[cam_body_id].reshape(3, 3).copy()
        cam_pos_local = body_xmat.T @ (np.asarray(cam_pos_world, dtype=float) - body_pos)
        R_cam_world = quat_wxyz_to_mat(cam_quat_world)
        R_local = body_xmat.T @ R_cam_world
        cam_quat_local = mat_to_quat_wxyz(R_local)
        env.sim.model.cam_pos[cam_id] = cam_pos_local
        env.sim.model.cam_quat[cam_id] = cam_quat_local
    else:
        env.sim.model.cam_pos[cam_id] = np.asarray(cam_pos_world, dtype=float)
        env.sim.model.cam_quat[cam_id] = cam_quat_world

    if hasattr(env, "_cam_configs"):
        env._cam_configs.setdefault(camera_name, {})
        env._cam_configs[camera_name]["pos"] = np.asarray(cam_pos_world, dtype=float).tolist()
        env._cam_configs[camera_name]["quat"] = cam_quat_world.tolist()

    return True
