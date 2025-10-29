import math
import numpy as np


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return the vector scaled to unit length (with numerical guard)."""
    vec = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vec)
    return vec / (norm + eps)


def mat_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix into a quaternion (wxyz ordering)."""
    R = np.asarray(rotation, dtype=float)
    t = np.trace(R)
    if t > 0.0:
        S = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
    quat = np.array([w, x, y, z], dtype=float)
    return quat / (np.linalg.norm(quat) + 1e-8)


def quat_wxyz_to_mat(quaternion: np.ndarray) -> np.ndarray:
    """Convert a quaternion (wxyz ordering) into a 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quaternion, dtype=float)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )


def orientation_from_direction(direction: np.ndarray, *, up_hint: np.ndarray | None = None) -> np.ndarray:
    """Generate a rotation matrix whose z-axis follows `direction` (pointing towards the target)."""
    direction_vec = np.asarray(direction, dtype=float)
    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=float)
    up_vec = np.asarray(up_hint, dtype=float)
    if np.linalg.norm(direction_vec) < 1e-6:
        direction_vec = np.array([0.0, 0.0, -1.0], dtype=float)
    z_axis = normalize(direction_vec)
    up_axis = normalize(up_vec)
    if abs(float(np.dot(z_axis, up_axis))) > 0.98:
        up_axis = normalize(np.array([1.0, 0.0, 0.0], dtype=float))
    x_axis = np.cross(up_axis, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        up_axis = normalize(np.array([0.0, 1.0, 0.0], dtype=float))
        x_axis = np.cross(up_axis, z_axis)
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack([x_axis, y_axis, z_axis])
