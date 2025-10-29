import numpy as np

from openpi_client import image_tools


def obs_to_robocasa_inputs(
    obs: dict,
    prompt: str,
    *,
    base_camera: str = "robot0_agentview_left",
    right_camera: str | None = "robot0_agentview_right",
    wrist_camera: str = "robot0_eye_in_hand",
    resize_hw: int | tuple[int, int] | None = None,
    include_right_camera: bool = True,
    flip_vertical: bool = True,
) -> dict:
    """Pack RoboCasa observations into the structure expected by RobocasaPolicy transforms."""

    def _get_image_for(camera: str | None, fallbacks: tuple[str, ...]) -> np.ndarray:
        keys = []
        if camera:
            keys.append(f"{camera}_image")
        keys.extend(fallbacks)
        for key in keys:
            img = obs.get(key)
            if img is not None:
                return np.asarray(img)
        raise KeyError(f"Missing camera image keys {keys} in observation")

    def _process_image(image: np.ndarray) -> np.ndarray:
        img_uint8 = image_tools.convert_to_uint8(image)
        if resize_hw is None:
            return img_uint8
        if isinstance(resize_hw, tuple):
            target_h, target_w = resize_hw
        else:
            target_h = target_w = int(resize_hw)
        return image_tools.resize_with_pad(img_uint8, target_h, target_w)

    base_img = _get_image_for(
        base_camera,
        (
            "robot0_agentview_left_image",
        ),
    )
    if flip_vertical:
        base_img = np.flipud(np.asarray(base_img))
    right_img = None
    if include_right_camera and right_camera is not None:
        right_img = _get_image_for(
            right_camera,
            (
                "robot0_agentview_right_image",
            ),
        )
        if flip_vertical and right_img is not None:
            right_img = np.flipud(np.asarray(right_img))
    wrist_img = _get_image_for(
        wrist_camera,
        (
            "robot0_eye_in_hand_image",
        ),
    )
    if flip_vertical:
        wrist_img = np.flipud(np.asarray(wrist_img))

    required_keys = (
        "robot0_base_pos",
        "robot0_base_quat",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    missing_state = [k for k in required_keys if obs.get(k) is None]
    if missing_state:
        raise KeyError(f"Missing expected proprio keys in observation: {missing_state}")

    base_pos = np.asarray(obs["robot0_base_pos"], dtype=np.float32).reshape(-1)
    base_quat = np.asarray(obs["robot0_base_quat"], dtype=np.float32).reshape(-1)
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)

    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    if gripper_qpos.size == 0:
        raise ValueError("robot0_gripper_qpos is empty")
    gripper_scalar = float(np.mean(gripper_qpos))

    state = np.concatenate(
        [base_pos[:3], base_quat[:4], eef_pos[:3], eef_quat[:4], np.array([gripper_scalar], dtype=np.float32)],
        axis=0,
    ).astype(np.float32, copy=False)

    base_img = _process_image(base_img)
    wrist_img = _process_image(wrist_img)
    if right_img is not None:
        right_img = _process_image(right_img)

    inputs: dict[str, object] = {
        "observation/image": base_img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
        "observation/gripper_position": np.array([gripper_scalar], dtype=np.float32),
        "prompt": prompt,
    }
    if right_img is not None:
        inputs["observation/image_right"] = right_img

    return inputs
