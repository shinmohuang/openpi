import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robocasa_example() -> dict:
    """Creates a random input example for the Robocasa policy."""
    return {
        "observation/image": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "observation/image_right": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "observation/wrist_images": np.random.randint(256, size=(128, 128, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    image_keys: tuple[str, ...] = _model.IMAGE_KEYS

    def __call__(self, data: dict) -> dict:
        if "observation/state" in data:
            state = np.asarray(data["observation/state"], dtype=np.float32)
        else:
            gripper_source = (
                data.get("observation/gripper_position")
                or data.get("observation/gripper_qpos")
                or data.get("observation/gripper")
            )
            if gripper_source is None:
                gripper_pos = np.zeros(1, dtype=np.float32)
            else:
                gripper_pos = np.asarray(gripper_source, dtype=np.float32)
                if gripper_pos.ndim == 0:
                    gripper_pos = gripper_pos[np.newaxis]
                else:
                    gripper_pos = gripper_pos.reshape(-1)
                    if gripper_pos.size > 1:
                        gripper_pos = np.array([gripper_pos.mean()], dtype=np.float32)

            joint_source = (
                data.get("observation/joint_position")
                or data.get("observation/joint_positions")
                or data.get("observation/robot0_joint_pos")
            )
            if joint_source is None:
                raise KeyError("observation/joint_position")
            joint = np.asarray(joint_source, dtype=np.float32)
            state = np.concatenate([joint, gripper_pos])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data.get("observation/image", data.get("robot0_agentview_left_image")))
        right_image_raw = data.get("observation/image_right", data.get("robot0_agentview_right_image"))
        wrist_image_raw = data.get("observation/wrist_image", data.get("robot0_eye_in_hand"))
        right_image = _parse_image(right_image_raw) if right_image_raw is not None else None
        wrist_image = _parse_image(wrist_image_raw) if wrist_image_raw is not None else None

        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.bool_] = {}

        for key in self.image_keys:
            match key:
                case "base_0_rgb":
                    source = base_image
                case "left_wrist_0_rgb" | "wrist_0_rgb":
                    source = wrist_image if wrist_image is not None else base_image
                case "right_wrist_0_rgb" | "base_1_rgb":
                    source = right_image if right_image is not None else base_image
                case _:
                    raise ValueError(f"Unsupported camera key: {key}")

            images[key] = source
            image_masks[key] = np.True_

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05 | _model.ModelType.PI0_FAST:
                pass
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": images,
            "image_mask": image_masks,
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 12 dims.
        return {"actions": np.asarray(data["actions"][:, :12])}
