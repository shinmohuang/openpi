import dataclasses
import pathlib
from typing import Annotated, Sequence

import tyro


@dataclasses.dataclass
class Args:
    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Evaluation
    task_names: Sequence[str] | None = None  # If None, sample from SINGLE_STAGE tasks
    use_multi_stage: bool = False  # Use multi-stage task registry instead of single-stage
    num_trials_per_task: int = 5
    replan_steps: int = 15  # how many steps from action chunk to execute before replanning

    # Environment
    camera_size: int = 512
    camera_names: Sequence[str] = (
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    )
    robots: str = "PandaOmron"
    randomize_cameras: bool = False
    # A list of (layout_id, style_id) tuples to sample from; if empty, use eval_utils defaults
    layout_and_style_ids: Sequence[tuple[int, int]] | None = None
    camera_resize: int | None = 224  # Resize camera feeds before sending to policy (None = keep original)
    include_right_camera: bool = True  # Send the right agentview image to the policy

    # Limits & output
    max_steps_override: int | None = 1500  # If set, overrides task horizon
    video_out_dir: str | None = "rollouts/robocasa"
    model_name: str = "lora_norm_1"  # Used for naming rollout directories
    num_workers: int = 1
    eval_all: Annotated[bool, tyro.conf.arg(name="all")] = False

    # Reproducibility
    seed: int = 7
