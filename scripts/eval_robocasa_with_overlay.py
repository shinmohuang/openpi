#!/usr/bin/env python
"""Evaluate a π0.5 policy on RoboCasa by applying a lightweight RL overlay on top of a base checkpoint.

The overlay is the small checkpoint saved by train_ppo_robocasa.py when --lightweight-save=True.
This script loads the base model weights, applies the overlay (only trainable keys), and runs evaluation.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import os

import numpy as np
import torch
import tyro
import imageio

from openpi.rl import (
    Pi05RobocasaPolicy,
    RobocasaEnvConfig,
    make_robocasa_vector_env,
)
from openpi.training import config as train_config_lib


@dataclasses.dataclass
class Args:
    base_checkpoint_dir: str
    overlay_checkpoint: str
    config_name: str = "pi05_robocasa_pandamobile_lora"
    task_name: str = "PnPCounterToSink"
    use_multi_stage: bool = False
    episodes: int = 5
    seed: int = 0
    device: str = "cuda"
    # Video options (defaults enabled for async eval)
    record_video: bool = True
    video_dir: str | None = None
    fps: int = 10
    # Limits / quality
    max_steps_override: int | None = None
    inference_steps: int = 8
    # Video camera
    video_camera_name: str = "robot0_agentview_left"
    video_cam_pos: str | None = None
    video_lookat: str | None = None
    # Policy input cameras (use standard three views)
    policy_base_camera_name: str = "robot0_agentview_left"
    policy_right_camera_name: str | None = "robot0_agentview_right"
    policy_wrist_camera_name: str | None = "robot0_eye_in_hand"


def _extract_prompts(info, fallback: list[str]) -> list[str]:
    if isinstance(info, dict) and "prompt" in info:
        prompts = info["prompt"]
        if isinstance(prompts, np.ndarray):
            return [str(p) for p in prompts.tolist()]
        if isinstance(prompts, (list, tuple)):
            return [str(p) for p in prompts]
        return [str(prompts)] * len(fallback)
    return list(fallback)


def main(args: Args) -> None:
    # Ensure headless EGL rendering for MuJoCo / robosuite
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_cfg = train_config_lib.get_config(args.config_name)
    model_config = train_cfg.model

    # Build single env for evaluation
    def _parse_vec3(s: str | None) -> tuple[float, float, float] | None:
        if not s:
            return None
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Expected three comma-separated values for a vector, got: {s}")
        return (float(parts[0]), float(parts[1]), float(parts[2]))

    env_cfg = RobocasaEnvConfig(
        task_name=args.task_name,
        use_multi_stage=args.use_multi_stage,
        seed=args.seed,
        max_steps_override=args.max_steps_override,
        video_camera_name=args.video_camera_name,
        video_camera_pos=_parse_vec3(args.video_cam_pos),
        video_lookat=_parse_vec3(args.video_lookat),
        policy_base_camera_name=args.policy_base_camera_name,
        policy_right_camera_name=args.policy_right_camera_name,
        policy_wrist_camera_name=args.policy_wrist_camera_name,
    )
    env = make_robocasa_vector_env(env_cfg, 1)
    obs, info = env.reset()
    prompts = _extract_prompts(info, [""])
    env_action_dim = env.single_action_space.shape[0]
    # Prepare video directory and state
    out_dir = Path(args.video_dir) if args.video_dir else (Path(args.overlay_checkpoint).parent / "videos")
    if args.record_video:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare model action bounds (pad env bounds to model action dim)
    def _pad_bounds(arr: np.ndarray, target_dim: int, fill: float) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape[0] == target_dim:
            return arr
        if arr.shape[0] > target_dim:
            return arr[:target_dim]
        pad = np.full((target_dim - arr.shape[0],), fill, dtype=np.float32)
        return np.concatenate([arr, pad], axis=0)

    model_action_dim = model_config.action_dim
    padded_low = _pad_bounds(env.single_action_space.low, model_action_dim, -1.0)
    padded_high = _pad_bounds(env.single_action_space.high, model_action_dim, 1.0)

    # Load base policy then apply overlay
    policy = Pi05RobocasaPolicy.from_checkpoint(
        args.base_checkpoint_dir,
        model_config=model_config,
        device=device,
        state_dim=model_config.action_dim,
        action_bounds=(padded_low, padded_high),
        inference_steps=args.inference_steps,
        min_std=0.05,
        max_std=0.2,
        include_initial_logprob=True,
    )

    overlay = torch.load(Path(args.overlay_checkpoint), map_location=device)
    overlay_state = overlay.get("policy", overlay)  # support either full payload or raw state dict
    missing, unexpected = policy.load_state_dict(overlay_state, strict=False)
    print(f"Applied overlay: missing={len(missing)} unexpected={len(unexpected)}")

    # Evaluate
    successes = 0
    returns: list[float] = []
    episode_return = 0.0
    completed = 0

    writer = None
    while completed < args.episodes:
        with torch.no_grad():
            actions, _, _ = policy.sample_actions(obs, prompts, eval_mode=True, return_chain=False)
        # Clip to environment action dimension
        action = actions[0, 0, :env_action_dim].cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action[None, :])
        episode_return += float(reward[0])
        done = bool(terminated[0] or truncated[0])
        prompts = _extract_prompts(info, prompts)

        # Record frame
        if args.record_video:
            frame = None
            try:
                frame = env.envs[0].render()
            except Exception:
                frame = None
            if frame is not None:
                if writer is None:
                    tag = Path(args.overlay_checkpoint).stem
                    out_path = out_dir / f"eval_{tag}_ep{completed}.mp4"
                    writer = imageio.get_writer(
                        out_path,
                        format="ffmpeg",
                        mode="I",
                        fps=args.fps,
                        codec="libx264",
                        ffmpeg_params=["-crf", "28", "-preset", "fast"],
                    )
                writer.append_data(frame)

        if done:
            info_dict = info[0] if isinstance(info, (list, tuple)) else info
            success_flag = bool(info_dict.get("success", False)) if isinstance(info_dict, dict) else False
            successes += int(success_flag)
            returns.append(episode_return)
            episode_return = 0.0
            completed += 1
            if writer is not None:
                try:
                    writer.close()
                except Exception:
                    pass
                writer = None
            obs, info = env.reset()
            prompts = _extract_prompts(info, [""])

    env.close()
    avg_return = float(np.mean(returns)) if returns else 0.0
    success_rate = successes / max(args.episodes, 1)
    print(f"Eval results :: success={100*success_rate:.1f}% avg_return={avg_return:.3f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
