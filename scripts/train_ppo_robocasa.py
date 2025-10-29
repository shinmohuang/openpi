#!/usr/bin/env python
"""Minimal PPO fine-tuning loop for π₀․₅ on RoboCasa using in-repo utilities."""

from __future__ import annotations

import dataclasses
import functools
import math
import os
import subprocess
import time
from pathlib import Path
import imageio
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
import tyro
from torch.nn.parallel import DistributedDataParallel as DDP

from openpi.rl.robocasa_env import RobocasaEnvConfig, make_robocasa_vector_env
from openpi.rl.pi05_policy_wrapper import Pi05RobocasaPolicy
from openpi.rl.ppo_buffer import PPORolloutBuffer
from openpi.rl.value_network import StateValueNetwork
from openpi.training import config as train_config_lib


def _setup_distributed() -> tuple[int, int, int]:
    """Initialise torch.distributed if launched via torchrun."""
    # Make NCCL a bit more robust by default.
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_mean(value: float, device: torch.device, world_size: int) -> float:
    tensor = torch.tensor(value, device=device)
    if dist.is_initialized() and world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return float(tensor.item())


def _distributed_sync_int(value: int, device: torch.device, world_size: int) -> int:
    t = torch.tensor([value], device=device if device.type == "cuda" else "cpu", dtype=torch.int64)
    if dist.is_initialized() and world_size > 1:
        dist.broadcast(t, src=0)
    return int(t.item())


def _cast_transformer_blocks_dtype(module: torch.nn.Module, dtype: torch.dtype, device: torch.device) -> None:
    """Force specific transformer blocks to a uniform dtype to satisfy FSDP flattening.

    This targets HF Gemma and SigLIP encoder layers, which otherwise may mix bf16/fp32
    (e.g., LayerNorm in fp32)."""
    try:
        from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
        from transformers.models.siglip.modeling_siglip import SiglipEncoderLayer
        target_types = (GemmaDecoderLayer, SiglipEncoderLayer)
    except Exception:
        target_types = tuple()

    for sub in module.modules():
        if target_types and isinstance(sub, target_types):
            sub.to(device=device, dtype=dtype)


def _extract_prompts(info, fallback: Sequence[str]) -> list[str]:
    if isinstance(info, dict) and "prompt" in info:
        prompts = info["prompt"]
        if isinstance(prompts, np.ndarray):
            return [str(p) for p in prompts.tolist()]
        if isinstance(prompts, (list, tuple)):
            return [str(p) for p in prompts]
        return [str(prompts)] * len(fallback)
    return list(fallback)


def save_checkpoint(
    log_dir: Path,
    update: int,
    policy: Pi05RobocasaPolicy,
    value_net: StateValueNetwork,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    rank: int,
    *,
    lightweight: bool,
) -> None:
    """Rank-0 checkpoint saving for DDP (no FSDP dependency)."""
    if dist.is_initialized() and rank != 0:
        return
    checkpoint_path = log_dir / f"checkpoint_{update:05d}.pt"
    if lightweight:
        # Save only trainable parameters and buffers belonging to trainable modules (exact module, not whole subtree)
        pol_state_full = policy.state_dict()
        trainable_param_keys = {name for name, p in policy.named_parameters() if p.requires_grad}
        # module prefixes owning trainable params (e.g., 'state_encoder', 'time_embedding', 'explore_noise_net', or specific expert blocks)
        trainable_module_prefixes = {name.rsplit(".", 1)[0] for name in trainable_param_keys if "." in name}
        # Include params explicitly by key
        filtered: dict[str, torch.Tensor] = {k: v for k, v in pol_state_full.items() if k in trainable_param_keys}
        # Include buffers that live exactly under trainable modules (no subtree expansion)
        for buf_name, _ in policy.named_buffers():
            mod_pref = buf_name.rsplit(".", 1)[0] if "." in buf_name else ""
            if mod_pref in trainable_module_prefixes and buf_name in pol_state_full:
                filtered[buf_name] = pol_state_full[buf_name]
        pol_state = filtered
        payload = {
            "update": update,
            "policy": pol_state,
            "value_net": value_net.state_dict(),
        }
    else:
        pol_state = policy.state_dict()
        payload = {
            "update": update,
            "policy": pol_state,
            "value_net": value_net.state_dict(),
            "actor_optim": actor_optimizer.state_dict(),
            "critic_optim": critic_optimizer.state_dict(),
        }
    torch.save(payload, checkpoint_path)


def evaluate_policy(
    policy: Pi05RobocasaPolicy,
    env_cfg: RobocasaEnvConfig,
    episodes: int,
    device: torch.device,
    *,
    record_video: bool = False,
    video_dir: Path | None = None,
    update_tag: int | None = None,
    max_steps_override: int | None = None,
    inference_steps: int | None = None,
) -> tuple[float, float]:
    eval_cfg = dataclasses.replace(
        env_cfg,
        seed=env_cfg.seed + 10_000,
        max_steps_override=max_steps_override if max_steps_override is not None else env_cfg.max_steps_override,
    )
    env = make_robocasa_vector_env(eval_cfg, 1)
    obs, info = env.reset()
    prompts = _extract_prompts(info, [""])
    env_action_dim = env.single_action_space.shape[0]

    successes = 0
    returns: list[float] = []
    episode_return = 0.0
    completed = 0
    writer = None  # lazy video writer

    while completed < episodes:
        with torch.no_grad():
            # Optionally override inference steps during eval
            old_steps = None
            if inference_steps is not None:
                old_steps = getattr(policy, "inference_steps", None)
                policy.inference_steps = inference_steps
            actions, _, _ = policy.sample_actions(obs, prompts, eval_mode=True, return_chain=False)
            if inference_steps is not None and old_steps is not None:
                policy.inference_steps = old_steps
        # Clip to environment action dimension (e.g., 12-D for PANDAMOBILE_12D)
        action = actions[0, 0, :env_action_dim].cpu().numpy()
        obs, reward, terminated, truncated, info = env.step(action[None, :])
        episode_return += float(reward[0])
        done = bool(terminated[0] or truncated[0])
        prompts = _extract_prompts(info, prompts)

        # Write frame if required
        if record_video:
            try:
                # Access underlying single env render
                frame = None
                try:
                    frame = env.envs[0].render()
                except Exception:
                    pass
                if frame is not None:
                    if writer is None:
                        # Lazily open writer for this episode
                        out_dir = video_dir or Path("runs")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        tag = f"upd{update_tag}_" if update_tag is not None else ""
                        out_path = out_dir / f"eval_{tag}ep{completed}.mp4"
                        writer = imageio.get_writer(
                            out_path,
                            format="ffmpeg",
                            mode="I",
                            fps=10,
                            codec="libx264",
                            ffmpeg_params=["-crf", "28", "-preset", "fast"],
                        )
                    writer.append_data(frame)
            except Exception:
                pass

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
    success_rate = successes / max(episodes, 1)
    return success_rate, avg_return


@dataclasses.dataclass
class Args:
    checkpoint: str
    config_name: str = "pi05_robocasa_pandamobile_lora"
    total_updates: int = 500
    rollout_length: int = 128
    num_envs: int = 4
    ppo_epochs: int = 2
    mini_batch_size: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    actor_lr: float = 5e-6
    critic_lr: float = 1e-4
    max_grad_norm: float = 1.0
    inference_steps: int = 6
    min_std: float = 0.05
    max_std: float = 0.2
    include_initial_logprob: bool = True
    noise_hold_ratio: float = 0.3
    noise_decay_ratio: float = 0.7
    task_name: str = "CloseDrawer"
    use_multi_stage: bool = False
    # Scene randomization
    randomize_cameras: bool = False
    # Layout/style pool as comma-separated pairs like "1:0,2:0"; leave empty to use env default
    layout_style_ids: str | None = None
    # If True and a pool is provided, resample (layout,style) at each episode reset
    resample_layout_each_episode: bool = True
    # Curriculum: start with a subset of layouts and expand when eval success >= threshold
    curriculum_enable: bool = True
    curriculum_initial_k: int = 4
    curriculum_expand_threshold: float = 0.30
    curriculum_expand_step: int = 4
    curriculum_max_k: int = 16
    seed: int = 0
    device: str = "cuda"
    log_interval: int = 10
    log_dir: str | None = None
    save_interval: int = 0
    eval_interval: int = 0
    eval_episodes: int = 3
    eval_max_steps: int | None = 200
    eval_inference_steps: int | None = None
    lightweight_save: bool = True
    # Eval video
    eval_record_video: bool = True
    eval_video_dir: str | None = None
    # Async eval on a separate GPU/process so training GPUs keep working
    async_eval: bool = True
    eval_device: int | None = 2  # e.g., 2 for GPU:2; if None uses current device
    # Vectorization controls
    vector_async: bool = True
    vector_context: str | None = "spawn"
    vector_shared_memory: bool = False
    # Video camera controls (for eval videos)
    video_camera_name: str = "robot0_agentview_left"
    video_cam_pos: str | None = None  # e.g., "1.2, -0.3, 1.8"
    video_lookat: str | None = None   # e.g., "0.0, 0.0, 0.9"
    # Policy input camera controls (use standard three views)
    policy_base_camera_name: str = "robot0_agentview_left"
    policy_right_camera_name: str | None = "robot0_agentview_right"
    policy_wrist_camera_name: str | None = "robot0_eye_in_hand"
    # Action expert fine-tuning controls
    keep_lora: bool = False


def train(args: Args, rank: int, world_size: int, local_rank: int):
    is_distributed = world_size > 1
    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
        # Route MuJoCo offscreen rendering to the local GPU when using EGL.
        os.environ.setdefault("MUJOCO_GL", "egl")
        if os.environ.get("MUJOCO_GL", "").lower() == "egl":
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    else:
        device = torch.device("cpu")

    log_dir = Path(args.log_dir or f"runs/ppo_robocasa_{int(time.time())}")
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Logging to {log_dir}")
    if is_distributed:
        dist.barrier()

    train_cfg = train_config_lib.get_config(args.config_name)
    model_config = train_cfg.model

    if rank == 0:
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "unavailable"
            print(f"Using CUDA device: {gpu_name}")
        else:
            print(f"Using device: {device}")

    if is_distributed and args.num_envs % world_size != 0:
        raise ValueError(f"num_envs={args.num_envs} must be divisible by world_size={world_size}")
    local_num_envs = args.num_envs // world_size if is_distributed else args.num_envs

    # Parse layout/style ids if provided (format "L:S,L:S")
    layout_ids_parsed: tuple[tuple[int, int], ...] | None = None
    if args.layout_style_ids:
        pairs = []
        for token in args.layout_style_ids.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                l, s = token.split(":")
                pairs.append((int(l), int(s)))
            except ValueError:
                raise ValueError(f"Invalid layout_style_ids token '{token}', expected 'L:S'.")
        layout_ids_parsed = tuple(pairs) if pairs else None

    full_layouts = layout_ids_parsed
    current_layouts = layout_ids_parsed
    if args.curriculum_enable and full_layouts:
        # Start with a subset
        k0 = max(1, min(args.curriculum_initial_k, len(full_layouts)))
        current_layouts = tuple(full_layouts[:k0])
        # Force per-episode resample when using curriculum
        args.resample_layout_each_episode = True

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
        randomize_cameras=args.randomize_cameras,
        layout_and_style_ids=current_layouts,
        resample_layout_each_episode=args.resample_layout_each_episode,
        seed=seed,
        vector_async=args.vector_async,
        vector_context=args.vector_context,
        vector_shared_memory=args.vector_shared_memory,
        video_camera_name=args.video_camera_name,
        video_camera_pos=_parse_vec3(args.video_cam_pos),
        video_lookat=_parse_vec3(args.video_lookat),
        policy_base_camera_name=args.policy_base_camera_name,
        policy_right_camera_name=args.policy_right_camera_name,
        policy_wrist_camera_name=args.policy_wrist_camera_name,
    )
    if rank == 0:
        print(
            f"Creating RoboCasa Env: task={env_cfg.task_name} "
            f"(multi_stage={env_cfg.use_multi_stage}) num_envs_per_rank={local_num_envs}",
            flush=True,
        )
    env = make_robocasa_vector_env(env_cfg, local_num_envs)
    if rank == 0:
        print("Environment construction finished, resetting…", flush=True)
    obs, info = env.reset()
    if rank == 0:
        print("Initial reset complete, starting training loop.", flush=True)
    prompts = _extract_prompts(info, [""] * local_num_envs)

    action_space = env.single_action_space
    env_action_dim = action_space.shape[0]
    model_action_dim = model_config.action_dim

    def _pad_bounds(array: np.ndarray, target_dim: int, fill: float) -> np.ndarray:
        if array.shape[0] == target_dim:
            return array
        if array.shape[0] > target_dim:
            return array[:target_dim]
        pad = np.full((target_dim - array.shape[0],), fill, dtype=array.dtype)
        return np.concatenate([array, pad], axis=0)

    padded_low = _pad_bounds(action_space.low, model_action_dim, -1.0)
    padded_high = _pad_bounds(action_space.high, model_action_dim, 1.0)

    state_dim_env = obs["state"].shape[-1]

    policy_module = Pi05RobocasaPolicy.from_checkpoint(
        args.checkpoint,
        model_config=model_config,
        device=device,
        state_dim=model_action_dim,
        action_bounds=(padded_low, padded_high),
        inference_steps=args.inference_steps,
        min_std=args.min_std,
        max_std=args.max_std,
        include_initial_logprob=args.include_initial_logprob,
    )
    policy_module = policy_module.to(device=device)
    # Freeze backbones (vision + language). Train only RL heads / expert / noise / value.
    policy_module.freeze_backbones()
    # Keep only LoRA adapters trainable within the action expert; freeze base weights
    policy_module.freeze_action_expert_base(keep_lora=args.keep_lora)

    value_module = StateValueNetwork(state_dim=state_dim_env).to(device)

    if is_distributed and device.type == "cuda":
        policy = DDP(policy_module, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        value_net = DDP(value_module, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        policy = policy_module
        value_net = value_module

    actor_optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.critic_lr)

    buffer = PPORolloutBuffer(args.rollout_length, local_num_envs, device)

    global_step = 0
    start_time = time.time()
    base_min_std = args.min_std
    base_max_std = args.max_std
    for update in range(1, args.total_updates + 1):
        if rank == 0:
            print(
                f"\n== Update {update}/{args.total_updates} :: collecting {args.rollout_length} steps "
                f"per rank ({local_num_envs} envs) ==",
                flush=True,
            )
        progress_interval = max(1, args.rollout_length // 4)
        for step in range(args.rollout_length):
            state_tensor = torch.from_numpy(obs["state"]).float().to(device)
            values = value_net(state_tensor)

            pol_forward = policy.module if isinstance(policy, DDP) else policy
            with torch.no_grad():
                actions, chains, logprobs = pol_forward.sample_actions(obs, prompts, eval_mode=False, return_chain=True)
            control_actions = actions[:, 0, :env_action_dim]
            next_obs, rewards, terminated, truncated, info = env.step(control_actions.cpu().numpy())
            dones = np.logical_or(terminated, truncated)

            buffer.add(obs, prompts, actions, chains, logprobs, values, rewards, dones)
            obs = next_obs
            prompts = _extract_prompts(info, prompts)
            global_step += local_num_envs * world_size

            if rank == 0 and ((step + 1) % progress_interval == 0 or step == args.rollout_length - 1):
                mean_step_reward = float(np.mean(rewards)) if rewards is not None else 0.0
                print(
                    f"  Collected step {step + 1}/{args.rollout_length} "
                    f"(mean reward this step {mean_step_reward:.3f})",
                    flush=True,
                )

        buffer.set_next_observation(obs, prompts)
        last_values = value_net(torch.from_numpy(obs["state"]).float().to(device))
        buffer.compute_returns(last_values.detach(), args.gamma, args.gae_lambda)

        advantages = buffer.advantages
        flat_adv = advantages.view(-1)
        advantages = (advantages - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        buffer.advantages = advantages

        for epoch in range(args.ppo_epochs):
            if rank == 0:
                print(f"  Optimizing epoch {epoch + 1}/{args.ppo_epochs}", flush=True)
            for batch_idx, batch in enumerate(buffer.iter_minibatches(args.mini_batch_size), start=1):
                pol_forward = policy.module if isinstance(policy, DDP) else policy
                new_logprob, entropy = pol_forward.compute_log_prob(
                    batch["observations"], batch["prompts"], batch["chains"], return_entropy=True
                )
                old_logprob = batch["old_logprobs"]
                advantages_batch = batch["advantages"]
                returns_batch = batch["returns"]

                ratio = torch.exp(new_logprob - old_logprob)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * advantages_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                state_batch = torch.from_numpy(batch["observations"]["state"]).float().to(device)
                value_pred = value_net(state_batch)
                value_loss = torch.nn.functional.mse_loss(value_pred, returns_batch)

                entropy_mean = entropy.mean()
                loss = actor_loss + args.value_coef * value_loss - args.entropy_coef * entropy_mean

                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                pol_clip = policy.module if isinstance(policy, DDP) else policy
                val_clip = value_net.module if isinstance(value_net, DDP) else value_net
                torch.nn.utils.clip_grad_norm_(pol_clip.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(val_clip.parameters(), args.max_grad_norm)
                actor_optimizer.step()
                critic_optimizer.step()
                if rank == 0 and batch_idx % 10 == 0:
                    print(f"    Processed minibatch {batch_idx}", flush=True)

        mean_reward_local = torch.stack(buffer.rewards).float().mean().item() if buffer.rewards else 0.0
        mean_return_local = buffer.returns.mean().item()
        mean_reward = _distributed_mean(mean_reward_local, device, world_size)
        mean_return = _distributed_mean(mean_return_local, device, world_size)
        if rank == 0 and (update % args.log_interval == 0 or update == 1):
            elapsed = time.time() - start_time
            print(
                f"[update {update}/{args.total_updates}] step={global_step} "
                f"reward={mean_reward:.3f} return={mean_return:.3f} "
                f"time={elapsed:.1f}s"
            )

        if args.save_interval and update % args.save_interval == 0:
            # unwrap DDP for saving
            pol_to_save = policy.module if isinstance(policy, DDP) else policy
            val_to_save = value_net.module if isinstance(value_net, DDP) else value_net
            save_checkpoint(
                log_dir,
                update,
                pol_to_save,
                val_to_save,
                actor_optimizer,
                critic_optimizer,
                rank,
                lightweight=args.lightweight_save,
            )

        if args.eval_interval and update % args.eval_interval == 0:
            if rank == 0:
                # Print eval start banner
                print(
                    f"[Eval] Start @ update {update}: episodes={args.eval_episodes} "
                    f"task={env_cfg.task_name} seed={env_cfg.seed}",
                    flush=True,
                )
                success_rate = None
                eval_return = None
                if args.async_eval:
                    overlays_dir = Path(args.log_dir or f"runs/ppo_robocasa_{int(start_time)}") / "overlays"
                    overlays_dir.mkdir(parents=True, exist_ok=True)
                    overlay_path = overlays_dir / f"overlay_upd_{update:05d}.pt"
                    pol_state_full = (policy.module if isinstance(policy, DDP) else policy).state_dict()
                    trainable_param_keys = {name for name, p in (policy.module if isinstance(policy, DDP) else policy).named_parameters() if p.requires_grad}
                    trainable_module_prefixes = {name.rsplit('.', 1)[0] for name in trainable_param_keys if '.' in name}
                    filtered = {k: v for k, v in pol_state_full.items() if k in trainable_param_keys}
                    for buf_name, _ in (policy.module if isinstance(policy, DDP) else policy).named_buffers():
                        mod_pref = buf_name.rsplit('.', 1)[0] if '.' in buf_name else ''
                        if mod_pref in trainable_module_prefixes and buf_name in pol_state_full:
                            filtered[buf_name] = pol_state_full[buf_name]
                    torch.save({"policy": filtered}, overlay_path)

                    base_dir = args.checkpoint
                    cmd = [
                        "python", "scripts/eval_robocasa_with_overlay.py",
                        "--base-checkpoint-dir", str(base_dir),
                        "--overlay-checkpoint", str(overlay_path),
                        "--config-name", args.config_name,
                        "--task-name", args.task_name,
                        "--episodes", str(args.eval_episodes),
                        "--device", "cuda" if args.eval_device is not None else args.device,
                    ]
                    if args.eval_max_steps is not None:
                        cmd += ["--max-steps-override", str(args.eval_max_steps)]
                    if args.eval_inference_steps is not None:
                        cmd += ["--inference-steps", str(args.eval_inference_steps)]
                    if args.video_camera_name:
                        cmd += ["--video-camera-name", args.video_camera_name]
                    if args.video_cam_pos:
                        cmd += ["--video-cam-pos", args.video_cam_pos]
                    if args.video_lookat:
                        cmd += ["--video-lookat", args.video_lookat]
                    envp = os.environ.copy()
                    if args.eval_device is not None:
                        # For robosuite EGL binding, MUJOCO_EGL_DEVICE_ID must match one of the ids
                        # in CUDA_VISIBLE_DEVICES (not an index). Use the same value here.
                        envp["CUDA_VISIBLE_DEVICES"] = str(args.eval_device)
                        envp["MUJOCO_GL"] = envp.get("MUJOCO_GL", "egl")
                        envp["PYOPENGL_PLATFORM"] = envp.get("PYOPENGL_PLATFORM", "egl")
                        envp["MUJOCO_EGL_DEVICE_ID"] = str(args.eval_device)
                    print(f"[Eval-async] Spawning: {' '.join(cmd)} on GPU {args.eval_device}")
                    subprocess.Popen(cmd, env=envp)
                else:
                    was_training = policy.training
                    eval_policy = policy.module if isinstance(policy, DDP) else policy
                    eval_policy.eval()
                    video_dir = Path(args.eval_video_dir) if args.eval_video_dir else (Path(args.log_dir or f"runs/ppo_robocasa_{int(start_time)}") / "videos")
                    success_rate, eval_return = evaluate_policy(
                        eval_policy,
                        env_cfg,
                        args.eval_episodes,
                        device,
                        record_video=args.eval_record_video,
                        video_dir=video_dir,
                        update_tag=update,
                        max_steps_override=args.eval_max_steps,
                        inference_steps=args.eval_inference_steps,
                    )
                    if was_training:
                        policy.train()
                if success_rate is not None and eval_return is not None:
                    print(f"  Eval @ update {update}: success={100*success_rate:.1f}% return={eval_return:.3f}")
                else:
                    print(f"  Eval @ update {update}: launched async eval on GPU {args.eval_device}")
            # Curriculum expansion (synchronize across ranks)
            if args.curriculum_enable and full_layouts and not args.async_eval:
                # Broadcast eval success rate from rank0
                sr_local = float(success_rate) if rank == 0 else 0.0
                sr_synced = _distributed_mean(sr_local, device, world_size)
                # Decide target K
                current_k = len(current_layouts) if current_layouts else 0
                target_k = current_k
                if sr_synced >= args.curriculum_expand_threshold and current_k < len(full_layouts):
                    target_k = min(
                        len(full_layouts),
                        max(current_k + 1, min(current_k + args.curriculum_expand_step, args.curriculum_max_k)),
                    )
                # Sync target_k from rank0
                target_k = _distributed_sync_int(target_k if rank == 0 else 0, device, world_size)
                if target_k > current_k:
                    # Expand and rebuild env
                    current_layouts = tuple(full_layouts[:target_k])
                    if rank == 0:
                        print(f"[Curriculum] Expanding layouts: {current_k} -> {target_k}")
                    env_cfg = dataclasses.replace(
                        env_cfg,
                        layout_and_style_ids=current_layouts,
                        resample_layout_each_episode=True,
                    )
                    env.close()
                    env = make_robocasa_vector_env(env_cfg, local_num_envs)
                    obs, info = env.reset()
                    prompts = _extract_prompts(info, [""] * local_num_envs)
            if is_distributed:
                dist.barrier()

        hold_steps = int(args.noise_hold_ratio * args.total_updates)
        decay_steps = max(int(args.noise_decay_ratio * args.total_updates), 1)
        if update <= hold_steps:
            current_max_std = base_max_std
        else:
            progress = min((update - hold_steps) / decay_steps, 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            current_max_std = base_min_std + (base_max_std - base_min_std) * cosine
        pol_for_noise = policy.module if isinstance(policy, DDP) else policy
        pol_for_noise.set_noise_range(base_min_std, current_max_std)
        if rank == 0 and (update % args.log_interval == 0 or update == 1):
            print(f"  [noise] min_std={base_min_std:.3f} max_std_sched={current_max_std:.3f}")

        buffer.clear()

    env.close()


def main():
    rank, world_size, local_rank = _setup_distributed()
    try:
        args = tyro.cli(Args)
        train(args, rank, world_size, local_rank)
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    main()
