import dataclasses
import functools
import json
import pathlib
import random
from collections.abc import Callable

import gym
import numpy as np
import robosuite
from robosuite.environments.base import REGISTERED_ENVS

# Ensure custom RoboCasa controller registers itself with robosuite.
from examples.robocasa import pandamobile_12d_controller as _pandamobile_controller  # noqa: F401

from examples.robocasa.robocasa_observations import obs_to_robocasa_inputs
from examples.robocasa.robocasa_camera import set_camera_pose as _set_camera_pose
from examples.robocasa.robocasa_tasks import task_horizon, task_prompt


@dataclasses.dataclass
class RobocasaEnvConfig:
    """Configuration bundle for constructing RoboCasa RL environments."""

    task_name: str = "HANG_MUG"
    use_multi_stage: bool = False
    robots: str = "PandaOmron"
    camera_names: tuple[str, ...] = (
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    )
    camera_size: int = 512
    include_right_camera: bool = True
    camera_resize: int | None = 224
    max_steps_override: int | None = None
    randomize_cameras: bool = False
    layout_and_style_ids: tuple[tuple[int, int], ...] | None = None
    # If True and multiple layouts are provided, rebuild env at each reset with a randomly chosen (layout, style)
    resample_layout_each_episode: bool = False
    seed: int = 0
    # Vectorization controls
    vector_async: bool = True
    vector_context: str | None = "spawn"  # multiprocessing start method for AsyncVectorEnv
    vector_shared_memory: bool = False
    # Video rendering camera (decoupled from policy input cameras)
    video_camera_name: str = "robot0_agentview_left"
    # Optional fixed camera pose for video camera
    video_camera_pos: tuple[float, float, float] | None = None
    video_lookat: tuple[float, float, float] | None = None
    # Policy input cameras (decoupled from video)
    # Use the three standard RoboCasa views for policy inputs
    policy_base_camera_name: str = "robot0_agentview_left"
    policy_right_camera_name: str | None = "robot0_agentview_right"
    policy_wrist_camera_name: str | None = "robot0_eye_in_hand"


class RobocasaGymEnv(gym.Env):
    """Thin Gym wrapper for RoboCasa Kitchen tasks."""

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config: RobocasaEnvConfig):
        super().__init__()
        self._config = config
        self._rng = random.Random(config.seed)

        self._env = self._make_env()
        self._episode_horizon = self._resolve_horizon()

        low, high = self._env.action_spec
        self.action_space = gym.spaces.Box(
            low=np.asarray(low, dtype=np.float32),
            high=np.asarray(high, dtype=np.float32),
        )

        processed = self._process_observation(self._reset_env())
        image_spaces: dict[str, gym.spaces.Box] = {
            key: gym.spaces.Box(low=0, high=255, shape=value.shape, dtype=np.uint8)
            for key, value in processed["image"].items()
        }
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=processed["state"].shape,
                    dtype=np.float32,
                ),
                "image": gym.spaces.Dict(image_spaces),
            }
        )

        self._last_obs = processed
        self._last_prompt = ""
        self._steps_taken = 0

    def _controller_config(self) -> dict:
        controller_cfg_path = (
            pathlib.Path(__file__).resolve().parents[3]
            / "examples"
            / "robocasa"
            / "controller_pandamobile_12d.json"
        )
        with controller_cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _make_env(self):
        controller_cfg = self._controller_config()
        # Ensure video camera is included in the observation cameras
        camera_names = list(self._config.camera_names)
        if self._config.video_camera_name not in camera_names:
            camera_names.append(self._config.video_camera_name)
        # Ensure policy input cameras are also present
        if self._config.policy_base_camera_name not in camera_names:
            camera_names.append(self._config.policy_base_camera_name)
        if self._config.include_right_camera and self._config.policy_right_camera_name:
            if self._config.policy_right_camera_name not in camera_names:
                camera_names.append(self._config.policy_right_camera_name)
        wrist_name = self._config.policy_wrist_camera_name or self._config.policy_base_camera_name
        if wrist_name not in camera_names:
            camera_names.append(wrist_name)
        return robosuite.make(
            env_name=self._config.task_name,
            robots=self._config.robots,
            controller_configs=controller_cfg,
            camera_names=camera_names,
            camera_widths=self._config.camera_size,
            camera_heights=self._config.camera_size,
            has_renderer=False,
            has_offscreen_renderer=True,
            renderer="mujoco",
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=True,
            camera_depths=False,
            seed=self._config.seed,
            obj_instance_split="B",
            randomize_cameras=self._config.randomize_cameras,
            layout_and_style_ids=self._config.layout_and_style_ids,
            translucent_robot=False,
        )

    def _resolve_horizon(self) -> int:
        if self._config.max_steps_override is not None:
            return self._config.max_steps_override
        return task_horizon(self._config.task_name, use_multi_stage=self._config.use_multi_stage)

    def _reset_env(self):
        # Optionally resample layout/style by rebuilding the env
        if self._config.resample_layout_each_episode:
            # Choose (layout, style) automatically if not provided: use full registry, minus EXCLUDE_LAYOUTS
            chosen_ls = None
            if self._config.layout_and_style_ids and len(self._config.layout_and_style_ids) > 0:
                chosen_ls = random.choice(list(self._config.layout_and_style_ids))
            else:
                try:
                    from robocasa.models.scenes.scene_registry import unpack_layout_ids, unpack_style_ids

                    env_cls = REGISTERED_ENVS.get(self._config.task_name)
                    excluded = set(getattr(env_cls, "EXCLUDE_LAYOUTS", [])) if env_cls else set()
                    layout_ids = [int(l) for l in unpack_layout_ids(None) if int(l) not in excluded]
                    style_ids = [int(s) for s in unpack_style_ids(None)]
                    if layout_ids and style_ids:
                        chosen_ls = (random.choice(layout_ids), random.choice(style_ids))
                except Exception:
                    chosen_ls = None

            if chosen_ls is not None:
                controller_cfg = self._controller_config()
                camera_names = list(self._config.camera_names)
                if self._config.video_camera_name not in camera_names:
                    camera_names.append(self._config.video_camera_name)
                if self._config.policy_base_camera_name not in camera_names:
                    camera_names.append(self._config.policy_base_camera_name)
                if self._config.include_right_camera and self._config.policy_right_camera_name:
                    if self._config.policy_right_camera_name not in camera_names:
                        camera_names.append(self._config.policy_right_camera_name)
                wrist_name = self._config.policy_wrist_camera_name or self._config.policy_base_camera_name
                if wrist_name not in camera_names:
                    camera_names.append(wrist_name)
                self._env = robosuite.make(
                    env_name=self._config.task_name,
                    robots=self._config.robots,
                    controller_configs=controller_cfg,
                    camera_names=camera_names,
                    camera_widths=self._config.camera_size,
                    camera_heights=self._config.camera_size,
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    renderer="mujoco",
                    ignore_done=True,
                    use_object_obs=True,
                    use_camera_obs=True,
                    camera_depths=False,
                    seed=self._config.seed,
                    obj_instance_split="B",
                    randomize_cameras=self._config.randomize_cameras,
                    layout_and_style_ids=(tuple(map(int, chosen_ls)),),
                    translucent_robot=False,
                )
            else:
                self._env.reset()
        else:
            self._env.reset()
        return self._env._get_observations(force_update=True)

    def _process_observation(self, obs: dict) -> dict:
        prompt = task_prompt(self._config.task_name, env=self._env)
        # Optionally override the video camera pose on the MuJoCo model
        try:
            if self._config.video_camera_pos is not None and self._config.video_lookat is not None:
                _set_camera_pose(
                    self._env,
                    self._config.video_camera_name,
                    np.asarray(self._config.video_camera_pos, dtype=float),
                    np.asarray(self._config.video_lookat, dtype=float),
                )
        except Exception:
            pass
        # Build policy inputs using configured policy cameras (not the wrist-following view)
        right_cam = (
            self._config.policy_right_camera_name if self._config.include_right_camera else None
        )
        wrist_cam = self._config.policy_wrist_camera_name or self._config.policy_base_camera_name
        inputs = obs_to_robocasa_inputs(
            dict(obs),
            prompt=prompt,
            base_camera=self._config.policy_base_camera_name,
            right_camera=right_cam,
            wrist_camera=wrist_cam,
            resize_hw=self._config.camera_resize,
            include_right_camera=self._config.include_right_camera,
            flip_vertical=True,
        )

        right_image = inputs.get("observation/image_right")
        if right_image is None:
            right_image = np.zeros_like(inputs["observation/image"])

        # Save a dedicated video frame from the chosen video camera
        video_key = f"{self._config.video_camera_name}_image"
        try:
            raw_video = np.asarray(obs.get(video_key), dtype=np.uint8)
        except Exception:
            raw_video = None

        processed = {
            "state": np.asarray(inputs["observation/state"], dtype=np.float32),
            "image": {
                "observation/image": np.asarray(inputs["observation/image"], dtype=np.uint8),
                "observation/image_right": np.asarray(right_image, dtype=np.uint8),
                "observation/wrist_image": np.asarray(inputs["observation/wrist_image"], dtype=np.uint8),
            },
        }
        self._last_prompt = inputs["prompt"]
        if raw_video is None:
            # Fallback: use base image as video
            self._last_video_frame = processed["image"]["observation/image"]
        else:
            # Resize / pad to camera_size for uniform videos
            try:
                from openpi_client import image_tools as _img_tools

                # Flip vertically to match policy input preprocessing
                flipped = np.flipud(raw_video)
                self._last_video_frame = _img_tools.resize_with_pad(
                    flipped, self._config.camera_size, self._config.camera_size
                ).astype(np.uint8)
            except Exception:
                self._last_video_frame = np.flipud(raw_video).astype(np.uint8)
        return processed

    def seed(self, seed: int | None = None):
        if seed is not None:
            self._config.seed = seed
        self._rng.seed(self._config.seed)
        return [self._config.seed]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.seed(seed)
        obs_raw = self._reset_env()
        # Print chosen layout / style if available
        layout_id = getattr(self._env, "layout_id", None)
        style_id = getattr(self._env, "style_id", None)
        if layout_id is not None and style_id is not None:
            print(f"[Env Reset] task={self._config.task_name} layout={layout_id} style={style_id} seed={self._config.seed}")
        obs = self._process_observation(obs_raw)
        self._last_obs = obs
        self._steps_taken = 0
        info = {"prompt": self._last_prompt}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self._steps_taken += 1

        _, reward, done, info = self._env.step(action.tolist())
        obs = self._process_observation(self._env._get_observations(force_update=True))
        self._last_obs = obs

        env_success = self._env._check_success()
        success = bool(env_success.get("task", False) if isinstance(env_success, dict) else env_success)
        reward = 1.0 if success else float(reward)

        truncated = self._steps_taken >= self._episode_horizon
        terminated = bool(done) or success
        info = {
            **(info or {}),
            "success": success,
            "prompt": self._last_prompt,
            "elapsed_steps": self._steps_taken,
        }
        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError(f"Unsupported render mode {mode}")
        if self._last_obs is None:
            return None
        frame = getattr(self, "_last_video_frame", None)
        if frame is None:
            frame = self._last_obs["image"]["observation/image"]
        # Ensure contiguous uint8 HWC
        return np.ascontiguousarray(frame.astype(np.uint8))


def make_robocasa_vector_env(config: RobocasaEnvConfig, num_envs: int) -> gym.vector.VectorEnv:
    """Create a vectorized RoboCasa environment.

    Falls back to `SyncVectorEnv` for single-environment setups to avoid multiprocessing overhead.
    """

    def make_single(seed_offset: int):
        cfg = dataclasses.replace(config, seed=config.seed + seed_offset)
        return RobocasaGymEnv(cfg)

    env_fns: list[Callable[[], gym.Env]] = [
        functools.partial(make_single, idx) for idx in range(num_envs)
    ]
    if num_envs == 1 or not config.vector_async:
        return gym.vector.SyncVectorEnv(env_fns)
    # Use spawn context and disable shared memory to avoid EGL / CUDA fork hangs
    return gym.vector.AsyncVectorEnv(
        env_fns,
        shared_memory=config.vector_shared_memory,
        context=config.vector_context,
    )
