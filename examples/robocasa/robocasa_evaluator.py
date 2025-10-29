import collections
import datetime
import json
import logging
import pathlib
import random

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

import robosuite
from robosuite.environments.base import REGISTERED_ENVS

from robocasa_args import Args
from robocasa_observations import obs_to_robocasa_inputs
from robocasa_tasks import pick_tasks, task_horizon, task_prompt

# Register custom 12D controller (ensure local module is importable when run as script).
try:
    from pandamobile_12d_controller import PandaMobile12DController  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - runtime fallback when executed as script
    import pathlib as _pathlib
    import sys as _sys

    _sys.path.append(str(_pathlib.Path(__file__).parent.resolve()))
    from pandamobile_12d_controller import PandaMobile12DController  # noqa: F401

try:
    from robocasa.environments.kitchen import kitchen as _kitchen_module
except Exception:  # pragma: no cover - fallback if kitchen module unavailable
    _kitchen_module = None


def eval_robocasa(args: Args) -> None:
    # Seeding
    np.random.seed(args.seed)
    random.seed(args.seed)

    if _kitchen_module is not None:
        desired_offset = [0.0, 0.0, 0.0]
        current_offset = _kitchen_module._ROBOT_POS_OFFSETS.get("Panda")
        if current_offset != desired_offset:
            _kitchen_module._ROBOT_POS_OFFSETS["Panda"] = desired_offset
            logging.info(
                "Adjusted Panda robot base offset %s in kitchen environment.",
                _kitchen_module._ROBOT_POS_OFFSETS["Panda"],
            )

    tasks = pick_tasks(args)
    logging.info("Selected tasks: %s", tasks)

    run_video_dir: pathlib.Path | None = None
    if args.video_out_dir is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        stage_label = "multi-stage" if args.use_multi_stage else "single-stage"
        model_slug = (args.model_name or "model").replace(" ", "_")
        run_video_dir = pathlib.Path(args.video_out_dir) / f"{timestamp}_{model_slug}_{stage_label}"
        run_video_dir.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info("Server metadata: %s", client.get_server_metadata())

    total_episodes = 0
    total_successes = 0

    for task_name in tasks:
        horizon = args.max_steps_override or task_horizon(task_name, use_multi_stage=args.use_multi_stage)
        logging.info("Task %s | horizon=%s", task_name, horizon)
        env_cls = REGISTERED_ENVS.get(task_name)
        excluded_layouts = set(getattr(env_cls, "EXCLUDE_LAYOUTS", [])) if env_cls else set()

        task_successes = 0
        task_episodes = 0

        for episode_idx in range(args.num_trials_per_task):
            layout_and_style_ids = (
                list(args.layout_and_style_ids)
                if args.layout_and_style_ids is not None
                else None
            )
            if layout_and_style_ids:
                filtered = [
                    (int(ls[0]), int(ls[1]))
                    for ls in layout_and_style_ids
                    if int(ls[0]) not in excluded_layouts
                ]
                if not filtered:
                    logging.warning(
                        "Requested layout/style IDs incompatible with %s exclusions %s; falling back to environment defaults.",
                        task_name,
                        sorted(excluded_layouts),
                    )
                    chosen_ls = None
                else:
                    ls = random.choice(filtered)
                    chosen_ls = (ls,)
            else:
                chosen_ls = None

            controller_cfg_path = pathlib.Path(__file__).with_name("controller_pandamobile_12d.json")
            with controller_cfg_path.open("r") as f:
                controller_cfg = json.load(f)

            camera_names = list(dict.fromkeys(args.camera_names))
            if not args.include_right_camera:
                camera_names = [name for name in camera_names if "agentview_right" not in name]

            env = robosuite.make(
                env_name=task_name,
                robots=args.robots,
                controller_configs=controller_cfg,
                camera_names=camera_names,
                camera_widths=args.camera_size,
                camera_heights=args.camera_size,
                has_renderer=False,
                has_offscreen_renderer=True,
                renderer="mujoco",
                ignore_done=True,
                use_object_obs=True,
                use_camera_obs=True,
                camera_depths=False,
                seed=args.seed + episode_idx,
                obj_instance_split="B",
                randomize_cameras=args.randomize_cameras,
                layout_and_style_ids=chosen_ls,
                translucent_robot=False,
            )
            logging.info("Env action_dim=%s", env.action_dim)

            try:
                obs = env.reset()
                action_low, action_high = env.action_spec
                action_low = np.asarray(action_low, dtype=float)
                action_high = np.asarray(action_high, dtype=float)
                obs = env._get_observations(force_update=True)

                try:
                    base_candidates = ["robot0_base", "robot0:base", "panda0_link0", "panda_link0"]
                    base_name = None
                    for cand in base_candidates:
                        try:
                            env.sim.model.body_name2id(cand)
                            base_name = cand
                            break
                        except Exception:
                            continue
                    if base_name is None:
                        try:
                            names = [
                                n.decode() if isinstance(n, bytes) else str(n)
                                for n in env.sim.model.body_names
                            ]
                            for name in names:
                                if "robot0" in name and "base" in name:
                                    base_name = name
                                    break
                        except Exception:
                            base_name = None

                    env.sim.forward()
                except Exception:
                    pass
                action_plan = collections.deque()
                writer = None
                out_tmp = None
                success = False

                if run_video_dir is not None:
                    out_dir = run_video_dir
                    try:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        out_tmp = out_dir / f"{task_name}_ep{episode_idx}.mp4"
                        writer = imageio.get_writer(
                            out_tmp,
                            format="ffmpeg",
                            mode="I",
                            fps=10,
                            codec="libx264",
                            ffmpeg_params=["-crf", "28", "-preset", "fast"],
                        )
                    except Exception as exc:
                        logging.error("Could not open video writer %s: %s", out_dir, exc)
                        writer = None

                prompt_logged = False
                for _ in range(horizon):
                    prompt = task_prompt(task_name, env=env)
                    if not prompt_logged:
                        logging.info("Episode prompt (%s ep=%s): %s", task_name, episode_idx, prompt)
                        prompt_logged = True
                    obs_policy = dict(obs)
                    flip_keys = (
                        "robot0_agentview_left_image",
                        "robot0_agentview_center_image",
                        "robot0_agentview_right_image",
                        "robot0_robotview_image",
                        "robot0_eye_in_hand_image",
                        "eye_in_hand_image",
                        "agentview_image",
                    )
                    for key in flip_keys:
                        if key in obs_policy:
                            obs_policy[key] = np.flipud(np.asarray(obs_policy[key]))
                    available_cams = camera_names or [
                        "robot0_agentview_left",
                        "robot0_agentview_right",
                        "robot0_eye_in_hand",
                    ]
                    base_cam = available_cams[0] if available_cams else "robot0_agentview_left"
                    right_cam = None
                    if args.include_right_camera:
                        right_cam = next(
                            (cam for cam in available_cams if "agentview_right" in cam),
                            None,
                        )
                    wrist_cam = next(
                        (cam for cam in available_cams if "eye_in_hand" in cam),
                        "robot0_eye_in_hand",
                    )
                    inputs = obs_to_robocasa_inputs(
                        obs_policy,
                        prompt=prompt,
                        base_camera=base_cam,
                        right_camera=right_cam,
                        wrist_camera=wrist_cam,
                        resize_hw=args.camera_resize,
                        include_right_camera=args.include_right_camera,
                    )
                    policy_base_img = np.asarray(inputs["observation/image"])
                    right_raw = inputs.get("observation/image_right")
                    policy_right_img = np.asarray(right_raw) if right_raw is not None else None
                    policy_wrist_img = np.asarray(inputs["observation/wrist_image"])

                    if not action_plan:
                        action_chunk = client.infer(inputs)["actions"]
                        if len(action_chunk) < args.replan_steps:
                            raise RuntimeError(
                                f"Policy predicted only {len(action_chunk)} steps (< replan_steps={args.replan_steps})."
                            )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = np.asarray(action_plan.popleft(), dtype=float)
                    if action.shape[0] != action_low.shape[0]:
                        raise ValueError(
                            f"Policy action dimension mismatch: expected {action_low.shape[0]}, got {action.shape[0]}"
                        )
                    action_cmd = np.clip(action, action_low, action_high)

                    obs, reward, done, info = env.step(action_cmd.tolist())
                    obs = env._get_observations(force_update=True)
                    if writer is not None:
                        try:
                            tiles = []
                            for view in (policy_base_img, policy_right_img, policy_wrist_img):
                                if view is None:
                                    continue
                                img = np.asarray(view)
                                if np.issubdtype(img.dtype, np.floating):
                                    img = image_tools.convert_to_uint8(img)
                                img = image_tools.resize_with_pad(img, args.camera_size, args.camera_size)
                                tiles.append(img)
                            if tiles:
                                frame = np.concatenate(tiles, axis=1)
                                writer.append_data(frame)
                        except Exception as exc:
                            logging.error("Video write error: %s", exc)
                            try:
                                writer.close()
                            except Exception:
                                pass
                            writer = None

                    succ = env._check_success()
                    if isinstance(succ, dict):
                        success = bool(succ.get("task", False))
                    else:
                        success = bool(succ)
                    if success:
                        task_successes += 1
                        total_successes += 1
                        break

                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                    if out_tmp is not None:
                        try:
                            final_path = out_tmp.with_name(
                                f"{task_name}_ep{episode_idx}_{'succ' if success else 'fail'}.mp4"
                            )
                            out_tmp.rename(final_path)
                            logging.info("Saved %s", final_path)
                        except Exception as exc:
                            logging.info("Saved %s (rename skipped: %s)", out_tmp, exc)

            except Exception as exc:
                logging.error("Episode exception (%s ep=%s): %s", task_name, episode_idx, exc)

            finally:
                try:
                    env.close()
                except Exception:
                    pass

            task_episodes += 1
            total_episodes += 1
            logging.info(
                "Task %s: success so far %s/%s | total %s/%s",
                task_name,
                task_successes,
                task_episodes,
                total_successes,
                total_episodes,
            )

        logging.info(
            "Task %s success rate: %s/%s = %.2f",
            task_name,
            task_successes,
            task_episodes,
            (task_successes / max(task_episodes, 1)),
        )

    logging.info(
        "Overall success rate: %s/%s = %.2f",
        total_successes,
        total_episodes,
        (total_successes / max(total_episodes, 1)),
    )
