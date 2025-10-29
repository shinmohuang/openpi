#!/usr/bin/env python3
"""
Convert RoboCasa (robomimic-style) HDF5 datasets to LeRobot format.

The converter can ingest a single .hdf5 file or walk a directory tree and
append all demonstrations into one LeRobot dataset. Example usage:

    uv run examples/robocasa/convert_robocasa_data_to_lerobot.py \
        --input-path third_party/robocasa/datasets/v0.1 \
        --repo-name your_hf_username/robocasa_v01

Optional flags:
    --push-to-hub    Push the resulting dataset to Hugging Face Hub.
    --max-demos N    Only convert the first N demos (useful for smoke tests).
    --single-repo-name / --multi-repo-name  Override default repo names per stage.

The output is written under $HF_LEROBOT_HOME/<repo-name>. Existing content is
removed unless --no-clean is specified. Single-stage and multi-stage datasets
are exported separately.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def _quat_xyzw_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to axis-angle (shape (..., 3))."""
    if quat_xyzw.shape[-1] != 4:
        raise ValueError(f"Expected quaternion with shape (..., 4), got {quat_xyzw.shape}")
    # robomimic stores xyzw; convert to wxyz
    quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
    w = np.clip(quat_wxyz[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(1.0 - w * w)
    axis = np.zeros(quat_wxyz.shape[:-1] + (3,), dtype=np.float32)
    mask = sin_half > 1e-8
    axis[mask] = quat_wxyz[..., 1:][mask] / sin_half[mask][..., None]
    return axis * angle[..., None]


def _find_first_available(obs_group: h5py.Group, keys: Iterable[str]) -> np.ndarray:
    for key in keys:
        if key in obs_group:
            return np.asarray(obs_group[key], dtype=np.uint8)
    raise KeyError(f"None of the observation keys {list(keys)} found in dataset (available: {list(obs_group.keys())})")


def collect_hdf5_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*.hdf5") if p.is_file())
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def stage_from_path(path: Path) -> str | None:
    parts = set(path.parts)
    if "single_stage" in parts:
        return "single"
    if "multi_stage" in parts:
        return "multi"
    return None


def source_from_path(path: Path) -> str:
    return "mg" if "mg" in path.parts else "human"


def infer_specs(h5_path: Path) -> dict[str, int | tuple[int, ...]]:
    with h5py.File(h5_path, "r") as f:
        data_group = f["data"]
        demo = data_group[next(iter(data_group))]
        obs = demo["obs"]
        main_imgs = _find_first_available(
            obs,
            [
                "robot0_agentview_left_image",
                "robot0_agentview_center_image",
                "robot0_robotview_image",
            ],
        )
        _ = _find_first_available(
            obs,
            [
                "robot0_agentview_right_image",
                "robot0_agentview_center_image",
                "robot0_robotview_image",
            ],
        )
        img_shape = main_imgs[0].shape
        action_dim = int(demo["actions"].shape[-1])
        base_pos_dim = int(np.asarray(obs["robot0_base_pos"]).shape[-1])
        base_quat_dim = int(np.asarray(obs["robot0_base_quat"]).shape[-1])
        eef_pos_dim = int(np.asarray(obs["robot0_eef_pos"]).shape[-1])
        eef_quat_dim = int(np.asarray(obs["robot0_eef_quat"]).shape[-1])
        state_dim = base_pos_dim + base_quat_dim + eef_pos_dim + eef_quat_dim + 1  # + gripper
        joint_dim = int(np.asarray(obs["robot0_joint_pos"]).shape[-1]) if "robot0_joint_pos" in obs else 0
        joint_trig_dim = int(np.asarray(obs["robot0_joint_pos_cos"]).shape[-1]) if "robot0_joint_pos_cos" in obs else 0
        joint_vel_dim = int(np.asarray(obs["robot0_joint_vel"]).shape[-1]) if "robot0_joint_vel" in obs else 0
        return {
            "image_shape": img_shape,
            "action_dim": action_dim,
            "state_dim": state_dim,
            "joint_dim": joint_dim,
            "joint_trig_dim": joint_trig_dim,
            "joint_vel_dim": joint_vel_dim,
        }


def create_dataset(
    repo_name: str,
    clean: bool,
    *,
    image_shape: tuple[int, ...],
    action_dim: int,
    state_dim: int,
    joint_dim: int,
    joint_trig_dim: int,
    joint_vel_dim: int,
) -> LeRobotDataset:
    out_dir = HF_LEROBOT_HOME / repo_name
    if out_dir.exists():
        if clean:
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"Output directory already exists: {out_dir}. Use --clean to overwrite.")

    # Features align with OpenPI / DROID expectations
    features: dict[str, dict[str, object]] = {
        "image": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "image_right": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "gripper_position": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_position"],
        },
        "robot0_base_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["xyz"],
        },
        "robot0_base_quat": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["xyzw"],
        },
        "robot0_eef_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["xyz"],
        },
        "robot0_eef_quat": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["xyzw"],
        },
        "robot0_base_to_eef_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["xyz"],
        },
        "robot0_base_to_eef_quat": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["xyzw"],
        },
        "robot0_gripper_qpos": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["finger"],
        },
        "robot0_gripper_qvel": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["finger_vel"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["actions"],
        },
    }
    if joint_dim > 0:
        features["robot0_joint_pos"] = {
            "dtype": "float32",
            "shape": (joint_dim,),
            "names": ["joint"],
        }
    if joint_trig_dim > 0:
        features["robot0_joint_pos_cos"] = {
            "dtype": "float32",
            "shape": (joint_trig_dim,),
            "names": ["joint_cos"],
        }
        features["robot0_joint_pos_sin"] = {
            "dtype": "float32",
            "shape": (joint_trig_dim,),
            "names": ["joint_sin"],
        }
    if joint_vel_dim > 0:
        features["robot0_joint_vel"] = {
            "dtype": "float32",
            "shape": (joint_vel_dim,),
            "names": ["joint_vel"],
        }

    return LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        features=features,
        image_writer_threads=8,
        image_writer_processes=4,
    )


def convert_file(
    dataset: LeRobotDataset,
    dataset_path: Path,
    *,
    max_demos: int | None = None,
    action_dim: int,
    state_dim: int,
    joint_dim: int,
    joint_trig_dim: int,
    joint_vel_dim: int,
) -> int:
    """Convert a single RoboCasa HDF5 file into the provided LeRobot dataset."""
    demo_count = 0
    with h5py.File(dataset_path, "r") as f:
        data_group = f["data"]
        for demo_name in sorted(data_group.keys()):
            if max_demos is not None and demo_count >= max_demos:
                break

            demo = data_group[demo_name]
            obs = demo["obs"]
            actions = np.asarray(demo["actions"], dtype=np.float32)[..., :action_dim]

            ep_meta = json.loads(demo.attrs.get("ep_meta", "{}"))
            task_lang = ep_meta.get("lang", "")

            # Main / wrist camera fallbacks
            main_imgs = _find_first_available(
                obs,
                [
                    "robot0_agentview_left_image",
                    "robot0_agentview_center_image",
                    "robot0_robotview_image",
                ],
            )
            right_imgs = _find_first_available(
                obs,
                [
                    "robot0_agentview_right_image",
                    "robot0_agentview_center_image",
                    "robot0_robotview_image",
                ],
            )
            try:
                wrist_imgs = _find_first_available(obs, ["robot0_eye_in_hand_image"])
            except KeyError:
                wrist_imgs = main_imgs

            base_pos = np.asarray(obs["robot0_base_pos"], dtype=np.float32)
            base_quat = np.asarray(obs["robot0_base_quat"], dtype=np.float32)
            eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
            eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
            base_to_eef_pos = (
                np.asarray(obs["robot0_base_to_eef_pos"], dtype=np.float32)
                if "robot0_base_to_eef_pos" in obs
                else np.zeros_like(eef_pos)
            )
            base_to_eef_quat = (
                np.asarray(obs["robot0_base_to_eef_quat"], dtype=np.float32)
                if "robot0_base_to_eef_quat" in obs
                else np.zeros_like(eef_quat)
            )
            joint_pos = (
                np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
                if joint_dim > 0 and "robot0_joint_pos" in obs
                else (np.zeros((len(actions), joint_dim), dtype=np.float32) if joint_dim > 0 else None)
            )
            joint_pos_cos = (
                np.asarray(obs["robot0_joint_pos_cos"], dtype=np.float32)
                if joint_trig_dim > 0 and "robot0_joint_pos_cos" in obs
                else (np.zeros((len(actions), joint_trig_dim), dtype=np.float32) if joint_trig_dim > 0 else None)
            )
            joint_pos_sin = (
                np.asarray(obs["robot0_joint_pos_sin"], dtype=np.float32)
                if joint_trig_dim > 0 and "robot0_joint_pos_sin" in obs
                else (np.zeros((len(actions), joint_trig_dim), dtype=np.float32) if joint_trig_dim > 0 else None)
            )
            joint_vel = (
                np.asarray(obs["robot0_joint_vel"], dtype=np.float32)
                if joint_vel_dim > 0 and "robot0_joint_vel" in obs
                else (np.zeros((len(actions), joint_vel_dim), dtype=np.float32) if joint_vel_dim > 0 else None)
            )
            gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
            if gripper_qpos.ndim == 1:
                gripper_qpos = gripper_qpos.reshape(-1, 2)
            gripper_scalar = np.mean(gripper_qpos, axis=-1, keepdims=True)
            gripper_qvel = (
                np.asarray(obs["robot0_gripper_qvel"], dtype=np.float32)
                if "robot0_gripper_qvel" in obs
                else np.zeros_like(gripper_qpos)
            )

            horizon_candidates = [
                len(actions),
                len(main_imgs),
                len(right_imgs),
                len(wrist_imgs),
                len(base_pos),
                len(base_quat),
                len(eef_pos),
                len(eef_quat),
                len(base_to_eef_pos),
                len(base_to_eef_quat),
                len(gripper_scalar),
                len(gripper_qpos),
                len(gripper_qvel),
            ]
            if joint_pos is not None:
                horizon_candidates.append(len(joint_pos))
            if joint_pos_cos is not None:
                horizon_candidates.append(len(joint_pos_cos))
            if joint_pos_sin is not None:
                horizon_candidates.append(len(joint_pos_sin))
            if joint_vel is not None:
                horizon_candidates.append(len(joint_vel))
            horizon = min(horizon_candidates)
            if horizon == 0:
                continue

            actions = actions[:horizon]
            main_imgs = main_imgs[:horizon]
            right_imgs = right_imgs[:horizon]
            wrist_imgs = wrist_imgs[:horizon]
            base_pos = base_pos[:horizon]
            base_quat = base_quat[:horizon]
            eef_pos = eef_pos[:horizon]
            eef_quat = eef_quat[:horizon]
            base_to_eef_pos = base_to_eef_pos[:horizon]
            base_to_eef_quat = base_to_eef_quat[:horizon]
            if joint_pos is not None:
                joint_pos = joint_pos[:horizon]
            if joint_pos_cos is not None:
                joint_pos_cos = joint_pos_cos[:horizon]
            if joint_pos_sin is not None:
                joint_pos_sin = joint_pos_sin[:horizon]
            if joint_vel is not None:
                joint_vel = joint_vel[:horizon]
            gripper_qpos = gripper_qpos[:horizon]
            gripper_qvel = gripper_qvel[:horizon]
            gripper_scalar = gripper_scalar[:horizon]
            state = np.concatenate(
                [base_pos, base_quat, eef_pos, eef_quat, gripper_scalar],
                axis=-1,
            )
            if state.shape[-1] != state_dim:
                raise ValueError(f"State dimension mismatch: expected {state_dim}, got {state.shape[-1]}")

            for t in range(horizon):
                dataset.add_frame(
                    {
                        "image": main_imgs[t],
                        "image_right": right_imgs[t],
                        "wrist_image": wrist_imgs[t],
                        "state": state[t],
                        "gripper_position": gripper_scalar[t],
                        "robot0_base_pos": base_pos[t],
                        "robot0_base_quat": base_quat[t],
                        "robot0_eef_pos": eef_pos[t],
                        "robot0_eef_quat": eef_quat[t],
                        "robot0_base_to_eef_pos": base_to_eef_pos[t],
                        "robot0_base_to_eef_quat": base_to_eef_quat[t],
                        **({"robot0_joint_pos": joint_pos[t]} if joint_pos is not None else {}),
                        **({"robot0_joint_pos_cos": joint_pos_cos[t]} if joint_pos_cos is not None else {}),
                        **({"robot0_joint_pos_sin": joint_pos_sin[t]} if joint_pos_sin is not None else {}),
                        **({"robot0_joint_vel": joint_vel[t]} if joint_vel is not None else {}),
                        "robot0_gripper_qpos": gripper_qpos[t],
                        "robot0_gripper_qvel": gripper_qvel[t],
                        "actions": actions[t],
                        "task": task_lang,
                    }
                )
            dataset.save_episode()
            demo_count += 1
    return demo_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to a RoboCasa .hdf5 file or directory containing multiple datasets.",
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default="your_hf_username/robocasa",
        help="Base name of the output LeRobot repo. '_single' and '_multi' suffixes will be appended unless dedicated names are provided.",
    )
    parser.add_argument(
        "--single-repo-name",
        type=str,
        default=None,
        help="Optional explicit repo name for single-stage datasets.",
    )
    parser.add_argument(
        "--multi-repo-name",
        type=str,
        default=None,
        help="Optional explicit repo name for multi-stage datasets.",
    )
    parser.add_argument("--push-to-hub", action="store_true", help="Push resulting dataset to Hugging Face Hub.")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete existing output directory.")
    parser.add_argument("--max-demos", type=int, default=None, help="Convert at most this many demos across all datasets for testing.")
    parser.add_argument(
        "--source",
        choices=("all", "human", "mg"),
        default="all",
        help="Which data source(s) to include: human-collected (dated folders), MimicGen (mg), or both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_files = collect_hdf5_files(args.input_path)
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {args.input_path}")

    single_repo = args.single_repo_name or f"{args.repo_name}_single"
    multi_repo = args.multi_repo_name or f"{args.repo_name}_multi"

    stage_files: dict[str, list[Path]] = {"single": [], "multi": []}
    skipped_stage = 0
    skipped_source = 0
    for file_path in hdf5_files:
        stage = stage_from_path(file_path)
        if stage is None:
            skipped_stage += 1
            continue
        source = source_from_path(file_path)
        if args.source != "all" and source != args.source:
            skipped_source += 1
            continue
        stage_files[stage].append(file_path)

    if skipped_stage:
        print(f"Warning: skipped {skipped_stage} file(s) with unknown stage.")
    if skipped_source:
        print(f"Skipped {skipped_source} file(s) due to source filter '{args.source}'.")

    total_demos = 0
    remaining = args.max_demos
    stage_reports: dict[str, int] = {}

    stage_configs = {
        "single": {"repo": single_repo},
        "multi": {"repo": multi_repo},
    }

    for stage, files in stage_files.items():
        if not files:
            continue
        specs = infer_specs(files[0])
        dataset = create_dataset(
            stage_configs[stage]["repo"],
            clean=not args.no_clean,
            image_shape=specs["image_shape"],
            action_dim=int(specs["action_dim"]),
            state_dim=int(specs["state_dim"]),
            joint_dim=int(specs["joint_dim"]),
            joint_trig_dim=int(specs["joint_trig_dim"]),
            joint_vel_dim=int(specs["joint_vel_dim"]),
        )
        stage_configs[stage].update({
            "dataset": dataset,
            "action_dim": int(specs["action_dim"]),
            "state_dim": int(specs["state_dim"]),
            "joint_dim": int(specs["joint_dim"]),
            "joint_trig_dim": int(specs["joint_trig_dim"]),
            "joint_vel_dim": int(specs["joint_vel_dim"]),
        })

        stage_demo_count = 0
        for file_path in files:
            if remaining is not None and remaining <= 0:
                break
            demos_written = convert_file(
                dataset,
                file_path,
                max_demos=remaining,
                action_dim=stage_configs[stage]["action_dim"],
                state_dim=stage_configs[stage]["state_dim"],
                joint_dim=stage_configs[stage]["joint_dim"],
                joint_trig_dim=stage_configs[stage]["joint_trig_dim"],
                joint_vel_dim=stage_configs[stage]["joint_vel_dim"],
            )
            stage_demo_count += demos_written
            total_demos += demos_written
            if remaining is not None:
                remaining = max(0, remaining - demos_written)
        stage_reports[stage] = stage_demo_count

        if args.push_to_hub:
            dataset.push_to_hub(
                tags=["robocasa", stage, "panda"],
                private=False,
                push_videos=True,
                license="apache-2.0",
            )

        print(
            f"Stage {stage}: converted {stage_demo_count} demos from {len(files)} file(s) into {stage_configs[stage]['repo']}."
        )

    print(f"Total demos converted: {total_demos} (source filter: {args.source})")


if __name__ == "__main__":
    main()
