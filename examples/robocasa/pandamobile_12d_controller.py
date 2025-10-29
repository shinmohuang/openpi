from __future__ import annotations

import numpy as np

from robosuite.controllers.composite.composite_controller import (
    HybridMobileBase,
    register_composite_controller,
)


@register_composite_controller
class PandaMobile12DController(HybridMobileBase):
    """Hybrid mobile base controller with a 12D action interface matching RoboCasa datasets."""

    name = "PANDAMOBILE_12D"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slices_ready = False

    def setup_action_split_idx(self):
        super().setup_action_split_idx()
        # Cache slices for downstream mapping (arm, gripper, base, torso)
        right_arm = self.arms[0]
        gripper_name = f"{right_arm}_gripper"
        self._right_slice = slice(*self._action_split_indexes[right_arm])
        self._gripper_slice = slice(*self._action_split_indexes.get(gripper_name, (0, 0)))
        self._base_slice = slice(*self._action_split_indexes.get("base", (0, 0)))
        self._torso_slice = slice(*self._action_split_indexes.get("torso", (0, 0)))
        self._slices_ready = True

    def _expand_action(self, action: np.ndarray) -> np.ndarray:
        if not self._slices_ready:
            self.setup_action_split_idx()
        expected_dim = 12
        action = np.asarray(action, dtype=np.float32)
        if action.ndim != 1 or action.shape[0] != expected_dim:
            raise ValueError(f"PANDAMOBILE_12D expects a 12D action, got shape {action.shape}")

        low_full, _ = super().action_limits
        full_dim = low_full.shape[0]
        expanded = np.zeros(full_dim, dtype=np.float32)

        # Arm pose deltas (6 dims)
        expanded[self._right_slice] = action[0:6]

        # Gripper: replicate single scalar across both fingers
        grip_val = float(action[6])
        if self._gripper_slice.stop > self._gripper_slice.start:
            expanded[self._gripper_slice] = grip_val

        # Base velocities (3 dims)
        expanded[self._base_slice] = action[7:10]

        # Torso delta or setpoint (1 dim)
        if self._torso_slice.stop > self._torso_slice.start:
            expanded[self._torso_slice] = action[10]

        # Hybrid mode flag (final dim)
        expanded[-1] = action[11]
        return expanded

    def set_goal(self, all_action: np.ndarray):
        expanded = self._expand_action(all_action)
        super().set_goal(expanded)

    @property
    def action_limits(self):
        low_full, high_full = super().action_limits
        if not self._slices_ready:
            self.setup_action_split_idx()

        components = [
            (self._right_slice, slice(0, 6)),
            (self._gripper_slice, slice(6, 7)),
            (self._base_slice, slice(7, 10)),
            (self._torso_slice, slice(10, 11)),
        ]

        low_parts = []
        high_parts = []
        for full_slice, target_slice in components:
            if full_slice.stop > full_slice.start:
                low_parts.append(low_full[full_slice][:(target_slice.stop - target_slice.start)])
                high_parts.append(high_full[full_slice][:(target_slice.stop - target_slice.start)])
        # Arm, gripper, base, torso segments were appended in order; append hybrid mode flag.
        low = np.concatenate(low_parts + [low_full[-1:].copy()])
        high = np.concatenate(high_parts + [high_full[-1:].copy()])
        return low, high

    def create_action_vector(self, action_dict):
        # Let parent build the full vector, then compress to 12D
        full_vector = super().create_action_vector(action_dict)
        if not self._slices_ready:
            self.setup_action_split_idx()
        compressed = np.concatenate(
            [
                full_vector[self._right_slice],
                [np.mean(full_vector[self._gripper_slice])],
                full_vector[self._base_slice],
                [np.mean(full_vector[self._torso_slice]) if self._torso_slice.stop > self._torso_slice.start else 0.0],
                [full_vector[-1]],
            ]
        )
        return compressed
