import dataclasses
from collections.abc import Sequence

import numpy as np
import torch

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.policies import robocasa_policy


@dataclasses.dataclass
class RobocasaPi05Preprocessor:
    """Converts RoboCasa environment observations into π₀․₅ model inputs."""

    model_config: pi0_config.Pi0Config
    device: torch.device | str = "cuda"

    def __post_init__(self):
        if not self.model_config.pi05:
            raise ValueError("RobocasaPi05Preprocessor requires a pi0.5 model configuration.")

        self._device = torch.device(self.device)
        self._input_transform = robocasa_policy.RobocasaInputs(model_type=_model.ModelType.PI05)
        tokenizer = _tokenizer.PaligemmaTokenizer(self.model_config.max_token_len)
        self._model_transforms = (
            _transforms.ResizeImages(*_model.IMAGE_RESOLUTION),
            _transforms.TokenizePrompt(
                tokenizer,
                discrete_state_input=self.model_config.discrete_state_input,
            ),
            _transforms.PadStatesAndActions(self.model_config.action_dim),
        )
        self._image_keys = _model.IMAGE_KEYS

    def __call__(
        self,
        observations: dict[str, np.ndarray],
        prompts: Sequence[str] | str,
    ) -> _model.Observation[torch.Tensor]:
        """Prepare a batch of observations for π₀․₅ inference.

        Args:
            observations: Dictionary returned by the RoboCasa Gym wrapper.
                Expected keys:
                    - "state": np.ndarray[..., state_dim]
                    - "image": dict with camera views (`observation/image`, `observation/image_right`,
                      `observation/wrist_image`)
            prompts: Per-environment language prompts (sequence length must match batch size) or a single string.

        Returns:
            `_model.Observation` whose leaves are torch tensors on the configured device.
        """
        batch_states = np.asarray(observations["state"])
        if batch_states.ndim == 1:
            batch_states = batch_states[None, ...]

        image_dict = observations["image"]
        base_images = np.asarray(image_dict["observation/image"])
        if base_images.ndim == 3:
            base_images = base_images[None, ...]

        right_images_raw = image_dict.get("observation/image_right")
        if right_images_raw is None:
            right_images_raw = np.zeros_like(base_images)
        right_images = np.asarray(right_images_raw)
        if right_images.ndim == 3:
            right_images = right_images[None, ...]

        wrist_images = np.asarray(image_dict["observation/wrist_image"])
        if wrist_images.ndim == 3:
            wrist_images = wrist_images[None, ...]

        batch_size = batch_states.shape[0]
        if isinstance(prompts, str):
            prompt_list = [prompts] * batch_size
        else:
            prompt_list = list(prompts)
            if len(prompt_list) != batch_size:
                raise ValueError(f"Expected {batch_size} prompts, got {len(prompt_list)}")

        processed_samples = []
        for idx in range(batch_size):
            sample = {
                "observation/state": batch_states[idx],
                "observation/image": base_images[idx],
                "observation/image_right": right_images[idx],
                "observation/wrist_image": wrist_images[idx],
                "prompt": prompt_list[idx],
            }
            data = self._input_transform(sample)
            for transform in self._model_transforms:
                data = transform(data)
            processed_samples.append(data)

        stacked = self._stack_samples(processed_samples)
        return self._to_torch_observation(stacked)

    def _stack_samples(self, samples: list[dict]) -> dict:
        def stack_leaf(key: str):
            return np.stack([np.asarray(sample[key]) for sample in samples], axis=0)

        images = {
            key: np.stack([np.asarray(sample["image"][key]) for sample in samples], axis=0)
            for key in self._image_keys
        }
        image_masks = {
            key: np.stack([np.asarray(sample["image_mask"][key]) for sample in samples], axis=0)
            for key in self._image_keys
        }

        stacked: dict = {
            "state": stack_leaf("state"),
            "image": images,
            "image_mask": image_masks,
        }

        if "tokenized_prompt" in samples[0]:
            stacked["tokenized_prompt"] = stack_leaf("tokenized_prompt")
        if "tokenized_prompt_mask" in samples[0]:
            stacked["tokenized_prompt_mask"] = stack_leaf("tokenized_prompt_mask")

        return stacked

    def _to_torch_observation(self, data: dict) -> _model.Observation[torch.Tensor]:
        images = {}
        for key in self._image_keys:
            array = data["image"][key]
            tensor = torch.from_numpy(array).to(self._device, dtype=torch.float32)
            if tensor.ndim == 4:
                # Convert HWC -> CHW if needed
                if tensor.shape[-1] == 3 and tensor.shape[1] != 3:
                    tensor = tensor.permute(0, 3, 1, 2)
            images[key] = tensor / 255.0 * 2.0 - 1.0
        image_masks = {
            key: torch.from_numpy(data["image_mask"][key]).to(self._device, dtype=torch.bool)
            for key in self._image_keys
        }
        state = torch.from_numpy(data["state"]).to(self._device, dtype=torch.float32)

        tokenized_prompt = None
        tokenized_prompt_mask = None
        if "tokenized_prompt" in data:
            tokenized_prompt = torch.from_numpy(data["tokenized_prompt"]).to(self._device, dtype=torch.int32)
        if "tokenized_prompt_mask" in data:
            tokenized_prompt_mask = torch.from_numpy(data["tokenized_prompt_mask"]).to(self._device, dtype=torch.bool)

        return _model.Observation(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )
