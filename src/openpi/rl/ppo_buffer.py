from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch


def _clone_observation(obs: dict[str, Any]) -> dict[str, Any]:
    cloned = {
        "state": np.array(obs["state"], copy=True),
        "image": {k: np.array(v, copy=True) for k, v in obs["image"].items()},
    }
    return cloned


@dataclass
class PPORolloutBuffer:
    num_steps: int
    num_envs: int
    device: torch.device

    def __post_init__(self):
        self.clear()

    def clear(self):
        self.observations: list[dict[str, Any]] = []
        self.prompts: list[list[str]] = []
        self.actions: list[torch.Tensor] = []
        self.chains: list[torch.Tensor] = []
        self.logprobs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None
        self.next_observation: dict[str, Any] | None = None
        self.next_prompts: list[str] | None = None

    def add(
        self,
        observation: dict[str, Any],
        prompts: list[str],
        actions: torch.Tensor,
        chains: torch.Tensor,
        logprobs: torch.Tensor,
        values: torch.Tensor,
        rewards: np.ndarray,
        dones: np.ndarray,
    ):
        self.observations.append(_clone_observation(observation))
        self.prompts.append(list(prompts))
        self.actions.append(actions.detach().cpu())
        self.chains.append(chains.detach().cpu())
        self.logprobs.append(logprobs.detach().cpu())
        self.values.append(values.detach().cpu())
        self.rewards.append(torch.from_numpy(rewards).float())
        self.dones.append(torch.from_numpy(dones.astype(np.bool_)))

    def set_next_observation(self, observation: dict[str, Any], prompts: list[str]):
        self.next_observation = _clone_observation(observation)
        self.next_prompts = list(prompts)

    def compute_returns(self, last_values: torch.Tensor, gamma: float, gae_lambda: float):
        rewards = torch.stack(self.rewards).to(self.device)  # [T, N]
        values = torch.stack(self.values).to(self.device)  # [T, N]
        dones = torch.stack(self.dones).to(self.device)  # [T, N]
        advantages = torch.zeros_like(rewards, device=self.device)

        next_values = last_values.to(self.device)
        next_advantage = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.num_steps)):
            mask = 1.0 - dones[t].float()
            delta = rewards[t] + gamma * next_values * mask - values[t]
            next_advantage = delta + gamma * gae_lambda * mask * next_advantage
            advantages[t] = next_advantage
            next_values = values[t]

        self.advantages = advantages
        self.returns = advantages + values

    def iter_minibatches(self, batch_size: int) -> Iterator[dict[str, Any]]:
        if self.advantages is None or self.returns is None:
            raise ValueError("Call compute_returns before iterating minibatches.")

        total_samples = self.num_steps * self.num_envs
        indices = torch.randperm(total_samples)

        for start in range(0, total_samples, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield self._gather_batch(batch_idx.tolist())

    # Internal helpers -----------------------------------------------------

    def _gather_batch(self, flat_indices: list[int]) -> dict[str, Any]:
        obs_batch = []
        prompt_batch: list[str] = []
        chain_batch = []
        old_logprob_batch = []
        advantage_batch = []
        return_batch = []

        for flat in flat_indices:
            step = flat // self.num_envs
            env_idx = flat % self.num_envs
            obs = self.observations[step]
            prompt = self.prompts[step][env_idx]
            chain = self.chains[step][env_idx]
            logprob = self.logprobs[step][env_idx]

            obs_batch.append(self._slice_observation(obs, env_idx))
            prompt_batch.append(prompt)
            chain_batch.append(chain)
            old_logprob_batch.append(logprob)
            advantage_batch.append(self.advantages[step, env_idx])
            return_batch.append(self.returns[step, env_idx])

        batched_obs = self._stack_observations(obs_batch)
        chains = torch.stack(chain_batch, dim=0).to(self.device)
        old_logprobs = torch.stack(old_logprob_batch, dim=0).to(self.device)
        advantages = torch.stack(advantage_batch, dim=0).to(self.device)
        returns = torch.stack(return_batch, dim=0).to(self.device)

        return {
            "observations": batched_obs,
            "prompts": prompt_batch,
            "chains": chains,
            "old_logprobs": old_logprobs,
            "advantages": advantages,
            "returns": returns,
        }

    @staticmethod
    def _slice_observation(observation: dict[str, Any], index: int) -> dict[str, Any]:
        return {
            "state": observation["state"][index],
            "image": {k: v[index] for k, v in observation["image"].items()},
        }

    @staticmethod
    def _stack_observations(samples: list[dict[str, Any]]) -> dict[str, Any]:
        state = np.stack([sample["state"] for sample in samples], axis=0)
        image_keys = samples[0]["image"].keys()
        images = {key: np.stack([sample["image"][key] for sample in samples], axis=0) for key in image_keys}
        return {"state": state, "image": images}
