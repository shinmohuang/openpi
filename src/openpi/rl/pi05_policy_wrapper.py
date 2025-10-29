from __future__ import annotations

import math
import pathlib
from collections.abc import Sequence

import safetensors.torch
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.rl.robocasa_preprocess import RobocasaPi05Preprocessor


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class ExploreNoiseNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, min_std: float, max_std: float, hidden_dims: Sequence[int] = (256, 256), activation: str = "Tanh"):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [in_dim, *hidden_dims, out_dim]
        act_cls = getattr(nn, activation)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("min_logvar", torch.log(torch.tensor(min_std**2, dtype=torch.float32)))
        self.register_buffer("max_logvar", torch.log(torch.tensor(max_std**2, dtype=torch.float32)))

    def set_range(self, min_std: float, max_std: float):
        self.min_logvar.data = torch.log(torch.tensor(min_std**2, dtype=torch.float32, device=self.min_logvar.device))
        self.max_logvar.data = torch.log(torch.tensor(max_std**2, dtype=torch.float32, device=self.max_logvar.device))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logvar = torch.tanh(self.mlp(features))
        logvar = self.min_logvar + (self.max_logvar - self.min_logvar) * (logvar + 1.0) * 0.5
        std = torch.exp(0.5 * logvar)
        return std


class Pi05RobocasaPolicy(nn.Module):
    """π₀․₅ PyTorch policy wrapper with diffusion log-probabilities for RoboCasa RL.

    Only the first action chunk is executed by the environment; PPO log-probabilities
    are computed consistently on that chunk.
    """

    def __init__(
        self,
        *,
        model: pi0_pytorch.PI0Pytorch,
        preprocessor: RobocasaPi05Preprocessor,
        device: torch.device,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        state_dim: int,
        inference_steps: int,
        min_std: float,
        max_std: float,
        include_initial_logprob: bool,
        noise_hidden_dims: Sequence[int] = (256, 256),
        time_dim: int = 32,
    ):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.inference_steps = inference_steps
        self.include_initial_logprob = include_initial_logprob
        self.register_buffer("action_low", self._to_bound_tensor(action_low))
        self.register_buffer("action_high", self._to_bound_tensor(action_high))
        self.min_std = min_std
        self.max_std = max_std
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
        )
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
            nn.Mish(),
        )
        explore_in_dim = 256 + time_dim
        total_act_dim = self.model.config.action_dim * self.model.config.action_horizon
        self.explore_noise_net = ExploreNoiseNet(explore_in_dim, total_act_dim, min_std, max_std, hidden_dims=noise_hidden_dims)
        self.time_dim = time_dim
        self._backbones_frozen = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | pathlib.Path,
        model_config: pi0_config.Pi0Config,
        *,
        device: str | torch.device = "cuda",
        state_dim: int,
        action_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
        inference_steps: int = 10,
        min_std: float | None = None,
        max_std: float | None = None,
        include_initial_logprob: bool = True,
    ) -> "Pi05RobocasaPolicy":
        device = torch.device(device)
        model = pi0_pytorch.PI0Pytorch(model_config).to(device)
        model.eval()

        ckpt_dir = pathlib.Path(checkpoint_dir)
        ckpt_path = ckpt_dir / "model.safetensors"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        weights = safetensors.torch.load_file(str(ckpt_path))
        model.load_state_dict(weights, strict=False)

        preprocessor = RobocasaPi05Preprocessor(model_config=model_config, device=device)

        if action_bounds is None:
            dim = model_config.action_dim
            low = torch.full((dim,), -1.0)
            high = torch.full((dim,), 1.0)
        else:
            low = torch.as_tensor(action_bounds[0], dtype=torch.float32)
            high = torch.as_tensor(action_bounds[1], dtype=torch.float32)

        if min_std is None:
            min_std = 0.05
        if max_std is None:
            max_std = 0.2

        policy = cls(
            model=model,
            preprocessor=preprocessor,
            device=device,
            action_low=low,
            action_high=high,
            state_dim=state_dim,
            inference_steps=inference_steps,
            min_std=min_std,
            max_std=max_std,
            include_initial_logprob=include_initial_logprob,
        )
        return policy.to(device)

    def freeze_backbones(self, *, freeze_vision: bool = True, freeze_language: bool = True) -> None:
        paligemma = getattr(self.model.paligemma_with_expert, "paligemma", None)
        if paligemma is None:
            return
        if freeze_vision and hasattr(paligemma, "vision_tower"):
            paligemma.vision_tower.eval()
            for p in paligemma.vision_tower.parameters():
                p.requires_grad = False
        if freeze_language and hasattr(paligemma, "language_model"):
            paligemma.language_model.eval()
            for p in paligemma.language_model.parameters():
                p.requires_grad = False
        self._backbones_frozen = True

    def freeze_action_expert_base(self, *, keep_lora: bool = True) -> None:
        gemma_expert = getattr(self.model.paligemma_with_expert, "gemma_expert", None)
        if gemma_expert is None:
            return
        mdl = getattr(gemma_expert, "model", None)
        if mdl is None:
            return
        for name, p in mdl.named_parameters():
            if keep_lora and ("lora" in name.lower()):
                p.requires_grad = True
            else:
                p.requires_grad = False
        mdl.train()

    def set_noise_range(self, min_std: float, max_std: float) -> None:
        self.min_std = min_std
        self.max_std = max_std
        self.explore_noise_net.set_range(min_std, max_std)

    def sample_actions(  # type: ignore[override]
        self,
        observations: dict,
        prompts: Sequence[str] | str,
        *,
        eval_mode: bool = False,
        return_chain: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        inputs = self.preprocessor(observations, prompts)
        context = self._prepare_context(inputs)

        noise = self.model.sample_noise(
            (inputs.state.shape[0], self.model.config.action_horizon, self.model.config.action_dim),
            self.device,
        )
        actions, chain, logprob = self._integrate_chain(context, noise, eval_mode=eval_mode, return_chain=True)

        if return_chain:
            return actions, chain, logprob
        return actions, None, None

    def compute_log_prob(
        self,
        observations: dict,
        prompts: Sequence[str] | str,
        chain: torch.Tensor,
        *,
        return_entropy: bool = False,
    ):
        inputs = self.preprocessor(observations, prompts)
        context = self._prepare_context(inputs)
        return self._evaluate_chain_logprob(context, chain, return_entropy=return_entropy)

    def _to_bound_tensor(self, bounds: torch.Tensor | Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(bounds, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.view(1, 1, -1)
        return tensor

    def _prepare_context(self, observation):
        images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(observation, train=False)

        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

            _, past_key_values = self.model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        state_dtype = self.state_encoder[0].weight.dtype
        state = state.to(self.device, dtype=state_dtype)

        return {
            "state": state,
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
        }

    def _integrate_chain(self, context, initial_noise, *, eval_mode: bool, return_chain: bool):
        batch_size = initial_noise.shape[0]
        dt = torch.tensor(-1.0 / self.inference_steps, dtype=torch.float32, device=self.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        x_t = initial_noise
        chain = [x_t]
        state_emb = context["state"].to(self.state_encoder[0].weight.dtype)
        cond_emb = self.state_encoder(state_emb)

        # Only the executed first chunk contributes to log-probability
        logprob = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        if self.include_initial_logprob:
            x0 = x_t[:, 0, :]
            dist0 = Normal(torch.zeros_like(x0, dtype=torch.float32), torch.ones_like(x0, dtype=torch.float32))
            logprob = dist0.log_prob(x0.float()).sum(dim=-1)

        for step in range(self.inference_steps):
            expanded_time = time.expand(batch_size)
            v_t = self.model.denoise_step(
                context["state"],
                context["prefix_pad_masks"],
                context["past_key_values"],
                x_t,
                expanded_time,
            )
            mean = x_t + dt * v_t
            time_feat = self.time_embedding(expanded_time).to(self.device)
            if time_feat.dim() > 2:
                time_feat = time_feat.view(batch_size, -1)
            noise_feat = torch.cat([cond_emb, time_feat], dim=-1)
            sigma = self.explore_noise_net(noise_feat).view(batch_size, self.model.config.action_horizon, self.model.config.action_dim)
            if eval_mode:
                x_next = mean
            else:
                eps = torch.randn_like(mean)
                x_next = mean + sigma * eps

            if step == self.inference_steps - 1:
                x_next = torch.max(torch.min(x_next, self.action_high), self.action_low)

            # Accumulate log-prob for first (executed) chunk only
            mean0 = mean[:, 0, :]
            sigma0 = sigma[:, 0, :]
            x_next0 = x_next[:, 0, :]
            dist0 = Normal(mean0.float(), sigma0.float())
            logprob = logprob + dist0.log_prob(x_next0.float()).sum(dim=-1)

            chain.append(x_next)
            x_t = x_next
            time = time + dt

        actions = x_t.float()
        chain_tensor = torch.stack(chain, dim=1) if return_chain else None
        return actions, chain_tensor, logprob

    def _evaluate_chain_logprob(self, context, chain: torch.Tensor, *, return_entropy: bool):
        batch_size = chain.shape[0]
        dt = torch.tensor(-1.0 / self.inference_steps, dtype=torch.float32, device=self.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        logprob = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        entropy = torch.zeros(batch_size, device=self.device) if return_entropy else None
        if self.include_initial_logprob:
            x0 = chain[:, 0, 0, :]
            dist0 = Normal(torch.zeros_like(x0, dtype=torch.float32), torch.ones_like(x0, dtype=torch.float32))
            logprob = dist0.log_prob(x0.float()).sum(dim=-1)
            if return_entropy:
                entropy = dist0.entropy().sum(dim=-1)

        x_t = chain[:, 0]
        cond_emb = self.state_encoder(context["state"].to(self.state_encoder[0].weight.dtype))
        for step in range(self.inference_steps):
            expanded_time = time.expand(batch_size)
            v_t = self.model.denoise_step(
                context["state"],
                context["prefix_pad_masks"],
                context["past_key_values"],
                x_t,
                expanded_time,
            )
            mean = x_t + dt * v_t
            time_feat = self.time_embedding(expanded_time)
            if time_feat.dim() > 2:
                time_feat = time_feat.view(batch_size, -1)
            noise_feat = torch.cat([cond_emb, time_feat], dim=-1)
            sigma = self.explore_noise_net(noise_feat).view(batch_size, self.model.config.action_horizon, self.model.config.action_dim)

            x_next = chain[:, step + 1]
            mean0 = mean[:, 0, :]
            sigma0 = sigma[:, 0, :]
            x_next0 = x_next[:, 0, :]
            dist0 = Normal(mean0.float(), sigma0.float())
            logprob = logprob + dist0.log_prob(x_next0.float()).sum(dim=-1)
            if return_entropy:
                entropy = entropy + dist0.entropy().sum(dim=-1)

            x_t = x_next
            time = time + dt

        if return_entropy:
            return logprob, entropy
        return logprob
from __future__ import annotations

import math
import pathlib
from collections.abc import Sequence

import safetensors.torch
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.rl.robocasa_preprocess import RobocasaPi05Preprocessor


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class ExploreNoiseNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, min_std: float, max_std: float, hidden_dims: Sequence[int] = (256, 256), activation: str = "Tanh"):
        super().__init__()
        layers: list[nn.Module] = []
        dims = [in_dim, *hidden_dims, out_dim]
        act_cls = getattr(nn, activation)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("min_logvar", torch.log(torch.tensor(min_std**2, dtype=torch.float32)))
        self.register_buffer("max_logvar", torch.log(torch.tensor(max_std**2, dtype=torch.float32)))

    def set_range(self, min_std: float, max_std: float):
        self.min_logvar.data = torch.log(torch.tensor(min_std**2, dtype=torch.float32, device=self.min_logvar.device))
        self.max_logvar.data = torch.log(torch.tensor(max_std**2, dtype=torch.float32, device=self.max_logvar.device))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logvar = torch.tanh(self.mlp(features))
        logvar = self.min_logvar + (self.max_logvar - self.min_logvar) * (logvar + 1.0) * 0.5
        std = torch.exp(0.5 * logvar)
        return std


class Pi05RobocasaPolicy(nn.Module):
    """π₀․₅ PyTorch policy wrapper with diffusion log-probabilities for RoboCasa RL.

    Only the first action chunk is executed by the environment; PPO log-probabilities
    are computed consistently on that chunk.
    """

    def __init__(
        self,
        *,
        model: pi0_pytorch.PI0Pytorch,
        preprocessor: RobocasaPi05Preprocessor,
        device: torch.device,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        state_dim: int,
        inference_steps: int,
        min_std: float,
        max_std: float,
        include_initial_logprob: bool,
        noise_hidden_dims: Sequence[int] = (256, 256),
        time_dim: int = 32,
    ):
        super().__init__()
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.inference_steps = inference_steps
        self.include_initial_logprob = include_initial_logprob
        self.register_buffer("action_low", self._to_bound_tensor(action_low))
        self.register_buffer("action_high", self._to_bound_tensor(action_high))
        self.min_std = min_std
        self.max_std = max_std
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Mish(),
            nn.Linear(256, 256),
            nn.Mish(),
        )
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
            nn.Mish(),
        )
        explore_in_dim = 256 + time_dim
        total_act_dim = self.model.config.action_dim * self.model.config.action_horizon
        self.explore_noise_net = ExploreNoiseNet(explore_in_dim, total_act_dim, min_std, max_std, hidden_dims=noise_hidden_dims)
        self.time_dim = time_dim
        self._backbones_frozen = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str | pathlib.Path,
        model_config: pi0_config.Pi0Config,
        *,
        device: str | torch.device = "cuda",
        state_dim: int,
        action_bounds: tuple[Sequence[float], Sequence[float]] | None = None,
        inference_steps: int = 10,
        min_std: float | None = None,
        max_std: float | None = None,
        include_initial_logprob: bool = True,
    ) -> "Pi05RobocasaPolicy":
        device = torch.device(device)
        model = pi0_pytorch.PI0Pytorch(model_config).to(device)
        model.eval()

        ckpt_dir = pathlib.Path(checkpoint_dir)
        ckpt_path = ckpt_dir / "model.safetensors"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        weights = safetensors.torch.load_file(str(ckpt_path))
        model.load_state_dict(weights, strict=False)

        preprocessor = RobocasaPi05Preprocessor(model_config=model_config, device=device)

        if action_bounds is None:
            dim = model_config.action_dim
            low = torch.full((dim,), -1.0)
            high = torch.full((dim,), 1.0)
        else:
            low = torch.as_tensor(action_bounds[0], dtype=torch.float32)
            high = torch.as_tensor(action_bounds[1], dtype=torch.float32)

        if min_std is None:
            min_std = 0.05
        if max_std is None:
            max_std = 0.2

        policy = cls(
            model=model,
            preprocessor=preprocessor,
            device=device,
            action_low=low,
            action_high=high,
            state_dim=state_dim,
            inference_steps=inference_steps,
            min_std=min_std,
            max_std=max_std,
            include_initial_logprob=include_initial_logprob,
        )
        return policy.to(device)

    def freeze_backbones(self, *, freeze_vision: bool = True, freeze_language: bool = True) -> None:
        paligemma = getattr(self.model.paligemma_with_expert, "paligemma", None)
        if paligemma is None:
            return
        if freeze_vision and hasattr(paligemma, "vision_tower"):
            paligemma.vision_tower.eval()
            for p in paligemma.vision_tower.parameters():
                p.requires_grad = False
        if freeze_language and hasattr(paligemma, "language_model"):
            paligemma.language_model.eval()
            for p in paligemma.language_model.parameters():
                p.requires_grad = False
        self._backbones_frozen = True

    def freeze_action_expert_base(self, *, keep_lora: bool = True) -> None:
        gemma_expert = getattr(self.model.paligemma_with_expert, "gemma_expert", None)
        if gemma_expert is None:
            return
        mdl = getattr(gemma_expert, "model", None)
        if mdl is None:
            return
        for name, p in mdl.named_parameters():
            if keep_lora and ("lora" in name.lower()):
                p.requires_grad = True
            else:
                p.requires_grad = False
        mdl.train()

    def set_noise_range(self, min_std: float, max_std: float) -> None:
        self.min_std = min_std
        self.max_std = max_std
        self.explore_noise_net.set_range(min_std, max_std)

    def sample_actions(  # type: ignore[override]
        self,
        observations: dict,
        prompts: Sequence[str] | str,
        *,
        eval_mode: bool = False,
        return_chain: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        inputs = self.preprocessor(observations, prompts)
        context = self._prepare_context(inputs)

        noise = self.model.sample_noise(
            (inputs.state.shape[0], self.model.config.action_horizon, self.model.config.action_dim),
            self.device,
        )
        actions, chain, logprob = self._integrate_chain(context, noise, eval_mode=eval_mode, return_chain=True)

        if return_chain:
            return actions, chain, logprob
        return actions, None, None

    def compute_log_prob(
        self,
        observations: dict,
        prompts: Sequence[str] | str,
        chain: torch.Tensor,
        *,
        return_entropy: bool = False,
    ):
        inputs = self.preprocessor(observations, prompts)
        context = self._prepare_context(inputs)
        return self._evaluate_chain_logprob(context, chain, return_entropy=return_entropy)

    def _to_bound_tensor(self, bounds: torch.Tensor | Sequence[float]) -> torch.Tensor:
        tensor = torch.as_tensor(bounds, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.view(1, 1, -1)
        return tensor

    def _prepare_context(self, observation):
        images, img_masks, lang_tokens, lang_masks, state = self.model._preprocess_observation(observation, train=False)

        with torch.no_grad():
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            prefix_att_2d_masks_4d = self.model._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

            _, past_key_values = self.model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        state_dtype = self.state_encoder[0].weight.dtype
        state = state.to(self.device, dtype=state_dtype)

        return {
            "state": state,
            "prefix_pad_masks": prefix_pad_masks,
            "past_key_values": past_key_values,
        }

    def _integrate_chain(self, context, initial_noise, *, eval_mode: bool, return_chain: bool):
        batch_size = initial_noise.shape[0]
        dt = torch.tensor(-1.0 / self.inference_steps, dtype=torch.float32, device=self.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        x_t = initial_noise
        chain = [x_t]
        state_emb = context["state"].to(self.state_encoder[0].weight.dtype)
        cond_emb = self.state_encoder(state_emb)

        # Only the executed first chunk contributes to log-probability
        logprob = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        if self.include_initial_logprob:
            x0 = x_t[:, 0, :]
            dist0 = Normal(torch.zeros_like(x0, dtype=torch.float32), torch.ones_like(x0, dtype=torch.float32))
            logprob = dist0.log_prob(x0.float()).sum(dim=-1)

        for step in range(self.inference_steps):
            expanded_time = time.expand(batch_size)
            v_t = self.model.denoise_step(
                context["state"],
                context["prefix_pad_masks"],
                context["past_key_values"],
                x_t,
                expanded_time,
            )
            mean = x_t + dt * v_t
            time_feat = self.time_embedding(expanded_time).to(self.device)
            if time_feat.dim() > 2:
                time_feat = time_feat.view(batch_size, -1)
            noise_feat = torch.cat([cond_emb, time_feat], dim=-1)
            sigma = self.explore_noise_net(noise_feat).view(batch_size, self.model.config.action_horizon, self.model.config.action_dim)
            if eval_mode:
                x_next = mean
            else:
                eps = torch.randn_like(mean)
                x_next = mean + sigma * eps

            if step == self.inference_steps - 1:
                x_next = torch.max(torch.min(x_next, self.action_high), self.action_low)

            # Accumulate log-prob for first (executed) chunk only
            mean0 = mean[:, 0, :]
            sigma0 = sigma[:, 0, :]
            x_next0 = x_next[:, 0, :]
            dist0 = Normal(mean0.float(), sigma0.float())
            logprob = logprob + dist0.log_prob(x_next0.float()).sum(dim=-1)

            chain.append(x_next)
            x_t = x_next
            time = time + dt

        actions = x_t.float()
        chain_tensor = torch.stack(chain, dim=1) if return_chain else None
        return actions, chain_tensor, logprob

    def _evaluate_chain_logprob(self, context, chain: torch.Tensor, *, return_entropy: bool):
        batch_size = chain.shape[0]
        dt = torch.tensor(-1.0 / self.inference_steps, dtype=torch.float32, device=self.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=self.device)

        logprob = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        entropy = torch.zeros(batch_size, device=self.device) if return_entropy else None
        if self.include_initial_logprob:
            x0 = chain[:, 0, 0, :]
            dist0 = Normal(torch.zeros_like(x0, dtype=torch.float32), torch.ones_like(x0, dtype=torch.float32))
            logprob = dist0.log_prob(x0.float()).sum(dim=-1)
            if return_entropy:
                entropy = dist0.entropy().sum(dim=-1)

        x_t = chain[:, 0]
        cond_emb = self.state_encoder(context["state"].to(self.state_encoder[0].weight.dtype))
        for step in range(self.inference_steps):
            expanded_time = time.expand(batch_size)
            v_t = self.model.denoise_step(
                context["state"],
                context["prefix_pad_masks"],
                context["past_key_values"],
                x_t,
                expanded_time,
            )
            mean = x_t + dt * v_t
            time_feat = self.time_embedding(expanded_time)
            if time_feat.dim() > 2:
                time_feat = time_feat.view(batch_size, -1)
            noise_feat = torch.cat([cond_emb, time_feat], dim=-1)
            sigma = self.explore_noise_net(noise_feat).view(batch_size, self.model.config.action_horizon, self.model.config.action_dim)

            x_next = chain[:, step + 1]
            mean0 = mean[:, 0, :]
            sigma0 = sigma[:, 0, :]
            x_next0 = x_next[:, 0, :]
            dist0 = Normal(mean0.float(), sigma0.float())
            logprob = logprob + dist0.log_prob(x_next0.float()).sum(dim=-1)
            if return_entropy:
                entropy = entropy + dist0.entropy().sum(dim=-1)

            x_t = x_next
            time = time + dt

        if return_entropy:
            return logprob, entropy
        return logprob
