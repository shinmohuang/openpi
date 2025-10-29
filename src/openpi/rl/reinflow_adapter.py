import dataclasses
from collections.abc import Mapping

import torch

from third_party.ReinFlow.model.flow.ft_ppo.ppoflow import PPOFlow
from third_party.ReinFlow.model.flow.mlp_flow import FlowMLP

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch
from openpi.rl.robocasa_preprocess import RobocasaPi05Preprocessor


@dataclasses.dataclass
class Pi05FlowBackbone(torch.nn.Module):
    """Wrap π₀․₅ policy to expose a ReinFlow-compatible interface."""

    model: pi0_pytorch.PI0Pytorch
    preprocessor: RobocasaPi05Preprocessor

    def forward(self, action: torch.Tensor, time: torch.Tensor, cond: Mapping[str, torch.Tensor]):
        raise NotImplementedError("Pi05FlowBackbone forward is not implemented.")
