#!/usr/bin/env python
"""Extract a lightweight policy overlay from a full training checkpoint.

This reads a checkpoint produced by scripts/train_ppo_robocasa.py and writes a
small .pt file that only contains trainable policy parameters (RL heads + LoRA),
which can be applied on top of the base π0.5 checkpoint at eval time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import tyro


ALLOWED_PREFIXES: tuple[str, ...] = (
    # RL heads
    "state_encoder.",
    "time_embedding.",
    "explore_noise_net.",
)


def _looks_like_lora_key(key: str) -> bool:
    k = key.lower()
    # Typical LoRA naming in Gemma expert
    return ("lora" in k) or ("l0ra" in k)


def filter_policy_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep only RL heads + LoRA adapter parameters and their immediate buffers."""
    out: dict[str, torch.Tensor] = {}

    # 1) Direct allow-list prefixes
    for k, v in state.items():
        if any(k.startswith(pref) for pref in ALLOWED_PREFIXES):
            out[k] = v

    # 2) LoRA params in Gemma expert
    for k, v in state.items():
        if "gemma_expert.model" in k and _looks_like_lora_key(k):
            out[k] = v

    # 3) Shallow buffers under allowed modules (e.g., running stats if any)
    allowed_modules: set[str] = set(k.rsplit(".", 1)[0] for k in out.keys() if "." in k)
    for k, v in state.items():
        if k in out:
            continue
        mod = k.rsplit(".", 1)[0] if "." in k else ""
        if mod in allowed_modules:
            out[k] = v

    return out


@dataclass
class Args:
    checkpoint: str
    out: str


def main(args: Args) -> None:
    ckpt = torch.load(Path(args.checkpoint), map_location="cpu")
    # Support either full payload (with 'policy') or raw state dict
    pol_state = ckpt.get("policy", ckpt)
    if not isinstance(pol_state, dict):
        raise ValueError("Checkpoint does not contain a valid 'policy' state_dict")

    filtered = filter_policy_state(pol_state)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": filtered}, args.out)
    print(f"Wrote overlay with {len(filtered)} tensors -> {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))

