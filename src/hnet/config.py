"""Configuration dataclasses for H-Net components."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AttnConfig:
    """Attention-related per-stage configuration.

    Fields can be lists to support multi-stage architectures.
    """

    num_heads: list[int] = field(default_factory=list)
    rotary_emb_dim: list[int] = field(default_factory=list)
    window_size: list[int] = field(default_factory=list)


@dataclass
class SSMConfig:
    """Simplified SSM mixer configuration."""

    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256


@dataclass
class HNetConfig:
    """Top-level H-Net configuration.

    arch_layout describes the recursive encoder/main/decoder nesting. Each stage can
    refine dimensions and sub-configs via per-stage lists.
    """

    arch_layout: list[str | list] = field(default_factory=list)
    d_model: list[int] = field(default_factory=list)
    d_intermediate: list[int] = field(default_factory=list)
    vocab_size: int = 256
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    tie_embeddings: bool = False
