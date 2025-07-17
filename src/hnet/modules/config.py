"""Configuration dataclasses for H-Net modules."""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class Mamba2Config:
    """Configuration for Mamba2 layers."""

    d_model: int
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    A_init_range: tuple[float, float] = (1.0, 16.0)
    D_has_hdim: bool = False
    rmsnorm: bool = True
    norm_before_gate: bool = False
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    bias: bool = False
    conv_bias: bool = True
    chunk_size: int = 256
    norm_epsilon: float = 1e-5
    layer_idx: int | None = None


@dataclass
class AttentionConfig:
    """Configuration for attention layers."""

    d_model: int
    num_heads: int
    qkv_proj_bias: bool = False
    out_proj_bias: bool = False
    window_size: int = -1
    softmax_scale: float | None = None
    layer_idx: int | None = None
    rotary_emb_dim: int = 0
    rotary_emb_base: float = 10000.0
    rotary_emb_interleaved: bool = False


@dataclass
class HybridConfig:
    """Configuration for hybrid blocks."""

    d_model: int
    n_heads: int
    use_mamba: bool = True
    mlp_expand: int = 4
    norm_epsilon: float = 1e-5
    layer_idx: int | None = None
    # Mamba-specific
    mamba_config: Mamba2Config | None = None
    # Attention-specific
    attention_config: AttentionConfig | None = None

    def __post_init__(self):
        """Initialize sub-configs if not provided."""
        if self.use_mamba and self.mamba_config is None:
            self.mamba_config = Mamba2Config(d_model=self.d_model)
        elif not self.use_mamba and self.attention_config is None:
            self.attention_config = AttentionConfig(
                d_model=self.d_model, num_heads=self.n_heads
            )


@dataclass
class InferenceConfig:
    """Unified configuration for inference/generation mode."""

    batch_size: int
    max_seq_len: int
    current_seq_len: int = 0
    dtype: jnp.dtype = jnp.float32


@dataclass
class MLPConfig:
    """Configuration for MLP/FFN layers."""

    d_model: int
    d_intermediate: int | None = None
    bias: bool = False
    multiple_of: int = 128
    dtype: jnp.dtype | None = None
