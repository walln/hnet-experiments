import re
from dataclasses import dataclass

import flax.nnx as nnx
import jax.numpy as jnp

from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from hnet.modules.block import HybridBlock, create_block
from hnet.modules.cache import CacheState
from hnet.modules.utils import get_seq_idx, get_stage_cfg


@dataclass
class IsotropicInferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficiently calculate and store the context during inference."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    cache_state: CacheState | None = None
    lengths_per_sample: jnp.ndarray | None = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.lengths_per_sample is not None:
            self.lengths_per_sample = jnp.zeros_like(self.lengths_per_sample)


def _get_stage_ssm_cfg(base_cfg: SSMConfig, stage_idx: int) -> SSMConfig:
    """Extract stage-specific SSM config."""
    cfg_dict = get_stage_cfg(base_cfg, stage_idx)
    return SSMConfig(**cfg_dict)


def _get_stage_attn_cfg(base_cfg: AttnConfig, stage_idx: int) -> AttnConfig:
    """Extract stage-specific attention config."""
    cfg_dict = get_stage_cfg(base_cfg, stage_idx)
    return AttnConfig(**cfg_dict)


class Isotropic(nnx.Module):
    """Isotropic architecture module."""

    layers: list[HybridBlock]

    def __init__(
        self,
        config: HNetConfig,
        pos_idx: int,
        stage_idx: int,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize Isotropic module.

        Args:
            config: HNetConfig configuration
            pos_idx: Position index in the architecture layout
            stage_idx: Stage index for multi-stage models
            rngs: Random number generators
        """
        self.stage_idx = stage_idx

        # Handle potential None values in config lists
        if config.d_model is None or len(config.d_model) <= stage_idx:
            raise ValueError(f"d_model not configured for stage {stage_idx}")
        if config.d_intermediate is None or len(config.d_intermediate) <= stage_idx:
            raise ValueError(f"d_intermediate not configured for stage {stage_idx}")
        if config.arch_layout is None or len(config.arch_layout) == 0:
            raise ValueError("arch_layout must be provided")

        self.d_model = config.d_model[self.stage_idx]

        # Ensure configs are initialized
        if config.ssm_cfg is None:
            raise ValueError("ssm_cfg must be provided")
        if config.attn_cfg is None:
            raise ValueError("attn_cfg must be provided")

        self.ssm_cfg = _get_stage_ssm_cfg(config.ssm_cfg, stage_idx)
        self.attn_cfg = _get_stage_attn_cfg(config.attn_cfg, stage_idx)

        # Parse architecture layout
        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            if isinstance(arch_layout, list) and len(arch_layout) > 1:
                arch_layout = arch_layout[1]
            else:
                raise ValueError(f"Invalid arch_layout for stage {stage_idx}")

        if isinstance(arch_layout, list):
            arch_layout = arch_layout[pos_idx]

        # Ensure arch_layout is a string for regex
        if not isinstance(arch_layout, str):
            raise ValueError(
                f"Expected string arch_layout at position {pos_idx}, got {type(arch_layout)}"
            )

        layout_parse = re.findall(r"([mMtT])(\d+)", arch_layout)

        # Create layers
        layers = []
        layer_idx = 0
        self.arch_full = []

        for arch, n_layer in layout_parse:
            assert arch in ("m", "M", "t", "T")
            assert n_layer.isdigit()

            for i in range(int(n_layer)):
                block = create_block(
                    arch,
                    self.d_model,
                    d_intermediate=config.d_intermediate[self.stage_idx],
                    ssm_cfg=self.ssm_cfg,
                    attn_cfg=self.attn_cfg,
                    layer_idx=(layer_idx + i),
                    rngs=rngs,
                )
                layers.append(block)
                self.arch_full.append(arch)
            layer_idx += int(n_layer)

        self.layers = layers
        self.rmsnorm = nnx.RMSNorm(self.d_model, epsilon=1e-5, rngs=rngs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """
        Allocate the inference cache for the Isotropic module.

        Arguments:
            batch_size: int. The number of sequences in the batch.
            max_seqlen: int. The maximum sequence length in the batch.
            dtype: jnp.dtype. The dtype of the inference cache.

        Returns:
            IsotropicInferenceParams with allocated cache
        """
        # Create empty cache state
        cache = CacheState.empty()

        return IsotropicInferenceParams(
            cache_state=cache,
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
        )

    def __call__(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        mask=None,
        inference_params=None,
        **mixer_kwargs,
    ):
        """
        Forward pass through the Isotropic module.

        Args:
            hidden_states: Input hidden states
            cu_seqlens: Cumulative sequence lengths for packed format
            max_seqlen: Maximum sequence length
            mask: Attention mask for unpacked format
            inference_params: Inference parameters with cache
            **mixer_kwargs: Additional arguments for mixers

        Returns:
            Output hidden states
        """
        assert (mask is not None) or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Either mask or cu_seqlens and max_seqlen must be provided"

        # Prepare mixer kwargs
        attn_mixer_kwargs = dict(mixer_kwargs)
        ssm_mixer_kwargs = dict(mixer_kwargs)

        if mask is not None:
            packed = False
            assert hidden_states.ndim == 3, (
                "Hidden states must be (B, L, D) in unpacked mode"
            )
        else:
            if cu_seqlens is not None:
                attn_mixer_kwargs.update(
                    {
                        "cu_seqlens": cu_seqlens.astype(jnp.int32),
                        "max_seqlen": max_seqlen,
                    }
                )
                ssm_mixer_kwargs.update({"seq_idx": get_seq_idx(cu_seqlens)})
            packed = True

        cache = inference_params.cache_state if inference_params else None

        for _layer_idx, (layer, arch) in enumerate(
            zip(self.layers, self.arch_full, strict=False)
        ):
            if arch in ("m", "M"):
                # Mamba expects (B, L, D) format
                if hidden_states.ndim == 2:
                    hidden_states = hidden_states[None, :, :]
            elif arch in ("t", "T"):
                # Attention in packed mode expects (total_tokens, D)
                if hidden_states.ndim == 3 and packed:
                    hidden_states = jnp.squeeze(hidden_states, axis=0)
            else:
                raise NotImplementedError(f"Architecture {arch} not supported")

            # Forward through layer
            hidden_states, cache = layer(hidden_states, cache=cache)

        # Final norm
        hidden_states = self.rmsnorm(hidden_states)

        if hidden_states.ndim == 3 and packed:
            hidden_states = jnp.squeeze(hidden_states, axis=0)

        if inference_params is not None and mask is not None:
            # Update sequence offset
            assert mask.shape[0] == 1, "seqlen_offset handling assumes batch size 1"
            inference_params.seqlen_offset += hidden_states.shape[1]

        return hidden_states

    def step(self, hidden_states, inference_params):
        """
        Single step for autoregressive generation.

        Assumes hidden_states is (B, 1, D). Steps each of the layers in order.

        Args:
            hidden_states: Input hidden states (B, 1, D)
            inference_params: Inference parameters with cache

        Returns:
            Output hidden states (B, 1, D)
        """
        cache = inference_params.cache_state

        for _layer_idx, layer in enumerate(self.layers):
            # Forward through layer (step is just forward with seq_len=1)
            hidden_states, cache = layer(hidden_states, cache=cache)

        # Final norm
        hidden_states = self.rmsnorm(hidden_states)
        inference_params.seqlen_offset += 1

        return hidden_states
