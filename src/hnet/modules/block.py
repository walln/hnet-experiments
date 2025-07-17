# Copyright (c) 2025, Nick Wall.
# JAX implementation of hybrid transformer blocks combining Mamba2 and attention


import flax.nnx as nnx
import jax

from hnet.modules.cache import CacheState, create_mamba2_cache
from hnet.modules.config import HybridConfig, Mamba2Config
from hnet.modules.mamba2 import Mamba2Block
from hnet.modules.mha import CausalMHA
from hnet.modules.swiglu import SwiGLU


class HybridBlock(nnx.Module):
    """
    A hybrid block that can use either Mamba2 or Attention as the sequence mixing layer.

    This follows the structure of modern hybrid architectures like Jamba.
    """

    mixer: Mamba2Block | CausalMHA

    def __init__(
        self,
        config: HybridConfig,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize hybrid block.

        Args:
            config: Hybrid block configuration
            rngs: Random number generators
        """
        self.config = config
        self.use_mamba = config.use_mamba
        self.layer_idx = config.layer_idx

        # Pre-norm for sequence mixing
        self.norm1 = nnx.RMSNorm(config.d_model, epsilon=config.norm_epsilon, rngs=rngs)

        # Sequence mixing layer
        if config.use_mamba:
            assert config.mamba_config is not None
            self.mixer = Mamba2Block(
                config=config.mamba_config,
                rngs=rngs,
            )
        else:
            assert config.attention_config is not None
            self.mixer = CausalMHA(
                config=config.attention_config,
                rngs=rngs,
            )

        # Pre-norm for MLP
        self.norm2 = nnx.RMSNorm(config.d_model, epsilon=config.norm_epsilon, rngs=rngs)

        # MLP
        d_intermediate = config.d_model * config.mlp_expand
        self.mlp = SwiGLU(
            d_model=config.d_model,
            d_intermediate=d_intermediate,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        cache: CacheState | None = None,
    ) -> tuple[jax.Array, CacheState | None]:
        """
        Forward pass of hybrid block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            cache: Optional cache state for inference mode

        Returns:
            output: Output tensor of shape (batch, seq_len, d_model)
            updated_cache: Updated cache state (if cache was provided)
        """
        # Sequence mixing with residual
        residual = x
        x = self.norm1(x)

        # Use isinstance to help the type checker
        if isinstance(self.mixer, Mamba2Block):
            # Get Mamba cache for this layer if available
            mamba_cache = (
                cache.get_mamba(self.layer_idx)
                if cache and self.layer_idx is not None
                else None
            )
            x, updated_mamba_cache = self.mixer(x, cache=mamba_cache)

            # Update cache if needed
            if (
                cache is not None
                and updated_mamba_cache is not None
                and self.layer_idx is not None
            ):
                cache = cache.update_mamba(self.layer_idx, updated_mamba_cache)

        elif isinstance(self.mixer, CausalMHA):
            # Get attention cache for this layer if available
            attn_cache = (
                cache.get_attention(self.layer_idx)
                if cache and self.layer_idx is not None
                else None
            )
            x, updated_attn_cache = self.mixer(x, cache=attn_cache)

            # Update cache if needed
            if (
                cache is not None
                and updated_attn_cache is not None
                and self.layer_idx is not None
            ):
                cache = cache.update_attention(self.layer_idx, updated_attn_cache)
        else:
            raise ValueError(f"Unknown mixer type: {type(self.mixer)}")

        x = residual + x

        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x, cache


class Mamba2Wrapper(nnx.Module):
    """
    Mamba2 wrapper class that has the same inference interface as the CausalMHA class.

    This provides compatibility with existing transformer code.
    """

    def __init__(
        self,
        config: Mamba2Config,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize Mamba2 wrapper."""
        self.config = config
        self.layer_idx = config.layer_idx
        self.mamba = Mamba2Block(
            config=config,
            rngs=rngs,
        )

    def __call__(
        self, x: jax.Array, cache: CacheState | None = None
    ) -> tuple[jax.Array, CacheState | None]:
        """Forward pass through wrapped Mamba2Block."""
        # Get Mamba cache for this layer if available
        mamba_cache = (
            cache.get_mamba(self.layer_idx)
            if cache and self.layer_idx is not None
            else None
        )
        output, updated_mamba_cache = self.mamba(x, cache=mamba_cache)

        # Update cache if needed
        if (
            cache is not None
            and updated_mamba_cache is not None
            and self.layer_idx is not None
        ):
            cache = cache.update_mamba(self.layer_idx, updated_mamba_cache)

        return output, cache

    def step(self, x: jax.Array, cache: CacheState) -> tuple[jax.Array, CacheState]:
        """Single step for generation."""
        # Get or create Mamba cache for this layer
        mamba_cache = (
            cache.get_mamba(self.layer_idx) if self.layer_idx is not None else None
        )

        if mamba_cache is None and self.layer_idx is not None:
            # Initialize cache if not present
            batch_size = x.shape[0]
            mamba_cache = create_mamba2_cache(
                batch_size,
                self.mamba.mamba.d_inner,
                self.mamba.mamba.d_state,
                self.mamba.mamba.d_conv,
                self.mamba.mamba.nheads,
                self.mamba.mamba.headdim,
            )
            cache = cache.update_mamba(self.layer_idx, mamba_cache)

        # Call Mamba2Block with cache
        output, updated_cache = self.mamba(x, cache=mamba_cache)

        # Update cache if needed
        if updated_cache is not None and self.layer_idx is not None:
            cache = cache.update_mamba(self.layer_idx, updated_cache)

        return output, cache
