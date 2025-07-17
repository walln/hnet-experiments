# Copyright (c) 2025, Nick Wall.
# JAX implementation of hybrid transformer blocks combining Mamba2 and attention


import flax.nnx as nnx
import jax

from hnet.modules.mamba2 import Mamba2Block
from hnet.modules.mha import CausalMHA, InferenceParams
from hnet.modules.swiglu import SwiGLU


class HybridBlock(nnx.Module):
    """
    A hybrid block that can use either Mamba2 or Attention as the sequence mixing layer.

    This follows the structure of modern hybrid architectures like Jamba.
    """

    mixer: Mamba2Block | CausalMHA

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_mamba: bool = True,
        d_state: int = 128,
        d_conv: int = 4,
        expand_factor: int = 2,
        mlp_expand: int = 4,
        layer_idx: int | None = None,
        norm_epsilon: float = 1e-5,
        window_size: int = -1,
        chunk_size: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize hybrid block.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads (if using attention)
            use_mamba: Whether to use Mamba2 (True) or Attention (False)
            d_state: SSM state dimension (for Mamba2)
            d_conv: Convolution kernel size (for Mamba2)
            expand_factor: Expansion factor for Mamba2 inner dimension
            mlp_expand: Expansion factor for MLP
            layer_idx: Layer index for caching
            norm_epsilon: Epsilon for layer normalization
            window_size: Window size for attention (-1 for global)
            chunk_size: Chunk size for Mamba2 SSD algorithm
            rngs: Random number generators
        """
        self.use_mamba = use_mamba
        self.layer_idx = layer_idx

        # Pre-norm for sequence mixing
        self.norm1 = nnx.RMSNorm(d_model, epsilon=norm_epsilon, rngs=rngs)

        # Sequence mixing layer
        if use_mamba:
            self.mixer = Mamba2Block(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand_factor,
                headdim=64,  # Default head dimension
                ngroups=1,  # Default number of groups
                layer_idx=layer_idx,
                norm_epsilon=norm_epsilon,
                chunk_size=chunk_size,
                rngs=rngs,
            )
        else:
            self.mixer = CausalMHA(
                d_model=d_model,
                num_heads=n_heads,
                window_size=window_size,
                layer_idx=layer_idx,
                rngs=rngs,
            )

        # Pre-norm for MLP
        self.norm2 = nnx.RMSNorm(d_model, epsilon=norm_epsilon, rngs=rngs)

        # MLP
        d_intermediate = d_model * mlp_expand
        self.mlp = SwiGLU(
            d_model=d_model,
            d_intermediate=d_intermediate,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        inference_params: dict | None = None,
    ) -> jax.Array:
        """
        Forward pass of hybrid block.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            inference_params: Optional params for inference mode

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Sequence mixing with residual
        residual = x
        x = self.norm1(x)

        # Use isinstance to help the type checker
        if isinstance(self.mixer, Mamba2Block):
            # Mamba2Block expects cache argument and returns a tuple
            x, _ = self.mixer(x, cache=None)
        elif isinstance(self.mixer, CausalMHA):
            # CausalMHA expects InferenceParams object
            # Convert dict to InferenceParams for attention
            ip = None
            if inference_params is not None and isinstance(inference_params, dict):
                # Create InferenceParams from dict
                ip = InferenceParams(
                    max_batch_size=x.shape[0],
                    max_seqlen=x.shape[1],
                )
                if "key_value_memory_dict" in inference_params:
                    ip.key_value_memory_dict = inference_params["key_value_memory_dict"]
            elif inference_params is not None:
                ip = inference_params

            # Call CausalMHA with proper parameters
            x = self.mixer(x, inference_params=ip)
        else:
            raise ValueError(f"Unknown mixer type: {type(self.mixer)}")

        x = residual + x

        # MLP with residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Mamba2Wrapper(nnx.Module):
    """
    Mamba2 wrapper class that has the same inference interface as the CausalMHA class.

    This provides compatibility with existing transformer code.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        layer_idx: int | None = None,
        chunk_size: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize Mamba2 wrapper."""
        self.layer_idx = layer_idx
        self.mamba = Mamba2Block(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            chunk_size=chunk_size,
            headdim=64,  # Default head dimension
            ngroups=1,  # Default number of groups
            layer_idx=layer_idx,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, inference_params: dict | None = None) -> jax.Array:
        """Forward pass through wrapped Mamba2Block."""
        # Mamba2Block returns tuple, extract just the output
        output, _ = self.mamba(x, cache=None)
        return output

    def step(self, x: jax.Array, inference_params: dict | None = None) -> jax.Array:
        """Single step for generation."""
        if inference_params is None:
            inference_params = {}

        # Initialize cache if not present
        if "cache" not in inference_params:
            from .mamba2 import InferenceCache

            batch_size = x.shape[0]
            inference_params["cache"] = InferenceCache.alloc(
                batch_size,
                self.mamba.mamba.d_inner,
                self.mamba.mamba.d_state,
                self.mamba.mamba.d_conv,
                self.mamba.mamba.nheads,
                self.mamba.mamba.headdim,
            )

        # Call Mamba2Block with cache
        output, updated_cache = self.mamba(x, cache=inference_params["cache"])

        # Update cache in params
        inference_params["cache"] = updated_cache

        return output
