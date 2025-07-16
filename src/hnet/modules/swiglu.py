# Copyright (c) 2025, Nick Wall.
# JAX implementation of SwiGLU (Swish-Gated Linear Unit)
# Based on the original implementation from state-spaces/mamba

import flax.nnx as nnx
import jax
import jax.numpy as jnp


def swiglu(gate: jax.Array, x: jax.Array) -> jax.Array:
    """
    Apply SwiGLU activation function.

    SwiGLU(gate, x) = Swish(gate) * x = (gate * sigmoid(gate)) * x

    Args:
        gate: Gate tensor
        x: Input tensor

    Returns:
        Activated tensor
    """
    return jax.nn.silu(gate) * x  # silu is the same as swish


class SwiGLU(nnx.Module):
    """
    SwiGLU (Swish-Gated Linear Unit) module.

    Applies two linear transformations with a gated activation in between.
    """

    def __init__(
        self,
        d_model: int,
        d_intermediate: int | None = None,
        bias: bool = False,
        multiple_of: int = 128,
        dtype: jnp.dtype | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize SwiGLU module.

        Args:
            d_model: Input and output dimension
            d_intermediate: Intermediate dimension (defaults to 8/3 * d_model)
            bias: Whether to use bias in linear layers
            multiple_of: Round intermediate dimension to multiple of this value
            dtype: Data type for parameters
            rngs: Random number generators for initialization
        """
        # Calculate intermediate dimension
        if d_intermediate is None:
            d_intermediate = int(8 * d_model / 3)

        # Round to multiple_of
        d_intermediate = (d_intermediate + multiple_of - 1) // multiple_of * multiple_of

        # Initialize linear layers
        self.fc1 = nnx.Linear(
            d_model, 2 * d_intermediate, use_bias=bias, dtype=dtype, rngs=rngs
        )
        self.fc2 = nnx.Linear(
            d_intermediate, d_model, use_bias=bias, dtype=dtype, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply SwiGLU transformation.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Output tensor of shape (..., d_model)
        """
        # First linear transformation to 2 * d_intermediate
        y = self.fc1(x)

        # Split into gate and value
        y, gate = jnp.split(y, 2, axis=-1)

        # Apply SwiGLU activation
        y = swiglu(gate, y)

        # Second linear transformation back to d_model
        y = self.fc2(y)

        return y
