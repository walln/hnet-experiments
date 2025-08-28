"""Feed-forward modules."""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class SwiGLU(nnx.Module):
    """SwiGLU MLP block with two linear projections.

    Keeps numerical casting consistent with attention/mixer weights.
    """

    def __init__(
        self,
        d_model: int,
        d_intermediate: int,
        dtype=jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.fc1 = nnx.Linear(
            d_model, 2 * d_intermediate, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.fc2 = nnx.Linear(
            d_intermediate, d_model, use_bias=False, dtype=dtype, rngs=rngs
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_dtype = x.dtype
        w_dtype = self.fc1.kernel.value.dtype
        if x_dtype != w_dtype:
            x = x.astype(w_dtype)
        gate_up = self.fc1(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        out = self.fc2(up * jax.nn.silu(gate))
        return out.astype(x_dtype)
