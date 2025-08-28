"""Normalization layers."""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from flax.nnx import Param


class RMSNorm(nnx.Module):
    """Root Mean Square LayerNorm with optional residual merging.

    Matches the behavior used in the original single-file implementation.
    """

    def __init__(
        self, d: int, eps: float = 1e-5, dtype=jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0)
    ):
        self.eps = eps
        self.weight = Param(jnp.ones((d,), dtype=dtype))

    def __call__(
        self,
        x: jnp.ndarray,
        residual: jnp.ndarray | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ):
        if residual is not None:
            if residual_in_fp32:
                residual = residual.astype(jnp.float32)
            x = (x + residual).astype(x.dtype)

        if prenorm:
            normed = (
                x
                * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
                * self.weight.value
            )
            return normed, x
        else:
            return (
                x
                * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
                * self.weight.value
            )
