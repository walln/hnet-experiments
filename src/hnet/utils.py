"""Utility helpers used across the H-Net implementation."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import jax
import jax.numpy as jnp


def get_seq_idx(cu_seqlens: jnp.ndarray) -> jnp.ndarray:
    """Return sequence indices for packed representation.

    Args:
        cu_seqlens: (B+1,) cumulative sequence lengths.

    Returns:
        (1, T) int tensor mapping each token to its batch index.
    """

    seq_idx = jnp.zeros(cu_seqlens[-1], dtype=jnp.int32)
    seq_idx = seq_idx.at[cu_seqlens[:-1]].set(1)
    seq_idx = (jnp.cumsum(seq_idx) - 1)[None, :]
    return seq_idx


def get_stage_cfg(cfg: Any, stage_idx: int):
    """Extract the stage-specific dict view from a dataclass config instance."""

    return {
        k: (v[stage_idx] if isinstance(v, list) else v) for k, v in asdict(cfg).items()
    }


def ste_func(x):
    """Straight-through estimator helper compatible with JAX/NNX modules."""

    return jax.lax.stop_gradient(jnp.ones_like(x)) + (x - jax.lax.stop_gradient(x))
