"""Unified cache structures for JAX-idiomatic inference."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class Mamba2CacheState:
    """Cache state for Mamba2 layers."""

    conv_state: jax.Array  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: jax.Array  # (batch, nheads, headdim, d_state)


@dataclass
class AttentionCacheState:
    """Cache state for attention layers."""

    key_cache: jax.Array  # (batch, max_seq_len, num_heads, head_dim)
    value_cache: jax.Array  # (batch, max_seq_len, num_heads, head_dim)
    cached_len: int  # Current length of cached sequences


@dataclass
class CacheState:
    """Unified cache state for hybrid models.

    This is a JAX pytree that can contain caches for multiple layers.
    Each layer's cache is accessed by its layer index.
    """

    mamba_caches: dict[int, Mamba2CacheState]
    attention_caches: dict[int, AttentionCacheState]

    @staticmethod
    def empty() -> "CacheState":
        """Create an empty cache state."""
        return CacheState(mamba_caches={}, attention_caches={})

    def update_mamba(self, layer_idx: int, cache: Mamba2CacheState) -> "CacheState":
        """Return a new CacheState with updated Mamba cache for a specific layer."""
        new_mamba_caches = self.mamba_caches.copy()
        new_mamba_caches[layer_idx] = cache
        return CacheState(
            mamba_caches=new_mamba_caches,
            attention_caches=self.attention_caches,
        )

    def update_attention(
        self, layer_idx: int, cache: AttentionCacheState
    ) -> "CacheState":
        """Return a new CacheState with updated attention cache for a specific layer."""
        new_attention_caches = self.attention_caches.copy()
        new_attention_caches[layer_idx] = cache
        return CacheState(
            mamba_caches=self.mamba_caches,
            attention_caches=new_attention_caches,
        )

    def get_mamba(self, layer_idx: int) -> Mamba2CacheState | None:
        """Get Mamba cache for a specific layer."""
        return self.mamba_caches.get(layer_idx)

    def get_attention(self, layer_idx: int) -> AttentionCacheState | None:
        """Get attention cache for a specific layer."""
        return self.attention_caches.get(layer_idx)


# Register as JAX pytrees for proper handling in transformations
def _flatten_cache_state(x: CacheState) -> tuple[list, None]:
    return ([x.mamba_caches, x.attention_caches], None)


def _unflatten_cache_state(_: None, caches: list) -> CacheState:
    mamba_caches: dict[int, Mamba2CacheState] = caches[0]
    attention_caches: dict[int, AttentionCacheState] = caches[1]
    return CacheState(mamba_caches, attention_caches)


jax.tree_util.register_pytree_node(
    Mamba2CacheState,
    lambda x: ([x.conv_state, x.ssm_state], None),
    lambda _, arrays: Mamba2CacheState(arrays[0], arrays[1]),
)

jax.tree_util.register_pytree_node(
    AttentionCacheState,
    lambda x: ([x.key_cache, x.value_cache], x.cached_len),
    lambda cached_len, arrays: AttentionCacheState(
        arrays[0], arrays[1], cached_len=cached_len
    ),
)

jax.tree_util.register_pytree_node(
    CacheState,
    _flatten_cache_state,
    _unflatten_cache_state,
)


# Factory functions for creating caches
def create_mamba2_cache(
    batch_size: int,
    d_inner: int,
    d_state: int,
    d_conv: int,
    nheads: int,
    headdim: int,
    dtype: jnp.dtype = jnp.float32,
) -> Mamba2CacheState:
    """Create an initialized Mamba2 cache."""
    conv_state = jnp.zeros((batch_size, d_inner + 2 * d_state, d_conv), dtype=dtype)
    ssm_state = jnp.zeros((batch_size, nheads, headdim, d_state), dtype=dtype)
    return Mamba2CacheState(conv_state, ssm_state)


def create_attention_cache(
    batch_size: int,
    max_seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.float32,
) -> AttentionCacheState:
    """Create an initialized attention cache."""
    key_cache = jnp.zeros((batch_size, max_seq_len, num_heads, head_dim), dtype=dtype)
    value_cache = jnp.zeros((batch_size, max_seq_len, num_heads, head_dim), dtype=dtype)
    return AttentionCacheState(key_cache, value_cache, cached_len=0)
