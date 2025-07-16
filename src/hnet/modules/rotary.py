# Copyright (c) 2025, Nick Wall.
# Reimplementation of rotary positional encoding based on the Tri Dao implementation.
# Copyright (c) 2023, Tri Dao.

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from einops import rearrange, repeat


def rotate_half(x: jax.Array, interleaved: bool = False) -> jax.Array:
    """Rotate half the hidden dims of the input."""
    if not interleaved:
        x1, x2 = jnp.split(x, 2, axis=-1)
        return jnp.concatenate([-x2, x1], axis=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = jnp.stack([-x2, x1], axis=-1)
        return rearrange(rotated, "... d two -> ... (d two)", two=2)


def apply_rotary_emb_jax(
    x: jax.Array, cos: jax.Array, sin: jax.Array, interleaved: bool = False
) -> jax.Array:
    """
    Apply rotary embeddings to input tensors.

    Args:
        x: (batch_size, seqlen, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style)

    Returns:
        Tensor with rotary embeddings applied
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1], f"Rotary dim {ro_dim} exceeds head dim {x.shape[-1]}"

    # Repeat cos and sin to match x's shape
    cos = repeat(
        cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )
    sin = repeat(
        sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)"
    )

    # Apply rotary embedding
    x_rot = x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin

    # Concatenate rotated and non-rotated parts
    return jnp.concatenate([x_rot, x[..., ro_dim:]], axis=-1)


def apply_rotary_emb(
    x: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    interleaved: bool = False,
    seqlen_offsets: int | jax.Array = 0,
    cu_seqlens: jax.Array | None = None,
    max_seqlen: int | None = None,
) -> jax.Array:
    """
    Apply rotary embeddings with support for offsets and variable sequence lengths.

    Args:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
           else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions
        seqlen_offsets: (batch_size,) or int. Each sequence is shifted by this amount.
        cu_seqlens: (batch + 1,) cumulative sequence lengths for packed sequences
        max_seqlen: maximum sequence length when using packed sequences

    Returns:
        Tensor with rotary embeddings applied
    """
    if cu_seqlens is not None:
        # Handle packed sequences - this would require custom implementation
        # For now, we'll raise NotImplementedError
        raise NotImplementedError(
            "Packed sequence support not yet implemented in JAX version"
        )

    # Get the sequence length from input
    seqlen = x.shape[1] if cu_seqlens is None else max_seqlen

    # Handle sequence offsets and ensure cos/sin match sequence length
    if isinstance(seqlen_offsets, int):
        # Slice cos and sin to get the positions we need
        cos = cos[seqlen_offsets : seqlen_offsets + seqlen]
        sin = sin[seqlen_offsets : seqlen_offsets + seqlen]
    else:
        # Handle per-sequence offsets
        # This would require more complex indexing
        raise NotImplementedError(
            "Per-sequence offsets not yet implemented in JAX version"
        )

    # If cos/sin are longer than needed (e.g., from cache), slice to match
    if cos.shape[0] > seqlen:
        cos = cos[:seqlen]
        sin = sin[:seqlen]

    return apply_rotary_emb_jax(x, cos, sin, interleaved)


def apply_rotary_emb_qkv(
    qkv: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    cos_k: jax.Array | None = None,
    sin_k: jax.Array | None = None,
    interleaved: bool = False,
    seqlen_offsets: int | jax.Array = 0,
    num_heads_q: int | None = None,
) -> jax.Array:
    """
    Apply rotary embeddings to Q, K, V tensors.

    Args:
        qkv: (batch_size, seqlen, 3, nheads, headdim) or
             (batch_size, seqlen, num_heads_q + 2 * num_heads_k, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        cos_k, sin_k: (seqlen, rotary_dim / 2), optional separate embeddings for K
        interleaved: if True, rotate pairs of even and odd dimensions
        seqlen_offsets: sequence offsets for KV cache
        num_heads_q: number of query heads (for MQA/GQA)

    Returns:
        QKV tensor with rotary embeddings applied to Q and K
    """
    if cos_k is None:
        cos_k = cos
    if sin_k is None:
        sin_k = sin

    if qkv.ndim == 5:
        # Standard shape: (batch, seqlen, 3, nheads, headdim)
        batch, seqlen, three, nheads, headdim = qkv.shape
        assert three == 3

        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        # Apply rotary to Q and K
        q_rot = apply_rotary_emb(q, cos, sin, interleaved, seqlen_offsets)
        k_rot = apply_rotary_emb(k, cos_k, sin_k, interleaved, seqlen_offsets)

        # Stack back together
        qkv_rot = jnp.stack([q_rot, k_rot, v], axis=2)

    else:
        # MQA/GQA shape: (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim)
        assert qkv.ndim == 4
        assert num_heads_q is not None

        batch, seqlen, total_heads, headdim = qkv.shape
        num_heads_k = (total_heads - num_heads_q) // 2
        assert total_heads == num_heads_q + 2 * num_heads_k

        q = qkv[:, :, :num_heads_q]
        k = qkv[:, :, num_heads_q : num_heads_q + num_heads_k]
        v = qkv[:, :, num_heads_q + num_heads_k :]

        # Apply rotary to Q and K
        q_rot = apply_rotary_emb(q, cos, sin, interleaved, seqlen_offsets)
        k_rot = apply_rotary_emb(k, cos_k, sin_k, interleaved, seqlen_offsets)

        # Concatenate back
        qkv_rot = jnp.concatenate([q_rot, k_rot, v], axis=2)

    return qkv_rot


def apply_rotary_emb_kv(
    kv: jax.Array,
    cos: jax.Array,
    sin: jax.Array,
    interleaved: bool = False,
    seqlen_offsets: int | jax.Array = 0,
) -> jax.Array:
    """
    Apply rotary embeddings to K in KV tensor.

    Args:
        kv: (batch_size, seqlen, 2, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions
        seqlen_offsets: sequence offsets for KV cache

    Returns:
        KV tensor with rotary embeddings applied to K
    """
    batch, seqlen, two, nheads, headdim = kv.shape
    assert two == 2

    k = kv[:, :, 0]
    v = kv[:, :, 1]

    # Apply rotary to K only
    k_rot = apply_rotary_emb(k, cos, sin, interleaved, seqlen_offsets)

    # Stack back together
    kv_rot = jnp.stack([k_rot, v], axis=2)

    return kv_rot


class RotaryEmbedding(nnx.Module):
    """
    Rotary position embeddings from RoFormer (Su et. al).

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: float | None = None,
        pos_idx_in_fp32: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize rotary embeddings.

        Args:
            dim: dimension of the embeddings (must be even)
            base: base for the frequency computation
            interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style)
            scale_base: if not None, implement XPos
            pos_idx_in_fp32: if True, compute position indices in fp32 for precision
            rngs: random number generators (required by nnx but not used here)
        """
        self.dim = dim
        self.base = float(base)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.pos_idx_in_fp32 = pos_idx_in_fp32

        # Compute inverse frequencies
        self.inv_freq = self._compute_inv_freq()

        # Compute scale if using XPos
        if scale_base is not None:
            scale = (jnp.arange(0, dim, 2, dtype=jnp.float32) + 0.4 * dim) / (1.4 * dim)
            self.scale = nnx.Variable(scale)
        else:
            self.scale = None

        # Cache for precomputed cos/sin values
        self._seq_len_cached = nnx.Variable(0)
        self._cos_cached = nnx.Variable(jnp.array([]))
        self._sin_cached = nnx.Variable(jnp.array([]))
        self._cos_k_cached = nnx.Variable(jnp.array([]))
        self._sin_k_cached = nnx.Variable(jnp.array([]))

    def _compute_inv_freq(self) -> jax.Array:
        """Compute inverse frequencies for rotary embeddings."""
        return 1.0 / (
            self.base ** (jnp.arange(0, self.dim, 2, dtype=jnp.float32) / self.dim)
        )

    def _update_cos_sin_cache(self, seqlen: int, dtype=None) -> None:
        """Update the cached cos/sin values if needed."""
        if dtype is None:
            dtype = jnp.float32

        # Check if we need to recompute
        if (
            seqlen > self._seq_len_cached.value
            or self._cos_cached.value.size == 0
            or self._cos_cached.value.dtype != dtype
        ):
            self._seq_len_cached.value = seqlen

            # Compute position indices
            if self.pos_idx_in_fp32:
                t = jnp.arange(seqlen, dtype=jnp.float32)
                inv_freq = self.inv_freq.astype(jnp.float32)
            else:
                t = jnp.arange(seqlen, dtype=dtype)
                inv_freq = self.inv_freq.astype(dtype)

            # Compute frequencies
            freqs = jnp.outer(t, inv_freq)

            if self.scale is None:
                # Standard rotary embeddings
                self._cos_cached.value = jnp.cos(freqs).astype(dtype)
                self._sin_cached.value = jnp.sin(freqs).astype(dtype)
            else:
                # XPos embeddings with scaling
                power = (t - seqlen // 2) / self.scale_base
                scale = self.scale.value ** power[:, None]

                self._cos_cached.value = (jnp.cos(freqs) * scale).astype(dtype)
                self._sin_cached.value = (jnp.sin(freqs) * scale).astype(dtype)
                self._cos_k_cached.value = (jnp.cos(freqs) / scale).astype(dtype)
                self._sin_k_cached.value = (jnp.sin(freqs) / scale).astype(dtype)

    def __call__(
        self,
        qkv: jax.Array,
        kv: jax.Array | None = None,
        seqlen_offset: int | jax.Array = 0,
        cu_seqlens: jax.Array | None = None,
        max_seqlen: int | None = None,
        num_heads_q: int | None = None,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """
        Apply rotary embeddings to Q, K, V tensors.

        Args:
            qkv: (batch, seqlen, 3, nheads, headdim) or (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim)
                if kv is None, else just q of shape (batch, seqlen, nheads, headdim)
            kv: (batch, seqlen, 2, nheads, headdim), optional separate KV tensor
            seqlen_offset: offset for each sequence (for KV cache)
            cu_seqlens: cumulative sequence lengths (for packed sequences)
            max_seqlen: maximum sequence length
            num_heads_q: number of query heads (for MQA/GQA)

        Returns:
            Rotary-embedded tensors
        """
        if cu_seqlens is not None:
            raise NotImplementedError("Packed sequence support not yet implemented")

        # Get sequence length and update cache
        seqlen = qkv.shape[1]
        if max_seqlen is not None:
            self._update_cos_sin_cache(max_seqlen, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(seqlen + seqlen_offset, dtype=qkv.dtype)
        else:
            self._update_cos_sin_cache(seqlen, dtype=qkv.dtype)

        # Get cached cos/sin values
        cos = self._cos_cached.value
        sin = self._sin_cached.value
        cos_k = self._cos_k_cached.value if self.scale is not None else cos
        sin_k = self._sin_k_cached.value if self.scale is not None else sin

        if kv is None:
            # Apply to QKV tensor
            return apply_rotary_emb_qkv(
                qkv,
                cos,
                sin,
                cos_k,
                sin_k,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                num_heads_q=num_heads_q,
            )
        else:
            # Apply to separate Q and KV tensors
            q = apply_rotary_emb(
                qkv,
                cos,
                sin,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
            )
            kv = apply_rotary_emb_kv(
                kv,
                cos_k,
                sin_k,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
            )
            return q, kv


# Aliases for backward compatibility
apply_rotary_emb_func = apply_rotary_emb
