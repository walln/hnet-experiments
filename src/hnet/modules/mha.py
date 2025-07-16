# Copyright (c) 2025, Nick Wall.
# JAX implementation of multi-head attention based on the Tri Dao PyTorch implementation.
# Copyright (c) 2023, Tri Dao.

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from einops import rearrange

from .rotary import RotaryEmbedding


def causal_mask(
    seq_len: int, seq_len_k: int | None = None, dtype=jnp.float32
) -> jax.Array:
    """Create a causal mask for self-attention."""
    if seq_len_k is None:
        seq_len_k = seq_len
    # Create a mask where positions can only attend to earlier positions
    row_indices = jnp.arange(seq_len)[:, None]
    col_indices = jnp.arange(seq_len_k)[None, :]
    mask = (row_indices >= col_indices).astype(dtype)
    return mask


def sliding_window_mask(
    seq_len: int, window_size: int, seq_len_k: int | None = None, dtype=jnp.float32
) -> jax.Array:
    """Create a sliding window attention mask."""
    if seq_len_k is None:
        seq_len_k = seq_len

    if window_size < 0:
        return causal_mask(seq_len, seq_len_k, dtype)

    mask = jnp.zeros((seq_len, seq_len_k), dtype=dtype)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        end = min(seq_len_k, i + 1)
        mask = mask.at[i, start:end].set(1.0)
    return mask


def scaled_dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array | None = None,
    softmax_scale: float | None = None,
    causal: bool = True,
    window_size: int = -1,
) -> jax.Array:
    """
    Scaled dot-product attention.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch, seq_len_k, num_heads, head_dim)
        v: Value tensor of shape (batch, seq_len_k, num_heads, head_dim)
        mask: Optional attention mask
        softmax_scale: Scaling factor for attention scores
        causal: Whether to use causal masking
        window_size: Sliding window size (-1 for global attention)

    Returns:
        Attention output of shape (batch, seq_len, num_heads, head_dim)
    """
    batch, seq_len, num_heads, head_dim = q.shape
    seq_len_k = k.shape[1]

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    # Compute attention scores
    # Reshape for batched matrix multiply: (batch * num_heads, seq_len, head_dim)
    q = rearrange(q, "b s h d -> (b h) s d")
    k = rearrange(k, "b s h d -> (b h) s d")
    v = rearrange(v, "b s h d -> (b h) s d")

    # Compute scores: (batch * num_heads, seq_len, seq_len_k)
    scores = jnp.matmul(q, k.swapaxes(-2, -1)) * softmax_scale

    # Apply mask if needed
    if causal or window_size > 0:
        if window_size > 0:
            attn_mask = sliding_window_mask(seq_len, window_size, seq_len_k)
        else:
            attn_mask = causal_mask(seq_len, seq_len_k)

        # Apply mask directly to the (batch * num_heads, seq_len, seq_len_k) scores
        # The mask is broadcasted across the batch dimension
        scores = jnp.where(attn_mask[None, :, :], scores, -1e9)

    if mask is not None:
        scores = scores + mask

    # Apply softmax
    attn_weights = jax.nn.softmax(scores, axis=-1)

    # Apply attention to values
    output = jnp.matmul(attn_weights, v)

    # Reshape back
    output = rearrange(output, "(b h) s d -> b s h d", b=batch, h=num_heads)

    return output


class CausalSelfAttention(nnx.Module):
    """JAX implementation of scaled dot product attention with softmax."""

    def __init__(
        self,
        softmax_scale: float | None = None,
        window_size: tuple[int, int] = (-1, -1),
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize causal self-attention.

        Args:
            softmax_scale: The temperature to use for the softmax attention.
                         (default: 1/sqrt(d_keys) where d_keys is computed at runtime)
            window_size: Sliding window size for local attention
            rngs: Random number generators (required by nnx)
        """
        self.softmax_scale = softmax_scale
        self.window_size = window_size[0]  # Use first element for self-attention

    def __call__(
        self,
        qkv: jax.Array,
        cu_seqlens: jax.Array | None = None,
        max_seqlen: int | None = None,
    ) -> jax.Array:
        """
        Implements the multihead softmax attention.

        Args:
            qkv: The tensor containing the query, key, and value.
                If cu_seqlens is None, then qkv has shape (B, S, 3, H, D).
                Packed sequences not yet supported in JAX version.
            cu_seqlens: Not yet supported
            max_seqlen: Not yet supported

        Returns:
            out: (B, S, H, D)
        """
        if cu_seqlens is not None:
            raise NotImplementedError(
                "Packed sequence support not yet implemented in JAX version"
            )

        assert qkv.ndim == 5, f"Expected qkv to have 5 dimensions, got {qkv.ndim}"
        batch, seq_len, three, num_heads, head_dim = qkv.shape
        assert three == 3, f"Expected 3 QKV components, got {three}"

        q = qkv[:, :, 0]  # (B, S, H, D)
        k = qkv[:, :, 1]  # (B, S, H, D)
        v = qkv[:, :, 2]  # (B, S, H, D)

        return scaled_dot_product_attention(
            q,
            k,
            v,
            softmax_scale=self.softmax_scale,
            causal=True,
            window_size=self.window_size,
        )


class CausalCrossAttention(nnx.Module):
    """JAX implementation of scaled dot product cross-attention."""

    def __init__(
        self,
        softmax_scale: float | None = None,
        window_size: tuple[int, int] = (-1, -1),
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize causal cross-attention.

        Args:
            softmax_scale: The temperature to use for the softmax attention.
            window_size: Sliding window size for local attention
            rngs: Random number generators (required by nnx)
        """
        self.softmax_scale = softmax_scale
        self.window_size = window_size[0]

    def __call__(
        self,
        q: jax.Array,
        kv: jax.Array,
        cu_seqlens: jax.Array | None = None,
        max_seqlen: int | None = None,
        cu_seqlens_k: jax.Array | None = None,
        max_seqlen_k: int | None = None,
    ) -> jax.Array:
        """
        Implements the multihead softmax cross-attention.

        Args:
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
            cu_seqlens: Not yet supported
            max_seqlen: Not yet supported
            cu_seqlens_k: Not yet supported
            max_seqlen_k: Not yet supported

        Returns:
            Attention output of shape (B, Sq, H, D)
        """
        if cu_seqlens is not None:
            raise NotImplementedError(
                "Packed sequence support not yet implemented in JAX version"
            )

        assert q.ndim == 4, f"Expected q to have 4 dimensions, got {q.ndim}"
        assert kv.ndim == 5, f"Expected kv to have 5 dimensions, got {kv.ndim}"
        assert kv.shape[2] == 2, f"Expected 2 KV components, got {kv.shape[2]}"

        k = kv[:, :, 0]  # (B, Sk, H_k, D)
        v = kv[:, :, 1]  # (B, Sk, H_k, D)

        return scaled_dot_product_attention(
            q,
            k,
            v,
            softmax_scale=self.softmax_scale,
            causal=True,
            window_size=self.window_size,
        )


class InferenceParams:
    """Parameters for generation/inference mode with KV caching."""

    def __init__(
        self,
        max_batch_size: int,
        max_seqlen: int,
        seqlen_offset: int = 0,
        batch_size_offset: int = 0,
        lengths_per_sample: jax.Array | None = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_seqlen = max_seqlen
        self.seqlen_offset = seqlen_offset
        self.batch_size_offset = batch_size_offset
        self.lengths_per_sample = lengths_per_sample
        # In JAX, we'll handle KV cache differently than PyTorch
        # Instead of a mutable dict, we'll use explicit state management
        self.key_value_memory_dict = {}


class CausalMHA(nnx.Module):
    """Causal Multi-Head Attention module in JAX."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        qkv_proj_bias: bool = False,
        out_proj_bias: bool = False,
        window_size: int = -1,
        softmax_scale: float | None = None,
        layer_idx: int | None = None,
        rotary_emb_dim: int = 0,
        rotary_emb_base: float = 10000.0,
        rotary_emb_interleaved: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize Causal MHA.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            qkv_proj_bias: Whether to use bias in QKV projection
            out_proj_bias: Whether to use bias in output projection
            window_size: Sliding window size (-1 for global attention)
            softmax_scale: Attention scaling factor
            layer_idx: Layer index (for KV caching)
            rotary_emb_dim: Dimension of rotary embeddings (0 to disable)
            rotary_emb_base: Base for rotary embeddings
            rotary_emb_interleaved: Whether to use interleaved rotary
            rngs: Random number generators
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.layer_idx = layer_idx
        self.softmax_scale = softmax_scale
        self.rotary_emb_dim = rotary_emb_dim

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        qkv_dim = self.head_dim * (3 * self.num_heads)

        # Initialize rotary embeddings if needed
        if self.rotary_emb_dim > 0:
            self.rotary_emb = RotaryEmbedding(
                dim=rotary_emb_dim,
                base=rotary_emb_base,
                interleaved=rotary_emb_interleaved,
                rngs=rngs,
            )
        else:
            self.rotary_emb = None

        # Initialize linear layers
        self.Wqkv = nnx.Linear(d_model, qkv_dim, use_bias=qkv_proj_bias, rngs=rngs)

        # Initialize attention modules
        self.inner_attn = CausalSelfAttention(
            softmax_scale=softmax_scale,
            window_size=(window_size, -1),
            rngs=rngs,
        )
        self.inner_cross_attn = CausalCrossAttention(
            softmax_scale=softmax_scale,
            window_size=(window_size, -1),
            rngs=rngs,
        )

        self.out_proj = nnx.Linear(d_model, d_model, use_bias=out_proj_bias, rngs=rngs)

        # For KV caching - initialize with empty array instead of None
        self._kv_cache = nnx.Variable(jnp.zeros((0, 0, 0, 0, 0), dtype=jnp.float32))

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype=None
    ) -> jax.Array:
        """Allocate KV cache for inference."""
        if dtype is None:
            dtype = self.out_proj.kernel.value.dtype

        cache = jnp.zeros(
            (batch_size, max_seqlen, 2, self.num_heads, self.head_dim),
            dtype=dtype,
        )
        self._kv_cache.value = cache
        return cache

    def _update_kv_cache(
        self, kv: jax.Array, cache_position: int
    ) -> tuple[jax.Array, jax.Array]:
        """
        Update KV cache and return updated cache.

        Args:
            kv: New KV values of shape (batch, seqlen, 2, nheads, head_dim)
            cache_position: Position in cache to update

        Returns:
            Updated KV cache and the cache up to current position
        """
        if self._kv_cache.value.size == 0:
            raise ValueError(
                "KV cache not allocated. Call allocate_inference_cache first."
            )

        batch_size, seqlen, _, _, _ = kv.shape

        # Update cache
        new_cache = self._kv_cache.value.at[
            :batch_size, cache_position : cache_position + seqlen
        ].set(kv)
        self._kv_cache.value = new_cache

        # Return cache up to current position
        return new_cache, new_cache[:batch_size, : cache_position + seqlen]

    def __call__(
        self,
        x: jax.Array,
        cu_seqlens: jax.Array | None = None,
        max_seqlen: int | None = None,
        inference_params: InferenceParams | None = None,
        **kwargs,
    ) -> jax.Array:
        """
        Forward pass of Causal MHA.

        Args:
            x: Input tensor of shape (batch, seqlen, d_model)
            cu_seqlens: Not yet supported
            max_seqlen: Not yet supported
            inference_params: Parameters for inference mode with KV caching

        Returns:
            Output tensor of shape (batch, seqlen, d_model)
        """
        if cu_seqlens is not None:
            raise NotImplementedError(
                "Packed sequence support not yet implemented in JAX version"
            )

        batch_size, seqlen, _ = x.shape

        # Project to QKV
        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, d=self.head_dim
        )

        # Apply rotary embeddings if enabled
        if self.rotary_emb is not None:
            if inference_params is not None:
                seqlen_offset = inference_params.seqlen_offset
            else:
                seqlen_offset = 0
            qkv = self.rotary_emb(qkv, seqlen_offset=seqlen_offset)

        # Handle inference mode with KV caching
        if inference_params is not None and inference_params.seqlen_offset > 0:
            # We're in generation mode
            assert isinstance(qkv, jax.Array), "qkv must be a JAX array"
            q = qkv[:, :, 0]  # (batch, seqlen, nheads, head_dim)
            kv_new = qkv[:, :, 1:]  # (batch, seqlen, 2, nheads, head_dim)

            # Update KV cache
            _, kv_cache = self._update_kv_cache(kv_new, inference_params.seqlen_offset)

            # Cross attention with cached KV
            context = self.inner_cross_attn(q, kv_cache)
        else:
            # Standard self-attention
            assert isinstance(qkv, jax.Array), "qkv must be a JAX array"
            context = self.inner_attn(qkv)

        # Project output
        assert isinstance(context, jax.Array), "context must be a JAX array"
        out = self.out_proj(rearrange(context, "b s h d -> b s (h d)"))
        return out

    def step(self, x: jax.Array, inference_params: InferenceParams) -> jax.Array:
        """Single step for autoregressive generation."""
        return self(x, inference_params=inference_params)
