"""Causal Multi-Head Attention without FlashAttention kernels."""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class CausalMHA(nnx.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        qkv_proj_bias: bool = False,
        out_proj_bias: bool = False,
        rotary_emb_dim: int = 0,
        window_size: int = -1,
        dtype=jnp.float32,
        layer_idx: int | None = None,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.layer_idx = layer_idx
        self.rotary_emb_dim = int(rotary_emb_dim) if rotary_emb_dim is not None else 0
        self.rotary_base = 10000.0
        self.window_size = int(window_size) if window_size is not None else -1

        self.Wqkv = nnx.Linear(
            d_model, 3 * d_model, use_bias=qkv_proj_bias, dtype=dtype, rngs=rngs
        )
        self.out_proj = nnx.Linear(
            d_model, d_model, use_bias=out_proj_bias, dtype=dtype, rngs=rngs
        )

    def _split_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape(*x.shape[:-1], self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

    def _merge_heads(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.transpose(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], self.d_model)

    def _causal_attn(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        attn_mask: jnp.ndarray | None,
    ) -> jnp.ndarray:
        # q, k, v: (B, H, L, D)
        scale = self.head_dim**-0.5
        attn_scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # (B, H, L, L)
        L = q.shape[-2]
        causal = jnp.triu(jnp.ones((L, L), dtype=jnp.bool_), k=1)
        attn_scores = jnp.where(causal, -jnp.inf, attn_scores)

        if self.window_size is not None and self.window_size > 0:
            idx = jnp.arange(L)
            win_mask = idx[None, :] < (idx[:, None] - (self.window_size - 1))
            attn_scores = jnp.where(win_mask, -jnp.inf, attn_scores)

        if attn_mask is not None:
            mask = (
                attn_mask[:, None, None, :]
                .repeat(self.num_heads, axis=1)
                .repeat(L, axis=2)
            )
            attn_scores = jnp.where(~mask, -jnp.inf, attn_scores)

        attn = jax.nn.softmax(attn_scores, axis=-1)
        return jnp.matmul(attn, v)

    def _rotary_cos_sin(self, L: int, offset: int, dtype):
        """Compute rotary cos/sin tables for sequence length L starting at offset."""

        ro_dim = self.rotary_emb_dim
        assert ro_dim % 2 == 0 and ro_dim <= self.head_dim
        inv_freq = 1.0 / (
            self.rotary_base ** (jnp.arange(0, ro_dim, 2, dtype=jnp.float32) / ro_dim)
        )
        t = jnp.arange(offset, offset + L, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)  # (L, ro_dim/2)
        cos = jnp.cos(freqs).astype(dtype)
        sin = jnp.sin(freqs).astype(dtype)
        return cos, sin

    def _apply_rotary(self, q: jnp.ndarray, k: jnp.ndarray, offset: int = 0):
        if self.rotary_emb_dim <= 0:
            return q, k
        B, H, L, Dh = q.shape
        ro_dim = self.rotary_emb_dim
        cos, sin = self._rotary_cos_sin(L, offset, q.dtype)
        cos = cos.reshape(1, 1, L, -1)
        sin = sin.reshape(1, 1, L, -1)

        def rotate_half(x):
            x1, x2 = jnp.split(x, 2, axis=-1)
            xr1 = x1 * cos - x2 * sin
            xr2 = x2 * cos + x1 * sin
            return jnp.concatenate([xr1, xr2], axis=-1)

        def apply(x):
            x_ro = x[..., :ro_dim]
            x_rest = x[..., ro_dim:]
            x_ro_new = rotate_half(x_ro)
            return jnp.concatenate([x_ro_new, x_rest], axis=-1)

        return apply(q), apply(k)

    def __call__(
        self,
        x: jnp.ndarray,
        cu_seqlens: jnp.ndarray | None = None,
        max_seqlen: int | None = None,
        attn_mask: jnp.ndarray | None = None,
        inference_params=None,
    ) -> jnp.ndarray:
        packed = cu_seqlens is not None and max_seqlen is not None
        if packed:
            assert cu_seqlens is not None and max_seqlen is not None
            T, D = x.shape
            B = len(cu_seqlens) - 1
            Lmax = int(max_seqlen)
            padded = jnp.zeros((B, Lmax, D), dtype=x.dtype)
            mask = jnp.zeros((B, Lmax), dtype=jnp.bool_)

            for b in range(B):
                s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
                L = e - s
                padded = padded.at[b, :L].set(x[s:e])
                mask = mask.at[b, :L].set(True)

            out = self._forward_padded(padded, attn_mask=mask, inference_params=None)
            parts = []
            for b in range(B):
                L = int(jnp.sum(mask[b]))
                parts.append(out[b, :L])
            return jnp.concatenate(parts, axis=0)
        else:
            assert x.ndim == 3 and attn_mask is not None, (
                "Provide attn_mask for padded input"
            )
            return self._forward_padded(
                x, attn_mask=attn_mask, inference_params=inference_params
            )

    def _forward_padded(
        self, x: jnp.ndarray, attn_mask: jnp.ndarray | None, inference_params=None
    ) -> jnp.ndarray:
        B, L, _ = x.shape
        x_dtype = x.dtype
        w_dtype = self.Wqkv.kernel.value.dtype
        if x_dtype != w_dtype:
            x = x.astype(w_dtype)

        qkv = self.Wqkv(x)  # (B, L, 3D)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = self._split_heads(q)  # (B, H, L, Dh)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.rotary_emb_dim > 0:
            offset = int(getattr(inference_params, "seqlen_offset", 0) or 0)
            q, k = self._apply_rotary(q, k, offset=offset)

        ctx = self._causal_attn(q, k, v, attn_mask)  # (B, H, L, Dh)
        ctx = self._merge_heads(ctx)  # (B, L, D)
        out = self.out_proj(ctx)

        if inference_params is not None:
            assert getattr(self, "layer_idx", None) is not None, (
                "layer_idx required for KV cache"
            )
            kv_cache = inference_params.key_value_memory_dict.get(self.layer_idx)
            if kv_cache is None:
                dtype = self.out_proj.kernel.value.dtype
                kv_cache = jnp.zeros(
                    (
                        inference_params.max_batch_size,
                        inference_params.max_seqlen,
                        2,
                        self.num_heads,
                        self.head_dim,
                    ),
                    dtype=dtype,
                )
                inference_params.key_value_memory_dict[self.layer_idx] = kv_cache

            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + B
            seq_start = inference_params.seqlen_offset
            seq_end = seq_start + L
            kv_cur = jnp.stack(
                [k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)], axis=2
            )  # (B, L, 2, H, Dh)
            kv_cache = kv_cache.at[batch_start:batch_end, seq_start:seq_end, ...].set(
                kv_cur
            )
            inference_params.key_value_memory_dict[self.layer_idx] = kv_cache

        return out.astype(x_dtype)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        return None

    def step(self, x: jnp.ndarray, inference_params=None) -> jnp.ndarray:
        assert x.ndim == 3 and x.shape[1] == 1
        B, L, _ = x.shape
        x_dtype = x.dtype
        w_dtype = self.Wqkv.kernel.value.dtype
        if x_dtype != w_dtype:
            x = x.astype(w_dtype)

        qkv = self.Wqkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.rotary_emb_dim > 0:
            offset = int(getattr(inference_params, "seqlen_offset", 0) or 0)
            q, k = self._apply_rotary(q, k, offset=offset)

        assert getattr(self, "layer_idx", None) is not None, (
            "layer_idx required for KV cache"
        )
        kv_cache = inference_params.key_value_memory_dict.get(self.layer_idx)
        if kv_cache is None:
            dtype = self.out_proj.kernel.value.dtype
            kv_cache = jnp.zeros(
                (
                    inference_params.max_batch_size,
                    inference_params.max_seqlen,
                    2,
                    self.num_heads,
                    self.head_dim,
                ),
                dtype=dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = kv_cache

        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + B
        seq_start = inference_params.seqlen_offset
        seq_end = seq_start + 1

        kv_cache = kv_cache.at[batch_start:batch_end, seq_start:seq_end, 0, ...].set(
            k.transpose(0, 2, 1, 3)
        )
        kv_cache = kv_cache.at[batch_start:batch_end, seq_start:seq_end, 1, ...].set(
            v.transpose(0, 2, 1, 3)
        )
        inference_params.key_value_memory_dict[self.layer_idx] = kv_cache

        start = (
            max(0, seq_end - self.window_size)
            if (self.window_size is not None and self.window_size > 0)
            else 0
        )
        K_all = kv_cache[batch_start:batch_end, start:seq_end, 0, ...].transpose(
            0, 2, 1, 3
        )
        V_all = kv_cache[batch_start:batch_end, start:seq_end, 1, ...].transpose(
            0, 2, 1, 3
        )

        scale = self.head_dim**-0.5
        attn_scores = jnp.matmul(q, K_all.transpose(0, 1, 3, 2)) * scale
        attn = jax.nn.softmax(attn_scores, axis=-1)
        ctx = jnp.matmul(attn, V_all)
        ctx = self._merge_heads(ctx)
        out = self.out_proj(ctx)
        return out.astype(x_dtype)
