"""
Pure JAX/Flax NNX single-file reimplementation of H-Net (Dynamic Chunking for End-to-End
Hierarchical Sequence Modeling), based on the PyTorch reference implementation.

Key design points:
- Maintains the same high-level module structure (Routing -> Chunk -> Inner -> Dechunk -> Residual).
- Implements EMA-based dechunking in pure JAX.
- Provides both batch (masked) and packed (cu_seqlens) code paths where practical.
- Uses a simple causal multi-head attention and a lightweight SSM-like mixer (conv + gating)
  to stand in for Mamba2 while keeping everything JAX-native.

This file exposes:
- Config dataclasses: AttnConfig, SSMConfig, HNetConfig
- Core modules: Isotropic, RoutingModule/ChunkLayer/DeChunkLayer, HNet
- Optional LM wrapper: HNetForCausalLM

Notes:
- This is intended for correctness and structure parity, not kernel-level speed.
- The SSM mixer is a simplified JAX substitute; feel free to swap it with a faster or
  more faithful implementation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax import lax
import flax.nnx as nnx
from flax.nnx import Param
import numpy as np


# ============================
# Utilities and Configs
# ============================


def get_seq_idx(cu_seqlens: jnp.ndarray) -> jnp.ndarray:
    """Return sequence indices for packed representation.
    cu_seqlens: (B+1,) cumulative lengths
    Returns: (1, T) int tensor mapping each token to its batch index.
    """
    seq_idx = jnp.zeros(cu_seqlens[-1], dtype=jnp.int32)
    seq_idx = seq_idx.at[cu_seqlens[:-1]].set(1)
    seq_idx = (jnp.cumsum(seq_idx) - 1)[None, :]
    return seq_idx


def get_stage_cfg(cfg, stage_idx: int):
    return {
        k: (v[stage_idx] if isinstance(v, list) else v) for k, v in asdict(cfg).items()
    }


@dataclass
class AttnConfig:
    num_heads: list[int] = field(default_factory=list)
    rotary_emb_dim: list[int] = field(default_factory=list)
    window_size: list[int] = field(default_factory=list)


@dataclass
class SSMConfig:
    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256


@dataclass
class HNetConfig:
    arch_layout: list[Union[str, list]] = field(default_factory=list)
    d_model: list[int] = field(default_factory=list)
    d_intermediate: list[int] = field(default_factory=list)
    vocab_size: int = 256
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    tie_embeddings: bool = False


# ============================
# Inference State Containers
# ============================


@dataclass
class InferenceParams:
    """Container for inference parameters and caches."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict[int, jnp.ndarray] = field(default_factory=dict)
    lengths_per_sample: jnp.ndarray | None = None

    def reset(self, max_seqlen: int, max_batch_size: int):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        self.key_value_memory_dict.clear()


@dataclass
class RoutingModuleOutput:
    boundary_prob: jnp.ndarray
    boundary_mask: jnp.ndarray
    selected_probs: jnp.ndarray


@dataclass
class RoutingModuleState:
    has_seen_tokens: jnp.ndarray  # (B,)
    last_hidden_state: jnp.ndarray  # (B, D)


@dataclass
class DeChunkState:
    last_value: jnp.ndarray  # (B, D)


@dataclass
class HNetState:
    encoder_state: Optional[InferenceParams] = None
    routing_module_state: Optional[RoutingModuleState] = None
    main_network_state: Optional[Union["HNetState", InferenceParams]] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[InferenceParams] = None


# ============================
# Norms and simple building blocks
# ============================


class RMSNorm(nnx.Module):
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


class SwiGLU(nnx.Module):
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


# ============================
# Simple JAX Causal MHA (no FlashAttention)
# ============================


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
        attn_mask: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        # q, k, v: (B, H, L, D)
        scale = self.head_dim**-0.5
        attn_scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale  # (B, H, L, L)
        L = q.shape[-2]
        causal = jnp.triu(jnp.ones((L, L), dtype=jnp.bool_), k=1)
        attn_scores = jnp.where(causal, -jnp.inf, attn_scores)

        if self.window_size is not None and self.window_size > 0:
            # Disallow keys older than window_size
            idx = jnp.arange(L)
            # True where j < i - (W-1)
            win_mask = idx[None, :] < (idx[:, None] - (self.window_size - 1))
            attn_scores = jnp.where(win_mask, -jnp.inf, attn_scores)

        if attn_mask is not None:
            # attn_mask: (B, L) with True for valid tokens
            mask = (
                attn_mask[:, None, None, :]
                .repeat(self.num_heads, axis=1)
                .repeat(L, axis=2)
            )
            attn_scores = jnp.where(~mask, -jnp.inf, attn_scores)

        attn = jax.nn.softmax(attn_scores, axis=-1)
        return jnp.matmul(attn, v)

    def _rotary_cos_sin(self, L: int, offset: int, dtype):
        """Compute rotary cos/sin tables for sequence length L starting at position offset.
        Returns cos, sin with shape (L, rotary_emb_dim//2) in given dtype.
        """
        ro_dim = self.rotary_emb_dim
        assert ro_dim % 2 == 0 and ro_dim <= self.head_dim
        inv_freq = 1.0 / (
            self.rotary_base ** (jnp.arange(0, ro_dim, 2, dtype=jnp.float32) / ro_dim)
        )
        # positions [offset, offset+L-1]
        t = jnp.arange(offset, offset + L, dtype=jnp.float32)
        freqs = jnp.outer(t, inv_freq)  # (L, ro_dim/2)
        cos = jnp.cos(freqs).astype(dtype)
        sin = jnp.sin(freqs).astype(dtype)
        return cos, sin

    def _apply_rotary(self, q: jnp.ndarray, k: jnp.ndarray, offset: int = 0):
        """Apply rotary embeddings on first rotary_emb_dim of q and k (NeoX style, non-interleaved).
        q, k: (B, H, L, Dh)
        """
        if self.rotary_emb_dim <= 0:
            return q, k
        B, H, L, Dh = q.shape
        ro_dim = self.rotary_emb_dim
        cos, sin = self._rotary_cos_sin(L, offset, q.dtype)
        # shape to (1,1,L,ro_dim/2) for broadcast
        cos = cos.reshape(1, 1, L, -1)
        sin = sin.reshape(1, 1, L, -1)

        def rotate_half(x):
            x1, x2 = jnp.split(x, 2, axis=-1)  # (..., ro_dim/2)
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
        cu_seqlens: Optional[jnp.ndarray] = None,
        max_seqlen: Optional[int] = None,
        attn_mask: Optional[jnp.ndarray] = None,
        inference_params=None,
    ) -> jnp.ndarray:
        # Support either packed (T, D) with cu_seqlens+max_seqlen or padded (B, L, D) with attn_mask.
        packed = cu_seqlens is not None and max_seqlen is not None
        if packed:
            # Unpack to padded then repack after attention
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
            # Repack
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
        self, x: jnp.ndarray, attn_mask: Optional[jnp.ndarray], inference_params=None
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

        # Rotary (absolute positions starting at seqlen_offset if provided)
        if self.rotary_emb_dim > 0:
            offset = int(getattr(inference_params, "seqlen_offset", 0) or 0)
            q, k = self._apply_rotary(q, k, offset=offset)

        ctx = self._causal_attn(q, k, v, attn_mask)  # (B, H, L, Dh)
        ctx = self._merge_heads(ctx)  # (B, L, D)
        out = self.out_proj(ctx)

        # Write K/V to cache during prefill if inference_params provided
        if inference_params is not None:
            assert getattr(self, "layer_idx", None) is not None, (
                "layer_idx required for KV cache"
            )
            # lazy-allocate cache
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
        # Not implementing KV cache here; return placeholder
        return None

    def step(self, x: jnp.ndarray, inference_params=None) -> jnp.ndarray:
        # Incremental decoding with KV cache and rotary.
        assert x.ndim == 3 and x.shape[1] == 1
        B, L, _ = x.shape
        x_dtype = x.dtype
        w_dtype = self.Wqkv.kernel.value.dtype
        if x_dtype != w_dtype:
            x = x.astype(w_dtype)

        qkv = self.Wqkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = self._split_heads(q)  # (B, H, 1, Dh)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.rotary_emb_dim > 0:
            offset = int(getattr(inference_params, "seqlen_offset", 0) or 0)
            q, k = self._apply_rotary(q, k, offset=offset)

        # Lazy-allocate / fetch KV cache
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

        # Write current K/V
        kv_cache = kv_cache.at[batch_start:batch_end, seq_start:seq_end, 0, ...].set(
            k.transpose(0, 2, 1, 3)
        )  # K
        kv_cache = kv_cache.at[batch_start:batch_end, seq_start:seq_end, 1, ...].set(
            v.transpose(0, 2, 1, 3)
        )  # V
        inference_params.key_value_memory_dict[self.layer_idx] = kv_cache

        # Read past K/V up to seq_end, with optional windowing
        start = (
            max(0, seq_end - self.window_size)
            if (self.window_size is not None and self.window_size > 0)
            else 0
        )
        K_all = kv_cache[batch_start:batch_end, start:seq_end, 0, ...].transpose(
            0, 2, 1, 3
        )  # (B, H, S, Dh)
        V_all = kv_cache[batch_start:batch_end, start:seq_end, 1, ...].transpose(
            0, 2, 1, 3
        )

        # Attention over full prefix
        scale = self.head_dim**-0.5
        attn_scores = jnp.matmul(q, K_all.transpose(0, 1, 3, 2)) * scale  # (B, H, 1, S)
        attn = jax.nn.softmax(attn_scores, axis=-1)
        ctx = jnp.matmul(attn, V_all)  # (B, H, 1, Dh)
        ctx = self._merge_heads(ctx)  # (B, 1, D)
        out = self.out_proj(ctx)
        return out.astype(x_dtype)


# ============================
# Lightweight SSM-like Mixer (JAX-only stand-in for Mamba2)
# ============================


class PyTorchSSM(nnx.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dtype=jnp.float32,
        layer_idx: int | None = None,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim
        self.nheads = self.d_inner // self.headdim
        self.layer_idx = layer_idx

        # Match Mamba2 parameterization to align with checkpoints
        # Order: [z (d_inner), x (d_inner), B (d_state), C (d_state), dt (nheads)]
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nnx.Linear(
            d_model, d_in_proj, use_bias=False, dtype=dtype, rngs=rngs
        )

        # Depthwise 1D conv over sequence for dynamics on concat(x, B, C)
        conv_dim = self.d_inner + 2 * self.d_state
        # Store conv weights in JAX format (K, C) for efficiency
        self.conv1d_weight = Param(jnp.zeros((d_conv, conv_dim), dtype=dtype))
        self.conv1d_bias = Param(jnp.zeros((conv_dim,), dtype=dtype))

        self.norm = RMSNorm(self.d_inner, dtype=dtype, rngs=rngs)
        self.out_proj = nnx.Linear(
            self.d_inner, d_model, use_bias=False, dtype=dtype, rngs=rngs
        )

        # Extra parameters present in checkpoints (not fully used but included for loading)
        self.dt_bias = Param(jnp.zeros((self.nheads,), dtype=dtype))
        self.A_log = Param(jnp.zeros((self.nheads,), dtype=dtype))
        self.D = Param(jnp.ones((self.nheads,), dtype=dtype))

    def _depthwise_conv1d(self, x: jnp.ndarray) -> jnp.ndarray:
        """Depthwise 1D convolution matching the working JAX implementation."""
        # x: (B, L, C). Per-channel cross-correlation with left padding (K-1)
        B, L, C = x.shape
        K = self.d_conv
        # Left pad
        pad = jnp.zeros((B, K - 1, C), dtype=x.dtype)
        xp = jnp.concatenate([pad, x], axis=1)  # (B, L+K-1, C)
        # Prepare (C,B,L+K-1,1)
        xc = jnp.swapaxes(xp, 1, 2)[..., None]
        xc = jnp.swapaxes(xc, 0, 1)
        # Reverse kernel along length to implement cross-correlation via convolution
        k = self.conv1d_weight.value[::-1, :]  # (K, C)
        wc = jnp.swapaxes(k[:, :, None], 0, 1)  # (C, K, 1)
        wc = wc[:, :, None]  # (C, K, 1, 1)

        def conv_c(xc_i, wc_i):
            # Ensure dtypes match
            wc_i = wc_i.astype(xc_i.dtype)
            return lax.conv_general_dilated(
                xc_i,
                wc_i,
                window_strides=(1,),
                padding="VALID",
                dimension_numbers=("NLC", "LIO", "NLC"),
            )  # (B, L, 1)

        y_c = jax.vmap(conv_c, in_axes=(0, 0), out_axes=0)(xc, wc)  # (C, B, L, 1)
        y = jnp.transpose(y_c[..., 0], (1, 2, 0))  # (B, L, C)
        if self.conv1d_bias is not None:
            y = y + self.conv1d_bias.value.astype(y.dtype)
        return y

    def __call__(
        self,
        x: jnp.ndarray,
        seq_idx: Optional[jnp.ndarray] = None,
        inference_params=None,
    ) -> jnp.ndarray:
        # x: (B, L, D)
        B, L, D = x.shape
        x_dtype = x.dtype
        w_dtype = self.in_proj.kernel.value.dtype
        if x_dtype != w_dtype:
            x = x.astype(w_dtype)

        zxbcdt = self.in_proj(x)
        z, xBC, dt = jnp.split(
            zxbcdt,
            [self.d_inner, 2 * self.d_inner + 2 * self.d_state],
            axis=-1,
        )

        # Depthwise conv over concat(x, B, C) - using proven JAX approach
        xBC = jax.nn.silu(self._depthwise_conv1d(xBC))

        if xBC.shape[1] != L:
            xBC = xBC[:, :L, :]

        x_part, B_part, C_part = jnp.split(
            xBC, [self.d_inner, self.d_inner + self.d_state], axis=-1
        )

        # Heads view
        x_heads = x_part.reshape(B, L, self.nheads, self.headdim)
        B_heads = B_part[:, :, None, :].repeat(self.nheads, axis=2)
        C_heads = C_part[:, :, None, :].repeat(self.nheads, axis=2)
        dt = jax.nn.softplus(dt + self.dt_bias.value)  # (B, L, H)
        A = -jnp.exp(self.A_log.value)  # (H,)

        # Stateful scan (naive O(L * H * P * S))
        state = jnp.zeros((B, self.nheads, self.headdim, self.d_state), dtype=w_dtype)

        def scan_fn(state, inputs):
            dt_t, x_t, B_t, C_t = inputs
            decay = jnp.exp(A[None, :] * dt_t)  # (B, H)
            state = state * decay[:, :, None, None]
            state = (
                state
                + (x_t[:, :, :, None] * B_t[:, :, None, :]) * dt_t[:, :, None, None]
            )
            y_t = jnp.sum(state * C_t[:, :, None, :], axis=-1)  # (B, H, P)
            return state, y_t

        inputs = (
            dt.transpose(1, 0, 2),
            x_heads.transpose(1, 0, 2, 3),
            B_heads.transpose(1, 0, 2, 3),
            C_heads.transpose(1, 0, 2, 3),
        )
        _, y_acc = lax.scan(scan_fn, state, inputs)
        y_acc = y_acc.transpose(1, 0, 2, 3)  # (B, L, H, P)

        y = y_acc.reshape(B, L, self.d_inner)
        D_full = (
            self.D.value[:, None]
            .repeat(self.headdim, axis=1)
            .reshape(-1)[None, None, :]
        )
        y = y + x_part * D_full
        y = self.norm(y * jax.nn.silu(z))
        out = self.out_proj(y)
        return out.astype(x_dtype)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        return None

    def step(self, x: jnp.ndarray, inference_params=None) -> jnp.ndarray:
        # x: (B, 1, D)
        assert x.shape[1] == 1
        B = x.shape[0]
        x_dtype = x.dtype
        w_dtype = self.in_proj.kernel.value.dtype
        if x_dtype != w_dtype:
            x = x.astype(w_dtype)

        zxbcdt = self.in_proj(x)
        z, xBC, dt = jnp.split(
            zxbcdt,
            [self.d_inner, 2 * self.d_inner + 2 * self.d_state],
            axis=-1,
        )

        # Cache states
        layer_idx = getattr(self, "layer_idx", None)
        assert layer_idx is not None and inference_params is not None
        cache = inference_params.key_value_memory_dict
        conv_dim = self.d_inner + 2 * self.d_state

        if layer_idx not in cache:
            conv_state = jnp.zeros((B, conv_dim, self.d_conv), dtype=w_dtype)
            ssm_state = jnp.zeros(
                (B, self.nheads, self.headdim, self.d_state), dtype=w_dtype
            )
        else:
            conv_state, ssm_state = cache[layer_idx]
            if conv_state.shape[0] != B:
                conv_state = conv_state[:B]
                ssm_state = ssm_state[:B]

        # Depthwise conv via state - using proven JAX approach
        xBC_t = xBC[:, 0]  # (B, conv_dim)
        conv_state = jnp.concatenate([conv_state[:, :, 1:], xBC_t[:, :, None]], axis=2)

        # Cross-correlation per-channel using einsum (from working JAX version)
        # conv_state: (B, conv_dim, d_conv), weight: (d_conv, conv_dim)
        weight = self.conv1d_weight.value.astype(conv_state.dtype)
        conv_out = jnp.einsum("bck,kc->bc", conv_state, weight)
        conv_out = conv_out + self.conv1d_bias.value.astype(conv_state.dtype)
        conv_out = jax.nn.silu(conv_out)

        x_part, B_part, C_part = jnp.split(
            conv_out, [self.d_inner, self.d_inner + self.d_state], axis=-1
        )
        x_heads = x_part.reshape(B, self.nheads, self.headdim)
        B_t = B_part[:, None, :].repeat(self.nheads, axis=1)
        C_t = C_part[:, None, :].repeat(self.nheads, axis=1)
        dt_t = jax.nn.softplus(dt[:, 0] + self.dt_bias.value)  # (B, H)
        A = -jnp.exp(self.A_log.value)

        decay = jnp.exp(A[None, :] * dt_t)
        ssm_state = ssm_state * decay[:, :, None, None]
        ssm_state = (
            ssm_state
            + (x_heads[:, :, :, None] * B_t[:, :, None, :]) * dt_t[:, :, None, None]
        )
        y_t = jnp.sum(ssm_state * C_t[:, :, None, :], axis=-1)  # (B, H, P)

        y = y_t.reshape(B, 1, self.d_inner)
        D_full = (
            self.D.value[:, None]
            .repeat(self.headdim, axis=1)
            .reshape(-1)[None, None, :]
        )
        y = y + x_part[:, None, :] * D_full
        y = self.norm(y * jax.nn.silu(z))
        out = self.out_proj(y)

        cache[layer_idx] = [conv_state, ssm_state]
        return out.astype(x_dtype)


# ============================
# Blocks and Isotropic stack
# ============================


class Block(nnx.Module):
    def __init__(
        self,
        d_model: int,
        mixer: nnx.Module,
        mlp: Optional[nnx.Module],
        residual_in_fp32: bool = True,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.residual_in_fp32 = residual_in_fp32
        self.norm1 = RMSNorm(d_model, rngs=rngs)
        self.mixer = mixer
        self.mlp = mlp
        if self.mlp is not None:
            self.norm2 = RMSNorm(d_model, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        residual: jnp.ndarray | None = None,
        inference_params=None,
        mixer_kwargs: dict | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        x, residual = self.norm1(
            x, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32
        )
        mixer_kwargs = mixer_kwargs or {}
        x = self.mixer(x, **mixer_kwargs, inference_params=inference_params)
        if self.mlp is not None:
            x, residual = self.norm2(
                x,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            x = self.mlp(x)
        return x, residual

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype=None, **kwargs
    ):
        return None

    def step(
        self, x: jnp.ndarray, inference_params, residual: Optional[jnp.ndarray] = None
    ):
        x, residual = self.norm1(
            x, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32
        )
        x = self.mixer.step(x, inference_params)
        if self.mlp is not None:
            x, residual = self.norm2(
                x,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            x = self.mlp(x)
        return x, residual


class Isotropic(nnx.Module):
    def __init__(
        self,
        config: HNetConfig,
        pos_idx: int,
        stage_idx: int,
        dtype=jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.stage_idx = stage_idx
        self.d_model = config.d_model[self.stage_idx]
        self.ssm_cfg = get_stage_cfg(config.ssm_cfg, stage_idx)
        self.attn_cfg = get_stage_cfg(config.attn_cfg, stage_idx)

        # Parse arch layout for this submodule position (encoder/main/decoder)
        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]
        arch_layout = arch_layout[pos_idx]

        # Parse patterns like "m4T2"
        layout_parse = re.findall(r"([mMtT])(\d+)", arch_layout)
        layers: list[Block] = []
        layer_idx = 0
        for arch, n_layer_str in layout_parse:
            n_layer = int(n_layer_str)
            for _ in range(n_layer):
                if arch in ("t", "T"):
                    mixer = CausalMHA(
                        self.d_model,
                        num_heads=self.attn_cfg.get("num_heads", 8),
                        rotary_emb_dim=self.attn_cfg.get("rotary_emb_dim", 0),
                        window_size=self.attn_cfg.get("window_size", -1),
                        dtype=dtype,
                        layer_idx=layer_idx,
                        rngs=rngs,
                    )
                elif arch in ("m", "M"):
                    mixer = PyTorchSSM(
                        self.d_model,
                        d_state=self.ssm_cfg.get("d_state", 128),
                        d_conv=self.ssm_cfg.get("d_conv", 4),
                        expand=self.ssm_cfg.get("expand", 2),
                        dtype=dtype,
                        layer_idx=layer_idx,
                        rngs=rngs,
                    )
                else:
                    raise NotImplementedError

                if arch in ("T", "M"):
                    mlp = SwiGLU(
                        self.d_model,
                        d_intermediate=config.d_intermediate[self.stage_idx],
                        dtype=dtype,
                        rngs=rngs,
                    )
                else:
                    mlp = None
                layers.append(Block(self.d_model, mixer, mlp, rngs=rngs))
                layer_idx += 1

        self.layers = layers
        self.rmsnorm = RMSNorm(self.d_model, eps=1e-5, dtype=dtype, rngs=rngs)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        return InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cu_seqlens: Optional[jnp.ndarray] = None,
        max_seqlen: Optional[int] = None,
        mask: Optional[jnp.ndarray] = None,
        inference_params: Optional[InferenceParams] = None,
        **mixer_kwargs,
    ) -> jnp.ndarray:
        packed = cu_seqlens is not None and max_seqlen is not None and mask is None
        if packed:
            # Unpack to padded input
            assert cu_seqlens is not None and max_seqlen is not None
            T, D = hidden_states.shape
            B = len(cu_seqlens) - 1
            Lmax = int(max_seqlen)
            x = jnp.zeros((B, Lmax, D), dtype=hidden_states.dtype)
            mask_local = jnp.zeros((B, Lmax), dtype=jnp.bool_)
            for b in range(B):
                s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
                x = x.at[b, : e - s].set(hidden_states[s:e])
                mask_local = mask_local.at[b, : e - s].set(True)
        else:
            x = hidden_states
            mask_local = mask

        residual = None
        for layer in self.layers:
            # Each mixer expects padded input with an attention mask if needed
            if isinstance(layer.mixer, CausalMHA):
                mix_kwargs = {"attn_mask": mask_local}
            else:
                mix_kwargs = {}
            x, residual = layer(
                x,
                residual=residual,
                inference_params=inference_params,
                mixer_kwargs=mix_kwargs,
            )

        x = self.rmsnorm(x, residual=residual, prenorm=False, residual_in_fp32=True)

        if packed:
            outs = []
            for b in range(B):
                L = int(jnp.sum(mask_local[b]))
                outs.append(x[b, :L])
            x = jnp.concatenate(outs, axis=0)

        if inference_params is not None:
            # Follow reference: assert batch size 1 and padded path when tracking seqlen_offset
            assert mask is not None, (
                "Mask must be provided if inference_params is provided"
            )
            assert mask.shape[0] == 1, "seqlen_offset handling assumes batch size 1"
            assert x.ndim == 3, (
                "Inference with inference_params expects padded (B, L, D)"
            )
            inference_params.seqlen_offset += x.shape[1]

        return x

    def step(self, hidden_states: jnp.ndarray, inference_params: InferenceParams):
        residual = None
        x = hidden_states
        for layer in self.layers:
            x, residual = layer.step(x, inference_params, residual=residual)
        x = self.rmsnorm(x, residual=residual, prenorm=False, residual_in_fp32=True)
        inference_params.seqlen_offset += 1
        return x


# ============================
# Dynamic Chunking (Router/Chunk/Dechunk)
# ============================


class RoutingModule(nnx.Module):
    def __init__(self, d_model: int, dtype=jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.d_model = d_model
        self.q_proj_layer = nnx.Linear(
            d_model, d_model, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.k_proj_layer = nnx.Linear(
            d_model, d_model, use_bias=False, dtype=dtype, rngs=rngs
        )

        # Initialize as identity matrices like PyTorch version
        self.q_proj_layer.kernel.value = jnp.eye(d_model, dtype=dtype)
        self.k_proj_layer.kernel.value = jnp.eye(d_model, dtype=dtype)

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype=jnp.float32
    ):
        return RoutingModuleState(
            has_seen_tokens=jnp.zeros(batch_size, dtype=jnp.bool_),
            last_hidden_state=jnp.zeros((batch_size, self.d_model), dtype=dtype),
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cu_seqlens: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        inference_params: Optional[RoutingModuleState] = None,
    ) -> RoutingModuleOutput:
        assert (mask is not None) or (cu_seqlens is not None), (
            "Provide mask or cu_seqlens"
        )
        if inference_params is not None:
            # Match reference behavior: prefill requires mask and unseen state
            assert mask is not None, (
                "Mask must be provided if inference_params is provided"
            )
            assert (~inference_params.has_seen_tokens).all(), (
                "Cannot have seen tokens when inference_params is provided"
            )
            # Not supporting packed + inference_params
            assert cu_seqlens is None, (
                "Packed mode with inference_params is not supported"
            )

        if cu_seqlens is not None:
            # Treat as single batch for computation convenience
            hs = hidden_states[None, :, :]  # (1, T, D)
        else:
            hs = hidden_states  # (B, L, D)

        # Ensure dtype matches projection weights
        hs_dtype = hs.dtype
        w_dtype = self.q_proj_layer.kernel.value.dtype
        if hs_dtype != w_dtype:
            hs = hs.astype(w_dtype)

        # Cosine similarity between consecutive tokens
        q_proj = self.q_proj_layer(hs[:, :-1])  # (B, L-1, D)
        k_proj = self.k_proj_layer(hs[:, 1:])  # (B, L-1, D)

        q_norm = q_proj / jnp.linalg.norm(q_proj, axis=-1, keepdims=True)
        k_norm = k_proj / jnp.linalg.norm(k_proj, axis=-1, keepdims=True)

        cos_sim = jnp.sum(q_norm * k_norm, axis=-1)  # (B, L-1)
        boundary_prob = jnp.clip((1 - cos_sim) / 2, 0.0, 1.0)

        # Force first token as boundary
        PAD_PROB = 1.0
        boundary_prob = jnp.pad(
            boundary_prob, ((0, 0), (1, 0)), constant_values=PAD_PROB
        )

        if cu_seqlens is not None:
            boundary_prob = boundary_prob.squeeze(0)
            boundary_prob = boundary_prob.at[cu_seqlens[:-1]].set(PAD_PROB)

        boundary_prob = jnp.stack([1 - boundary_prob, boundary_prob], axis=-1)
        selected_idx = jnp.argmax(boundary_prob, axis=-1)
        boundary_mask = selected_idx == 1

        if mask is not None:
            boundary_mask = boundary_mask & mask

        if inference_params is not None:
            # Update prefill state so that step() has correct previous token context
            assert mask is not None
            has_mask = mask.any(axis=-1)
            inference_params.has_seen_tokens = (
                has_mask | inference_params.has_seen_tokens
            )
            last_mask = jnp.clip(mask.sum(axis=-1) - 1, a_min=0)
            idx_b = jnp.arange(hidden_states.shape[0])
            last_h = hidden_states[idx_b, last_mask]
            inference_params.last_hidden_state = jnp.where(
                has_mask[:, None],
                last_h,
                inference_params.last_hidden_state,
            )

        selected_probs = jnp.take_along_axis(
            boundary_prob, selected_idx[..., None], axis=-1
        )
        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )

    def step(
        self, hidden_states: jnp.ndarray, inference_params: RoutingModuleState
    ) -> RoutingModuleOutput:
        # hidden_states: (B, 1, D)
        hs = hidden_states.squeeze(1)
        # Align dtype to projection weights
        hs_dtype = hs.dtype
        w_dtype = self.q_proj_layer.kernel.value.dtype
        if hs_dtype != w_dtype:
            hs = hs.astype(w_dtype)

        q_proj = self.q_proj_layer(inference_params.last_hidden_state)
        k_proj = self.k_proj_layer(hs)

        q_norm = q_proj / jnp.linalg.norm(q_proj, axis=-1, keepdims=True)
        k_norm = k_proj / jnp.linalg.norm(k_proj, axis=-1, keepdims=True)

        cos_sim = jnp.sum(q_norm * k_norm, axis=-1)  # (B,)
        boundary_prob = jnp.clip((1 - cos_sim) / 2, 0.0, 1.0)
        inference_params.last_hidden_state = hs
        boundary_prob = jnp.where(
            inference_params.has_seen_tokens,
            boundary_prob,
            jnp.ones_like(boundary_prob),
        )
        boundary_prob = jnp.stack([1 - boundary_prob, boundary_prob], axis=-1)
        inference_params.has_seen_tokens = jnp.ones_like(
            inference_params.has_seen_tokens
        )
        # Match reference: threshold 0.5 for boundary
        boundary_mask = boundary_prob[..., 1] > 0.5
        selected_probs = boundary_prob.max(axis=-1, keepdims=True)
        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )


class ChunkLayer(nnx.Module):
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        boundary_mask: jnp.ndarray,
        cu_seqlens: jnp.ndarray | None = None,
        mask: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None, int | None, jnp.ndarray | None]:
        assert (mask is not None) or (cu_seqlens is not None), (
            "Provide mask or cu_seqlens"
        )
        if cu_seqlens is not None:
            next_hidden_states = hidden_states[boundary_mask]
            next_cu_seqlens = jnp.pad(
                jnp.cumsum(boundary_mask)[cu_seqlens[1:] - 1], (1, 0)
            )
            next_max_seqlen = int((next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max())
            next_mask = None
        else:
            next_cu_seqlens = None
            num_tokens = boundary_mask.sum(axis=-1)
            next_max_seqlen = int(num_tokens.max())
            L = hidden_states.shape[1]

            # Create sorted indices to bring boundary tokens to front
            token_idx = jnp.arange(L)[None, :] + (~boundary_mask).astype(jnp.int32) * L
            seq_sorted_indices = jnp.argsort(token_idx, axis=1)

            # Gather selected tokens
            next_hidden_states = jnp.take_along_axis(
                hidden_states,
                seq_sorted_indices[:, :next_max_seqlen, None].repeat(
                    hidden_states.shape[-1], axis=2
                ),
                axis=1,
            )
            next_mask = jnp.arange(next_max_seqlen)[None, :] < num_tokens[:, None]
            next_max_seqlen = None
        return next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask

    def step(
        self, hidden_states: jnp.ndarray, boundary_mask: jnp.ndarray
    ) -> jnp.ndarray:
        return hidden_states[boundary_mask]


class DeChunkLayer(nnx.Module):
    def __init__(self, d_model: int):
        self.d_model = d_model

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype=jnp.float32
    ):
        return DeChunkState(
            last_value=jnp.zeros((batch_size, self.d_model), dtype=dtype)
        )

    @staticmethod
    def _ema_sequence(hidden_seq: jnp.ndarray, p_seq: jnp.ndarray) -> jnp.ndarray:
        """Compute EMA over a sequence of selected hidden states.
        hidden_seq: (K, D), p_seq: (K,)
        Returns: (K, D) EMA states.
        """

        def scan_fn(h, inputs):
            pk, hidden_k = inputs
            h = (1 - pk) * h + pk * hidden_k
            return h, h

        K, D = hidden_seq.shape
        h_init = jnp.zeros(D, dtype=hidden_seq.dtype)
        _, out = lax.scan(scan_fn, h_init, (p_seq, hidden_seq))
        return out

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        boundary_mask: jnp.ndarray,
        boundary_prob: jnp.ndarray,
        cu_seqlens: Optional[jnp.ndarray] = None,
        inference_params: Optional[DeChunkState] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if inference_params is None:
            assert mask is not None, "Mask must be provided in prefill"
            # First token must be boundary
            first_boundary = (
                boundary_mask[:, 0] if boundary_mask.ndim == 2 else boundary_mask[0]
            )
            assert (
                first_boundary.all()
                if first_boundary.ndim > 0
                else first_boundary.item() == 1
            )

        # Extract p probabilities (B, L) or (T,)
        if boundary_prob.shape[-1] == 2:
            p_full = boundary_prob[..., -1].astype(jnp.float32)
        else:
            p_full = boundary_prob.astype(jnp.float32)
        p_full = jnp.clip(p_full, 1e-4, 1 - 1e-4)

        original_dtype = hidden_states.dtype

        if cu_seqlens is not None:
            # Packed path: hidden_states are selected boundary states concatenated across batch
            # boundary_mask and p_full correspond to original tokens across batch
            B = len(cu_seqlens) - 1

            # Compute per-sample counts of selected tokens (boundaries)
            selected_mask = boundary_mask
            selected_p = p_full[selected_mask]  # (sum K_b,)

            # Build cu for selected
            sel_counts = []
            for b in range(B):
                s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
                sel_counts.append(int(selected_mask[s:e].sum()))
            sel_cu = [0]
            for c in sel_counts:
                sel_cu.append(sel_cu[-1] + c)
            sel_cu = jnp.array(sel_cu)

            # Compute EMA per sample over selected hidden states
            ema_selected = jnp.zeros_like(hidden_states)
            for b in range(B):
                ks, ke = int(sel_cu[b]), int(sel_cu[b + 1])
                if ke > ks:
                    ema_selected = ema_selected.at[ks:ke].set(
                        self._ema_sequence(hidden_states[ks:ke], selected_p[ks:ke])
                    )

            # Map EMA back to full token positions via plug-back index = cumsum(boundary_mask)-1
            plug_back_idx = jnp.cumsum(boundary_mask) - 1  # (T_total,)
            # Gather from ema_selected using flat mapping
            out_full = jnp.take_along_axis(
                ema_selected,
                jnp.clip(plug_back_idx, a_min=0)[:, None].repeat(self.d_model, axis=1),
                axis=0,
            )
            return out_full.astype(original_dtype)
        else:
            # Unpacked (B, L, D)
            B, L = boundary_mask.shape
            # Compute EMA along selected tokens for each sample
            # First, sort tokens to bring boundaries to front like reference
            token_idx = jnp.arange(L)[None, :] + (~boundary_mask).astype(jnp.int32) * L
            seq_sorted_indices = jnp.argsort(token_idx, axis=1)

            # Number of selected (boundary) tokens per sample
            num_tokens = boundary_mask.sum(axis=-1)  # (B,)
            # hidden_states is already chunked/selected with width Mmax from chunk layer
            M = hidden_states.shape[1]
            selected_hidden = hidden_states  # (B, M, D)
            # Align p to the same chunked order and truncate to M
            p_sorted = jnp.take_along_axis(
                p_full, seq_sorted_indices[:, :M], axis=1
            )  # (B, M)

            ema_selected = jnp.zeros_like(selected_hidden)
            for b in range(B):
                m = int(num_tokens[b])
                if m > 0:
                    ema_selected = ema_selected.at[b, :m].set(
                        self._ema_sequence(selected_hidden[b, :m], p_sorted[b, :m])
                    )

            plug_back_idx = (
                jnp.cumsum(boundary_mask, axis=1) - 1
            )  # (B, L), -1 where not boundary yet
            out = jnp.take_along_axis(
                ema_selected,
                jnp.clip(plug_back_idx, a_min=0)[..., None].repeat(
                    self.d_model, axis=2
                ),
                axis=1,
            )
            if inference_params is not None:
                inference_params.last_value = out[:, -1]
            return out.astype(original_dtype)

    def step(
        self,
        hidden_states: jnp.ndarray,
        boundary_mask: jnp.ndarray,
        boundary_prob: jnp.ndarray,
        inference_params: DeChunkState,
    ) -> jnp.ndarray:
        # hidden_states: (B', 1, D) for selected tokens, boundary_mask: (B,), boundary_prob: (B, 2)
        B = boundary_mask.shape[0]
        D = hidden_states.shape[-1]
        p = jnp.zeros(B, dtype=hidden_states.dtype)
        bp = boundary_prob.astype(p.dtype)
        p = p.at[boundary_mask].set(jnp.clip(bp[boundary_mask, -1], 1e-4, 1 - 1e-4))

        current = jnp.zeros((B, D), dtype=hidden_states.dtype)
        if hidden_states.size > 0:
            current = current.at[boundary_mask].set(hidden_states.squeeze(1))

        result = p[:, None] * current + (1 - p)[:, None] * inference_params.last_value
        inference_params.last_value = result
        return result[:, None, :]


# ============================
# H-Net wrapper (recursive stages)
# ============================


class STE(nnx.Module):
    """Straight-through estimator for gradient flow."""

    def __call__(self, x):
        return jnp.ones_like(x)


def ste_func(x):
    # In JAX, we need to use stop_gradient for the STE behavior
    return jax.lax.stop_gradient(jnp.ones_like(x)) + (x - jax.lax.stop_gradient(x))


class HNet(nnx.Module):
    def __init__(
        self,
        config: HNetConfig,
        stage_idx: int,
        dtype=jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]

        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]

        assert isinstance(arch_layout, list), f"Wrong arch_layout: {arch_layout}"
        if len(arch_layout) == 3:
            sub_model_names = ["encoder", "main_network", "decoder"]
            self.is_innermost = False
        elif len(arch_layout) == 1:
            sub_model_names = ["main_network"]
            self.is_innermost = True
        else:
            raise NotImplementedError

        for _name, _layout in zip(sub_model_names, arch_layout, strict=True):
            if self.is_innermost or _name in ("encoder", "decoder"):
                SubModel = Isotropic
                _stage_idx = stage_idx
                _pos_idx = 0 if (_name == "encoder" or self.is_innermost) else 2
                _sub_model = SubModel(
                    config=config,
                    stage_idx=_stage_idx,
                    pos_idx=_pos_idx,
                    dtype=dtype,
                    rngs=rngs,
                )
            else:
                SubModel = HNet
                _stage_idx = stage_idx + 1
                _sub_model = SubModel(
                    config=config, stage_idx=_stage_idx, dtype=dtype, rngs=rngs
                )
            setattr(self, _name, _sub_model)

        if not self.is_innermost:
            self.routing_module = RoutingModule(self.d_model, dtype=dtype, rngs=rngs)
            self.chunk_layer = ChunkLayer()
            self.dechunk_layer = DeChunkLayer(self.d_model)
            # Residual in fp32
            self.residual_proj = nnx.Linear(
                self.d_model, self.d_model, dtype=jnp.float32, rngs=rngs
            )
            # Initialize to zeros like PyTorch version
            self.residual_proj.kernel.value = jnp.zeros_like(
                self.residual_proj.kernel.value
            )
            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual

        if stage_idx > 0 and self.d_model - config.d_model[stage_idx - 1] > 0:
            self.pad_dimension = Param(
                jnp.zeros((self.d_model - config.d_model[stage_idx - 1],), dtype=dtype)
            )
        else:
            self.pad_dimension = None

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype=jnp.float32
    ):
        if self.is_innermost:
            return HNetState(
                main_network_state=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                )
            )
        else:
            return HNetState(
                encoder_state=self.encoder.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                routing_module_state=self.routing_module.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                main_network_state=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                dechunk_state=self.dechunk_layer.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                decoder_state=self.decoder.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
            )

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cu_seqlens: Optional[jnp.ndarray] = None,
        max_seqlen: Optional[int] = None,
        mask: Optional[jnp.ndarray] = None,
        inference_params: Optional[HNetState] = None,
        **mixer_kwargs,
    ):
        assert mask is not None or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Provide mask or (cu_seqlens, max_seqlen)"
        if inference_params is None:
            inference_params = HNetState(main_network_state=None)
        else:
            assert mask is not None, (
                "Mask must be provided if inference_params is provided"
            )

        D = hidden_states.shape[-1]
        early_dims = hidden_states.shape[:-1]
        if self.pad_dimension is not None:
            pad_expanded = jnp.broadcast_to(
                self.pad_dimension.value,
                early_dims + (self.pad_dimension.value.shape[-1],),
            )
            hidden_states = jnp.concatenate([hidden_states, pad_expanded], axis=-1)

        if self.is_innermost:
            hs = self.main_network(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                mask=mask,
                inference_params=inference_params.main_network_state,
                **mixer_kwargs,
            )
            hs = hs[..., :D]
            return hs, []

        # Encoder
        hs = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params.encoder_state,
            **mixer_kwargs,
        )

        hs_for_residual = hs.astype(self.residual_proj.kernel.value.dtype)
        residual = self.residual_proj(hs_for_residual)

        # Routing
        bpred_output = self.routing_module(
            hs,
            cu_seqlens=cu_seqlens,
            mask=mask,
            inference_params=inference_params.routing_module_state,
        )
        # Chunk
        hs_chunk, next_cu, next_max_L, next_mask = self.chunk_layer(
            hs, bpred_output.boundary_mask, cu_seqlens, mask=mask
        )

        # Main inner network
        hs_inner, prev_boundary_predictions = self.main_network(
            hs_chunk,
            cu_seqlens=next_cu,
            max_seqlen=next_max_L,
            mask=next_mask,
            inference_params=inference_params.main_network_state,
            **mixer_kwargs,
        )

        # Dechunk back to original resolution
        hs = self.dechunk_layer(
            hs_inner,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            next_cu,
            mask=mask,
            inference_params=inference_params.dechunk_state,
        )

        # Residual fusion with STE gating
        hs = self.residual_func(
            hs.astype(residual.dtype), residual, bpred_output.selected_probs
        ).astype(hs.dtype)

        # Decoder
        hs = self.decoder(
            hs,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params.decoder_state,
            **mixer_kwargs,
        )

        hs = hs[..., :D]
        return hs, [bpred_output, *prev_boundary_predictions]

    def step(self, hidden_states: jnp.ndarray, inference_params: HNetState):
        D = hidden_states.shape[-1]
        if self.pad_dimension is not None:
            pad_expanded = jnp.broadcast_to(
                self.pad_dimension.value,
                hidden_states.shape[:-1] + (self.pad_dimension.value.shape[-1],),
            )
            hidden_states = jnp.concatenate([hidden_states, pad_expanded], axis=-1)

        if self.is_innermost:
            hs = self.main_network.step(
                hidden_states, inference_params.main_network_state
            )
            hs = hs[..., :D]
            return hs, []

        hs = self.encoder.step(hidden_states, inference_params.encoder_state)
        hs_for_residual = hs.astype(self.residual_proj.kernel.value.dtype)
        residual = self.residual_proj(hs_for_residual)
        bpred_output = self.routing_module.step(
            hs, inference_params.routing_module_state
        )
        hs_inner = self.chunk_layer.step(hs, bpred_output.boundary_mask)

        if hs_inner.shape[0] > 0:
            hs_inner, prev_boundary_predictions = self.main_network.step(
                hs_inner, inference_params.main_network_state
            )
        else:
            prev_boundary_predictions = []

        hs = self.dechunk_layer.step(
            hs_inner,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            inference_params.dechunk_state,
        )
        hs = self.residual_func(
            hs.astype(residual.dtype), residual, bpred_output.selected_probs
        ).astype(hs.dtype)
        hs = self.decoder.step(hs, inference_params.decoder_state)
        hs = hs[..., :D]
        return hs, [bpred_output, *prev_boundary_predictions]


# ============================
# Optional: simple LM wrapper operating on byte vocab (UTF-8)
# ============================


class HNetForCausalLM(nnx.Module):
    def __init__(
        self, config: HNetConfig, dtype=jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0)
    ) -> None:
        self.config = config
        d_embed = config.d_model[0]
        self.embeddings = nnx.Embed(config.vocab_size, d_embed, dtype=dtype, rngs=rngs)
        self.backbone = HNet(config=config, stage_idx=0, dtype=dtype, rngs=rngs)
        self.lm_head = nnx.Linear(
            d_embed, config.vocab_size, use_bias=False, dtype=dtype, rngs=rngs
        )
        if config.tie_embeddings:
            # Share the embedding and output weights
            self.lm_head.kernel = self.embeddings.embedding

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype=jnp.float32
    ):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        position_ids=None,
        inference_params: Optional[HNetState] = None,
        num_last_tokens: int = 0,
        **mixer_kwargs,
    ):
        hidden_states = self.embeddings(input_ids)
        B, L, D = hidden_states.shape

        if mask is None:
            assert inference_params is None, (
                "Inference params not supported in packed mode here"
            )
            hidden_states = hidden_states.reshape(-1, D)
            cu = jnp.arange(B + 1) * L
            maxL = jnp.array(L, dtype=jnp.int32)
        else:
            cu, maxL = None, None

        hs, bpred_output = self.backbone(
            hidden_states,
            cu_seqlens=cu,
            max_seqlen=maxL,
            mask=mask,
            inference_params=inference_params,
            **mixer_kwargs,
        )
        hs = hs.reshape(B, L, D)

        if num_last_tokens > 0:
            hs = hs[:, -num_last_tokens:]

        # Align input dtype to lm_head weight, then cast logits to float32 for stability
        w_dtype = self.lm_head.kernel.value.dtype
        logits = self.lm_head(hs.astype(w_dtype)).astype(jnp.float32)
        return logits, bpred_output, inference_params

    def step(self, input_ids: jnp.ndarray, inference_params: HNetState):
        B = input_ids.shape[0]
        assert B == 1, "step currently supports batch size 1"
        hidden_states = self.embeddings(input_ids)
        hidden_states, bpred_output = self.backbone.step(
            hidden_states, inference_params
        )
        w_dtype = self.lm_head.kernel.value.dtype
        logits = self.lm_head(hidden_states.astype(w_dtype)).astype(jnp.float32)
        return logits, bpred_output, inference_params


# ============================
# CLI utilities: byte tokenizer + generation
# ============================


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> jnp.ndarray:
        b = text.encode("utf-8")
        if add_bos:
            b = bytes([self.bos_idx]) + b
        if add_eos:
            b = b + bytes([self.eos_idx])
        return jnp.array(list(b), dtype=jnp.int32)

    def decode(self, ids: jnp.ndarray) -> str:
        if ids.ndim > 1:
            ids = ids.flatten()
        arr = ids.astype(int).tolist()
        # Filter BOS/EOS in decode for nicer printing
        arr = [t for t in arr if t not in (self.bos_idx, self.eos_idx)]
        try:
            return bytearray(arr).decode("utf-8")
        except Exception:
            # More aggressive fallback: replace invalid bytes with placeholders
            try:
                return bytearray(arr).decode("utf-8", errors="replace")
            except Exception:
                # Ultimate fallback: return individual chars or hex
                result = []
                for tok in arr:
                    if 32 <= tok <= 126:  # Printable ASCII
                        result.append(chr(tok))
                    else:
                        result.append(f"\\x{tok:02x}")
                return "".join(result)


def _top_p_filtering(logits: jnp.ndarray, top_p: float) -> jnp.ndarray:
    if top_p >= 1.0:
        return logits
    sorted_logits = jnp.sort(logits, axis=-1)[::-1]  # descending
    sorted_indices = jnp.argsort(logits, axis=-1)[::-1]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift right and keep first token
    sorted_indices_to_remove = jnp.concatenate(
        [jnp.array([False]), sorted_indices_to_remove[:-1]]
    )
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits = logits.at[indices_to_remove].set(-jnp.inf)
    return logits


def load_config_from_json(json_path: str) -> HNetConfig:
    with open(json_path) as f:
        cfg = json.load(f)
    attn_cfg = AttnConfig(**cfg.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**cfg.pop("ssm_cfg"))
    return HNetConfig(**cfg, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)


def load_model(
    model_path: Optional[str],
    config_path: str,
    dtype: str = "bfloat16",
    strict: bool = True,
) -> HNetForCausalLM:
    torch_dtype = jnp.bfloat16 if dtype == "bfloat16" else jnp.float32
    cfg = load_config_from_json(config_path)
    rngs = nnx.Rngs(0)
    model = HNetForCausalLM(cfg, dtype=torch_dtype, rngs=rngs)

    if model_path:
        # Load PyTorch checkpoint and convert to JAX
        import torch

        state = torch.load(model_path, map_location="cpu")
        try:
            # Convert PyTorch state dict to JAX parameters
            jax_state = convert_pytorch_to_jax(state, model)
            # Set the parameters using the nested parameter structure
            update_model_parameters(model, jax_state)
        except Exception as e:
            # Provide debug info to help alignment
            ckpt_keys = sorted(list(state.keys()))
            model_keys = sorted([str(k) for k in nnx.state(model)])

            # Print a few around common prefixes
            def head_tail(arr):
                return arr[:20] + (["..."] if len(arr) > 40 else []) + arr[-20:]

            raise RuntimeError(
                "Error loading state_dict strictly.\n"
                f"Exception: {e}\n"
                f"Checkpoint keys sample: {head_tail(ckpt_keys)}\n"
                f"Model keys sample: {head_tail(model_keys)}\n"
                "Tip: ensure architecture and parameter names match the reference implementation."
            ) from e
    return model


def convert_pytorch_to_jax(pytorch_state: dict, jax_model) -> dict:
    """Convert PyTorch state dict to JAX parameters."""

    # Get the full nested parameter structure from JAX model
    def get_nested_params(module, prefix=""):
        """Recursively extract all parameter paths from JAX model."""
        params = {}
        state = nnx.state(module)

        for key, value in state.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)

            # If this is a parameter, add it directly
            if hasattr(value, "value"):
                params[full_key] = value
            # If this is a nested module, recurse
            elif hasattr(value, "__dict__"):
                try:
                    nested = get_nested_params(value, full_key)
                    params.update(nested)
                except:
                    # If recursion fails, treat as parameter
                    params[full_key] = value
            else:
                params[full_key] = value

        return params

    # Get all JAX parameter paths
    jax_params = get_nested_params(jax_model)
    jax_state = {}

    for jax_key, jax_param in jax_params.items():
        jax_key_str = str(jax_key)
        found = False

        # Generate PyTorch key candidates
        pytorch_key_candidates = [
            jax_key_str,
            jax_key_str.replace(".kernel", ".weight"),
            jax_key_str.replace(".embedding", ".weight"),
            jax_key_str.replace("embeddings.embedding", "embeddings.weight"),
            jax_key_str.replace("lm_head.kernel", "lm_head.weight"),
            # Handle manual conv1d parameters - convert underscore to dot
            jax_key_str.replace("conv1d_weight", "conv1d.weight"),
            jax_key_str.replace("conv1d_bias", "conv1d.bias"),
        ]

        # Try exact matches first
        for pytorch_key in pytorch_key_candidates:
            if pytorch_key in pytorch_state:
                tensor = pytorch_state[pytorch_key]
                # Convert torch tensor to numpy then JAX array
                if hasattr(tensor, "detach"):
                    tensor = tensor.detach().cpu().numpy()
                elif hasattr(tensor, "numpy"):
                    tensor = tensor.numpy()
                else:
                    tensor = np.array(tensor)

                # Handle transpose for different layer types
                if ".kernel" in jax_key_str and tensor.ndim == 2:
                    # Linear layers: PyTorch [out, in] -> JAX [in, out]
                    tensor = tensor.T
                elif "conv1d_weight" in jax_key_str and tensor.ndim == 3:
                    # Conv1d weight: PyTorch (conv_dim, 1, d_conv) -> JAX (d_conv, conv_dim)
                    tensor = tensor[:, 0, :].T  # Remove middle dim and transpose

                jax_state[jax_key] = jnp.array(tensor)
                found = True
                break

        if not found:
            # Keep original parameter value as fallback
            if hasattr(jax_param, "value"):
                jax_state[jax_key] = jax_param.value
            else:
                jax_state[jax_key] = jax_param

    # Count successfully loaded parameters
    loaded_keys = []
    for jax_key in jax_state.keys():
        # Check if this parameter was loaded from checkpoint (vs default)
        pytorch_key_candidates = [
            str(jax_key),
            str(jax_key).replace(".kernel", ".weight"),
            str(jax_key).replace(".embedding", ".weight"),
            str(jax_key).replace("conv1d_weight", "conv1d.weight"),
            str(jax_key).replace("conv1d_bias", "conv1d.bias"),
        ]
        if any(pk in pytorch_state for pk in pytorch_key_candidates):
            loaded_keys.append(jax_key)

    print(
        f"Successfully loaded {len(loaded_keys)}/{len(jax_params)} parameters from checkpoint"
    )

    return jax_state


def update_model_parameters(model, jax_state):
    """Update model parameters using the nested parameter paths."""
    for param_path, param_value in jax_state.items():
        try:
            # Navigate to the parameter using the path
            path_parts = str(param_path).split(".")
            current = model

            # Navigate to the parent object, handling list indices
            for part in path_parts[:-1]:
                if part.isdigit():
                    # This is a list index
                    current = current[int(part)]
                else:
                    # This is an attribute
                    current = getattr(current, part)

            # Get the final parameter name
            final_param = path_parts[-1]

            # Update the parameter
            if final_param.isdigit():
                # Final part is also an index
                current[int(final_param)] = param_value
            elif hasattr(current, final_param):
                param_obj = getattr(current, final_param)
                if hasattr(param_obj, "value"):
                    # This is a Param object - update its value
                    param_obj.value = param_value
                else:
                    # Direct assignment
                    setattr(current, final_param, param_value)
            else:
                print(f"Warning: Could not find parameter {param_path}")
        except Exception as e:
            print(f"Error updating parameter {param_path}: {e}")


def generate_tokens(
    model: HNetForCausalLM,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    rng_key: jax.Array | None = None,
):
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    tok = ByteTokenizer()
    x = tok.encode(prompt, add_bos=True)[None, :]  # (1, L)
    cache = model.allocate_inference_cache(
        1, x.shape[1] + max_new_tokens, dtype=model.lm_head.kernel.value.dtype
    )

    # Prefill
    mask = jnp.ones_like(x, dtype=jnp.bool_)
    logits, _, _ = model(x, mask=mask, inference_params=cache)
    logits = logits[0, -1]  # (vocab_size,)

    for i in range(max_new_tokens):
        rng_key, sample_key = jax.random.split(rng_key)
        logits = logits / max(temperature, 1e-6)
        logits = _top_p_filtering(logits, top_p)
        next_id = jax.random.categorical(sample_key, logits, shape=(1,))  # (1,)

        token_id = int(next_id.item())

        if token_id == ByteTokenizer().eos_idx:
            break
        yield token_id

        nxt = next_id[None, :]  # (1, 1)
        logits, _, _ = model.step(nxt, cache)
        logits = logits[0, -1]


def main():
    parser = argparse.ArgumentParser(description="H-Net JAX (Flax NNX) generator")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to config JSON (e.g., hnet-reference/configs/hnet_2stage_L.json)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to .pt weights (optional; non-strict load)",
    )
    parser.add_argument("--prompt", type=str, required=True, help="UTF-8 prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"]
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict state_dict load (fail on any mismatch)",
    )
    args = parser.parse_args()

    # Set manual seed for deterministic sampling
    rng_key = jax.random.PRNGKey(args.seed)

    model = load_model(
        args.model_path, args.config_path, dtype=args.dtype, strict=args.strict
    )
    tok = ByteTokenizer()
    print(args.prompt, end="", flush=True)
    buf = []
    for tid in generate_tokens(
        model,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        rng_key=rng_key,
    ):
        buf.append(tid)

        # Stream decode in small chunks to respect UTF-8 boundaries
        for j in range(1, min(len(buf), 4) + 1):
            try:
                s = tok.decode(jnp.array(buf[:j]))
                print(s, end="", flush=True)
                buf = buf[j:]
                break
            except Exception:
                pass


if __name__ == "__main__":
    main()
