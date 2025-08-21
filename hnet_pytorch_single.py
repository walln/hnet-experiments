"""
Pure PyTorch single-file reimplementation of H-Net (Dynamic Chunking for End-to-End
Hierarchical Sequence Modeling), based on the reference architecture in
./hnet-reference, but without any Triton or FlashAttention dependencies.

Key design points:
- Maintains the same high-level module structure (Routing -> Chunk -> Inner -> Dechunk -> Residual).
- Implements EMA-based dechunking in pure PyTorch.
- Provides both batch (masked) and packed (cu_seqlens) code paths where practical.
- Uses a simple causal multi-head attention and a lightweight SSM-like mixer (conv + gating)
  to stand in for Mamba2 while keeping everything PyTorch-native.

This file exposes:
- Config dataclasses: AttnConfig, SSMConfig, HNetConfig
- Core modules: Isotropic, RoutingModule/ChunkLayer/DeChunkLayer, HNet
- Optional LM wrapper: HNetForCausalLM

Notes:
- This is intended for correctness and structure parity, not kernel-level speed.
- The SSM mixer is a simplified PyTorch substitute; feel free to swap it with a faster or
  more faithful implementation as long as it stays Triton-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================
# Utilities and Configs
# ============================


def get_seq_idx(cu_seqlens: torch.Tensor, device=None) -> torch.Tensor:
    """Return sequence indices for packed representation.
    cu_seqlens: (B+1,) cumulative lengths
    Returns: (1, T) int tensor mapping each token to its batch index.
    """
    seq_idx = torch.zeros(cu_seqlens[-1], dtype=torch.long, device=device)
    seq_idx[cu_seqlens[:-1]] = 1
    seq_idx = (torch.cumsum(seq_idx, dim=0) - 1).unsqueeze(0).int()
    return seq_idx


def get_stage_cfg(cfg, stage_idx: int):
    return {k: (v[stage_idx] if isinstance(v, list) else v) for k, v in asdict(cfg).items()}


@dataclass
class AttnConfig:
    num_heads: List[int] = field(default_factory=list)
    rotary_emb_dim: List[int] = field(default_factory=list)
    window_size: List[int] = field(default_factory=list)


@dataclass
class SSMConfig:
    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256


@dataclass
class HNetConfig:
    arch_layout: List[Union[str, List]] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    d_intermediate: List[int] = field(default_factory=list)
    vocab_size: int = 256
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    tie_embeddings: bool = False


# ============================
# Norms and simple building blocks
# ============================


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device, dtype=dtype))

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ):
        if residual is not None:
            if residual_in_fp32:
                residual = residual.to(torch.float32)
            x = (x + residual).to(x.dtype)
        if prenorm:
            normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight
            return normed, x
        else:
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_intermediate: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.fc1 = nn.Linear(d_model, 2 * d_intermediate, bias=False, **factory_kwargs)
        self.fc2 = nn.Linear(d_intermediate, d_model, bias=False, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        w_dtype = self.fc1.weight.dtype
        if x_dtype != w_dtype:
            x = x.to(w_dtype)
        gate_up = self.fc1(x)
        gate, up = gate_up.chunk(2, dim=-1)
        out = self.fc2(up * F.silu(gate))
        return out.to(x_dtype)


# ============================
# Simple PyTorch Causal MHA (no FlashAttention)
# ============================


class CausalMHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        qkv_proj_bias: bool = False,
        out_proj_bias: bool = False,
        rotary_emb_dim: int = 0,
        window_size: int = -1,
        device=None,
        dtype=None,
        layer_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.layer_idx = layer_idx
        self.rotary_emb_dim = int(rotary_emb_dim) if rotary_emb_dim is not None else 0
        self.rotary_base = 10000.0
        self.window_size = int(window_size) if window_size is not None else -1
        factory_kwargs = {"device": device, "dtype": dtype}

        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=qkv_proj_bias, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias, **factory_kwargs)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.size(0), x.size(2), self.d_model)

    def _causal_attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # q, k, v: (B, H, L, D)
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, L, L)
        L = q.size(-2)
        causal = torch.triu(torch.ones(L, L, device=q.device, dtype=torch.bool), diagonal=1)
        attn_scores.masked_fill_(causal, float("-inf"))
        if self.window_size is not None and self.window_size > 0:
            # Disallow keys older than window_size
            idx = torch.arange(L, device=q.device)
            # True where j < i - (W-1)
            win_mask = idx[None, :] < (idx[:, None] - (self.window_size - 1))
            attn_scores.masked_fill_(win_mask, float("-inf"))
        if attn_mask is not None:
            # attn_mask: (B, L) with True for valid tokens
            mask = attn_mask[:, None, None, :].expand(-1, self.num_heads, L, -1)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(attn_scores, dim=-1)
        return torch.matmul(attn, v)

    def _rotary_cos_sin(self, L: int, offset: int, device, dtype):
        """Compute rotary cos/sin tables for sequence length L starting at position offset.
        Returns cos, sin with shape (L, rotary_emb_dim//2) in given dtype.
        """
        ro_dim = self.rotary_emb_dim
        assert ro_dim % 2 == 0 and ro_dim <= self.head_dim
        inv_freq = 1.0 / (
            self.rotary_base ** (torch.arange(0, ro_dim, 2, device=device, dtype=torch.float32) / ro_dim)
        )
        # positions [offset, offset+L-1]
        t = torch.arange(offset, offset + L, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (L, ro_dim/2)
        cos = torch.cos(freqs).to(dtype)
        sin = torch.sin(freqs).to(dtype)
        return cos, sin

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """Apply rotary embeddings on first rotary_emb_dim of q and k (NeoX style, non-interleaved).
        q, k: (B, H, L, Dh)
        """
        if self.rotary_emb_dim <= 0:
            return q, k
        B, H, L, Dh = q.shape
        ro_dim = self.rotary_emb_dim
        cos, sin = self._rotary_cos_sin(L, offset, q.device, q.dtype)
        # shape to (1,1,L,ro_dim/2) for broadcast
        cos = cos.view(1, 1, L, -1)
        sin = sin.view(1, 1, L, -1)
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)  # (..., ro_dim/2)
            xr1 = x1 * cos - x2 * sin
            xr2 = x2 * cos + x1 * sin
            return torch.cat([xr1, xr2], dim=-1)
        def apply(x):
            x_ro = x[..., :ro_dim]
            x_rest = x[..., ro_dim:]
            x_ro_new = rotate_half(x_ro)
            return torch.cat([x_ro_new, x_rest], dim=-1)
        return apply(q), apply(k)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        attn_mask: Optional[torch.Tensor] = None,
        inference_params=None,
    ) -> torch.Tensor:
        # Support either packed (T, D) with cu_seqlens+max_seqlen or padded (B, L, D) with attn_mask.
        packed = cu_seqlens is not None and max_seqlen is not None
        if packed:
            # Unpack to padded then repack after attention
            T, D = x.shape
            B = cu_seqlens.numel() - 1
            Lmax = int(max_seqlen)
            device = x.device
            padded = torch.zeros(B, Lmax, D, device=device, dtype=x.dtype)
            mask = torch.zeros(B, Lmax, device=device, dtype=torch.bool)
            for b in range(B):
                s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
                L = e - s
                padded[b, :L] = x[s:e]
                mask[b, :L] = True
            out = self._forward_padded(padded, attn_mask=mask, inference_params=None)
            # Repack
            parts = []
            for b in range(B):
                L = int(mask[b].sum())
                parts.append(out[b, :L])
            return torch.cat(parts, dim=0)
        else:
            assert x.dim() == 3 and attn_mask is not None, "Provide attn_mask for padded input"
            return self._forward_padded(x, attn_mask=attn_mask, inference_params=inference_params)

    def _forward_padded(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor], inference_params=None) -> torch.Tensor:
        B, L, _ = x.shape
        x_dtype = x.dtype
        w_dtype = self.Wqkv.weight.dtype
        if x_dtype != w_dtype:
            x = x.to(w_dtype)
        qkv = self.Wqkv(x)  # (B, L, 3D)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
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
            assert getattr(self, "layer_idx", None) is not None, "layer_idx required for KV cache"
            # lazy-allocate cache
            kv_cache = inference_params.key_value_memory_dict.get(self.layer_idx)
            if kv_cache is None:
                dtype = self.out_proj.weight.dtype
                device = self.out_proj.weight.device
                kv_cache = torch.empty(
                    inference_params.max_batch_size,
                    inference_params.max_seqlen,
                    2,
                    self.num_heads,
                    self.head_dim,
                    dtype=dtype,
                    device=device,
                )
                inference_params.key_value_memory_dict[self.layer_idx] = kv_cache
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + B
            seq_start = inference_params.seqlen_offset
            seq_end = seq_start + L
            kv_cur = torch.stack([k.transpose(1, 2), v.transpose(1, 2)], dim=2)  # (B, L, 2, H, Dh)
            kv_cache[batch_start:batch_end, seq_start:seq_end, ...] = kv_cur
        return out.to(x_dtype)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        # Not implementing KV cache here; return placeholder
        return None

    def step(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        # Incremental decoding with KV cache and rotary.
        assert x.dim() == 3 and x.size(1) == 1
        B, L, _ = x.shape
        x_dtype = x.dtype
        w_dtype = self.Wqkv.weight.dtype
        if x_dtype != w_dtype:
            x = x.to(w_dtype)
        qkv = self.Wqkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = self._split_heads(q)  # (B, H, 1, Dh)
        k = self._split_heads(k)
        v = self._split_heads(v)
        if self.rotary_emb_dim > 0:
            offset = int(getattr(inference_params, "seqlen_offset", 0) or 0)
            q, k = self._apply_rotary(q, k, offset=offset)
        # Lazy-allocate / fetch KV cache
        assert getattr(self, "layer_idx", None) is not None, "layer_idx required for KV cache"
        kv_cache = inference_params.key_value_memory_dict.get(self.layer_idx)
        if kv_cache is None:
            dtype = self.out_proj.weight.dtype
            device = self.out_proj.weight.device
            kv_cache = torch.empty(
                inference_params.max_batch_size,
                inference_params.max_seqlen,
                2,
                self.num_heads,
                self.head_dim,
                dtype=dtype,
                device=device,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = kv_cache
        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + B
        seq_start = inference_params.seqlen_offset
        seq_end = seq_start + 1
        # Write current K/V
        kv_cache[batch_start:batch_end, seq_start:seq_end, 0, ...] = k.transpose(1, 2)  # K
        kv_cache[batch_start:batch_end, seq_start:seq_end, 1, ...] = v.transpose(1, 2)  # V
        # Read past K/V up to seq_end, with optional windowing
        start = max(0, seq_end - self.window_size) if (self.window_size is not None and self.window_size > 0) else 0
        K_all = kv_cache[batch_start:batch_end, start:seq_end, 0, ...].transpose(1, 2)  # (B, H, S, Dh)
        V_all = kv_cache[batch_start:batch_end, start:seq_end, 1, ...].transpose(1, 2)
        # Attention over full prefix
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, K_all.transpose(-2, -1)) * scale  # (B, H, 1, S)
        attn = torch.softmax(attn_scores, dim=-1)
        ctx = torch.matmul(attn, V_all)  # (B, H, 1, Dh)
        ctx = self._merge_heads(ctx)  # (B, 1, D)
        out = self.out_proj(ctx)
        return out.to(x_dtype)


# ============================
# Lightweight SSM-like Mixer (PyTorch-only stand-in for Mamba2)
# ============================


class PyTorchSSM(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        device=None,
        dtype=None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.headdim = headdim
        self.nheads = self.d_inner // self.headdim
        self.layer_idx = layer_idx
        factory_kwargs = {"device": device, "dtype": dtype}

        # Match Mamba2 parameterization to align with checkpoints
        # Order: [z (d_inner), x (d_inner), B (d_state), C (d_state), dt (nheads)]
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False, **factory_kwargs)
        # Depthwise 1D conv over sequence for dynamics on concat(x, B, C)
        conv_dim = self.d_inner + 2 * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            bias=True,
            **factory_kwargs,
        )
        self.norm = RMSNorm(self.d_inner, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory_kwargs)

        # Extra parameters present in checkpoints (not fully used but included for loading)
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads, **factory_kwargs))
        self.A_log = nn.Parameter(torch.zeros(self.nheads, **factory_kwargs))
        self.D = nn.Parameter(torch.ones(self.nheads, **factory_kwargs))

    def forward(self, x: torch.Tensor, seq_idx: Optional[torch.Tensor] = None, inference_params=None) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        x_dtype = x.dtype
        w_dtype = self.in_proj.weight.dtype
        if x_dtype != w_dtype:
            x = x.to(w_dtype)
        zxbcdt = self.in_proj(x)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads],
            dim=-1,
        )
        # Depthwise conv over concat(x, B, C)
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2))
        if xBC.shape[1] != L:
            xBC = xBC[:, :L, :]
        x_part, B_part, C_part = torch.split(xBC, [self.d_inner, self.d_state, self.d_state], dim=-1)

        # Heads view
        x_heads = x_part.view(B, L, self.nheads, self.headdim)
        B_heads = B_part.unsqueeze(2).expand(-1, -1, self.nheads, -1)
        C_heads = C_part.unsqueeze(2).expand(-1, -1, self.nheads, -1)
        dt = F.softplus(dt + self.dt_bias)  # (B, L, H)
        A = -torch.exp(self.A_log)  # (H,)

        # Stateful scan (naive O(L * H * P * S))
        state = torch.zeros(B, self.nheads, self.headdim, self.d_state, device=x.device, dtype=w_dtype)
        y_acc = torch.zeros(B, L, self.nheads, self.headdim, device=x.device, dtype=w_dtype)
        for t in range(L):
            dt_t = dt[:, t]  # (B, H)
            decay = torch.exp(A.unsqueeze(0) * dt_t)  # (B, H)
            state = state * decay.unsqueeze(-1).unsqueeze(-1)
            x_t = x_heads[:, t]  # (B, H, P)
            B_t = B_heads[:, t]  # (B, H, S)
            state = state + (x_t.unsqueeze(-1) * B_t.unsqueeze(-2)) * dt_t.unsqueeze(-1).unsqueeze(-1)
            C_t = C_heads[:, t]
            y_t = (state * C_t.unsqueeze(-2)).sum(dim=-1)  # (B, H, P)
            y_acc[:, t] = y_t

        y = y_acc.reshape(B, L, self.d_inner)
        D_full = self.D.unsqueeze(-1).expand(self.nheads, self.headdim).reshape(1, 1, -1)
        y = y + x_part * D_full
        y = self.norm(y * F.silu(z))
        out = self.out_proj(y)
        return out.to(x_dtype)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        return None

    def step(self, x: torch.Tensor, inference_params=None) -> torch.Tensor:
        # x: (B, 1, D)
        assert x.shape[1] == 1
        B = x.shape[0]
        x_dtype = x.dtype
        w_dtype = self.in_proj.weight.dtype
        if x_dtype != w_dtype:
            x = x.to(w_dtype)
        zxbcdt = self.in_proj(x)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads],
            dim=-1,
        )
        # Cache states
        layer_idx = getattr(self, 'layer_idx', None)
        assert layer_idx is not None and inference_params is not None
        cache = inference_params.key_value_memory_dict
        conv_dim = self.d_inner + 2 * self.d_state
        if layer_idx not in cache:
            conv_state = torch.zeros(B, conv_dim, self.conv1d.kernel_size[0], device=x.device, dtype=w_dtype)
            ssm_state = torch.zeros(B, self.nheads, self.headdim, self.d_state, device=x.device, dtype=w_dtype)
        else:
            conv_state, ssm_state = cache[layer_idx]
            if conv_state.size(0) != B:
                conv_state = conv_state[:B].contiguous()
                ssm_state = ssm_state[:B].contiguous()

        # Depthwise conv via state
        xBC_t = xBC[:, 0]  # (B, conv_dim)
        conv_state = torch.cat([conv_state[:, :, 1:], xBC_t.unsqueeze(-1)], dim=-1)
        w = self.conv1d.weight.squeeze(1)  # (conv_dim, k)
        conv_out = torch.einsum('bck,ck->bc', conv_state, w)
        if self.conv1d.bias is not None:
            conv_out = conv_out + self.conv1d.bias
        conv_out = F.silu(conv_out)
        x_part, B_part, C_part = torch.split(conv_out, [self.d_inner, self.d_state, self.d_state], dim=-1)
        x_heads = x_part.view(B, self.nheads, self.headdim)
        B_t = B_part.unsqueeze(1).expand(-1, self.nheads, -1)
        C_t = C_part.unsqueeze(1).expand(-1, self.nheads, -1)
        dt_t = F.softplus(dt[:, 0] + self.dt_bias)  # (B, H)
        A = -torch.exp(self.A_log)
        decay = torch.exp(A.unsqueeze(0) * dt_t)
        ssm_state = ssm_state * decay.unsqueeze(-1).unsqueeze(-1)
        ssm_state = ssm_state + (x_heads.unsqueeze(-1) * B_t.unsqueeze(-2)) * dt_t.unsqueeze(-1).unsqueeze(-1)
        y_t = (ssm_state * C_t.unsqueeze(-2)).sum(dim=-1)  # (B, H, P)
        y = y_t.reshape(B, 1, self.d_inner)
        D_full = self.D.unsqueeze(-1).expand(self.nheads, self.headdim).reshape(1, 1, -1)
        y = y + x_part.view(B, 1, -1) * D_full
        y = self.norm(y * F.silu(z))
        out = self.out_proj(y)
        cache[layer_idx] = [conv_state, ssm_state]
        return out.to(x_dtype)


# ============================
# Blocks and Isotropic stack
# ============================


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        mixer: nn.Module,
        mlp: Optional[nn.Module],
        residual_in_fp32: bool = True,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm1 = RMSNorm(d_model)
        self.mixer = mixer
        self.mlp = mlp
        if self.mlp is not None:
            self.norm2 = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inference_params=None,
        mixer_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, residual = self.norm1(x, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32)
        mixer_kwargs = mixer_kwargs or {}
        x = self.mixer(x, **mixer_kwargs, inference_params=inference_params)
        if self.mlp is not None:
            x, residual = self.norm2(x, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32)
            x = self.mlp(x)
        return x, residual

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        return None

    def step(self, x: torch.Tensor, inference_params, residual: Optional[torch.Tensor] = None):
        x, residual = self.norm1(x, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32)
        x = self.mixer.step(x, inference_params)
        if self.mlp is not None:
            x, residual = self.norm2(x, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32)
            x = self.mlp(x)
        return x, residual


@dataclass
class IsotropicInferenceParams:
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None

    def reset(self, max_seqlen: int, max_batch_size: int):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        self.key_value_memory_dict.clear()


class Isotropic(nn.Module):
    def __init__(
        self,
        config: HNetConfig,
        pos_idx: int,
        stage_idx: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
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
        import re

        layout_parse = re.findall(r"([mMtT])(\d+)", arch_layout)
        layers: List[Block] = []
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
                        **factory_kwargs,
                        layer_idx=layer_idx,
                    )
                elif arch in ("m", "M"):
                    mixer = PyTorchSSM(
                        self.d_model,
                        d_state=self.ssm_cfg.get("d_state", 128),
                        d_conv=self.ssm_cfg.get("d_conv", 4),
                        expand=self.ssm_cfg.get("expand", 2),
                        **factory_kwargs,
                        layer_idx=layer_idx,
                    )
                else:
                    raise NotImplementedError

                if arch in ("T", "M"):
                    mlp = SwiGLU(self.d_model, d_intermediate=config.d_intermediate[self.stage_idx], **factory_kwargs)
                else:
                    mlp = None
                layers.append(Block(self.d_model, mixer, mlp))
                layer_idx += 1

        self.layers = nn.ModuleList(layers)
        self.rmsnorm = RMSNorm(self.d_model, eps=1e-5, **factory_kwargs)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        return IsotropicInferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        inference_params: Optional[IsotropicInferenceParams] = None,
        **mixer_kwargs,
    ) -> torch.Tensor:
        packed = cu_seqlens is not None and max_seqlen is not None and mask is None
        if packed:
            # Unpack to padded input
            T, D = hidden_states.shape
            B = cu_seqlens.numel() - 1
            Lmax = int(max_seqlen)
            device = hidden_states.device
            x = torch.zeros(B, Lmax, D, device=device, dtype=hidden_states.dtype)
            mask_local = torch.zeros(B, Lmax, device=device, dtype=torch.bool)
            for b in range(B):
                s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
                x[b, : e - s] = hidden_states[s:e]
                mask_local[b, : e - s] = True
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
            x, residual = layer(x, residual=residual, inference_params=inference_params, mixer_kwargs=mix_kwargs)

        x = self.rmsnorm(x, residual=residual, prenorm=False, residual_in_fp32=True)

        if packed:
            outs = []
            for b in range(B):
                L = int(mask_local[b].sum())
                outs.append(x[b, :L])
            x = torch.cat(outs, dim=0)

        if inference_params is not None:
            # Follow reference: assert batch size 1 and padded path when tracking seqlen_offset
            assert mask is not None, "Mask must be provided if inference_params is provided"
            assert mask.shape[0] == 1, "seqlen_offset handling assumes batch size 1"
            assert x.dim() == 3, "Inference with inference_params expects padded (B, L, D)"
            inference_params.seqlen_offset += x.shape[1]

        return x

    def step(self, hidden_states: torch.Tensor, inference_params: IsotropicInferenceParams):
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


@dataclass
class RoutingModuleOutput:
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor


@dataclass
class RoutingModuleState:
    has_seen_tokens: torch.Tensor  # (B,)
    last_hidden_state: torch.Tensor  # (B, D)


@dataclass
class DeChunkState:
    last_value: torch.Tensor  # (B, D)


class RoutingModule(nn.Module):
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model, device=self.q_proj_layer.weight.device))
            self.k_proj_layer.weight.copy_(torch.eye(d_model, device=self.k_proj_layer.weight.device))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, device, dtype=None):
        return RoutingModuleState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        inference_params: Optional[RoutingModuleState] = None,
    ) -> RoutingModuleOutput:
        assert (mask is not None) or (cu_seqlens is not None), "Provide mask or cu_seqlens"
        if inference_params is not None:
            # Match reference behavior: prefill requires mask and unseen state
            assert mask is not None, "Mask must be provided if inference_params is provided"
            assert (
                (~inference_params.has_seen_tokens).all()
            ), "Cannot have seen tokens when inference_params is provided"
            # Not supporting packed + inference_params
            assert cu_seqlens is None, "Packed mode with inference_params is not supported"

        if cu_seqlens is not None:
            # Treat as single batch for computation convenience
            hs = hidden_states.unsqueeze(0)  # (1, T, D)
        else:
            hs = hidden_states  # (B, L, D)

        # Ensure dtype matches projection weights
        hs_dtype = hs.dtype
        w_dtype = self.q_proj_layer.weight.dtype
        if hs_dtype != w_dtype:
            hs = hs.to(w_dtype)

        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hs[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hs[:, 1:]), dim=-1),
        )
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        # Force first token as boundary
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), value=PAD_PROB)
        if cu_seqlens is not None:
            boundary_prob = boundary_prob.squeeze(0)
            boundary_prob[cu_seqlens[:-1]] = PAD_PROB
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)
        selected_idx = torch.argmax(boundary_prob, dim=-1)
        boundary_mask = selected_idx == 1
        if mask is not None:
            boundary_mask = boundary_mask & mask

        if inference_params is not None:
            # Update prefill state so that step() has correct previous token context
            has_mask = mask.any(dim=-1)
            inference_params.has_seen_tokens.copy_(has_mask | inference_params.has_seen_tokens)
            last_mask = torch.clamp(mask.sum(dim=-1) - 1, min=0)
            idx_b = torch.arange(hidden_states.shape[0], device=hidden_states.device)
            last_h = hidden_states[idx_b, last_mask]
            inference_params.last_hidden_state.copy_(
                torch.where(
                    has_mask.unsqueeze(-1),
                    last_h,
                    inference_params.last_hidden_state,
                )
            )
        selected_probs = boundary_prob.gather(dim=-1, index=selected_idx.unsqueeze(-1))
        return RoutingModuleOutput(boundary_prob=boundary_prob, boundary_mask=boundary_mask, selected_probs=selected_probs)

    def step(self, hidden_states: torch.Tensor, inference_params: RoutingModuleState) -> RoutingModuleOutput:
        # hidden_states: (B, 1, D)
        hs = hidden_states.squeeze(1)
        # Align dtype to projection weights
        hs_dtype = hs.dtype
        w_dtype = self.q_proj_layer.weight.dtype
        if hs_dtype != w_dtype:
            hs = hs.to(w_dtype)
        cos_sim = torch.einsum(
            "b d, b d -> b",
            F.normalize(self.q_proj_layer(inference_params.last_hidden_state), dim=-1),
            F.normalize(self.k_proj_layer(hs), dim=-1),
        )
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        inference_params.last_hidden_state.copy_(hs)
        boundary_prob = torch.where(inference_params.has_seen_tokens, boundary_prob, torch.ones_like(boundary_prob))
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)
        inference_params.has_seen_tokens.copy_(torch.ones_like(inference_params.has_seen_tokens))
        selected_idx = torch.argmax(boundary_prob, dim=-1)
        boundary_mask = selected_idx == 1
        selected_probs = boundary_prob.gather(dim=-1, index=selected_idx.unsqueeze(-1))
        return RoutingModuleOutput(boundary_prob=boundary_prob, boundary_mask=boundary_mask, selected_probs=selected_probs)


class ChunkLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int], Optional[torch.Tensor]]:
        assert (mask is not None) or (cu_seqlens is not None), "Provide mask or cu_seqlens"
        if cu_seqlens is not None:
            next_hidden_states = hidden_states[boundary_mask]
            next_cu_seqlens = F.pad(boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0))
            next_max_seqlen = int((next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max())
            next_mask = None
        else:
            next_cu_seqlens = None
            num_tokens = boundary_mask.sum(dim=-1)
            next_max_seqlen = int(num_tokens.max())
            device = hidden_states.device
            L = hidden_states.shape[1]
            token_idx = torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
            seq_sorted_indices = torch.argsort(token_idx, dim=1)
            next_hidden_states = torch.gather(
                hidden_states,
                dim=1,
                index=seq_sorted_indices[:, :next_max_seqlen, None].expand(-1, -1, hidden_states.shape[-1]),
            )
            next_mask = (torch.arange(next_max_seqlen, device=device)[None, :] < num_tokens[:, None])
            next_max_seqlen = None
        return next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask

    def step(self, hidden_states: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
        return hidden_states[boundary_mask]


class DeChunkLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, device, dtype=None):
        return DeChunkState(last_value=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype))

    @staticmethod
    def _ema_sequence(hidden_seq: torch.Tensor, p_seq: torch.Tensor) -> torch.Tensor:
        """Compute EMA over a sequence of selected hidden states.
        hidden_seq: (K, D), p_seq: (K,)
        Returns: (K, D) EMA states.
        """
        K, D = hidden_seq.shape
        out = torch.zeros_like(hidden_seq)
        h = torch.zeros(D, device=hidden_seq.device, dtype=hidden_seq.dtype)
        for k in range(K):
            pk = p_seq[k]
            h = (1 - pk) * h + pk * hidden_seq[k]
            out[k] = h
        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_prob: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        inference_params: Optional[DeChunkState] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inference_params is None:
            assert mask is not None, "Mask must be provided in prefill"
            assert boundary_mask[:, 0].all() if boundary_mask.dim() == 2 else boundary_mask[0].item() == 1

        # Extract p probabilities (B, L) or (T,)
        if boundary_prob.shape[-1] == 2:
            p_full = boundary_prob[..., -1].float().clamp(1e-4, 1 - 1e-4)
        else:
            p_full = boundary_prob.float().clamp(1e-4, 1 - 1e-4)

        original_dtype = hidden_states.dtype

        if cu_seqlens is not None:
            # Packed path: hidden_states are selected boundary states concatenated across batch
            # boundary_mask and p_full correspond to original tokens across batch
            device = hidden_states.device
            T_total = p_full.shape[0]
            B = cu_seqlens.numel() - 1

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
            sel_cu = torch.tensor(sel_cu, device=device, dtype=torch.long)

            # Compute EMA per sample over selected hidden states
            ema_selected = torch.zeros_like(hidden_states)
            for b in range(B):
                ks, ke = int(sel_cu[b]), int(sel_cu[b + 1])
                if ke > ks:
                    ema_selected[ks:ke] = self._ema_sequence(hidden_states[ks:ke], selected_p[ks:ke])

            # Map EMA back to full token positions via plug-back index = cumsum(boundary_mask)-1
            plug_back_idx = boundary_mask.cumsum(dim=0) - 1  # (T_total,)
            # Gather from ema_selected using flat mapping
            out_full = torch.gather(
                ema_selected,
                dim=0,
                index=plug_back_idx.clamp(min=0).unsqueeze(-1).expand(-1, self.d_model),
            )
            return out_full.to(original_dtype)
        else:
            # Unpacked (B, L, D)
            B, L = boundary_mask.shape
            # Compute EMA along selected tokens for each sample
            # First, sort tokens to bring boundaries to front like reference
            device = hidden_states.device
            token_idx = torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            # Number of selected (boundary) tokens per sample
            num_tokens = boundary_mask.sum(dim=-1)  # (B,)
            # hidden_states is already chunked/selected with width Mmax from chunk layer
            M = hidden_states.shape[1]
            selected_hidden = hidden_states  # (B, M, D)
            # Align p to the same chunked order and truncate to M
            p_sorted = torch.gather(p_full, dim=1, index=seq_sorted_indices[:, :M])  # (B, M)

            ema_selected = torch.zeros_like(selected_hidden)
            for b in range(B):
                m = int(num_tokens[b])
                if m > 0:
                    ema_selected[b, :m] = self._ema_sequence(selected_hidden[b, :m], p_sorted[b, :m])

            plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L), -1 where not boundary yet
            out = torch.gather(
                ema_selected,
                dim=1,
                index=plug_back_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.d_model),
            )
            if inference_params is not None:
                inference_params.last_value.copy_(out[:, -1])
            return out.to(original_dtype)

    def step(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_prob: torch.Tensor,
        inference_params: DeChunkState,
    ) -> torch.Tensor:
        # hidden_states: (B', 1, D) for selected tokens, boundary_mask: (B,), boundary_prob: (B, 2)
        B = boundary_mask.shape[0]
        D = hidden_states.shape[-1]
        p = torch.zeros(B, device=hidden_states.device, dtype=hidden_states.dtype)
        bp = boundary_prob.to(p.dtype)
        p[boundary_mask] = bp[boundary_mask, -1].clamp(1e-4, 1 - 1e-4)
        current = torch.zeros(B, D, device=hidden_states.device, dtype=hidden_states.dtype)
        if hidden_states.numel() > 0:
            current[boundary_mask] = hidden_states.squeeze(1)
        result = p.unsqueeze(-1) * current + (1 - p).unsqueeze(-1) * inference_params.last_value
        inference_params.last_value.copy_(result)
        return result.unsqueeze(1)


# ============================
# H-Net wrapper (recursive stages)
# ============================


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_func(x):
    return STE.apply(x)


@dataclass
class HNetState:
    encoder_state: Optional[IsotropicInferenceParams] = None
    routing_module_state: Optional[RoutingModuleState] = None
    main_network_state: Optional[Union["HNetState", IsotropicInferenceParams]] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[IsotropicInferenceParams] = None


class HNet(nn.Module):
    def __init__(self, config: HNetConfig, stage_idx: int, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
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

        for _name, _layout in zip(sub_model_names, arch_layout):
            if self.is_innermost or _name in ("encoder", "decoder"):
                SubModel = Isotropic
                _stage_idx = stage_idx
                _pos_idx = 0 if (_name == "encoder" or self.is_innermost) else 2
                _pos_idx_dict = {"pos_idx": _pos_idx}
            else:
                SubModel = HNet
                _stage_idx = stage_idx + 1
                _pos_idx_dict = {}
            _sub_model = SubModel(config=config, stage_idx=_stage_idx, **_pos_idx_dict, **factory_kwargs)
            self.add_module(_name, _sub_model)

        if not self.is_innermost:
            self.routing_module = RoutingModule(self.d_model, **factory_kwargs)
            self.chunk_layer = ChunkLayer()
            self.dechunk_layer = DeChunkLayer(self.d_model)
            # Residual in fp32
            self.residual_proj = nn.Linear(self.d_model, self.d_model, device=device, dtype=torch.float32)
            nn.init.zeros_(self.residual_proj.weight)
            self.residual_proj.weight._no_reinit = True
            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual

        if stage_idx > 0 and self.d_model - config.d_model[stage_idx - 1] > 0:
            self.pad_dimension = nn.Parameter(torch.zeros(self.d_model - config.d_model[stage_idx - 1], **factory_kwargs))
        else:
            self.pad_dimension = None

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        if self.is_innermost:
            return HNetState(main_network_state=self.main_network.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype))
        else:
            device = self.residual_proj.weight.device
            return HNetState(
                encoder_state=self.encoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype),
                routing_module_state=self.routing_module.allocate_inference_cache(batch_size, max_seqlen, device, dtype=dtype),
                main_network_state=self.main_network.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype),
                dechunk_state=self.dechunk_layer.allocate_inference_cache(batch_size, max_seqlen, device, dtype=dtype),
                decoder_state=self.decoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        inference_params: Optional[HNetState] = None,
        **mixer_kwargs,
    ):
        assert mask is not None or (cu_seqlens is not None and max_seqlen is not None), "Provide mask or (cu_seqlens, max_seqlen)"
        if inference_params is None:
            inference_params = HNetState(main_network_state=None)
        else:
            assert mask is not None, "Mask must be provided if inference_params is provided"

        D = hidden_states.shape[-1]
        early_dims = hidden_states.shape[:-1]
        if self.pad_dimension is not None:
            hidden_states = torch.cat((hidden_states, self.pad_dimension.expand(early_dims + (-1,))), dim=-1)

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

        hs_for_residual = hs.to(dtype=self.residual_proj.weight.dtype)
        residual = self.residual_proj(hs_for_residual)

        # Routing
        bpred_output = self.routing_module(hs, cu_seqlens=cu_seqlens, mask=mask, inference_params=inference_params.routing_module_state)
        # Chunk
        hs_chunk, next_cu, next_max_L, next_mask = self.chunk_layer(hs, bpred_output.boundary_mask, cu_seqlens, mask=mask)

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
        hs = self.residual_func(hs.to(dtype=residual.dtype), residual, bpred_output.selected_probs).to(hs.dtype)

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

    def step(self, hidden_states: torch.Tensor, inference_params: HNetState):
        D = hidden_states.shape[-1]
        if self.pad_dimension is not None:
            hidden_states = torch.cat((hidden_states, self.pad_dimension.expand(hidden_states.shape[:-1] + (-1,))), dim=-1)
        if self.is_innermost:
            hs = self.main_network.step(hidden_states, inference_params.main_network_state)
            hs = hs[..., :D]
            return hs, []
        hs = self.encoder.step(hidden_states, inference_params.encoder_state)
        hs_for_residual = hs.to(dtype=self.residual_proj.weight.dtype)
        residual = self.residual_proj(hs_for_residual)
        bpred_output = self.routing_module.step(hs, inference_params.routing_module_state)
        hs_inner = self.chunk_layer.step(hs, bpred_output.boundary_mask)
        if hs_inner.shape[0] > 0:
            hs_inner, prev_boundary_predictions = self.main_network.step(hs_inner, inference_params.main_network_state)
        else:
            prev_boundary_predictions = []
        hs = self.dechunk_layer.step(hs_inner, bpred_output.boundary_mask, bpred_output.boundary_prob, inference_params.dechunk_state)
        hs = self.residual_func(hs.to(dtype=residual.dtype), residual, bpred_output.selected_probs).to(hs.dtype)
        hs = self.decoder.step(hs, inference_params.decoder_state)
        hs = hs[..., :D]
        return hs, [bpred_output, *prev_boundary_predictions]


# ============================
# Optional: simple LM wrapper operating on byte vocab (UTF-8)
# ============================


class HNetForCausalLM(nn.Module):
    def __init__(self, config: HNetConfig, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.config = config
        d_embed = config.d_model[0]
        self.embeddings = nn.Embedding(config.vocab_size, d_embed, **factory_kwargs)
        self.backbone = HNet(config=config, stage_idx=0, **factory_kwargs)
        self.lm_head = nn.Linear(d_embed, config.vocab_size, bias=False, **factory_kwargs)
        if config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids=None,
        inference_params: Optional[HNetState] = None,
        num_last_tokens: int = 0,
        **mixer_kwargs,
    ):
        hidden_states = self.embeddings(input_ids)
        B, L, D = hidden_states.shape
        if mask is None:
            assert inference_params is None, "Inference params not supported in packed mode here"
            hidden_states = hidden_states.flatten(0, 1)
            cu = torch.arange(B + 1, device=hidden_states.device) * L
            maxL = torch.tensor(L, dtype=torch.int, device=hidden_states.device)
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
        hs = hs.view(B, L, D)
        if num_last_tokens > 0:
            hs = hs[:, -num_last_tokens:]
        # Align input dtype to lm_head weight, then cast logits to float32 for stability
        w_dtype = self.lm_head.weight.dtype
        logits = self.lm_head(hs.to(w_dtype)).to(torch.float32)
        return logits, bpred_output, inference_params

    def step(self, input_ids: torch.Tensor, inference_params: HNetState):
        B = input_ids.shape[0]
        assert B == 1, "step currently supports batch size 1"
        hidden_states = self.embeddings(input_ids)
        hidden_states, bpred_output = self.backbone.step(hidden_states, inference_params)
        w_dtype = self.lm_head.weight.dtype
        logits = self.lm_head(hidden_states.to(w_dtype)).to(torch.float32)
        return logits, bpred_output, inference_params


# ============================
# CLI utilities: byte tokenizer + generation
# ============================


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> torch.Tensor:
        b = text.encode("utf-8")
        if add_bos:
            b = bytes([self.bos_idx]) + b
        if add_eos:
            b = b + bytes([self.eos_idx])
        return torch.tensor(list(b), dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        if ids.dim() > 1:
            ids = ids.view(-1)
        arr = ids.detach().cpu().tolist()
        # Filter BOS/EOS in decode for nicer printing
        arr = [t for t in arr if t not in (self.bos_idx, self.eos_idx)]
        try:
            return bytearray(arr).decode("utf-8")
        except Exception:
            # Return partial best-effort string
            s = []
            buf = []
            for tok in arr:
                buf.append(tok)
                try:
                    s.append(bytearray(buf).decode("utf-8"))
                    buf = []
                except Exception:
                    pass
            return "".join(s)


def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float("inf")
    return logits


def load_config_from_json(json_path: str) -> HNetConfig:
    import json

    with open(json_path, "r") as f:
        cfg = json.load(f)
    attn_cfg = AttnConfig(**cfg.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**cfg.pop("ssm_cfg"))
    return HNetConfig(**cfg, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)


def load_model(
    model_path: Optional[str],
    config_path: str,
    device: str = None,
    dtype: str = "bfloat16",
    strict: bool = True,
) -> HNetForCausalLM:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    cfg = load_config_from_json(config_path)
    model = HNetForCausalLM(cfg, device=device, dtype=torch_dtype)
    model.eval()
    if model_path:
        state = torch.load(model_path, map_location=device)
        try:
            result = model.load_state_dict(state, strict=strict)
            # If strict=True, PyTorch still returns IncompatibleKeys (should be empty)
            missing = getattr(result, "missing_keys", [])
            unexpected = getattr(result, "unexpected_keys", [])
            if strict and (missing or unexpected):
                raise RuntimeError(
                    f"Strict load failed: missing={len(missing)}, unexpected={len(unexpected)}\n"
                    f"Missing (first 20): {missing[:20]}\nUnexpected (first 20): {unexpected[:20]}"
                )
        except Exception as e:
            # Provide debug info to help alignment
            ckpt_keys = sorted(list(state.keys()))
            model_keys = sorted(list(model.state_dict().keys()))
            # Print a few around common prefixes
            def head_tail(arr):
                return arr[:20] + (["..."] if len(arr) > 40 else []) + arr[-20:]
            raise RuntimeError(
                "Error loading state_dict strictly.\n"
                f"Exception: {e}\n"
                f"Checkpoint keys sample: {head_tail(ckpt_keys)}\n"
                f"Model keys sample: {head_tail(model_keys)}\n"
                "Tip: ensure architecture and parameter names match the reference implementation."
            )
    return model


def generate_tokens(
    model: HNetForCausalLM,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
):
    device = next(model.parameters()).device
    tok = ByteTokenizer()
    x = tok.encode(prompt, add_bos=True).to(device)[None, :]
    cache = model.allocate_inference_cache(1, x.shape[1] + max_new_tokens, dtype=next(model.parameters()).dtype)
    with torch.inference_mode():
        mask = torch.ones_like(x, dtype=torch.bool, device=device)
        logits, _, _ = model.forward(x, mask=mask, inference_params=cache)
    logits = logits[0, -1]
    for _ in range(max_new_tokens):
        logits = logits / max(temperature, 1e-6)
        logits = _top_p_filtering(logits, top_p)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)  # (1,)
        if next_id.item() == ByteTokenizer().eos_idx:
            break
        yield next_id.item()
        with torch.inference_mode():
            nxt = next_id.view(1, 1).to(device)
            logits, _, _ = model.step(nxt, cache)
        logits = logits[0, -1]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="H-Net PyTorch (no Triton) generator")
    parser.add_argument("--config-path", type=str, required=True, help="Path to config JSON (e.g., hnet-reference/configs/hnet_2stage_L.json)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to .pt weights (optional; non-strict load)")
    parser.add_argument("--prompt", type=str, required=True, help="UTF-8 prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"]) 
    parser.add_argument("--strict", action="store_true", help="Strict state_dict load (fail on any mismatch)")
    args = parser.parse_args()

    # Set manual seed for deterministic sampling
    torch.manual_seed(args.seed)

    model = load_model(args.model_path, args.config_path, device=args.device, dtype=args.dtype, strict=args.strict or True)
    tok = ByteTokenizer()
    print(args.prompt, end="", flush=True)
    buf = []
    for tid in generate_tokens(
        model,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    ):
        buf.append(tid)
        # Stream decode in small chunks to respect UTF-8 boundaries
        for j in range(1, min(len(buf), 4) + 1):
            try:
                s = tok.decode(torch.tensor(buf[:j]))
                print(s, end="", flush=True)
                buf = buf[j:]
                break
            except Exception:
                pass


if __name__ == "__main__":
    main()
