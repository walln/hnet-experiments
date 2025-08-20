#!/usr/bin/env python3
"""
Complete standalone H-Net implementation - single file with no dependencies
All code copied directly from working multi-file version
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# ============================================================================
# Mamba2 Components (copied from mamba_no_triton.py)
# ============================================================================


def segsum(x):
    """More stable segment sum calculation (from ssd_minimal.py)"""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """
    Pure PyTorch SSD implementation from ssd_minimal.py

    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
        block_len: chunk size
    Return:
        Y: (batch, length, n_heads, d_head)
        final_state: (batch, n_heads, d_head, d_state)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype

    original_length = X.shape[1]

    # Pad sequence to be divisible by block_len if necessary
    if X.shape[1] % block_len != 0:
        pad_len = block_len - (X.shape[1] % block_len)
        X = F.pad(X, (0, 0, 0, 0, 0, pad_len))
        A = F.pad(A, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len))

    # Rearrange into blocks/chunks
    X, A, B, C = [
        rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")

    # Trim back to original length if we padded
    if Y.shape[1] > original_length:
        Y = Y[:, :original_length]

    return Y, final_state


class RMSNorm(nn.Module):
    """RMSNorm implementation matching flash_attn.ops.triton.layer_norm.RMSNorm interface"""

    def __init__(self, d, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device, dtype=dtype))

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        """
        Forward pass matching flash_attn RMSNorm interface
        
        Args:
            x: input tensor
            residual: optional residual tensor to add
            prenorm: if True, apply residual connection after normalization
            residual_in_fp32: if True, convert residual to fp32
        """
        if residual is not None:
            if residual_in_fp32:
                residual = residual.to(torch.float32)
            x = (x + residual).to(x.dtype)
        
        if prenorm:
            # Pre-norm: normalize input, return (normalized, residual_for_next_layer)
            normalized = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
            return normalized, x
        else:
            # Post-norm: just normalize and return
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RMSNormGated(nn.Module):
    """Gated RMSNorm for Mamba2"""

    def __init__(self, d, eps=1e-5, norm_before_gate=True, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.weight = nn.Parameter(torch.ones(d, device=device, dtype=dtype))

    def forward(self, x, z=None):
        if z is not None:
            if self.norm_before_gate:
                x = (
                    x
                    * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                    * self.weight
                )
                return x * F.silu(z)
            else:
                return (
                    (x * F.silu(z))
                    * torch.rsqrt(
                        (x * F.silu(z)).pow(2).mean(-1, keepdim=True) + self.eps
                    )
                    * self.weight
                )
        else:
            return (
                x
                * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                * self.weight
            )


class RotaryEmbedding(nn.Module):
    """Pure PyTorch Rotary Position Embedding"""
    
    def __init__(self, dim, base=10000.0, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # Precompute inverse frequencies for half the dimension
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, seq_len, device, dtype):
        if seq_len > self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device:
            self._seq_len_cached = seq_len
            
            # Create position indices
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
            
            # Compute frequencies
            freqs = torch.outer(t, self.inv_freq.to(device))
            
            # Cache cos and sin
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)
    
    def rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embedding to query and key tensors."""
        # cos and sin: (seq_len, dim//2)
        # q, k: (batch_size, seq_len, num_heads, head_dim)
        
        seq_len = q.shape[1]
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim//2)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, dim//2)
        
        # Only apply rotary to the first rotary_dim dimensions
        rotary_dim = self.dim  # Use the actual rotary dimension
        
        q_rot = q[..., :rotary_dim]
        q_pass = q[..., rotary_dim:]
        k_rot = k[..., :rotary_dim]
        k_pass = k[..., rotary_dim:]
        
        # Split into pairs for rotation (last dim should be even)
        assert rotary_dim % 2 == 0
        q_rot_reshaped = q_rot.view(*q_rot.shape[:-1], rotary_dim // 2, 2)
        k_rot_reshaped = k_rot.view(*k_rot.shape[:-1], rotary_dim // 2, 2)
        
        # Apply rotation: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
        q_rot_real = q_rot_reshaped[..., 0] * cos - q_rot_reshaped[..., 1] * sin
        q_rot_imag = q_rot_reshaped[..., 0] * sin + q_rot_reshaped[..., 1] * cos
        k_rot_real = k_rot_reshaped[..., 0] * cos - k_rot_reshaped[..., 1] * sin
        k_rot_imag = k_rot_reshaped[..., 0] * sin + k_rot_reshaped[..., 1] * cos
        
        # Reshape back
        q_rot = torch.stack([q_rot_real, q_rot_imag], dim=-1).view(*q_rot.shape)
        k_rot = torch.stack([k_rot_real, k_rot_imag], dim=-1).view(*k_rot.shape)
        
        # Concatenate rotated and non-rotated parts
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
        
        return q, k
    
    def forward(self, q, k):
        seq_len = q.shape[1]
        self._update_cos_sin_cache(seq_len, q.device, q.dtype)
        return self.apply_rotary_pos_emb(q, k, self._cos_cached, self._sin_cached)


class Mamba2NoTriton(nn.Module):
    """Mamba2 implementation without Triton dependencies"""

    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="swish",
        bias=False,
        conv_bias=True,
        chunk_size=256,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        if self.learnable_init_states:
            self.init_states = nn.Parameter(
                torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs)
            )

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))

        # Extra normalization layer right before output projection
        self.norm = RMSNormGated(
            self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs
        )

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

    def forward(self, u, seq_idx=None, inference_params=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        # Handle inference mode
        if inference_params is not None:
            return self.step(u, inference_params)

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads)
        initial_states = (
            repeat(self.init_states, "... -> b ...", b=batch)
            if self.learnable_init_states
            else None
        )

        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

        # 1D Convolution
        xBC = self.act(
            self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)
        )  # (B, L, d_inner + 2 * ngroups * d_state)
        xBC = xBC[:, :seqlen, :]  # Trim padding

        # Split into 3 main branches: X, B, C
        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )

        # Reshape for SSD computation
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups).expand(
            -1, -1, self.nheads, -1
        )  # Expand groups to heads
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups).expand(
            -1, -1, self.nheads, -1
        )  # Expand groups to heads

        # Use pure PyTorch SSD implementation
        # Need to handle dt properly - expand to match heads and multiply with x
        dt_expanded = dt.unsqueeze(-1).expand(
            -1, -1, -1, self.headdim
        )  # (B, L, nheads, headdim)
        x_scaled = x * dt_expanded
        A_scaled = (
            A.unsqueeze(0).unsqueeze(0).expand(batch, seqlen, -1)
        )  # (B, L, nheads)
        A_dt = A_scaled * dt  # (B, L, nheads)

        y, final_state = ssd_minimal_discrete(
            x_scaled, A_dt, B, C, self.chunk_size, initial_states
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        # Add skip connection
        if hasattr(self, "D"):
            x_flat = rearrange(x, "b l h p -> b l (h p)")
            D_expanded = self.D.unsqueeze(-1).expand(-1, self.headdim).flatten()
            y = y + x_flat * D_expanded

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, inference_params):
        """Step function for autoregressive generation with proper state management"""
        batch_size, seq_len, dim = hidden_states.shape
        assert seq_len == 1, "Step function expects single token input"
        
        # Get cached states
        layer_idx = getattr(self, 'layer_idx', 0)
        if not hasattr(inference_params, 'key_value_memory_dict'):
            inference_params.key_value_memory_dict = {}
            
        if layer_idx not in inference_params.key_value_memory_dict:
            # Initialize cache: [conv_state, ssm_state]
            conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
            conv_state = torch.zeros(batch_size, conv_dim, self.d_conv, 
                                   device=hidden_states.device, dtype=hidden_states.dtype)
            ssm_state = torch.zeros(batch_size, self.nheads, self.headdim, self.d_state,
                                  device=hidden_states.device, dtype=hidden_states.dtype)
            inference_params.key_value_memory_dict[layer_idx] = [conv_state, ssm_state]
        
        conv_state, ssm_state = inference_params.key_value_memory_dict[layer_idx]
        
        # Input projection
        u = hidden_states.squeeze(1)  # (B, D)
        zxbcdt = self.in_proj(u)  # (B, d_in_proj)
        
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, nheads)
        
        # Update convolution state by rolling and adding new input
        conv_state = torch.cat([conv_state[:, :, 1:], xBC.unsqueeze(-1)], dim=-1)
        
        # Apply convolution using cached state
        # conv1d weights are (out_channels, in_channels//groups, kernel_size)
        conv_weight = self.conv1d.weight.squeeze(1)  # (conv_dim, d_conv) - remove groups dim
        xBC = torch.sum(conv_state * conv_weight.unsqueeze(0), dim=-1)  # (B, conv_dim)
        if self.conv1d.bias is not None:
            xBC = xBC + self.conv1d.bias
        xBC = self.act(xBC)
        
        # Split convolution output
        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )
        
        # Reshape for computation
        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)  # (B, nheads, headdim)
        B = rearrange(B, "b (g n) -> b g n", g=self.ngroups).expand(-1, self.nheads, -1)
        C = rearrange(C, "b (g n) -> b g n", g=self.ngroups).expand(-1, self.nheads, -1)
        
        # SSM step computation
        A = -torch.exp(self.A_log)  # (nheads,)
        
        # Compute discrete A
        dt_expanded = dt.unsqueeze(-1)  # (B, nheads, 1)
        dA = torch.exp(dt_expanded * A.unsqueeze(0).unsqueeze(-1))  # (B, nheads, 1)
        
        # Update SSM state: state = dA * state + dt * B * x
        dBx = torch.einsum("bh,bhn,bhp->bhpn", dt, B, x)  # B: (B, nheads, d_state), x: (B, nheads, headdim)
        ssm_state = ssm_state * dA.unsqueeze(-1) + dBx  # (B, nheads, headdim, d_state)
        
        # Compute output: y = C * state
        y = torch.einsum("bhpn,bhn->bhp", ssm_state, C)  # C: (B, nheads, d_state)
        
        # Add skip connection
        if hasattr(self, "D"):
            D_expanded = self.D.unsqueeze(-1).expand(-1, self.headdim).flatten()
            x_flat = rearrange(x, "b h p -> b (h p)")
            y_flat = rearrange(y, "b h p -> b (h p)")
            y_flat = y_flat + x_flat * D_expanded
        else:
            y_flat = rearrange(y, "b h p -> b (h p)")
        
        # Apply normalization and output projection
        y = self.norm(y_flat, z)
        out = self.out_proj(y)
        
        # Update cache
        inference_params.key_value_memory_dict[layer_idx] = [conv_state, ssm_state]
        
        return out.unsqueeze(1)  # Return (B, 1, D)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate inference cache (simplified)"""
        device = self.out_proj.weight.device
        # Return a simple dict - more complete implementation would have proper state management
        return {"batch_size": batch_size, "max_seqlen": max_seqlen, "seqlen_offset": 0}


# ============================================================================
# Proper mamba_chunk_scan_combined Implementation
# ============================================================================


def chunk_state_pytorch(B, x, dt, dA_cumsum):
    """
    Equivalent to the Triton chunk_state kernel.
    Computes the SSM state for each chunk.
    
    Args:
        B: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim) 
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Returns:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    _, _, nchunks, chunk_size = dt.shape
    
    # Ensure dimensions match
    assert nheads % ngroups == 0
    nheads_per_group = nheads // ngroups
    
    # Reshape to chunks
    x_chunk = rearrange(x, "b (c l) h p -> b c l h p", c=nchunks)
    B_chunk = rearrange(B, "b (c l) g n -> b c l g n", c=nchunks)
    
    # Expand B to match heads
    B_chunk = repeat(B_chunk, "b c l g n -> b c l (g h) n", h=nheads_per_group)
    
    # Initialize states
    states = torch.zeros(batch, nchunks, nheads, headdim, dstate, 
                        device=x.device, dtype=torch.float32)
    
    # Process each chunk
    for chunk_idx in range(nchunks):
        x_c = x_chunk[:, chunk_idx]  # (batch, chunk_size, nheads, headdim)
        B_c = B_chunk[:, chunk_idx]  # (batch, chunk_size, nheads, dstate)
        dt_c = dt[:, :, chunk_idx]   # (batch, nheads, chunk_size)
        dA_c = dA_cumsum[:, :, chunk_idx]  # (batch, nheads, chunk_size)
        
        # Compute state contribution for this chunk
        for pos in range(chunk_size):
            dt_pos = dt_c[:, :, pos:pos+1]      # (batch, nheads, 1)
            dA_pos = dA_c[:, :, pos:pos+1]      # (batch, nheads, 1)
            x_pos = x_c[:, pos]                 # (batch, nheads, headdim)
            B_pos = B_c[:, pos]                 # (batch, nheads, dstate)
            
            # Weight by decay from end of chunk
            decay_weight = torch.exp(dA_c[:, :, -1:] - dA_pos)  # (batch, nheads, 1)
            
            # Accumulate: states += decay_weight * dt * B * x
            contribution = torch.einsum('bh,bhp,bhn->bhpn', 
                                       decay_weight.squeeze(-1) * dt_pos.squeeze(-1),
                                       x_pos, B_pos)
            states[:, chunk_idx] += contribution
    
    return states


def state_passing_pytorch(states, dA_chunk_cumsum):
    """
    Equivalent to the Triton state_passing kernel.
    Passes states between chunks with weighted cumsum.
    
    Args:
        states: (batch, nchunks, nheads, headdim, dstate)
        dA_chunk_cumsum: (batch, nheads, nchunks) - cumsum of dA over chunks
    Returns:
        states_out: (batch, nchunks, nheads, headdim, dstate) 
    """
    batch, nchunks, nheads, headdim, dstate = states.shape
    
    # Initialize output with initial states (zeros)
    states_out = torch.zeros_like(states)
    
    # Process sequentially through chunks
    current_state = torch.zeros(batch, nheads, headdim, dstate, 
                               device=states.device, dtype=states.dtype)
    
    for chunk_idx in range(nchunks):
        # Get decay factor to current chunk
        if chunk_idx > 0:
            decay = torch.exp(dA_chunk_cumsum[:, :, chunk_idx-1])  # (batch, nheads)
            decay = decay.unsqueeze(-1).unsqueeze(-1)  # (batch, nheads, 1, 1)
            current_state = current_state * decay  # (batch, nheads, headdim, dstate)
        
        # Add current chunk's state contribution
        current_state = current_state + states[:, chunk_idx]  # (batch, nheads, headdim, dstate)
        
        # Store the updated state
        states_out[:, chunk_idx] = current_state
    
    return states_out


def chunk_scan_pytorch(B, C, x, dt, dA_cumsum, prev_states, D=None):
    """
    Equivalent to the Triton chunk_scan kernel.
    Computes output using BMM and state contributions.
    
    Args:
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate) 
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,) optional skip connection
    Returns:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    _, _, nchunks, chunk_size = dt.shape
    
    nheads_per_group = nheads // ngroups
    
    # Reshape to chunks
    x_chunk = rearrange(x, "b (c l) h p -> b c l h p", c=nchunks)
    B_chunk = rearrange(B, "b (c l) g n -> b c l g n", c=nchunks)
    C_chunk = rearrange(C, "b (c l) g n -> b c l g n", c=nchunks)
    
    # Expand B and C to match heads
    B_chunk = repeat(B_chunk, "b c l g n -> b c l (g h) n", h=nheads_per_group)
    C_chunk = repeat(C_chunk, "b c l g n -> b c l (g h) n", h=nheads_per_group)
    
    # Initialize output
    out = torch.zeros_like(x_chunk)
    
    # Process each chunk
    for chunk_idx in range(nchunks):
        x_c = x_chunk[:, chunk_idx]    # (batch, chunk_size, nheads, headdim)
        B_c = B_chunk[:, chunk_idx]    # (batch, chunk_size, nheads, dstate)  
        C_c = C_chunk[:, chunk_idx]    # (batch, chunk_size, nheads, dstate)
        dt_c = dt[:, :, chunk_idx]     # (batch, nheads, chunk_size)
        dA_c = dA_cumsum[:, :, chunk_idx]  # (batch, nheads, chunk_size)
        
        # 1. Intra-chunk interactions (BMM)
        # Compute CB matrix
        CB = torch.einsum("blhn,bshn->bhls", C_c, B_c)  # (batch, nheads, chunk_size, chunk_size)
        
        # Apply causal mask and decay 
        causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device), diagonal=0)
        dt_diff = dA_c[:, :, :, None] - dA_c[:, :, None, :]  # (batch, nheads, chunk_size, chunk_size)
        decay = torch.exp(dt_diff) * causal_mask[None, None, :, :]
        
        CB_masked = CB * decay
        
        # Apply to input with dt weighting
        dt_expanded = dt_c.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, headdim)  # (batch, chunk_size, nheads, headdim)
        x_dt = x_c * dt_expanded  # (batch, chunk_size, nheads, headdim)
        
        intra_out = torch.einsum("bhls,bshp->blhp", CB_masked, x_dt)
        
        # 2. Inter-chunk contribution from previous states
        state_decay = torch.exp(dA_c).unsqueeze(-1)  # (batch, nheads, chunk_size, 1)
        prev_state = prev_states[:, chunk_idx]  # (batch, nheads, headdim, dstate)
        
        inter_out = torch.einsum("blhn,bhpn,bhl->blhp", 
                                C_c, prev_state, state_decay.squeeze(-1))
        
        # Combine contributions
        out[:, chunk_idx] = intra_out + inter_out
    
    # Reshape back and add skip connection
    out = rearrange(out, "b c l h p -> b (c l) h p")
    
    if D is not None:
        if D.dim() == 1:
            D = D.unsqueeze(-1).expand(-1, headdim)  # (nheads, headdim)
        D_expanded = D.unsqueeze(0).unsqueeze(0)  # (1, 1, nheads, headdim)
        out = out + x * D_expanded
    
    return out


def mamba_chunk_scan_combined_pytorch(x, dt, A, B, C, chunk_size, D=None, dt_bias=None, 
                                     seq_idx=None, dt_softplus=False):
    """
    Correct PyTorch implementation of mamba_chunk_scan_combined for EMA reconstruction.
    
    Based on analysis of reference implementation, this performs:
    h[t] = (1-p[t]) * h[t-1] + p[t] * hidden_states[t]  
    y[t] = h[t]
    
    Where:
    - x = hidden_states / dt (input scaled by inverse time constant)
    - p = B (boundary probabilities)  
    - dt = log(1/(1-p)) (time constant from boundary probability)
    - A = -1 (exponential decay)
    - C = 1 (unity output weights)
    
    Args:
        x: (batch, seqlen, nheads, headdim) - input/dt 
        dt: (batch, seqlen, nheads) - time constants
        A: (nheads,) - decay factors (should be -1)
        B: (batch, seqlen, 1, 1) - boundary probabilities p
        C: (batch, seqlen, 1, 1) - output weights (should be 1)
        
    Returns:
        y: (batch, seqlen, nheads, headdim) - EMA reconstruction
    """
    batch, seqlen, nheads, headdim = x.shape
    device = x.device
    dtype = x.dtype
    
    # Extract boundary probabilities from B: (batch, seqlen, 1, 1) -> (batch, seqlen)
    p = B.squeeze(-1).squeeze(-1)  # boundary probabilities
    
    # The mathematical relationship is:
    # hidden_states[t] = x[t] * dt[t] (since x = hidden_states / dt in reference)
    # h[t] = (1-p[t]) * h[t-1] + p[t] * hidden_states[t]
    hidden_states = x * dt.unsqueeze(-1)  # (batch, seqlen, nheads, headdim)
    
    # Initialize output
    y = torch.zeros_like(x)
    
    # Process each sequence in batch
    for b in range(batch):
        # Initialize state for this sequence
        h = torch.zeros(nheads, headdim, device=device, dtype=dtype)
        
        for t in range(seqlen):
            # Current boundary probability
            p_t = p[b, t]  # scalar
            
            # EMA update: h[t] = (1-p[t]) * h[t-1] + p[t] * hidden_states[t]
            h = (1 - p_t) * h + p_t * hidden_states[b, t]
            
            # Output is the current state
            y[b, t] = h
    
    return y


# ============================================================================
# H-Net Components (copied from hnet_reference_correct.py)
# ============================================================================


@dataclass
class RoutingModuleOutput:
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor


@dataclass
class RoutingModuleState:
    """State for routing module during inference"""

    has_seen_tokens: torch.Tensor  # (batch_size,) bool
    last_hidden_state: torch.Tensor  # (batch_size, d_model)


@dataclass
class DeChunkState:
    """State for dechunk layer during inference"""

    last_value: torch.Tensor  # (batch_size, d_model)


@dataclass
class IsotropicInferenceParams:
    """Simplified isotropic inference params"""

    key_value_memory_dict: Dict = None

    def __post_init__(self):
        if self.key_value_memory_dict is None:
            self.key_value_memory_dict = {}


@dataclass
class HNetState:
    """Complete H-Net state for inference"""

    encoder_state: Optional[IsotropicInferenceParams] = None
    routing_module_state: Optional[RoutingModuleState] = None
    main_network_state: Optional[Union["HNetState", IsotropicInferenceParams]] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[IsotropicInferenceParams] = None


@dataclass
class HNetReferenceConfig:
    """H-Net configuration matching reference"""

    d_model: List[int]
    d_intermediate: List[int]
    vocab_size: int
    arch_layout: List
    ssm_cfg: Dict
    attn_cfg: Dict
    tie_embeddings: bool = False


class STE(torch.autograd.Function):
    """Straight-Through Estimator - exact from reference"""

    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_func(x):
    return STE.apply(x)


class RoutingModuleReference(nn.Module):
    """Exact routing module from reference"""

    def __init__(self, d_model, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}

        # Routing projection layers - let weights be loaded from checkpoint
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)

        # CRITICAL: Initialize as identity matrices like reference
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model, **factory_kwargs))
            self.k_proj_layer.weight.copy_(torch.eye(d_model, **factory_kwargs))

        # Mark as no-reinit to allow weight loading to override if needed
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return RoutingModuleState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def forward(self, hidden_states, mask=None, inference_params=None):
        """Exact forward from reference"""
        if inference_params is not None:
            assert mask is not None, (
                "Mask must be provided if inference_params is not provided"
            )
            assert (~inference_params.has_seen_tokens).all(), (
                "Cannot have seen tokens when inference_params is not provided"
            )

        # Compute cosine similarity between consecutive tokens
        cos_sim = torch.einsum(
            "b l d, b l d -> b l",
            F.normalize(self.q_proj_layer(hidden_states[:, :-1]), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states[:, 1:]), dim=-1),
        )

        # Convert to boundary probability
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

        # Force boundary probability of the first element to 1.0
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

        # CRITICAL: Create 2-channel categorical distribution like reference
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        # EXACT selection logic from reference: use argmax NOT threshold
        selected_idx = torch.argmax(boundary_prob, dim=-1)
        boundary_mask = selected_idx == 1

        # Apply mask constraint from reference - no invalid tokens can be selected
        if mask is not None:
            boundary_mask = boundary_mask & mask

        # Selected probs like reference using gather with selected indices
        selected_probs = boundary_prob.gather(dim=-1, index=selected_idx.unsqueeze(-1))

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )

    def step(self, hidden_states, inference_params):
        """Exact step function from reference"""
        # hidden_states is (B, 1, D)
        hidden_states_squeezed = hidden_states.squeeze(1)

        cos_sim = torch.einsum(
            "b d, b d -> b",
            F.normalize(self.q_proj_layer(inference_params.last_hidden_state), dim=-1),
            F.normalize(self.k_proj_layer(hidden_states_squeezed), dim=-1),
        )

        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        inference_params.last_hidden_state.copy_(hidden_states_squeezed)

        # Handle first token
        boundary_prob = torch.where(
            inference_params.has_seen_tokens,
            boundary_prob,
            torch.ones_like(boundary_prob),
        )

        # Create (B, 2) tensor as in reference - categorical distribution
        boundary_prob_2d = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        inference_params.has_seen_tokens.copy_(
            torch.ones_like(inference_params.has_seen_tokens)
        )

        # Match reference: use argmax for boundary selection
        selected_idx = torch.argmax(boundary_prob_2d, dim=-1)  # (B,)
        boundary_mask = selected_idx == 1  # (B,)
        selected_probs = boundary_prob_2d.gather(dim=-1, index=selected_idx.unsqueeze(-1))  # (B, 1)

        return RoutingModuleOutput(
            boundary_prob=boundary_prob_2d,  # (B, 2)
            boundary_mask=boundary_mask,  # (B,)
            selected_probs=selected_probs,  # (B, 1)
        )


class ChunkLayerReference(nn.Module):
    """Exact chunk layer from reference"""

    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, boundary_mask, mask=None):
        """Exact forward from reference"""
        # Simplified version without cu_seqlens for now
        B, L, D = hidden_states.shape

        num_tokens = boundary_mask.sum(dim=-1)
        next_max_seqlen = int(num_tokens.max())

        device = hidden_states.device

        # Exact logic from reference
        token_idx = (
            torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)

        next_hidden_states = torch.gather(
            hidden_states,
            dim=1,
            index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                -1, -1, hidden_states.shape[-1]
            ),
        )

        next_mask = (
            torch.arange(next_max_seqlen, device=device)[None, :] < num_tokens[:, None]
        )

        return next_hidden_states, None, None, next_mask

    def step(self, hidden_states, boundary_mask):
        """Exact step from reference"""
        return hidden_states[boundary_mask]


class DeChunkLayerReference(nn.Module):
    """Exact dechunk layer from reference using SSD for EMA"""

    def __init__(
        self, d_model, dtype=torch.float32, block_size=256, headdim=32, device=None
    ):
        super().__init__()
        self.d_model = d_model
        self.dtype = dtype
        self.block_size = block_size
        self.headdim = headdim
        self.device = device
        assert d_model % self.headdim == 0
        self.nheads = d_model // self.headdim

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return DeChunkState(
            last_value=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def mamba_chunk_scan_combined_equivalent(self, x, dt, A, B, C, chunk_size):
        """
        Exact PyTorch equivalent of mamba_chunk_scan_combined for EMA case.
        
        Args:
            x: (batch, seqlen, nheads, headdim) - hidden_states / dt  
            dt: (batch, seqlen, nheads) - log(1/(1-p))
            A: (nheads,) - all -1
            B: (batch, seqlen, 1, 1) - boundary probabilities p
            C: (batch, seqlen, 1, 1) - all ones
            chunk_size: int
            
        Returns:
            out: (batch, seqlen, nheads, headdim) - EMA reconstructed sequence
        """
        batch, seqlen, nheads, headdim = x.shape
        dstate = B.shape[-1]  # Should be 1 for EMA case
        
        # Pad sequence to chunk boundary if needed
        original_seqlen = seqlen
        if seqlen % chunk_size != 0:
            pad_len = chunk_size - seqlen % chunk_size
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            dt = F.pad(dt, (0, 0, 0, pad_len))
            B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
            seqlen = seqlen + pad_len

        nchunks = seqlen // chunk_size
        
        # Reshape to chunks: (batch, nchunks, chunk_size, ...)
        x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
        dt = rearrange(dt, "b (c l) h -> b c l h", l=chunk_size)
        B = rearrange(B, "b (c l) g n -> b c l g n", l=chunk_size)
        C = rearrange(C, "b (c l) g n -> b c l g n", l=chunk_size)
        
        # Broadcast B, C to match number of heads (they should be 1 for EMA)
        if B.shape[3] == 1 and nheads > 1:
            B = B.expand(-1, -1, -1, nheads, -1)  # (batch, nchunks, chunk_size, nheads, dstate)
        if C.shape[3] == 1 and nheads > 1:
            C = C.expand(-1, -1, -1, nheads, -1)  # (batch, nchunks, chunk_size, nheads, dstate)
        
        # Convert to float32 for numerical stability
        x = x.float()
        dt = dt.float()
        B = B.float()
        C = C.float()
        
        # Initialize output and state
        out = torch.zeros_like(x)
        state = torch.zeros(batch, nheads, headdim, dstate, dtype=torch.float32, device=x.device)
        
        # Process each chunk sequentially
        for chunk_idx in range(nchunks):
            # Current chunk data
            x_chunk = x[:, chunk_idx]        # (batch, chunk_size, nheads, headdim)
            dt_chunk = dt[:, chunk_idx]      # (batch, chunk_size, nheads)
            B_chunk = B[:, chunk_idx]        # (batch, chunk_size, nheads, dstate)
            C_chunk = C[:, chunk_idx]        # (batch, chunk_size, nheads, dstate)
            
            # Process each position in chunk
            for pos in range(chunk_size):
                # Get position data
                dt_pos = dt_chunk[:, pos]       # (batch, nheads)
                B_pos = B_chunk[:, pos]         # (batch, nheads, dstate)
                C_pos = C_chunk[:, pos]         # (batch, nheads, dstate)
                x_pos = x_chunk[:, pos]         # (batch, nheads, headdim)
                
                # SSM update: h[t] = exp(A * dt) * h[t-1] + B * dt * x[t]
                # For A = -1: h[t] = exp(-dt) * h[t-1] + B * dt * x[t]
                decay_factor = torch.exp(A.unsqueeze(0) * dt_pos)  # (batch, nheads)
                
                # Apply decay to previous state
                state = state * decay_factor.unsqueeze(-1).unsqueeze(-1)  # (batch, nheads, headdim, dstate)
                
                # Add input contribution: B * dt * x
                # B: (batch, nheads, dstate), dt: (batch, nheads), x: (batch, nheads, headdim)
                input_contrib = torch.einsum('bhn,bh,bhp->bhpn', B_pos, dt_pos, x_pos)
                state = state + input_contrib
                
                # Compute output: y[t] = C * h[t]
                out_pos = torch.einsum('bhn,bhpn->bhp', C_pos, state)
                out[:, chunk_idx, pos] = out_pos
        
        # Reshape back to original format
        out = rearrange(out, "b c l h p -> b (c l) h p")
        
        # Remove padding if added
        if original_seqlen != seqlen:
            out = out[:, :original_seqlen]
        
        # Convert back to original dtype
        return out.to(self.dtype)

    def forward(
        self,
        hidden_states,
        boundary_mask,
        boundary_prob,
        mask=None,
        inference_params=None,
    ):
        """Exact forward from reference using SSD for mamba_chunk_scan_combined"""
        if inference_params is None:
            assert mask is not None, (
                "Mask must be provided if inference_params is not provided"
            )
            assert boundary_mask[:, 0].all(), (
                "First token must be a boundary if running prefill"
            )

        # Exact logic from reference with proper dtype handling
        if boundary_prob.dim() == 3 and boundary_prob.shape[-1] > 1:
            # boundary_prob is (B, L, 2) from forward pass
            p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - (1e-4))
        else:
            # boundary_prob is (B, L) from forward pass  
            p = torch.clamp(boundary_prob.float(), min=1e-4, max=1 - (1e-4))
        

        B, L = boundary_mask.shape
        M = hidden_states.shape[1]  # Number of boundary tokens

        # Only apply token sorting if we have the full sequence
        if p.shape[1] == L:
            # Token sorting logic from reference
            token_idx = (
                torch.arange(L, device=hidden_states.device)[None, :]
                + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            p = torch.gather(p, dim=1, index=seq_sorted_indices[:, :M])  # (B, M)
        else:
            # p is already the right shape (B, M)
            pass

        original_dtype = hidden_states.dtype

        # Exact EMA computation from reference using SSD
        dt = torch.log(1 / (1 - p)).to(self.dtype)
        x = (hidden_states / dt[..., None]).to(self.dtype)
        A = -torch.ones(
            (self.nheads,), device=hidden_states.device, dtype=torch.float32
        )
        b = p.to(self.dtype)
        c = torch.ones_like(b)

        # CRITICAL: Use exact reference implementation convention
        # mamba_chunk_scan_combined uses x/dt, matching reference exactly
        
        B, L, D = hidden_states.shape
        
        # Prepare parameters exactly as in reference
        p = torch.clamp(p, min=1e-4, max=1 - (1e-4))
        dt = torch.log(1 / (1 - p)).to(self.dtype)  # Convert boundary prob to timestep
        
        # Reference convention: divide by dt (not multiply!)
        x = (hidden_states / dt[..., None]).to(self.dtype)
        A = -torch.ones((self.nheads,), device=hidden_states.device, dtype=torch.float32)
        b = p.to(self.dtype)
        c = torch.ones_like(b)
        
        # Use mamba_chunk_scan_combined exactly as in reference
        out = mamba_chunk_scan_combined_pytorch(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            repeat(dt, "b l -> b l h", h=self.nheads),
            A,
            rearrange(b, "b l -> b l 1 1"),
            rearrange(c, "b l -> b l 1 1"),
            chunk_size=self.block_size
        )
        
        # Reshape back to original format
        out = rearrange(out, "b l h p -> b l (h p)")

        # Exact plug back logic from reference
        plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
        out = torch.gather(
            out,
            dim=1,
            index=plug_back_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
        )

        if inference_params is not None:
            inference_params.last_value.copy_(out[:, -1])

        return out.to(original_dtype)

    def step(self, hidden_states, boundary_mask, boundary_prob, inference_params):
        """Exact step from reference"""
        # hidden_states is (B', 1, D), where B' = boundary_mask.sum()
        # boundary_mask is (B,) and boundary_prob is (B, 2)

        B = boundary_mask.shape[0]
        D = hidden_states.shape[-1]

        p = torch.zeros(B, device=hidden_states.device, dtype=hidden_states.dtype)
        p[boundary_mask] = boundary_prob[boundary_mask, -1].clamp(
            min=1e-4, max=1 - (1e-4)
        )

        current_hidden_states = torch.zeros(
            B, D, device=hidden_states.device, dtype=hidden_states.dtype
        )
        current_hidden_states[boundary_mask] = hidden_states.squeeze(1)

        # Exact EMA formula from reference
        result = p * current_hidden_states + (1 - p) * inference_params.last_value
        inference_params.last_value.copy_(result)

        return result.unsqueeze(1)


class IsotropicReference(nn.Module):
    """Isotropic layer stack matching reference"""

    def __init__(self, config, stage_idx, pos_idx, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = config.d_model[stage_idx]

        # Parse architecture layout exactly like reference
        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = (
                arch_layout[1]
                if isinstance(arch_layout, list) and len(arch_layout) > 1
                else arch_layout
            )

        # Get the specific layout for this position
        if isinstance(arch_layout, list):
            if pos_idx == 0:  # encoder
                layout = arch_layout[0]
            elif pos_idx == 2:  # decoder
                layout = arch_layout[2]
            else:
                layout = arch_layout[pos_idx]
        else:
            layout = arch_layout

        # Parse complex mixed architectures like "T1m4", "m4T1", etc.
        def parse_mixed_layout(layout_str):
            """Parse mixed layout strings like 'T1m4' or 'm4T1'"""
            components = []
            i = 0
            while i < len(layout_str):
                if layout_str[i] in ["T", "m"]:
                    layer_type = layout_str[i]
                    i += 1
                    # Extract number
                    num_str = ""
                    while i < len(layout_str) and layout_str[i].isdigit():
                        num_str += layout_str[i]
                        i += 1
                    if num_str:
                        components.append((layer_type, int(num_str)))
                else:
                    i += 1
            return components

        # Create layers based on layout - match exact weight structure
        if isinstance(layout, str) and ("m" in layout or "T" in layout):
            # Handle mixed architectures like "T1m4", "m4T1", or simple "m4", "T22"
            if layout.startswith("m") and layout[1:].isdigit():
                # Simple mamba: "m4"
                components = [("m", int(layout[1:]))]
            elif layout.startswith("T") and layout[1:].isdigit():
                # Simple transformer: "T22"
                components = [("T", int(layout[1:]))]
            else:
                # Mixed architecture: "T1m4", "m4T1", etc.
                components = parse_mixed_layout(layout)

            # Helper classes for different layer types
            class MambaBlock(nn.Module):
                def __init__(self, d_model, layer_idx, **kwargs):
                    super().__init__()
                    self.norm1 = RMSNorm(d_model)
                    self.mixer = Mamba2NoTriton(
                        d_model=d_model,
                        d_state=config.ssm_cfg["d_state"],
                        d_conv=config.ssm_cfg["d_conv"],
                        expand=config.ssm_cfg["expand"],
                        headdim=64,
                        chunk_size=config.ssm_cfg["chunk_size"],
                        layer_idx=layer_idx,
                        **kwargs,
                    )

                def forward(self, x):
                    # Pre-norm with proper residual handling like reference
                    residual = None
                    x_norm, residual = self.norm1(x, residual=residual, prenorm=True, residual_in_fp32=True)
                    mixed = self.mixer(x_norm)
                    return mixed, residual

            class TransformerLayerCorrect(nn.Module):
                def __init__(
                    self, d_model, num_heads=16, intermediate_size=None, **kwargs
                ):
                    super().__init__()
                    self.d_model = d_model
                    self.num_heads = num_heads
                    self.head_dim = d_model // num_heads

                    # Determine intermediate size based on stage
                    if intermediate_size is None:
                        if (
                            len(config.d_intermediate) > stage_idx
                            and config.d_intermediate[stage_idx] > 0
                        ):
                            intermediate_size = config.d_intermediate[stage_idx]
                        else:
                            intermediate_size = 4096  # Default

                    # Get rotary embedding dimension from config
                    rotary_emb_dim = 0
                    if (
                        hasattr(config, 'attn_cfg') and 
                        isinstance(config.attn_cfg, dict) and 
                        'rotary_emb_dim' in config.attn_cfg
                    ):
                        if isinstance(config.attn_cfg['rotary_emb_dim'], list):
                            if stage_idx < len(config.attn_cfg['rotary_emb_dim']):
                                rotary_emb_dim = config.attn_cfg['rotary_emb_dim'][stage_idx]
                        else:
                            rotary_emb_dim = config.attn_cfg['rotary_emb_dim']
                    elif hasattr(config, 'rotary_emb_dim'):
                        if isinstance(config.rotary_emb_dim, list):
                            if stage_idx < len(config.rotary_emb_dim):
                                rotary_emb_dim = config.rotary_emb_dim[stage_idx]
                        else:
                            rotary_emb_dim = config.rotary_emb_dim
                    
                    self.rotary_emb_dim = rotary_emb_dim

                    # Match exact weight structure: mixer.Wqkv, mixer.out_proj
                    self.norm1 = RMSNorm(d_model)

                    # Create mixer module matching weight structure
                    self.mixer = nn.Module()
                    self.mixer.Wqkv = nn.Linear(
                        d_model, 3 * d_model, bias=False, **kwargs
                    )
                    self.mixer.out_proj = nn.Linear(
                        d_model, d_model, bias=False, **kwargs
                    )
                    
                    # Add rotary embeddings if specified
                    if rotary_emb_dim > 0:
                        device = kwargs.get('device', None)
                        self.rotary_emb = RotaryEmbedding(rotary_emb_dim, device=device)

                    # SwiGLU/Gated MLP matching exact weight structure
                    self.norm2 = RMSNorm(d_model)
                    self.mlp = nn.Module()
                    self.mlp.fc1 = nn.Linear(
                        d_model, 2 * intermediate_size, bias=False, **kwargs
                    )
                    self.mlp.fc2 = nn.Linear(
                        intermediate_size, d_model, bias=False, **kwargs
                    )

                def forward(self, x, inference_params=None):
                    # Pre-norm attention with proper residual handling
                    residual = None
                    x_norm, residual = self.norm1(x, residual=residual, prenorm=True, residual_in_fp32=True)

                    # QKV projection
                    B, L, D = x_norm.shape
                    qkv = self.mixer.Wqkv(x_norm)  # (B, L, 3*D)
                    qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
                    q, k, v = qkv.unbind(2)  # Each is (B, L, num_heads, head_dim)

                    # Handle KV caching for inference like reference
                    if inference_params is not None and hasattr(inference_params, 'key_value_memory_dict'):
                        # During inference, use KV cache with proper rotary embedding handling
                        layer_idx = getattr(self, 'layer_idx', 0)
                        seqlen_offset = getattr(inference_params, 'seqlen_offset', 0)
                        
                        # Apply rotary embeddings BEFORE caching with correct positions
                        if hasattr(self, 'rotary_emb') and self.rotary_emb_dim > 0:
                            # Update cos/sin cache for current sequence length
                            self.rotary_emb._update_cos_sin_cache(seqlen_offset + L, q.device, q.dtype)
                            # Get cos/sin for current positions
                            cos = self.rotary_emb._cos_cached[seqlen_offset:seqlen_offset + L]
                            sin = self.rotary_emb._sin_cached[seqlen_offset:seqlen_offset + L]
                            q, k = self.rotary_emb.apply_rotary_pos_emb(q, k, cos, sin)
                        
                        # Combine K and V for caching: (B, L, 2, num_heads, head_dim)
                        kv = torch.stack([k, v], dim=2)
                        
                        # Update KV cache (similar to reference _update_kv_cache)
                        if layer_idx not in inference_params.key_value_memory_dict:
                            # Initialize cache
                            max_seqlen = getattr(inference_params, 'max_seqlen', 2048)
                            kv_cache = torch.zeros(
                                B, max_seqlen, 2, self.num_heads, self.head_dim,
                                dtype=kv.dtype, device=kv.device
                            )
                            inference_params.key_value_memory_dict[layer_idx] = kv_cache
                        else:
                            kv_cache = inference_params.key_value_memory_dict[layer_idx]
                        
                        # Update cache with new KV pairs (already with correct rotary embeddings)
                        kv_cache[:, seqlen_offset:seqlen_offset + L] = kv
                        
                        # Use cached KV for attention (all previous + current)
                        cached_kv = kv_cache[:, :seqlen_offset + L]  # (B, seq_len_so_far, 2, num_heads, head_dim)
                        k_all = cached_kv[:, :, 0]  # (B, seq_len_so_far, num_heads, head_dim)
                        v_all = cached_kv[:, :, 1]  # (B, seq_len_so_far, num_heads, head_dim)
                        
                        # Transpose for attention
                        q = q.transpose(1, 2)  # (B, num_heads, L, head_dim)
                        k_all = k_all.transpose(1, 2)  # (B, num_heads, seq_len_so_far, head_dim)
                        v_all = v_all.transpose(1, 2)  # (B, num_heads, seq_len_so_far, head_dim)
                        
                        # Attention computation with full cached context
                        scale = self.head_dim**-0.5
                        scores = torch.matmul(q, k_all.transpose(-2, -1)) * scale  # (B, num_heads, L, seq_len_so_far)
                        
                        # Causal mask for cached context
                        seq_len_so_far = seqlen_offset + L
                        mask = torch.triu(
                            torch.ones(L, seq_len_so_far, device=x.device), 
                            diagonal=seqlen_offset + 1
                        ).bool()
                        scores.masked_fill_(mask, -float("inf"))
                        
                        attn = F.softmax(scores, dim=-1)
                        out = torch.matmul(attn, v_all)  # (B, num_heads, L, head_dim)
                        
                    else:
                        # Training mode: standard attention without caching
                        
                        # Apply rotary embeddings if available
                        if hasattr(self, 'rotary_emb') and self.rotary_emb_dim > 0:
                            # For training, use full sequence
                            q, k = self.rotary_emb(q, k)
                        
                        q = q.transpose(1, 2)  # (B, num_heads, L, head_dim)
                        k = k.transpose(1, 2)
                        v = v.transpose(1, 2)

                        scale = self.head_dim**-0.5
                        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

                        # Causal mask
                        mask = torch.triu(
                            torch.ones(L, L, device=x.device), diagonal=1
                        ).bool()
                        scores.masked_fill_(mask, -float("inf"))

                        attn = F.softmax(scores, dim=-1)
                        out = torch.matmul(attn, v)  # (B, num_heads, L, head_dim)

                    out = out.transpose(1, 2).reshape(B, L, D)  # (B, L, D)

                    # Output projection
                    attn_out = self.mixer.out_proj(out)

                    # Pre-norm MLP with proper residual handling
                    attn_out, residual = self.norm2(attn_out, residual=residual, prenorm=True, residual_in_fp32=True)
                    
                    # fc1 produces both gate and up projections
                    gate_up = self.mlp.fc1(attn_out)  # (B, L, 2*intermediate_size)
                    gate, up = gate_up.chunk(
                        2, dim=-1
                    )  # Each (B, L, intermediate_size)
                    # Apply gating: up * silu(gate) (corrected SwiGLU)
                    intermediate = up * F.silu(gate)  # (B, L, intermediate_size)
                    # fc2 projects down
                    mlp_out = self.mlp.fc2(intermediate)  # (B, L, d_model)

                    return mlp_out, residual

            # Build layers from components
            layers = []
            layer_idx = 0

            # Determine number of attention heads for this stage
            if stage_idx < len(config.attn_cfg["num_heads"]):
                num_heads = config.attn_cfg["num_heads"][stage_idx]
            else:
                num_heads = 16  # Default

            # Determine intermediate size for this stage
            if stage_idx < len(config.d_intermediate):
                intermediate_size = config.d_intermediate[stage_idx]
            else:
                intermediate_size = 4096  # Default

            for layer_type, num_layers in components:
                if layer_type == "m":
                    # Add Mamba layers
                    for i in range(num_layers):
                        layers.append(
                            MambaBlock(
                                self.d_model, layer_idx=layer_idx, **factory_kwargs
                            )
                        )
                        layer_idx += 1
                elif layer_type == "T":
                    # Add Transformer layers
                    for i in range(num_layers):
                        layers.append(
                            TransformerLayerCorrect(
                                self.d_model,
                                num_heads=num_heads,
                                intermediate_size=intermediate_size,
                                **factory_kwargs,
                            )
                        )
                        layer_idx += 1

            self.layers = nn.Sequential(*layers)
            # Final norm matching weight structure
            self.rmsnorm = RMSNorm(self.d_model)

    def forward(self, hidden_states, mask=None, inference_params=None, **kwargs):
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states)
        
        # Final normalization with residual (prenorm=False)
        hidden_states = self.rmsnorm(hidden_states, residual=residual, prenorm=False, residual_in_fp32=True)
        return hidden_states

    def step(self, hidden_states, inference_params):
        return self.forward(hidden_states, inference_params=inference_params)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        return IsotropicInferenceParams()


class HNetReference(nn.Module):
    """Complete H-Net exactly matching reference implementation"""

    def __init__(
        self, config: HNetReferenceConfig, stage_idx=0, device=None, dtype=None
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]

        # Parse architecture layout exactly like reference
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

        # Create submodules exactly like reference
        for _name, _layout in zip(sub_model_names, arch_layout):
            if self.is_innermost or _name in ("encoder", "decoder"):
                SubModel = IsotropicReference
                _stage_idx = stage_idx
                _pos_idx = None
                if _name == "encoder":
                    _pos_idx = 0
                elif self.is_innermost:
                    _pos_idx = 0
                elif _name == "decoder":
                    _pos_idx = 2
                _pos_idx_dict = {"pos_idx": _pos_idx}
            else:
                SubModel = HNetReference
                _stage_idx = stage_idx + 1
                _pos_idx_dict = {}

            _sub_model = SubModel(
                config=config,
                stage_idx=_stage_idx,
                **_pos_idx_dict,
                **factory_kwargs,
            )
            self.add_module(_name, _sub_model)

        # Add chunking components for non-innermost layers
        if not self.is_innermost:
            self.routing_module = RoutingModuleReference(self.d_model, **factory_kwargs)
            self.chunk_layer = ChunkLayerReference()
            self.dechunk_layer = DeChunkLayerReference(
                d_model=self.d_model, dtype=dtype or torch.float32, device=device
            )

            # Residual projection in fp32 exactly like reference
            self.residual_proj = nn.Linear(
                self.d_model, self.d_model, bias=True, device=device, dtype=torch.float32
            )
            # Initialize to zeros like reference, then allow weight loading to overwrite
            nn.init.zeros_(self.residual_proj.weight)
            nn.init.zeros_(self.residual_proj.bias)
            self.residual_proj.weight._no_reinit = True
            self.residual_proj.bias._no_reinit = True
            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual

        # Dimension padding exactly like reference
        if stage_idx > 0 and self.d_model - config.d_model[stage_idx - 1] > 0:
            self.pad_dimension = nn.Parameter(
                torch.zeros(
                    self.d_model - config.d_model[stage_idx - 1], **factory_kwargs
                )
            )
        else:
            self.pad_dimension = None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """Exact allocation from reference"""
        if self.is_innermost:
            return HNetState(
                main_network_state=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                )
            )
        else:
            device = self.residual_proj.weight.device
            return HNetState(
                encoder_state=self.encoder.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                routing_module_state=self.routing_module.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype
                ),
                main_network_state=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                dechunk_state=self.dechunk_layer.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype
                ),
                decoder_state=self.decoder.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
            )

    def forward(self, hidden_states, mask=None, inference_params=None, bypass_chunking=False, **mixer_kwargs):
        """Exact forward from reference with optional chunking bypass"""
        if inference_params is None:
            inference_params = HNetState(main_network_state=None)
        else:
            assert mask is not None, (
                "Mask must be provided if inference_params is provided"
            )

        D = hidden_states.shape[-1]
        EARLY_DIMS = hidden_states.shape[:-1]

        # Dimension padding exactly like reference
        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (hidden_states, self.pad_dimension.expand(EARLY_DIMS + (-1,))), dim=-1
            )

        if self.is_innermost:
            hidden_states = self.main_network(
                hidden_states,
                mask=mask,
                inference_params=inference_params.main_network_state,
                **mixer_kwargs,
            )
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        # Encoder
        hidden_states = self.encoder(
            hidden_states,
            mask=mask,
            inference_params=inference_params.encoder_state,
            **mixer_kwargs,
        )

        # Residual computation in fp32 exactly like reference
        hidden_states_for_residual = hidden_states.to(
            dtype=self.residual_proj.weight.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)

        if bypass_chunking:
            # BYPASS MODE: Skip chunking and just run main network directly
            print("DEBUG: Using bypass mode - skipping dynamic chunking")
            
            # Process with main network directly (no chunking)
            hidden_states, _ = self.main_network(
                hidden_states,
                mask=mask,
                inference_params=inference_params.main_network_state,
                **mixer_kwargs,
            )
            
            # Simple residual (no hierarchical modulation)
            hidden_states = hidden_states + residual.to(hidden_states.dtype)
            
            # Create dummy outputs for consistency
            from hnet_standalone import RoutingModuleOutput
            bpred_output = RoutingModuleOutput(
                boundary_prob=torch.zeros(hidden_states.shape[0], hidden_states.shape[1], 2),
                boundary_mask=torch.zeros(hidden_states.shape[0], hidden_states.shape[1], dtype=torch.bool),
                selected_probs=torch.zeros(hidden_states.shape[0], hidden_states.shape[1], 1)
            )
            prev_boundary_predictions = []
            
        else:
            # NORMAL MODE: Full dynamic chunking pipeline 
            bpred_output = self.routing_module(
                hidden_states,
                mask=mask,
                inference_params=inference_params.routing_module_state,
            )

            hidden_states, next_cu_seqlens, next_max_seqlen, next_mask = self.chunk_layer(
                hidden_states, bpred_output.boundary_mask, mask=mask
            )

            # Process with main network
            hidden_states, prev_boundary_predictions = self.main_network(
                hidden_states,
                mask=next_mask,
                inference_params=inference_params.main_network_state,
                **mixer_kwargs,
            )

            # Dechunk
            hidden_states = self.dechunk_layer(
                hidden_states,
                bpred_output.boundary_mask,
                bpred_output.boundary_prob,
                mask=mask,
                inference_params=inference_params.dechunk_state,
            )

            # Apply hierarchical residual exactly like reference
            hidden_states = self.residual_func(
                hidden_states.to(dtype=residual.dtype),
                residual,
                bpred_output.selected_probs,
            ).to(hidden_states.dtype)

        # Decoder
        hidden_states = self.decoder(
            hidden_states,
            mask=mask,
            inference_params=inference_params.decoder_state,
            **mixer_kwargs,
        )

        hidden_states = hidden_states[..., :D]
        return hidden_states, [bpred_output, *prev_boundary_predictions]

    def step(self, hidden_states, inference_params):
        """Exact step from reference"""
        D = hidden_states.shape[-1]

        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (
                    hidden_states,
                    self.pad_dimension.expand(hidden_states.shape[:-1] + (-1,)),
                ),
                dim=-1,
            )

        if self.is_innermost:
            hidden_states = self.main_network.step(
                hidden_states, inference_params.main_network_state
            )
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        # Step functions exactly like reference
        hidden_states = self.encoder.step(hidden_states, inference_params.encoder_state)
        hidden_states_for_residual = hidden_states.to(
            dtype=self.residual_proj.weight.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)

        bpred_output = self.routing_module.step(
            hidden_states, inference_params.routing_module_state
        )
        hidden_states_inner = self.chunk_layer.step(
            hidden_states, bpred_output.boundary_mask
        )

        if hidden_states_inner.shape[0] > 0:
            hidden_states_inner, prev_boundary_predictions = self.main_network.step(
                hidden_states_inner, inference_params.main_network_state
            )
        else:
            prev_boundary_predictions = []

        hidden_states = self.dechunk_layer.step(
            hidden_states_inner,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            inference_params.dechunk_state,
        )

        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype),
            residual,
            bpred_output.selected_probs,
        ).to(hidden_states.dtype)

        hidden_states = self.decoder.step(hidden_states, inference_params.decoder_state)

        hidden_states = hidden_states[..., :D]
        return hidden_states, [bpred_output, *prev_boundary_predictions]


class HNetReferenceForCausalLM(nn.Module):
    """Complete H-Net for Causal LM exactly matching reference"""

    def __init__(self, config: HNetReferenceConfig, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.config = config
        vocab_size = config.vocab_size
        d_embed = config.d_model[0]

        self.embeddings = nn.Embedding(vocab_size, d_embed, **factory_kwargs)
        self.backbone = HNetReference(config, stage_idx=0, **factory_kwargs)
        self.lm_head = nn.Linear(d_embed, vocab_size, bias=False, **factory_kwargs)

        if config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight

    def forward(self, input_ids, mask=None, inference_params=None, bypass_chunking=False, **kwargs):
        x = self.embeddings(input_ids)

        if mask is None:
            batch, seqlen = input_ids.shape
            mask = torch.ones(batch, seqlen, device=input_ids.device, dtype=torch.bool)

        x, bpred_output = self.backbone(
            x, mask=mask, inference_params=inference_params, bypass_chunking=bypass_chunking, **kwargs
        )
        logits = self.lm_head(x)

        from collections import namedtuple

        CausalLMOutput = namedtuple(
            "CausalLMOutput", ["logits", "bpred_output", "inference_params"]
        )
        return CausalLMOutput(
            logits=logits, bpred_output=bpred_output, inference_params=inference_params
        )

    def step(self, input_ids, inference_params):
        """Step function for autoregressive generation"""
        batch = input_ids.shape[0]
        assert batch == 1, "Only support batch size 1 for step"

        x = self.embeddings(input_ids)
        x, bpred_output = self.backbone.step(x, inference_params)
        logits = self.lm_head(x)

        # CRITICAL FIX: Proper state management like reference implementation
        # Update sequence length offset for KV caching and state management
        if not hasattr(inference_params, 'seqlen_offset'):
            inference_params.seqlen_offset = 0
        inference_params.seqlen_offset += 1

        from collections import namedtuple

        CausalLMOutput = namedtuple(
            "CausalLMOutput", ["logits", "bpred_output", "inference_params"]
        )
        return CausalLMOutput(
            logits=logits, bpred_output=bpred_output, inference_params=inference_params
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        cache = self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype
        )
        # Initialize sequence offset for proper state management
        cache.seqlen_offset = 0
        cache.max_seqlen = max_seqlen
        return cache


def create_reference_hnet_from_config(config_dict, device=None, dtype=None):
    """Create reference H-Net from config dictionary"""
    config = HNetReferenceConfig(
        d_model=config_dict["d_model"],
        d_intermediate=config_dict["d_intermediate"],
        vocab_size=config_dict["vocab_size"],
        arch_layout=config_dict["arch_layout"],
        ssm_cfg=config_dict["ssm_cfg"],
        attn_cfg=config_dict["attn_cfg"],
        tie_embeddings=config_dict.get("tie_embeddings", False),
    )

    model = HNetReferenceForCausalLM(config, device=device, dtype=dtype)
    
    # Add method to zero residual projections after weight loading
    def zero_residual_projections():
        """Zero out residual projections that cause instability"""
        def _zero_residual_recursive(module):
            if hasattr(module, '_zero_residual_after_load') and module._zero_residual_after_load:
                with torch.no_grad():
                    if hasattr(module, 'residual_proj'):
                        module.residual_proj.weight.zero_()
                        module.residual_proj.bias.zero_()
                        print(f"✅ Zeroed residual projection in {module.__class__.__name__}")
            
            for child in module.children():
                _zero_residual_recursive(child)
        
        _zero_residual_recursive(model)
    
    model.zero_residual_projections = zero_residual_projections
    return model


# ============================================================================
# Test with BOS Token with configurable model path
# ============================================================================


def test_with_proper_bos(
    model_path=None,
    config_path=None,
    max_tokens=20,
    temperature=1.0,
    top_p=0.9,
    prompts=None,
):
    """Test generation with BOS token like reference"""
    print("Testing H-Net with Proper BOS Token (Standalone)")
    print("=" * 50)

    # Load config from file if provided, otherwise use default 1-stage config
    if config_path:
        import json

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"✅ Config loaded from {config_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Failed to load config from {config_path}: {e}")
            print("Using default 1-stage config")
            config = {
                "arch_layout": ["m4", ["T22"], "m4"],
                "d_model": [1024, 1536],
                "d_intermediate": [0, 4096],
                "vocab_size": 256,
                "ssm_cfg": {"chunk_size": 1, "d_conv": 4, "d_state": 128, "expand": 2},
                "attn_cfg": {
                    "num_heads": [16, 16],
                    "rotary_emb_dim": [32, 48],
                    "window_size": [1023, -1],
                },
                "tie_embeddings": False,
            }
    else:
        # Default 1-stage config
        config = {
            "arch_layout": ["m4", ["T22"], "m4"],
            "d_model": [1024, 1536],
            "d_intermediate": [0, 4096],
            "vocab_size": 256,
            "ssm_cfg": {"chunk_size": 1, "d_conv": 4, "d_state": 128, "expand": 2},
            "attn_cfg": {
                "num_heads": [16, 16],
                "rotary_emb_dim": [32, 48],
                "window_size": [1023, -1],
            },
            "tie_embeddings": False,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use float32 on CPU, bfloat16 on GPU
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = create_reference_hnet_from_config(
        config, device=device, dtype=dtype
    )
    model.eval()

    # Load weights
    if model_path:
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint, strict=True)
            print(f"✅ All weights loaded from {model_path}")
            
        except (FileNotFoundError, RuntimeError) as e:
            print(f"❌ Failed to load weights from {model_path}: {e}")
            print("Using random initialization for testing")
    else:
        print("⚠️ No model path provided, using random initialization")

    # ByteTokenizer settings from reference
    BOS_IDX = 254
    EOS_IDX = 255

    def encode_with_bos(text):
        """Encode text like reference ByteTokenizer"""
        text_bytes = text.encode("utf-8")
        # Add BOS token
        tokens = [BOS_IDX] + list(text_bytes)
        return tokens

    def decode_bytes(tokens):
        """Decode bytes like reference"""
        # Filter out special tokens
        filtered = [t for t in tokens if t not in [BOS_IDX, EOS_IDX]]
        try:
            return bytearray(filtered).decode("utf-8", errors="replace")
        except:
            return str(filtered)

    # Use provided prompts or defaults
    if prompts is None:
        test_prompts = [
            "Hello",
            "The quick brown",
            "I am a",
            "What is",
        ]
    else:
        test_prompts = prompts

    print(f"\n🎯 Testing with BOS token (254) + nucleus sampling:")
    print(f"  Max tokens: {max_tokens}, Temperature: {temperature}, Top-p: {top_p}")
    print(f"=" * 60)

    for prompt_text in test_prompts:
        print(f"\nPrompt: '{prompt_text}'")

        # Encode with BOS like reference
        prompt_tokens = encode_with_bos(prompt_text)
        print(f"Encoded: {prompt_tokens} (BOS + UTF-8 bytes)")

        # Generate with reference-like settings
        input_ids = torch.tensor([prompt_tokens], device=device)
        cache = model.allocate_inference_cache(batch_size=1, max_seqlen=100, dtype=dtype)

        generated_tokens = []

        # Prefill (like reference)
        with torch.no_grad():
            mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
            output = model(input_ids, mask=mask, inference_params=cache)
            logits = output.logits[0, -1, :]

        # Generate tokens (like reference)
        for step in range(max_tokens):
            # Handle temperature=0.0 case (deterministic sampling)
            if temperature == 0.0:
                # Deterministic: take argmax 
                next_token = torch.argmax(logits, dim=-1).item()
            else:
                logits_temp = logits / temperature

                # Top-p (nucleus) sampling like reference
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits_temp, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits_temp[indices_to_remove] = -float("inf")

                # Sample next token
                probs = F.softmax(logits_temp, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            
            generated_tokens.append(next_token)

            # Check for EOS
            if next_token == EOS_IDX:
                print(f"   <EOS at step {step}>")
                break

            # Next step like reference
            current_token = torch.tensor([[next_token]], device=device)
            with torch.no_grad():
                output = model.step(current_token, cache)
                cache = output.inference_params  # CRITICAL: Update cache with new state!
                logits = output.logits[0, -1, :]

        # Decode full sequence
        full_tokens = prompt_tokens + generated_tokens
        decoded_text = decode_bytes(full_tokens)

        print(
            f"Generated tokens: {generated_tokens[:10]}{'...' if len(generated_tokens) > 10 else ''}"
        )
        print(f"Full decoded: '{decoded_text}'")

        # Quality assessment
        words = decoded_text.split()
        if len(words) > 1:
            print(f"✅ Multi-word generation: {len(words)} words")
        else:
            print(f"❌ Single/no word generation")


def interactive_generation(
    model_path=None, config_path=None, max_tokens=100, temperature=1.0, top_p=0.9
):
    """Interactive text generation mode"""
    print("H-Net Interactive Generation")
    print("=" * 30)

    # Load config from file if provided, otherwise use default 1-stage config
    if config_path:
        import json

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"✅ Config loaded from {config_path}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"❌ Failed to load config from {config_path}: {e}")
            print("Using default 1-stage config")
            config = {
                "arch_layout": ["m4", ["T22"], "m4"],
                "d_model": [1024, 1536],
                "d_intermediate": [0, 4096],
                "vocab_size": 256,
                "ssm_cfg": {"chunk_size": 1, "d_conv": 4, "d_state": 128, "expand": 2},
                "attn_cfg": {
                    "num_heads": [16, 16],
                    "rotary_emb_dim": [32, 48],
                    "window_size": [1023, -1],
                },
                "tie_embeddings": False,
            }
    else:
        # Default 1-stage config
        config = {
            "arch_layout": ["m4", ["T22"], "m4"],
            "d_model": [1024, 1536],
            "d_intermediate": [0, 4096],
            "vocab_size": 256,
            "ssm_cfg": {"chunk_size": 1, "d_conv": 4, "d_state": 128, "expand": 2},
            "attn_cfg": {
                "num_heads": [16, 16],
                "rotary_emb_dim": [32, 48],
                "window_size": [1023, -1],
            },
            "tie_embeddings": False,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use float32 on CPU, bfloat16 on GPU
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = create_reference_hnet_from_config(
        config, device=device, dtype=dtype
    )
    model.eval()

    # Load weights
    if model_path:
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint, strict=True)
            print(f"✅ Model loaded from {model_path}")
            
        except (FileNotFoundError, RuntimeError) as e:
            print(f"❌ Failed to load model from {model_path}: {e}")
            print("Using random initialization")
    else:
        print("⚠️ No model path provided, using random initialization")

    # ByteTokenizer settings
    BOS_IDX = 254
    EOS_IDX = 255

    def encode_with_bos(text):
        text_bytes = text.encode("utf-8")
        return [BOS_IDX] + list(text_bytes)

    def decode_bytes(tokens):
        filtered = [t for t in tokens if t not in [BOS_IDX, EOS_IDX]]
        try:
            return bytearray(filtered).decode("utf-8", errors="replace")
        except:
            return str(filtered)

    print(
        f"\nSettings: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}"
    )
    print("Enter prompts (empty line to quit):")

    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            if not prompt:
                break

            # Encode with BOS
            prompt_tokens = encode_with_bos(prompt)
            input_ids = torch.tensor([prompt_tokens], device=device)
            cache = model.allocate_inference_cache(
                batch_size=1, max_seqlen=max_tokens + len(prompt_tokens), dtype=dtype
            )

            generated_tokens = []

            # Prefill
            with torch.no_grad():
                mask = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
                output = model(input_ids, mask=mask, inference_params=cache)
                logits = output.logits[0, -1, :]

            print(f"\033[92m{prompt}\033[0m", end="", flush=True)

            # Generate
            for step in range(max_tokens):
                logits_temp = logits / temperature

                # Top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits_temp, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits_temp[indices_to_remove] = -float("inf")

                probs = F.softmax(logits_temp, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                if next_token == EOS_IDX:
                    break

                generated_tokens.append(next_token)

                # Try to decode and print incrementally
                try:
                    partial_text = decode_bytes([next_token])
                    print(partial_text, end="", flush=True)
                except:
                    pass

                # Next step
                current_token = torch.tensor([[next_token]], device=device)
                with torch.no_grad():
                    output = model.step(current_token, cache)
                    cache = output.inference_params  # CRITICAL: Update cache with new state!
                    logits = output.logits[0, -1, :]

            print()  # New line after generation

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError during generation: {e}")


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="H-Net Standalone Implementation - Test and Generate Text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run test with default model path
  python hnet_standalone.py --mode test
  
  # Run test with custom model path  
  python hnet_standalone.py --mode test --model-path weights/custom_model.pt
  
  # Interactive generation mode
  python hnet_standalone.py --mode interactive --model-path weights/Hnet_1stage_L.pt --max-tokens 50
  
  # Test with custom prompts
  python hnet_standalone.py --mode test --prompts "Hello world" "The quick brown fox"
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["test", "interactive"],
        default="test",
        help="Run mode: 'test' for predefined prompts, 'interactive' for user input (default: test)",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/Hnet_1stage_L.pt",
        help="Path to the model checkpoint (.pt file) (default: weights/Hnet_1stage_L.pt)",
    )

    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to the model configuration (.json file) (optional, uses default 1-stage config if not provided)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Maximum number of tokens to generate (default: 20)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter (default: 0.9)",
    )

    parser.add_argument(
        "--prompts", nargs="+", help="Custom prompts for test mode (space-separated)"
    )

    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip model loading and use random weights (for architecture testing)",
    )

    args = parser.parse_args()

    # Handle model path
    model_path = None if args.no_model else args.model_path

    print("H-Net Standalone Implementation")
    print("=" * 40)
    print(f"Mode: {args.mode}")
    print(f"Model path: {model_path or 'None (random weights)'}")
    print(f"Config path: {args.config_path or 'None (using default 1-stage config)'}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")

    try:
        if args.mode == "test":
            test_with_proper_bos(
                model_path=model_path,
                config_path=args.config_path,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                prompts=args.prompts,
            )
        elif args.mode == "interactive":
            interactive_generation(
                model_path=model_path,
                config_path=args.config_path,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
