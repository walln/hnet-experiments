#!/usr/bin/env python3
"""
JAX implementation of Mamba2 based on the working PyTorch non-Triton version
"""

import math
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Any, Callable
from flax.linen.dtypes import promote_dtype
import einops


def segsum_jax(x):
    """JAX version of segsum - more stable segment sum calculation"""
    T = x.shape[-1]
    x = einops.repeat(x, "... d -> ... d e", e=T)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    x = jnp.where(mask, x, 0)
    x_segsum = jnp.cumsum(x, axis=-2)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    return x_segsum


def ssd_minimal_discrete_jax(X, A, B, C, block_len, initial_states=None):
    """
    JAX version of SSD implementation from working PyTorch version
    
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
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X = einops.rearrange(X, "b (c l) h p -> b c l h p", l=block_len)
    A = einops.rearrange(A, "b (c l) h -> b c l h", l=block_len)
    B = einops.rearrange(B, "b (c l) h n -> b c l h n", l=block_len)
    C = einops.rearrange(C, "b (c l) h n -> b c l h n", l=block_len)

    A = einops.rearrange(A, "b c l h -> b h c l")
    A_cumsum = jnp.cumsum(A, axis=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = jnp.exp(segsum_jax(A))
    Y_diag = jnp.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    decay_states = jnp.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    states = jnp.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1])
    states = jnp.concatenate([initial_states, states], axis=1)
    decay_chunk = jnp.exp(segsum_jax(jnp.pad(A_cumsum[:, :, :, -1], ((0, 0), (0, 0), (1, 0)))))
    new_states = jnp.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    state_decay_out = jnp.exp(A_cumsum)
    Y_off = jnp.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = einops.rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


class RMSNorm(nn.Module):
    """JAX/Flax RMSNorm implementation"""
    eps: float = 1e-5
    
    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, (x.shape[-1],))
        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(variance + self.eps) * weight


class RMSNormGated(nn.Module):
    """JAX/Flax Gated RMSNorm for Mamba2"""
    eps: float = 1e-5
    norm_before_gate: bool = True
    
    @nn.compact
    def __call__(self, x, z=None):
        weight = self.param('weight', nn.initializers.ones, (x.shape[-1],))
        
        if z is not None:
            if self.norm_before_gate:
                variance = jnp.mean(x**2, axis=-1, keepdims=True)
                x = x * jax.lax.rsqrt(variance + self.eps) * weight
                return x * nn.silu(z)
            else:
                x_gated = x * nn.silu(z)
                variance = jnp.mean(x_gated**2, axis=-1, keepdims=True)
                return x_gated * jax.lax.rsqrt(variance + self.eps) * weight
        else:
            variance = jnp.mean(x**2, axis=-1, keepdims=True)
            return x * jax.lax.rsqrt(variance + self.eps) * weight


class Mamba2JAX(nn.Module):
    """JAX/Flax Mamba2 implementation without Triton dependencies"""
    
    d_model: int
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 128
    ngroups: int = 1
    A_init_range: tuple = (1, 16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: tuple = (0.0, float("inf"))
    learnable_init_states: bool = False
    activation: str = "swish"
    bias: bool = False
    conv_bias: bool = True
    chunk_size: int = 256
    layer_idx: Optional[int] = None
    dtype: Any = jnp.float32
    
    def setup(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        
        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Dense(d_in_proj, use_bias=self.bias, dtype=self.dtype)
        
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv(
            features=conv_dim,
            kernel_size=(self.d_conv,),
            feature_group_count=conv_dim,
            use_bias=self.conv_bias,
            padding=self.d_conv - 1,
            dtype=self.dtype
        )
        
        if self.learnable_init_states:
            self.init_states = self.param(
                'init_states', 
                nn.initializers.zeros, 
                (self.nheads, self.headdim, self.d_state)
            )
        
        # Initialize dt_bias
        def dt_bias_init(key, shape):
            dt = jnp.exp(
                jax.random.uniform(key, shape, dtype=jnp.float32) * (math.log(self.dt_max) - math.log(self.dt_min))
                + math.log(self.dt_min)
            )
            dt = jnp.clip(dt, min=self.dt_init_floor)
            # Inverse of softplus
            inv_dt = dt + jnp.log(-jnp.expm1(-dt))
            return inv_dt.astype(self.dtype)
        
        self.dt_bias = self.param('dt_bias', dt_bias_init, (self.nheads,))
        
        # A parameter
        def A_log_init(key, shape):
            A = jax.random.uniform(key, shape, dtype=jnp.float32, minval=self.A_init_range[0], maxval=self.A_init_range[1])
            return jnp.log(A).astype(self.dtype)
        
        self.A_log = self.param('A_log', A_log_init, (self.nheads,))
        
        # D "skip" parameter
        self.D = self.param('D', nn.initializers.ones, (self.nheads,))
        
        # Extra normalization layer right before output projection
        self.norm = RMSNormGated(eps=1e-5, norm_before_gate=False)
        
        self.out_proj = nn.Dense(self.d_model, use_bias=self.bias, dtype=self.dtype)
    
    def __call__(self, u, seq_idx=None, inference_params=None, training=True):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape
        
        # Handle inference mode (simplified for now)
        if inference_params is not None:
            return self.step(u, inference_params)
        
        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -jnp.exp(self.A_log)  # (nheads)
        initial_states = (
            einops.repeat(self.init_states, "h p n -> b h p n", b=batch) 
            if self.learnable_init_states else None
        )
        
        # Split: z (d_inner), xBC (d_inner + 2*ngroups*d_state), dt (nheads)
        split_indices = [self.d_inner, self.d_inner + self.d_inner + 2 * self.ngroups * self.d_state]
        z, xBC, dt = jnp.split(zxbcdt, split_indices, axis=-1)
        dt = nn.softplus(dt + self.dt_bias)  # (B, L, nheads)
        
        # 1D Convolution
        xBC = nn.silu(self.conv1d(xBC))  # (B, L, d_inner + 2 * ngroups * d_state)
        xBC = xBC[:, :seqlen, :]  # Trim padding
        
        # Split into 3 main branches: X, B, C
        x, B, C = jnp.split(xBC, [self.d_inner, self.d_inner + self.ngroups * self.d_state], axis=-1)
        
        # Reshape for SSD computation
        x = einops.rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = einops.rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
        B = jnp.broadcast_to(B, (batch, seqlen, self.nheads, self.d_state))  # Expand groups to heads
        C = einops.rearrange(C, "b l (g n) -> b l g n", g=self.ngroups) 
        C = jnp.broadcast_to(C, (batch, seqlen, self.nheads, self.d_state))  # Expand groups to heads
        
        # Use JAX SSD implementation
        dt_expanded = einops.repeat(dt, "b l h -> b l h p", p=self.headdim)  # (B, L, nheads, headdim)
        x_scaled = x * dt_expanded
        A_scaled = einops.repeat(A, "h -> b l h", b=batch, l=seqlen)  # (B, L, nheads)
        A_dt = A_scaled * dt  # (B, L, nheads)
        
        y, final_state = ssd_minimal_discrete_jax(x_scaled, A_dt, B, C, self.chunk_size, initial_states)
        y = einops.rearrange(y, "b l h p -> b l (h p)")
        
        # Add skip connection
        x_flat = einops.rearrange(x, "b l h p -> b l (h p)")
        D_expanded = einops.repeat(self.D, "h -> h p", p=self.headdim)
        D_expanded = einops.rearrange(D_expanded, "h p -> (h p)")
        y = y + x_flat * D_expanded
        
        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.out_proj(y)
        return out
    
    def step(self, hidden_states, inference_params):
        """Step function for autoregressive generation (simplified)"""
        # For now, just run forward pass
        return self(hidden_states, inference_params=None)


def test_mamba2_jax():
    """Test the JAX Mamba2 implementation"""
    print("Testing Mamba2 JAX implementation...")
    
    # Test parameters similar to H-Net config
    batch, seqlen, d_model = 1, 16, 1024  # Use smaller seqlen for testing
    d_state = 128
    chunk_size = 16
    
    # Create model
    model = Mamba2JAX(
        d_model=d_model,
        d_state=d_state,
        d_conv=4,
        expand=2,
        headdim=64,  # Match real H-Net weights (32 heads * 64 headdim = 2048 d_inner)
        ngroups=1,
        chunk_size=chunk_size,
        dtype=jnp.float32
    )
    
    # Test input
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (batch, seqlen, d_model))
    
    try:
        print(f"Input shape: {x.shape}")
        
        # Initialize parameters
        params = model.init(rng, x)
        print(f"✅ Model initialized successfully")
        
        # Forward pass
        output = model.apply(params, x)
        
        print(f"✅ Forward pass successful!")
        print(f"   Output shape: {output.shape}")
        print(f"   Output mean: {jnp.mean(output):.6f}")
        print(f"   Output std: {jnp.std(output):.6f}")
        print(f"   Contains NaN: {jnp.any(jnp.isnan(output))}")
        print(f"   Contains Inf: {jnp.any(jnp.isinf(output))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Mamba2 JAX Implementation")
    print("=" * 40)
    
    success = test_mamba2_jax()
    
    if success:
        print("\n🎉 Mamba2 JAX implementation works!")
    else:
        print("\n❌ Mamba2 JAX implementation failed.")