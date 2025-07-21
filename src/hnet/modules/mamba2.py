# Copyright (c) 2025, Nick Wall.
# JAX implementation of Mamba2 (State Space Model)
# Based on the original implementation by Tri Dao and Albert Gu
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from einops import rearrange, repeat

from .cache import Mamba2CacheState
from .config import Mamba2Config


def softplus(x: jax.Array) -> jax.Array:
    return jax.nn.softplus(x)


def silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


# For backward compatibility, alias the new cache structure
InferenceCache = Mamba2CacheState


def split_tensor(x, split_sizes, dim):
    """
    Splits the tensor x into multiple tensors based on split_sizes along dimension dim.

    Args:
        x: The input tensor to split.
        split_sizes: The sizes to split the tensor into.
        dim: The dimension along which to split.

    Returns:
        List of split tensors.
    """
    splits = []
    start = 0
    for size in split_sizes:
        end = start + size
        indices = jnp.arange(start, end)
        split = jnp.take(x, indices=indices, axis=dim)
        splits.append(split)
        start = end
    return splits


def segsum(x: jax.Array) -> jax.Array:
    """
    Stable segment sum calculation.

    `exp(segsum(A))` produces a 1-semiseparable matrix, which is equivalent to a scalar SSM.

    Args:
        x: (..., d) input array

    Returns:
        (..., d, d) output where out[..., i, j] = sum(x[..., i] for i in [j+1, i])
    """
    T = x.shape[-1]

    # Expand x to (..., d, e) where e=T
    x = repeat(x, "... d -> ... d e", e=T)

    # Create lower triangular mask
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)

    # Apply mask
    x = jnp.where(mask, x, 0)

    # Cumulative sum along second-to-last axis
    x_segsum = jnp.cumsum(x, axis=-2)

    # Apply mask to keep only lower triangular part
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    x_segsum = jnp.where(mask, x_segsum, -jnp.inf)

    return x_segsum


def ssd(
    x: jax.Array,
    A: jax.Array,
    B: jax.Array,
    C: jax.Array,
    chunk_size: int,
    initial_states: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Structured State Space Duality (SSD) - the core Mamba2 algorithm.

    Args:
        x: (batch, seqlen, n_heads, d_head) - Input tensor
        A: (batch, seqlen, n_heads) - A matrix (diagonal values)
        B: (batch, seqlen, n_heads, d_state) - B matrix
        C: (batch, seqlen, n_heads, d_state) - C matrix
        chunk_size: Size of each chunk for parallel scan
        initial_states: Optional (batch, 1, n_heads, d_head, d_state) initial states

    Returns:
        Y: (batch, seqlen, n_heads, d_head) - Output
        final_state: (batch, n_heads, d_head, d_state) - Final states
    """
    original_seq_len = x.shape[1]

    # Pad sequence length to be divisible by chunk_size if necessary
    pad_len = (chunk_size - original_seq_len % chunk_size) % chunk_size
    if pad_len > 0:
        # Pad inputs with zeros
        x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0), (0, 0)), mode="constant")
        A = jnp.pad(A, ((0, 0), (0, pad_len), (0, 0)), mode="constant")
        B = jnp.pad(B, ((0, 0), (0, pad_len), (0, 0), (0, 0)), mode="constant")
        C = jnp.pad(C, ((0, 0), (0, pad_len), (0, 0), (0, 0)), mode="constant")

    # Rearrange into chunks
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]

    # Rearrange A for cumulative sum
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = jnp.cumsum(A, axis=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = jnp.exp(segsum(x=A))  # (b, h, c, l, l) - lower triangular

    # Compute Y_diag = C @ B @ L @ x
    Y_diag = jnp.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 2. Compute the state for each intra-chunk
    # decay_states: (b, h, c, l)
    decay_states = jnp.exp(A_cumsum[..., -1:] - A_cumsum)

    # states: (b, c, h, p, n)
    states = jnp.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 3. Compute the inter-chunk SSM recurrence
    if initial_states is None:
        initial_states = jnp.zeros_like(states[:, :1])

    states = jnp.concatenate([initial_states, states], axis=1)

    # Compute decay between chunks
    A_cumsum_padded = jnp.pad(
        A_cumsum[:, :, :, -1], ((0, 0), (0, 0), (1, 0)), mode="constant"
    )
    decay_chunk = jnp.exp(segsum(x=A_cumsum_padded))  # (b, h, c+1, c+1)

    # Apply decay to states
    new_states = jnp.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)

    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    state_decay_out = jnp.exp(A_cumsum)  # (b, h, c, l)
    Y_off = jnp.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)

    # Add intra-chunk and inter-chunk outputs
    Y = Y_diag + Y_off
    Y = rearrange(Y, "b c l h p -> b (c l) h p")

    # Remove padding if it was added
    if pad_len > 0:
        Y = Y[:, :original_seq_len, :, :]

    return Y, final_state


class RMSNorm(nnx.Module):
    """RMS Normalization with optional gating."""

    def __init__(self, d: int, eps: float = 1e-5, *, rngs: nnx.Rngs):
        self.d = d
        self.eps = eps
        self.weight = nnx.Param(jnp.ones((d,)))

    def __call__(self, x: jax.Array, z: jax.Array | None = None) -> jax.Array:
        """
        Apply RMS normalization with optional gating.

        Args:
            x: Input tensor
            z: Optional gating tensor
        """
        # Apply gating first if z is provided
        if z is not None:
            x = x * silu(z)

        # RMS normalization
        mean_sq = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        rsqrt = jax.lax.rsqrt(mean_sq + self.eps)
        return x * rsqrt * self.weight.value


class Mamba2Layer(nnx.Module):
    """
    A single Mamba2 layer implementing the SSD (Structured State Space Duality) framework.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        A_init_range: tuple[float, float] = (1, 16),
        D_has_hdim: bool = False,
        rmsnorm: bool = True,
        norm_before_gate: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
        chunk_size: int = 64,  # Changed default to match reference
        layer_idx: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize Mamba2 layer with reference-compatible defaults."""
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        # For compatibility - use d_inner as d_ssm
        self.d_ssm = self.d_inner

        assert self.d_ssm % self.headdim == 0, "d_ssm must be divisible by headdim"
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate

        # Input projection: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nnx.Linear(self.d_model, d_in_proj, use_bias=bias, rngs=rngs)

        # Convolution for local context mixing
        self.conv_dim = self.d_inner + 2 * self.d_state

        # Initialize convolution weights to match reference format
        self.conv_weight = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(
                rngs.params(), (self.conv_dim, 1, self.d_conv)
            )
        )
        if conv_bias:
            self.conv_bias = nnx.Param(jnp.zeros(self.conv_dim))
        else:
            self.conv_bias = None

        # Initialize log dt bias
        dt = jnp.exp(
            jax.random.uniform(rngs.params(), (self.nheads,))
            * (jnp.log(dt_max) - jnp.log(dt_min))
            + jnp.log(dt_min)
        )
        dt = jnp.clip(dt, min=dt_init_floor)
        # Inverse of softplus
        inv_dt = dt + jnp.log(jnp.expm1(dt))
        self.dt_bias = nnx.Param(inv_dt)

        # SSM parameters
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = jax.random.uniform(
            rngs.params(),
            (self.nheads,),
            minval=A_init_range[0],
            maxval=A_init_range[1],
        )
        self.A_log = nnx.Param(jnp.log(A))

        # D "skip" parameter
        D_shape = (self.d_ssm if self.D_has_hdim else self.nheads,)
        self.D = nnx.Param(jnp.ones(D_shape))

        # Normalization
        if self.rmsnorm:
            self.norm = RMSNorm(self.d_inner, rngs=rngs)
        else:
            self.norm = None

        # Output projection
        self.out_proj = nnx.Linear(self.d_inner, self.d_model, use_bias=bias, rngs=rngs)

    def __call__(
        self,
        u: jax.Array,
        step_mode: bool = False,
        h: InferenceCache | None = None,
    ) -> tuple[jax.Array, InferenceCache | None]:
        """
        Forward pass matching reference implementation.

        Args:
            u: Input tensor of shape (batch, seq_len, d_model)
            step_mode: Whether in step mode for inference
            h: Optional inference cache
        """
        if step_mode and h is not None:
            return self.step(u, h)

        # Input projection
        zxbcdt = self.in_proj(u)

        # Split into components using reference split logic
        split_sizes = [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads]
        z, xBC, dt = split_tensor(zxbcdt, split_sizes, dim=-1)

        # Apply dt bias
        dt = softplus(dt + self.dt_bias.value)

        # Convolution - simplified to match reference behavior
        # Pad for causal convolution
        pad_amount = self.d_conv - 1
        xBC_padded = jnp.pad(xBC, ((0, 0), (pad_amount, 0), (0, 0)), mode="constant")

        # Apply depthwise convolution manually
        conv_weight = self.conv_weight.value  # (conv_dim, 1, d_conv)
        conv_weight = jnp.squeeze(conv_weight, axis=1).T  # (d_conv, conv_dim)

        xBC_conv = jnp.zeros_like(xBC)
        for i in range(self.d_conv):
            xBC_conv += (
                xBC_padded[:, i : i + xBC.shape[1], :] * conv_weight[i : i + 1, :]
            )

        if self.conv_bias is not None:
            xBC_conv += self.conv_bias.value

        xBC = silu(xBC_conv)

        # Split convolution output
        split_sizes = [self.d_inner, self.d_state, self.d_state]
        x, B, C = split_tensor(xBC, split_sizes, dim=-1)

        # Reshape for multi-head processing
        x = rearrange(x, "b l (h p) -> b l h p", h=self.nheads, p=self.headdim)

        # Get A parameter
        A = -jnp.exp(self.A_log.value)  # (nheads,)

        # Run SSD algorithm
        y, ssm_state = ssd(
            x * dt[..., None],  # Apply dt scaling to input
            A[None, None, :] * dt,  # Broadcast A and apply dt scaling
            rearrange(B, "b l n -> b l 1 n"),  # Add head dimension
            rearrange(C, "b l n -> b l 1 n"),  # Add head dimension
            self.chunk_size,
        )

        # Apply D parameter (skip connection)
        if self.D_has_hdim:
            D = rearrange(self.D.value, "(h p) -> h p", h=self.nheads, p=self.headdim)
            y = y + D[None, None, :, :] * x
        else:
            D = self.D.value  # (nheads,)
            y = y + D[None, None, :, None] * x

        # Reshape back
        y = rearrange(y, "b l h p -> b l (h p)")

        # Apply normalization with gating
        y = self.norm(y, z) if self.norm is not None else y * silu(z)

        # Output projection
        y = self.out_proj(y)

        # Create cache for inference
        if step_mode:
            # Store convolution state - match reference format
            conv_state = rearrange(xBC, "b l d -> b d l")  # (batch, conv_dim, seqlen)
            # Pad or truncate to d_conv length
            pad_amount = self.d_conv - xBC.shape[1]
            if pad_amount > 0:
                conv_state = jnp.pad(
                    conv_state, ((0, 0), (0, 0), (pad_amount, 0)), mode="constant"
                )
            else:
                conv_state = conv_state[:, :, -self.d_conv :]

            h = InferenceCache(conv_state, ssm_state)

        return y, h

    def step(self, u: jax.Array, h: InferenceCache) -> tuple[jax.Array, InferenceCache]:
        """
        Single step inference matching reference implementation.

        Args:
            u: (batch, 1, d_model) single token input
            h: Current cache state
        """
        assert u.shape[1] == 1, "Only one token can be decoded per inference step"

        # Input projection
        zxbcdt = self.in_proj(jnp.squeeze(u, axis=1))

        # Split components
        split_sizes = [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads]
        z, xBC, dt = split_tensor(zxbcdt, split_sizes, dim=-1)

        # Advance convolution input - match reference logic
        rolled_conv_state = jnp.roll(h.conv_state, shift=-1, axis=-1)
        updated_conv_state = rolled_conv_state.at[:, :, -1].set(xBC)
        h = InferenceCache(conv_state=updated_conv_state, ssm_state=h.ssm_state)

        # Convolution step
        conv_weight = rearrange(self.conv_weight.value, "d 1 w -> d w")
        xBC = jnp.sum(h.conv_state * conv_weight, axis=-1)

        if self.conv_bias is not None:
            xBC += self.conv_bias.value

        xBC = silu(xBC)

        # Split convolution output
        split_sizes = [self.d_inner, self.d_state, self.d_state]
        x, B, C = split_tensor(xBC, split_sizes, dim=-1)

        # Calculate A
        A = -jnp.exp(self.A_log.value)

        # SSM step
        dt = softplus(dt + self.dt_bias.value)

        # Compute dA
        dA = jnp.exp(dt * A)

        # Reshape x for multi-head
        x = rearrange(x, "b (h p) -> b h p", h=self.nheads, p=self.headdim)

        # Update SSM state
        dBx = jnp.einsum("bh, bn, bhp -> bhpn", dt, B, x)
        updated_ssm_state = h.ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx
        h = InferenceCache(conv_state=h.conv_state, ssm_state=updated_ssm_state)

        # Compute output
        y = jnp.einsum("bhpn, bn -> bhp", updated_ssm_state, C)

        # Apply D parameter
        if self.D_has_hdim:
            D = rearrange(self.D.value, "(h p) -> h p", h=self.nheads, p=self.headdim)
            y = y + D[None, :, :] * x
        else:
            D = self.D.value  # (nheads,)
            y = y + rearrange(D, "h -> h 1") * x

        # Reshape back
        y = rearrange(y, "b h p -> b (h p)")

        # Apply normalization with gating
        y = self.norm(y, z) if self.norm is not None else y * silu(z)

        # Output projection
        y = self.out_proj(y)

        y = jnp.expand_dims(y, axis=1)

        return y, h


class Mamba2Block(nnx.Module):
    """
    A Mamba2 block with normalization and residual connection.
    """

    def __init__(
        self,
        config: Mamba2Config,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize Mamba2 block with norm and residual."""
        self.config = config
        self.norm = nnx.RMSNorm(config.d_model, epsilon=config.norm_epsilon, rngs=rngs)
        self.mamba = Mamba2Layer(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            headdim=config.headdim,
            ngroups=config.ngroups,
            chunk_size=config.chunk_size,
            layer_idx=config.layer_idx,
            A_init_range=config.A_init_range,
            D_has_hdim=config.D_has_hdim,
            rmsnorm=config.rmsnorm,
            norm_before_gate=config.norm_before_gate,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init_floor=config.dt_init_floor,
            bias=config.bias,
            conv_bias=config.conv_bias,
            rngs=rngs,
        )

    def __call__(
        self, x: jax.Array, cache: Mamba2CacheState | None = None
    ) -> tuple[jax.Array, Mamba2CacheState | None]:
        """Forward pass with residual connection."""
        residual = x
        x = self.norm(x)
        x, updated_cache = self.mamba(x, step_mode=(cache is not None), h=cache)
        return residual + x, updated_cache
