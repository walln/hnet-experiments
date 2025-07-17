# Copyright (c) 2025, Nick Wall.
# JAX implementation of Mamba2 (State Space Model)
# Based on the original implementation by Tri Dao and Albert Gu

from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from einops import rearrange, repeat


def softplus(x: jax.Array) -> jax.Array:
    """Softplus activation function."""
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0)


def silu(x: jax.Array) -> jax.Array:
    """SiLU (Swish) activation function."""
    return x * jax.nn.sigmoid(x)


@dataclass
class InferenceCache:
    """Cache for autoregressive inference."""

    conv_state: jax.Array  # (batch, d_inner + 2 * d_state, d_conv)
    ssm_state: jax.Array  # (batch, nheads, headdim, d_state)

    @staticmethod
    def alloc(
        batch_size: int,
        d_inner: int,
        d_state: int,
        d_conv: int,
        nheads: int,
        headdim: int,
    ) -> "InferenceCache":
        """Allocate a new inference cache."""
        conv_state = jnp.zeros((batch_size, d_inner + 2 * d_state, d_conv))
        ssm_state = jnp.zeros((batch_size, nheads, headdim, d_state))
        return InferenceCache(conv_state, ssm_state)


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
    assert x.shape[1] % chunk_size == 0, (
        f"Sequence length {x.shape[1]} must be divisible by chunk_size {chunk_size}"
    )

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
    else:
        # Ensure initial states have the right shape
        if initial_states.shape[1] == 1 and len(initial_states.shape) == 5:
            # Reshape from (batch, 1, nheads, headdim, d_state) to (batch, 1, headdim, d_state)
            initial_states = rearrange(initial_states, "b 1 h p n -> b 1 h p n")

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

    return Y, final_state


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
        chunk_size: int = 256,
        layer_idx: int | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize Mamba2 layer.

        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Local convolution kernel size
            expand: Expansion factor for inner dimension
            headdim: Head dimension for multi-head SSM
            ngroups: Number of groups for grouped SSM
            A_init_range: Range for initializing A matrix
            D_has_hdim: Whether D parameter has head dimension
            rmsnorm: Whether to use RMSNorm
            norm_before_gate: Whether to normalize before gating
            dt_min: Minimum dt value
            dt_max: Maximum dt value
            dt_init_floor: Floor for dt initialization
            bias: Whether to use bias in linear projections
            conv_bias: Whether to use bias in convolution
            chunk_size: Chunk size for scan
            layer_idx: Layer index (for caching)
            rngs: Random number generators
        """
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
        # z and x are for gated MLP, B and C are SSM parameters, dt is time step
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nnx.Linear(self.d_model, d_in_proj, use_bias=bias, rngs=rngs)

        # Convolution for local context mixing
        # Applied to [x, B, C] together
        self.conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state

        # Initialize convolution weights
        self.conv_weight = nnx.Param(
            nnx.initializers.uniform(scale=0.1)(
                rngs.params(), (self.d_conv, 1, self.conv_dim)
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
            self.norm = nnx.RMSNorm(
                self.d_ssm,
                epsilon=1e-5,
                rngs=rngs,
            )

        # Output projection
        self.out_proj = nnx.Linear(self.d_inner, self.d_model, use_bias=bias, rngs=rngs)

    def _conv1d(
        self, x: jax.Array, cache: InferenceCache | None = None
    ) -> tuple[jax.Array, jax.Array | None]:
        """
        Apply 1D depthwise convolution using JAX operations.

        Args:
            x: Input of shape (batch, seq_len, channels)
            cache: Optional cache for inference mode

        Returns:
            output: Convolved output
            updated_conv_state: Updated convolution state for caching
        """
        if cache is not None:
            # Inference mode: use cached convolution state
            assert x.shape[1] == 1, "Inference mode expects single token"
            _batch_size = x.shape[0]

            # Update cache: shift left and append new value
            conv_state = cache.conv_state  # (batch, channels, d_conv)
            conv_state = jnp.roll(conv_state, shift=-1, axis=-1)
            conv_state = conv_state.at[:, :, -1].set(x[:, 0, :])

            # Apply convolution weights
            conv_weight = rearrange(self.conv_weight.value, "d 1 c -> c d")
            output = jnp.sum(conv_state * conv_weight, axis=-1)  # (batch, channels)

            if self.conv_bias is not None:
                output = output + self.conv_bias.value

            return output[:, None, :], conv_state
        else:
            # Training mode: full sequence convolution
            batch, seq_len, channels = x.shape

            # Prepare input for depthwise conv
            # Need shape: (batch, seq_len, channels)

            # Prepare weights: (d_conv, 1, channels) -> (d_conv, channels)
            weights = self.conv_weight.value.squeeze(1)  # (d_conv, channels)

            # Manual convolution implementation for depthwise
            # Pad the input
            x_padded = jnp.pad(
                x, ((0, 0), (self.d_conv - 1, 0), (0, 0)), mode="constant"
            )

            # Apply convolution manually
            output = jnp.zeros_like(x)
            for i in range(self.d_conv):
                # For each position in the kernel
                output += x_padded[:, i : i + seq_len, :] * weights[i : i + 1, :]

            if self.conv_bias is not None:
                output = output + self.conv_bias.value

            return output, None

    def __call__(
        self, u: jax.Array, cache: InferenceCache | None = None
    ) -> tuple[jax.Array, InferenceCache | None]:
        """
        Forward pass of Mamba2 layer.

        Args:
            u: Input tensor of shape (batch, seq_len, d_model)
            cache: Optional inference cache

        Returns:
            output: Output tensor of shape (batch, seq_len, d_model)
            updated_cache: Updated cache if in inference mode
        """
        batch, seq_len, _ = u.shape

        # Check if we're in inference mode
        if cache is not None and seq_len == 1:
            return self._step(u, cache)

        # Input projection
        zxbcdt = self.in_proj(u)  # (batch, seq_len, d_in_proj)

        # Split into components
        z, xBC, dt = jnp.split(
            zxbcdt,
            [self.d_inner, self.d_inner + self.d_ssm + 2 * self.ngroups * self.d_state],
            axis=-1,
        )

        # Apply convolution to [x, B, C]
        xBC = silu(self._conv1d(xBC)[0])

        # Split convolution output
        x, B, C = jnp.split(
            xBC, [self.d_ssm, self.d_ssm + self.ngroups * self.d_state], axis=-1
        )

        # Prepare SSM inputs
        x = rearrange(x, "b l (h p) -> b l h p", h=self.nheads, p=self.headdim)
        dt = softplus(dt + self.dt_bias.value)  # (batch, seq_len, nheads)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups, n=self.d_state)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups, n=self.d_state)

        # Handle grouped B/C - for now just take first group
        if self.ngroups > 1:
            B = B[:, :, 0, :]
            C = C[:, :, 0, :]
        else:
            B = B[:, :, 0, :]
            C = C[:, :, 0, :]

        # Expand B and C to match head dimension
        B = repeat(B, "b l n -> b l h n", h=self.nheads)
        C = repeat(C, "b l n -> b l h n", h=self.nheads)

        # Apply dt to x
        x = x * dt[:, :, :, None]

        # Compute SSM with SSD algorithm
        A = -jnp.exp(self.A_log.value)  # (nheads,)
        A_expanded = repeat(A, "h -> b l h", b=batch, l=seq_len)

        y, final_state = ssd(x, A_expanded * dt, B, C, self.chunk_size)

        # Apply D parameter (skip connection)
        if self.D_has_hdim:
            D = rearrange(self.D.value, "(h p) -> h p", h=self.nheads, p=self.headdim)
            y = y + D[None, None, :, :] * x / dt[:, :, :, None]
        else:
            D = self.D.value  # (nheads,)
            y = y + D[None, None, :, None] * x / dt[:, :, :, None]

        # Reshape back
        y = rearrange(y, "b l h p -> b l (h p)")

        # Apply normalization
        if self.rmsnorm:
            y = self.norm(y)
        y = y * silu(z)

        # Output projection
        out = self.out_proj(y)

        # Create new cache if needed
        if cache is not None:
            # Store convolution state
            conv_state = rearrange(xBC, "b l d -> b d l")[:, :, -self.d_conv :]
            new_cache = InferenceCache(conv_state, final_state)
            return out, new_cache

        return out, None

    def _step(
        self, u: jax.Array, cache: InferenceCache
    ) -> tuple[jax.Array, InferenceCache]:
        """
        Single step for autoregressive inference.

        Args:
            u: (batch, 1, d_model) single token input
            cache: Current cache state

        Returns:
            output: (batch, 1, d_model) output
            updated_cache: Updated cache
        """
        assert u.shape[1] == 1, "Step mode expects single token"
        _batch = u.shape[0]

        # Input projection
        zxbcdt = self.in_proj(u[:, 0, :])  # (batch, d_in_proj)

        # Split into components
        z, xBC, dt = jnp.split(
            zxbcdt,
            [self.d_inner, self.d_inner + self.d_ssm + 2 * self.ngroups * self.d_state],
            axis=-1,
        )

        # Update convolution state and apply conv
        xBC_conv, updated_conv_state = self._conv1d(xBC[:, None, :], cache)
        xBC = silu(xBC_conv[:, 0, :])

        # conv_state should not be None in inference mode
        assert updated_conv_state is not None, (
            "Conv state should not be None in step mode"
        )
        conv_state = updated_conv_state

        # Split convolution output
        x, B, C = jnp.split(
            xBC, [self.d_ssm, self.d_ssm + self.ngroups * self.d_state], axis=-1
        )

        # Process SSM parameters
        x = rearrange(x, "b (h p) -> b h p", h=self.nheads, p=self.headdim)
        dt = softplus(dt + self.dt_bias.value)  # (batch, nheads)
        B = rearrange(B, "b (g n) -> b g n", g=self.ngroups, n=self.d_state)
        C = rearrange(C, "b (g n) -> b g n", g=self.ngroups, n=self.d_state)

        # Handle groups
        if self.ngroups > 1:
            B = B[:, 0, :]
            C = C[:, 0, :]
        else:
            B = B[:, 0, :]
            C = C[:, 0, :]

        # Compute SSM step
        A = -jnp.exp(self.A_log.value)  # (nheads,)

        # Compute dA = exp(A * dt)
        dA = jnp.exp(dt * A[None, :])  # (batch, nheads)

        # Update SSM state: h_new = h * dA + x * dt * B
        # x: (batch, nheads, headdim)
        # dt: (batch, nheads)
        # B: (batch, d_state)
        dBx = jnp.einsum("bh, bn, bhp -> bhpn", dt, B, x)

        ssm_state = cache.ssm_state * dA[:, :, None, None] + dBx

        # Compute output: y = C @ h
        y = jnp.einsum("bhpn, bn -> bhp", ssm_state, C)

        # Apply D parameter
        if self.D_has_hdim:
            D = rearrange(self.D.value, "(h p) -> h p", h=self.nheads, p=self.headdim)
            y = y + D[None, :, :] * x
        else:
            D = self.D.value  # (nheads,)
            y = y + D[None, :, None] * x

        # Reshape back
        y = rearrange(y, "b h p -> b (h p)")

        # Apply normalization
        if self.rmsnorm:
            y = self.norm(y)
        y = y * silu(z)

        # Output projection
        out = self.out_proj(y)

        # Create updated cache
        updated_cache = InferenceCache(conv_state, ssm_state)

        return out[:, None, :], updated_cache


class Mamba2Block(nnx.Module):
    """
    A Mamba2 block with normalization and residual connection.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        layer_idx: int | None = None,
        norm_epsilon: float = 1e-5,
        chunk_size: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize Mamba2 block with norm and residual."""
        self.norm = nnx.RMSNorm(d_model, epsilon=norm_epsilon, rngs=rngs)
        self.mamba = Mamba2Layer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            layer_idx=layer_idx,
            rngs=rngs,
        )

    def __call__(
        self, x: jax.Array, cache: InferenceCache | None = None
    ) -> tuple[jax.Array, InferenceCache | None]:
        """Forward pass with residual connection."""
        residual = x
        x = self.norm(x)
        x, updated_cache = self.mamba(x, cache)
        return residual + x, updated_cache
