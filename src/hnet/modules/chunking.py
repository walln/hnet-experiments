# Copyright (c) 2025, Nick Wall.
# JAX implementation of chunking module for dynamic sequence segmentation

from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from einops import rearrange, repeat

from hnet.modules.mamba2 import ssd


def normalize(x: jax.Array, axis: int = -1, eps: float = 1e-12) -> jax.Array:
    """Normalize a tensor along the specified axis."""
    return x / (jnp.linalg.norm(x, axis=axis, keepdims=True) + eps)


@dataclass
class RoutingModuleOutput:
    boundary_prob: jax.Array
    boundary_mask: jax.Array
    selected_probs: jax.Array


@dataclass
class RoutingModuleState:
    """
    The state of the routing module.

    Contains
        - [has_seen_tokens] (batch_size,) bool array. Whether that batch element has processed any tokens yet.
        - [last_hidden_state] (batch_size, d_model) array. The last hidden state of the batch element (used for boundary prediction).
    """

    has_seen_tokens: jax.Array  # (batch_size,)
    last_hidden_state: jax.Array  # (batch_size, d_model)


@dataclass
class DeChunkState:
    """
    The state of the dechunk.

    Contains
        - [last_value] (batch_size, d_model) array. The last value of the batch element (used for the EMA).
    """

    last_value: jax.Array  # (batch_size, d_model)


# Helper function to get sequence indices from cumulative sequence lengths
def get_seq_idx(cu_seqlens: jax.Array, device=None) -> jax.Array:
    """Convert cumulative sequence lengths to sequence indices."""
    # cu_seqlens is [0, len1, len1+len2, ..., total_len]
    # We need to create [0, 0, ..., 1, 1, ..., 2, 2, ...]
    seq_idx = jnp.zeros(cu_seqlens[-1], dtype=jnp.int32)
    for i in range(len(cu_seqlens) - 1):
        seq_idx = seq_idx.at[cu_seqlens[i] : cu_seqlens[i + 1]].set(i)
    return seq_idx


class RoutingModule(nnx.Module):
    def __init__(self, d_model, *, rngs: nnx.Rngs):
        self.d_model = d_model

        # Initialize projections with identity matrix
        self.q_proj_layer = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)
        self.k_proj_layer = nnx.Linear(d_model, d_model, use_bias=False, rngs=rngs)

        # Initialize weights as identity matrices
        self.q_proj_layer.kernel.value = jnp.eye(d_model)
        self.k_proj_layer.kernel.value = jnp.eye(d_model)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        return RoutingModuleState(
            has_seen_tokens=jnp.zeros(batch_size, dtype=jnp.bool_),
            last_hidden_state=jnp.zeros(
                (batch_size, self.d_model), dtype=dtype or jnp.float32
            ),
        )

    def __call__(
        self, hidden_states, cu_seqlens=None, mask=None, inference_params=None
    ):
        assert (mask is not None) or (cu_seqlens is not None), (
            "Either mask or cu_seqlens must be provided"
        )

        if inference_params is not None:
            assert mask is not None, (
                "Mask must be provided if inference_params is provided"
            )
            assert jnp.all(~inference_params.has_seen_tokens), (
                "Cannot have seen tokens when inference_params is provided"
            )

        if cu_seqlens is not None:
            # We are in packed mode, so hidden_states is (T, D). Make it (B, T, D)
            hidden_states = jnp.expand_dims(hidden_states, axis=0)

        cos_sim = jnp.einsum(
            "b l d, b l d -> b l",
            normalize(self.q_proj_layer(hidden_states[:, :-1]), axis=-1),
            normalize(self.k_proj_layer(hidden_states[:, 1:]), axis=-1),
        )
        # this clamp should no-op as long as no precision issues are encountered
        boundary_prob = jnp.clip(((1 - cos_sim) / 2), min=0.0, max=1.0)

        # Force boundary probability of the first element to 1.0
        PAD_PROB = 1.0
        boundary_prob = jnp.pad(
            boundary_prob, ((0, 0), (1, 0)), constant_values=PAD_PROB
        )

        if cu_seqlens is not None:
            boundary_prob = jnp.squeeze(boundary_prob, axis=0)
            boundary_prob = boundary_prob.at[cu_seqlens[:-1]].set(PAD_PROB)

        boundary_prob = jnp.stack(((1 - boundary_prob), boundary_prob), axis=-1)

        selected_idx = jnp.argmax(boundary_prob, axis=-1)

        boundary_mask = selected_idx == 1  # (shape hidden_states.shape[:-1])
        if mask is not None:
            # No invalid tokens can be selected
            boundary_mask = boundary_mask & mask

        if inference_params is not None and mask is not None:
            has_mask = jnp.any(mask, axis=-1)
            inference_params.has_seen_tokens = (
                has_mask | inference_params.has_seen_tokens
            )
            last_mask = jnp.clip(jnp.sum(mask, axis=-1) - 1, min=0)
            inference_params.last_hidden_state = jnp.where(
                has_mask[:, None],
                hidden_states[
                    jnp.arange(hidden_states.shape[0]),
                    last_mask,
                ],
                inference_params.last_hidden_state,
            )

        selected_probs = jnp.take_along_axis(
            boundary_prob, selected_idx[..., None], axis=-1
        )  # (shape hidden_states.shape[:-1], 1)

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,  # (shape hidden_states.shape[:-1], 2)
            boundary_mask=boundary_mask,  # (shape hidden_states.shape[:-1])
            selected_probs=selected_probs,  # (shape hidden_states.shape[:-1], 1)
        )

    def step(self, hidden_states, inference_params):
        # hidden_states is (B, 1, D)
        hidden_states = jnp.squeeze(hidden_states, axis=1)
        cos_sim = jnp.einsum(
            "b d, b d -> b",
            normalize(self.q_proj_layer(inference_params.last_hidden_state), axis=-1),
            normalize(self.k_proj_layer(hidden_states), axis=-1),
        )
        boundary_prob = jnp.clip(((1 - cos_sim) / 2), min=0.0, max=1.0)
        inference_params.last_hidden_state = hidden_states
        boundary_prob = jnp.where(
            inference_params.has_seen_tokens,
            boundary_prob,
            jnp.ones_like(boundary_prob),
        )
        boundary_prob = jnp.stack(((1 - boundary_prob), boundary_prob), axis=-1)

        inference_params.has_seen_tokens = jnp.ones_like(
            inference_params.has_seen_tokens
        )

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,  # (B, 2)
            boundary_mask=boundary_prob[..., 1] > 0.5,  # (B,)
            selected_probs=jnp.max(boundary_prob, axis=-1, keepdims=True),  # (B, 1)
        )


class ChunkLayer(nnx.Module):
    def __call__(self, hidden_states, boundary_mask, cu_seqlens=None, mask=None):
        assert (mask is not None) or (cu_seqlens is not None), (
            "Either mask or cu_seqlens must be provided"
        )

        if cu_seqlens is not None:
            next_hidden_states = hidden_states[boundary_mask]
            boundary_cumsum = jnp.cumsum(boundary_mask)
            next_cu_seqlens = jnp.pad(
                boundary_cumsum[cu_seqlens[1:] - 1], (1, 0), constant_values=0
            )
            next_max_seqlen = int(jnp.max(next_cu_seqlens[1:] - next_cu_seqlens[:-1]))
            next_mask = None
        else:
            next_cu_seqlens = None
            num_tokens = jnp.sum(boundary_mask, axis=-1)
            next_max_seqlen = int(jnp.max(num_tokens))

            L = hidden_states.shape[1]
            token_idx = jnp.arange(L)[None, :] + (~boundary_mask).astype(jnp.int32) * L
            seq_sorted_indices = jnp.argsort(token_idx, axis=1)

            next_hidden_states = jnp.take_along_axis(
                hidden_states, seq_sorted_indices[:, :next_max_seqlen, None], axis=1
            )

            next_mask = jnp.arange(next_max_seqlen)[None, :] < num_tokens[:, None]
            next_max_seqlen = None

        return next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask

    def step(self, hidden_states, boundary_mask):
        return hidden_states[boundary_mask]


class DeChunkLayer(nnx.Module):
    def __init__(
        self,
        d_model,
        dtype=jnp.float32,
        chunk_size=256,
        headdim=32,
        *,
        rngs: nnx.Rngs,
    ):
        self.d_model = d_model
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.headdim = headdim
        assert d_model % self.headdim == 0
        self.nheads = d_model // self.headdim

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        return DeChunkState(
            last_value=jnp.zeros(
                (batch_size, self.d_model), dtype=dtype or jnp.float32
            ),
        )

    def __call__(
        self,
        hidden_states,
        boundary_mask,
        boundary_prob,
        cu_seqlens=None,
        inference_params=None,
        mask=None,
    ):
        if inference_params is None:
            assert mask is not None, (
                "Mask must be provided if inference_params is not provided"
            )
            assert jnp.all(boundary_mask[:, 0]), (
                "First token must be a boundary if running prefill"
            )

        p = jnp.clip(boundary_prob[..., -1], min=1e-4, max=1 - 1e-4)

        if cu_seqlens is not None:
            p = p[boundary_mask]
            p = jnp.expand_dims(p, axis=0)
            _seq_idx = get_seq_idx(cu_seqlens)
        else:
            B, L = boundary_mask.shape
            _seq_idx = None

            token_idx = jnp.arange(L)[None, :] + (~boundary_mask).astype(jnp.int32) * L
            seq_sorted_indices = jnp.argsort(token_idx, axis=1)

            p = jnp.take_along_axis(
                p, seq_sorted_indices[:, : hidden_states.shape[1]], axis=1
            )  # (B, M)

        # Convert to log domain for numerical stability
        dt = jnp.log(1 / (1 - p))
        x = hidden_states / dt[..., None]

        # Use the full SSD algorithm from Mamba2
        # Prepare inputs for SSD
        x = rearrange(x, "b l (h p) -> b l h p", h=self.nheads, p=self.headdim)

        # Pad sequence length to be divisible by chunk_size
        # Note: This padding is safe because:
        # - Padded x values are 0 (no input contribution)
        # - Padded dt values are 0, so decay = exp(A * 0) = 1 (no time evolution)
        # - Padded p values are 0, so B matrix = 0 (no state update)
        # This makes padded positions effectively "invisible" to the algorithm
        seq_len = x.shape[1]
        pad_len = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

        if pad_len > 0:
            # Pad x, dt, and p
            x = jnp.pad(x, ((0, 0), (0, pad_len), (0, 0), (0, 0)), mode="constant")
            dt_expanded = repeat(dt, "b l -> b l h", h=self.nheads)
            dt_expanded = jnp.pad(
                dt_expanded, ((0, 0), (0, pad_len), (0, 0)), mode="constant"
            )
            p_padded = jnp.pad(p, ((0, 0), (0, pad_len)), mode="constant")
        else:
            dt_expanded = repeat(dt, "b l -> b l h", h=self.nheads)
            p_padded = p

        A = -jnp.ones((self.nheads,))
        A_expanded = repeat(A, "h -> b l h", b=x.shape[0], l=x.shape[1])

        # Use p as both B and C matrices (as in original PyTorch code)
        B = rearrange(p_padded, "b l -> b l 1 1")
        C = jnp.ones_like(B)

        # Expand B and C to match expected dimensions
        B = repeat(B, "b l 1 1 -> b l h n", h=self.nheads, n=1)
        C = repeat(C, "b l 1 1 -> b l h n", h=self.nheads, n=1)

        # Run SSD algorithm
        out, _ = ssd(x, A_expanded * dt_expanded, B, C, self.chunk_size)

        # Reshape back and remove padding
        out = rearrange(out, "b l h p -> b l (h p)")
        if pad_len > 0:
            out = out[:, :seq_len]

        if cu_seqlens is not None:
            out = jnp.squeeze(out, axis=0)
            plug_back_idx = jnp.cumsum(boundary_mask) - 1
            out = jnp.take_along_axis(out, plug_back_idx[:, None], axis=0)
        else:
            plug_back_idx = jnp.cumsum(boundary_mask, axis=1) - 1  # (B, L)
            out = jnp.take_along_axis(out, plug_back_idx[..., None], axis=1)

        if inference_params is not None:
            inference_params.last_value = out[:, -1]

        return out

    def step(self, hidden_states, boundary_mask, boundary_prob, inference_params):
        # hidden_states is (B', 1, D), where B' = boundary_mask.sum()
        # boundary_mask is (B,) and boundary_prob is (B, 2)

        B = boundary_mask.shape[0]
        D = hidden_states.shape[-1]

        p = jnp.zeros(B, dtype=hidden_states.dtype)
        p = p.at[boundary_mask].set(
            jnp.clip(boundary_prob[boundary_mask, -1], min=1e-4, max=1 - 1e-4)
        )

        current_hidden_states = jnp.zeros((B, D), dtype=hidden_states.dtype)
        current_hidden_states = current_hidden_states.at[boundary_mask].set(
            jnp.squeeze(hidden_states, axis=1)
        )

        result = (
            p[:, None] * current_hidden_states
            + (1 - p[:, None]) * inference_params.last_value
        )
        inference_params.last_value = result

        return jnp.expand_dims(result, axis=1)


# Register pytrees for JAX transformations
jax.tree_util.register_pytree_node(
    RoutingModuleOutput,
    lambda x: ([x.boundary_prob, x.boundary_mask, x.selected_probs], None),
    lambda _, arrays: RoutingModuleOutput(arrays[0], arrays[1], arrays[2]),
)

jax.tree_util.register_pytree_node(
    RoutingModuleState,
    lambda x: ([x.has_seen_tokens, x.last_hidden_state], None),
    lambda _, arrays: RoutingModuleState(arrays[0], arrays[1]),
)

jax.tree_util.register_pytree_node(
    DeChunkState,
    lambda x: ([x.last_value], None),
    lambda _, arrays: DeChunkState(arrays[0]),
)
