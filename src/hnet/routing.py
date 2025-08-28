"""Dynamic chunking components: Router, Chunk, Dechunk layers."""

from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp
from jax import lax

from .state import DeChunkState, RoutingModuleOutput, RoutingModuleState


class RoutingModule(nnx.Module):
    def __init__(self, d_model: int, dtype=jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0)):
        self.d_model = d_model
        self.q_proj_layer = nnx.Linear(
            d_model, d_model, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.k_proj_layer = nnx.Linear(
            d_model, d_model, use_bias=False, dtype=dtype, rngs=rngs
        )

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
        cu_seqlens: jnp.ndarray | None = None,
        mask: jnp.ndarray | None = None,
        inference_params: RoutingModuleState | None = None,
    ) -> RoutingModuleOutput:
        assert (mask is not None) or (cu_seqlens is not None), (
            "Provide mask or cu_seqlens"
        )
        if inference_params is not None:
            assert mask is not None
            assert (~inference_params.has_seen_tokens).all()
            assert cu_seqlens is None

        hs = hidden_states[None, :, :] if cu_seqlens is not None else hidden_states

        hs_dtype = hs.dtype
        w_dtype = self.q_proj_layer.kernel.value.dtype
        if hs_dtype != w_dtype:
            hs = hs.astype(w_dtype)

        q_proj = self.q_proj_layer(hs[:, :-1])
        k_proj = self.k_proj_layer(hs[:, 1:])

        q_norm = q_proj / jnp.linalg.norm(q_proj, axis=-1, keepdims=True)
        k_norm = k_proj / jnp.linalg.norm(k_proj, axis=-1, keepdims=True)

        cos_sim = jnp.sum(q_norm * k_norm, axis=-1)
        boundary_prob = jnp.clip((1 - cos_sim) / 2, 0.0, 1.0)

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
            assert mask is not None
            has_mask = mask.any(axis=-1)
            inference_params.has_seen_tokens = (
                has_mask | inference_params.has_seen_tokens
            )
            last_mask = jnp.clip(mask.sum(axis=-1) - 1, a_min=0)
            idx_b = jnp.arange(hidden_states.shape[0])
            last_h = hidden_states[idx_b, last_mask]
            inference_params.last_hidden_state = jnp.where(
                has_mask[:, None], last_h, inference_params.last_hidden_state
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
        hs = hidden_states.squeeze(1)
        hs_dtype = hs.dtype
        w_dtype = self.q_proj_layer.kernel.value.dtype
        if hs_dtype != w_dtype:
            hs = hs.astype(w_dtype)

        q_proj = self.q_proj_layer(inference_params.last_hidden_state)
        k_proj = self.k_proj_layer(hs)

        q_norm = q_proj / jnp.linalg.norm(q_proj, axis=-1, keepdims=True)
        k_norm = k_proj / jnp.linalg.norm(k_proj, axis=-1, keepdims=True)

        cos_sim = jnp.sum(q_norm * k_norm, axis=-1)
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

            token_idx = jnp.arange(L)[None, :] + (~boundary_mask).astype(jnp.int32) * L
            seq_sorted_indices = jnp.argsort(token_idx, axis=1)

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
        cu_seqlens: jnp.ndarray | None = None,
        inference_params: DeChunkState | None = None,
        mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if inference_params is None:
            assert mask is not None, "Mask must be provided in prefill"
            first_boundary = (
                boundary_mask[:, 0] if boundary_mask.ndim == 2 else boundary_mask[0]
            )
            assert (
                first_boundary.all()
                if first_boundary.ndim > 0
                else first_boundary.item() == 1
            )

        if boundary_prob.shape[-1] == 2:
            p_full = boundary_prob[..., -1].astype(jnp.float32)
        else:
            p_full = boundary_prob.astype(jnp.float32)
        p_full = jnp.clip(p_full, 1e-4, 1 - 1e-4)

        original_dtype = hidden_states.dtype

        if cu_seqlens is not None:
            B = len(cu_seqlens) - 1
            selected_mask = boundary_mask
            selected_p = p_full[selected_mask]

            sel_counts = []
            for b in range(B):
                s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
                sel_counts.append(int(selected_mask[s:e].sum()))
            sel_cu = [0]
            for c in sel_counts:
                sel_cu.append(sel_cu[-1] + c)
            sel_cu = jnp.array(sel_cu)

            ema_selected = jnp.zeros_like(hidden_states)
            for b in range(B):
                ks, ke = int(sel_cu[b]), int(sel_cu[b + 1])
                if ke > ks:
                    ema_selected = ema_selected.at[ks:ke].set(
                        self._ema_sequence(hidden_states[ks:ke], selected_p[ks:ke])
                    )

            plug_back_idx = jnp.cumsum(boundary_mask) - 1
            out_full = jnp.take_along_axis(
                ema_selected,
                jnp.clip(plug_back_idx, a_min=0)[:, None].repeat(self.d_model, axis=1),
                axis=0,
            )
            return out_full.astype(original_dtype)
        else:
            B, L = boundary_mask.shape
            token_idx = jnp.arange(L)[None, :] + (~boundary_mask).astype(jnp.int32) * L
            seq_sorted_indices = jnp.argsort(token_idx, axis=1)

            num_tokens = boundary_mask.sum(axis=-1)
            M = hidden_states.shape[1]
            selected_hidden = hidden_states
            p_sorted = jnp.take_along_axis(p_full, seq_sorted_indices[:, :M], axis=1)

            ema_selected = jnp.zeros_like(selected_hidden)
            for b in range(B):
                m = int(num_tokens[b])
                if m > 0:
                    ema_selected = ema_selected.at[b, :m].set(
                        self._ema_sequence(selected_hidden[b, :m], p_sorted[b, :m])
                    )

            plug_back_idx = jnp.cumsum(boundary_mask, axis=1) - 1
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
