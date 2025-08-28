"""H-Net recursive wrapper assembling isotropic stacks with dynamic chunking."""

from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp

from .config import HNetConfig
from .routing import ChunkLayer, DeChunkLayer, RoutingModule
from .stack import Isotropic
from .state import HNetState
from .utils import ste_func


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
            self.residual_proj = nnx.Linear(
                self.d_model, self.d_model, dtype=jnp.float32, rngs=rngs
            )
            self.residual_proj.kernel.value = jnp.zeros_like(
                self.residual_proj.kernel.value
            )
            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual

        if stage_idx > 0 and self.d_model - config.d_model[stage_idx - 1] > 0:
            from flax.nnx import Param

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
        cu_seqlens: jnp.ndarray | None = None,
        max_seqlen: int | None = None,
        mask: jnp.ndarray | None = None,
        inference_params: HNetState | None = None,
        **mixer_kwargs,
    ):
        assert mask is not None or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Provide mask or (cu_seqlens, max_seqlen)"
        if inference_params is None:
            inference_params = HNetState(main_network_state=None)
        else:
            assert mask is not None

        D = hidden_states.shape[-1]
        early_dims = hidden_states.shape[:-1]
        if self.pad_dimension is not None:
            pad_expanded = jnp.broadcast_to(
                self.pad_dimension.value,
                (*early_dims, self.pad_dimension.value.shape[-1]),
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

        bpred_output = self.routing_module(
            hs,
            cu_seqlens=cu_seqlens,
            mask=mask,
            inference_params=inference_params.routing_module_state,
        )
        hs_chunk, next_cu, next_max_L, next_mask = self.chunk_layer(
            hs, bpred_output.boundary_mask, cu_seqlens, mask=mask
        )

        hs_inner, prev_boundary_predictions = self.main_network(
            hs_chunk,
            cu_seqlens=next_cu,
            max_seqlen=next_max_L,
            mask=next_mask,
            inference_params=inference_params.main_network_state,
            **mixer_kwargs,
        )

        hs = self.dechunk_layer(
            hs_inner,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            next_cu,
            mask=mask,
            inference_params=inference_params.dechunk_state,
        )

        hs = self.residual_func(
            hs.astype(residual.dtype), residual, bpred_output.selected_probs
        ).astype(hs.dtype)

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
                (*hidden_states.shape[:-1], self.pad_dimension.value.shape[-1]),
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
