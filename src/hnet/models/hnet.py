from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from hnet.modules.chunking import (
    ChunkLayer,
    DeChunkLayer,
    DeChunkState,
    RoutingModule,
    RoutingModuleState,
)
from hnet.modules.config import HNetConfig
from hnet.modules.isotropic import Isotropic, IsotropicInferenceParams


def ste_func(x):
    """Straight-through estimator that returns ones but passes gradients through unchanged."""

    @jax.custom_vjp
    def _ste(x):
        return jnp.ones_like(x)

    def _ste_fwd(x):
        return jnp.ones_like(x), x

    def _ste_bwd(x, g):
        return (g,)

    _ste.defvjp(_ste_fwd, _ste_bwd)
    return _ste(x)


@dataclass
class HNetState:
    encoder_state: IsotropicInferenceParams | None = None
    routing_module_state: RoutingModuleState | None = None
    main_network_state: Union["HNetState", IsotropicInferenceParams] | None = None
    dechunk_state: DeChunkState | None = None
    decoder_state: IsotropicInferenceParams | None = None


class HNet(nnx.Module):
    # Declare attributes for type checking
    encoder: Isotropic | None = None
    main_network: Union["HNet", Isotropic] | None = None
    decoder: Isotropic | None = None
    routing_module: RoutingModule | None = None
    chunk_layer: ChunkLayer | None = None
    dechunk_layer: DeChunkLayer | None = None
    residual_proj: nnx.Linear | None = None
    residual_func: Callable | None = None
    pad_dimension: nnx.Param | None = None

    def __init__(
        self,
        config: HNetConfig,
        stage_idx: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()

        # Add type checks for config
        if config.d_model is None or len(config.d_model) <= stage_idx:
            raise ValueError(f"d_model not configured for stage {stage_idx}")
        if config.arch_layout is None:
            raise ValueError("arch_layout must be provided")

        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]

        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            if not isinstance(arch_layout, list) or len(arch_layout) <= 1:
                raise ValueError(f"Invalid arch_layout for stage {stage_idx}")
            arch_layout = arch_layout[1]

        assert isinstance(arch_layout, list), f"Wrong arch_layout: {arch_layout}"
        if len(arch_layout) == 3:
            sub_model_names = ["encoder", "main_network", "decoder"]
            self.is_innermost = False
        elif len(arch_layout) == 2:
            # Handle 2-element layout: assume first is encoder and second is main_network
            sub_model_names = ["encoder", "main_network"]
            self.is_innermost = False
        elif len(arch_layout) == 1:
            sub_model_names = ["main_network"]
            self.is_innermost = True
        else:
            raise NotImplementedError(
                f"Unsupported arch_layout length: {len(arch_layout)}"
            )

        for _name, _layout in zip(sub_model_names, arch_layout, strict=False):
            if self.is_innermost or _name in ("encoder", "decoder"):
                SubModel = Isotropic
                _stage_idx = stage_idx
                if _name == "encoder":
                    _pos_idx = 0
                elif self.is_innermost:
                    # if innermost, then len(layer_layout) == 1
                    _pos_idx = 0
                elif _name == "decoder":
                    _pos_idx = 2
                else:
                    raise ValueError(f"Unexpected model name: {_name}")

                _sub_model = SubModel(
                    config=config,
                    stage_idx=_stage_idx,
                    pos_idx=_pos_idx,
                    rngs=rngs,
                )
            else:
                SubModel = HNet
                _stage_idx = stage_idx + 1
                _sub_model = SubModel(
                    config=config,
                    stage_idx=_stage_idx,
                    rngs=rngs,
                )

            setattr(self, _name, _sub_model)

        if not self.is_innermost:
            self.routing_module = RoutingModule(self.d_model, rngs=rngs)
            self.chunk_layer = ChunkLayer()
            self.dechunk_layer = DeChunkLayer(self.d_model, rngs=rngs)

            # do the residual in fp32
            self.residual_proj = nnx.Linear(
                self.d_model, self.d_model, dtype=jnp.float32, rngs=rngs
            )
            # Initialize to zeros
            self.residual_proj.kernel.value = jnp.zeros_like(
                self.residual_proj.kernel.value
            )
            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual

        if (
            stage_idx > 0
            and len(config.d_model) > stage_idx - 1
            and self.d_model - config.d_model[stage_idx - 1] > 0
        ):
            self.pad_dimension = nnx.Param(
                jnp.zeros(self.d_model - config.d_model[stage_idx - 1])
            )
        else:
            self.pad_dimension = None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """
        Allocate the inference cache for the HNet.

        Arguments:
            batch_size: int. The number of sequences in the batch.
            max_seqlen: int. The maximum sequence length in the batch.
            dtype: jax.numpy.dtype. The dtype of the inference cache.

        The structure of the inference cache is as follows:
            - [encoder state]
            - [routing module state]
            - [main network state]
            - [dechunk state]
            - [decoder state]
        It is thus a list of length 5.
        """
        if self.is_innermost:
            if self.main_network is None:
                raise ValueError("main_network not initialized")
            return HNetState(
                main_network_state=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                )
            )
        else:
            if (
                self.encoder is None
                or self.routing_module is None
                or self.main_network is None
                or self.dechunk_layer is None
            ):
                raise ValueError("Not all submodules are initialized")

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
                decoder_state=(
                    self.decoder.allocate_inference_cache(
                        batch_size, max_seqlen, dtype=dtype
                    )
                    if self.decoder is not None
                    else None
                ),
            )

    def __call__(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        mask=None,
        inference_params=None,
        **mixer_kwargs,
    ):
        assert mask is not None or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Either mask or cu_seqlens and max_seqlen must be provided"

        if inference_params is None:
            inference_params = HNetState(main_network_state=None)
        else:
            assert mask is not None, (
                "Mask must be provided if inference_params is provided"
            )

        D = hidden_states.shape[-1]
        EARLY_DIMS = hidden_states.shape[:-1]

        if self.pad_dimension is not None:
            pad_shape = [*EARLY_DIMS, self.pad_dimension.value.shape[0]]
            hidden_states = jnp.concatenate(
                (
                    hidden_states,
                    jnp.broadcast_to(self.pad_dimension.value, pad_shape),
                ),
                axis=-1,
            )

        if self.is_innermost:
            if self.main_network is None:
                raise ValueError("main_network not initialized")
            # Defensive programming: runtime type validation
            assert self.main_network is not None
            assert isinstance(self.main_network, Isotropic), (
                f"Expected main_network to be Isotropic, got {type(self.main_network)}"
            )
            hidden_states = self.main_network(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                mask=mask,
                inference_params=inference_params.main_network_state,
                **mixer_kwargs,
            )
            # Fix slice syntax
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        # Non-innermost case
        if (
            self.encoder is None
            or self.routing_module is None
            or self.main_network is None
            or self.chunk_layer is None
            or self.dechunk_layer is None
            or self.residual_proj is None
            or self.residual_func is None
        ):
            raise ValueError("Not all submodules are initialized")

        # Defensive programming: runtime type validation
        assert self.encoder is not None
        assert isinstance(self.encoder, Isotropic), (
            f"Expected encoder to be Isotropic, got {type(self.encoder)}"
        )
        assert self.routing_module is not None
        assert isinstance(self.routing_module, RoutingModule), (
            f"Expected routing_module to be RoutingModule, got {type(self.routing_module)}"
        )
        assert self.main_network is not None
        assert isinstance(self.main_network, HNet | Isotropic), (
            f"Expected main_network to be HNet or Isotropic, got {type(self.main_network)}"
        )
        assert self.chunk_layer is not None
        assert isinstance(self.chunk_layer, ChunkLayer), (
            f"Expected chunk_layer to be ChunkLayer, got {type(self.chunk_layer)}"
        )
        assert self.dechunk_layer is not None
        assert isinstance(self.dechunk_layer, DeChunkLayer), (
            f"Expected dechunk_layer to be DeChunkLayer, got {type(self.dechunk_layer)}"
        )
        assert self.residual_proj is not None
        assert isinstance(self.residual_proj, nnx.Linear), (
            f"Expected residual_proj to be nnx.Linear, got {type(self.residual_proj)}"
        )
        assert self.residual_func is not None
        assert callable(self.residual_func), (
            f"Expected residual_func to be callable, got {type(self.residual_func)}"
        )

        hidden_states = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params.encoder_state,
            **mixer_kwargs,
        )

        hidden_states_for_residual = hidden_states.astype(
            self.residual_proj.kernel.value.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)

        bpred_output = self.routing_module(
            hidden_states,
            cu_seqlens=cu_seqlens,
            mask=mask,
            inference_params=inference_params.routing_module_state,
        )
        hidden_states, next_cu_seqlens, next_max_seqlen, next_mask = self.chunk_layer(
            hidden_states, bpred_output.boundary_mask, cu_seqlens, mask=mask
        )

        hidden_states, prev_boundary_predictions = self.main_network(
            hidden_states,
            cu_seqlens=next_cu_seqlens,
            max_seqlen=next_max_seqlen,
            mask=next_mask,
            inference_params=inference_params.main_network_state,
            **mixer_kwargs,
        )

        hidden_states = self.dechunk_layer(
            hidden_states,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            next_cu_seqlens,
            mask=mask,
            inference_params=inference_params.dechunk_state,
        )

        hidden_states = self.residual_func(
            hidden_states.astype(residual.dtype),
            residual,
            bpred_output.selected_probs,
        ).astype(hidden_states.dtype)

        if self.decoder is not None:
            assert isinstance(self.decoder, Isotropic), (
                f"Expected decoder to be Isotropic, got {type(self.decoder)}"
            )
            hidden_states = self.decoder(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                mask=mask,
                inference_params=inference_params.decoder_state,
                **mixer_kwargs,
            )

        # Fix slice syntax
        hidden_states = hidden_states[..., :D]
        return hidden_states, [bpred_output, *prev_boundary_predictions]

    def step(self, hidden_states, inference_params):
        D = hidden_states.shape[-1]

        if self.pad_dimension is not None:
            pad_shape = [*hidden_states.shape[:-1], self.pad_dimension.value.shape[0]]
            hidden_states = jnp.concatenate(
                (
                    hidden_states,
                    jnp.broadcast_to(self.pad_dimension.value, pad_shape),
                ),
                axis=-1,
            )

        if self.is_innermost:
            if self.main_network is None:
                raise ValueError("main_network not initialized")
            # Defensive programming: runtime type validation
            assert self.main_network is not None
            assert isinstance(self.main_network, Isotropic), (
                f"Expected main_network to be Isotropic, got {type(self.main_network)}"
            )
            hidden_states = self.main_network.step(
                hidden_states, inference_params.main_network_state
            )
            # Fix slice syntax
            hidden_states = hidden_states[..., :D]
            return hidden_states, []

        if (
            self.encoder is None
            or self.routing_module is None
            or self.main_network is None
            or self.chunk_layer is None
            or self.dechunk_layer is None
            or self.residual_proj is None
            or self.residual_func is None
        ):
            raise ValueError("Not all submodules are initialized")

        # Defensive programming: runtime type validation
        assert self.encoder is not None
        assert isinstance(self.encoder, Isotropic), (
            f"Expected encoder to be Isotropic, got {type(self.encoder)}"
        )
        assert self.routing_module is not None
        assert isinstance(self.routing_module, RoutingModule), (
            f"Expected routing_module to be RoutingModule, got {type(self.routing_module)}"
        )
        assert self.main_network is not None
        assert isinstance(self.main_network, HNet | Isotropic), (
            f"Expected main_network to be HNet or Isotropic, got {type(self.main_network)}"
        )
        assert self.chunk_layer is not None
        assert isinstance(self.chunk_layer, ChunkLayer), (
            f"Expected chunk_layer to be ChunkLayer, got {type(self.chunk_layer)}"
        )
        assert self.dechunk_layer is not None
        assert isinstance(self.dechunk_layer, DeChunkLayer), (
            f"Expected dechunk_layer to be DeChunkLayer, got {type(self.dechunk_layer)}"
        )
        assert self.residual_proj is not None
        assert isinstance(self.residual_proj, nnx.Linear), (
            f"Expected residual_proj to be nnx.Linear, got {type(self.residual_proj)}"
        )
        assert self.residual_func is not None
        assert callable(self.residual_func), (
            f"Expected residual_func to be callable, got {type(self.residual_func)}"
        )

        hidden_states = self.encoder.step(hidden_states, inference_params.encoder_state)
        hidden_states_for_residual = hidden_states.astype(
            self.residual_proj.kernel.value.dtype
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
            hidden_states.astype(residual.dtype),
            residual,
            bpred_output.selected_probs,
        ).astype(hidden_states.dtype)

        if self.decoder is not None:
            assert isinstance(self.decoder, Isotropic), (
                f"Expected decoder to be Isotropic, got {type(self.decoder)}"
            )
            hidden_states = self.decoder.step(
                hidden_states, inference_params.decoder_state
            )

        # Fix slice syntax
        hidden_states = hidden_states[..., :D]

        return hidden_states, [bpred_output, *prev_boundary_predictions]
