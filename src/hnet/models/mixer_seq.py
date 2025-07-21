from collections import namedtuple
from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from hnet.models.config_hnet import HNetConfig
from hnet.models.hnet import HNet, HNetState
from hnet.modules.chunking import RoutingModuleOutput


@dataclass
class CausalLMOutput:
    logits: jnp.ndarray
    bpred_output: list[RoutingModuleOutput]
    inference_params: HNetState


class HNetForCausalLM(nnx.Module):
    def __init__(
        self,
        config: HNetConfig,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.config = config

        vocab_size = self.config.vocab_size
        d_embed = self.config.d_model[0]

        super().__init__()

        # We consider the HNet as a map (B, L, D[0]) -> (B, L, D[0])
        # Thus, the embedding is defined outside of the HNet.
        self.embeddings = nnx.Embed(
            num_embeddings=vocab_size,
            features=d_embed,
            rngs=rngs,
        )

        self.backbone = HNet(
            config=config,
            # We pass in the stage_idx as an HNet needs to know what
            # depth of the hierarchy it is in.
            stage_idx=0,
            rngs=rngs,
        )
        self.lm_head = nnx.Linear(
            in_features=d_embed,
            out_features=vocab_size,
            use_bias=False,
            rngs=rngs,
        )
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_embeddings:
            # In Flax, we share the embedding matrix directly
            self.lm_head.kernel = self.embeddings.embedding

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids,
        mask=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        **mixer_kwargs,
    ):
        """
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.embeddings(input_ids)

        B, L, D = hidden_states.shape

        assert position_ids is None, (
            "Position ids are not supported for HNet due to the subsampling hierarchical structure"
        )

        if mask is None:
            # Absent a mask, we assume we are running in packed mode
            assert inference_params is None, (
                "Inference params are not supported in packed mode"
            )
            hidden_states = hidden_states.reshape(
                -1, D
            )  # Flatten batch and length dims
            cu_seqlens = jnp.arange(B + 1) * L
            max_seqlen = jnp.array(L, dtype=jnp.int32)
        else:
            cu_seqlens = None
            max_seqlen = None

        hidden_states, bpred_output = self.backbone(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params,
            **mixer_kwargs,
        )

        hidden_states = hidden_states.reshape(B, L, D)

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)

        CausalLMOutput = namedtuple(
            "CausalLMOutput", ["logits", "bpred_output", "inference_params"]
        )
        return CausalLMOutput(
            logits=lm_logits,
            bpred_output=bpred_output,
            inference_params=inference_params,
        )

    def step(self, input_ids, inference_params):
        B = input_ids.shape[0]
        assert B == 1, (
            "HNetForCausalLM step currently only supports batch size 1 -- need to handle different-size lengths for each sample"
        )

        hidden_states = self.embeddings(input_ids)

        hidden_states, bpred_output = self.backbone.step(
            hidden_states, inference_params
        )
        logits = self.lm_head(hidden_states)

        return CausalLMOutput(
            logits=logits, bpred_output=bpred_output, inference_params=inference_params
        )
