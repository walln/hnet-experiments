"""Byte-level LM wrapper around H-Net backbone."""

from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp

from .config import HNetConfig
from .model import HNet


class HNetForCausalLM(nnx.Module):
    def __init__(
        self, config: HNetConfig, dtype=jnp.float32, rngs: nnx.Rngs = nnx.Rngs(0)
    ) -> None:
        self.config = config
        d_embed = config.d_model[0]
        self.embeddings = nnx.Embed(config.vocab_size, d_embed, dtype=dtype, rngs=rngs)
        self.backbone = HNet(config=config, stage_idx=0, dtype=dtype, rngs=rngs)
        self.lm_head = nnx.Linear(
            d_embed, config.vocab_size, use_bias=False, dtype=dtype, rngs=rngs
        )
        if config.tie_embeddings:
            self.lm_head.kernel = self.embeddings.embedding

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype=jnp.float32
    ):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        mask: jnp.ndarray | None = None,
        position_ids=None,
        inference_params=None,
        num_last_tokens: int = 0,
        **mixer_kwargs,
    ):
        hidden_states = self.embeddings(input_ids)
        B, L, D = hidden_states.shape

        if mask is None:
            assert inference_params is None
            hidden_states = hidden_states.reshape(-1, D)
            cu = jnp.arange(B + 1) * L
            maxL = jnp.array(L, dtype=jnp.int32)
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
        hs = hs.reshape(B, L, D)

        if num_last_tokens > 0:
            hs = hs[:, -num_last_tokens:]

        w_dtype = self.lm_head.kernel.value.dtype
        logits = self.lm_head(hs.astype(w_dtype)).astype(jnp.float32)
        return logits, bpred_output, inference_params

    def step(self, input_ids: jnp.ndarray, inference_params):
        B = input_ids.shape[0]
        assert B == 1
        hidden_states = self.embeddings(input_ids)
        hidden_states, bpred_output = self.backbone.step(
            hidden_states, inference_params
        )
        w_dtype = self.lm_head.kernel.value.dtype
        logits = self.lm_head(hidden_states.astype(w_dtype)).astype(jnp.float32)
        return logits, bpred_output, inference_params
