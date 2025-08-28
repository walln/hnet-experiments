"""Isotropic stack of mixer blocks, used for encoder/decoder/main sub-networks."""

from __future__ import annotations

import re
from typing import cast

import flax.nnx as nnx
import jax.numpy as jnp

from hnet.config import HNetConfig
from hnet.modules.block import Block
from hnet.modules.mha import CausalMHA
from hnet.modules.mlp import SwiGLU
from hnet.modules.norms import RMSNorm
from hnet.modules.ssm import SSM
from hnet.state import InferenceParams
from hnet.utils import get_stage_cfg


class Isotropic(nnx.Module):
    def __init__(
        self,
        config: HNetConfig,
        pos_idx: int,
        stage_idx: int,
        dtype=jnp.float32,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.stage_idx = stage_idx
        self.d_model = config.d_model[self.stage_idx]
        self.ssm_cfg = get_stage_cfg(config.ssm_cfg, stage_idx)
        self.attn_cfg = get_stage_cfg(config.attn_cfg, stage_idx)

        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]
        arch_layout = arch_layout[pos_idx]

        # Help the type checker: we expect a string layout like "m4T2"
        if not isinstance(arch_layout, str):
            raise TypeError("Expected string arch layout at this position")
        arch_str = arch_layout

        pattern = re.compile(r"([mMtT])(\d+)")
        layout_parse = pattern.findall(arch_str)
        layers: list[Block] = []
        layer_idx = 0
        for arch, n_layer_str in layout_parse:
            n_layer = int(n_layer_str)
            for _ in range(n_layer):
                if arch in ("t", "T"):
                    mixer = CausalMHA(
                        self.d_model,
                        num_heads=self.attn_cfg.get("num_heads", 8),
                        rotary_emb_dim=self.attn_cfg.get("rotary_emb_dim", 0),
                        window_size=self.attn_cfg.get("window_size", -1),
                        dtype=dtype,
                        layer_idx=layer_idx,
                        rngs=rngs,
                    )
                elif arch in ("m", "M"):
                    mixer = SSM(
                        self.d_model,
                        d_state=self.ssm_cfg.get("d_state", 128),
                        d_conv=self.ssm_cfg.get("d_conv", 4),
                        expand=self.ssm_cfg.get("expand", 2),
                        dtype=dtype,
                        layer_idx=layer_idx,
                        rngs=rngs,
                    )
                else:
                    raise NotImplementedError

                if arch in ("T", "M"):
                    mlp = SwiGLU(
                        self.d_model,
                        d_intermediate=config.d_intermediate[self.stage_idx],
                        dtype=dtype,
                        rngs=rngs,
                    )
                else:
                    mlp = None
                layers.append(Block(self.d_model, mixer, mlp, rngs=rngs))
                layer_idx += 1

        self.layers = layers
        self.rmsnorm = RMSNorm(self.d_model, eps=1e-5, dtype=dtype, rngs=rngs)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        return InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)

    def __call__(
        self,
        hidden_states: jnp.ndarray,
        cu_seqlens: jnp.ndarray | None = None,
        max_seqlen: int | None = None,
        mask: jnp.ndarray | None = None,
        inference_params: InferenceParams | None = None,
        **mixer_kwargs,
    ) -> jnp.ndarray:
        packed = cu_seqlens is not None and max_seqlen is not None and mask is None
        if packed:
            assert cu_seqlens is not None and max_seqlen is not None
            T, D = hidden_states.shape
            B = len(cu_seqlens) - 1
            Lmax = int(max_seqlen)
            x = jnp.zeros((B, Lmax, D), dtype=hidden_states.dtype)
            mask_local = jnp.zeros((B, Lmax), dtype=jnp.bool_)
            for b in range(B):
                s, e = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
                x = x.at[b, : e - s].set(hidden_states[s:e])
                mask_local = mask_local.at[b, : e - s].set(True)
        else:
            x = hidden_states
            mask_local = mask

        residual = None
        for layer in self.layers:
            if isinstance(layer.mixer, CausalMHA):
                mix_kwargs = {"attn_mask": mask_local}
            else:
                mix_kwargs = {}
            x, residual = layer(
                x,
                residual=residual,
                inference_params=inference_params,
                mixer_kwargs=mix_kwargs,
            )

        x = self.rmsnorm(x, residual=residual, prenorm=False, residual_in_fp32=True)

        if packed:
            mask_arr = cast(jnp.ndarray, mask_local)
            outs = []
            for b in range(B):
                Lb = int(jnp.sum(mask_arr[b]))
                outs.append(x[b, :Lb])
            x = jnp.concatenate(outs, axis=0)

        if inference_params is not None:
            assert mask is not None
            assert mask.shape[0] == 1
            assert x.ndim == 3
            inference_params.seqlen_offset += x.shape[1]

        return x

    def step(self, hidden_states: jnp.ndarray, inference_params: InferenceParams):
        residual = None
        x = hidden_states
        for layer in self.layers:
            x, residual = layer.step(x, inference_params, residual=residual)
        x = self.rmsnorm(x, residual=residual, prenorm=False, residual_in_fp32=True)
        inference_params.seqlen_offset += 1
        return x
