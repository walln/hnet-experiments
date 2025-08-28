"""High-level block and isotropic stack primitives."""

from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp

from hnet.modules.norms import RMSNorm


class Block(nnx.Module):
    def __init__(
        self,
        d_model: int,
        mixer: nnx.Module,
        mlp: nnx.Module | None,
        residual_in_fp32: bool = True,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.residual_in_fp32 = residual_in_fp32
        self.norm1 = RMSNorm(d_model, rngs=rngs)
        self.mixer = mixer
        self.mlp = mlp
        if self.mlp is not None:
            self.norm2 = RMSNorm(d_model, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        residual: jnp.ndarray | None = None,
        inference_params=None,
        mixer_kwargs: dict | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        x, residual = self.norm1(
            x, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32
        )
        mixer_kwargs = mixer_kwargs or {}
        x = self.mixer(x, **mixer_kwargs, inference_params=inference_params)
        if self.mlp is not None:
            x, residual = self.norm2(
                x,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            x = self.mlp(x)
        return x, residual

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, dtype=None, **kwargs
    ):
        return None

    def step(
        self, x: jnp.ndarray, inference_params, residual: jnp.ndarray | None = None
    ):
        x, residual = self.norm1(
            x, residual=residual, prenorm=True, residual_in_fp32=self.residual_in_fp32
        )
        x = self.mixer.step(x, inference_params)
        if self.mlp is not None:
            x, residual = self.norm2(
                x,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            x = self.mlp(x)
        return x, residual
