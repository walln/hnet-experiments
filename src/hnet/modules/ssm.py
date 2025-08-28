"""Lightweight SSM-like mixer (JAX stand-in for Mamba2)."""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from flax.nnx import Param
from jax import lax

from hnet.modules.norms import RMSNorm


class PyTorchSSM(nnx.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        dtype=jnp.float32,
        layer_idx: int | None = None,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.headdim = headdim
        self.nheads = self.d_inner // self.headdim
        self.layer_idx = layer_idx

        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.nheads
        self.in_proj = nnx.Linear(
            d_model, d_in_proj, use_bias=False, dtype=dtype, rngs=rngs
        )

        conv_dim = self.d_inner + 2 * self.d_state
        self.conv1d_weight = Param(jnp.zeros((d_conv, conv_dim), dtype=dtype))
        self.conv1d_bias = Param(jnp.zeros((conv_dim,), dtype=dtype))

        self.norm = RMSNorm(self.d_inner, dtype=dtype, rngs=rngs)
        self.out_proj = nnx.Linear(
            self.d_inner, d_model, use_bias=False, dtype=dtype, rngs=rngs
        )

        self.dt_bias = Param(jnp.zeros((self.nheads,), dtype=dtype))
        self.A_log = Param(jnp.zeros((self.nheads,), dtype=dtype))
        self.D = Param(jnp.ones((self.nheads,), dtype=dtype))

    def _depthwise_conv1d(self, x: jnp.ndarray) -> jnp.ndarray:
        B, L, C = x.shape
        K = self.d_conv
        pad = jnp.zeros((B, K - 1, C), dtype=x.dtype)
        xp = jnp.concatenate([pad, x], axis=1)
        xc = jnp.swapaxes(xp, 1, 2)[..., None]
        xc = jnp.swapaxes(xc, 0, 1)
        k = self.conv1d_weight.value[::-1, :]  # (K, C)
        wc = jnp.swapaxes(k[:, :, None], 0, 1)  # (C, K, 1)
        wc = wc[:, :, None]  # (C, K, 1, 1)

        def conv_c(xc_i, wc_i):
            wc_i = wc_i.astype(xc_i.dtype)
            return lax.conv_general_dilated(
                xc_i,
                wc_i,
                window_strides=(1,),
                padding="VALID",
                dimension_numbers=("NLC", "LIO", "NLC"),
            )

        y_c = jax.vmap(conv_c, in_axes=(0, 0), out_axes=0)(xc, wc)
        y = jnp.transpose(y_c[..., 0], (1, 2, 0))
        if self.conv1d_bias is not None:
            y = y + self.conv1d_bias.value.astype(y.dtype)
        return y

    def __call__(
        self,
        x: jnp.ndarray,
        seq_idx: jnp.ndarray | None = None,
        inference_params=None,
    ) -> jnp.ndarray:
        B, L, D = x.shape
        x_dtype = x.dtype
        w_dtype = self.in_proj.kernel.value.dtype
        if x_dtype != w_dtype:
            x = x.astype(w_dtype)

        zxbcdt = self.in_proj(x)
        z, xBC, dt = jnp.split(
            zxbcdt,
            [self.d_inner, 2 * self.d_inner + 2 * self.d_state],
            axis=-1,
        )

        xBC = jax.nn.silu(self._depthwise_conv1d(xBC))
        if xBC.shape[1] != L:
            xBC = xBC[:, :L, :]

        x_part, B_part, C_part = jnp.split(
            xBC, [self.d_inner, self.d_inner + self.d_state], axis=-1
        )

        x_heads = x_part.reshape(B, L, self.nheads, self.headdim)
        B_heads = B_part[:, :, None, :].repeat(self.nheads, axis=2)
        C_heads = C_part[:, :, None, :].repeat(self.nheads, axis=2)
        dt = jax.nn.softplus(dt + self.dt_bias.value)
        A = -jnp.exp(self.A_log.value)

        state = jnp.zeros((B, self.nheads, self.headdim, self.d_state), dtype=w_dtype)

        def scan_fn(state, inputs):
            dt_t, x_t, B_t, C_t = inputs
            decay = jnp.exp(A[None, :] * dt_t)
            state = state * decay[:, :, None, None]
            state = (
                state
                + (x_t[:, :, :, None] * B_t[:, :, None, :]) * dt_t[:, :, None, None]
            )
            y_t = jnp.sum(state * C_t[:, :, None, :], axis=-1)
            return state, y_t

        inputs = (
            dt.transpose(1, 0, 2),
            x_heads.transpose(1, 0, 2, 3),
            B_heads.transpose(1, 0, 2, 3),
            C_heads.transpose(1, 0, 2, 3),
        )
        _, y_acc = lax.scan(scan_fn, state, inputs)
        y_acc = y_acc.transpose(1, 0, 2, 3)

        y = y_acc.reshape(B, L, self.d_inner)
        D_full = (
            self.D.value[:, None]
            .repeat(self.headdim, axis=1)
            .reshape(-1)[None, None, :]
        )
        y = y + x_part * D_full
        y = self.norm(y * jax.nn.silu(z))
        out = self.out_proj(y)
        return out.astype(x_dtype)

    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None):
        return None

    def step(self, x: jnp.ndarray, inference_params=None) -> jnp.ndarray:
        assert x.shape[1] == 1
        B = x.shape[0]
        x_dtype = x.dtype
        w_dtype = self.in_proj.kernel.value.dtype
        if x_dtype != w_dtype:
            x = x.astype(w_dtype)

        zxbcdt = self.in_proj(x)
        z, xBC, dt = jnp.split(
            zxbcdt,
            [self.d_inner, 2 * self.d_inner + 2 * self.d_state],
            axis=-1,
        )

        layer_idx = getattr(self, "layer_idx", None)
        assert layer_idx is not None and inference_params is not None
        cache = inference_params.key_value_memory_dict
        conv_dim = self.d_inner + 2 * self.d_state

        if layer_idx not in cache:
            conv_state = jnp.zeros((B, conv_dim, self.d_conv), dtype=w_dtype)
            ssm_state = jnp.zeros(
                (B, self.nheads, self.headdim, self.d_state), dtype=w_dtype
            )
        else:
            conv_state, ssm_state = cache[layer_idx]
            if conv_state.shape[0] != B:
                conv_state = conv_state[:B]
                ssm_state = ssm_state[:B]

        xBC_t = xBC[:, 0]
        conv_state = jnp.concatenate([conv_state[:, :, 1:], xBC_t[:, :, None]], axis=2)
        weight = self.conv1d_weight.value.astype(conv_state.dtype)
        conv_out = jnp.einsum("bck,kc->bc", conv_state, weight)
        conv_out = conv_out + self.conv1d_bias.value.astype(conv_state.dtype)
        conv_out = jax.nn.silu(conv_out)

        x_part, B_part, C_part = jnp.split(
            conv_out, [self.d_inner, self.d_inner + self.d_state], axis=-1
        )
        x_heads = x_part.reshape(B, self.nheads, self.headdim)
        B_t = B_part[:, None, :].repeat(self.nheads, axis=1)
        C_t = C_part[:, None, :].repeat(self.nheads, axis=1)
        dt_t = jax.nn.softplus(dt[:, 0] + self.dt_bias.value)
        A = -jnp.exp(self.A_log.value)

        decay = jnp.exp(A[None, :] * dt_t)
        ssm_state = ssm_state * decay[:, :, None, None]
        ssm_state = (
            ssm_state
            + (x_heads[:, :, :, None] * B_t[:, :, None, :]) * dt_t[:, :, None, None]
        )
        y_t = jnp.sum(ssm_state * C_t[:, :, None, :], axis=-1)

        y = y_t.reshape(B, 1, self.d_inner)
        D_full = (
            self.D.value[:, None]
            .repeat(self.headdim, axis=1)
            .reshape(-1)[None, None, :]
        )
        y = y + x_part[:, None, :] * D_full
        y = self.norm(y * jax.nn.silu(z))
        out = self.out_proj(y)

        cache[layer_idx] = [conv_state, ssm_state]
        return out.astype(x_dtype)
