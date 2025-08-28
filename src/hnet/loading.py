"""Config loading and checkpoint conversion utilities."""

from __future__ import annotations

import json

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from .config import AttnConfig, HNetConfig, SSMConfig
from .lm import HNetForCausalLM


def load_config_from_json(json_path: str) -> HNetConfig:
    with open(json_path) as f:
        cfg = json.load(f)
    attn_cfg = AttnConfig(**cfg.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**cfg.pop("ssm_cfg"))
    return HNetConfig(**cfg, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)


def load_model(
    model_path: str | None,
    config_path: str,
    dtype: str = "bfloat16",
    strict: bool = True,
) -> HNetForCausalLM:
    torch_dtype = jnp.bfloat16 if dtype == "bfloat16" else jnp.float32
    cfg = load_config_from_json(config_path)
    rngs = nnx.Rngs(0)
    model = HNetForCausalLM(cfg, dtype=torch_dtype, rngs=rngs)

    if model_path:
        import torch

        state = torch.load(model_path, map_location="cpu")
        try:
            jax_state = convert_pytorch_to_jax(state, model)
            update_model_parameters(model, jax_state)
        except Exception as e:
            ckpt_keys = sorted(list(state.keys()))
            model_keys = sorted([str(k) for k in nnx.state(model)])

            def head_tail(arr):
                return arr[:20] + (["..."] if len(arr) > 40 else []) + arr[-20:]

            raise RuntimeError(
                "Error loading state_dict strictly.\n"
                f"Exception: {e}\n"
                f"Checkpoint keys sample: {head_tail(ckpt_keys)}\n"
                f"Model keys sample: {head_tail(model_keys)}\n"
                "Tip: ensure architecture and parameter names match the reference implementation."
            ) from e
    return model


def convert_pytorch_to_jax(pytorch_state: dict, jax_model) -> dict:
    """Convert PyTorch state dict to JAX parameters."""

    def get_nested_params(module, prefix=""):
        params = {}
        state = nnx.state(module)
        for key, value in state.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            if hasattr(value, "value"):
                params[full_key] = value
            elif hasattr(value, "__dict__"):
                try:
                    nested = get_nested_params(value, full_key)
                    params.update(nested)
                except Exception:
                    params[full_key] = value
            else:
                params[full_key] = value
        return params

    jax_params = get_nested_params(jax_model)
    jax_state = {}

    for jax_key, jax_param in jax_params.items():
        jax_key_str = str(jax_key)
        found = False

        pytorch_key_candidates = [
            jax_key_str,
            jax_key_str.replace(".kernel", ".weight"),
            jax_key_str.replace(".embedding", ".weight"),
            jax_key_str.replace("embeddings.embedding", "embeddings.weight"),
            jax_key_str.replace("lm_head.kernel", "lm_head.weight"),
            jax_key_str.replace("conv1d_weight", "conv1d.weight"),
            jax_key_str.replace("conv1d_bias", "conv1d.bias"),
        ]

        for pytorch_key in pytorch_key_candidates:
            if pytorch_key in pytorch_state:
                tensor = pytorch_state[pytorch_key]
                if hasattr(tensor, "detach"):
                    tensor = tensor.detach().cpu().numpy()
                elif hasattr(tensor, "numpy"):
                    tensor = tensor.numpy()
                else:
                    tensor = np.array(tensor)

                if ".kernel" in jax_key_str and tensor.ndim == 2:
                    tensor = tensor.T
                elif "conv1d_weight" in jax_key_str and tensor.ndim == 3:
                    tensor = tensor[:, 0, :].T

                jax_state[jax_key] = jnp.array(tensor)
                found = True
                break

        if not found:
            if hasattr(jax_param, "value"):
                jax_state[jax_key] = jax_param.value
            else:
                jax_state[jax_key] = jax_param

    loaded_keys = []
    for jax_key in jax_state:
        pytorch_key_candidates = [
            str(jax_key),
            str(jax_key).replace(".kernel", ".weight"),
            str(jax_key).replace(".embedding", ".weight"),
            str(jax_key).replace("conv1d_weight", "conv1d.weight"),
            str(jax_key).replace("conv1d_bias", "conv1d.bias"),
        ]
        if any(pk in pytorch_state for pk in pytorch_key_candidates):
            loaded_keys.append(jax_key)

    print(
        f"Successfully loaded {len(loaded_keys)}/{len(jax_params)} parameters from checkpoint"
    )
    return jax_state


def update_model_parameters(model, jax_state):
    """Update model parameters using the nested parameter paths."""

    for param_path, param_value in jax_state.items():
        try:
            path_parts = str(param_path).split(".")
            current = model
            for part in path_parts[:-1]:
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = getattr(current, part)

            final_param = path_parts[-1]
            if final_param.isdigit():
                current[int(final_param)] = param_value
            elif hasattr(current, final_param):
                param_obj = getattr(current, final_param)
                if hasattr(param_obj, "value"):
                    param_obj.value = param_value
                else:
                    setattr(current, final_param, param_value)
            else:
                print(f"Warning: Could not find parameter {param_path}")
        except Exception as e:
            print(f"Error updating parameter {param_path}: {e}")
