"""H-Net JAX modular package.

This package splits the original single-file implementation into composable modules
so you can experiment with the architecture while preserving identical behavior.
"""

from .config import AttnConfig, HNetConfig, SSMConfig
from .generation import generate_tokens
from .lm import HNetForCausalLM
from .loading import (
    convert_pytorch_to_jax,
    load_config_from_json,
    load_model,
    update_model_parameters,
)
from .model import HNet
from .tokenizer import ByteTokenizer

__all__ = [
    "AttnConfig",
    "ByteTokenizer",
    "HNet",
    "HNetConfig",
    "HNetForCausalLM",
    "SSMConfig",
    "convert_pytorch_to_jax",
    "generate_tokens",
    "load_config_from_json",
    "load_model",
    "update_model_parameters",
]
