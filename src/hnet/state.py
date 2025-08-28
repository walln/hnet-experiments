"""Dataclass containers for inference and routing state."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp


@dataclass
class InferenceParams:
    """Container for inference parameters and caches."""

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict[int, jnp.ndarray] = field(default_factory=dict)
    lengths_per_sample: jnp.ndarray | None = None

    def reset(self, max_seqlen: int, max_batch_size: int):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        self.key_value_memory_dict.clear()


@dataclass
class RoutingModuleOutput:
    boundary_prob: jnp.ndarray
    boundary_mask: jnp.ndarray
    selected_probs: jnp.ndarray


@dataclass
class RoutingModuleState:
    has_seen_tokens: jnp.ndarray  # (B,)
    last_hidden_state: jnp.ndarray  # (B, D)


@dataclass
class DeChunkState:
    last_value: jnp.ndarray  # (B, D)


@dataclass
class HNetState:
    encoder_state: InferenceParams | None = None
    routing_module_state: RoutingModuleState | None = None
    main_network_state: HNetState | InferenceParams | None = None
    dechunk_state: DeChunkState | None = None
    decoder_state: InferenceParams | None = None
