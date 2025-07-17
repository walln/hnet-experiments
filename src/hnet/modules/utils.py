from dataclasses import asdict

import jax.numpy as jnp


def get_seq_idx(cu_seqlens, device=None):
    """
    Generate sequence indices from cumulative sequence lengths.

    Args:
        cu_seqlens: Array of cumulative sequence lengths [batch_size + 1]
        device: Unused, kept for API compatibility

    Returns:
        Array of sequence indices [1, total_seq_len]
    """
    seq_idx = jnp.zeros(cu_seqlens[-1], dtype=jnp.int32)
    seq_idx = seq_idx.at[cu_seqlens[:-1]].set(1)
    seq_idx = (jnp.cumsum(seq_idx, axis=0) - 1)[None, :].astype(jnp.int32)

    return seq_idx


def get_stage_cfg(cfg, stage_idx):
    """
    Extract stage-specific configuration from a multi-stage config.

    Args:
        cfg: Configuration dataclass
        stage_idx: Index of the stage

    Returns:
        Dictionary with stage-specific values
    """
    return {
        k: v[stage_idx] if isinstance(v, list) else v for k, v in asdict(cfg).items()
    }
