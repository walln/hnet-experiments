from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from hnet.modules.utils import get_seq_idx, get_stage_cfg


def test_basic_get_seq_idx():
    cu_seqlens = jnp.array([0, 3, 7, 10])
    seq_idx = get_seq_idx(cu_seqlens)

    expected = jnp.array([[0, 0, 0, 1, 1, 1, 1, 2, 2, 2]], dtype=jnp.int32)
    assert seq_idx.shape == (1, 10)
    assert jnp.array_equal(seq_idx, expected)


def test_single_sequence_get_seq_idx():
    cu_seqlens = jnp.array([0, 5])
    seq_idx = get_seq_idx(cu_seqlens)

    expected = jnp.array([[0, 0, 0, 0, 0]], dtype=jnp.int32)
    assert jnp.array_equal(seq_idx, expected)


def test_empty_get_seq_idx():
    cu_seqlens = jnp.array([0])
    seq_idx = get_seq_idx(cu_seqlens)

    assert seq_idx.shape == (1, 0)


@dataclass
class Config:
    param1: list[int]
    param2: int
    param3: list[str]


def test_basic_get_stage_cfg():
    cfg = Config(param1=[10, 20, 30], param2=42, param3=["a", "b", "c"])

    # Test stage 0
    stage0_cfg = get_stage_cfg(cfg, 0)
    assert stage0_cfg == {"param1": 10, "param2": 42, "param3": "a"}

    # Test stage 1
    stage1_cfg = get_stage_cfg(cfg, 1)
    assert stage1_cfg == {"param1": 20, "param2": 42, "param3": "b"}

    # Test stage 2
    stage2_cfg = get_stage_cfg(cfg, 2)
    assert stage2_cfg == {"param1": 30, "param2": 42, "param3": "c"}


def test_scalar_params_get_stage_cfg():
    cfg = Config(param1=[100], param2=50, param3=["test"])

    stage_cfg = get_stage_cfg(cfg, 0)
    assert stage_cfg["param2"] == 50  # Scalar preserved
    assert stage_cfg["param1"] == 100  # List indexed


def test_empty_lists_get_stage_cfg():
    cfg = Config(param1=[], param2=10, param3=[])

    # This should raise an IndexError when trying to access empty lists
    with pytest.raises(IndexError):
        get_stage_cfg(cfg, 0)
