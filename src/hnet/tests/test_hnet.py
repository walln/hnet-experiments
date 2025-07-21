"""Test the HNet JAX model."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from hnet.models.config_hnet import AttnConfig, SSMConfig
from hnet.models.hnet import HNet, HNetConfig


@pytest.fixture
def simple_config():
    """Simple HNet configuration for testing."""
    return HNetConfig(
        arch_layout=["m2t2"],  # 2 Mamba blocks + 2 Transformer blocks
        d_model=[128],
        d_intermediate=[512],
        vocab_size=256,
        ssm_cfg=SSMConfig(
            d_conv=4,
            expand=2,
            d_state=16,
            chunk_size=64,  # Use default chunk size
        ),
        attn_cfg=AttnConfig(
            num_heads=[8],
            rotary_emb_dim=[16],
            window_size=[-1],
        ),
    )


@pytest.fixture
def hierarchical_config():
    """Hierarchical HNet configuration for testing."""
    return HNetConfig(
        arch_layout=["m2", ["m1t1"], "t2"],  # Hierarchical structure with 2 stages
        d_model=[64, 128],
        d_intermediate=[256, 512],
        vocab_size=256,
        ssm_cfg=SSMConfig(
            d_conv=4,
            expand=2,
            d_state=16,
            chunk_size=16,  # Use chunk size that divides test sequence lengths well
        ),
        attn_cfg=AttnConfig(
            num_heads=[4, 8],
            rotary_emb_dim=[16, 16],
            window_size=[-1, -1],
        ),
    )


def test_hnet_initialization(simple_config):
    """Test HNet model initialization."""
    rngs = nnx.Rngs(0)
    model = HNet(simple_config, stage_idx=0, rngs=rngs)

    assert model.stage_idx == 0
    assert model.d_model == 128
    assert model.is_innermost is True  # Single stage model
    assert hasattr(model, "main_network")


def test_hnet_forward_pass(simple_config):
    """Test HNet forward pass."""
    rngs = nnx.Rngs(0)
    model = HNet(simple_config, stage_idx=0, rngs=rngs)

    # Create dummy input with length divisible by chunk_size
    batch_size, seq_len = 2, 128  # Divisible by default chunk_size=64
    hidden_states = jax.random.normal(
        jax.random.key(0), (batch_size, seq_len, simple_config.d_model[0])
    )
    mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)

    # Forward pass
    output, boundary_predictions = model(hidden_states, mask=mask)

    assert output.shape == hidden_states.shape
    assert isinstance(boundary_predictions, list)


def test_hnet_inference_cache(simple_config):
    """Test HNet inference cache allocation."""
    rngs = nnx.Rngs(0)
    model = HNet(simple_config, stage_idx=0, rngs=rngs)

    batch_size, max_seqlen = 2, 64
    cache = model.allocate_inference_cache(batch_size, max_seqlen)

    assert cache is not None
    assert cache.main_network_state is not None


def test_hnet_step_function(simple_config):
    """Test HNet step function for autoregressive generation."""
    rngs = nnx.Rngs(0)
    model = HNet(simple_config, stage_idx=0, rngs=rngs)

    batch_size = 2
    cache = model.allocate_inference_cache(batch_size, 64)

    # Single token input
    single_token = jax.random.normal(
        jax.random.key(1), (batch_size, 1, simple_config.d_model[0])
    )

    # Step
    output, _ = model.step(single_token, cache)

    assert output.shape == single_token.shape


def test_hierarchical_hnet_initialization(hierarchical_config):
    """Test hierarchical HNet initialization."""
    rngs = nnx.Rngs(42)
    model = HNet(hierarchical_config, stage_idx=0, rngs=rngs)

    assert model.stage_idx == 0
    assert model.d_model == 64
    assert model.is_innermost is False
    assert hasattr(model, "encoder")
    assert hasattr(model, "main_network")
    assert hasattr(model, "decoder")
    assert hasattr(model, "routing_module")
    assert hasattr(model, "chunk_layer")
    assert hasattr(model, "dechunk_layer")
    assert hasattr(model, "residual_proj")


def test_hierarchical_hnet_forward(hierarchical_config):
    """Test hierarchical HNet forward pass."""
    rngs = nnx.Rngs(42)
    model = HNet(hierarchical_config, stage_idx=0, rngs=rngs)

    # Create dummy input with length divisible by chunk_size
    batch_size, seq_len = 1, 48  # Divisible by chunk_size=16
    hidden_states = jax.random.normal(
        jax.random.key(2), (batch_size, seq_len, hierarchical_config.d_model[0])
    )
    mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)

    # Forward pass
    output, boundary_predictions = model(hidden_states, mask=mask)

    assert output.shape == hidden_states.shape
    assert len(boundary_predictions) > 0

    # Check that we have routing decisions
    for pred in boundary_predictions:
        assert hasattr(pred, "boundary_mask")
        assert hasattr(pred, "boundary_prob")
        assert hasattr(pred, "selected_probs")


def test_hnet_with_padding(hierarchical_config):
    """Test HNet with dimension padding."""
    rngs = nnx.Rngs(0)

    # Create inner stage model (stage 1) which should have padding
    model = HNet(hierarchical_config, stage_idx=1, rngs=rngs)

    assert model.d_model == 128  # d_model[1]
    assert model.pad_dimension is not None
    assert model.pad_dimension.value.shape == (128 - 64,)  # Padding from 64 to 128
