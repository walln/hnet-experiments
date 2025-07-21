import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from hnet.modules.block import HybridBlock, Mamba2Wrapper
from hnet.modules.cache import CacheState, create_mamba2_cache
from hnet.modules.config import HybridConfig, Mamba2Config
from hnet.modules.mamba2 import (
    Mamba2Block,
    Mamba2Layer,
    segsum,
    silu,
    softplus,
    ssd,
)


def test_activations():
    """Test activation functions."""
    x = jax.random.normal(jax.random.PRNGKey(0), (10,))

    # Test softplus
    sp = softplus(x)
    assert sp.shape == x.shape
    assert jnp.all(sp >= 0)

    # Test silu
    s = silu(x)
    assert s.shape == x.shape


def test_segsum():
    """Test segsum function."""
    # Test 1D case
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = segsum(x)
    assert result.shape == (4, 4)

    # Test that exp(segsum) creates proper lower triangular structure
    L = jnp.exp(result)
    # Check lower triangular
    assert jnp.allclose(L, jnp.tril(L))

    # Test batched case
    x_batch = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 8))
    result_batch = segsum(x_batch)
    assert result_batch.shape == (2, 3, 8, 8)


def test_ssd():
    """Test SSD (Structured State Space Duality) algorithm."""
    batch = 2
    seq_len = 64  # Must be divisible by chunk_size
    nheads = 4
    headdim = 16
    d_state = 8
    chunk_size = 16

    # Create inputs
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 4)

    x = jax.random.normal(keys[0], (batch, seq_len, nheads, headdim))
    A = jax.random.normal(keys[1], (batch, seq_len, nheads)) * 0.1
    B = jax.random.normal(keys[2], (batch, seq_len, nheads, d_state))
    C = jax.random.normal(keys[3], (batch, seq_len, nheads, d_state))

    # Test without initial states
    y, final_state = ssd(x, A, B, C, chunk_size)

    assert y.shape == (batch, seq_len, nheads, headdim)
    assert final_state.shape == (batch, nheads, headdim, d_state)

    # Test with initial states
    # Use non-zero initial states to ensure they affect the output
    initial_states = (
        jax.random.normal(keys[0], (batch, 1, nheads, headdim, d_state)) * 0.1
    )
    y2, final_state2 = ssd(x, A, B, C, chunk_size, initial_states)

    assert y2.shape == (batch, seq_len, nheads, headdim)
    assert final_state2.shape == (batch, nheads, headdim, d_state)

    # Results should be different with non-zero initial states
    assert not jnp.allclose(y, y2)


def test_inference_cache():
    """Test InferenceCache dataclass."""
    batch_size = 2
    d_inner = 128
    d_state = 16
    d_conv = 4
    nheads = 8
    headdim = 16

    cache = create_mamba2_cache(batch_size, d_inner, d_state, d_conv, nheads, headdim)

    assert cache.conv_state.shape == (batch_size, d_inner + 2 * d_state, d_conv)
    assert cache.ssm_state.shape == (batch_size, nheads, headdim, d_state)

    # Check initialized to zeros
    assert jnp.allclose(cache.conv_state, 0)
    assert jnp.allclose(cache.ssm_state, 0)


def test_mamba2_layer():
    """Test Mamba2Layer implementation."""
    rngs = nnx.Rngs(0)
    d_model = 128
    batch_size = 2
    seq_len = 64  # Must be divisible by chunk_size (default 64)
    headdim = 32
    chunk_size = 64  # Use default chunk size

    # Create module
    mamba = Mamba2Layer(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=headdim,
        ngroups=1,
        chunk_size=chunk_size,
        rngs=rngs,
    )

    # Create input
    x = jax.random.normal(rngs(), (batch_size, seq_len, d_model))

    # Forward pass without step mode
    output, cache = mamba(x, step_mode=False)

    # Check output shape
    assert output.shape == x.shape
    # Cache should be None when not in step_mode
    assert cache is None


def test_mamba2_inference_step():
    """Test Mamba2 single-step inference."""
    rngs = nnx.Rngs(0)
    d_model = 128
    batch_size = 2
    headdim = 32
    chunk_size = 64  # Use default chunk size

    # Create module
    mamba = Mamba2Layer(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=headdim,
        ngroups=1,
        chunk_size=chunk_size,
        rngs=rngs,
    )

    # Initialize cache
    cache = create_mamba2_cache(
        batch_size,
        mamba.d_inner,
        mamba.d_state,
        mamba.d_conv,
        mamba.nheads,
        mamba.headdim,
    )

    # Single token input
    x_single = jax.random.normal(rngs(), (batch_size, 1, d_model))

    # Step through multiple tokens using step method
    outputs = []
    for _ in range(5):
        output, cache = mamba.step(x_single, cache)
        outputs.append(output)
        assert output.shape == (batch_size, 1, d_model)
        assert cache is not None

    # Check that outputs are different (state is being updated)
    for i in range(1, len(outputs)):
        assert not jnp.allclose(outputs[i], outputs[0])


def test_mamba2_block():
    """Test Mamba2Block with norm and residual."""
    rngs = nnx.Rngs(0)
    d_model = 128
    batch_size = 2
    seq_len = 64
    chunk_size = 64

    # Create config
    config = Mamba2Config(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=32,
        ngroups=1,
        chunk_size=chunk_size,
        layer_idx=0,
    )

    # Create module
    block = Mamba2Block(
        config=config,
        rngs=rngs,
    )

    # Create input
    x = jax.random.normal(rngs(), (batch_size, seq_len, d_model))

    # Forward pass
    output, cache = block(x)

    # Check output shape
    assert output.shape == x.shape
    assert cache is None

    # Check that residual connection is applied (output should be different from normalized input)
    normed_input = block.norm(x)
    assert not jnp.allclose(output, normed_input)


def test_hybrid_block_mamba():
    """Test HybridBlock with Mamba2."""
    rngs = nnx.Rngs(0)
    d_model = 128
    batch_size = 2
    seq_len = 64  # Use seq_len divisible by default chunk_size=64

    # Create config with Mamba2
    mamba_config = Mamba2Config(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        chunk_size=64,  # Use default chunk size
        layer_idx=0,
    )

    config = HybridConfig(
        d_model=d_model,
        n_heads=8,  # Not used when use_mamba=True
        use_mamba=True,
        mlp_expand=4,
        layer_idx=0,
        mamba_config=mamba_config,
    )

    # Create module with Mamba2
    block = HybridBlock(
        config=config,
        rngs=rngs,
    )

    # Create input
    x = jax.random.normal(rngs(), (batch_size, seq_len, d_model))

    # Forward pass
    output, cache = block(x)

    # Check output shape
    assert output.shape == x.shape
    assert cache is None


def test_hybrid_block_attention():
    """Test HybridBlock with Attention."""
    rngs = nnx.Rngs(0)
    d_model = 128
    batch_size = 2
    seq_len = 16

    # Create config with Attention
    from hnet.modules.config import AttentionConfig

    attention_config = AttentionConfig(
        d_model=d_model,
        num_heads=8,
        layer_idx=0,
    )

    config = HybridConfig(
        d_model=d_model,
        n_heads=8,
        use_mamba=False,
        layer_idx=0,
        attention_config=attention_config,
    )

    # Create module with Attention
    block = HybridBlock(
        config=config,
        rngs=rngs,
    )

    # Create input
    x = jax.random.normal(rngs(), (batch_size, seq_len, d_model))

    # Forward pass
    output, cache = block(x)

    # Check output shape
    assert output.shape == x.shape
    assert cache is None


def test_mamba2_wrapper():
    """Test Mamba2Wrapper for compatibility."""
    rngs = nnx.Rngs(0)
    d_model = 128
    batch_size = 2
    seq_len = 64

    # Create config
    config = Mamba2Config(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        layer_idx=0,
        chunk_size=64,  # Use default chunk size
    )

    # Create module
    wrapper = Mamba2Wrapper(
        config=config,
        rngs=rngs,
    )

    # Create input
    x = jax.random.normal(rngs(), (batch_size, seq_len, d_model))

    # Forward pass
    output, cache = wrapper(x)

    # Check output shape
    assert output.shape == x.shape
    assert cache is None

    # Test step method for generation
    single_token = jax.random.normal(rngs(), (batch_size, 1, d_model))
    cache_state = CacheState.empty()
    step_output, updated_cache = wrapper.step(single_token, cache_state)
    assert step_output.shape == single_token.shape
    assert updated_cache is not None


@pytest.mark.parametrize(
    "d_model,d_state,headdim", [(64, 8, 16), (128, 16, 32), (256, 32, 64)]
)
def test_different_sizes(d_model, d_state, headdim):
    """Test with different model sizes."""
    rngs = nnx.Rngs(0)
    batch_size = 1
    seq_len = 64  # Must be divisible by chunk_size (default 64)
    chunk_size = 64

    # Create module
    mamba = Mamba2Layer(
        d_model=d_model,
        d_state=d_state,
        headdim=headdim,
        chunk_size=chunk_size,
        rngs=rngs,
    )

    # Create input
    x = jax.random.normal(rngs(), (batch_size, seq_len, d_model))

    # Forward pass
    output, _ = mamba(x, step_mode=False)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model)


def test_mamba2_convolution():
    """Test Mamba2 convolution implementation through forward pass."""
    rngs = nnx.Rngs(0)
    d_model = 64
    batch_size = 2
    seq_len = 64  # Use compatible seq_len

    mamba = Mamba2Layer(
        d_model=d_model,
        d_state=8,
        d_conv=4,
        expand=2,
        headdim=32,
        chunk_size=64,
        rngs=rngs,
    )

    # Test training mode - full forward pass
    x = jax.random.normal(rngs(), (batch_size, seq_len, d_model))
    output, _ = mamba(x, step_mode=False)
    assert output.shape == x.shape

    # Test inference mode - step by step
    cache = create_mamba2_cache(
        batch_size,
        mamba.d_inner,
        mamba.d_state,
        mamba.d_conv,
        mamba.nheads,
        mamba.headdim,
    )
    x_single = jax.random.normal(rngs(), (batch_size, 1, d_model))

    # Step through a few tokens to test convolution state updates
    for _ in range(3):
        step_output, cache = mamba.step(x_single, cache)
        assert step_output.shape == x_single.shape
        assert cache is not None


def test_mamba2_groups():
    """Test Mamba2 with multiple groups."""
    rngs = nnx.Rngs(0)
    d_model = 128
    batch_size = 2
    seq_len = 64
    chunk_size = 64
    ngroups = 2  # Test with multiple groups

    # Create module with groups
    mamba = Mamba2Layer(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=32,
        ngroups=ngroups,
        chunk_size=chunk_size,
        rngs=rngs,
    )

    # Create input
    x = jax.random.normal(rngs(), (batch_size, seq_len, d_model))

    # Forward pass
    output, _ = mamba(x, step_mode=False)
    assert output.shape == x.shape


def test_mamba2_d_has_hdim():
    """Test Mamba2 with D parameter having head dimension."""
    rngs = nnx.Rngs(0)
    d_model = 128
    batch_size = 2
    seq_len = 64
    chunk_size = 64

    # Create module with D_has_hdim=True
    mamba = Mamba2Layer(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=32,
        D_has_hdim=True,
        chunk_size=chunk_size,
        rngs=rngs,
    )

    # Check D parameter shape - should be (d_ssm,) when D_has_hdim=True
    expected_d_shape = mamba.d_ssm if mamba.D_has_hdim else mamba.nheads
    assert mamba.D.value.shape == (expected_d_shape,)

    # Create input and test forward pass
    x = jax.random.normal(rngs(), (batch_size, seq_len, d_model))
    output, _ = mamba(x, step_mode=False)
    assert output.shape == x.shape


def test_sequence_length_compatibility():
    """Test that arbitrary sequence lengths are supported through padding."""
    rngs = nnx.Rngs(0)
    d_model = 64
    batch_size = 1
    chunk_size = 64

    mamba = Mamba2Layer(
        d_model=d_model,
        d_state=8,
        d_conv=4,
        headdim=16,
        chunk_size=chunk_size,
        rngs=rngs,
    )

    # Test with sequence length not divisible by chunk_size
    seq_len_irregular = 65  # Not divisible by chunk_size
    x_irregular = jax.random.normal(rngs(), (batch_size, seq_len_irregular, d_model))

    # This should now work due to internal padding
    output_irregular, _ = mamba(x_irregular, step_mode=False)
    assert output_irregular.shape == x_irregular.shape

    # Test with compatible sequence length
    seq_len_good = 128  # Divisible by chunk_size
    x_good = jax.random.normal(rngs(), (batch_size, seq_len_good, d_model))
    output_good, _ = mamba(x_good, step_mode=False)
    assert output_good.shape == x_good.shape

    # Test various sequence lengths to ensure robustness
    for seq_len in [1, 7, 33, 63, 65, 100]:
        x_test = jax.random.normal(rngs(), (batch_size, seq_len, d_model))
        output_test, _ = mamba(x_test, step_mode=False)
        assert output_test.shape == x_test.shape
