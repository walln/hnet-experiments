import flax.nnx as nnx
import jax
import jax.numpy as jnp

from hnet.modules.swiglu import SwiGLU, swiglu


def test_swiglu_shape():
    """Test that swiglu preserves shape."""
    batch_size, seq_len, dim = 2, 10, 64
    gate = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, dim))
    x = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, dim))

    output = swiglu(gate, x)
    assert output.shape == x.shape


def test_swiglu_values():
    """Test swiglu computation is correct."""
    # Test with known values
    gate = jnp.array([0.0, 1.0, -1.0])
    x = jnp.array([1.0, 1.0, 1.0])

    expected = jax.nn.silu(gate) * x
    output = swiglu(gate, x)

    assert jnp.allclose(output, expected)


def test_swiglu_gradient():
    """Test that swiglu is differentiable."""

    def loss_fn(gate, x):
        return jnp.sum(swiglu(gate, x))

    gate = jax.random.normal(jax.random.PRNGKey(0), (10,))
    x = jax.random.normal(jax.random.PRNGKey(1), (10,))

    # Should not raise any errors
    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    gate_grad, x_grad = grad_fn(gate, x)

    assert gate_grad.shape == gate.shape
    assert x_grad.shape == x.shape


def test_swiglu_module_init():
    """Test module initialization."""
    rngs = nnx.Rngs(0)
    d_model = 128

    module = SwiGLU(d_model, rngs=rngs)

    # Check that layers are initialized
    assert hasattr(module, "fc1")
    assert hasattr(module, "fc2")

    # Check dimensions
    assert module.fc1.in_features == d_model
    # Default d_intermediate should be rounded to multiple of 128
    expected_d_intermediate = ((8 * d_model // 3) + 127) // 128 * 128
    assert module.fc1.out_features == 2 * expected_d_intermediate
    assert module.fc2.in_features == expected_d_intermediate
    assert module.fc2.out_features == d_model


def test_swiglu_module_custom_intermediate():
    """Test module with custom intermediate dimension."""
    rngs = nnx.Rngs(0)
    d_model = 128
    d_intermediate = 256

    module = SwiGLU(d_model, d_intermediate=d_intermediate, rngs=rngs)

    assert module.fc1.out_features == 2 * d_intermediate
    assert module.fc2.in_features == d_intermediate


def test_swiglu_module_multiple_of():
    """Test that intermediate dimension is rounded correctly."""
    rngs = nnx.Rngs(0)
    d_model = 100
    multiple_of = 64

    module = SwiGLU(d_model, multiple_of=multiple_of, rngs=rngs)

    # Calculate expected dimension
    d_intermediate = int(8 * d_model / 3)  # 266
    expected = ((d_intermediate + multiple_of - 1) // multiple_of) * multiple_of  # 320

    assert module.fc1.out_features == 2 * expected
    assert module.fc2.in_features == expected


def test_swiglu_module_forward():
    """Test forward pass."""
    rngs = nnx.Rngs(0)
    d_model = 128
    batch_size, seq_len = 2, 10

    module = SwiGLU(d_model, rngs=rngs)
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))

    output = module(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, d_model)

    # Check that output is not NaN
    assert not jnp.any(jnp.isnan(output))


def test_swiglu_module_bias():
    """Test module with bias."""
    rngs = nnx.Rngs(0)
    d_model = 64

    # Without bias
    module_no_bias = SwiGLU(d_model, bias=False, rngs=rngs)
    assert not module_no_bias.fc1.use_bias
    assert not module_no_bias.fc2.use_bias

    # With bias
    module_with_bias = SwiGLU(d_model, bias=True, rngs=rngs)
    assert module_with_bias.fc1.use_bias
    assert module_with_bias.fc2.use_bias


def test_swiglu_module_dtype():
    """Test module with different dtypes."""
    rngs = nnx.Rngs(0)
    d_model = 64

    # Float32 (default param_dtype)
    module_f32 = SwiGLU(d_model, dtype=jnp.float32, rngs=rngs)
    assert module_f32.fc1.kernel.value.dtype == jnp.float32

    # Float16 computation dtype - params still stored as float32 by default
    module_f16 = SwiGLU(d_model, dtype=jnp.float16, rngs=rngs)
    assert module_f16.fc1.kernel.value.dtype == jnp.float32  # params use param_dtype

    # To actually store params as float16, we need to set param_dtype
    # Note: Flax NNX Linear doesn't have param_dtype parameter yet,
    # so we'll just test that computation dtype is set correctly
    x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, d_model))
    y_f32 = module_f32(x.astype(jnp.float32))
    y_f16 = module_f16(x.astype(jnp.float16))
    assert y_f32.dtype == jnp.float32
    assert y_f16.dtype == jnp.float16


def test_swiglu_gradient_flow():
    """Test gradient flow through the module."""
    rngs = nnx.Rngs(0)
    d_model = 64

    module = SwiGLU(d_model, rngs=rngs)

    def loss_fn(module, x):
        return jnp.mean(module(x) ** 2)

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, d_model))

    # Compute gradients
    grads = nnx.grad(loss_fn)(module, x)

    # Check that gradients exist and are not NaN
    assert grads.fc1.kernel.value is not None
    assert grads.fc2.kernel.value is not None
    assert not jnp.any(jnp.isnan(grads.fc1.kernel.value))
    assert not jnp.any(jnp.isnan(grads.fc2.kernel.value))
