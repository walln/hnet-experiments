import flax.nnx as nnx
import jax
import jax.numpy as jnp

from hnet.modules.rotary import (
    RotaryEmbedding,
    apply_rotary_emb,
    apply_rotary_emb_jax,
    apply_rotary_emb_kv,
    apply_rotary_emb_qkv,
    rotate_half,
)


def test_rotate_half():
    """Test the rotate_half function."""
    # Test non-interleaved
    x = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    result = rotate_half(x, interleaved=False)
    expected = jnp.array([[-3, -4, 1, 2], [-7, -8, 5, 6]])
    assert jnp.allclose(result, expected)

    # Test interleaved
    result_interleaved = rotate_half(x, interleaved=True)
    expected_interleaved = jnp.array([[-2, 1, -4, 3], [-6, 5, -8, 7]])
    assert jnp.allclose(result_interleaved, expected_interleaved)


def test_rotate_half_edge_cases():
    """Test rotate_half with edge cases."""
    # Single element per half
    x = jnp.array([1.0, 2.0])
    result = rotate_half(x)
    expected = jnp.array([-2.0, 1.0])
    assert jnp.allclose(result, expected)

    # Higher dimensional tensors
    x = jax.random.normal(jax.random.PRNGKey(42), (2, 3, 4, 8))
    result = rotate_half(x, interleaved=False)
    assert result.shape == x.shape

    # Verify the rotation property: rotate_half(rotate_half(x)) = -x
    x = jax.random.normal(jax.random.PRNGKey(43), (10, 16))
    double_rotated = rotate_half(rotate_half(x))
    assert jnp.allclose(double_rotated, -x, atol=1e-6)


def test_apply_rotary_emb_jax():
    """Test basic rotary embedding application."""
    batch, seq, heads, dim = 2, 4, 2, 8
    x = jax.random.normal(jax.random.PRNGKey(0), (batch, seq, heads, dim))

    # Create cos and sin
    rotary_dim = dim // 2
    cos = jnp.ones((seq, rotary_dim))
    sin = jnp.zeros((seq, rotary_dim))

    # Apply rotary
    result = apply_rotary_emb_jax(x, cos, sin)

    # With cos=1 and sin=0, result should equal input
    assert jnp.allclose(result, x, atol=1e-6)


def test_apply_rotary_emb_different_dtypes():
    """Test rotary embeddings with different data types."""
    rngs = nnx.Rngs(0)
    dim = 64

    # Test common dtypes used in ML workloads
    dtypes = [jnp.float16, jnp.bfloat16, jnp.float32]

    for dtype in dtypes:
        rotary_emb = RotaryEmbedding(dim=dim, rngs=rngs)

        batch, seq, heads = 1, 16, 4
        qkv = jax.random.normal(
            jax.random.PRNGKey(1), (batch, seq, 3, heads, dim)
        ).astype(dtype)

        qkv_rot = rotary_emb(qkv)
        assert isinstance(qkv_rot, jax.Array)
        assert qkv_rot.dtype == dtype
        assert qkv_rot.shape == qkv.shape


def test_apply_rotary_emb_partial_dim():
    """Test rotary embeddings applied to partial dimensions."""
    batch, seq, heads, dim = 2, 8, 4, 16
    x = jax.random.normal(jax.random.PRNGKey(10), (batch, seq, heads, dim))

    # Apply rotary to only half the dimensions
    rotary_dim = dim // 4  # Only rotate 1/4 of head dim
    cos = jnp.ones((seq, rotary_dim))
    sin = jnp.ones((seq, rotary_dim)) * 0.5

    result = apply_rotary_emb_jax(x, cos, sin)

    # Check that only first part is modified
    rotated_dims = rotary_dim * 2
    assert not jnp.allclose(result[..., :rotated_dims], x[..., :rotated_dims])
    assert jnp.allclose(result[..., rotated_dims:], x[..., rotated_dims:])


def test_apply_rotary_emb_functions():
    """Test the internal apply_rotary_emb functions directly."""
    batch, seq, heads, dim = 2, 16, 8, 64
    rotary_dim = dim // 2

    # Create test data
    x = jax.random.normal(jax.random.PRNGKey(20), (batch, seq, heads, dim))
    qkv = jax.random.normal(jax.random.PRNGKey(21), (batch, seq, 3, heads, dim))
    kv = jax.random.normal(jax.random.PRNGKey(22), (batch, seq, 2, heads, dim))

    # Create cos/sin
    cos = jnp.ones((seq, rotary_dim))
    sin = jnp.zeros((seq, rotary_dim))

    # Test apply_rotary_emb
    result = apply_rotary_emb(x, cos, sin)
    assert result.shape == x.shape
    assert jnp.allclose(result, x, atol=1e-6)  # Since sin=0

    # Test apply_rotary_emb_qkv
    qkv_result = apply_rotary_emb_qkv(qkv, cos, sin, cos, sin)
    assert qkv_result.shape == qkv.shape
    # V should be unchanged
    assert jnp.allclose(qkv_result[:, :, 2], qkv[:, :, 2])

    # Test apply_rotary_emb_kv
    kv_result = apply_rotary_emb_kv(kv, cos, sin)
    assert kv_result.shape == kv.shape
    # V should be unchanged
    assert jnp.allclose(kv_result[:, :, 1], kv[:, :, 1])


def test_rotary_embedding_module():
    """Test the RotaryEmbedding module."""
    rngs = nnx.Rngs(0)
    dim = 64
    rotary_emb = RotaryEmbedding(dim=dim, rngs=rngs)

    # Test with standard QKV
    batch, seq, heads = 2, 32, 8
    qkv = jax.random.normal(jax.random.PRNGKey(1), (batch, seq, 3, heads, dim))

    qkv_rot = rotary_emb(qkv)

    # Assert return type for type checker
    assert isinstance(qkv_rot, jax.Array), "Expected single array return when kv=None"

    # Check shape is preserved
    assert qkv_rot.shape == qkv.shape

    # Check that V (index 2) is unchanged
    assert jnp.allclose(qkv_rot[:, :, 2], qkv[:, :, 2])

    # Check that Q and K are modified
    assert not jnp.allclose(qkv_rot[:, :, 0], qkv[:, :, 0])
    assert not jnp.allclose(qkv_rot[:, :, 1], qkv[:, :, 1])


def test_rotary_with_separate_kv():
    """Test rotary embeddings with separate Q and KV tensors."""
    rngs = nnx.Rngs(0)
    dim = 64
    rotary_emb = RotaryEmbedding(dim=dim, rngs=rngs)

    batch, seq, heads = 2, 16, 8
    q = jax.random.normal(jax.random.PRNGKey(2), (batch, seq, heads, dim))
    kv = jax.random.normal(jax.random.PRNGKey(3), (batch, seq, 2, heads, dim))

    result = rotary_emb(q, kv=kv)

    # Assert return type for type checker
    assert isinstance(result, tuple), "Expected tuple return when kv is provided"
    assert len(result) == 2, "Expected tuple of length 2"

    q_rot, kv_rot = result

    # Check shapes
    assert q_rot.shape == q.shape
    assert kv_rot.shape == kv.shape

    # Check V is unchanged
    assert jnp.allclose(kv_rot[:, :, 1], kv[:, :, 1])

    # Check Q and K are modified
    assert not jnp.allclose(q_rot, q)
    assert not jnp.allclose(kv_rot[:, :, 0], kv[:, :, 0])


def test_rotary_with_offset():
    """Test rotary embeddings with sequence offset (for KV cache)."""
    rngs = nnx.Rngs(0)
    dim = 64
    rotary_emb = RotaryEmbedding(dim=dim, rngs=rngs)

    # First, process initial sequence
    batch, seq1, heads = 1, 10, 8
    qkv1 = jax.random.normal(jax.random.PRNGKey(4), (batch, seq1, 3, heads, dim))
    _qkv1_rot = rotary_emb(qkv1)

    # Now process new token with offset
    seq2 = 1
    q_new = jax.random.normal(jax.random.PRNGKey(5), (batch, seq2, heads, dim))
    kv_new = jax.random.normal(jax.random.PRNGKey(6), (batch, seq2, 2, heads, dim))

    result = rotary_emb(q_new, kv=kv_new, seqlen_offset=seq1)

    # Assert return type for type checker
    assert isinstance(result, tuple), "Expected tuple return when kv is provided"
    assert len(result) == 2, "Expected tuple of length 2"

    q_rot, kv_rot = result

    # The rotary embeddings should be different due to offset
    result_no_offset = rotary_emb(q_new, kv=kv_new, seqlen_offset=0)
    assert isinstance(result_no_offset, tuple), (
        "Expected tuple return when kv is provided"
    )
    q_no_offset, _ = result_no_offset
    assert not jnp.allclose(q_rot, q_no_offset)


def test_xpos():
    """Test XPos (exponential positional scaling)."""
    rngs = nnx.Rngs(0)
    dim = 64

    # Create two models - one with XPos, one without
    rotary_standard = RotaryEmbedding(dim=dim, rngs=rngs)
    rotary_xpos = RotaryEmbedding(dim=dim, scale_base=512.0, rngs=rngs)

    batch, seq, heads = 1, 128, 4
    qkv = jax.random.normal(jax.random.PRNGKey(7), (batch, seq, 3, heads, dim))

    qkv_standard = rotary_standard(qkv)
    qkv_xpos = rotary_xpos(qkv)

    # Assert return types for type checker
    assert isinstance(qkv_standard, jax.Array), "Expected single array return"
    assert isinstance(qkv_xpos, jax.Array), "Expected single array return"

    # Results should be different
    assert not jnp.allclose(qkv_standard, qkv_xpos)

    # Both should preserve shape
    assert qkv_standard.shape == qkv.shape
    assert qkv_xpos.shape == qkv.shape


def test_mqa_gqa():
    """Test Multi-Query Attention (MQA) / Grouped-Query Attention (GQA)."""
    rngs = nnx.Rngs(0)
    dim = 64
    rotary_emb = RotaryEmbedding(dim=dim, rngs=rngs)

    batch, seq = 2, 16
    num_heads_q = 8
    num_heads_k = 2  # Fewer KV heads for GQA

    # Create QKV with different head counts
    qkv = jax.random.normal(
        jax.random.PRNGKey(8), (batch, seq, num_heads_q + 2 * num_heads_k, dim)
    )

    qkv_rot = rotary_emb(qkv, num_heads_q=num_heads_q)

    # Assert return type for type checker
    assert isinstance(qkv_rot, jax.Array), "Expected single array return"

    # Check shape is preserved
    assert qkv_rot.shape == qkv.shape

    # Extract and verify Q, K, V
    q_rot = qkv_rot[:, :, :num_heads_q]
    k_rot = qkv_rot[:, :, num_heads_q : num_heads_q + num_heads_k]
    v_rot = qkv_rot[:, :, num_heads_q + num_heads_k :]

    # Q and K should be modified
    assert not jnp.allclose(q_rot, qkv[:, :, :num_heads_q])
    assert not jnp.allclose(k_rot, qkv[:, :, num_heads_q : num_heads_q + num_heads_k])

    # V should be unchanged
    assert jnp.allclose(v_rot, qkv[:, :, num_heads_q + num_heads_k :])


def test_cache_behavior():
    """Test the caching behavior of cos/sin values."""
    rngs = nnx.Rngs(0)
    dim = 64
    rotary_emb = RotaryEmbedding(dim=dim, rngs=rngs)

    # Initial state - no cache
    assert rotary_emb._seq_len_cached.value == 0
    assert rotary_emb._cos_cached.value.size == 0

    # Process a sequence - should populate cache
    batch, seq1, heads = 1, 32, 4
    qkv1 = jax.random.normal(jax.random.PRNGKey(30), (batch, seq1, 3, heads, dim))
    _ = rotary_emb(qkv1)

    # Check cache is populated
    assert rotary_emb._seq_len_cached.value == seq1
    assert rotary_emb._cos_cached.value.shape[0] == seq1
    assert rotary_emb._sin_cached.value.shape[0] == seq1

    # Process smaller sequence - should use existing cache
    seq2 = 16
    qkv2 = jax.random.normal(jax.random.PRNGKey(31), (batch, seq2, 3, heads, dim))
    old_cos = rotary_emb._cos_cached.value
    _ = rotary_emb(qkv2)

    # Cache should not change
    assert jnp.array_equal(rotary_emb._cos_cached.value, old_cos)

    # Process larger sequence - should update cache
    seq3 = 64
    qkv3 = jax.random.normal(jax.random.PRNGKey(32), (batch, seq3, 3, heads, dim))
    _ = rotary_emb(qkv3)

    # Cache should be updated
    assert rotary_emb._seq_len_cached.value == seq3
    assert rotary_emb._cos_cached.value.shape[0] == seq3


def test_different_bases():
    """Test rotary embeddings with different base values."""
    rngs = nnx.Rngs(0)
    dim = 64

    bases = [1000.0, 10000.0, 100000.0]
    results = []

    batch, seq, heads = 1, 32, 4
    qkv = jax.random.normal(jax.random.PRNGKey(40), (batch, seq, 3, heads, dim))

    for base in bases:
        rotary_emb = RotaryEmbedding(dim=dim, base=base, rngs=rngs)
        result = rotary_emb(qkv)
        results.append(result)

    # Different bases should produce different results
    for i in range(len(results) - 1):
        assert not jnp.allclose(results[i], results[i + 1])


def test_gradient_flow():
    """Test that gradients flow through rotary embeddings."""
    # Test gradient flow through the functional interface
    batch, seq, heads, dim = 1, 16, 4, 64

    # Create inputs
    x = jax.random.normal(jax.random.PRNGKey(50), (batch, seq, heads, dim))
    rotary_dim = dim // 2
    cos = jnp.ones((seq, rotary_dim))
    sin = jnp.ones((seq, rotary_dim)) * 0.5

    # Define loss function using the functional interface
    def loss_fn(x):
        x_rot = apply_rotary_emb_jax(x, cos, sin)
        return jnp.sum(x_rot**2)

    # Compute gradient
    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(x)

    # Gradient should be non-zero
    assert not jnp.allclose(grad, 0.0)
    assert grad.shape == x.shape

    # Also test with apply_rotary_emb_qkv
    qkv = jax.random.normal(jax.random.PRNGKey(51), (batch, seq, 3, heads, dim))

    def loss_fn_qkv(qkv):
        qkv_rot = apply_rotary_emb_qkv(qkv, cos, sin, cos, sin)
        return jnp.sum(qkv_rot**2)

    grad_fn_qkv = jax.grad(loss_fn_qkv)
    grad_qkv = grad_fn_qkv(qkv)

    assert not jnp.allclose(grad_qkv, 0.0)
    assert grad_qkv.shape == qkv.shape


def test_numerical_stability():
    """Test numerical stability with large sequence lengths."""
    rngs = nnx.Rngs(0)
    dim = 64
    rotary_emb = RotaryEmbedding(dim=dim, rngs=rngs)

    # Test with large sequence length
    batch, seq, heads = 1, 8192, 2
    qkv = jax.random.normal(jax.random.PRNGKey(60), (batch, seq, 3, heads, dim))

    qkv_rot = rotary_emb(qkv)
    assert isinstance(qkv_rot, jax.Array)

    # Check for NaN or Inf
    assert not jnp.any(jnp.isnan(qkv_rot))
    assert not jnp.any(jnp.isinf(qkv_rot))

    # Check that norms are reasonable
    input_norm = jnp.linalg.norm(qkv)
    output_norm = jnp.linalg.norm(qkv_rot)
    assert jnp.abs(output_norm - input_norm) / input_norm < 0.1  # Within 10%


def test_interleaved_mode():
    """Test interleaved mode more thoroughly."""
    rngs = nnx.Rngs(0)
    dim = 64

    # Create rotary embeddings with and without interleaving
    rotary_standard = RotaryEmbedding(dim=dim, interleaved=False, rngs=rngs)
    rotary_interleaved = RotaryEmbedding(dim=dim, interleaved=True, rngs=rngs)

    batch, seq, heads = 1, 16, 4
    qkv = jax.random.normal(jax.random.PRNGKey(70), (batch, seq, 3, heads, dim))

    result_standard = rotary_standard(qkv)
    result_interleaved = rotary_interleaved(qkv)

    assert isinstance(result_standard, jax.Array)
    assert isinstance(result_interleaved, jax.Array)

    # Results should be different
    assert not jnp.allclose(result_standard, result_interleaved)

    # But shapes should be the same
    assert result_standard.shape == result_interleaved.shape


def test_broadcasting():
    """Test broadcasting behavior with different input shapes."""
    dim = 64
    seq = 16
    heads = 8

    # Test functional interface with different shapes
    shapes_functional = [
        (seq, heads, dim),  # No batch
        (1, seq, heads, dim),  # Single batch
        (2, seq, heads, dim),  # Multiple batch
        (2, 4, seq, heads, dim),  # Multiple batch dims
    ]

    rotary_dim = dim // 2
    cos = jnp.ones((seq, rotary_dim))
    sin = jnp.ones((seq, rotary_dim)) * 0.5

    for shape in shapes_functional:
        x = jax.random.normal(jax.random.PRNGKey(80), shape)
        result = apply_rotary_emb_jax(x, cos, sin)
        assert result.shape == shape

    # Test module interface with proper QKV format
    rngs = nnx.Rngs(0)
    rotary_emb = RotaryEmbedding(dim=dim, rngs=rngs)

    # Standard shapes for QKV tensors
    shapes_qkv = [
        (1, seq, 3, heads, dim),  # Single batch
        (2, seq, 3, heads, dim),  # Multiple batch
        (4, seq, 3, heads, dim),  # Larger batch
    ]

    for i, shape in enumerate(shapes_qkv):
        qkv = jax.random.normal(jax.random.PRNGKey(81 + i), shape)
        result = rotary_emb(qkv)
        assert isinstance(result, jax.Array)
        assert result.shape == shape


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    rngs = nnx.Rngs(0)

    # Test with minimum dimension (2)
    rotary_emb_min = RotaryEmbedding(dim=2, rngs=rngs)
    qkv = jax.random.normal(jax.random.PRNGKey(90), (1, 4, 3, 1, 2))
    result = rotary_emb_min(qkv)
    assert isinstance(result, jax.Array)
    assert result.shape == qkv.shape

    # Test with single sequence position
    dim = 64
    rotary_emb = RotaryEmbedding(dim=dim, rngs=rngs)
    qkv_single = jax.random.normal(jax.random.PRNGKey(91), (1, 1, 3, 4, dim))
    result_single = rotary_emb(qkv_single)
    assert isinstance(result_single, jax.Array)
    assert result_single.shape == qkv_single.shape


def test_odd_dimensions():
    """Test behavior with odd dimensions (should work with even rotary_dim)."""
    rngs = nnx.Rngs(0)

    # Odd total dimension but even rotary dimension
    dim = 63
    rotary_emb = RotaryEmbedding(dim=62, rngs=rngs)  # Use even rotary dim

    batch, seq, heads = 1, 8, 4
    qkv = jax.random.normal(jax.random.PRNGKey(100), (batch, seq, 3, heads, dim))

    result = rotary_emb(qkv)
    assert isinstance(result, jax.Array)
    assert result.shape == qkv.shape

    # Last dimension should be unchanged
    assert jnp.allclose(result[..., -1], qkv[..., -1])
