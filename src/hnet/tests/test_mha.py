import flax.nnx as nnx
import jax
import jax.numpy as jnp

from hnet.modules.mha import (
    CausalCrossAttention,
    CausalMHA,
    CausalSelfAttention,
    InferenceParams,
    causal_mask,
    scaled_dot_product_attention,
    sliding_window_mask,
)


def test_causal_mask():
    """Test causal mask generation."""
    # Test square mask
    seq_len = 4
    mask = causal_mask(seq_len)

    expected = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    assert jnp.allclose(mask, expected)

    # Test rectangular mask (cross-attention)
    seq_len = 3
    seq_len_k = 5
    mask = causal_mask(seq_len, seq_len_k)

    expected = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0],
        ]
    )

    assert jnp.allclose(mask, expected)
    assert mask.shape == (seq_len, seq_len_k)


def test_sliding_window_mask():
    """Test sliding window mask generation."""
    seq_len = 5
    window_size = 2
    mask = sliding_window_mask(seq_len, window_size)

    expected = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )

    assert jnp.allclose(mask, expected)

    # Test that window_size=-1 gives causal mask
    full_mask = sliding_window_mask(seq_len, -1)
    causal = causal_mask(seq_len)
    assert jnp.allclose(full_mask, causal)

    # Test with different sequence lengths
    seq_len = 3
    seq_len_k = 5
    window_size = 2
    mask = sliding_window_mask(seq_len, window_size, seq_len_k)

    expected = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
        ]
    )

    assert jnp.allclose(mask, expected)
    assert mask.shape == (seq_len, seq_len_k)


def test_scaled_dot_product_attention():
    """Test scaled dot product attention computation."""
    batch_size, seq_len, num_heads, head_dim = 2, 4, 2, 8
    key = jax.random.PRNGKey(0)

    q = jax.random.normal(key, (batch_size, seq_len, num_heads, head_dim))
    k = jax.random.normal(key, (batch_size, seq_len, num_heads, head_dim))
    v = jax.random.normal(key, (batch_size, seq_len, num_heads, head_dim))

    # Test without causal mask
    output = scaled_dot_product_attention(q, k, v, causal=False)
    assert output.shape == (batch_size, seq_len, num_heads, head_dim)

    # Test with causal mask
    output_causal = scaled_dot_product_attention(q, k, v, causal=True)
    assert output_causal.shape == (batch_size, seq_len, num_heads, head_dim)

    # Causal and non-causal should be different
    assert not jnp.allclose(output, output_causal)

    # Test with sliding window
    output_window = scaled_dot_product_attention(q, k, v, causal=True, window_size=2)
    assert output_window.shape == (batch_size, seq_len, num_heads, head_dim)


def test_causal_self_attention():
    """Test CausalSelfAttention module."""
    rngs = nnx.Rngs(0)
    batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 16

    # Initialize module
    attn = CausalSelfAttention(rngs=rngs)

    # Create QKV tensor
    qkv = jax.random.normal(rngs(), (batch_size, seq_len, 3, num_heads, head_dim))

    # Forward pass
    output = attn(qkv)
    assert output.shape == (batch_size, seq_len, num_heads, head_dim)

    # Test with custom softmax scale
    attn_scaled = CausalSelfAttention(softmax_scale=0.1, rngs=rngs)
    output_scaled = attn_scaled(qkv)
    assert output_scaled.shape == (batch_size, seq_len, num_heads, head_dim)

    # Test with window size
    attn_window = CausalSelfAttention(window_size=(3, -1), rngs=rngs)
    output_window = attn_window(qkv)
    assert output_window.shape == (batch_size, seq_len, num_heads, head_dim)


def test_causal_cross_attention():
    """Test CausalCrossAttention module."""
    rngs = nnx.Rngs(0)
    batch_size, seq_len_q, seq_len_kv, num_heads, head_dim = 2, 6, 8, 4, 16

    # Initialize module
    attn = CausalCrossAttention(rngs=rngs)

    # Create Q and KV tensors
    q = jax.random.normal(rngs(), (batch_size, seq_len_q, num_heads, head_dim))
    kv = jax.random.normal(rngs(), (batch_size, seq_len_kv, 2, num_heads, head_dim))

    # Forward pass
    output = attn(q, kv)
    assert output.shape == (batch_size, seq_len_q, num_heads, head_dim)


def test_causal_mha_basic():
    """Test basic CausalMHA functionality."""
    rngs = nnx.Rngs(0)
    d_model, num_heads = 256, 8
    batch_size, seq_len = 2, 16

    # Initialize module
    mha = CausalMHA(
        d_model=d_model,
        num_heads=num_heads,
        qkv_proj_bias=True,
        out_proj_bias=True,
        rngs=rngs,
    )

    # Create input
    x = jax.random.normal(rngs(), (batch_size, seq_len, d_model))

    # Forward pass
    output = mha(x)
    assert output.shape == (batch_size, seq_len, d_model)

    # Test gradient flow
    def loss_fn(x):
        return jnp.sum(mha(x))

    grad_fn = jax.grad(loss_fn)
    grad = grad_fn(x)
    assert grad.shape == x.shape


def test_causal_mha_with_rotary():
    """Test CausalMHA with rotary embeddings."""
    rngs = nnx.Rngs(0)
    d_model, num_heads = 256, 8
    # Rotary dim should be <= head_dim (256/8 = 32)
    rotary_dim = 16
    batch_size, seq_len = 2, 16

    # Initialize module with rotary embeddings
    mha = CausalMHA(
        d_model=d_model,
        num_heads=num_heads,
        rotary_emb_dim=rotary_dim,
        rotary_emb_base=10000.0,
        rotary_emb_interleaved=False,
        rngs=rngs,
    )

    # Create input
    x = jax.random.normal(rngs(), (batch_size, seq_len, d_model))

    # Forward pass
    output = mha(x)
    assert output.shape == (batch_size, seq_len, d_model)

    # Test with interleaved rotary
    mha_interleaved = CausalMHA(
        d_model=d_model,
        num_heads=num_heads,
        rotary_emb_dim=rotary_dim,
        rotary_emb_interleaved=True,
        rngs=rngs,
    )

    output_interleaved = mha_interleaved(x)
    assert output_interleaved.shape == (batch_size, seq_len, d_model)


def test_causal_mha_kv_cache():
    """Test CausalMHA with KV caching for autoregressive generation."""
    rngs = nnx.Rngs(0)
    d_model, num_heads = 256, 8
    batch_size, max_seq_len = 2, 32

    # Initialize module
    mha = CausalMHA(
        d_model=d_model,
        num_heads=num_heads,
        layer_idx=0,  # Required for KV caching
        rngs=rngs,
    )

    # Allocate KV cache
    mha.allocate_inference_cache(batch_size, max_seq_len)

    # Test generation mode
    inference_params = InferenceParams(
        max_batch_size=batch_size,
        max_seqlen=max_seq_len,
        seqlen_offset=0,
        batch_size_offset=0,
    )

    # Generate token by token
    for _ in range(10):
        # Single token input
        x = jax.random.normal(rngs(), (batch_size, 1, d_model))

        # Forward pass with caching
        output = mha(x, inference_params=inference_params)
        assert output.shape == (batch_size, 1, d_model)

        # Update sequence offset
        inference_params.seqlen_offset += 1

    # Test step method
    x = jax.random.normal(rngs(), (batch_size, 1, d_model))
    output = mha.step(x, inference_params)
    assert output.shape == (batch_size, 1, d_model)


def test_causal_mha_window_size():
    """Test CausalMHA with sliding window attention."""
    rngs = nnx.Rngs(0)
    d_model, num_heads = 256, 8
    window_size = 4
    batch_size, seq_len = 2, 16

    # Initialize module with window size
    mha = CausalMHA(
        d_model=d_model, num_heads=num_heads, window_size=window_size, rngs=rngs
    )

    # Create input
    x = jax.random.normal(rngs(), (batch_size, seq_len, d_model))

    # Forward pass
    output = mha(x)
    assert output.shape == (batch_size, seq_len, d_model)


def test_causal_mha_shapes():
    """Test CausalMHA with various input shapes."""
    rngs = nnx.Rngs(0)
    d_model, num_heads = 128, 4

    # Test different configurations
    configs = [
        (1, 8),  # Single sequence
        (4, 16),  # Multiple sequences
        (2, 1),  # Single token per sequence
        (8, 64),  # Longer sequences
    ]

    for batch_size, seq_len in configs:
        mha = CausalMHA(d_model=d_model, num_heads=num_heads, rngs=rngs)

        x = jax.random.normal(rngs(), (batch_size, seq_len, d_model))
        output = mha(x)
        assert output.shape == (batch_size, seq_len, d_model)


def test_causal_mha_dtypes():
    """Test CausalMHA with different data types."""
    rngs = nnx.Rngs(0)
    d_model, num_heads = 128, 4
    batch_size, seq_len = 2, 8

    for dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
        mha = CausalMHA(d_model=d_model, num_heads=num_heads, rngs=rngs)

        x = jax.random.normal(rngs(), (batch_size, seq_len, d_model), dtype=dtype)
        output = mha(x)
        assert output.shape == (batch_size, seq_len, d_model)
        # Linear layers may cast to float32 for numerical stability
        # So we only check that the output is a floating type
        assert jnp.issubdtype(output.dtype, jnp.floating)


def test_causal_mha_jit():
    """Test that CausalMHA works with JIT compilation."""
    rngs = nnx.Rngs(0)
    d_model, num_heads = 128, 4
    batch_size, seq_len = 2, 8

    # Initialize module
    mha = CausalMHA(d_model=d_model, num_heads=num_heads, rngs=rngs)

    # Create JIT-compiled forward function using nnx.jit
    @nnx.jit
    def forward(model, x):
        return model(x)

    # Test forward pass
    x = jax.random.normal(rngs(), (batch_size, seq_len, d_model))
    output = forward(mha, x)
    assert output.shape == (batch_size, seq_len, d_model)


def test_causal_mha_vmap():
    """Test that CausalMHA works with vmap."""
    rngs = nnx.Rngs(0)
    d_model, num_heads = 128, 4
    num_examples, batch_size, seq_len = 3, 2, 8

    # Initialize module
    mha = CausalMHA(d_model=d_model, num_heads=num_heads, rngs=rngs)

    # Create vmapped forward function
    @nnx.vmap(in_axes=(None, 0), out_axes=0)
    def forward(mha, x):
        return mha(x)

    # Test forward pass
    x = jax.random.normal(rngs(), (num_examples, batch_size, seq_len, d_model))
    output = forward(mha, x)
    assert output.shape == (num_examples, batch_size, seq_len, d_model)


def test_inference_params():
    """Test InferenceParams initialization."""
    params = InferenceParams(
        max_batch_size=4, max_seqlen=128, seqlen_offset=10, batch_size_offset=2
    )

    assert params.max_batch_size == 4
    assert params.max_seqlen == 128
    assert params.seqlen_offset == 10
    assert params.batch_size_offset == 2
    assert params.lengths_per_sample is None
    assert isinstance(params.key_value_memory_dict, dict)
