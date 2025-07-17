import flax.nnx as nnx
import jax
import jax.numpy as jnp

from hnet.modules.chunking import (
    ChunkLayer,
    DeChunkLayer,
    DeChunkState,
    RoutingModule,
    RoutingModuleOutput,
    RoutingModuleState,
    get_seq_idx,
    normalize,
)


def test_normalize():
    """Test the normalize function."""
    # Test basic normalization
    x = jnp.array([[1.0, 0.0], [3.0, 4.0]])
    normalized = normalize(x, axis=-1)

    # Check that norms are 1
    norms = jnp.linalg.norm(normalized, axis=-1)
    assert jnp.allclose(norms, jnp.ones_like(norms), atol=1e-6)

    # Test with zero vector (should handle eps)
    x_zero = jnp.zeros((2, 3))
    normalized_zero = normalize(x_zero)
    assert not jnp.any(jnp.isnan(normalized_zero))


def test_get_seq_idx():
    """Test sequence index generation from cumulative lengths."""
    cu_seqlens = jnp.array([0, 3, 7, 10])
    expected = jnp.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])

    result = get_seq_idx(cu_seqlens)
    assert jnp.array_equal(result, expected)


def test_routing_module_init():
    """Test RoutingModule initialization."""
    rngs = nnx.Rngs(0)
    d_model = 128

    module = RoutingModule(d_model, rngs=rngs)

    # Check attributes
    assert module.d_model == d_model
    assert hasattr(module, "q_proj_layer")
    assert hasattr(module, "k_proj_layer")

    # Check that projections are initialized as identity
    assert jnp.allclose(module.q_proj_layer.kernel.value, jnp.eye(d_model), atol=1e-5)
    assert jnp.allclose(module.k_proj_layer.kernel.value, jnp.eye(d_model), atol=1e-5)


def test_routing_module_cache_allocation():
    """Test cache allocation for inference."""
    rngs = nnx.Rngs(0)
    d_model = 64
    batch_size = 2
    max_seqlen = 100

    module = RoutingModule(d_model, rngs=rngs)
    cache = module.allocate_inference_cache(batch_size, max_seqlen)

    assert isinstance(cache, RoutingModuleState)
    assert cache.has_seen_tokens.shape == (batch_size,)
    assert cache.last_hidden_state.shape == (batch_size, d_model)
    assert cache.has_seen_tokens.dtype == jnp.bool_


def test_routing_module_forward():
    """Test RoutingModule forward pass."""
    rngs = nnx.Rngs(0)
    d_model = 64
    batch_size, seq_len = 2, 10

    module = RoutingModule(d_model, rngs=rngs)
    hidden_states = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, d_model)
    )
    mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)

    output = module(hidden_states, mask=mask)

    assert isinstance(output, RoutingModuleOutput)
    assert output.boundary_prob.shape == (batch_size, seq_len, 2)
    assert output.boundary_mask.shape == (batch_size, seq_len)
    assert output.selected_probs.shape == (batch_size, seq_len, 1)

    # Check that probabilities sum to 1
    prob_sums = jnp.sum(output.boundary_prob, axis=-1)
    assert jnp.allclose(prob_sums, jnp.ones_like(prob_sums))

    # Check that first token is always a boundary
    assert jnp.all(output.boundary_mask[:, 0])


def test_routing_module_packed_sequences():
    """Test RoutingModule with packed sequences."""
    rngs = nnx.Rngs(0)
    d_model = 64
    total_len = 15

    module = RoutingModule(d_model, rngs=rngs)
    hidden_states = jax.random.normal(jax.random.PRNGKey(0), (total_len, d_model))
    cu_seqlens = jnp.array([0, 5, 10, 15])

    output = module(hidden_states, cu_seqlens=cu_seqlens)

    assert output.boundary_prob.shape == (total_len, 2)
    assert output.boundary_mask.shape == (total_len,)

    # Check that sequence starts are boundaries
    assert output.boundary_mask[0]
    assert output.boundary_mask[5]
    assert output.boundary_mask[10]


def test_routing_module_step():
    """Test single-step inference."""
    rngs = nnx.Rngs(0)
    d_model = 64
    batch_size = 2

    module = RoutingModule(d_model, rngs=rngs)
    cache = module.allocate_inference_cache(batch_size, 1)

    # First step
    hidden = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 1, d_model))
    output = module.step(hidden, cache)

    assert output.boundary_prob.shape == (batch_size, 2)
    assert output.boundary_mask.shape == (batch_size,)
    assert jnp.all(cache.has_seen_tokens)

    # Second step should use cached state
    hidden2 = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 1, d_model))
    output2 = module.step(hidden2, cache)

    # Boundary probability should be different from 1.0 for second token
    assert not jnp.all(output2.boundary_prob[:, 1] == 1.0)


def test_chunk_layer():
    """Test ChunkLayer forward pass."""
    batch_size, seq_len, d_model = 2, 10, 64
    hidden_states = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, d_model)
    )

    # Create boundary mask with some boundaries
    boundary_mask = jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)
    boundary_mask = boundary_mask.at[:, jnp.array([0, 3, 7])].set(True)

    mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)

    chunker = ChunkLayer()
    next_hidden, next_cu_seqlens, next_max_seqlen, next_mask = chunker(
        hidden_states, boundary_mask, mask=mask
    )

    # Check output shapes
    _expected_tokens = jnp.sum(boundary_mask)
    assert next_hidden.shape[0] == batch_size
    assert next_hidden.shape[2] == d_model
    # next_max_seqlen is None for non-packed sequences
    assert next_max_seqlen is None

    # Check mask validity
    assert next_mask is not None
    # Since next_max_seqlen is None, check against actual mask shape
    num_tokens = jnp.sum(boundary_mask, axis=-1)
    max_tokens = int(jnp.max(num_tokens))
    assert next_mask.shape == (batch_size, max_tokens)
    # Check that valid positions are True
    for i in range(batch_size):
        assert jnp.all(next_mask[i, : num_tokens[i]])


def test_chunk_layer_packed():
    """Test ChunkLayer with packed sequences."""
    total_len, d_model = 15, 64
    hidden_states = jax.random.normal(jax.random.PRNGKey(0), (total_len, d_model))

    # Boundaries at positions 0, 5, 10
    boundary_mask = jnp.zeros(total_len, dtype=jnp.bool_)
    boundary_mask = boundary_mask.at[jnp.array([0, 5, 10])].set(True)

    cu_seqlens = jnp.array([0, 5, 10, 15])

    chunker = ChunkLayer()
    next_hidden, next_cu_seqlens, next_max_seqlen, next_mask = chunker(
        hidden_states, boundary_mask, cu_seqlens=cu_seqlens
    )

    assert next_hidden.shape == (3, d_model)  # 3 boundaries
    assert next_cu_seqlens is not None
    assert len(next_cu_seqlens) == 4  # batch_size + 1
    assert next_cu_seqlens[0] == 0
    assert next_cu_seqlens[-1] == 3


def test_chunk_layer_step():
    """Test ChunkLayer step mode."""
    batch_size, d_model = 3, 64
    hidden_states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, d_model))

    # Only select some elements
    boundary_mask = jnp.array([True, False, True])

    chunker = ChunkLayer()
    selected = chunker.step(hidden_states, boundary_mask)

    assert selected.shape == (2, d_model)  # Only 2 selected
    assert jnp.allclose(selected[0], hidden_states[0])
    assert jnp.allclose(selected[1], hidden_states[2])


def test_dechunk_layer_init():
    """Test DeChunkLayer initialization."""
    rngs = nnx.Rngs(0)
    d_model = 128

    layer = DeChunkLayer(d_model, chunk_size=256, headdim=32, rngs=rngs)

    assert layer.d_model == d_model
    assert layer.dtype == jnp.float32
    assert layer.chunk_size == 256
    assert layer.headdim == 32
    assert layer.nheads == d_model // 32


def test_dechunk_layer_cache():
    """Test DeChunkLayer cache allocation."""
    rngs = nnx.Rngs(0)
    d_model = 64
    batch_size = 2

    layer = DeChunkLayer(d_model, chunk_size=256, headdim=32, rngs=rngs)
    cache = layer.allocate_inference_cache(batch_size, 100)

    assert isinstance(cache, DeChunkState)
    assert cache.last_value.shape == (batch_size, d_model)


def test_dechunk_layer_forward():
    """Test DeChunkLayer forward pass."""
    rngs = nnx.Rngs(0)
    d_model = 64
    batch_size, seq_len = 2, 8  # Changed to be divisible by chunk_size

    # Create chunked hidden states (fewer tokens)
    chunked_len = 4  # Adjusted for new seq_len
    hidden_states = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, chunked_len, d_model)
    )

    # Create boundary mask indicating which positions are boundaries
    boundary_mask = jnp.zeros((batch_size, seq_len), dtype=jnp.bool_)
    boundary_mask = boundary_mask.at[:, jnp.array([0, 2, 4, 6])].set(True)

    # Create boundary probabilities
    boundary_prob = jnp.ones((batch_size, seq_len, 2)) * 0.5
    boundary_prob = boundary_prob.at[:, :, 1].set(0.8)  # High boundary probability

    mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)

    layer = DeChunkLayer(d_model, chunk_size=8, headdim=32, rngs=rngs)
    output = layer(hidden_states, boundary_mask, boundary_prob, mask=mask)

    # Output should have original sequence length
    assert output.shape == (batch_size, seq_len, d_model)
    assert not jnp.any(jnp.isnan(output))


def test_dechunk_layer_step():
    """Test DeChunkLayer step mode."""
    rngs = nnx.Rngs(0)
    d_model = 64
    batch_size = 3

    layer = DeChunkLayer(d_model, chunk_size=256, headdim=32, rngs=rngs)
    cache = layer.allocate_inference_cache(batch_size, 1)

    # Only some positions have new values
    boundary_mask = jnp.array([True, False, True])
    hidden_states = jax.random.normal(
        jax.random.PRNGKey(0), (2, 1, d_model)
    )  # Only 2 values
    boundary_prob = jnp.array([[0.2, 0.8], [0.5, 0.5], [0.1, 0.9]])

    output = layer.step(hidden_states, boundary_mask, boundary_prob, cache)

    assert output.shape == (batch_size, 1, d_model)
    assert not jnp.any(jnp.isnan(output))


def test_end_to_end_chunking():
    """Test complete chunking pipeline."""
    rngs = nnx.Rngs(0)
    d_model = 64
    batch_size, seq_len = 2, 16  # Changed to be divisible by chunk_size

    # Create modules
    router = RoutingModule(d_model, rngs=rngs)
    chunker = ChunkLayer()
    dechunker = DeChunkLayer(d_model, chunk_size=16, headdim=32, rngs=rngs)

    # Create input
    hidden_states = jax.random.normal(
        jax.random.PRNGKey(0), (batch_size, seq_len, d_model)
    )
    mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)

    # Forward pass
    routing_output = router(hidden_states, mask=mask)
    chunked, _, _, _ = chunker(hidden_states, routing_output.boundary_mask, mask=mask)
    reconstructed = dechunker(
        chunked, routing_output.boundary_mask, routing_output.boundary_prob, mask=mask
    )

    # Check shape preservation
    assert reconstructed.shape == hidden_states.shape

    # The reconstruction won't be exact due to the EMA operation,
    # but it should be reasonable
    assert not jnp.any(jnp.isnan(reconstructed))


def test_gradient_flow():
    """Test gradient flow through the chunking pipeline."""
    rngs = nnx.Rngs(0)
    d_model = 32
    batch_size, seq_len = 1, 8  # Changed to be divisible by smaller chunk_size

    # Create modules
    router = RoutingModule(d_model, rngs=rngs)
    chunker = ChunkLayer()
    dechunker = DeChunkLayer(d_model, chunk_size=8, headdim=16, rngs=rngs)

    def loss_fn(router, dechunker, x, mask):
        routing_output = router(x, mask=mask)
        chunked, _, _, _ = chunker(x, routing_output.boundary_mask, mask=mask)
        reconstructed = dechunker(
            chunked,
            routing_output.boundary_mask,
            routing_output.boundary_prob,
            mask=mask,
        )
        return jnp.mean((reconstructed - x) ** 2)

    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, d_model))
    mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)

    # Compute gradients
    grad_fn = nnx.grad(loss_fn, argnums=(0, 1))
    router_grads, dechunker_grads = grad_fn(router, dechunker, x, mask)

    # Check that gradients exist
    assert router_grads.q_proj_layer.kernel.value is not None
    assert router_grads.k_proj_layer.kernel.value is not None
    assert not jnp.any(jnp.isnan(router_grads.q_proj_layer.kernel.value))
    assert not jnp.any(jnp.isnan(router_grads.k_proj_layer.kernel.value))


def test_pytree_registration():
    """Test that dataclasses are properly registered as pytrees."""
    # Test RoutingModuleOutput
    output = RoutingModuleOutput(
        boundary_prob=jnp.ones((2, 10, 2)),
        boundary_mask=jnp.ones((2, 10), dtype=jnp.bool_),
        selected_probs=jnp.ones((2, 10, 1)),
    )

    # Test tree map
    doubled = jax.tree_util.tree_map(lambda x: x * 2, output)
    assert jnp.allclose(doubled.boundary_prob, output.boundary_prob * 2)

    # Test RoutingModuleState
    state = RoutingModuleState(
        has_seen_tokens=jnp.ones(2, dtype=jnp.bool_),
        last_hidden_state=jnp.ones((2, 64)),
    )

    # Test tree flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(state)
    assert len(leaves) == 2
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.array_equal(reconstructed.has_seen_tokens, state.has_seen_tokens)
