import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from hnet.modules.config import AttnConfig, HNetConfig, SSMConfig
from hnet.modules.isotropic import Isotropic, IsotropicInferenceParams


@pytest.fixture
def basic_config():
    return HNetConfig(
        arch_layout=["m4t4"],  # 4 Mamba blocks followed by 4 Transformer blocks
        d_model=[256],
        d_intermediate=[1024],
        vocab_size=256,
        ssm_cfg=SSMConfig(
            d_conv=4,
            expand=2,
            d_state=16,  # Small for testing
            chunk_size=32,  # Smaller chunk size for testing
        ),
        attn_cfg=AttnConfig(
            num_heads=[8],
            rotary_emb_dim=[32],  # Must not exceed head_dim (256/8 = 32)
            window_size=[-1],
        ),
    )


@pytest.fixture
def multi_stage_config():
    return HNetConfig(
        arch_layout=["m2t2", ["m4", "t4"]],  # 2-stage architecture
        d_model=[128, 256],
        d_intermediate=[512, 1024],
        vocab_size=256,
        ssm_cfg=SSMConfig(
            d_conv=4,
            expand=2,
            d_state=16,
            chunk_size=32,  # Smaller chunk size
        ),
        attn_cfg=AttnConfig(
            num_heads=[4, 8],
            rotary_emb_dim=[32, 32],  # head_dim = 128/4 = 32 and 256/8 = 32
            window_size=[-1, -1],
        ),
    )


def test_init_basic(basic_config):
    rngs = nnx.Rngs(0)
    model = Isotropic(basic_config, pos_idx=0, stage_idx=0, rngs=rngs)

    assert model.d_model == 256
    assert len(model.layers) == 8  # 4 Mamba + 4 Transformer
    assert len(model.arch_full) == 8
    assert model.arch_full == ["m", "m", "m", "m", "t", "t", "t", "t"]


def test_init_multi_stage(multi_stage_config):
    rngs = nnx.Rngs(0)

    # Stage 0
    model_stage0 = Isotropic(multi_stage_config, pos_idx=0, stage_idx=0, rngs=rngs)
    assert model_stage0.d_model == 128
    assert len(model_stage0.layers) == 4  # 2 Mamba + 2 Transformer

    # Stage 1, position 0
    model_stage1_pos0 = Isotropic(multi_stage_config, pos_idx=0, stage_idx=1, rngs=rngs)
    assert model_stage1_pos0.d_model == 256
    assert len(model_stage1_pos0.layers) == 4  # 4 Mamba
    assert all(arch == "m" for arch in model_stage1_pos0.arch_full)

    # Stage 1, position 1
    model_stage1_pos1 = Isotropic(multi_stage_config, pos_idx=1, stage_idx=1, rngs=rngs)
    assert model_stage1_pos1.d_model == 256
    assert len(model_stage1_pos1.layers) == 4  # 4 Transformer
    assert all(arch == "t" for arch in model_stage1_pos1.arch_full)


def test_forward_unpacked(basic_config):
    """Test forward pass with unpacked format (B, L, D)."""
    rngs = nnx.Rngs(0)
    model = Isotropic(basic_config, pos_idx=0, stage_idx=0, rngs=rngs)

    batch_size, seq_len = 2, 128  # seq_len divisible by chunk_size
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, 256))
    mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)

    output = model(x, mask=mask)

    assert output.shape == (batch_size, seq_len, 256)
    assert jnp.isfinite(output).all()


def test_forward_packed(basic_config):
    # Note: Currently only Mamba blocks support packed format
    # Create a config with only Mamba blocks
    config = HNetConfig(
        arch_layout=["m4"],  # Only Mamba blocks
        d_model=[256],
        d_intermediate=[1024],
        vocab_size=256,
        ssm_cfg=SSMConfig(
            d_conv=4,
            expand=2,
            d_state=16,
            chunk_size=32,
        ),
        attn_cfg=AttnConfig(
            num_heads=[8],
            rotary_emb_dim=[32],
            window_size=[-1],
        ),
    )

    rngs = nnx.Rngs(0)
    model = Isotropic(config, pos_idx=0, stage_idx=0, rngs=rngs)

    # Create packed sequences: 2 sequences of length 64 and 32 (both divisible by chunk_size=32)
    _seq_lens = jnp.array([64, 32])
    cu_seqlens = jnp.array([0, 64, 96])
    total_len = int(cu_seqlens[-1])

    x = jax.random.normal(jax.random.PRNGKey(0), (total_len, 256))

    output = model(x, cu_seqlens=cu_seqlens, max_seqlen=64)

    assert output.shape == (total_len, 256)
    assert jnp.isfinite(output).all()


def test_allocate_inference_cache(basic_config):
    rngs = nnx.Rngs(0)
    model = Isotropic(basic_config, pos_idx=0, stage_idx=0, rngs=rngs)

    batch_size, max_seqlen = 2, 512
    inference_params = model.allocate_inference_cache(batch_size, max_seqlen)

    assert isinstance(inference_params, IsotropicInferenceParams)
    assert inference_params.max_seqlen == max_seqlen
    assert inference_params.max_batch_size == batch_size
    assert inference_params.seqlen_offset == 0
    assert inference_params.cache_state is not None


def test_step():
    # Create a config with chunk_size=1 for single-token generation
    config = HNetConfig(
        arch_layout=["m2t2"],
        d_model=[256],
        d_intermediate=[1024],
        vocab_size=256,
        ssm_cfg=SSMConfig(
            d_conv=4,
            expand=2,
            d_state=16,
            chunk_size=1,  # Allow single token processing
        ),
        attn_cfg=AttnConfig(
            num_heads=[8],
            rotary_emb_dim=[32],
            window_size=[-1],
        ),
    )

    rngs = nnx.Rngs(0)
    model = Isotropic(config, pos_idx=0, stage_idx=0, rngs=rngs)

    batch_size = 2
    inference_params = model.allocate_inference_cache(batch_size, 512)

    # Single token input
    x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 1, 256))

    output = model.step(x, inference_params)

    assert output.shape == (batch_size, 1, 256)
    assert jnp.isfinite(output).all()
    assert inference_params.seqlen_offset == 1


def test_reset_inference_params():
    params = IsotropicInferenceParams(
        max_seqlen=512,
        max_batch_size=4,
        seqlen_offset=100,
        lengths_per_sample=jnp.ones((4,)),
    )

    params.reset(256, 2)

    assert params.max_seqlen == 256
    assert params.max_batch_size == 2
    assert params.seqlen_offset == 0
    assert jnp.all(params.lengths_per_sample == 0)


def test_invalid_config():
    rngs = nnx.Rngs(0)

    # Missing d_model for stage
    config = HNetConfig(
        arch_layout=["m4"],
        d_model=[256],
        d_intermediate=[1024],
        ssm_cfg=SSMConfig(),
        attn_cfg=AttnConfig(
            num_heads=[8],
            rotary_emb_dim=[32],
            window_size=[-1],
        ),
    )

    with pytest.raises(ValueError, match="d_model not configured"):
        Isotropic(config, pos_idx=0, stage_idx=1, rngs=rngs)

    # Missing arch_layout
    config = HNetConfig(
        d_model=[256],
        d_intermediate=[1024],
        ssm_cfg=SSMConfig(),
        attn_cfg=AttnConfig(
            num_heads=[8],
            rotary_emb_dim=[32],
            window_size=[-1],
        ),
    )

    with pytest.raises(ValueError, match="arch_layout must be provided"):
        Isotropic(config, pos_idx=0, stage_idx=0, rngs=rngs)


def test_different_architectures():
    rngs = nnx.Rngs(0)

    # Test uppercase variants
    config = HNetConfig(
        arch_layout=["M2T2m1t1"],  # Mix of uppercase and lowercase
        d_model=[256],
        d_intermediate=[1024],
        ssm_cfg=SSMConfig(chunk_size=32),
        attn_cfg=AttnConfig(
            num_heads=[8],
            rotary_emb_dim=[32],
            window_size=[-1],
        ),
    )

    model = Isotropic(config, pos_idx=0, stage_idx=0, rngs=rngs)
    assert len(model.layers) == 6
    assert model.arch_full == ["M", "M", "T", "T", "m", "t"]

    # Test longer sequences
    config = HNetConfig(
        arch_layout=["m10t5"],
        d_model=[256],
        d_intermediate=[1024],
        ssm_cfg=SSMConfig(chunk_size=32),
        attn_cfg=AttnConfig(
            num_heads=[8],
            rotary_emb_dim=[32],
            window_size=[-1],
        ),
    )

    model = Isotropic(config, pos_idx=0, stage_idx=0, rngs=rngs)
    assert len(model.layers) == 15
    assert model.arch_full.count("m") == 10
    assert model.arch_full.count("t") == 5


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
def test_dtype_support(dtype: jnp.dtype):
    config = HNetConfig(
        arch_layout=["m2t2"],
        d_model=[256],
        d_intermediate=[1024],
        vocab_size=256,
        ssm_cfg=SSMConfig(
            d_conv=4,
            expand=2,
            d_state=16,
            chunk_size=32,  # Make sure seq_len is divisible by chunk_size
        ),
        attn_cfg=AttnConfig(
            num_heads=[8],
            rotary_emb_dim=[32],
            window_size=[-1],
        ),
    )

    rngs = nnx.Rngs(0)
    model = Isotropic(config, pos_idx=0, stage_idx=0, rngs=rngs)

    x = jax.random.normal(
        jax.random.PRNGKey(0), (1, 64, 256), dtype=dtype
    )  # seq_len=64 divisible by 32
    mask = jnp.ones((1, 64), dtype=jnp.bool_)

    # Simply test that the model can process different dtype inputs
    output = model(x, mask=mask)

    # Output dtype may be model-dependent, just check it's finite
    assert jnp.isfinite(output).all()
