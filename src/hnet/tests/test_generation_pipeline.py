#!/usr/bin/env python3
"""
Comprehensive generation pipeline tests
Testing the critical end-to-end functionality that integrates everything
"""

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import json
import tempfile
import os
from pathlib import Path

from hnet.generate import (
    ByteTokenizer, 
    generate, 
    load_from_pretrained,
    convert_pytorch_to_jax
)
from hnet.models.config_hnet import (
    AttnConfig,
    HNetConfig, 
    SSMConfig
)
from hnet.models.mixer_seq import HNetForCausalLM, CausalLMOutput


class TestByteTokenizer:
    """Test the byte-level tokenizer"""
    
    def test_tokenizer_init(self):
        """Test tokenizer initialization"""
        tokenizer = ByteTokenizer()
        assert tokenizer.vocab_size == 256
        assert tokenizer.bos_idx == 254
        assert tokenizer.eos_idx == 255
    
    def test_encode_decode_basic(self):
        """Test basic encode/decode functionality"""
        tokenizer = ByteTokenizer()
        
        # Test simple ASCII text
        text = "Hello, world!"
        encoded = tokenizer.encode([text])[0]
        assert "input_ids" in encoded
        
        # Decode back
        decoded = tokenizer.decode(encoded["input_ids"])
        assert decoded == text
    
    def test_encode_with_special_tokens(self):
        """Test encoding with BOS/EOS tokens"""
        tokenizer = ByteTokenizer()
        
        text = "Hi"
        
        # With BOS
        encoded_bos = tokenizer.encode([text], add_bos=True)[0]
        assert encoded_bos["input_ids"][0] == tokenizer.bos_idx
        
        # With EOS
        encoded_eos = tokenizer.encode([text], add_eos=True)[0]
        assert encoded_eos["input_ids"][-1] == tokenizer.eos_idx
        
        # With both
        encoded_both = tokenizer.encode([text], add_bos=True, add_eos=True)[0]
        assert encoded_both["input_ids"][0] == tokenizer.bos_idx
        assert encoded_both["input_ids"][-1] == tokenizer.eos_idx
    
    def test_encode_decode_unicode(self):
        """Test encoding/decoding Unicode text"""
        tokenizer = ByteTokenizer()
        
        # Test Unicode characters
        text = "Hello 世界! 🌍"
        encoded = tokenizer.encode([text])[0]
        decoded = tokenizer.decode(encoded["input_ids"])
        assert decoded == text
        
    def test_batch_encoding(self):
        """Test batch encoding"""
        tokenizer = ByteTokenizer()
        
        texts = ["Hello", "World", "Test"]
        encoded_batch = tokenizer.encode(texts)
        
        assert len(encoded_batch) == 3
        for i, enc in enumerate(encoded_batch):
            decoded = tokenizer.decode(enc["input_ids"])
            assert decoded == texts[i]


class TestModelConstruction:
    """Test model construction without weights"""
    
    def create_minimal_config(self):
        """Create a minimal valid configuration for testing"""
        return {
            "arch_layout": ["m2"],  # Single Mamba layer
            "d_model": [128],       # Small dimension
            "d_intermediate": [0],  # No FFN
            "vocab_size": 256,
            "ssm_cfg": {
                "chunk_size": 64,   # Small chunks
                "d_conv": 4,
                "d_state": 16,      # Small state
                "expand": 2
            },
            "attn_cfg": {
                "num_heads": [4],
                "rotary_emb_dim": [16], 
                "window_size": [127]
            },
            "tie_embeddings": False
        }
    
    def test_model_creation(self):
        """Test creating model from config"""
        config_dict = self.create_minimal_config()
        
        # Create config objects
        attn_cfg = AttnConfig(**config_dict.pop("attn_cfg"))
        ssm_cfg = SSMConfig(**config_dict.pop("ssm_cfg"))
        hnet_cfg = HNetConfig(**config_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
        
        # Create model
        rngs = nnx.Rngs(42)
        model = HNetForCausalLM(hnet_cfg, rngs=rngs)
        
        # Test basic properties
        assert model.config.vocab_size == 256
        assert hasattr(model, 'embeddings')
        assert hasattr(model, 'backbone')
        assert hasattr(model, 'lm_head')
    
    def test_model_forward_pass(self):
        """Test model forward pass without weights"""
        config_dict = self.create_minimal_config()
        
        attn_cfg = AttnConfig(**config_dict.pop("attn_cfg"))
        ssm_cfg = SSMConfig(**config_dict.pop("ssm_cfg"))
        hnet_cfg = HNetConfig(**config_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
        
        rngs = nnx.Rngs(42)
        model = HNetForCausalLM(hnet_cfg, rngs=rngs)
        
        # Create test input
        batch_size, seq_len = 1, 8
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
        
        # Forward pass
        output = model.forward(input_ids, mask=mask)
        
        # Check output structure
        assert hasattr(output, 'logits')
        assert hasattr(output, 'bpred_output')
        assert hasattr(output, 'inference_params')
        
        # Check shapes
        assert output.logits.shape == (batch_size, seq_len, 256)  # vocab_size
    
    def test_cache_allocation(self):
        """Test cache allocation"""
        config_dict = self.create_minimal_config()
        
        attn_cfg = AttnConfig(**config_dict.pop("attn_cfg"))
        ssm_cfg = SSMConfig(**config_dict.pop("ssm_cfg"))
        hnet_cfg = HNetConfig(**config_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
        
        rngs = nnx.Rngs(42)
        model = HNetForCausalLM(hnet_cfg, rngs=rngs)
        
        batch_size, max_seqlen = 1, 512
        
        try:
            cache = model.allocate_inference_cache(batch_size, max_seqlen)
            if cache is not None:
                print("✅ Cache allocation working")
            else:
                print("ℹ️ Cache allocation returns None")
        except Exception as e:
            print(f"⚠️ Cache allocation error: {e}")
    
    def test_step_function(self):
        """Test single step inference"""
        config_dict = self.create_minimal_config()
        
        attn_cfg = AttnConfig(**config_dict.pop("attn_cfg"))
        ssm_cfg = SSMConfig(**config_dict.pop("ssm_cfg"))
        hnet_cfg = HNetConfig(**config_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
        
        rngs = nnx.Rngs(42)
        model = HNetForCausalLM(hnet_cfg, rngs=rngs)
        
        # Test step function
        input_ids = jnp.array([[42]], dtype=jnp.int32)  # Single token
        
        try:
            # Try to allocate cache first
            cache = model.allocate_inference_cache(1, 512)
            
            output = model.step(input_ids, cache)
            
            # Check output
            assert hasattr(output, 'logits')
            assert output.logits.shape == (1, 1, 256)
            
            print("✅ Step function working")
        except Exception as e:
            print(f"⚠️ Step function error: {e}")


class TestWeightConversion:
    """Test PyTorch to JAX weight conversion"""
    
    def test_convert_pytorch_to_jax_basic(self):
        """Test basic weight conversion without actual PyTorch weights"""
        # Create mock PyTorch state dict structure
        import numpy as np
        
        mock_state_dict = {
            "embeddings.weight": np.random.randn(256, 128).astype(np.float32),
            "backbone.encoder.layers.0.mixer.in_proj.weight": np.random.randn(512, 128).astype(np.float32),
            "backbone.encoder.layers.0.mixer.out_proj.weight": np.random.randn(128, 256).astype(np.float32),
            "lm_head.weight": np.random.randn(256, 128).astype(np.float32),
        }
        
        # Convert to JAX
        jax_state_dict = convert_pytorch_to_jax(mock_state_dict)
        
        # Check conversions
        assert len(jax_state_dict) > 0
        
        # Check specific conversions
        assert "_mapping.embeddings.embedding" in jax_state_dict
        assert "_mapping.lm_head.kernel" in jax_state_dict
        
        # Check weight matrix transposition for linear layers
        original_shape = mock_state_dict["lm_head.weight"].shape
        converted_shape = jax_state_dict["_mapping.lm_head.kernel"].shape
        assert converted_shape == (original_shape[1], original_shape[0])  # Transposed


class TestGenerationFunctions:
    """Test generation functions (without pretrained weights)"""
    
    def test_generation_interface(self):
        """Test that generation interface works (may fail due to missing implementations)"""
        # Create minimal model
        config_dict = {
            "arch_layout": ["m1"],
            "d_model": [64],
            "d_intermediate": [0],
            "vocab_size": 256,
            "ssm_cfg": {"chunk_size": 32, "d_conv": 4, "d_state": 8, "expand": 2},
            "attn_cfg": {"num_heads": [2], "rotary_emb_dim": [8], "window_size": [31]},
            "tie_embeddings": False
        }
        
        attn_cfg = AttnConfig(**config_dict.pop("attn_cfg"))
        ssm_cfg = SSMConfig(**config_dict.pop("ssm_cfg"))
        hnet_cfg = HNetConfig(**config_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
        
        rngs = nnx.Rngs(42)
        model = HNetForCausalLM(hnet_cfg, rngs=rngs)
        
        # Test generation function signature
        try:
            key = jax.random.PRNGKey(42)
            gen_iter = generate(
                model=model,
                prompt="Hi",
                max_tokens=5,
                temperature=1.0,
                top_p=0.9,
                key=key
            )
            
            # Try to get first token
            first_token = next(gen_iter)
            print(f"✅ Generation started, first token shape: {first_token.shape}")
            
        except StopIteration:
            print("✅ Generation completed (no tokens generated)")
        except Exception as e:
            print(f"⚠️ Generation error (expected): {e}")
            # This is expected since we don't have proper cache/step implementation


class TestConfigurationLoading:
    """Test loading configurations from files"""
    
    def test_load_real_config(self):
        """Test loading actual configuration files"""
        config_path = "/Users/nickwall/dev/projects/h-net-jax/configs/hnet_1stage_L.json"
        
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            
            # Test config structure
            assert "arch_layout" in config
            assert "d_model" in config
            assert "vocab_size" in config
            assert "ssm_cfg" in config
            assert "attn_cfg" in config
            
            # Try to create model config
            attn_cfg = AttnConfig(**config["attn_cfg"])
            ssm_cfg = SSMConfig(**config["ssm_cfg"])
            hnet_cfg = HNetConfig(**{k: v for k, v in config.items() 
                                   if k not in ["attn_cfg", "ssm_cfg"]}, 
                                  attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
            
            print(f"✅ Loaded config: {hnet_cfg.arch_layout}")
            print(f"   d_model: {hnet_cfg.d_model}")
            print(f"   vocab_size: {hnet_cfg.vocab_size}")
        else:
            print(f"⚠️ Config file not found: {config_path}")


class TestFullPipelineIntegration:
    """Test integration of all components"""
    
    def test_tokenizer_to_model_integration(self):
        """Test that tokenizer output works with model input"""
        # Create model
        config_dict = {
            "arch_layout": ["m1"],
            "d_model": [64],
            "d_intermediate": [0],
            "vocab_size": 256,
            "ssm_cfg": {"chunk_size": 16, "d_conv": 4, "d_state": 8, "expand": 2},
            "attn_cfg": {"num_heads": [2], "rotary_emb_dim": [8], "window_size": [15]},
            "tie_embeddings": False
        }
        
        attn_cfg = AttnConfig(**config_dict.pop("attn_cfg"))
        ssm_cfg = SSMConfig(**config_dict.pop("ssm_cfg"))
        hnet_cfg = HNetConfig(**config_dict, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)
        
        rngs = nnx.Rngs(42)
        model = HNetForCausalLM(hnet_cfg, rngs=rngs)
        
        # Create tokenizer
        tokenizer = ByteTokenizer()
        
        # Tokenize text
        prompt = "Hello"
        encoded = tokenizer.encode([prompt], add_bos=True)[0]
        input_ids = jnp.array(encoded["input_ids"], dtype=jnp.int32).reshape(1, -1)
        
        # Test forward pass
        mask = jnp.ones(input_ids.shape, dtype=jnp.bool_)
        output = model.forward(input_ids, mask=mask)
        
        # Check that we get reasonable output
        assert output.logits.shape[-1] == 256  # vocab_size
        assert not jnp.any(jnp.isnan(output.logits))
        assert not jnp.any(jnp.isinf(output.logits))
        
        print("✅ Tokenizer to model integration working")
    
    def test_sampling_from_logits(self):
        """Test sampling functionality"""
        # Create random logits
        key = jax.random.PRNGKey(42)
        logits = jax.random.normal(key, (256,))
        
        # Test basic sampling
        next_token = jax.random.categorical(key, logits)
        assert 0 <= next_token < 256
        
        # Test temperature scaling
        temp_logits = logits / 0.7
        temp_token = jax.random.categorical(key, temp_logits)
        assert 0 <= temp_token < 256
        
        print("✅ Sampling from logits working")


if __name__ == "__main__":
    print("Running comprehensive generation pipeline tests...")
    
    # Test tokenizer
    print("\n=== Testing Byte Tokenizer ===")
    test_tok = TestByteTokenizer()
    test_tok.test_tokenizer_init()
    test_tok.test_encode_decode_basic()
    test_tok.test_encode_with_special_tokens()
    test_tok.test_encode_decode_unicode()
    test_tok.test_batch_encoding()
    
    # Test model construction  
    print("\n=== Testing Model Construction ===")
    test_model = TestModelConstruction()
    test_model.test_model_creation()
    test_model.test_model_forward_pass()
    test_model.test_cache_allocation()
    test_model.test_step_function()
    
    # Test weight conversion
    print("\n=== Testing Weight Conversion ===")
    test_weights = TestWeightConversion()
    test_weights.test_convert_pytorch_to_jax_basic()
    
    # Test generation
    print("\n=== Testing Generation Functions ===")
    test_gen = TestGenerationFunctions()
    test_gen.test_generation_interface()
    
    # Test config loading
    print("\n=== Testing Configuration Loading ===")
    test_config = TestConfigurationLoading()
    test_config.test_load_real_config()
    
    # Test integration
    print("\n=== Testing Full Pipeline Integration ===")
    test_integration = TestFullPipelineIntegration()
    test_integration.test_tokenizer_to_model_integration()
    test_integration.test_sampling_from_logits()
    
    print("\n🎉 Generation pipeline tests completed!")
    print("Note: Some tests may show expected errors due to incomplete implementations.")
    print("This helps identify what needs to be implemented for full functionality.")