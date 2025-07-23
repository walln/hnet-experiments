#!/usr/bin/env python3
"""
JAX H-Net implementation based on the working PyTorch non-Triton version
"""

import json
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Any, Dict, List
from dataclasses import dataclass
import einops

from .mamba_jax import Mamba2JAX, RMSNorm


@dataclass
class HNetJAXConfig:
    """JAX H-Net configuration"""
    d_model: List[int]
    d_intermediate: List[int] 
    vocab_size: int
    arch_layout: List
    ssm_cfg: Dict
    attn_cfg: Dict
    tie_embeddings: bool = False


class SimpleMHAJAX(nn.Module):
    """JAX Multi-Head Attention for testing"""
    
    d_model: int
    num_heads: int = 16
    layer_idx: Optional[int] = None
    dtype: Any = jnp.float32
    
    def setup(self):
        self.head_dim = self.d_model // self.num_heads
        
        self.Wqkv = nn.Dense(3 * self.d_model, use_bias=False, dtype=self.dtype)
        self.out_proj = nn.Dense(self.d_model, use_bias=False, dtype=self.dtype)
    
    def __call__(self, x, inference_params=None, **kwargs):
        batch, seqlen, d_model = x.shape
        
        qkv = self.Wqkv(x)  # (batch, seqlen, 3 * d_model)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head attention
        q = einops.rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = einops.rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = einops.rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)
        
        # Scaled dot-product attention with causal mask
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn = jnp.einsum("bhld,bhsd->bhls", q, k) * scale
        
        # Apply causal mask
        causal_mask = jnp.tril(jnp.ones((seqlen, seqlen), dtype=bool))
        attn = jnp.where(causal_mask, attn, -jnp.inf)
        
        attn = nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhls,bhsd->bhld", attn, v)
        
        # Reshape back
        out = einops.rearrange(out, "b h l d -> b l (h d)")
        
        return self.out_proj(out)
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {'batch_size': batch_size, 'max_seqlen': max_seqlen}


class SimpleBlockJAX(nn.Module):
    """JAX transformer block"""
    
    d_model: int
    mixer_cls: Any
    mlp_cls: Optional[Any] = None
    layer_idx: Optional[int] = None
    dtype: Any = jnp.float32
    
    def setup(self):
        self.norm1 = RMSNorm(eps=1e-5)
        self.mixer = self.mixer_cls(self.d_model, layer_idx=self.layer_idx, dtype=self.dtype)
        
        if self.mlp_cls is not None:
            self.norm2 = RMSNorm(eps=1e-5)
            self.mlp = self.mlp_cls(self.d_model, dtype=self.dtype)
        else:
            self.mlp = None
    
    def __call__(self, x, inference_params=None, **kwargs):
        # Pre-norm architecture
        residual = x
        x = self.norm1(x)
        x = self.mixer(x, inference_params=inference_params, **kwargs)
        x = x + residual
        
        if self.mlp is not None:
            residual = x
            x = self.norm2(x)
            x = self.mlp(x)
            x = x + residual
        
        return x
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class SimpleSwiGLUJAX(nn.Module):
    """JAX SwiGLU MLP"""
    
    d_model: int
    d_intermediate: Optional[int] = None
    dtype: Any = jnp.float32
    
    def setup(self):
        if self.d_intermediate is None:
            self.d_intermediate = 4 * self.d_model
            
        self.fc1 = nn.Dense(2 * self.d_intermediate, use_bias=False, dtype=self.dtype)
        self.fc2 = nn.Dense(self.d_model, use_bias=False, dtype=self.dtype)
    
    def __call__(self, x):
        x1, x2 = jnp.split(self.fc1(x), 2, axis=-1)
        return self.fc2(x1 * nn.silu(x2))


class SimpleIsotropicJAX(nn.Module):
    """JAX isotropic layer stack"""
    
    config: HNetJAXConfig
    stage_idx: int
    pos_idx: int
    dtype: Any = jnp.float32
    
    def setup(self):
        self.d_model = self.config.d_model[self.stage_idx]
        
        # Parse architecture layout
        arch_layout = self.config.arch_layout
        for _ in range(self.stage_idx):
            arch_layout = arch_layout[self.pos_idx] if isinstance(arch_layout[0], list) else arch_layout
        
        # For simplicity, assume m4 = 4 Mamba layers, T22 = 22 Transformer layers
        if isinstance(arch_layout, str):
            if arch_layout.startswith('m'):
                num_layers = int(arch_layout[1:])
                
                def mixer_cls(d_model, layer_idx=None, dtype=jnp.float32):
                    return Mamba2JAX(
                        d_model=d_model, 
                        d_state=self.config.ssm_cfg['d_state'],
                        d_conv=self.config.ssm_cfg['d_conv'],
                        expand=self.config.ssm_cfg['expand'],
                        headdim=64,  # Match real H-Net weights (32 heads * 64 headdim = 2048 d_inner) 
                        chunk_size=self.config.ssm_cfg['chunk_size'],
                        layer_idx=layer_idx,
                        dtype=dtype
                    )
                mlp_cls = None
            elif arch_layout.startswith('T'):
                num_layers = int(arch_layout[1:])
                
                def mixer_cls(d_model, layer_idx=None, dtype=jnp.float32):
                    return SimpleMHAJAX(
                        d_model=d_model,
                        num_heads=self.config.attn_cfg['num_heads'][self.stage_idx],
                        layer_idx=layer_idx,
                        dtype=dtype
                    )
                
                def mlp_cls(d_model, dtype=jnp.float32):
                    return SimpleSwiGLUJAX(
                        d_model=d_model, 
                        d_intermediate=self.config.d_intermediate[self.stage_idx],
                        dtype=dtype
                    )
        else:
            # Default to 4 Mamba layers
            num_layers = 4
            
            def mixer_cls(d_model, layer_idx=None, dtype=jnp.float32):
                return Mamba2JAX(
                    d_model=d_model, 
                    d_state=self.config.ssm_cfg['d_state'],
                    d_conv=self.config.ssm_cfg['d_conv'],
                    expand=self.config.ssm_cfg['expand'],
                    headdim=64,  # Match real H-Net weights (32 heads * 64 headdim = 2048 d_inner)
                    chunk_size=self.config.ssm_cfg['chunk_size'],
                    layer_idx=layer_idx,
                    dtype=dtype
                )
            mlp_cls = None
        
        # Create layers
        self.layers = [
            SimpleBlockJAX(
                d_model=self.d_model,
                mixer_cls=mixer_cls,
                mlp_cls=mlp_cls,
                layer_idx=i,
                dtype=self.dtype
            )
            for i in range(num_layers)
        ]
        
        self.norm = RMSNorm(eps=1e-5)
    
    def __call__(self, x, mask=None, inference_params=None, **kwargs):
        for layer in self.layers:
            x = layer(x, inference_params=inference_params, **kwargs)
        
        x = self.norm(x)
        return x
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        cache_dict = {}
        for i, layer in enumerate(self.layers):
            cache_dict[i] = layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
        return cache_dict


class SimpleHNetJAX(nn.Module):
    """JAX H-Net implementation"""
    
    config: HNetJAXConfig
    dtype: Any = jnp.float32
    
    def setup(self):
        self.d_model = self.config.d_model[0]  # First stage
        
        # For 1-stage model, arch_layout = ["m4", ["T22"], "m4"]
        # This means: encoder=m4, main_network=T22, decoder=m4
        
        self.encoder = SimpleIsotropicJAX(
            config=self.config, stage_idx=0, pos_idx=0, dtype=self.dtype
        )     # m4
        self.main_network = SimpleIsotropicJAX(
            config=self.config, stage_idx=1, pos_idx=0, dtype=self.dtype
        ) # T22  
        self.decoder = SimpleIsotropicJAX(
            config=self.config, stage_idx=0, pos_idx=2, dtype=self.dtype
        )    # m4
        
        # Add projection layers for dimension changes
        self.enc_to_main = nn.Dense(
            self.config.d_model[1], use_bias=False, dtype=self.dtype
        )
        self.main_to_dec = nn.Dense(
            self.config.d_model[0], use_bias=False, dtype=self.dtype
        )
        
        # Simplified - no chunking/routing for now
        self.is_innermost = False
    
    def __call__(self, x, mask=None, inference_params=None, **kwargs):
        # Simplified forward pass without hierarchical chunking
        
        # Encoder (1024-dim)
        x = self.encoder(
            x, mask=mask, 
            inference_params=inference_params.get('encoder') if inference_params else None, 
            **kwargs
        )
        
        # Project to main network dimensions (1024 -> 1536)
        x = self.enc_to_main(x)
        
        # Main network (1536-dim, at higher resolution)
        # For now, just pass through - real H-Net would do chunking here
        x = self.main_network(
            x, mask=mask, 
            inference_params=inference_params.get('main_network') if inference_params else None, 
            **kwargs
        )
        
        # Project back to decoder dimensions (1536 -> 1024)
        x = self.main_to_dec(x)  
        
        # Decoder (1024-dim)
        x = self.decoder(
            x, mask=mask, 
            inference_params=inference_params.get('decoder') if inference_params else None, 
            **kwargs
        )
        
        return x, []  # Empty bpred_output for simplicity
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        return {
            'encoder': self.encoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype),
            'main_network': self.main_network.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype),
            'decoder': self.decoder.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype),
        }


class SimpleHNetForCausalLMJAX(nn.Module):
    """JAX H-Net for Causal Language Modeling"""
    
    config: HNetJAXConfig
    dtype: Any = jnp.float32
    
    def setup(self):
        vocab_size = self.config.vocab_size
        d_embed = self.config.d_model[0]
        
        self.embeddings = nn.Embed(vocab_size, d_embed, dtype=self.dtype)
        self.backbone = SimpleHNetJAX(config=self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(vocab_size, use_bias=False, dtype=self.dtype)
        
        # Note: tie_embeddings would require special handling in JAX/Flax
        # For now, we skip this feature
    
    def __call__(self, input_ids, mask=None, inference_params=None, **kwargs):
        x = self.embeddings(input_ids)
        
        if mask is None:
            batch, seqlen = input_ids.shape
            mask = jnp.ones((batch, seqlen), dtype=bool)
        
        x, bpred_output = self.backbone(x, mask=mask, inference_params=inference_params, **kwargs)
        logits = self.lm_head(x)
        
        # Return in expected format (simplified)
        return {
            'logits': logits,
            'bpred_output': bpred_output,
            'inference_params': inference_params
        }
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype)
    
    def step(self, input_ids, inference_params):
        # Simplified step function
        batch = input_ids.shape[0]
        assert batch == 1, "Only support batch size 1 for step"
        
        x = self.embeddings(input_ids)
        x, bpred_output = self.backbone(x, inference_params=inference_params)
        logits = self.lm_head(x)
        
        return {
            'logits': logits,
            'bpred_output': bpred_output,
            'inference_params': inference_params
        }


def create_simple_hnet_jax_from_config(config_dict, dtype=jnp.float32):
    """Create JAX SimpleHNet from config dictionary"""
    
    # Convert to our JAX config format
    jax_config = HNetJAXConfig(
        d_model=config_dict['d_model'],
        d_intermediate=config_dict['d_intermediate'],
        vocab_size=config_dict['vocab_size'],
        arch_layout=config_dict['arch_layout'],
        ssm_cfg=config_dict['ssm_cfg'],
        attn_cfg=config_dict['attn_cfg'],
        tie_embeddings=config_dict.get('tie_embeddings', False)
    )
    
    return SimpleHNetForCausalLMJAX(config=jax_config, dtype=dtype)


def test_simple_hnet_jax():
    """Test the JAX H-Net implementation"""
    print("Testing JAX H-Net...")
    
    # Test config similar to real H-Net
    config_dict = {
        "arch_layout": ["m4", ["T22"], "m4"],
        "d_model": [1024, 1536],
        "d_intermediate": [0, 4096],
        "vocab_size": 256,
        "ssm_cfg": {
            "chunk_size": 1,  # Use chunk_size=1 to avoid divisibility issues
            "d_conv": 4,
            "d_state": 128,
            "expand": 2
        },
        "attn_cfg": {
            "num_heads": [16, 16],
            "rotary_emb_dim": [32, 48],
            "window_size": [1023, -1]
        },
        "tie_embeddings": False
    }
    
    try:
        # Create model
        model = create_simple_hnet_jax_from_config(config_dict, dtype=jnp.float32)
        print("✅ Model created successfully")
        
        # Test forward pass
        batch, seqlen = 1, 16
        rng = jax.random.PRNGKey(42)
        input_ids = jax.random.randint(rng, (batch, seqlen), minval=0, maxval=256)
        
        print(f"Input shape: {input_ids.shape}")
        
        # Initialize parameters
        params = model.init(rng, input_ids)
        print("✅ Parameters initialized successfully")
        
        # Forward pass
        output = model.apply(params, input_ids)
        
        print(f"✅ Forward pass successful!")
        print(f"   Logits shape: {output['logits'].shape}")
        print(f"   Logits mean: {jnp.mean(output['logits']):.6f}")
        print(f"   Logits std: {jnp.std(output['logits']):.6f}")
        print(f"   Contains NaN: {jnp.any(jnp.isnan(output['logits']))}")
        print(f"   Contains Inf: {jnp.any(jnp.isinf(output['logits']))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("JAX H-Net Implementation")
    print("=" * 40)
    
    success = test_simple_hnet_jax()
    
    if success:
        print("\n🎉 JAX H-Net implementation works!")
    else:
        print("\n❌ JAX H-Net implementation failed.")