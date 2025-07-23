# H-Net JAX Implementation

## Project Overview
This repository implements the H-Net paper in JAX/Flax, with reference implementations from PyTorch available for comparison. The project focuses on creating a pure JAX implementation that can run on macOS without CUDA/Triton dependencies.

## Key Directories
- `src/hnet/` - Main JAX implementation
- `configs/` - Model configuration files
- `weights/` - Pre-trained model weights in various formats
- `hnet-reference/` - Reference PyTorch implementation
- `mamba/` - Mamba reference implementation (CUDA-dependent)
- `causal-conv1d/` - Causal convolution reference (CUDA-dependent)
- `src/hnet/tests/` - Test suite for components

## Development Environment
- **Platform**: macOS (no CUDA/Triton compilation available)
- **Python**: >=3.12
- **Package Manager**: uv
- **Testing**: pytest
- **Linting**: ruff

## Common Commands
```bash
# Run tests (preferred method for validation)
uv run pytest -v

# Run specific test file
uv run pytest src/hnet/tests/test_mamba2.py -v

# Run linting
uv run ruff check

# Format code
uv run ruff format
```

## Testing Strategy
Since CUDA/Triton code cannot be compiled on macOS, validation relies on:
1. Mathematical analysis and careful programming
2. Comprehensive test suite (`uv run pytest -v`)
3. Component-by-component validation
4. Cross-reference with PyTorch implementations through mathematical equivalence

## Implementation Notes
- Focus on JAX/Flax native implementations
- Avoid CUDA-specific optimizations from reference code
- Trace through reference implementations for mathematical correctness
- Use tests extensively to validate each component
- Convert weights from PyTorch format as needed

## Reference Materials
- Paper: `files/paper.pdf`
- PyTorch reference: `hnet-reference/`
- Mamba reference: `mamba/` (for SSM components)
- Causal conv1d reference: `causal-conv1d/` (for convolution layers)

## Weight Management
Multiple weight formats available in `weights/`:
- Original PyTorch weights (.pt)
- Converted pickle formats (.pkl)
- JAX checkpoint format (.ckpt)

## H-Net Architecture Breakdown

Based on analysis of the reference implementation (`./hnet-reference/`), H-Net's core innovation is **dynamic chunking** with hierarchical processing.

### Core Architecture

H-Net is **NOT** just a standard transformer or Mamba model. The key insight is:

**H-Net = Standard Mamba2 + Dynamic Chunking + Hierarchical Processing**

### Architecture Layout

The canonical H-Net architecture follows this pattern:
```json
{
  "arch_layout": ["m4", ["T22"], "m4"],
  "d_model": [1024, 1536],
  "d_intermediate": [0, 4096]
}
```

This means:
- **Encoder**: 4 Mamba2 layers (1024-dim) → `"m4"`
- **Main Network**: 22 Transformer layers (1536-dim) → `["T22"]`  
- **Decoder**: 4 Mamba2 layers (1024-dim) → `"m4"`

### Dynamic Chunking Pipeline

The revolutionary aspect of H-Net is its dynamic chunking mechanism:

```
Input → Encoder → RoutingModule → ChunkLayer → MainNetwork → DeChunkLayer → Decoder → Output
                      ↓              ↓                           ↑
               [Dynamic Boundaries] [Compressed Sequence]  [Reconstructed]
```

### Key Components

#### 1. RoutingModule (`routing_module`)
- **Purpose**: Detects semantic boundaries in the sequence
- **Method**: Computes cosine similarity between consecutive tokens
- **Formula**: `boundary_prob = (1 - cos_sim) / 2`
- **Innovation**: Identifies natural breakpoints (word boundaries, semantic shifts)
- **Compression**: Typically achieves 30-70% compression depending on input structure

#### 2. ChunkLayer (`chunk_layer`)
- **Purpose**: Extracts only boundary tokens to create compressed representation
- **Input**: Full sequence + boundary mask
- **Output**: Compressed sequence containing only semantic boundary tokens
- **Effect**: Allows main network to process at higher semantic granularity

#### 3. DeChunkLayer (`dechunk_layer`)
- **Purpose**: Reconstructs full sequence from compressed representation
- **Method**: Sophisticated EMA (Exponential Moving Average) algorithm via SSD kernel
- **Formula**: Uses `dt = log(1/(1-p))` where `p` is boundary probability
- **SSD Integration**: Leverages Mamba2's SSD kernel for efficient EMA computation
- **Innovation**: Learns to propagate boundary information across entire segments

### Hierarchical Residual Processing

H-Net uses a sophisticated residual connection with STE (Straight-Through Estimator):

```python
# Residual computed BEFORE chunking
residual = residual_proj(hidden_states.to(fp32))

# ... dynamic chunking pipeline ...

# Applied AFTER dechunking with STE modulation
output = output * ste_func(boundary_probs) + residual
```

### State Management for Generation

H-Net requires complex state management for autoregressive generation:

```python
@dataclass
class HNetState:
    encoder_state: Optional[IsotropicInferenceParams]
    routing_module_state: Optional[RoutingModuleState]  # tracks boundaries
    main_network_state: Optional[Union[HNetState, IsotropicInferenceParams]]
    dechunk_state: Optional[DeChunkState]  # EMA continuation
    decoder_state: Optional[IsotropicInferenceParams]
```

### Weight Structure Analysis

From `weights/Hnet_1stage_L.pt`, the key weight patterns:

#### Mamba2 Components (Encoder/Decoder):
- `dt_bias`: (32,) - timestep bias for 32 heads
- `A_log`: (32,) - state matrix (log space)  
- `D`: (32,) - skip connection weights
- `in_proj.weight`: (4384, 1024) - [z, x, B, C, dt] projection
- `conv1d.weight`: (2304, 1, 4) - causal convolution
- `norm.weight`: (2048,) - RMSNorm for gated output

#### Dynamic Chunking Components:
- `routing_module.q_proj_layer.weight`: (1024, 1024) - identity matrix
- `routing_module.k_proj_layer.weight`: (1024, 1024) - identity matrix  
- `residual_proj.weight`: (1024, 1024) - hierarchical residual

#### Transformer Components (Main Network):
- `mixer.Wqkv.weight`: (4608, 1536) - attention projection
- `mlp.fc1.weight`: (8192, 1536) - MLP expansion
- `mlp.fc2.weight`: (1536, 8192) - MLP contraction

### Why H-Net Works

1. **Adaptive Processing**: Dynamic chunking allows processing at multiple granularities
2. **Semantic Awareness**: Boundary detection captures linguistic structure  
3. **Efficiency**: Compression reduces computation in main network
4. **Hierarchical**: Multi-stage architecture captures different abstraction levels
5. **EMA Reconstruction**: Learned interpolation preserves semantic continuity

### Critical Implementation Details

1. **Boundary Detection**: Must use cosine similarity between consecutive normalized token representations
2. **EMA Algorithm**: Must use SSD kernel with `dt = log(1/(1-p))` formulation
3. **Residual Connections**: Must use STE function with fp32 residual computation
4. **State Management**: Must properly handle complex inference cache for generation
5. **Dimension Handling**: Must handle dimension changes between stages (1024 ↔ 1536)

### Common Implementation Pitfalls

❌ **Wrong**: Implementing H-Net as just "Mamba + Transformer layers"
✅ **Correct**: Implementing full dynamic chunking pipeline

❌ **Wrong**: Simple reconstruction from boundaries
✅ **Correct**: EMA-based reconstruction using SSD algorithm

❌ **Wrong**: Standard residual connections
✅ **Correct**: STE-modulated hierarchical residuals

## Development Workflow
1. Implement components in `src/hnet/modules/`
2. Create corresponding tests in `src/hnet/tests/`
3. Run `uv run pytest -v` to validate implementation
4. Cross-reference with PyTorch implementation for correctness
5. Use mathematical analysis when direct execution comparison isn't possible