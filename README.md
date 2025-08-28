# H-Net JAX

A JAX implementation of H-Net, a hierarchical sequence modeling architecture with dynamic chunking capabilities, originally developed by Carnegie Mellon University and Cartesia.

## Citation

This work is based on the H-Net model from CMU + Cartesia..

**Paper**: [Dynamic Chunking for End-to-End Hierarchical Sequence Modeling](https://arxiv.org/abs/2507.07955)

```bibtex
@article{hwang2025hnet,
  title={Dynamic Chunking for End-to-End Hierarchical Sequence Modeling},
  author={Hwang, Sukjun and Wang, Brandon and Gu, Albert},
  journal={arXiv preprint arXiv:2507.07955},
  year={2025}
}
```

## Overview

This JAX port aims to explore and extend H-Net's capabilities through several research directions:

### 1. Performance Optimization & Batching

Investigating opportunities to introduce clever batching mechanisms and solve performance bottlenecks inherent in this model architecture. The original PyTorch implementation has several areas where JAX's functional programming paradigm could enable more efficient computation.

### 2. Multimodal Capabilities & Audio Processing

Conducting experiments and ablations on multimodal capabilities, with a particular focus on native audio processing. This research aligns with my work at [Maple Inc.](https://maple.inc), where we continuously explore innovative techniques in audio modeling and multimodal AI systems.

### 3. Scaling Pretraining Techniques

Developing and implementing scalable pretraining techniques specifically tailored for this architecture, exploring how H-Net's hierarchical structure can be leveraged for more efficient large-scale training.

## Current Status

**⚠️ Work in Progress**: The JAX version is currently unoptimized and experimental. Key limitations include:

- **Dynamic Chunking Challenges**: The dynamic chunking mechanism is particularly unfriendly with JAX's JIT compiler, leading to compilation overhead and performance issues
- **Unoptimized Implementation**: Many operations haven't been optimized for JAX's vectorization capabilities yet
- **Active Development**: The codebase is rapidly evolving as we experiment with different approaches

## Contributing

I'm open to **contributions and ideas** in any of the research directions mentioned above! Whether you have:

- Ideas for optimizing JAX implementations
- Experience with multimodal architectures
- Insights into scaling techniques
- General feedback on the codebase

Please feel free to:

- Open an issue for discussion
- Submit a pull request
- Reach out with research collaboration ideas

## Getting Started

```bash
# Install dependencies
uv sync

# Run a simple example
python hnet_jax.py
```

## Project Structure

```txt
src/hnet/
├── model.py          # Core H-Net model implementation
├── modules/          # Model components (attention, MLP, SSM, etc.)
├── generation.py     # Text generation utilities
├── config.py         # Configuration management
├── cli.py           # Command-line interface
└── ...
```

## Acknowledgments

This work builds upon the foundational research from CMU's Goomba Lab. Special thanks to the original authors for their innovative approach to hierarchical sequence modeling.
