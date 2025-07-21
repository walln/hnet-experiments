"""
Unified Modal deployment for H-Net JAX and PyTorch model comparison.

This script automatically downloads model weights from Hugging Face if they don't
exist in the persistent volume, using the efficient hf_transfer client.

Run with --jax, --torch, or both (default).

Usage:
    modal run modal_hnet_comparison.py --jax
    modal run modal_hnet_comparison.py --model-repo cartesia-ai/hnet_2stage_XL

Note: If the default model repository doesn't contain .pt files, you may need to:
1. Check the repository structure on Hugging Face directly
2. Use a different repository that contains the model weights
3. Manually upload model files to the persistent volume
"""

import argparse
import json
from pathlib import Path

import modal

# Create Modal app
app = modal.App("hnet-comparison-unified")

# Create volume for model weights
weights_volume = modal.Volume.from_name("hnet-model-weights", create_if_missing=True)
output_volume = modal.Volume.from_name("hnet-outputs", create_if_missing=True)
MODEL_DIR = Path("/models")

# Define dependencies for downloading model
download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub[hf_transfer]")  # install fast Rust download client
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # and enable it
)

# Define the container image with both JAX and PyTorch
# Using new API pattern - add source code directly to image
unified_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "setuptools",
        "ninja",
        "psutil",
    )
    .uv_pip_install(
        "torch==2.9.0.dev20250715+cu126",
        index_url="https://download.pytorch.org/whl/nightly/cu126",
    )
    .uv_pip_install(
        # JAX dependencies
        "jax[cuda12]",
        "flax",
        # PyTorch dependencies
        "torch>=2.7.1",
        # Shared dependencies
        "einops>=0.8.1",
        "numpy",
        "omegaconf>=2.3.0",
    )
    .add_local_dir("src", "/app/src")  # Add source code to image
)


@app.function(
    volumes={
        MODEL_DIR.as_posix(): weights_volume
    },  # "mount" the Volume, sharing it with your function
    image=download_image,  # only download dependencies needed here
)
def download_model(
    repo_id: str = "cartesia-ai/hnet_1stage_L",
    revision: str | None = None,  # include a revision to prevent surprises!
):
    """Download model weights from Hugging Face if they don't exist."""
    from huggingface_hub import (
        snapshot_download,
    )  # Note: this import will show linter errors but works in Modal

    model_path = MODEL_DIR / repo_id.split("/")[-1]  # e.g., /models/hnet_1stage_L

    # Check if model already exists
    if model_path.exists() and any(model_path.iterdir()):
        print(f"Model already exists at {model_path}")
        return str(model_path)

    print(f"Downloading model {repo_id} to {model_path}")
    snapshot_download(repo_id=repo_id, local_dir=model_path, revision=revision)
    print(f"Model downloaded to {model_path}")
    return str(model_path)


@app.function(
    image=unified_image,
    gpu="A10G",
    volumes={
        MODEL_DIR.as_posix(): weights_volume,
        "/outputs": output_volume,
    },
    timeout=600,
)
def run_jax_model(
    config_dict, test_prompts, repo_id: str = "cartesia-ai/hnet_1stage_L"
):
    """Run JAX model inference."""
    import sys

    sys.path.insert(0, "/app/src")  # Add source directory to Python path

    import jax
    import jax.numpy as jnp

    from hnet.generate import ByteTokenizer, load_from_pretrained

    print("=== JAX Model ===")
    print(f"JAX devices: {jax.devices()}")
    device_kind = jax.devices()[0].device_kind
    print(f"Device kind: {device_kind}")
    print(f"GPU/CUDA available: {device_kind in ['gpu', 'cuda']}")

    # Ensure model weights are available
    weights_volume.reload()  # Refresh volume to see any new downloads

    # Construct model path based on repo_id
    model_name = repo_id.split("/")[-1]  # e.g., "hnet_1stage_L"
    model_dir = MODEL_DIR / model_name

    # Look for various model file formats
    model_files = []
    for pattern in ["*.pt", "*.pth", "*.safetensors", "*.bin", "*.ckpt"]:
        model_files.extend(model_dir.glob(pattern))

    # Also check in subdirectories
    if not model_files:
        for subdir in model_dir.iterdir():
            if subdir.is_dir():
                for pattern in ["*.pt", "*.pth", "*.safetensors", "*.bin", "*.ckpt"]:
                    model_files.extend(subdir.glob(pattern))

        print(f"Model directory contents: {list(model_dir.iterdir())}")

    if not model_files:
        # List all files recursively for debugging
        all_files = list(model_dir.rglob("*"))
        print(f"All files in {model_dir} (recursive):")
        for f in all_files:
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f.relative_to(model_dir)} ({size_mb:.2f} MB)")

        # Check if there might be a different file we can use
        large_files = [
            f for f in all_files if f.is_file() and f.stat().st_size > 10 * 1024 * 1024
        ]  # > 10MB
        if large_files:
            print(f"Large files that might be models: {[f.name for f in large_files]}")
            # For now, raise error but suggest checking the repository
            raise FileNotFoundError(
                f"No standard model files (.pt, .pth, .safetensors, .bin, .ckpt) found in {model_dir}.\n"
                f"Found {len(large_files)} large files that might be models: {[f.name for f in large_files]}\n"
                f"The repository structure might be different than expected. Please check the Hugging Face repository directly."
            )
        else:
            raise FileNotFoundError(
                f"No model files found in {model_dir}.\n"
                f"The repository might be empty or use Git LFS. Please check the Hugging Face repository directly.\n"
                f"Available files: {[f.name for f in all_files if f.is_file()]}"
            )

    # Prefer .pt files, then .pth, then others
    pt_files = [f for f in model_files if f.suffix in [".pt", ".pth"]]
    model_path = pt_files[0] if pt_files else model_files[0]

    print(f"Using model weights: {model_path}")

    # Save config to temp file for loading
    config_path = "/tmp/config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f)

    # Load model
    print("Loading JAX model...")
    model = load_from_pretrained(str(model_path), config_path)
    tokenizer = ByteTokenizer()

    results = {}

    for prompt in test_prompts:
        print(f"\nProcessing: '{prompt}'")

        # Tokenize
        encoded = tokenizer.encode([prompt], add_bos=True)[0]
        input_ids = jnp.array(encoded["input_ids"], dtype=jnp.int32).reshape(1, -1)

        # Store input info
        prompt_results = {
            "prompt": prompt,
            "input_ids": input_ids.tolist(),
            "input_shape": list(input_ids.shape),
        }

        # Forward pass
        mask = jnp.ones(input_ids.shape, dtype=jnp.bool_)

        # Get embeddings
        embeddings = model.embeddings(input_ids)
        prompt_results["embeddings_stats"] = {
            "shape": list(embeddings.shape),
            "mean": float(jnp.mean(embeddings)),
            "std": float(jnp.std(embeddings)),
            "min": float(jnp.min(embeddings)),
            "max": float(jnp.max(embeddings)),
            "first_10": embeddings.flatten()[:10].tolist(),
        }

        # Full forward pass
        output = model.forward(input_ids, mask=mask, inference_params=None)
        logits = output.logits[0, -1, :]

        # Store logits info
        prompt_results["logits_stats"] = {
            "shape": list(logits.shape),
            "mean": float(jnp.mean(logits)),
            "std": float(jnp.std(logits)),
            "min": float(jnp.min(logits)),
            "max": float(jnp.max(logits)),
        }

        # Store full logits
        prompt_results["logits"] = logits.tolist()

        # Get top predictions
        top_k = 10
        top_indices = jnp.argsort(logits)[-top_k:][::-1]
        probs = jax.nn.softmax(logits)
        top_probs = probs[top_indices]

        top_predictions = []
        for idx, prob in zip(top_indices, top_probs, strict=False):
            char = chr(int(idx)) if 32 <= idx < 127 else f"[{int(idx)}]"
            top_predictions.append(
                {
                    "index": int(idx),
                    "char": char,
                    "logit": float(logits[idx]),
                    "prob": float(prob),
                }
            )

        prompt_results["top_predictions"] = top_predictions
        results[prompt] = prompt_results

    # Save results
    output_path = "/outputs/jax_model_outputs.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    output_volume.commit()

    # Print summary
    print("\n=== JAX Summary ===")
    for prompt, data in results.items():
        print(f"\nPrompt: '{prompt}'")
        print("Top 5 predictions:")
        for pred in data["top_predictions"][:5]:
            print(
                f"  {pred['char']} (idx={pred['index']}): logit={pred['logit']:.4f}, prob={pred['prob']:.4f}"
            )

    return results


@app.function(
    image=unified_image,
    gpu="A10G",
    volumes={
        MODEL_DIR.as_posix(): weights_volume,
        "/outputs": output_volume,
    },
    timeout=600,
)
def run_pytorch_model(
    config_dict, test_prompts, repo_id: str = "cartesia-ai/hnet_1stage_L"
):
    """Run PyTorch model inference."""
    import torch

    print("=== PyTorch Model ===")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch version: {torch.__version__}")

    # TODO: Add your PyTorch implementation here
    # This is a placeholder that returns dummy results

    print("\n⚠️  PyTorch implementation not yet added")
    print("Please implement the PyTorch model loading and inference")

    # Placeholder results structure matching JAX format
    results = {}
    for prompt in test_prompts:
        results[prompt] = {
            "prompt": prompt,
            "status": "not_implemented",
            "message": "Add PyTorch implementation in run_pytorch_model()",
        }

    # When you implement PyTorch, save results same as JAX:
    output_path = "/outputs/pytorch_model_outputs.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    output_volume.commit()

    return results


@app.function(
    image=unified_image,
    volumes={"/outputs": output_volume},
)
def compare_results():
    """Compare JAX and PyTorch results if both exist."""
    import numpy as np

    output_volume.reload()

    # Load results
    try:
        with open("/outputs/jax_model_outputs.json") as f:
            jax_results = json.load(f)
    except FileNotFoundError:
        print("JAX results not found")
        return None

    try:
        with open("/outputs/pytorch_model_outputs.json") as f:
            pytorch_results = json.load(f)
    except FileNotFoundError:
        print("PyTorch results not found")
        return None

    # Check if PyTorch is implemented
    first_prompt = next(iter(pytorch_results.keys()))
    if pytorch_results[first_prompt].get("status") == "not_implemented":
        print("PyTorch implementation not yet added")
        return None

    print("\n=== Comparing Results ===")

    comparison = {}

    for prompt in jax_results:
        if prompt not in pytorch_results:
            continue

        print(f"\nPrompt: '{prompt}'")

        jax_data = jax_results[prompt]
        pytorch_data = pytorch_results[prompt]

        # Compare top predictions
        jax_preds = jax_data["top_predictions"][:5]
        pytorch_preds = pytorch_data["top_predictions"][:5]

        matches = sum(
            1
            for j, p in zip(jax_preds, pytorch_preds, strict=False)
            if j["index"] == p["index"]
        )
        print(f"Top-5 prediction matches: {matches}/5")

        # Compare logits if available
        if "logits" in jax_data and "logits" in pytorch_data:
            jax_logits = np.array(jax_data["logits"])
            pytorch_logits = np.array(pytorch_data["logits"])

            # Cosine similarity
            cos_sim = np.dot(jax_logits, pytorch_logits) / (
                np.linalg.norm(jax_logits) * np.linalg.norm(pytorch_logits)
            )

            # MSE
            mse = np.mean((jax_logits - pytorch_logits) ** 2)

            print(f"Cosine similarity: {cos_sim:.6f}")
            print(f"MSE: {mse:.6f}")

            comparison[prompt] = {
                "top5_matches": matches,
                "cosine_similarity": float(cos_sim),
                "mse": float(mse),
            }

    # Save comparison
    with open("/outputs/comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)

    output_volume.commit()

    return comparison


@app.function(volumes={"/outputs": output_volume})
def get_results(result_type="all"):
    """Download results from Modal volumes."""
    output_volume.reload()

    results = {}

    if result_type in ["all", "jax"]:
        try:
            with open("/outputs/jax_model_outputs.json") as f:
                results["jax"] = json.load(f)
        except FileNotFoundError:
            results["jax"] = None

    if result_type in ["all", "pytorch"]:
        try:
            with open("/outputs/pytorch_model_outputs.json") as f:
                results["pytorch"] = json.load(f)
        except FileNotFoundError:
            results["pytorch"] = None

    if result_type in ["all", "comparison"]:
        try:
            with open("/outputs/comparison_results.json") as f:
                results["comparison"] = json.load(f)
        except FileNotFoundError:
            results["comparison"] = None

    return results


@app.local_entrypoint()
def main(*arglist):
    """Main entrypoint with CLI."""
    parser = argparse.ArgumentParser(
        description="Run H-Net model comparison on Modal GPUs"
    )
    parser.add_argument("--jax", action="store_true", help="Run JAX model only")
    parser.add_argument("--torch", action="store_true", help="Run PyTorch model only")
    parser.add_argument(
        "--compare", action="store_true", help="Compare existing results"
    )
    parser.add_argument(
        "--download", action="store_true", help="Download results to local files"
    )
    parser.add_argument(
        "--config",
        default="configs/hnet_1stage_L.json",
        help="Path to config file (default: configs/hnet_1stage_L.json)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["Hello", "The", "A"],
        help="Test prompts (default: Hello The A)",
    )
    parser.add_argument(
        "--model-repo",
        default="cartesia-ai/hnet_1stage_L",
        help="Hugging Face model repository (default: cartesia-ai/hnet_1stage_L). "
        "Note: Some repositories may not contain .pt files directly.",
    )

    args = parser.parse_args(args=arglist)

    # If no specific model selected, run both
    run_jax = args.jax or (not args.torch and not args.compare)
    run_torch = args.torch or (not args.jax and not args.compare)

    # Load config file locally
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return

    with open(config_path) as f:
        config_dict = json.load(f)

    print(f"Using config: {config_path}")
    print(f"Test prompts: {args.prompts}")

    # Ensure model weights are downloaded first
    print("\n" + "=" * 60)
    print("Ensuring model weights are available...")
    print("=" * 60)
    model_path = download_model.remote(repo_id=args.model_repo)
    print(f"Model weights ready at: {model_path}")

    # Run models
    if run_jax:
        print("\n" + "=" * 60)
        print("Running JAX model on Modal GPU...")
        print("=" * 60)
        _jax_results = run_jax_model.remote(config_dict, args.prompts, args.model_repo)

    if run_torch:
        print("\n" + "=" * 60)
        print("Running PyTorch model on Modal GPU...")
        print("=" * 60)
        _pytorch_results = run_pytorch_model.remote(
            config_dict, args.prompts, args.model_repo
        )

    # Compare if requested or if both models were run
    if args.compare or (run_jax and run_torch):
        print("\n" + "=" * 60)
        print("Comparing results...")
        print("=" * 60)
        _comparison = compare_results.remote()

    # Download results if requested
    if args.download:
        print("\n" + "=" * 60)
        print("Downloading results...")
        print("=" * 60)

        results = get_results.remote()

        if results["jax"]:
            with open("modal_jax_outputs.json", "w") as f:
                json.dump(results["jax"], f, indent=2)
            print("✓ Downloaded JAX results to modal_jax_outputs.json")

        if results["pytorch"]:
            with open("modal_pytorch_outputs.json", "w") as f:
                json.dump(results["pytorch"], f, indent=2)
            print("✓ Downloaded PyTorch results to modal_pytorch_outputs.json")

        if results["comparison"]:
            with open("modal_comparison_results.json", "w") as f:
                json.dump(results["comparison"], f, indent=2)
            print("✓ Downloaded comparison to modal_comparison_results.json")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
