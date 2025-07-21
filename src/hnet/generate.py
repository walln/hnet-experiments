import argparse
import json
import sys
import pickle

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import ListConfig

from hnet.models.config_hnet import (
    AttnConfig,
    HNetConfig,
    SSMConfig,
)
from hnet.models.mixer_seq import HNetForCausalLM


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255
        self.dtype = np.uint8

    def encode(self, seqs, add_bos=False, add_eos=False, **kwargs):
        total_outputs = []
        for text in seqs:
            text_byte = text.encode("utf-8")

            if add_bos:
                text_byte = bytes([self.bos_idx]) + text_byte
            if add_eos:
                text_byte = text_byte + bytes([self.eos_idx])
            text_byte = bytearray(text_byte)
            text_byte_ids = np.array(text_byte, dtype=self.dtype)

            total_outputs.append({"input_ids": text_byte_ids})

        return total_outputs

    def decode(self, tokens, **kwargs):
        if isinstance(tokens, (np.ndarray, jnp.ndarray)):
            tokens = np.array(tokens).tolist()
        return bytearray(tokens).decode("utf-8", **kwargs)


def convert_pytorch_to_jax(pytorch_state_dict):
    """Convert PyTorch state dict to JAX/Flax format with proper mappings."""
    jax_state_dict = {}
    print(f"Converting {len(pytorch_state_dict)} parameters from PyTorch to JAX...")

    conversion_count = 0
    skipped_count = 0

    for pt_key, pt_value in pytorch_state_dict.items():
        # Convert tensor to numpy array
        if hasattr(pt_value, "numpy"):
            pt_value = pt_value.numpy()
        elif not isinstance(pt_value, np.ndarray):
            continue

        # Create JAX key with proper mapping
        jax_key = f"_mapping.{pt_key}"

        # First, handle Mamba layers that need special prefix
        if (
            "mixer." in pt_key
            and any(
                x in pt_key
                for x in [
                    "in_proj",
                    "out_proj",
                    "conv1d",
                    "A_log",
                    "D",
                    "dt_bias",
                    "norm",
                ]
            )
            and "mamba" not in pt_key
            and ("encoder.layers" in pt_key or "decoder.layers" in pt_key)
        ):
            # This is a Mamba mixer parameter that needs the .mamba prefix
            parts = jax_key.split(".")
            mixer_idx = parts.index("mixer")
            # Insert "mamba" after "mixer"
            parts.insert(mixer_idx + 1, "mamba")
            jax_key = ".".join(parts)

        # Handle specific parameter conversions
        if pt_key.endswith(".weight"):
            # Linear/Dense layers: weight -> kernel with transpose
            if any(
                x in pt_key
                for x in [
                    "fc1",
                    "fc2",
                    "out_proj",
                    "in_proj",
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "Wqkv",
                    "lm_head",
                    "residual_proj",
                ]
            ):
                jax_key = jax_key.replace(".weight", ".kernel")
                # Transpose: PyTorch uses (out, in), Flax uses (in, out)
                if pt_value.ndim == 2:
                    pt_value = pt_value.T
            # RMSNorm layers in Flax: weight -> scale
            elif (
                "norm1.weight" in pt_key
                or "norm2.weight" in pt_key
                or "rmsnorm.weight" in pt_key
            ):
                jax_key = jax_key.replace(".weight", ".scale")
            # Other norm layers keep as weight (like Mamba's internal norm)
            elif "norm" in pt_key:
                pass  # Keep as .weight
            # Embedding layers: weight -> embedding
            elif "embeddings" in pt_key:
                jax_key = jax_key.replace(".weight", ".embedding")
            # Conv1d layers: need special handling
            elif "conv1d" in pt_key:
                # Map conv1d.weight to conv_weight
                jax_key = jax_key.replace("conv1d.weight", "conv_weight")

        # Handle bias mappings
        elif pt_key.endswith(".bias"):
            if "conv1d" in pt_key:
                jax_key = jax_key.replace("conv1d.bias", "conv_bias")

        # Don't skip MLP weights - we'll handle shape mismatches during loading

        # Store in JAX format
        jax_state_dict[jax_key] = jnp.array(pt_value)
        conversion_count += 1

    print(
        f"Converted {conversion_count} parameters successfully, skipped {skipped_count}"
    )
    return jax_state_dict


def load_from_pretrained(model_path: str, model_config_path: str):
    """Load model from pretrained checkpoint with full weight loading."""
    # Load configuration
    with open(model_config_path) as f:
        config = json.load(f)

    # Create config objects
    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config.pop("ssm_cfg"))
    hnet_cfg = HNetConfig(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)

    # Create model
    rngs = nnx.Rngs(0)
    model = HNetForCausalLM(hnet_cfg, rngs=rngs)

    # Load checkpoint - handle both PyTorch and JAX formats
    if model_path.endswith(".pt"):
        # Load PyTorch checkpoint
        try:
            import torch

            with torch.serialization.safe_globals([ListConfig]):
                state_dict = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )
            # Convert to JAX format
            state_dict = convert_pytorch_to_jax(state_dict)
        except ImportError:
            raise ImportError(
                "PyTorch is required to load .pt files. Install it or convert your weights to JAX format."
            )
    else:
        # Load JAX checkpoint
        with open(model_path, "rb") as f:
            state_dict = pickle.load(f)

    # Update model state with loaded weights
    print("Updating model state with loaded weights...")

    # Split model to get state
    graphdef, model_state = nnx.split(model)

    # Get the current state as a flat dictionary
    def flatten_state(state, prefix=""):
        """Flatten the state to a dictionary with dot-separated keys."""
        flat = {}

        def traverse(obj, path):
            if isinstance(obj, nnx.VariableState):
                if hasattr(obj, "value"):
                    flat[path] = obj
            elif hasattr(obj, "__dict__"):
                for k, v in obj.__dict__.items():
                    new_path = f"{path}.{k}" if path else k
                    traverse(v, new_path)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    new_path = f"{path}.{k}" if path else k
                    traverse(v, new_path)

        traverse(state, prefix)
        return flat

    # Flatten the current state
    flat_state = flatten_state(model_state)

    # Helper function to handle shape mismatches
    def handle_shape_mismatch(key, current_shape, value):
        """Try to handle shape mismatches for specific parameter types."""
        # With the fix to use exact d_intermediate, we shouldn't need truncation anymore
        # Just return False to report any mismatches
        return value, False

    # Update parameters
    updated_count = 0
    mismatch_count = 0
    missing_count = 0

    for key, value in state_dict.items():
        if key in flat_state:
            current_param = flat_state[key]
            if hasattr(current_param, "value"):
                current_shape = current_param.value.shape
                if current_shape == value.shape:
                    # Update the parameter value
                    current_param.value = value
                    updated_count += 1
                else:
                    # Try to handle shape mismatch
                    adapted_value, handled = handle_shape_mismatch(
                        key, current_shape, value
                    )
                    if handled:
                        current_param.value = adapted_value
                        updated_count += 1
                    else:
                        print(
                            f"Shape mismatch for {key}: expected {current_shape}, got {value.shape}"
                        )
                        mismatch_count += 1
        else:
            # Try without _mapping prefix if present
            if key.startswith("_mapping."):
                alt_key = key[9:]  # Remove "_mapping." prefix
                if alt_key in flat_state:
                    current_param = flat_state[alt_key]
                    if hasattr(current_param, "value"):
                        current_shape = current_param.value.shape
                        if current_shape == value.shape:
                            current_param.value = value
                            updated_count += 1
                        else:
                            # Try to handle shape mismatch
                            adapted_value, handled = handle_shape_mismatch(
                                alt_key, current_shape, value
                            )
                            if handled:
                                current_param.value = adapted_value
                                updated_count += 1
                            else:
                                print(
                                    f"Shape mismatch for {alt_key}: expected {current_shape}, got {value.shape}"
                                )
                                mismatch_count += 1
                else:
                    missing_count += 1
            else:
                missing_count += 1

    print(f"\nWeight loading summary:")
    print(f"  Successfully updated: {updated_count} parameters")
    print(f"  Shape mismatches: {mismatch_count} parameters")
    print(f"  Not found in model: {missing_count} parameters")

    # Merge state back into model
    model = nnx.merge(graphdef, model_state)

    return model


def generate(
    model,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 0.9,
    key=None,
):
    """Generate text from the model, yielding tokens as they're generated.

    Args:
        model: HNetForCausalLM model
        prompt: Input text prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Top-p sampling parameter
        key: JAX PRNG key for sampling

    Yields:
        Generated text token by token as strings
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    tokenizer = ByteTokenizer()

    # Tokenize prompt
    encoded = tokenizer.encode([prompt], add_bos=True)[0]
    input_ids = jnp.array(encoded["input_ids"], dtype=jnp.int32).reshape(1, -1)

    inference_cache = model.allocate_inference_cache(
        1, input_ids.shape[1] + max_tokens, dtype=jnp.bfloat16
    )

    # Initial forward pass
    mask = jnp.ones(input_ids.shape, dtype=jnp.bool_)
    output = model.forward(input_ids, mask=mask, inference_params=inference_cache)

    logits = output.logits[0, -1, :] / temperature

    for i in range(max_tokens):
        key, subkey = jax.random.split(key)

        # Apply top-p sampling
        if top_p < 1.0:
            sorted_indices = jnp.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove = jnp.concatenate(
                [jnp.array([False]), sorted_indices_to_remove[:-1]]
            )

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits = logits.at[indices_to_remove].set(-float("inf"))

        # Sample next token
        next_token = jax.random.categorical(subkey, logits)

        if next_token.item() == tokenizer.eos_idx:
            break

        current_token = next_token.reshape(1, 1)
        yield current_token

        # Step the model
        output = model.step(current_token, inference_cache)

        # Get logits and apply temperature
        logits = output.logits[0, -1, :] / temperature


def main():
    parser = argparse.ArgumentParser(description="Generate text from an H-Net model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt or JAX file)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the model configuration (.json file)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (default: 1024)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter (default: 1.0)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt to generate from (default: empty)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation",
    )
    args = parser.parse_args()

    print("Loading model...")
    try:
        model = load_from_pretrained(args.model_path, args.config_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    tokenizer = ByteTokenizer()
    key = jax.random.PRNGKey(args.seed if args.seed is not None else 0)

    # If a prompt is provided via command line, generate once and exit
    if args.prompt:
        prompt = args.prompt
    else:
        # Interactive mode
        while True:
            prompt = input("\nPrompt: ").strip()
            if not prompt:
                continue
            break

    if prompt:
        print(
            f"\nGenerating (max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p})"
        )

        print(f"\033[92m{prompt}\033[0m", end="")
        token_count = 0
        buf = []

        for token in generate(
            model,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            key=key,
        ):
            buf.append(token)
            token_count += 1

            decoded = None
            res = None
            for j in range(1, min(len(buf), 4)):
                try:
                    # Convert to numpy for decoding
                    token_array = jnp.concatenate([b.flatten() for b in buf[:j]])
                    res = tokenizer.decode(token_array)
                    decoded = j
                except:
                    pass

            if res is not None:
                print(res, end="", flush=True)
                buf = buf[decoded:]


if __name__ == "__main__":
    main()
