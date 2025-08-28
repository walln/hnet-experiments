"""CLI mirroring the original single-file interface for generation."""

from __future__ import annotations

import argparse

import jax

from .generation import generate_tokens
from .loading import load_model
from .tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser(description="H-Net JAX (Flax NNX) generator")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to config JSON (e.g., hnet-reference/configs/hnet_2stage_L.json)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to .pt weights (optional; non-strict load)",
    )
    parser.add_argument("--prompt", type=str, required=True, help="UTF-8 prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"]
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict state_dict load (fail on any mismatch)",
    )
    parser.add_argument(
        "--jit", action="store_true", help="Use JIT compilation (stateless mode)"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Enable performance profiling"
    )
    args = parser.parse_args()

    rng_key = jax.random.PRNGKey(args.seed)
    model = load_model(
        args.model_path, args.config_path, dtype=args.dtype, strict=args.strict
    )
    tok = ByteTokenizer()
    print(args.prompt, end="", flush=True)
    buf: list[int] = []
    for tid in generate_tokens(
        model,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        rng_key=rng_key,
        use_jit=args.jit,
        profile=args.profile,
    ):
        buf.append(tid)
        for j in range(1, min(len(buf), 4) + 1):
            try:
                s = tok.decode(jax.numpy.array(buf[:j]))
                print(s, end="", flush=True)
                buf = buf[j:]
                break
            except Exception:
                pass


if __name__ == "__main__":
    main()
