"""Token generation utilities with optional JIT and profiling."""

from __future__ import annotations

import time
from collections.abc import Generator

import jax
import jax.numpy as jnp

from .lm import HNetForCausalLM
from .tokenizer import ByteTokenizer


def _top_p_filtering(logits: jnp.ndarray, top_p: float) -> jnp.ndarray:
    if top_p >= 1.0:
        return logits
    sorted_logits = jnp.sort(logits, axis=-1)[::-1]
    sorted_indices = jnp.argsort(logits, axis=-1)[::-1]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove = jnp.concatenate(
        [jnp.array([False]), sorted_indices_to_remove[:-1]]
    )
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits = logits.at[indices_to_remove].set(-jnp.inf)
    return logits


def generate_tokens(
    model: HNetForCausalLM,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 1.0,
    rng_key: jax.Array | None = None,
    use_jit: bool = False,
    profile: bool = False,
) -> Generator[int, None, None]:
    """Stream token ids generated from the model for a text prompt."""

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    tok = ByteTokenizer()
    x = tok.encode(prompt, add_bos=True)[None, :]
    cache = model.allocate_inference_cache(
        1, x.shape[1] + max_new_tokens, dtype=model.lm_head.kernel.value.dtype
    )

    mask = jnp.ones_like(x, dtype=jnp.bool_)

    if use_jit:
        try:

            @jax.jit
            def jit_forward_stateless(input_ids, mask):
                return model(input_ids, mask=mask, inference_params=None)

            logits, _, _ = jit_forward_stateless(x, mask)
            print("[JIT] ", end="")
        except Exception as e:
            print(f"[JIT failed: {e}] ", end="")
            logits, _, _ = model(x, mask=mask, inference_params=cache)
    else:
        logits, _, _ = model(x, mask=mask, inference_params=cache)

    logits = logits[0, -1]

    tokens_generated = 0
    gen_start = time.perf_counter() if profile else None

    for i in range(int(max_new_tokens)):
        step_start = time.perf_counter() if profile else None

        rng_key, sample_key = jax.random.split(rng_key)
        logits = logits / max(temperature, 1e-6)
        logits = _top_p_filtering(logits, top_p)
        next_id = jax.random.categorical(sample_key, logits, shape=(1,))

        token_id = int(next_id.item())
        if token_id == ByteTokenizer().eos_idx:
            break

        tokens_generated += 1
        yield token_id

        nxt = next_id[None, :]
        logits, _, _ = model.step(nxt, cache)
        logits = logits[0, -1]

        if profile and step_start is not None and i < 5:
            step_time = time.time() - step_start
            print(f" [Step {i}: {step_time:.3f}s]", end="")

    if profile and tokens_generated > 0:
        assert gen_start is not None, (
            "gen_start should not be None when profiling is enabled"
        )
        total_gen_time = time.time() - float(gen_start)
        tokens_per_sec = (
            tokens_generated / total_gen_time if total_gen_time > 0 else 0.0
        )
        print(f" [Generation: {tokens_per_sec:.1f} tok/s]", end="")
