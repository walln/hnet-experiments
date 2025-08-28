"""Simple UTF-8 byte tokenizer with BOS/EOS handling."""

from __future__ import annotations

import jax.numpy as jnp


class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256
        self.bos_idx = 254
        self.eos_idx = 255

    def encode(
        self, text: str, add_bos: bool = False, add_eos: bool = False
    ) -> jnp.ndarray:
        b = text.encode("utf-8")
        if add_bos:
            b = bytes([self.bos_idx]) + b
        if add_eos:
            b = b + bytes([self.eos_idx])
        return jnp.array(list(b), dtype=jnp.int32)

    def decode(self, ids: jnp.ndarray) -> str:
        if ids.ndim > 1:
            ids = ids.flatten()
        arr = ids.astype(int).tolist()
        arr = [t for t in arr if t not in (self.bos_idx, self.eos_idx)]
        try:
            return bytearray(arr).decode("utf-8")
        except Exception:
            try:
                return bytearray(arr).decode("utf-8", errors="replace")
            except Exception:
                result = []
                for tok in arr:
                    if 32 <= tok <= 126:
                        result.append(chr(tok))
                    else:
                        result.append(f"\\x{tok:02x}")
                return "".join(result)
