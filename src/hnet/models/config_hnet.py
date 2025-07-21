from dataclasses import dataclass, field


@dataclass
class AttnConfig:
    num_heads: list[int] = field(default_factory=list)
    rotary_emb_dim: list[int] = field(default_factory=list)
    window_size: list[int] = field(default_factory=list)


@dataclass
class SSMConfig:
    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256


@dataclass
class HNetConfig:
    arch_layout: list[str | list] = field(default_factory=list)
    d_model: list[int] = field(default_factory=list)
    # intermediate dimension for the FFNs (0 indicates no FFN)
    d_intermediate: list[int] = field(default_factory=list)
    vocab_size: int = 256
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    tie_embeddings: bool = False
