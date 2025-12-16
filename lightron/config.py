from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None  # For GQA
    vocab_size: int = 32000
    multiple_of: int = 256  # MLP hidden dim multiple
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads