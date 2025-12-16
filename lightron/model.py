import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelArgs
from .layers import RMSNorm, apply_rotary_emb, precompute_freqs_cis


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU: (xW1 * SiLU(xW3)) * W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cis):
        B, S, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(B, S, self.n_heads, self.head_dim)
        xk = xk.view(B, S, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, S, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # GQA repeat if needed (omitted for brevity, assume n_heads == n_kv_heads for now)

        # Flash Attention
        # is_causal=True 自动处理 mask
        output = F.scaled_dot_product_attention(
            xq.transpose(1, 2),
            xk.transpose(1, 2),
            xv.transpose(1, 2),
            is_causal=True
        )

        output = output.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(output)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LightronTransformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)

    def forward(self, tokens):
        B, S = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:S].to(h.device)

        for layer in self.layers:
            h = layer(h, freqs_cis)

        h = self.norm(h)
        return self.output(h)
