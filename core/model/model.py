import torch
import torch.nn as nn
import torch.nn.functional as F
from core.config.config import ModelArgs
from core.layers import RMSNorm, apply_rotary_emb, precompute_freqs_cis


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 KV 头的数量复制 n_rep 倍，以匹配 Query 的头数。

    Args:
        x: 输入张量，形状为 (Batch, SeqLen, n_kv_heads, HeadDim)
        n_rep: 复制倍数

    Returns:
        输出张量，形状为 (Batch, SeqLen, n_kv_heads * n_rep, HeadDim)
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    # 核心逻辑：
    # 1. 增加一个维度: (B, S, n_kv, 1, D)
    # 2. 在新维度复制: (B, S, n_kv, n_rep, D)
    # 3. 展平维度: (B, S, n_kv * n_rep, D)
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


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
        # 如果配置中没写 n_kv_heads，默认等于 n_heads (即退化为标准 MHA)
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.head_dim = args.dim // args.n_heads

        # 计算复制倍数，例如 32 / 8 = 4
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cis):
        B, S, _ = x.shape

        # 1. 投影
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. 拆分 Head 维度
        # xq: [B, S, n_heads, D]
        # xk, xv: [B, S, n_kv_heads, D]
        xq = xq.view(B, S, self.n_heads, self.head_dim)
        xk = xk.view(B, S, self.n_kv_heads, self.head_dim)
        xv = xv.view(B, S, self.n_kv_heads, self.head_dim)

        # 3. Apply RoPE (旋转位置编码)
        # 注意：RoPE 是在 Attention 计算之前做的，且要在 repeat_kv 之前做
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # 4. GQA 核心步骤：如果 KV 头数少，就复制
        # 变换后 xk, xv 的形状将变为 [B, S, n_heads, D]
        if self.n_rep > 1:
            xk = repeat_kv(xk, self.n_rep)
            xv = repeat_kv(xv, self.n_rep)

        # 5. Flash Attention
        # 此时 xq, xk, xv 的维度完全对齐了
        # 需要转置为 [B, n_heads, S, D] 以符合 PyTorch API 要求
        output = F.scaled_dot_product_attention(
            xq.transpose(1, 2),
            xk.transpose(1, 2),
            xv.transpose(1, 2),
            is_causal=True
        )

        # 6. 还原形状并输出
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
        # 注意：这里只计算一次，并在 forward 中根据当前 seq_len 切片
        self.freqs_cis = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)

    def forward(self, tokens):
        B, S = tokens.shape
        h = self.tok_embeddings(tokens)

        # 确保 freqs_cis 在正确的设备上
        freqs_cis = self.freqs_cis[:S].to(h.device)

        for layer in self.layers:
            h = layer(h, freqs_cis)

        h = self.norm(h)
        return self.output(h)
