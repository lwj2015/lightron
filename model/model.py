import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import ModelArgs
from layers.layers import RMSNorm, apply_rotary_emb, precompute_freqs_cis
from parallel.parallel_tp import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from parallel.parallel_cp import ContextParallelAttention
from parallel.parallel_ep import ExpertParallel


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


def get_linear_cls(args: ModelArgs, parallel_type: str = None):
    """
    根据配置返回合适的 Linear 类
    parallel_type: 'col' (列并行), 'row' (行并行), None (普通)
    """
    if args.tp_size > 1:
        if parallel_type == 'col':
            return ColumnParallelLinear
        elif parallel_type == 'row':
            return RowParallelLinear
    return nn.Linear


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        ColLinear = get_linear_cls(args, 'col')
        RowLinear = get_linear_cls(args, 'row')

        # Llama 结构: w1(Gate), w3(Up), w2(Down)
        self.w1 = ColLinear(args.dim, hidden_dim, bias=False)  # Gate
        self.w3 = ColLinear(args.dim, hidden_dim, bias=False)  # Up
        self.w2 = RowLinear(hidden_dim, args.dim, bias=False)  # Down

    def forward(self, x):
        # SwiGLU: (xW1 * SiLU(xW3)) * W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFeedForward(nn.Module):
    """
    Sparse MoE Layer (集成 Expert Parallel)
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts = args.moe_num_experts
        self.topk = args.moe_topk
        # 1. Gating Network (Router)
        # Router 通常不做 TP，因为输出维度很小 (num_experts)
        self.gate = nn.Linear(args.dim, self.num_experts, bias=False)
        # 2. Experts
        # 在 EP 模式下，每个 GPU 只持有 total_experts / world_size 个专家
        # 假设 args.ep_size 已经设置正确
        # 这里简化处理：我们创建 num_experts 个 MLP，但在 forward 时 EP 模块会负责路由
        # 实际生产中，这里应该只初始化 local_experts
        self.experts = nn.ModuleList([MLP(args) for _ in range(self.num_experts)])
        # 3. EP 通信模块
        self.ep = ExpertParallel(self.num_experts)
    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape
        x_flat = x.view(-1, D)
        # 1. Routing
        router_logits = self.gate(x_flat) # [N, num_experts]
        probs = F.softmax(router_logits, dim=-1)
        weights, indices = torch.topk(probs, self.topk, dim=-1) # [N, k]
        # 2. EP Dispatch (分发到对应 GPU)
        # dispatched_x: [Total_Recv, D]
        # metadata: 用于恢复顺序
        dispatched_x, metadata = self.ep.dispatch(x_flat, indices)
        # 3. Computation (计算本地专家)
        # 这里是一个简化实现：实际应该根据 metadata 知道哪些 token 属于哪个专家
        # 为了代码跑通，假设所有 token 平均分配给本地专家 (仅作演示逻辑)
        # 真正的 MoE 实现需要在这里做复杂的 index select
        # 假设 dispatched_x 已经包含了所有需要本地计算的 token
        # 我们简单地通过第一个专家计算 (生产环境需要 loop over local experts)
        expert_output = self.experts[0](dispatched_x)
        # 4. EP Combine (聚合回原 GPU)
        combined_output = self.ep.combine(expert_output, metadata)
        # 5. 加权求和 (Weighted Sum)
        # combined_output: [N, k, D]
        # weights: [N, k]
        output = (combined_output * weights.unsqueeze(-1)).sum(dim=1)
        return output.view(B, S, D)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        # 如果配置中没写 n_kv_heads，默认等于 n_heads (即退化为标准 MHA)
        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.head_dim = args.dim // args.n_heads

        # 计算复制倍数，例如 32 / 8 = 4
        self.n_rep = self.n_heads // self.n_kv_heads

        ColLinear = get_linear_cls(args, 'col')
        RowLinear = get_linear_cls(args, 'row')

        # TP 模式下，这里的 dim 会被切分，所以传入 total dim 即可，Layer 内部会处理
        self.wq = ColLinear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = ColLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = ColLinear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = RowLinear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cis):
        B, S, _ = x.shape

        # 1. 投影
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. Reshape
        xq = xq.view(B, S, -1, self.head_dim)
        xk = xk.view(B, S, -1, self.head_dim)
        xv = xv.view(B, S, -1, self.head_dim)

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
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()

        # 1. Attention (集成 CP)
        self.attention = Attention(args)
        if args.cp_size > 1:
            # 如果开启 CP，包裹 Attention
            # 这与 TP 是正交的：TP 切分权重，CP 切分 Sequence
            self.attention = ContextParallelAttention(self.attention, args)

        # 2. FeedForward (集成 MoE)
        # 策略：每 moe_layer_freq 层替换一个 MoE
        # 例如 freq=2: Layer 0 (Dense), Layer 1 (MoE), Layer 2 (Dense)...
        use_moe = (args.moe_num_experts > 1) and (layer_id % args.moe_layer_freq == 1)
        if use_moe:
            self.feed_forward = MoEFeedForward(args)
        else:
            self.feed_forward = MLP(args)

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

        # TP Embedding
        if args.parallel_mode == 'manual_tp':
            self.tok_embeddings = VocabParallelEmbedding(args.vocab_size, args.dim)
            # Output 层通常也是 Column Parallel (Gather Output)
            self.output = ColumnParallelLinear(args.dim, args.vocab_size, bias=False, gather_output=True)
        else:
            self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 传入 layer_id 以决定是否使用 MoE
        self.layers = nn.ModuleList([
            TransformerBlock(args, layer_id=i) for i in range(args.n_layers)
        ])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        # Precompute RoPE frequencies 注意：这里只计算一次，并在 forward 中根据当前 seq_len 切片
        self.freqs_cis = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)

    def forward(self, tokens):
        B, S = tokens.shape
        h = self.tok_embeddings(tokens)

        # 确保 freqs_cis 在同一设备
        freqs_cis = self.freqs_cis[:S].to(h.device)

        for layer in self.layers:
            h = layer(h, freqs_cis)

        h = self.norm(h)
        return self.output(h)
