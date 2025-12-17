import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelArgs


class Expert(nn.Module):
    """ 一个标准的 MLP 专家 """

    def __init__(self, args: ModelArgs):
        super().__init__()
        # 专家的维度通常比标准 MLP 小，或者保持一致
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SparseMoE(nn.Module):
    def __init__(self, args: ModelArgs, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控网络 (Router)
        self.gate = nn.Linear(args.dim, num_experts, bias=False)

        # 专家列表
        self.experts = nn.ModuleList([Expert(args) for _ in range(num_experts)])

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(-1, D)  # [B*S, D]

        # 1. 计算路由概率
        logits = self.gate(x_flat)  # [B*S, num_experts]
        probs = F.softmax(logits, dim=-1)

        density = probs.mean(dim=0)
        self.aux_loss = (density ** 2).sum() * self.num_experts

        # 2. 选择 Top-K 专家
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        # 3. 归一化权重
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # 4. 专家计算 (这里使用简单的循环实现，生产环境通常用 CUDA kernel 或 Grouped GEMM 优化)
        final_output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]  # [B*S]
            prob = topk_probs[:, k].unsqueeze(-1)  # [B*S, 1]

            # 这一步效率较低，仅作演示原理。
            # 实际中会将相同 expert 的 token 聚合在一起计算
            for i in range(self.num_experts):
                mask = (expert_idx == i)
                if mask.any():
                    selected_input = x_flat[mask]
                    expert_output = self.experts[i](selected_input)
                    # 累加结果： output += prob * expert(input)
                    final_output[mask] += prob[mask] * expert_output

        return final_output.view(B, S, D)

# 注意：要在 model.py 中使用这个，需要在 ModelArgs 里加 use_moe 标志，
# 并在 TransformerBlock 初始化时判断：
# self.feed_forward = SparseMoE(args) if args.use_moe else FeedForward(args)
