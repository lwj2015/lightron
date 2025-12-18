import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol
from .distributed import get_tp_group


def _all_reduce(input_: torch.Tensor):
    """All-Reduce within TP group"""
    tp_group = get_tp_group()
    if tp_group is None: return input_
    dist.all_reduce(input_, op=dist.ReduceOp.SUM, group=tp_group)
    return input_


def _split(input_: torch.Tensor, dim=-1):
    """Split tensor into parts for TP"""
    tp_group = get_tp_group()
    if tp_group is None: return input_
    world_size = dist.get_world_size(group=tp_group)
    rank = dist.get_rank(group=tp_group)
    chunks = torch.chunk(input_, world_size, dim=dim)
    return chunks[rank].contiguous()


def _gather(input_: torch.Tensor, dim=-1):
    """All-Gather within TP group"""
    tp_group = get_tp_group()
    if tp_group is None: return input_
    return funcol.all_gather_tensor(input_, gather_dim=dim, group=tp_group)


# === Sequence Parallel (SP) 原语 ===
# SP 的核心是：在 LayerNorm/Dropout 时，按 Seq 维度切分；在 Linear 计算时，按 Hidden 维度切分。
# 这需要 all_to_all 通信 (Scatter/Gather 的变体)

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, gather_output=False, sequence_parallel=False):
        super().__init__()
        self.gather_output = gather_output
        self.sequence_parallel = sequence_parallel

        tp_group = get_tp_group()
        self.tp_size = dist.get_world_size(group=tp_group) if tp_group else 1

        assert out_features % self.tp_size == 0
        self.out_features_per_partition = out_features // self.tp_size

        self.weight = nn.Parameter(torch.empty(self.out_features_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)

        # Init logic omitted for brevity (use master_seed)

    def forward(self, x):
        # x: [B, S, H]

        if self.sequence_parallel:
            # SP 模式下，输入 x 是在 Seq 维度切分的 [B, S/TP, H]
            # 我们需要把它 gather 回来变成 [B, S, H] 才能做矩阵乘法
            # 或者使用 Ring 算法。这里简化为 All-Gather 输入
            x = _gather(x, dim=1)

            # Local MatMul
        # Output: [B, S, H/TP]
        output = F.linear(x, self.weight, self.bias)

        if self.gather_output:
            output = _gather(output, dim=-1)

        return output


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, input_is_parallel=True, sequence_parallel=False):
        super().__init__()
        self.input_is_parallel = input_is_parallel
        self.sequence_parallel = sequence_parallel

        tp_group = get_tp_group()
        self.tp_size = dist.get_world_size(group=tp_group) if tp_group else 1

        assert in_features % self.tp_size == 0
        self.in_features_per_partition = in_features // self.tp_size

        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: [B, S, H/TP]

        # Local MatMul
        # Output: [B, S, H] (Partial Sum)
        output = F.linear(x, self.weight)

        if self.sequence_parallel:
            # SP 模式下，我们不做 All-Reduce (Sum)，而是做 Reduce-Scatter
            # 结果变成 [B, S/TP, H]，保持 Seq 维度切分
            # 注意：PyTorch 的 reduce_scatter_tensor API
            output = funcol.reduce_scatter_tensor(output, reduceOp='sum', scatter_dim=1, group=get_tp_group())
        else:
            if self.input_is_parallel:
                output = _all_reduce(output)

        if self.bias is not None:
            output = output + self.bias
        return output


class VocabParallelEmbedding(nn.Module):
    """TP for Embedding Layer"""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        tp_group = get_tp_group()
        self.tp_size = dist.get_world_size(group=tp_group) if tp_group else 1
        self.tp_rank = dist.get_rank(group=tp_group) if tp_group else 0

        # 按 Vocab 维度切分
        self.vocab_start_index = self.tp_rank * (num_embeddings // self.tp_size)
        self.vocab_end_index = self.vocab_start_index + (num_embeddings // self.tp_size)

        self.weight = nn.Parameter(torch.empty(num_embeddings // self.tp_size, embedding_dim))

    def forward(self, input_):
        # input_: [B, S]
        # Mask out tokens not in this partition
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0

        output = F.embedding(masked_input, self.weight)
        output[input_mask, :] = 0.0

        # All-Reduce to sum up embeddings from all partitions
        output = _all_reduce(output)
        return output
