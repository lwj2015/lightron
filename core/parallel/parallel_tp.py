import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol


# 简单的 All-Reduce 封装
def _all_reduce(input_: torch.Tensor):
    if not dist.is_initialized():
        return input_
    # 在 TP 组内进行 All-Reduce
    # 注意：这里简化处理，假设整个 world 都是 TP 组。
    # 实际生产中需要根据 device_mesh 创建 process_group
    dist.all_reduce(input_, op=dist.ReduceOp.SUM)
    return input_


class ColumnParallelLinear(nn.Module):
    """
    列并行：将输出维度 (out_features) 切分。
    用于：Attention 的 QKV 投影，FFN 的第一层 (Up proj)。
    """

    def __init__(self, in_features, out_features, bias=False, gather_output=False):
        super().__init__()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.gather_output = gather_output

        assert out_features % world_size == 0, "Output features must be divisible by world size"
        self.output_size_per_partition = out_features // world_size

        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
        else:
            self.register_parameter('bias', None)

        # 初始化略...

    def forward(self, x):
        # x: [B, S, H] -> Output: [B, S, H / TP]
        output = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            # 如果需要聚合（比如在 Embedding 层），则 All-Gather
            output = funcol.all_gather_tensor(output, gather_dim=-1, group=None)
        return output


class RowParallelLinear(nn.Module):
    """
    行并行：将输入维度 (in_features) 切分。
    用于：Attention 的 Output 投影，FFN 的第二层 (Down proj)。
    """

    def __init__(self, in_features, out_features, bias=False, input_is_parallel=True):
        super().__init__()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.input_is_parallel = input_is_parallel

        assert in_features % world_size == 0, "Input features must be divisible by world size"
        self.input_size_per_partition = in_features // world_size

        self.weight = nn.Parameter(torch.empty(out_features, self.input_size_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # x: [B, S, H / TP] -> Output: [B, S, H]
        output = F.linear(x, self.weight)

        # Row Parallel 的核心：输出需要 All-Reduce 叠加
        if self.input_is_parallel:
            output = _all_reduce(output)

        if self.bias is not None:
            output = output + self.bias
        return output
