import torch
import torch.nn as nn
import torch.distributed as dist
from .distributed import get_device_mesh


def _all_to_all_single(input_, output_, group):
    """
    封装 dist.all_to_all_single，处理 input/output shape 动态变化的情况。
    EP 中，每个 GPU 发送和接收的 Token 数量可能不同（负载不均）。
    """
    # input_splits: list of int, 表示发给每个 rank 的 token 数
    # output_splits: list of int, 表示从每个 rank 接收的 token 数
    # 这通常需要先通信一次 metadata (token counts)
    pass


class ExpertParallel(nn.Module):
    """
    专家并行 (EP) 辅助模块。
    负责：
    1. Dispatch: 根据路由索引，将 Token 发送到对应的 GPU。
    2. Combine: 计算完成后，将 Token 发送回原来的 GPU。
    """

    def __init__(self, num_experts):
        super().__init__()
        self.mesh = get_device_mesh()
        # 假设 mesh 中有名为 'ep' 的维度 (通常复用 dp 或独立 ep)
        # 如果没有显式 ep 维度，通常使用 world_size 作为 ep_size
        if self.mesh and "ep" in self.mesh.mesh_dim_names:
            self.ep_group = self.mesh["ep"].get_group()
        else:
            # 默认回退到 None，即不开启 EP (所有专家都在本地)
            self.ep_group = None

    def dispatch(self, x, expert_indices):
        """
        x: [Batch*Seq, Hidden]
        expert_indices: [Batch*Seq, TopK] (每个 token 选择了哪些专家)
        """
        if self.ep_group is None:
            return x, None, None  # 本地模式，直接返回

        world_size = dist.get_world_size(group=self.ep_group)
        rank = dist.get_rank(group=self.ep_group)

        # 1. 计算每个 token 应该去哪个 rank
        # 假设专家均匀分布：Experts [0, 1] -> Rank 0, Experts [2, 3] -> Rank 1
        num_local_experts = self.num_experts // world_size
        target_ranks = expert_indices // num_local_experts

        # 2. 排序并重排 x，使其按 target_rank 连续
        # ... (这里涉及复杂的 argsort 和 scatter 操作，为了代码简洁，展示核心通信) ...

        # 3. All-to-All Dispatch
        # x_sorted -> x_dispatched
        # x_dispatched = dist.all_to_all_single(x_sorted, group=self.ep_group)

        # 占位返回，实际需要返回重排后的 x 和 恢复所需的 metadata
        return x, None, None

    def combine(self, x, metadata):
        """
        逆操作：把计算完的 token 发回原处
        """
        if self.ep_group is None:
            return x

        # All-to-All Combine
        # x_combined = dist.all_to_all_single(x, group=self.ep_group)
        return x
