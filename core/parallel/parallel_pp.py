import torch
import torch.nn as nn
import torch.distributed as dist
from .distributed import get_device_mesh


class PipelineStage(nn.Module):
    """
    PP Stage Wrapper.
    """

    def __init__(self, module, stage_id, num_stages, chunk_id=0):
        super().__init__()
        self.module = module
        self.stage_id = stage_id
        self.num_stages = num_stages

        self.mesh = get_device_mesh()
        self.pp_group = self.mesh["pp"].get_group() if self.mesh and "pp" in self.mesh.mesh_dim_names else None

        if self.pp_group is None:
            raise ValueError("PP group not initialized")

        # 计算上下游 Rank
        # 注意：这里假设 PP group 内 rank 是连续的 0, 1, 2...
        # 实际需要根据 global rank 映射
        my_rank = dist.get_rank(group=self.pp_group)
        self.prev_rank = (my_rank - 1) if my_rank > 0 else None
        self.next_rank = (my_rank + 1) if my_rank < num_stages - 1 else None

    def forward(self, x=None):
        # PP 的 forward 比较特殊，通常由外部调度器调用 send/recv
        # 这里仅作为单步执行的逻辑
        return self.module(x)

    def send_forward(self, output):
        if self.next_rank is not None:
            dist.send(output.contiguous(), dst=self.next_rank, group=self.pp_group)

    def recv_forward(self, tensor_shape):
        if self.prev_rank is not None:
            buffer = torch.empty(tensor_shape, device="cuda", dtype=torch.bfloat16)
            dist.recv(buffer, src=self.prev_rank, group=self.pp_group)
            return buffer
        return None

    def send_backward(self, grad):
        if self.prev_rank is not None:
            dist.send(grad.contiguous(), dst=self.prev_rank, group=self.pp_group)

    def recv_backward(self, tensor_shape):
        if self.next_rank is not None:
            buffer = torch.empty(tensor_shape, device="cuda", dtype=torch.bfloat16)
            dist.recv(buffer, src=self.next_rank, group=self.pp_group)
            return buffer
        return None
