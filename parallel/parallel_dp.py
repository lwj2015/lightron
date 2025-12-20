import torch
import torch.nn as nn
import torch.distributed as dist
from .distributed import get_device_mesh


class DataParallel(nn.Module):
    """
    手动实现的 DDP (类似于 PyTorch DDP，但为了配合 4D 并行而简化)。
    它不自动切分数据，只负责梯度的 All-Reduce。
    """

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.mesh = get_device_mesh()
        # 获取 DP 组 (可能包含 FSDP/DDP 语义)
        # 在 4D 并行中，DP 组通常是 mesh["dp"]
        if self.mesh and "dp" in self.mesh.mesh_dim_names:
            self.dp_group = self.mesh["dp"].get_group()
        else:
            self.dp_group = None

        # 注册 hook，在反向传播时自动同步梯度
        # 为了简化，我们这里不使用 bucket (桶)，而是逐个参数同步，或者在 step 前手动同步
        # 工业级实现会使用 Bucket 来合并小梯度通信
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._all_reduce_hook)

    def _all_reduce_hook(self, param):
        if self.dp_group is None or param.grad is None:
            return

        # 异步 All-Reduce
        # 注意：这里需要除以 dp_size 来求平均
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.dp_group)
        param.grad.div_(dist.get_world_size(group=self.dp_group))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
