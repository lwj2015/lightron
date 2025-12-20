import math
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F


def init_dist():
    """初始化分布式环境"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 初始化默认的全局进程组
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    return rank, world_size, device, local_rank


def ring_all_reduce_tp(tensor: torch.Tensor, group: dist.ProcessGroup = None) -> torch.Tensor:
    """
    支持 TP Group 的 Ring AllReduce
    :param tensor: 输入张量
    :param group: 通信组 (ProcessGroup)，如果为 None 则使用全局组
    """
    # 1. 统一 Group 处理
    if group is None:
        group = dist.group.WORLD

    # 获取组内信息 (Logical Rank)
    rank_in_group = dist.get_rank(group)
    world_size_in_group = dist.get_world_size(group)

    # 单卡直接返回
    if world_size_in_group == 1:
        return tensor

    # 2. 预处理：Flatten + Padding (处理不能整除的情况)
    original_shape = tensor.shape
    tensor_flat = tensor.flatten()
    numel = tensor_flat.numel()

    # 计算需要 Pad 的长度
    pad_len = (world_size_in_group - (numel % world_size_in_group)) % world_size_in_group
    if pad_len > 0:
        tensor_flat = F.pad(tensor_flat, (0, pad_len))

    # 3. 分块 (创建 View，不占用额外内存)
    chunk_size = tensor_flat.numel() // world_size_in_group
    chunks = list(tensor_flat.split(chunk_size))

    # 4. 邻居计算 (Logical Rank -> Global Rank)
    # Ring 拓扑基于 Group Rank 计算左右邻居
    right_rank_logical = (rank_in_group + 1) % world_size_in_group
    left_rank_logical = (rank_in_group - 1 + world_size_in_group) % world_size_in_group

    # 转换为 Global Rank 用于 P2P 通信 (NCCL backend 需要 Global Rank)
    right_rank_global = dist.get_global_rank(group, right_rank_logical)
    left_rank_global = dist.get_global_rank(group, left_rank_logical)

    # ----------------- 5. Reduce-Scatter 阶段 -----------------
    for step in range(world_size_in_group - 1):
        # 计算当前步的数据块索引
        send_idx = (rank_in_group - step + world_size_in_group) % world_size_in_group
        recv_idx = (rank_in_group - step - 1 + world_size_in_group) % world_size_in_group

        send_chunk = chunks[send_idx]
        # 接收 Buffer (必须新分配，不能覆盖正在使用的 chunk)
        recv_buffer = torch.empty_like(chunks[recv_idx])

        # 异步通信
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_chunk, right_rank_global, group=group),
            dist.P2POp(dist.irecv, recv_buffer, left_rank_global, group=group)
        ])
        for req in reqs: req.wait()

        # 累加 (In-place 修改 tensor_flat 的对应部分)
        chunks[recv_idx].add_(recv_buffer)

    # ----------------- 6. All-Gather 阶段 -----------------
    for step in range(world_size_in_group - 1):
        send_idx = (rank_in_group - step + 1 + world_size_in_group) % world_size_in_group
        recv_idx = (rank_in_group - step + world_size_in_group) % world_size_in_group

        send_chunk = chunks[send_idx]

        # AllGather 阶段可以直接写入目标 chunk，因为该 chunk 在当前 step 不会被读取发送
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_chunk, right_rank_global, group=group),
            dist.P2POp(dist.irecv, chunks[recv_idx], left_rank_global, group=group)
        ])
        for req in reqs: req.wait()

    # 7. 后处理：合并 + 去除 Padding + Reshape
    res = torch.cat(chunks)
    if pad_len > 0:
        res = res[:-pad_len]

    return res.reshape(original_shape)


# -------------------------- 使用示例 --------------------------
if __name__ == "__main__":
    rank, world_size, device, local_rank = init_dist()

    # --- 1. 模拟创建 TP Group ---
    # 假设 TP_SIZE = 2 (每两张卡组成一个 TP 组)
    tp_size = 2
    my_tp_group = None

    # 遍历所有可能的组，找到当前 rank 所属的组
    # 例如 8 卡：[0,1], [2,3], [4,5], [6,7]
    for i in range(0, world_size, tp_size):
        ranks = list(range(i, i + tp_size))
        # 注意：new_group 需要所有进程都调用，且顺序一致
        group = dist.new_group(ranks)
        if rank in ranks:
            my_tp_group = group

    # --- 2. 准备数据 ---
    # 故意构造一个不能整除的长度，测试 Padding 功能
    local_grad = torch.randn(4097, device=device) * (rank + 1)

    # 备份用于官方对比 (必须 Clone!)
    grad_for_official = local_grad.clone()

    # --- 3. 执行手写 Ring AllReduce (指定 TP Group) ---
    # 此时只会在 TP 组内聚合。例如 Rank 0 只会和 Rank 1 聚合。
    global_grad = ring_all_reduce_tp(local_grad, group=my_tp_group)

    # --- 4. 执行官方 AllReduce (指定 TP Group) ---
    dist.all_reduce(grad_for_official, op=dist.ReduceOp.SUM, group=my_tp_group)

    # --- 5. 验证 ---
    # 打印部分结果
    print(f"Rank {rank} (Group Rank {dist.get_rank(my_tp_group)}) | "
          f"手写: {global_grad[:3].cpu().numpy()} | "
          f"官方: {grad_for_official[:3].cpu().numpy()}")

    # 计算误差
    error = torch.mean((global_grad - grad_for_official) ** 2)
    print(f"Rank {rank} TP Group 误差: {error.item()}")

    # 销毁进程组
    dist.destroy_process_group()
