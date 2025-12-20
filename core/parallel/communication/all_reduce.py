import math
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F


# ==========================================
# 1. 基础环境初始化
# ==========================================
def init_dist():
    """初始化分布式环境"""
    # 从环境变量读取 Rank 信息 (torchrun 自动注入)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 绑定当前进程到指定的 GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 初始化 NCCL 后端
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            device_id=device
        )

    return rank, world_size, device, local_rank


# ==========================================
# 2. 3D 并行 Group 构建逻辑
# ==========================================
class ParallelGroups:
    def __init__(self, tp_size, pp_size, dp_size, rank, world_size):
        assert world_size == tp_size * pp_size * dp_size, \
            f"World Size ({world_size}) != TP({tp_size}) * PP({pp_size}) * DP({dp_size})"

        self.tp_group = None
        self.dp_group = None
        self.pp_group = None  # PP 通常不需要 AllReduce，但为了完整性列出逻辑

        print(f"[Rank {rank}] 初始化 Group: TP={tp_size}, PP={pp_size}, DP={dp_size}")

        # --- 构建 TP Group (连续切分) ---
        # 逻辑：[0,1], [2,3], [4,5], [6,7]
        num_tp_groups = world_size // tp_size
        for i in range(num_tp_groups):
            ranks = list(range(i * tp_size, (i + 1) * tp_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                self.tp_group = group
                # 仅在 Rank 0 打印一次拓扑结构
                if rank == 0:
                    print(f"  TP Group {i}: {ranks}")

        # --- 构建 DP Group (跨 PP 的同位切分) ---
        # 逻辑：步长为 tp_size * pp_size。
        # Rank 0 (Stage0, TP0) <-> Rank 4 (Stage0, TP0) [DP Group 0]
        # Rank 1 (Stage0, TP1) <-> Rank 5 (Stage0, TP1) [DP Group 1]
        # Rank 2 (Stage1, TP0) <-> Rank 6 (Stage1, TP0) [DP Group 2] ...

        # 这里的逻辑是：DP 组连接的是“完全相同的模型部分”但在“不同的数据副本”上的卡
        # 在 3D 并行中，通常 DP 维度的 stride 是最大的，或者取决于 rank 的排列方式。
        # 这里假设 Rank 排列顺序为：DP -> PP -> TP (Megatron 常见方式)
        # 但为了适配简单的 0-7 线性排列，我们假设排列是：
        # Rank ID = dp_idx * (pp * tp) + pp_idx * (tp) + tp_idx

        stride = tp_size * pp_size
        num_dp_groups = stride  # 有多少个并行的流水线/TP组合，就有多少个 DP 组

        for i in range(num_dp_groups):
            # i 表示 (PP_idx, TP_idx) 的组合索引
            ranks = [i + k * stride for k in range(dp_size)]
            group = dist.new_group(ranks)
            if rank in ranks:
                self.dp_group = group
                if rank == 0:
                    print(f"  DP Group (Base {i}): {ranks}")


# ==========================================
# 3. 通用 Ring AllReduce (支持任意 Group)
# ==========================================
def ring_all_reduce(tensor: torch.Tensor, group: dist.ProcessGroup = None) -> torch.Tensor:
    """
    通用的 Ring AllReduce 实现
    :param tensor: 输入张量
    :param group: 通信组 (TP组 或 DP组)
    """
    if group is None:
        group = dist.group.WORLD

    # 1. 获取组内逻辑 Rank (0 ~ group_size-1)
    rank_in_group = dist.get_rank(group)
    world_size_in_group = dist.get_world_size(group)

    if world_size_in_group == 1:
        return tensor

    # 2. 预处理：Flatten + Padding
    original_shape = tensor.shape
    tensor_flat = tensor.flatten()
    numel = tensor_flat.numel()

    pad_len = (world_size_in_group - (numel % world_size_in_group)) % world_size_in_group
    if pad_len > 0:
        tensor_flat = F.pad(tensor_flat, (0, pad_len))

    # 3. 分块
    chunk_size = tensor_flat.numel() // world_size_in_group
    chunks = list(tensor_flat.split(chunk_size))

    # 4. 计算环形邻居 (逻辑 Rank -> 物理 Global Rank)
    right_rank_logical = (rank_in_group + 1) % world_size_in_group
    left_rank_logical = (rank_in_group - 1 + world_size_in_group) % world_size_in_group

    right_rank_global = dist.get_global_rank(group, right_rank_logical)
    left_rank_global = dist.get_global_rank(group, left_rank_logical)

    # 5. Reduce-Scatter
    for step in range(world_size_in_group - 1):
        send_idx = (rank_in_group - step + world_size_in_group) % world_size_in_group
        recv_idx = (rank_in_group - step - 1 + world_size_in_group) % world_size_in_group

        send_chunk = chunks[send_idx]
        recv_buffer = torch.empty_like(chunks[recv_idx])

        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_chunk, right_rank_global, group=group),
            dist.P2POp(dist.irecv, recv_buffer, left_rank_global, group=group)
        ])
        for req in reqs: req.wait()
        chunks[recv_idx].add_(recv_buffer)

    # 6. All-Gather
    for step in range(world_size_in_group - 1):
        send_idx = (rank_in_group - step + 1 + world_size_in_group) % world_size_in_group
        recv_idx = (rank_in_group - step + world_size_in_group) % world_size_in_group

        send_chunk = chunks[send_idx]
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_chunk, right_rank_global, group=group),
            dist.P2POp(dist.irecv, chunks[recv_idx], left_rank_global, group=group)
        ])
        for req in reqs: req.wait()

    # 7. 恢复形状
    res = torch.cat(chunks)
    if pad_len > 0:
        res = res[:-pad_len]
    return res.reshape(original_shape)


# ==========================================
# 4. 主测试逻辑
# ==========================================
def main():
    rank, world_size, device, local_rank = init_dist()

    # 设定并行度：总卡数 8 = 2(TP) * 2(PP) * 2(DP)
    TP_SIZE = 2
    PP_SIZE = 2
    DP_SIZE = 2

    # 初始化 Group
    groups = ParallelGroups(TP_SIZE, PP_SIZE, DP_SIZE, rank, world_size)

    # 简单的同步屏障，防止打印混乱
    dist.barrier()
    if rank == 0:
        print("\n" + "=" * 50)
        print("🚀 开始测试 3D 并行 Ring AllReduce")
        print("=" * 50 + "\n")
    dist.barrier()

    # -------------------------------------------------
    # 测试场景 1: TP AllReduce (模拟前向/反向传播中的聚合)
    # -------------------------------------------------
    # 只有同一个 TP 组内的卡会聚合。
    # 例如 Rank 0 和 Rank 1 聚合；Rank 2 和 Rank 3 聚合。

    tensor_tp = torch.randn(1024, device=device) * (rank + 1)
    tensor_tp_ref = tensor_tp.clone()

    # 执行手写 Ring AllReduce
    res_tp = ring_all_reduce(tensor_tp, group=groups.tp_group)

    # 执行官方 AllReduce
    dist.all_reduce(tensor_tp_ref, op=dist.ReduceOp.SUM, group=groups.tp_group)

    # 验证误差
    err_tp = torch.mean((res_tp - tensor_tp_ref) ** 2)

    # 打印结果 (只打印 Rank 0 和 Rank 2，代表不同的 TP 组)
    if rank in [0, 2]:
        print(f"[TP Test] Rank {rank} (TP Group Rank {dist.get_rank(groups.tp_group)}): "
              f"Error = {err_tp.item():.5e} | "
              f"Val: {res_tp[0].item():.4f} vs Ref: {tensor_tp_ref[0].item():.4f}")

    dist.barrier()

    # -------------------------------------------------
    # 测试场景 2: DP AllReduce (模拟梯度同步)
    # -------------------------------------------------
    # 同一个 DP 组内的卡聚合。
    # 根据我们的逻辑，Rank 0 和 Rank 4 是一个 DP 组。

    tensor_dp = torch.randn(1024, device=device) + (rank + 10)
    tensor_dp_ref = tensor_dp.clone()

    # 执行手写 Ring AllReduce
    res_dp = ring_all_reduce(tensor_dp, group=groups.dp_group)

    # 执行官方 AllReduce
    dist.all_reduce(tensor_dp_ref, op=dist.ReduceOp.SUM, group=groups.dp_group)

    # 验证误差
    err_dp = torch.mean((res_dp - tensor_dp_ref) ** 2)

    # 打印结果 (只打印 Rank 0 和 Rank 1，代表不同的 DP 组基底)
    if rank in [0, 1]:
        print(f"[DP Test] Rank {rank} (DP Group Rank {dist.get_rank(groups.dp_group)}): "
              f"Error = {err_dp.item():.5e} | "
              f"Val: {res_dp[0].item():.4f} vs Ref: {tensor_dp_ref[0].item():.4f}")

    dist.barrier()
    if rank == 0:
        print("\n✅ 测试完成，所有误差应在 1e-10 级别 (浮点精度误差)")

    # 清理
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
