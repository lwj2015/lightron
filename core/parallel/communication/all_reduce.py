import os
import torch
import torch.distributed as dist
import torch.nn.functional as F

# 导入 Device Mesh (PyTorch 2.x 新特性)
from torch.distributed.device_mesh import init_device_mesh


# ==========================================
# 1. 基础环境初始化
# ==========================================
def init_dist():
    """初始化分布式环境 (地基)"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 1. 绑定设备
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 2. 初始化默认进程组 (虽然 init_device_mesh 可以自动初始化，
    #    但显式初始化并指定 device_id 是消除 Warning 的最佳实践)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)

    return rank, world_size, device, local_rank


# ==========================================
# 2. 通用 Ring AllReduce (核心逻辑完全不变)
# ==========================================
def ring_all_reduce(tensor: torch.Tensor, group: dist.ProcessGroup = None) -> torch.Tensor:
    """
    通用的 Ring AllReduce 实现
    :param tensor: 输入张量
    :param group: 通信组 (TP组 或 DP组)
    """
    if group is None:
        group = dist.group.WORLD

    # 获取组内逻辑 Rank
    rank_in_group = dist.get_rank(group)
    world_size_in_group = dist.get_world_size(group)

    if world_size_in_group == 1:
        return tensor

    # 预处理：Flatten + Padding
    original_shape = tensor.shape
    tensor_flat = tensor.flatten()
    numel = tensor_flat.numel()

    pad_len = (world_size_in_group - (numel % world_size_in_group)) % world_size_in_group
    if pad_len > 0:
        tensor_flat = F.pad(tensor_flat, (0, pad_len))

    # 分块
    chunk_size = tensor_flat.numel() // world_size_in_group
    chunks = list(tensor_flat.split(chunk_size))

    # 计算环形邻居 (逻辑 Rank -> 物理 Global Rank)
    right_rank_logical = (rank_in_group + 1) % world_size_in_group
    left_rank_logical = (rank_in_group - 1 + world_size_in_group) % world_size_in_group

    right_rank_global = dist.get_global_rank(group, right_rank_logical)
    left_rank_global = dist.get_global_rank(group, left_rank_logical)

    # Reduce-Scatter
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

    # All-Gather
    for step in range(world_size_in_group - 1):
        send_idx = (rank_in_group - step + 1 + world_size_in_group) % world_size_in_group
        recv_idx = (rank_in_group - step + world_size_in_group) % world_size_in_group

        send_chunk = chunks[send_idx]
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_chunk, right_rank_global, group=group),
            dist.P2POp(dist.irecv, chunks[recv_idx], left_rank_global, group=group)
        ])
        for req in reqs: req.wait()

    # 恢复形状
    res = torch.cat(chunks)
    if pad_len > 0:
        res = res[:-pad_len]
    return res.reshape(original_shape)


# ==========================================
# 3. 主测试逻辑 (使用 init_device_mesh)
# ==========================================
def main():
    rank, world_size, device, local_rank = init_dist()
    """
    # init_process_group不是必须的，可用这一段来代替init_dist
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    """

    # 设定并行度：总卡数 8 = 2(DP) * 2(PP) * 2(TP)
    # 注意：这里的顺序很重要，决定了 Rank 如何映射到 Mesh
    # 通常顺序是 (Data, Pipeline, Tensor)
    mesh_shape = (2, 2, 2)
    mesh_dim_names = ("dp", "pp", "tp")

    # --- 【核心修改】一键生成 3D Mesh ---
    # 这行代码自动完成了之前几十行的 Group 创建逻辑
    mesh_3d = init_device_mesh(
        "cuda",
        mesh_shape,
        mesh_dim_names=mesh_dim_names
    )

    # --- 【核心修改】直接通过名字获取 Group ---
    # 获取 TP 组 (沿着 "tp" 维度切分)
    tp_group = mesh_3d["tp"].get_group()

    # 获取 DP 组 (沿着 "dp" 维度切分)
    dp_group = mesh_3d["dp"].get_group()

    # 简单的同步屏障
    dist.barrier()
    if rank == 0:
        print("\n" + "=" * 50)
        print(f"🚀 Device Mesh 3D 并行测试 (Shape: {mesh_shape})")
        print("=" * 50 + "\n")
        # 打印 Mesh 结构看看
        print(f"Mesh Structure:\n{mesh_3d}")

    dist.barrier()

    # -------------------------------------------------
    # 测试场景 1: TP AllReduce
    # -------------------------------------------------
    tensor_tp = torch.randn(1024, device=device) * (rank + 1)
    tensor_tp_ref = tensor_tp.clone()

    # 传入从 Mesh 获取的 tp_group
    res_tp = ring_all_reduce(tensor_tp, group=tp_group)
    dist.all_reduce(tensor_tp_ref, op=dist.ReduceOp.SUM, group=tp_group)

    err_tp = torch.mean((res_tp - tensor_tp_ref) ** 2)

    if rank in [0, 1]:  # 打印 Rank 0 和 1 (它们应该在同一个 TP 组)
        print(f"[TP Test] Rank {rank} (TP-Group Rank {dist.get_rank(tp_group)}): "
              f"Error = {err_tp.item():.5e}")

    dist.barrier()

    # -------------------------------------------------
    # 测试场景 2: DP AllReduce
    # -------------------------------------------------
    tensor_dp = torch.randn(1024, device=device) + (rank + 10)
    tensor_dp_ref = tensor_dp.clone()

    # 传入从 Mesh 获取的 dp_group
    res_dp = ring_all_reduce(tensor_dp, group=dp_group)
    dist.all_reduce(tensor_dp_ref, op=dist.ReduceOp.SUM, group=dp_group)

    err_dp = torch.mean((res_dp - tensor_dp_ref) ** 2)

    if rank in [0, 4]:  # 打印 Rank 0 和 4 (它们应该在同一个 DP 组)
        print(f"[DP Test] Rank {rank} (DP-Group Rank {dist.get_rank(dp_group)}): "
              f"Error = {err_dp.item():.5e}")
    
    print(f"\n\n _flatten_mesh_list: {mesh_3d._flatten_mesh_list}")

    dist.barrier()

    # 清理
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
