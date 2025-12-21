import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from parallel.communication.all_reduce import ring_all_reduce

def init_dist():
    """初始化分布式环境"""
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

    # --- 一键生成 3D Mesh ---
    # 这行代码自动完成了之前几十行的 Group 创建逻辑
    mesh_3d = init_device_mesh(
        "cuda",
        mesh_shape,
        mesh_dim_names=mesh_dim_names
    )

    # --- 直接通过名字获取 Group ---
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
