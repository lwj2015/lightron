import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from parallel.communication.ring_reduce_scatter import ring_reduce_scatter


def init_dist():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)

    return rank, world_size, device


def get_global_rank(group, group_rank):
    return dist.get_global_rank(group, group_rank)


def main():
    rank, world_size, device = init_dist()

    # 构建 4D Device Mesh
    if world_size < 8:
        if rank == 0: print("⚠️ Warning: Less than 8 GPUs, using simplified mesh.")
        mesh_shape = (1, 1, 2, 2)
    else:
        mesh_shape = (2, 1, 2, 2)

    mesh_dim_names = ("dp", "pp", "tp", "cp")
    mesh = init_device_mesh("cuda", mesh_shape, mesh_dim_names=mesh_dim_names)

    if rank == 0:
        print(f"\n🚀 Device Mesh Created: {mesh_shape} {mesh_dim_names}")

    dist.barrier()

    # 测试 Reduce-Scatter (在 DP 组内)
    dp_group = mesh["dp"].get_group()
    dp_world_size = dist.get_world_size(dp_group)

    input_list = [torch.ones(10, device=device) * (rank + 1) * (i + 1) for i in range(dp_world_size)]

    ref_out = torch.zeros(10, device=device)
    dist.reduce_scatter(ref_out, input_list, group=dp_group)

    input_list_2 = [torch.ones(10, device=device) * (rank + 1) * (i + 1) for i in range(dp_world_size)]
    my_out = ring_reduce_scatter(input_list_2, group=dp_group)

    diff = (ref_out - my_out).abs().max()
    if dist.get_rank(dp_group) == 0 and rank == 0:
        print(f"\n[DP Group] Reduce-Scatter Test:")
        print(f"   Max Diff: {diff.item():.6f} {'✅' if diff < 1e-5 else '❌'}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
