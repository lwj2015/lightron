import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from parallel.communication.ring_all_gather import ring_all_gather


def init_dist():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)

    return rank, world_size, device


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

    # 测试 Ring All-Gather (在 CP 组内)
    cp_group = mesh["cp"].get_group()
    local_vec = torch.arange(10, device=device, dtype=torch.float32) + rank * 100

    gather_list = [torch.zeros_like(local_vec) for _ in range(dist.get_world_size(cp_group))]
    dist.all_gather(gather_list, local_vec, group=cp_group)
    ref_gather = torch.cat(gather_list)

    my_gather = ring_all_gather(local_vec, group=cp_group)

    diff = (ref_gather - my_gather).abs().max()
    if rank == 0:
        print(f"\n[CP Group] All-Gather Test:")
        print(f"   Max Diff: {diff.item():.6f} {'✅' if diff < 1e-5 else '❌'}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
