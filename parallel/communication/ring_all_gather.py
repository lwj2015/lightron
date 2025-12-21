import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


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


def ring_all_gather(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    if world_size == 1:
        return tensor.unsqueeze(0)

    output_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    output_tensors[rank] = tensor.clone()

    right_rank_logical = (rank + 1) % world_size
    left_rank_logical = (rank - 1 + world_size) % world_size

    right_rank_global = get_global_rank(group, right_rank_logical)
    left_rank_global = get_global_rank(group, left_rank_logical)

    curr_send_idx = rank
    curr_recv_idx = left_rank_logical

    for _ in range(world_size - 1):
        send_data = output_tensors[curr_send_idx]
        recv_data = output_tensors[curr_recv_idx]

        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_data, right_rank_global, group=group),
            dist.P2POp(dist.irecv, recv_data, left_rank_global, group=group)
        ])
        for req in reqs: req.wait()

        curr_send_idx = curr_recv_idx
        curr_recv_idx = (curr_recv_idx - 1 + world_size) % world_size

    return torch.cat(output_tensors, dim=0)


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
