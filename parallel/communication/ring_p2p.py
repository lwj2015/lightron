import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
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


def ring_p2p(send_tensor, recv_tensor, src_group_rank, dst_group_rank, group):
    my_group_rank = dist.get_rank(group)
    src_global_rank = get_global_rank(group, src_group_rank)
    dst_global_rank = get_global_rank(group, dst_group_rank)

    ops = []
    if my_group_rank == src_group_rank:
        ops.append(dist.P2POp(dist.isend, send_tensor, dst_global_rank, group=group))
    if my_group_rank == dst_group_rank:
        ops.append(dist.P2POp(dist.irecv, recv_tensor, src_global_rank, group=group))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs: req.wait()


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

    # 测试 P2P (在 PP 组内)
    pp_group = mesh["pp"].get_group()
    pp_rank = dist.get_rank(pp_group)
    pp_size = dist.get_world_size(pp_group)

    if pp_size > 1:
        tensor_p2p = torch.tensor([rank * 1.0], device=device)
        recv_p2p = torch.tensor([-1.0], device=device)

        if pp_rank % 2 == 0:
            target_logical = pp_rank + 1
            if target_logical < pp_size:
                ring_p2p(tensor_p2p, recv_p2p, pp_rank, target_logical, pp_group)
        else:
            src_logical = pp_rank - 1
            ring_p2p(tensor_p2p, recv_p2p, src_logical, pp_rank, pp_group)

            if rank == 1:
                print(f"\n[PP Group] P2P Test (Rank 1 received from Rank 0):")
                print(f"   Received: {recv_p2p.item()} (Expected 0.0) {'✅' if recv_p2p.item() == 0.0 else '❌'}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
