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


def ring_reduce_scatter(tensor_list: list, group: dist.ProcessGroup) -> torch.Tensor:
    """
    修正后的 Ring Reduce-Scatter
    逻辑：确保 Rank r 最终持有 Chunk r 的总和
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    if world_size == 1:
        return tensor_list[0]

    # 初始化：Result 包含我本地的贡献
    # 注意：我们直接在 tensor_list 上原地修改，这样最后 tensor_list[rank] 就是结果

    right_rank_logical = (rank + 1) % world_size
    left_rank_logical = (rank - 1 + world_size) % world_size

    right_rank_global = get_global_rank(group, right_rank_logical)
    left_rank_global = get_global_rank(group, left_rank_logical)

    recv_buffer = torch.zeros_like(tensor_list[0])

    for i in range(world_size - 1):
        # 【关键修正】索引偏移 -1
        # Step 0: Rank r 发送 Chunk r-1, 接收 Chunk r-2
        # 这样经过 N-1 步，Chunk r 会正好传回到 Rank r

        send_chunk_idx = (rank - i - 1 + world_size) % world_size
        recv_chunk_idx = (rank - i - 2 + world_size) % world_size

        send_data = tensor_list[send_chunk_idx]

        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, send_data, right_rank_global, group=group),
            dist.P2POp(dist.irecv, recv_buffer, left_rank_global, group=group)
        ])
        for req in reqs: req.wait()

        # 累加到对应的块
        tensor_list[recv_chunk_idx] += recv_buffer

    # 循环结束后，tensor_list[rank] 已经包含了所有人的贡献
    return tensor_list[rank]


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
