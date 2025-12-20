import os
import math
import torch
import torch.distributed as dist


# 本地测试使用
def init_dist():
    # 1. 读取LOCAL_RANK（torchrun自动设置，对应本地GPU编号0~7）
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 2. 绑定本地GPU
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # 3. 初始化分布式
    dist.init_process_group(backend="nccl")

    return rank, world_size, device, local_rank


def ring_all_reduce(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    tensor = tensor.contiguous()
    chunk_size = math.ceil(tensor.numel() / world_size)
    chunks = [tensor[i * chunk_size: (i + 1) * chunk_size] for i in range(world_size)]

    # 定义环形邻居
    right_rank = (rank + 1) % world_size
    left_rank = (rank - 1 + world_size) % world_size

    # ----------------- 2. Reduce-Scatter -----------------
    for step in range(world_size - 1):
        # 计算当前步需要发送和接收的数据块索引
        # 逻辑：第 i 步，rank r 发送 chunks[(r - i) % size]
        send_chunk_idx = (rank - step + world_size) % world_size
        recv_chunk_idx = (rank - step - 1 + world_size) % world_size

        send_chunk = chunks[send_chunk_idx]
        recv_chunk = torch.zeros_like(chunks[recv_chunk_idx])  # Buffer for incoming

        # 使用 batch_isend_irecv 避免死锁
        reqs = []
        reqs.append(dist.P2POp(dist.isend, send_chunk, right_rank))
        reqs.append(dist.P2POp(dist.irecv, recv_chunk, left_rank))

        reqs = dist.batch_isend_irecv(reqs)
        for req in reqs:
            req.wait()

        # 累加收到的梯度到对应的本地块
        chunks[recv_chunk_idx] += recv_chunk

    # ----------------- 3. AllGather -----------------
    for step in range(world_size - 1):
        # 逻辑：ReduceScatter 结束后，rank r 拥有完整的 chunks[(r + 1) % size]
        # 需要把这个完整块向右传
        send_chunk_idx = (rank - step + 1 + world_size) % world_size
        recv_chunk_idx = (rank - step + world_size) % world_size

        send_chunk = chunks[send_chunk_idx]
        recv_chunk = torch.zeros_like(chunks[recv_chunk_idx])

        reqs = []
        reqs.append(dist.P2POp(dist.isend, send_chunk, right_rank))
        reqs.append(dist.P2POp(dist.irecv, recv_chunk, left_rank))

        reqs = dist.batch_isend_irecv(reqs)
        for req in reqs:
            req.wait()
        chunks[recv_chunk_idx].copy_(recv_chunk)

    return torch.cat(chunks).reshape(tensor.shape)


# -------------------------- Ring AllReduce 使用示例 --------------------------
if __name__ == "__main__":
    rank, world_size, device, local_rank = init_dist()

    # 测试张量（模拟 TP 中的梯度张量）
    local_grad = torch.randn(4096, device=device) * (rank + 1)
    grad_for_official = local_grad.clone()
    print(f"Rank {rank} (LOCAL_RANK {local_rank}) 输入梯度前3值: {local_grad[:3]}")

    # 执行 Ring AllReduce（传入rank和world_size，避免依赖全局变量）
    global_grad = ring_all_reduce(local_grad, rank, world_size)
    print(f"Rank {rank} (LOCAL_RANK {local_rank}) 聚合后梯度前3值: {global_grad[:3]}")

    # 对比 PyTorch 官方 AllReduce（验证正确性）
    dist.all_reduce(grad_for_official, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} (LOCAL_RANK {local_rank}) 官方 AllReduce 前3值: {grad_for_official[:3]}")
    print(
        f"Rank {rank} (LOCAL_RANK {local_rank}) 手写 vs 官方 误差: {torch.mean((global_grad - grad_for_official) ** 2)}")