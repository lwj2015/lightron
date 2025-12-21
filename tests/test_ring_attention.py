import os
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from parallel.communication.ring_attention import ring_attention_kernel


def run_demo(rank, world_size):
    # 初始化进程组
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # --- A. 构建 Device Mesh (DP=1, PP=1, TP=2, CP=4) ---
    # 8 张卡：
    # Global: `[Batch, Seq, Head, Dim]`
    # Local: `[Batch, Seq/CP, Head/TP, Dim]`

    # TP 组: [0,1], [2,3], [4,5], [6,7] (负责切分 Head)
    # CP 组: [0,2,4,6], [1,3,5,7] (负责切分 Sequence)
    mesh = init_device_mesh("cuda", (1, 1, 2, 4), mesh_dim_names=("dp", "pp", "tp", "cp"))
    cp_group = mesh["cp"].get_group()
    tp_group = mesh["tp"].get_group()

    # --- B. 数据模拟 ---
    # 假设全局参数
    B, Global_Seq, Global_Head, Dim = 2, 32, 8, 64

    # 生成全局数据 (仅用于验证对比，实际训练中不会有这个变量)
    if rank == 0:
        global_q = torch.randn(B, Global_Seq, Global_Head, Dim, device="cuda")
        global_k = torch.randn(B, Global_Seq, Global_Head, Dim, device="cuda")
        global_v = torch.randn(B, Global_Seq, Global_Head, Dim, device="cuda")
    else:
        global_q = torch.empty(B, Global_Seq, Global_Head, Dim, device="cuda")
        global_k = torch.empty(B, Global_Seq, Global_Head, Dim, device="cuda")
        global_v = torch.empty(B, Global_Seq, Global_Head, Dim, device="cuda")

    # 广播全局数据，保证大家用来切分的数据源是一致的
    dist.broadcast(global_q, src=0)
    dist.broadcast(global_k, src=0)
    dist.broadcast(global_v, src=0)

    # --- C. 数据切分 (Sharding) ---
    # 1. TP 切分 (Head 维度)
    tp_rank = dist.get_rank(tp_group)
    tp_size = dist.get_world_size(tp_group)  # 2
    local_head_num = Global_Head // tp_size

    # 2. CP 切分 (Sequence 维度)
    cp_rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group)  # 4
    local_seq_len = Global_Seq // cp_size

    # 执行切分
    # 先切 Head (TP)
    temp_q = global_q.chunk(tp_size, dim=2)[tp_rank]
    temp_k = global_k.chunk(tp_size, dim=2)[tp_rank]
    temp_v = global_v.chunk(tp_size, dim=2)[tp_rank]

    # 再切 Seq (CP)
    local_q = temp_q.chunk(cp_size, dim=1)[cp_rank].clone()
    local_k = temp_k.chunk(cp_size, dim=1)[cp_rank].clone()
    local_v = temp_v.chunk(cp_size, dim=1)[cp_rank].clone()

    print(f"[Rank {rank}] Mesh Coord: TP={tp_rank}, CP={cp_rank} | "
          f"Local Shape: {local_q.shape} (Seq={local_seq_len}, Head={local_head_num})")

    # --- D. 运行 Ring Attention ---
    dist.barrier()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    # 调用我们手写的 Kernel
    local_out = ring_attention_kernel(local_q, local_k, local_v, cp_group, tp_group)
    end_event.record()
    torch.cuda.synchronize()

    if rank == 0:
        print(f"Ring Attention Time: {start_event.elapsed_time(end_event):.3f} ms")

    # --- E. 结果验证 (Gather vs Standard) ---

    # 1. 逆向还原 (Un-shard)
    # 先把 CP (Seq) 拼回去
    gathered_seq_out = [torch.zeros_like(local_out) for _ in range(cp_size)]
    dist.all_gather(gathered_seq_out, local_out, group=cp_group)
    # 此时我们有了 [Seq_Chunk0, Seq_Chunk1, ...] -> 拼成完整 Seq
    seq_restored = torch.cat(gathered_seq_out, dim=1)  # [B, Global_Seq, Local_Head, D]

    # 再把 TP (Head) 拼回去
    gathered_head_out = [torch.zeros_like(seq_restored) for _ in range(tp_size)]
    dist.all_gather(gathered_head_out, seq_restored, group=tp_group)
    # 此时我们有了 [Head_Chunk0, Head_Chunk1] -> 拼成完整 Head
    final_restored = torch.cat(gathered_head_out, dim=2)  # [B, Global_Seq, Global_Head, D]

    # 2. 运行标准 Attention (Reference)
    if rank == 0:
        # PyTorch 标准实现
        # 需要转置为 [B, H, S, D]
        ref_q = global_q.transpose(1, 2)
        ref_k = global_k.transpose(1, 2)
        ref_v = global_v.transpose(1, 2)

        ref_out = F.scaled_dot_product_attention(ref_q, ref_k, ref_v)
        ref_out = ref_out.transpose(1, 2)  # 转回 [B, S, H, D]

        # 3. 对比
        # 由于浮点数累加顺序不同，会有微小误差 (1e-5 级别)
        max_diff = (final_restored - ref_out).abs().max()
        print(f"\n=== Verification Result ===")
        print(f"Max Difference: {max_diff.item():.6f}")

        if max_diff < 1e-4:
            print("✅ SUCCESS: Ring Attention matches Standard Attention!")
        else:
            print("❌ FAILED: Difference is too large.")

    dist.destroy_process_group()


if __name__ == "__main__":
    WORLD_SIZE = 8
    # 检查是否有 8 个 GPU，没有的话报错
    if torch.cuda.device_count() < WORLD_SIZE:
        print(f"Error: This script requires {WORLD_SIZE} GPUs, but found {torch.cuda.device_count()}.")
    else:
        mp.spawn(run_demo, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
