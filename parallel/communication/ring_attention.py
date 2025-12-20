import os
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

# ==========================================
# 1. 核心算法: Ring Attention Kernel
# ==========================================
def ring_attention_kernel(local_q, local_k, local_v, cp_group, tp_group):
    """
    实现 Ring Attention + Online Softmax
    local_q: [B, S_local, H_local, D]
    """
    rank = dist.get_rank(group=cp_group)
    world_size = dist.get_world_size(group=cp_group)
    
    # 维度信息
    B, S_local, H_local, D = local_q.shape
    scale = 1.0 / math.sqrt(D)

    # === 初始化 Online Softmax 的统计量 ===
    # max_score: 当前行的最大值 (用于数值稳定)
    # sum_exp: 当前行的分母 (exp的和)
    # out: 当前的分子 (加权和)
    local_max = torch.full((B, S_local, H_local, 1), float('-inf'), device=local_q.device)
    local_sum_exp = torch.zeros((B, S_local, H_local, 1), device=local_q.device)
    local_out = torch.zeros_like(local_q)

    # === 准备通信缓冲区 ===
    # curr: 当前计算用的 KV
    # next: 接收下一个 Step 的 KV
    curr_k, curr_v = local_k.clone(), local_v.clone()
    next_k, next_v = torch.empty_like(local_k), torch.empty_like(local_v)

    # === Ring Loop ===
    # 环状通信：Rank i -> Rank i+1, Rank i-1 -> Rank i
    # 也就是向右发，从左接
    right_rank = (rank + 1) % world_size
    left_rank = (rank - 1 + world_size) % world_size

    # 获取全局 Rank ID，因为 P2P 通信需要全局 ID
    global_right_rank = dist.get_global_rank(cp_group, right_rank)
    global_left_rank = dist.get_global_rank(cp_group, left_rank)

    for step in range(world_size):
        # === 1. 启动异步通信 (使用 batch_isend_irecv) ===
        ops = []
        if step < world_size - 1:
            # 定义操作列表
            # 发送 K, V 给右边
            ops.append(dist.P2POp(dist.isend, curr_k, global_right_rank, cp_group))
            ops.append(dist.P2POp(dist.isend, curr_v, global_right_rank, cp_group))
            # 从左边接收 K, V
            ops.append(dist.P2POp(dist.irecv, next_k, global_left_rank, cp_group))
            ops.append(dist.P2POp(dist.irecv, next_v, global_left_rank, cp_group))
            # 关键修改：原子提交所有 P2P 请求
            reqs = dist.batch_isend_irecv(ops)
        else:
            reqs = []

        # 2. 计算 Attention (Computation)
        # Q [B, S, H, D] @ K.T [B, H, D, S] -> Score [B, H, S, S]
        # 注意：这里的 K 是来自 Ring 的某一段 Sequence
        # 为了方便矩阵乘法，我们调整维度: [B, H, S, D]
        q_ = local_q.transpose(1, 2)
        k_ = curr_k.transpose(1, 2)
        v_ = curr_v.transpose(1, 2)
        
        # [B, H, S_local, D] @ [B, H, D, S_remote] -> [B, H, S_local, S_remote]
        attn_score = torch.matmul(q_, k_.transpose(-1, -2)) * scale
        
        # 3. Online Softmax 更新逻辑 (核心数学)
        # 这一步通常在 CUDA Kernel 内部完成，这里用 Python 模拟
        # 维度转回 [B, S, H, 1] 以便广播
        attn_score = attn_score.transpose(1, 2) # [B, S_local, H, S_remote]
        
        # 找当前块的最大值
        block_max = torch.max(attn_score, dim=-1, keepdim=True).values
        
        # 更新全局最大值
        new_max = torch.maximum(local_max, block_max)
        
        # 计算缩放因子
        scale_old = torch.exp(local_max - new_max)
        scale_block = torch.exp(block_max - new_max)
        
        # 计算当前块的 exp
        # P_block = exp(score - block_max)
        p_block = torch.exp(attn_score - block_max)
        
        # 更新分母 Sum_Exp
        # sum_new = sum_old * scale_old + sum_block * scale_block
        block_sum = torch.sum(p_block, dim=-1, keepdim=True)
        local_sum_exp = local_sum_exp * scale_old + block_sum * scale_block
        
        # 更新分子 Out
        # out_new = out_old * scale_old + (P_block @ V_block) * scale_block
        # 先算 P @ V
        # [B, H, S_local, S_remote] @ [B, H, S_remote, D] -> [B, H, S_local, D]
        p_v = torch.matmul(p_block.transpose(1, 2), v_) 
        p_v = p_v.transpose(1, 2) # [B, S_local, H, D]
        
        local_out = local_out * scale_old + p_v * scale_block
        
        # 更新 Max
        local_max = new_max

        # 4. 等待通信完成
        for req in reqs:
            req.wait()
            
        # 5. 交换 Buffer
        curr_k = next_k.clone()
        curr_v = next_v.clone()

    # 最后除以分母
    final_out = local_out / local_sum_exp
    return final_out

# ==========================================
# 2. 模拟主流程 (Device Mesh + Data Sim)
# ==========================================
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
    tp_size = dist.get_world_size(tp_group) # 2
    local_head_num = Global_Head // tp_size
    
    # 2. CP 切分 (Sequence 维度)
    cp_rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group) # 4
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
    seq_restored = torch.cat(gathered_seq_out, dim=1) # [B, Global_Seq, Local_Head, D]
    
    # 再把 TP (Head) 拼回去
    gathered_head_out = [torch.zeros_like(seq_restored) for _ in range(tp_size)]
    dist.all_gather(gathered_head_out, seq_restored, group=tp_group)
    # 此时我们有了 [Head_Chunk0, Head_Chunk1] -> 拼成完整 Head
    final_restored = torch.cat(gathered_head_out, dim=2) # [B, Global_Seq, Global_Head, D]

    # 2. 运行标准 Attention (Reference)
    if rank == 0:
        # PyTorch 标准实现
        # 需要转置为 [B, H, S, D]
        ref_q = global_q.transpose(1, 2)
        ref_k = global_k.transpose(1, 2)
        ref_v = global_v.transpose(1, 2)
        
        ref_out = F.scaled_dot_product_attention(ref_q, ref_k, ref_v)
        ref_out = ref_out.transpose(1, 2) # 转回 [B, S, H, D]
        
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
