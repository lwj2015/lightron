import os
import argparse
import torch
import torch.optim as optim
from config.config import ModelArgs
from model.model import LightronTransformer, TransformerBlock
from parallel.distributed import setup_distributed, get_device_mesh
from parallel.parallel_fsdp import apply_fsdp2
from parallel.parallel_pp import PipelineStage


def parse_args():
    parser = argparse.ArgumentParser()
    # 并行配置
    parser.add_argument("--tp", type=int, default=1, help="Tensor Parallel size")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline Parallel size")
    parser.add_argument("--cp", type=int, default=1, help="Context Parallel size")
    parser.add_argument("--ep", type=int, default=1, help="Expert Parallel size")
    parser.add_argument("--dp", type=int, default=1, help="Data Parallel size")
    # 模型配置
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=8)
    return parser.parse_args()


def train():
    args = parse_args()

    # 1. 启动 5D 并行环境
    setup_distributed(
        tp_size=args.tp,
        pp_size=args.pp,
        cp_size=args.cp,
        ep_size=args.ep,
        dp_size=args.dp
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 2. 构造 ModelArgs
    # 这里的 parallel_mode 决定了 Layer 内部的行为 (TP/CP/EP)
    # PP 和 DP 是 Layer 外部的行为
    model_args = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=8,
        # 告诉模型层：如果 TP>1，请初始化为 Column/Row Parallel, 如果 CP>1，请初始化 ContextAttention
        parallel_mode='manual_tp' if args.tp > 1 else 'fsdp1'
    )

    # 3. 实例化模型 (CPU/Meta)
    # 此时：
    # - 如果 TP>1: Linear 层已经是切分的
    # - 如果 CP>1: Attention 已经是 ContextParallelAttention
    # - 如果 EP>1: MLP 应该是 MoE Layer (需要在 model.py 里支持)
    model = LightronTransformer(model_args)

    # 4. 应用 Pipeline Parallel (PP)
    if args.pp > 1:
        # PP 需要把模型切成 Stage
        # 这通常需要改写 model 的 forward，或者在这里手动切分 layers
        # 示例：
        model = PipelineStage(model, stage_id=..., num_stages=args.pp)
        pass

    # 5. 应用 Data Parallel (DP/FSDP)
    # 即使有 TP/PP，剩下的维度通常用 FSDP 管理
    # FSDP2 支持在 DeviceMesh 的 "dp" 维度上进行 Sharding
    if args.dp > 1:
        # 这里的 mesh 已经是多维的了，FSDP2 会自动寻找名为 "dp" 的维度进行切分
        # 这就是 FSDP2 + DeviceMesh 的强大之处！
        model = apply_fsdp2(model, transformer_layer_cls=TransformerBlock)
    else:
        # 如果没有 DP (比如纯 TP 占满了所有卡)，直接移到 GPU
        model = model.to(local_rank)

    print(f"[Rank {local_rank}] Model Ready. Params: {sum(p.numel() for p in model.parameters())}")

    # 6. 训练循环 (伪代码)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    for step in range(100):
        # ... 数据加载 ...

        if args.pp > 1:
            # PP 需要特殊的调度 (1F1B)
            # loss = pp_schedule.step(...)
            pass
        else:
            # 标准 Forward
            optimizer.zero_grad()
            logits = model(inputs)
            loss = ...
            loss.backward()
            optimizer.step()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
