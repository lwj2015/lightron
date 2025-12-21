import os
import json
import argparse
import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoConfig

from config.config import ModelArgs
from model.model import LightronTransformer
from parallel.distributed import setup_distributed
from parallel.parallel_fsdp import apply_fsdp2
from data.dataloader import MicroBatchDataLoader


def get_args():
    parser = argparse.ArgumentParser(description="Lightron Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def train_step(model, batch, grad_acc_steps):
    """单步训练逻辑"""
    # 数据移动到 GPU
    input_ids = batch["input_ids"].cuda()
    target_ids = batch["target_ids"].cuda()

    # Forward
    # 注意：LightronTransformer 返回的是 [B, S, VocabSize]
    logits = model(input_ids)

    # Loss Calculation
    # Reshape: [B*S, V] vs [B*S]
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target_ids.view(-1)
    )

    # Scale loss for gradient accumulation
    loss = loss / grad_acc_steps
    loss.backward()

    return loss.item() * grad_acc_steps


def main():
    # 1. 解析参数与配置
    args = get_args()
    config = load_config(args.config)

    dist_cfg = config["distributed"]
    train_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["dataset"]

    # 2. 初始化分布式环境 (4D Parallel Setup)
    # 优先从环境变量读取 (torchrun)，如果没设则用 config 的默认值
    tp_size = int(os.environ.get("TP_SIZE", dist_cfg.get("tp_size", 1)))
    dp_size = int(os.environ.get("DP_SIZE", dist_cfg.get("dp_size", 1)))
    cp_size = int(os.environ.get("CP_SIZE", dist_cfg.get("cp_size", 1)))
    pp_size = int(os.environ.get("PP_SIZE", dist_cfg.get("pp_size", 1)))
    ep_size = int(os.environ.get("EP_SIZE", dist_cfg.get("ep_size", 1)))

    setup_distributed(
        tp_size=tp_size,
        pp_size=pp_size,
        cp_size=cp_size,
        ep_size=ep_size,
        dp_size=dp_size
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    if global_rank == 0:
        print(f"🚀 Starting training with config: {args.config}")
        print(f"   World Size: {world_size} | TP={tp_size} DP={dp_size}")

    # 3. 自动加载模型配置 (从 HF)
    # 使用 HF_ENDPOINT 环境变量确保国内能下载
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    if global_rank == 0:
        print(f"Loading model config from {model_cfg['name']}...")

    # 让所有进程都加载 Config (Config 文件很小，不会有并发问题)
    hf_config = AutoConfig.from_pretrained(model_cfg["name"], trust_remote_code=True)

    vocab_size = hf_config.vocab_size
    if tp_size > 1:
        # 计算需要填充多少才能被 tp_size 整除
        if vocab_size % tp_size != 0:
            new_vocab_size = ((vocab_size // tp_size) + 1) * tp_size
            if global_rank == 0:
                print(f"⚠️ Vocab size {vocab_size} is not divisible by TP={tp_size}.")
                print(f"   Padding vocab size to {new_vocab_size}...")
            vocab_size = new_vocab_size

    # 4. 转换为 Lightron ModelArgs
    # 自动映射 HF 参数到 Lightron 参数
    model_args = ModelArgs(
        dim=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads),
        vocab_size=vocab_size,
        max_seq_len=train_cfg["seq_length"],
        norm_eps=getattr(hf_config, "rms_norm_eps", 1e-5),
        # 并行模式：如果 TP > 1，开启手动 TP
        # parallel_mode='manual_tp' if tp_size > 1 else 'fsdp2',
        tp_size=tp_size,
        cp_size=cp_size,
        # MoE 配置 (从 config 读取，如果没有则默认为 1)
        moe_num_experts=model_cfg.get("moe_num_experts", 1),
        moe_topk=model_cfg.get("moe_topk", 2),
        moe_layer_freq=model_cfg.get("moe_layer_freq", 2)
    )

    # 5. 初始化模型
    # 使用 Meta Device 初始化，秒级构建，不占显存
    with torch.device("meta"):
        model = LightronTransformer(model_args)

    # 6. 应用并行策略
    # A. TP/CP/EP: 已经在 model.py 内部通过 parallel_mode 处理了层结构
    # B. FSDP (DP): 处理剩余的参数切分
    if dp_size > 1:
        # FSDP2 会自动处理 Meta 到 Real 的参数初始化
        # 注意：如果 TP>1，这里是混合并行，FSDP2 会在 DP 维度切分

        # 1. 先切分 (此时还是 Meta Tensor)
        model = apply_fsdp2(model)

        # 2. 分配物理显存 (Materialize), 这会在每张卡上只分配它负责的那一部分参数 (Local Shard)
        model = model.to_empty(device="cuda")

        # 3. 初始化参数数值
        # 因为是 Meta 初始化，现在显存里全是垃圾数据，必须 reset
        # 为了保证所有 DP Rank 初始权重一致，我们需要固定随机种子
        torch.manual_seed(42 + global_rank)  # 注意：通常 DP 需要相同种子，但 FSDP2 这种局部初始化比较特殊

        # 更严谨的做法：设置相同的种子，让大家算出一样的随机数（如果切分逻辑允许）, 或者 Rank 0 初始化后广播（太慢）。
        # 对于 FSDP2，最简单的做法是：设置全局统一种子，然后依靠 reset_parameters
        torch.manual_seed(train_cfg.get("seed", 42))

        def init_weights(m):
            # 如果模块有自定义的重置方法（如 Linear, Embedding, 或我们的 ParallelLinear）
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
            # 兜底逻辑：针对原生 PyTorch 层
            elif isinstance(m, (torch.nn.Linear, torch.nn.Embedding)):
                m.reset_parameters()

        model.apply(init_weights)
    else:
        # 纯 TP 模式或单卡模式，需要手动 materialize
        model = model.to_empty(device="cuda")
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

    if global_rank == 0:
        # 统计参数量 (FSDP 下可能不准，仅供参考)
        try:
            param_count = sum(p.numel() for p in model.parameters())
            print(f"Model initialized. Total Parameters (Local/Meta): {param_count / 1e9:.2f}B")
        except:
            pass

    # 7. 初始化 DataLoader
    # 使用我们刚刚测试通过的 MicroBatchDataLoader
    dataloader = MicroBatchDataLoader(
        micro_batch_size=train_cfg["micro_batch_size"],
        seq_length=train_cfg["seq_length"],
        dataset_name=data_cfg["name"],
        tokenizer_name=model_cfg["name"],  # 复用模型名作为 tokenizer 名
        grad_acc_steps=train_cfg["gradient_accumulation_steps"],
        num_workers=data_cfg.get("num_workers", 0),
        max_samples=train_cfg.get("max_samples", None),
        split=data_cfg.get("split", "train")
    )

    # 8. 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01)
    )

    # 9. 训练循环
    model.train()
    total_steps = train_cfg["total_steps"]
    step = 0
    tokens_seen = 0

    start_time = time.time()

    # 创建迭代器
    data_iter = iter(dataloader)

    if global_rank == 0:
        print("\n=== Start Training ===")

    while step < total_steps:
        optimizer.zero_grad()
        loss_accum = 0.0

        # Gradient Accumulation Loop
        for _ in range(train_cfg["gradient_accumulation_steps"]):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Epoch 结束，重新开始
                data_iter = iter(dataloader)
                batch = next(data_iter)

            loss_val = train_step(model, batch, train_cfg["gradient_accumulation_steps"])
            loss_accum += loss_val

        # Optimizer Step
        # FSDP 会自动处理梯度同步
        optimizer.step()

        step += 1
        # 计算吞吐量
        current_tokens = dataloader.global_batch_size * train_cfg["seq_length"]
        tokens_seen += current_tokens

        # Logging
        if global_rank == 0 and step % train_cfg.get("log_interval", 10) == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_seen / elapsed
            print(f"Step {step}/{total_steps} | Loss: {loss_accum:.4f} | TPS: {tokens_per_sec:.2f} tokens/s")

    if global_rank == 0:
        print("Training Finished!")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
