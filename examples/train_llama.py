import os
import time
import torch
import torch.optim as optim
from lightron.config import ModelArgs
from lightron.model import LightronTransformer, TransformerBlock
from lightron.parallel import get_fsdp_wrapper, save_fsdp_checkpoint
from lightron.data import create_dataloader


def main():
    # 1. 环境初始化
    torch.distributed.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"Starting training on {world_size} GPUs...")

    # 2. 模型配置
    args = ModelArgs(
        dim=1024,
        n_layers=8,
        n_heads=8,
        vocab_size=32000,
        max_seq_len=1024
    )

    # 3. 实例化模型并应用 FSDP
    model = LightronTransformer(args).to(local_rank)

    # use compile to fuse operator
    model = torch.compile(model)

    model = get_fsdp_wrapper(model, TransformerBlock, use_bf16=True)

    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 4. 优化器
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # 5. 数据加载
    train_loader, sampler = create_dataloader("train.bin", args.max_seq_len, batch_size=4, rank=rank,
                                              world_size=world_size)

    # 6. 训练循环
    model.train()
    for epoch in range(2):
        sampler.set_epoch(epoch)
        for step, (x, y) in enumerate(train_loader):
            t0 = time.time()
            x, y = x.to(local_rank), y.to(local_rank)

            optimizer.zero_grad()
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
            if hasattr(model, 'aux_loss'):
                loss += 0.01 * model.aux_loss
            loss.backward()

            # 梯度裁剪 (FSDP 需要调用 clip_grad_norm_)
            model.clip_grad_norm_(1.0)
            optimizer.step()

            torch.cuda.synchronize()
            t1 = time.time()

            if rank == 0 and step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | Time: {(t1 - t0) * 1000:.2f}ms")

        # 每个 Epoch 保存一次 Checkpoint
        save_fsdp_checkpoint(model, rank, f"checkpoint_epoch_{epoch}.pt")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
