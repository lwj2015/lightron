import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from config.config import ModelArgs
from model.model import LightronTransformer, TransformerBlock
from parallel.distributed import setup_distributed, get_device_mesh
from parallel.parallel_fsdp import apply_fsdp1, apply_fsdp2


def train():
    # 1. 初始化分布式环境 (包含 Device Mesh 的创建)
    # 假设我们用 4 卡，TP=2, DP=2
    # 实际使用时应该从 args 读取
    tp_size = int(os.environ.get("TP_SIZE", 1))
    setup_distributed(tp_size=tp_size)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # 2. 配置模型
    # 如果 TP > 1，parallel_mode 设为 'manual_tp'
    parallel_mode = 'manual_tp' if tp_size > 1 else 'fsdp1'

    args = ModelArgs(
        dim=1024,
        n_layers=8,
        n_heads=8,
        parallel_mode=parallel_mode
    )

    # 3. 创建模型 (此时已经在 CPU/Meta 上根据 args 决定了层结构)
    model = LightronTransformer(args)

    # 4. 并行化包装
    if parallel_mode == 'fsdp1':
        # 使用我们封装好的 FSDP1
        model = apply_fsdp1(
            model,
            transformer_layer_cls=TransformerBlock,
            use_bf16=True
        )
    elif parallel_mode == 'fsdp2':
        # 使用 FSDP2
        model = apply_fsdp2(model, transformer_layer_cls=TransformerBlock)
    elif parallel_mode == 'manual_tp':
        # TP 模式下，模型已经在 __init__ 里切分好了
        # 只需要把模型移动到 GPU 即可
        # 注意：如果是 TP + DDP 混合，这里还需要套一个 DDP
        model = model.to(local_rank)
        # 简易版：假设纯 TP，不做 DDP

    print(f"Rank {local_rank} Model initialized.")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    # 5. 训练循环
    model.train()
    for step in range(100):
        # 构造假数据
        inputs = torch.randint(0, 32000, (4, 128)).to(local_rank)

        optimizer.zero_grad()
        logits = model(inputs)

        # 计算 Loss
        # Shift labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        loss.backward()

        # TP 模式下需要手动同步梯度吗？
        # 如果是纯 TP (world_size == tp_size)，不需要，因为 RowParallel 已经在 forward 里做过 All-Reduce 了。
        # 如果是 TP + DP，需要在这里对 DP 组做 All-Reduce (DataParallel 模块的工作)。

        optimizer.step()

        if local_rank == 0 and step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    # cleanup 由 torchrun 自动处理，或者显式 destroy
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
