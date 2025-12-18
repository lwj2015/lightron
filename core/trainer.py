import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
)
from core.model.model import LightronTransformer, TransformerBlock
from core.config.config import ModelArgs


def setup():
    # 初始化分布式环境
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    torch.distributed.destroy_process_group()


def train():
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])

    # 1. 创建模型
    args = ModelArgs(dim=1024, n_layers=8, n_heads=8)
    model = LightronTransformer(args).to(local_rank)

    # 2. FSDP 核心配置
    # 自动包裹策略：告诉 FSDP 每一层 TransformerBlock 是一个独立的切分单元
    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # 混合精度策略：BF16 计算，FP32 梯度累加 (H100/A100 推荐)
    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )

    # 3. 包装模型
    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=bf16_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
        device_id=torch.cuda.current_device()
    )

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    # 4. 简易训练循环
    for step in range(100):
        inputs = torch.randint(0, 32000, (4, 128)).to(local_rank)  # Dummy data

        optimizer.zero_grad()
        logits = model(inputs)

        # 简单的 Next Token Prediction Loss
        targets = inputs  # 仅作演示
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss.backward()
        optimizer.step()

        if local_rank == 0 and step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")

    cleanup()


if __name__ == "__main__":
    train()
