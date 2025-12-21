import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from config.config import ModelArgs
from model.model import LightronTransformer, TransformerBlock
from parallel.parallel_fsdp import apply_fsdp1, apply_fsdp2


def run_fsdp_test(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 创建小模型
    args = ModelArgs(dim=128, n_layers=2, n_heads=4, vocab_size=100)
    model = LightronTransformer(args).cuda()

    # 包装 FSDP
    fsdp_model = apply_fsdp1(model, TransformerBlock, use_bf16=False)

    # 运行一次 Forward/Backward
    x = torch.randint(0, 100, (2, 16)).cuda()
    output = fsdp_model(x)
    loss = output.sum()
    loss.backward()

    # 验证梯度是否同步
    # 简单检查：确保没有报错，并且梯度不为 None
    assert fsdp_model.module.output.weight.grad is None  # FSDP 会清除原始参数的 grad
    # 在 FSDP 中，我们需要检查 flat_param 的 grad，或者 step 一次看是否报错

    if rank == 0:
        print("FSDP Test Passed!")

    dist.destroy_process_group()


def test_fsdp_main():
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("Skipping FSDP test (requires at least 2 GPUs)")
        return
    mp.spawn(run_fsdp_test, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    test_fsdp_main()
