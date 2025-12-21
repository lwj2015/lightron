import os
import sys
import torch
import torch.distributed as dist

from data.dataloader import MicroBatchDataLoader
from parallel.distributed import setup_distributed


def test_dataloader(tp_size = 1, dp_size = 1):
    # 1. 模拟分布式环境初始化
    if not dist.is_initialized():
      dist.init_process_group("nccl")
      local_rank = int(os.environ["LOCAL_RANK"])
      torch.cuda.set_device(local_rank)
      setup_distributed(tp_size=tp_size, dp_size=dp_size)

    print("\n=== 1. Environment Setup ===")
    print(f"Rank: {dist.get_rank()}, World Size: {dist.get_world_size()}")

    # 2. 配置参数
    model_name = "gpt2"
    dataset_name = "roneneldan/TinyStories"
    seq_length = 128
    micro_batch_size = 4

    print(f"\n=== 2. Initializing DataLoader ===")
    print(f"Model: {model_name}, Dataset: {dataset_name}, SeqLen: {seq_length}")

    try:
        dataloader = MicroBatchDataLoader(
            micro_batch_size=micro_batch_size,
            seq_length=seq_length,
            dataset_name=dataset_name,  # load_dataset 会自动处理
            tokenizer_name=model_name,
            split="train",
            max_samples=1000,  # 只取前1000条，加快测试速度
            num_workers=0  # 调试模式建议设为 0，避免多进程报错
        )
        print("✅ DataLoader initialized successfully.")
    except Exception as e:
        print(f"❌ DataLoader initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 验证数据迭代
    print("\n=== 3. Testing Iteration ===")
    try:
        iterator = iter(dataloader)
        batch = next(iterator)

        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]

        print(f"Batch keys: {batch.keys()}")
        print(f"Input shape: {input_ids.shape}")  # 预期: [4, 128]
        print(f"Target shape: {target_ids.shape}")  # 预期: [4, 128]

        # 4. 验证逻辑正确性
        # Target 应该是 Input 向右移动一位
        # 比如 Input: [A, B, C], Target: [B, C, D]
        # 但由于我们是从长文本截断的，无法直接验证 input[i+1] == target[i] (除非我们拿到原始数据)
        # 不过根据 collate_fn 的逻辑：
        # input = data[0:S], target = data[1:S+1]
        # 所以在同一行内，target[t] 应该等于 input[t+1] (如果它们来自同一个连续片段)
        # 验证最后 5 个 token

        print("\n--- Sample Check (First Sequence) ---")
        print(f"Input  (last 5): {input_ids[0, -5:].tolist()}")
        print(f"Target (last 5): {target_ids[0, -5:].tolist()}")

        # 验证形状
        assert input_ids.shape == (micro_batch_size, seq_length), "Input shape mismatch!"
        assert target_ids.shape == (micro_batch_size, seq_length), "Target shape mismatch!"

        # 验证类型
        assert input_ids.dtype == torch.long, "Input dtype should be long"

        print("✅ Data shape and type check passed.")

    except Exception as e:
        print(f"❌ Iteration failed: {e}")
        import traceback
        traceback.print_exc()

    # 5. 清理
    dist.destroy_process_group()
    print("\n=== Test Finished ===")


if __name__ == "__main__":
    tp_size = 2
    dp_size = 4
    test_dataloader(tp_size, dp_size)
