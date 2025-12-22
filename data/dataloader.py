import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset
from datasets import load_dataset, Features, Sequence, Value
from transformers import AutoTokenizer
from parallel.distributed import get_dp_group, get_cp_group


class MicroBatchDataLoader(DataLoader):
    def __init__(
            self,
            micro_batch_size,
            seq_length,
            dataset_name,
            tokenizer_name,
            num_workers=0,
            num_proc=4,
            grad_acc_steps=1,
            split="train",
            max_samples=None,
            seed=42
    ):
        """
        通用 MicroBatch DataLoader。
        支持：
        1. 任意 HF 数据集
        2. 自动 Tokenize 和 Chunking (拼接长文本)
        3. 分布式采样 (DP)
        4. 上下文并行切分 (CP)
        """
        self.micro_batch_size = micro_batch_size
        self.seq_length = seq_length
        self.grad_acc_steps = grad_acc_steps

        # 获取分布式信息
        dp_group = get_dp_group()
        self.dp_world_size = dist.get_world_size(group=dp_group) if dp_group else 1
        self.dp_rank = dist.get_rank(group=dp_group) if dp_group else 0

        cp_group = get_cp_group()
        self.cp_world_size = dist.get_world_size(group=cp_group) if cp_group else 1
        self.cp_rank = dist.get_rank(group=cp_group) if cp_group else 0

        self.global_batch_size = micro_batch_size * grad_acc_steps * self.dp_world_size

        # CP 模式下，每个 GPU 只负责序列的一部分
        self.seq_length_per_gpu = seq_length // self.cp_world_size

        # 1. 加载 Tokenizer (只在 Rank 0 加载，然后广播，或者利用 HF 的缓存机制)
        # 这里简化为每个进程都加载，HF 会处理缓存锁
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 2. 加载数据集
        # 使用 streaming=True 可以处理超大数据集，但为了 shuffle 和 map 方便，
        # 这里演示 map-style dataset (适合中小数据集如 TinyStories, Wikitext)
        # 如果是 TB 级数据，建议换成 IterableDataset + Buffer Shuffle
        print(f"[Rank {dist.get_rank()}] Loading dataset {dataset_name}...")
        dataset = load_dataset(dataset_name, split=split)

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        # 3. 预处理：Tokenize & Grouping
        # 将文本转换为定长的 input_ids
        self.tokenized_dataset = self.process_dataset(dataset, num_proc)

        # 4. 分布式采样器
        self.sampler = DistributedSampler(
            self.tokenized_dataset,
            num_replicas=self.dp_world_size,
            rank=self.dp_rank,
            shuffle=True,
            seed=seed
        )

        if dist.get_rank() == 0:
          print(
              f"[DataLoader] dp_world_size={self.dp_world_size} cp_world_size={self.cp_world_size} "
              f"seq_length={self.seq_length} seq_length_per_gpu={self.seq_length_per_gpu} "
              f"micro_batch_size={self.micro_batch_size} grad_acc_steps={self.grad_acc_steps}",
              flush=True
          )

        super().__init__(
            self.tokenized_dataset,
            batch_size=micro_batch_size,
            sampler=self.sampler,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            drop_last=True
        )

    def process_dataset(self, dataset, num_proc):
        """
        将文本数据 Tokenize 并拼接成 seq_length + 1 的块
        """
        block_size = self.seq_length + 1

        def group_texts(examples):
            # 1. Tokenize
            tokenized_inputs = self.tokenizer(
                examples["text"],
                return_special_tokens_mask=True,
                truncation=False  # 先不截断，全部拼起来
            )
            concatenated_ids = {}
            for k, v in tokenized_inputs.items():
                # 展平 list of list
                concatenated_ids[k] = sum(v, [])

            total_length = len(concatenated_ids["input_ids"])
            # 丢弃最后不够一个 block 的部分
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size

            # 切分
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_ids.items()
            }
            return result

        # 只有主进程打印进度条
        is_main_process = (dist.get_rank() == 0) if dist.is_initialized() else True

        tokenized_datasets = dataset.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
            desc=f"Grouping texts in chunks of {block_size}",
            load_from_cache_file=True,
            disable_nullable=True
        )
        return tokenized_datasets

    def collate_fn(self, batch):
        """
        处理 Batch，支持 CP 切分
        """
        # batch 是一个 list of dict: [{'input_ids': [...]}, ...]
        input_ids_list = [item['input_ids'] for item in batch]
        batch_tensor = torch.tensor(input_ids_list, dtype=torch.long)

        # 强校验：必须是 S+1
        assert batch_tensor.shape[1] == self.seq_length + 1, \
        f"Expected S+1={self.seq_length+1}, got {batch_tensor.shape[1]}"

        # batch_tensor: [B, S+1]
        # input: 0 ~ S-1
        # target: 1 ~ S

        # CP 切分逻辑
        # 如果 CP=1, start=0, end=S
        # 如果 CP=2, Rank0: 0~S/2, Rank1: S/2~S
        start_idx = self.cp_rank * self.seq_length_per_gpu
        end_idx = start_idx + self.seq_length_per_gpu

        # 强校验：边界必须合法
        assert end_idx <= self.seq_length, \
        f"CP slice out of range: end_idx={end_idx}, seq_length={self.seq_length}"

        # 注意：这里切分的是 input (0~S-1) 和 target (1~S)
        # 原始数据长度是 S+1
        # Input: [B, S_local]
        input_ids = batch_tensor[:, start_idx: end_idx].contiguous()
        # Target: [B, S_local]
        target_ids = batch_tensor[:, start_idx + 1: end_idx + 1].contiguous()

        # 强校验：输出必须是 S_local
        assert input_ids.shape[1] == self.seq_length_per_gpu, \
          f"Expected S_local={self.seq_length_per_gpu}, got {input_ids.shape[1]}"
        assert target_ids.shape == input_ids.shape

        return {
            "input_ids": input_ids,
            "target_ids": target_ids
        }
