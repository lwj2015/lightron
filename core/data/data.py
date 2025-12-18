import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os


class BinDataset(Dataset):
    """
    读取二进制 token 文件 (如 numpy memmap)
    假设数据格式为 uint16 (vocab size < 65535)
    """

    def __init__(self, data_path, seq_len):
        self.seq_len = seq_len
        # 使用 memmap 避免一次性加载大文件到内存
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)
        self.num_samples = self.total_tokens // (seq_len + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        # 取 seq_len + 1 个 token，前 seq_len 是 input，后 seq_len 是 target
        chunk = torch.from_numpy(self.data[start: start + self.seq_len + 1].astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def create_dataloader(data_path, seq_len, batch_size, rank, world_size):
    # 如果没有真实数据，生成假数据用于测试
    if not os.path.exists(data_path):
        if rank == 0:
            print(f"Data file {data_path} not found. Creating dummy data...")
            dummy_data = np.random.randint(0, 32000, (100000,), dtype=np.uint16)
            with open(data_path, 'wb') as f:
                f.write(dummy_data.tobytes())
        torch.distributed.barrier()  # 等待 rank 0 写完

    dataset = BinDataset(data_path, seq_len)

    # 关键：使用 DistributedSampler 确保每个 GPU 读不同的数据
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    return loader, sampler
