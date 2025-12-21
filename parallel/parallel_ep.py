import torch
import torch.nn as nn
import torch.distributed as dist
from .distributed import get_device_mesh


def _all_to_all_with_handshake(data, splits, group):
    """
    带有握手机制的 All-to-All 通信。

    Args:
        data: 本地要发送的排序后的数据 [Total_Tokens, Hidden]
        splits: List[int], 长度为 world_size，表示发给每个 rank 的 token 数量
        group: 通信组

    Returns:
        received_data: 接收到的数据 [Total_Received, Hidden]
        received_splits: List[int], 表示从每个 rank 接收到的 token 数量
    """
    world_size = dist.get_world_size(group=group)

    # 1. 握手 (Handshake): 交换数据量信息
    # 我们要发送的 counts
    send_counts = torch.tensor(splits, device=data.device, dtype=torch.long)
    # 我们准备接收的 counts
    recv_counts = torch.empty(world_size, device=data.device, dtype=torch.long)

    # All-to-All 交换 counts
    # 比如: Rank 0 发给 Rank 1 说 "我有 10 个 token"，Rank 1 就会在 recv_counts[0] 收到 10
    dist.all_to_all_single(recv_counts, send_counts, group=group)

    # 2. 准备接收缓冲区
    recv_splits = recv_counts.tolist()
    total_recv_tokens = sum(recv_splits)
    hidden_dim = data.size(1)

    # 如果没有数据要收发，直接返回空
    if total_recv_tokens == 0 and data.numel() == 0:
        return torch.empty(0, hidden_dim, device=data.device, dtype=data.dtype), recv_splits

    recv_data = torch.empty(total_recv_tokens, hidden_dim, device=data.device, dtype=data.dtype)

    # 3. 传输实际数据 (Payload)
    # PyTorch 的 all_to_all_single 需要 input_split_sizes 和 output_split_sizes
    dist.all_to_all_single(
        recv_data,
        data,
        output_split_sizes=recv_splits,
        input_split_sizes=splits,
        group=group
    )

    return recv_data, recv_splits


class ExpertParallel(nn.Module):
    """
    专家并行 (EP) 核心模块。
    实现了 Token 的 Dispatch (分发) 和 Combine (聚合)。
    """

    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.mesh = get_device_mesh()

        # 获取 EP 组
        if self.mesh and "ep" in self.mesh.mesh_dim_names:
            self.ep_group = self.mesh["ep"].get_group()
        else:
            self.ep_group = None

    def dispatch(self, x, expert_indices):
        """
        将 Token 分发到对应的 GPU。

        Args:
            x: [Batch * Seq, Hidden] 输入 Token
            expert_indices: [Batch * Seq, TopK] 每个 Token 选中的专家索引

        Returns:
            dispatched_x: [Total_Recv, Hidden] 接收到的需要计算的 Token
            metadata: dict, 包含恢复顺序所需的所有信息
        """
        # 1. 本地模式 (无 EP)
        if self.ep_group is None:
            # 为了接口一致性，我们需要把 TopK 展开
            # x: [N, D] -> [N, K, D] -> [N*K, D]
            topk = expert_indices.size(1)
            x_expanded = x.unsqueeze(1).expand(-1, topk, -1).reshape(-1, x.size(-1))
            return x_expanded, {"is_local": True, "topk": topk}

        world_size = dist.get_world_size(group=self.ep_group)
        rank = dist.get_rank(group=self.ep_group)

        # 2. 准备数据
        # x: [N, D], indices: [N, K]
        # 我们需要把 x 复制 K 份，因为一个 token 可能去多个专家
        topk = expert_indices.size(1)
        # [N, D] -> [N, K, D] -> [N*K, D]
        x_flat = x.unsqueeze(1).expand(-1, topk, -1).reshape(-1, x.size(-1))
        # [N, K] -> [N*K]
        indices_flat = expert_indices.view(-1)

        # 3. 计算目标 Rank
        # 假设专家是均匀切分的。例如 8 个专家，4 个 GPU，则每个 GPU 负责 2 个。
        # Rank 0: Experts [0, 1], Rank 1: Experts [2, 3] ...
        num_local_experts = self.num_experts // world_size
        target_ranks = indices_flat // num_local_experts

        # 4. 排序 (Sorting) - 核心步骤
        # 我们必须把发往 Rank 0 的数据排在前面，Rank 1 的排在后面...
        sort_indices = torch.argsort(target_ranks)

        # 根据排序结果重排 x 和 target_ranks
        x_sorted = x_flat[sort_indices]
        target_ranks_sorted = target_ranks[sort_indices]

        # 5. 计算 Split Sizes (发给每个 Rank 多少个)
        # 统计每个 rank 出现的次数
        # bincount 统计 [0, 1, 1, 2] -> [1, 2, 1]
        splits = torch.bincount(target_ranks_sorted, minlength=world_size).tolist()

        # 6. 执行 All-to-All 通信
        received_x, received_splits = _all_to_all_with_handshake(x_sorted, splits, self.ep_group)

        # 7. 保存 Metadata (用于 Combine 阶段恢复)
        metadata = {
            "is_local": False,
            "sort_indices": sort_indices,  # 用于恢复顺序
            "send_splits": splits,  # 发送时的切分 (Combine 时接收用)
            "recv_splits": received_splits,  # 接收时的切分 (Combine 时发送用)
            "original_batch_size": x.size(0),
            "topk": topk
        }

        return received_x, metadata

    def combine(self, x, metadata):
        """
        将计算完成的 Token 发回原 GPU，并恢复顺序。

        Args:
            x: [Total_Recv, Hidden] 计算后的 Token (通常是经过 MLP 后的)
            metadata: dispatch 返回的元数据
        """
        if metadata["is_local"]:
            # 恢复形状 [N*K, D] -> [N, K, D]
            return x.view(metadata["original_batch_size"], metadata["topk"], -1)

        # 1. 逆向通信
        # Dispatch: Send(splits) -> Recv(recv_splits)
        # Combine:  Send(recv_splits) -> Recv(splits)
        # 注意：这里的 x 是已经在当前 GPU 算好的，数量等于 recv_splits

        received_x, _ = _all_to_all_with_handshake(
            x,
            metadata["recv_splits"],  # 我现在发回去的数量，就是我之前收到的数量
            self.ep_group
        )

        # 2. 恢复顺序 (Un-sort)
        # received_x 目前是按 Rank 排序的 (Rank 0 发回来的, Rank 1 发回来的...)
        # 我们需要把它变回 dispatch 之前的顺序

        # 创建一个空的 buffer
        # sort_indices[i] = j 意味着：排序后的第 i 个元素来自原始的第 j 个位置
        # 所以：original[sort_indices[i]] = sorted[i]
        # 逆操作：original[j] = sorted[inverse_map[j]]

        # 更简单的方法：直接根据 sort_indices 赋值
        output = torch.empty_like(received_x)
        # output[sort_indices] = received_x
        # 上面这行是错的，应该是 scatter 或者 index_copy
        # 正确逻辑：received_x 是排序后的状态，我们要把它放回 sort_indices 指定的位置
        output.index_copy_(0, metadata["sort_indices"], received_x)

        # 3. 恢复形状
        # [N*K, D] -> [N, K, D]
        output = output.view(metadata["original_batch_size"], metadata["topk"], -1)

        return output
