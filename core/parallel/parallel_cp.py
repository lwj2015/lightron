import torch
import torch.nn as nn
import torch.distributed as dist
from .distributed import get_device_mesh


def _all_to_all(input_, scatter_dim, gather_dim, group):
    """
    All-to-All 通信原语 (基于 Ulysses 算法的核心)
    作用：将张量在 scatter_dim 上切分，发送给不同 rank，
          同时接收其他 rank 的数据，在 gather_dim 上拼接。
    """
    # 1. 基础检查
    if group is None:
        return input_

    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return input_

    # 2. 预处理：确保输入连续
    input_ = input_.contiguous()

    # 3. 准备输入切片 (Split)
    # input shape: [..., scatter_dim_size, ...]
    # chunks shape: world_size * [..., scatter_dim_size/P, ...]
    input_chunks = list(input_.chunk(world_size, dim=scatter_dim))

    # 4. 准备输出缓冲区
    # 输出形状与输入切片形状相同（假设切分均匀）
    output_chunks = [torch.empty_like(chunk) for chunk in input_chunks]

    # 5. 执行 All-to-All 通信
    # 这是一个同步操作
    dist.all_to_all(output_chunks, input_chunks, group=group)

    # 6. 后处理：拼接 (Concat)
    # output shape: [..., gather_dim_size * P, ...]
    return torch.cat(output_chunks, dim=gather_dim)


class ContextParallelAttention(nn.Module):
    """
    CP Attention Wrapper (DeepSpeed Ulysses Style)

    该模块用于包裹标准的 Attention 计算（如 FlashAttention）。
    它负责在计算前后进行数据的“转置”：

    流程：
    1. 输入: [Batch, Seq/P, Heads, Dim] (序列被切分)
    2. 通信: All-to-All (Seq -> Heads)
    3. 中间: [Batch, Seq, Heads/P, Dim] (现在拥有完整的 Seq，但只有部分的 Heads)
    4. 计算: Local Attention (因为有完整 Seq，所以可以算 Attention Score)
    5. 通信: All-to-All (Heads -> Seq)
    6. 输出: [Batch, Seq/P, Heads, Dim] (恢复为序列切分)
    """

    def __init__(self, local_attn_module, args):
        """
        Args:
            local_attn_module: 原始的 Attention 模块 (lightron.model.Attention)
            args: ModelArgs 配置，用于获取 head 数量等信息
        """
        super().__init__()
        self.local_attn = local_attn_module
        self.num_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        # 获取 CP 通信组
        self.mesh = get_device_mesh()
        # 检查 mesh 中是否有 'cp' 维度，如果没有则回退到 None (不做 CP)
        if self.mesh and "cp" in self.mesh.mesh_dim_names:
            self.cp_group = self.mesh["cp"].get_group()
        else:
            self.cp_group = None

    def forward(self, x, freqs_cis):
        """
        x: 输入 Tensor，通常是 LayerNorm 后的结果
           Shape: [Batch, Seq_Local, Hidden_Dim]
           注意：这里的 Seq_Local = Seq_Global / CP_Size
        """
        # 如果没有开启 CP，直接调用原始 Attention
        if self.cp_group is None:
            return self.local_attn(x, freqs_cis)

        # === 1. 准备阶段 ===
        B, S_local, Hidden = x.shape
        cp_size = dist.get_world_size(group=self.cp_group)

        # 此时 x 是 [B, S/P, H_total * D]
        # 我们需要先把它 reshape 成 [B, S/P, H_total, D]
        # 注意：这里假设 local_attn 内部的 wq, wk, wv 是线性的，
        # 为了适配 Ulysses，我们需要侵入到 Attention 内部，或者要求 Attention 的输入已经是 QKV。
        # **修正**：Lightron 的 Attention 模块内部包含了 WQ/WK/WV 投影。
        # Ulysses 通常要求投影后的 QKV 进行 All-to-All。
        # 为了不重写整个 Attention 类，我们采用一种更高级的策略：
        # 我们让 local_attn 正常计算 QKV 投影，但在 FlashAttention 之前拦截它。

        # 但由于 Python 无法直接拦截中间变量，我们需要修改 lightron/model.py 中的 Attention。
        # 为了让这个 Wrapper 生效，我们假设 self.local_attn 已经被修改为支持
        # 接收 "pre-computed QKV" 或者我们在这里手动执行投影。

        # 【方案 B：最稳健的实现】
        # 我们不 Wrap 整个 Attention，而是 Wrap "Attention 计算部分"。
        # 但为了符合你要求的代码结构，我们假设这个 forward 是替代原 Attention.forward 的。

        # 1. 投影 (Local Projection)
        # x: [B, S/P, Hidden]
        xq, xk, xv = self.local_attn.wq(x), self.local_attn.wk(x), self.local_attn.wv(x)

        # Reshape to heads
        # xq: [B, S/P, H_total, D]
        xq = xq.view(B, S_local, self.num_heads, self.head_dim)
        xk = xk.view(B, S_local, self.num_heads, self.head_dim)  # 暂不考虑 GQA/MQA 的复杂情况，假设 MHA
        xv = xv.view(B, S_local, self.num_heads, self.head_dim)

        # === 2. 第一次 All-to-All (Seq -> Head) ===
        # 目标: [B, S_global, H_local, D]
        # 操作: Scatter dim 1 (Seq), Gather dim 2 (Head)

        xq = _all_to_all(xq, scatter_dim=2, gather_dim=1, group=self.cp_group)
        xk = _all_to_all(xk, scatter_dim=2, gather_dim=1, group=self.cp_group)
        xv = _all_to_all(xv, scatter_dim=2, gather_dim=1, group=self.cp_group)

        # 现在的形状: [B, S_global, H_total/P, D]
        # 此时我们拥有了完整的 Sequence，但只有部分的 Heads。

        # === 3. RoPE (需要完整 Seq) ===
        # 因为现在 S 是完整的，所以可以直接应用 RoPE
        # 注意：freqs_cis 需要匹配 S_global
        # 如果传入的 freqs_cis 是切片过的，这里可能需要调整，假设传入的是完整的或自动广播的
        xq, xk = self.local_attn.apply_rotary_emb_custom(xq, xk, freqs_cis)

        # === 4. Local Attention (FlashAttn) ===
        # PyTorch 的 scaled_dot_product_attention 接受 [B, H, S, D]
        # 我们需要转置一下
        output = torch.nn.functional.scaled_dot_product_attention(
            xq.transpose(1, 2),  # [B, H/P, S_global, D]
            xk.transpose(1, 2),
            xv.transpose(1, 2),
            is_causal=True
        )
        # output: [B, H/P, S_global, D]
        output = output.transpose(1, 2).contiguous()  # [B, S_global, H/P, D]

        # === 5. 第二次 All-to-All (Head -> Seq) ===
        # 目标: [B, S_local, H_total, D]
        # 操作: Scatter dim 1 (Seq), Gather dim 2 (Head) -> 也就是逆操作

        # 注意：刚才 _all_to_all 是把 dim 2 切了拼到 dim 1
        # 现在我们要把 dim 1 切了拼到 dim 2
        output = _all_to_all(output, scatter_dim=1, gather_dim=2, group=self.cp_group)

        # 现在的形状: [B, S_local, H_total, D]

        # === 6. 输出投影 ===
        output = output.flatten(2)  # [B, S_local, Hidden]
        return self.local_attn.wo(output)
