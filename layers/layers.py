import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # 简化的 RoPE 预计算
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    freqs_cis: [S, D]
    x: [B, S, H, D]
    Target:    [1, S, 1, D]
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), \
        f"freqs_cis shape {freqs_cis.shape} does not match x shape {(x.shape[1], x.shape[-1])}"
    
    # 构造广播形状: [d if i==1 or i==ndim-1 else 1]
    # 对于 4D 输入 x [B, S, H, D]，这将生成 [1, S, 1, D]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    # 将 Q, K 转为复数进行旋转
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # === 添加调试日志 ===
    # import torch.distributed as dist
    # if dist.is_initialized() and dist.get_rank() == 0:
    #     print(f"[Debug Rank 0] Inside apply_rotary_emb:")
    #     print(f"  xq_ (complex): {xq_.shape}")
    #     print(f"  xk_ (complex): {xk_.shape}")
    #     print(f"  freqs_cis (raw): {freqs_cis.shape}")
    # ===================

    # freqs_cis = freqs_cis[:xq.shape[1]] # 切片匹配 seq_len

    # 尝试广播
    # freqs_cis 需要 reshape 成 [1, S, 1, head_dim/2] 才能跟 xq_ [B, S, n_heads, head_dim/2] 相乘
    # 这里的 reshape 逻辑是关键
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # if dist.is_initialized() and dist.get_rank() == 0:
    #     print(f"  freqs_cis (broadcast): {freqs_cis.shape}")

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
