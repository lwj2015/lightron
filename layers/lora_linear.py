import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.05, bias=False):
        super().__init__()
        # 冻结的主权重
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False

        # LoRA 分支
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / rank

        # 初始化
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        # Result = Wx + (BAx * scale)
        result = self.linear(x)
        lora_out = self.lora_b(self.lora_a(self.dropout(x))) * self.scaling
        return result + lora_out
