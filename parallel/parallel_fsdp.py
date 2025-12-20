import functools
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# FSDP2
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from .distributed import get_device_mesh


def apply_fsdp1(model, transformer_layer_cls, use_bf16=True, strategy="full", device_mesh=None):
    """
    Classic FSDP (Wrapper based)
    """
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_cls},
    )

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    ) if use_bf16 else None

    # HSDP Logic: 如果提供了 device_mesh 且是 2D 的 (Replicate, Shard)
    # FSDP1 的 HSDP 需要 process_group，这里简化处理
    sharding_strategy = ShardingStrategy.FULL_SHARD
    if strategy == "hybrid":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif strategy == "grad_op":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device(),
        device_mesh=device_mesh  # PyTorch 2.2+ 支持直接传 DeviceMesh
    )


def apply_fsdp2(model, transformer_layer_cls=None):
    """
    FSDP2 (Composable / DTensor based)
    """
    mesh = get_device_mesh()
    if mesh is None:
        raise ValueError("Device Mesh must be initialized for FSDP2")

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    # FSDP2 推荐做法：手动对每一层应用 fully_shard
    # 这样可以精确控制切分粒度
    for module in model.modules():
        if transformer_layer_cls and isinstance(module, transformer_layer_cls):
            fully_shard(module, mesh=mesh, policy=mp_policy)

    # 最后对整个模型应用，处理剩余参数（如 Embedding, Output Head）
    fully_shard(model, mesh=mesh, policy=mp_policy)
    return model
