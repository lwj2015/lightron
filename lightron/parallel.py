import functools
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)


def get_fsdp_wrapper(model, transformer_layer_cls, use_bf16=True, strategy="full"):
    """
    配置并返回 FSDP 包装后的模型
    """
    # 1. 自动包裹策略：按层切分
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_cls},
    )

    # 2. 混合精度策略
    if use_bf16 and torch.cuda.is_bf16_supported():
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,  # 梯度累加保持 FP32 以稳定训练
            buffer_dtype=torch.float32,
        )
    else:
        mp_policy = None  # 默认 FP32

    # 3. 切分策略
    if strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD  # ZeRO-3
    elif strategy == "grad_op":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP  # ZeRO-2
    elif strategy == "hybrid":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD  # HSDP
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD

    # 4. 包装模型
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,  # 限制同时进行的 all-gather 数量，防止 OOM
    )
    return fsdp_model


def save_fsdp_checkpoint(fsdp_model, rank, path):
    """
    FSDP 模型保存逻辑：将所有分片聚合为完整权重保存
    注意：这非常消耗内存，建议只在 rank 0 上做，或者使用 CPU offload
    """
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_policy):
        cpu_state = fsdp_model.state_dict()

    if rank == 0:
        torch.save(cpu_state, path)
        print(f"Model checkpoint saved to {path}")


def load_fsdp_checkpoint(fsdp_model, path):
    """
    FSDP 模型加载逻辑
    """
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        state_dict = torch.load(path)
    else:
        state_dict = None

    # FSDP 会自动处理 scatter，将完整的 state_dict 切分给各个 rank
    # 注意：这需要所有 rank 都运行
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
        fsdp_model.load_state_dict(state_dict)
