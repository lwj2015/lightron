import functools
import torch
import torch.nn as nn
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
    enable_wrap,
    wrap,
)
# FSDP2
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from .distributed import get_device_mesh


def apply_fsdp1(
        model,
        transformer_layer_cls,
        use_bf16=True,
        strategy="full",
        device_mesh=None,
        use_activation_checkpointing=False
):
    """
    Classic FSDP (Wrapper based) - 生产级增强版
    """
    # 1. 自动包裹策略 (Auto Wrap Policy)
    # 这是一个关键点：除了 TransformerBlock，有时我们也希望根据参数量包裹其他大层
    if transformer_layer_cls:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
        )
    else:
        # 回退策略：如果没指定层类，按参数量切分 (例如 > 10M 的层)
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1e7
        )

    # 2. 混合精度策略 (Mixed Precision)
    # 生产环境建议：reduce_dtype 保持 float32 以防止梯度下溢 (Underflow)
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    ) if use_bf16 else None

    # 3. 分片策略 (Sharding Strategy)
    # 处理 HSDP 的特殊逻辑
    if strategy == "hybrid":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
        # HSDP 必须提供 device_mesh，且必须是 2D 的 (Replicate, Shard)
        # 如果传入的 mesh 是 1D 的，这里会报错，需要做检查
        if device_mesh is None or device_mesh.ndim != 2:
            raise ValueError("HSDP requires a 2D DeviceMesh (Replicate, Shard).")
    elif strategy == "grad_op":
        sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD

    # 4. 初始化 FSDP
    # sync_module_states=True: 极其重要！确保所有 Rank 的随机初始化参数在开始前强制同步一致。
    # limit_all_gathers=True: 节省显存，防止在前向传播时同时 gather 太多层。
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device(),
        device_mesh=device_mesh,
        sync_module_states=True,  # [改进点] 强制同步参数
        limit_all_gathers=True,  # [改进点] 显存优化
        use_orig_params=True  # [改进点] 允许 torch.compile 优化
    )

    # 5. Activation Checkpointing (梯度检查点)
    # FSDP1 需要在 wrap 之后应用
    if use_activation_checkpointing and transformer_layer_cls:
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        check_fn = lambda submodule: isinstance(submodule, transformer_layer_cls)
        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=check_fn,
        )

    return fsdp_model


def apply_fsdp2(
        model,
        transformer_layer_cls=None,
        use_activation_checkpointing=False
):
    """
    FSDP2 (Composable / DTensor based) - 生产级增强版
    """
    mesh = get_device_mesh()
    if mesh is None:
        raise ValueError("Device Mesh must be initialized for FSDP2")

    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32
    )

    # FSDP2 的核心逻辑：自底向上应用 fully_shard

    # 1. 对 Transformer Block 应用 fully_shard
    # reshard_after_forward=True: 类似于 FSDP1 的 FULL_SHARD，算完就释放显存
    if transformer_layer_cls:
        for module in model.modules():
            if isinstance(module, transformer_layer_cls):
                fully_shard(
                    module=module,
                    mesh=mesh,
                    mp_policy=mp_policy,
                    reshard_after_forward=True
                )

                # [改进点] FSDP2 的 AC 集成
                # 在 fully_shard 之后应用 AC
                if use_activation_checkpointing:
                    checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

    # 2. 对整个模型应用 fully_shard
    # 这会处理剩下的 Embedding、Output Head 等层
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=True
    )

    return model
