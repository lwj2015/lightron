import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

# 全局 Mesh 管理器
_DEVICE_MESH = None
_MESH_DIMS = {}


def setup_distributed(
        tp_size: int = 1,
        pp_size: int = 1,
        cp_size: int = 1,
        ep_size: int = 1,
        dp_size: int = 1,
):
    """
    初始化 5D 并行环境。

    层级结构 (Hierarchy):
    1. PP (Pipeline): 最外层，通常跨机。
    2. DP (Data): 数据并行层。
       注意：EP (Expert) 通常是 DP 的一种变体。
       - 如果 ep_size == 1: 纯 DP，所有 DP rank 拥有相同的 MoE 参数。
       - 如果 ep_size == dp_size: 纯 EP，所有 DP rank 拥有不同的专家。
       - 如果 1 < ep_size < dp_size: 混合模式 (Hybrid EP)。
    3. CP (Context): 上下文并行，切分 Sequence。
    4. TP (Tensor): 最内层，切分算子，通常在单机 NVLink 范围内。
    """
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 1. 校验 World Size
    # 注意：EP 不增加总 World Size，它是寄生在 DP 维度上的
    # 总卡数 = PP * DP * CP * TP
    expected_world_size = pp_size * dp_size * cp_size * tp_size

    if world_size != expected_world_size:
        raise ValueError(
            f"World Size Mismatch! Real: {world_size}, "
            f"Configured: {pp_size}(PP) * {dp_size}(DP) * {cp_size}(CP) * {tp_size}(TP) = {expected_world_size}"
        )

    # 2. 校验 EP 合法性
    # EP 是在 DP 组内进行的，所以 ep_size 必须能整除 dp_size
    if dp_size % ep_size != 0:
        raise ValueError(f"DP size ({dp_size}) must be divisible by EP size ({ep_size})")

    # 3. 构建 Device Mesh
    # 维度顺序：(PP, DP, CP, TP)
    mesh_dims = []
    mesh_names = []

    if pp_size > 1:
        mesh_dims.append(pp_size)
        mesh_names.append("pp")

    if dp_size > 1:
        mesh_dims.append(dp_size)
        mesh_names.append("dp")

    if cp_size > 1:
        mesh_dims.append(cp_size)
        mesh_names.append("cp")

    if tp_size > 1:
        mesh_dims.append(tp_size)
        mesh_names.append("tp")

    global _DEVICE_MESH, _MESH_DIMS

    if len(mesh_dims) > 0:
        _DEVICE_MESH = init_device_mesh("cuda", tuple(mesh_dims), mesh_dim_names=tuple(mesh_names))
    else:
        # 单卡模式
        _DEVICE_MESH = init_device_mesh("cuda", (1,), mesh_dim_names=("dp",))

    # 4. 存储配置供后续查询
    _MESH_DIMS = {
        "tp": tp_size,
        "pp": pp_size,
        "cp": cp_size,
        "ep": ep_size,
        "dp": dp_size
    }

    if rank == 0:
        print(f"🚀 Distributed Init Success!")
        print(f"   Shape: PP={pp_size} | DP={dp_size} (EP={ep_size}) | CP={cp_size} | TP={tp_size}")
        print(f"   Mesh: {mesh_names}")


def get_device_mesh():
    return _DEVICE_MESH


def get_parallel_info():
    return _MESH_DIMS


# === 获取各个维度的 Process Group ===

def get_tp_group():
    return _DEVICE_MESH["tp"].get_group() if "tp" in _DEVICE_MESH.mesh_dim_names else None


def get_cp_group():
    return _DEVICE_MESH["cp"].get_group() if "cp" in _DEVICE_MESH.mesh_dim_names else None


def get_pp_group():
    return _DEVICE_MESH["pp"].get_group() if "pp" in _DEVICE_MESH.mesh_dim_names else None


def get_dp_group():
    # 纯 DP 组 (用于同步非 MoE 参数)
    return _DEVICE_MESH["dp"].get_group() if "dp" in _DEVICE_MESH.mesh_dim_names else None


def get_ep_group():
    """
    获取 EP 通信组。
    EP 比较特殊，它是在 DP 维度上切分的。
    如果 ep_size == dp_size，那么 EP group 就是 DP group。
    如果 ep_size < dp_size，我们需要在 DP group 内部再切分。
    (为了简化，这里假设 ep_size == dp_size，即标准 MoE)
    """
    return get_dp_group()
