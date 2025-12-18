import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

# 全局 Device Mesh
_DEVICE_MESH = None


def setup_distributed(
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        cp_size: int = 1,
        dp_size: int = -1  # -1 means auto-calculated
):
    """
    初始化分布式环境并构建多维 Device Mesh
    支持的维度顺序: (PP, DP, TP) 或更多
    这里采用通用的 2D/3D/4D Mesh 构建逻辑
    """
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 自动计算 DP size
    # Total = PP * EP * CP * TP * DP
    known_size = tp_size * pp_size * ep_size * cp_size
    if dp_size == -1:
        if world_size % known_size != 0:
            raise ValueError(f"World size {world_size} not divisible by parallel dims {known_size}")
        dp_size = world_size // known_size

    assert world_size == dp_size * tp_size * pp_size * ep_size * cp_size, \
        f"World size mismatch: {world_size} != {dp_size}*{tp_size}*{pp_size}*{ep_size}*{cp_size}"

    # 构建 Mesh 维度名称和大小
    # 这里的顺序决定了物理设备的映射，通常建议：
    # PP (最外层，跨机) -> DP (跨机/单机) -> TP (单机 NVLink)
    # 为了简化，我们先实现最常用的 (DP, TP) 或 (PP, DP, TP)

    mesh_dims = []
    mesh_names = []

    if pp_size > 1:
        mesh_dims.append(pp_size)
        mesh_names.append("pp")

    # DP 通常包含 FSDP/DDP/EP/CP 的混合语义，这里简化为 "dp"
    # 如果要支持 CP/EP，通常是在 DP 维度上再切分，或者独立维度
    # 这里为了通用性，我们将剩余维度统称为 "dp_replicate" (用于 FSDP/DDP)
    if dp_size > 1:
        mesh_dims.append(dp_size)
        mesh_names.append("dp")

    if tp_size > 1:
        mesh_dims.append(tp_size)
        mesh_names.append("tp")

    global _DEVICE_MESH
    if len(mesh_dims) > 0:
        _DEVICE_MESH = init_device_mesh("cuda", tuple(mesh_dims), mesh_names=tuple(mesh_names))
    else:
        # 单卡情况
        _DEVICE_MESH = init_device_mesh("cuda", (1,), mesh_names=("dp",))

    if rank == 0:
        print(f"Distributed Init: World={world_size}, Mesh={mesh_names}:{mesh_dims}")


def get_device_mesh():
    return _DEVICE_MESH


def get_tp_group():
    return _DEVICE_MESH["tp"].get_group() if "tp" in _DEVICE_MESH.mesh_dim_names else None


def get_dp_group():
    return _DEVICE_MESH["dp"].get_group() if "dp" in _DEVICE_MESH.mesh_dim_names else None


def get_pp_group():
    return _DEVICE_MESH["pp"].get_group() if "pp" in _DEVICE_MESH.mesh_dim_names else None
