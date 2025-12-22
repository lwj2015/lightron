import os
import torch
import torch.distributed as dist

from parallel.distributed import get_device_mesh

_STEP = 0
_VERBOSE = os.environ.get("VERBOSE", "0") == "1"


class PPGroupManager:
    """
    Lightron PP group helper, modeled after picotron.process_group_manager fields.
    """
    def __init__(self):
        mesh = get_device_mesh()

        if mesh is None or "pp" not in mesh.mesh_dim_names:
            self.pp_group = None
            self.pp_world_size = 1
            self.pp_rank = 0
            self.pp_is_first_stage = True
            self.pp_is_last_stage = True
            self.pp_prev_rank = None
            self.pp_next_rank = None
        else:
            self.pp_group = mesh["pp"].get_group()
            self.pp_world_size = dist.get_world_size(group=self.pp_group)
            self.pp_rank = dist.get_rank(group=self.pp_group)
            self.pp_is_first_stage = (self.pp_rank == 0)
            self.pp_is_last_stage = (self.pp_rank == self.pp_world_size - 1)

            # 计算上一个 Stage 的相对 Rank
            if not self.pp_is_first_stage:
                prev_relative_rank = self.pp_rank - 1
                # 转换为全局 Rank
                self.pp_prev_rank = dist.get_global_rank(self.pp_group, prev_relative_rank)
            else:
                self.pp_prev_rank = None
            # 计算下一个 Stage 的相对 Rank
            if not self.pp_is_last_stage:
                next_relative_rank = self.pp_rank + 1
                # 转换为全局 Rank
                self.pp_next_rank = dist.get_global_rank(self.pp_group, next_relative_rank)
            else:
                self.pp_next_rank = None
            
            # Debug 打印，确认转换是否正确
            print(f"[Debug Rank {dist.get_global_rank(self.pp_group, self.pp_rank)}] PP Group: {self.pp_rank}/{self.pp_world_size}, "
                f"Prev Global: {self.pp_prev_rank}, Next Global: {self.pp_next_rank}", flush=True)


_ppm: PPGroupManager = None


def init_pp_group_manager():
    """
    在 setup_distributed 之后调用，真正地初始化 PPGroupManager。
    """
    global _ppm
    if _ppm is None:
        _ppm = PPGroupManager()
        if dist.get_rank() == 0:
            print(f"✅ PPGroupManager initialized: pp_world_size={_ppm.pp_world_size}", flush=True)


def get_pp_group_manager():
    """
    获取 PPGroupManager 实例。
    确保它已经被初始化。
    """
    global _ppm
    assert _ppm is not None, \
        "PPGroupManager not initialized. Call init_pp_group_manager() after setup_distributed()."
    return _ppm


def pipeline_communicate(operation: str, device, dtype, tensor=None, shapes=None):
    """
    Picotron-style unidirectional PP communication.
    operation ∈ {'recv_forward','send_forward','recv_backward','send_backward'}

    Notes:
    - uses async isend/irecv via dist.batch_isend_irecv
    - creates recv buffers with requires_grad=True (same as picotron)
    - communication uses pp_group derived from DeviceMesh["pp"]
    """
    global _STEP, _VERBOSE, _ppm

    if _ppm.pp_group is None or _ppm.pp_world_size == 1:
        # no pipeline; behave as no-op
        return None if operation.startswith("recv") else None

    if operation == "recv_forward":
        if _ppm.pp_is_first_stage:
            return None
        assert shapes is not None, "recv_forward requires shapes"
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = _ppm.pp_prev_rank

    elif operation == "send_forward":
        if _ppm.pp_is_last_stage:
            return None
        assert tensor is not None, "send_forward requires tensor"
        dest = _ppm.pp_next_rank

    elif operation == "recv_backward":
        if _ppm.pp_is_last_stage:
            return None
        assert shapes is not None, "recv_backward requires shapes"
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = _ppm.pp_next_rank

    elif operation == "send_backward":
        if _ppm.pp_is_first_stage:
            return None
        assert tensor is not None, "send_backward requires tensor"
        dest = _ppm.pp_prev_rank

    else:
        raise ValueError(f"Unknown operation: {operation}")

    is_send = operation.startswith("send")
    peer_rank = dest if is_send else src
    p2p_fn = dist.isend if is_send else dist.irecv
    op = dist.P2POp(p2p_fn, tensor, peer_rank, group=_ppm.pp_group)
    # op = dist.P2POp(p2p_fn, tensor, peer_rank)

    if _VERBOSE:
        direction = "→" if is_send else "←"
        what = operation.split("_")[1]
        print(
            f"{operation} | {'sending' if is_send else 'receiving'} {what} "
            f"{_ppm.pp_rank} {direction} {peer_rank} | STEP:{_STEP}",
            flush=True,
        )

    reqs = dist.batch_isend_irecv([op])
    for r in reqs:
        r.wait()
    torch.cuda.synchronize()

    if _VERBOSE:
        _STEP += 1

    return tensor if not is_send else None


def bidirectional_pipeline_communicate(operation: str, send_tensor, recv_shapes, device, dtype):
    """
    Picotron-style bidirectional PP communication.
    operation ∈ {'send_fwd_recv_bwd', 'send_bwd_recv_fwd'}

    - send_fwd_recv_bwd: send activation to next stage and receive grad from next stage
    - send_bwd_recv_fwd: send grad to prev stage and receive next activation from prev stage
    """
    global _STEP, _VERBOSE, _ppm

    if _ppm.pp_group is None or _ppm.pp_world_size == 1:
        return None

    is_fwd = (operation == "send_fwd_recv_bwd")
    if is_fwd:
        if _ppm.pp_is_last_stage:
            return None
        peer_rank = _ppm.pp_next_rank
    else:
        if _ppm.pp_is_first_stage:
            return None
        peer_rank = _ppm.pp_prev_rank

    assert send_tensor is not None, \
        f"[PP Error] {operation} failed on rank {_ppm.pp_rank}. " \
        f"I am trying to send a gradient to my prev stage, but my calculated gradient is None! " \
        f"This means the computation graph is broken."

    assert send_tensor is not None, f"{operation} requires send_tensor"
    assert recv_shapes is not None, f"{operation} requires recv_shapes"

    recv_tensor = torch.empty(recv_shapes, requires_grad=True, device=device, dtype=dtype)

    ops = [
        dist.P2POp(dist.isend, send_tensor, peer_rank, group=_ppm.pp_group),
        dist.P2POp(dist.irecv, recv_tensor, peer_rank, group=_ppm.pp_group),
    ]

    if _VERBOSE:
        print(
            f"{operation} | sending {'next' if is_fwd else 'prev'} {_ppm.pp_rank} -> {peer_rank} | "
            f"receiving {'next' if is_fwd else 'prev'} {peer_rank} -> {_ppm.pp_rank} | STEP:{_STEP}",
            flush=True,
        )

    reqs = dist.batch_isend_irecv(ops)
    for r in reqs:
        r.wait()
    torch.cuda.synchronize()

    if _VERBOSE:
        _STEP += 1

    return recv_tensor

