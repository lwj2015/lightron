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
        else:
            self.pp_group = mesh["pp"].get_group()
            self.pp_world_size = dist.get_world_size(group=self.pp_group)
            self.pp_rank = dist.get_rank(group=self.pp_group)

        self.pp_is_first_stage = (self.pp_rank == 0)
        self.pp_is_last_stage = (self.pp_rank == self.pp_world_size - 1)
        self.pp_prev_rank = self.pp_rank - 1 if not self.pp_is_first_stage else None
        self.pp_next_rank = self.pp_rank + 1 if not self.pp_is_last_stage else None


_ppm = PPGroupManager()


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


def get_pp_group_manager():
    # 供外部查询 pp_rank / is_first / is_last 等
    return _ppm
