import torch
import torch.nn as nn
import torch.nn.functional as F

from parallel.communication.pipeline_parallel_p2p import (
    pipeline_communicate,
    bidirectional_pipeline_communicate,
    get_pp_group_manager,
)

import torch.distributed as dist


class PipelineParallel(nn.Module):
    """
    Picotron-style pipeline wrapper for LightronTransformer.

    Expected base model fields (LightronTransformer):
      - tok_embeddings (nn.Embedding)
      - layers (nn.ModuleList of TransformerBlock)
      - norm (nn.Module)
      - output (nn.Linear)
      - freqs_cis (buffer for RoPE)  # 你已经修过“不要过早切片”
    """

    def __init__(self, model: nn.Module, model_args):
        super().__init__()
        self.ppm = get_pp_group_manager()
        self.model_args = model_args  # ModelArgs（含 dim/n_layers/max_seq_len等）

        # 1) layer 分配（模仿 picotron.distribute_layers）
        self.layer_distribution = self.distribute_layers(model_args.n_layers)

        # 2) 只在 first stage 保留 embedding；否则 Identity
        self.tok_embeddings = model.tok_embeddings if self.ppm.pp_is_first_stage else nn.Identity()

        # 3) 只保留本 stage 的 transformer blocks（引用原层，避免复制参数）
        self.layers = nn.ModuleDict({str(i): model.layers[i] for i in self.layer_distribution})

        # 4) 只在 last stage 保留 norm+output；否则 Identity
        self.norm = model.norm if self.ppm.pp_is_last_stage else nn.Identity()
        self.output = model.output if self.ppm.pp_is_last_stage else nn.Identity()

        # 5) RoPE cache：建议每个 stage 都持有一份（引用即可）
        #    这样中间 stage 不需要重新 precompute（否则慢、且容易 shape 不一致）
        self.freqs_cis = getattr(model, "freqs_cis", None)

    def reset_parameters(self):
        """
        可选：如果你依赖 meta-init + reset_parameters，这里可以做分 stage reset。
        你当前 trainer 里是 model.apply(init_weights)，如果包了 PP，需要改成对 PP wrapper apply。
        """
        if self.ppm.pp_is_first_stage and hasattr(self.tok_embeddings, "reset_parameters"):
            self.tok_embeddings.reset_parameters()

        for layer in self.layers.values():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        if self.ppm.pp_is_last_stage:
            if hasattr(self.norm, "reset_parameters"):
                self.norm.reset_parameters()
            if hasattr(self.output, "reset_parameters"):
                self.output.reset_parameters()

    def distribute_layers(self, num_layers: int):
        """
        Evenly distribute layers across PP stages (same as picotron).
        """
        pp = self.ppm.pp_world_size
        r = self.ppm.pp_rank
        layers_per_stage = [
            num_layers // pp + (1 if i < (num_layers % pp) else 0)
            for i in range(pp)
        ]
        start = sum(layers_per_stage[:r])
        return list(range(start, start + layers_per_stage[r]))

    def forward(self, input_ids=None, hidden_states=None):
        """
        Picotron-style stage forward:
          - first stage consumes input_ids -> embeddings
          - other stages consume hidden_states
          - last stage outputs logits [B, S, V]
          - middle stages output hidden [B, S, H]
        """
        x = hidden_states if hidden_states is not None else input_ids
        x = self.tok_embeddings(x)

        # RoPE cache：沿用你修复后的策略，传“完整 freqs_cis”
        freqs_cis = self.freqs_cis

        for layer in self.layers.values():
            # 你的 block.forward 形参是 (x, freqs_cis)
            x = layer(x, freqs_cis)

        x = self.norm(x)
        x = self.output(x)
        return x

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        """
        Picotron-style stage backward:
          - input_tensor: activation received from prev stage (or None on first stage)
          - output_tensor: activation produced by this stage (or loss scalar on last stage)
          - output_tensor_grad: grad received from next stage (or None on last stage)
        """
        if input_tensor is not None:
            input_tensor.retain_grad()

        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)

        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)

        """
        print(f"global_rank={dist.get_rank()} pp_rank={self.ppm.pp_rank}")
        if (not self.ppm.pp_is_first_stage) and (input_tensor is not None):
            if not hasattr(self, "_printed_grad"):
                self._printed_grad = True
                g = input_tensor.grad
                gn = None if g is None else float(g.norm().detach().cpu())
                print(f"[PP][grad_check] pp_rank={self.ppm.pp_rank} input_grad_norm={gn}", flush=True)
        else:
            print(f"self.ppm.pp_rank: {self.ppm.pp_rank}, self.ppm.pp_is_first_stage: {self.ppm.pp_is_first_stage}, input_tensor: {input_tensor}")
        """

        return input_tensor.grad if input_tensor is not None else None


def train_step_pipeline_afab(model: PipelineParallel, data_loader, tensor_shapes, device, dtype):
    """
    All-Forward-All-Backward (AFAB) schedule, Picotron-style.

    Requirements:
      - data_loader yields dict with: input_ids, target_ids
      - Only last stage computes CE loss
      - tensor_shapes should match activation/grad communicated between stages
    """
    ppm = model.ppm
    logging_loss = 0.0
    input_tensors, output_tensors = [], []

    # 你后面如果把 DP/CP 梯度同步做成类似 picotron 的 bucket，
    # 可以在这里加 requires_grad_sync + require_backward_grad_sync 控制。
    requires_grad_sync = getattr(model, "require_backward_grad_sync", None) is not None

    # --- Forward all microbatches ---
    for _ in range(data_loader.grad_acc_steps):
        input_tensor = pipeline_communicate("recv_forward", shapes=tensor_shapes, device=device, dtype=dtype)

        batch = next(data_loader)
        if ppm.pp_is_first_stage:
            out = model.forward(input_ids=batch["input_ids"].to(device), hidden_states=None)
        else:
            out = model.forward(input_ids=None, hidden_states=input_tensor.to(device))

        pipeline_communicate("send_forward", tensor=out, device=device, dtype=dtype)

        if ppm.pp_is_last_stage:
            # out: logits [B,S,V], target_ids: [B,S]
            loss = F.cross_entropy(out.flatten(0, 1), batch["target_ids"].to(device).flatten(), reduction="mean")
            loss = loss / data_loader.grad_acc_steps
            logging_loss += loss.item()
            out = loss  # 让 backward 以 scalar loss 为起点

        input_tensors.append(input_tensor)
        output_tensors.append(out)

    # --- Backward all microbatches ---
    for ith_microbatch in range(data_loader.grad_acc_steps):
        if requires_grad_sync:
            model.require_backward_grad_sync = (ith_microbatch == data_loader.grad_acc_steps - 1)

        output_grad = pipeline_communicate("recv_backward", shapes=tensor_shapes, device=device, dtype=dtype)

        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)

        input_grad = model.backward(input_tensor, output_tensor, output_grad)
        pipeline_communicate("send_backward", tensor=input_grad, device=device, dtype=dtype)

    return logging_loss


def train_step_pipeline_1f1b(model: PipelineParallel, data_loader, tensor_shapes, device, dtype):
    """
    1F1B schedule, Picotron-style.
    """
    ppm = model.ppm

    num_warmup = min(ppm.pp_world_size - ppm.pp_rank - 1, data_loader.grad_acc_steps)
    num_remaining = data_loader.grad_acc_steps - num_warmup

    logging_loss = 0.0
    input_tensors, output_tensors = [], []

    requires_grad_sync = getattr(model, "require_backward_grad_sync", None) is not None

    def _forward_step(input_tensor):
        batch = next(data_loader)

        if ppm.pp_is_first_stage:
            out = model.forward(input_ids=batch["input_ids"].to(device), hidden_states=None)
        else:
            out = model.forward(input_ids=None, hidden_states=input_tensor.to(device))

        if ppm.pp_is_last_stage:
            loss = F.cross_entropy(out.flatten(0, 1), batch["target_ids"].to(device).flatten(), reduction="mean")
            loss = loss / data_loader.grad_acc_steps
            nonlocal logging_loss
            logging_loss += loss.item()
            return loss  # backward 从 scalar loss 开始

        return out

    # --- Warmup: only forward ---
    for _ in range(num_warmup):
        inp = pipeline_communicate("recv_forward", shapes=tensor_shapes, device=device, dtype=dtype)
        out = _forward_step(inp)
        pipeline_communicate("send_forward", tensor=out, device=device, dtype=dtype)
        input_tensors.append(inp)
        output_tensors.append(out)

    # --- Steady state: 1F1B ---
    # print(f"debug, num_remaining: {num_remaining}")
    if num_remaining > 0:
        input_tensor = pipeline_communicate("recv_forward", shapes=tensor_shapes, device=device, dtype=dtype)
        # print(f"debug, input_tensor: {input_tensor}, input_tensor.requires_grad: {input_tensor.requires_grad}")
        if input_tensor is not None and input_tensor.requires_grad is False:
            print(f"[FATAL] Rank {dist.get_rank()} received input_tensor but requires_grad is False! Graph will break!", flush=True)
            input_tensor.requires_grad_(True) # 尝试强制修复

    if requires_grad_sync:
        model.require_backward_grad_sync = False
    
    # print(f"[Debug None Tensor {dist.get_global_rank(ppm.pp_group, ppm.pp_rank)}] PP Group: {ppm.pp_rank}/{ppm.pp_world_size}")

    for i in range(num_remaining):
        is_last_iter = (i == num_remaining - 1)

        out = _forward_step(input_tensor)

        out_grad = bidirectional_pipeline_communicate(
            operation="send_fwd_recv_bwd",
            send_tensor=out,
            recv_shapes=tensor_shapes,
            device=device,
            dtype=dtype,
        )

        input_tensors.append(input_tensor)
        output_tensors.append(out)

        # FIFO：取最早的进行 backward
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)

        if num_warmup == 0 and is_last_iter and requires_grad_sync:
            model.require_backward_grad_sync = True

        in_grad = model.backward(input_tensor, output_tensor, out_grad)

        if is_last_iter:
            input_tensor = None
            pipeline_communicate("send_backward", tensor=in_grad, device=device, dtype=dtype)
        else:
            input_tensor = bidirectional_pipeline_communicate(
                operation="send_bwd_recv_fwd",
                send_tensor=in_grad,
                recv_shapes=tensor_shapes,
                device=device,
                dtype=dtype,
            )

    # --- Cooldown: remaining backward ---
    for i in range(num_warmup):
        if requires_grad_sync:
            model.require_backward_grad_sync = (i == num_warmup - 1)

        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)

        out_grad = pipeline_communicate("recv_backward", shapes=tensor_shapes, device=device, dtype=dtype)
        in_grad = model.backward(input_tensor, output_tensor, out_grad)
        # print(f"model.ppm.pp_is_first_stage: {model.ppm.pp_is_first_stage}, in_grad: {in_grad}")
        if not model.ppm.pp_is_first_stage and in_grad is None:
            print(f"[FATAL] Rank {dist.get_rank()} (Stage {model.ppm.pp_rank}) backward produced None grad! "
                f"Input tensor requires_grad: {input_tensor.requires_grad if input_tensor is not None else 'None'}. "
                f"Output tensor grad fn: {output_tensor.grad_fn if output_tensor is not None else 'None'}", flush=True)
        pipeline_communicate("send_backward", tensor=in_grad, device=device, dtype=dtype)

    return logging_loss
