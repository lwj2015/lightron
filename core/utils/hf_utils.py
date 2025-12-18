import torch
import os
from core.model.model import LightronTransformer
from core.config.config import ModelArgs


def load_hf_llama_weights(model: LightronTransformer, hf_model_path: str):
    """
    从 HuggingFace Llama 加载权重到 Lightron。
    需要处理 key 的映射 (例如: model.layers.0.self_attn.q_proj -> layers.0.attention.wq)
    """
    from transformers import AutoModelForCausalLM
    print(f"Loading HF weights from {hf_model_path}...")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_path, torch_dtype=torch.bfloat16)
    hf_sd = hf_model.state_dict()

    my_sd = model.state_dict()

    # 简单的映射规则示例 (需要根据实际层名调整)
    mapping = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight"
    }

    # 遍历转换
    for hf_key, hf_val in hf_sd.items():
        # 1. 处理基础层
        if hf_key in mapping:
            my_sd[mapping[hf_key]].copy_(hf_val)
            continue

        # 2. 处理 Block 层
        # HF: model.layers.0.self_attn.q_proj.weight
        # My: layers.0.attention.wq.weight
        if "layers" in hf_key:
            key_parts = hf_key.split(".")
            layer_idx = key_parts[2]

            new_key = None
            if "self_attn.q_proj" in hf_key:
                new_key = f"layers.{layer_idx}.attention.wq.weight"
            elif "self_attn.k_proj" in hf_key:
                new_key = f"layers.{layer_idx}.attention.wk.weight"
            elif "self_attn.v_proj" in hf_key:
                new_key = f"layers.{layer_idx}.attention.wv.weight"
            elif "self_attn.o_proj" in hf_key:
                new_key = f"layers.{layer_idx}.attention.wo.weight"
            elif "mlp.gate_proj" in hf_key:
                new_key = f"layers.{layer_idx}.feed_forward.w1.weight"
            elif "mlp.down_proj" in hf_key:
                new_key = f"layers.{layer_idx}.feed_forward.w2.weight"
            elif "mlp.up_proj" in hf_key:
                new_key = f"layers.{layer_idx}.feed_forward.w3.weight"
            elif "input_layernorm" in hf_key:
                new_key = f"layers.{layer_idx}.attention_norm.weight"
            elif "post_attention_layernorm" in hf_key:
                new_key = f"layers.{layer_idx}.ffn_norm.weight"

            if new_key and new_key in my_sd:
                my_sd[new_key].copy_(hf_val)

    model.load_state_dict(my_sd)
    print("Weights loaded successfully!")
