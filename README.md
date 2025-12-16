# Lightron ⚡️

**Lightron** is a lightweight, educational, yet modern distributed training framework for LLMs. 
Lightron aims to bridge the gap between minimal implementations and modern production features like **FSDP**, **FlashAttention-2**, and **Llama-3 architectures**.

## 🚀 Key Features
- **Modern Architecture**: RMSNorm, SwiGLU, Rotary Embeddings (RoPE).
- **Efficiency**: Native PyTorch `scaled_dot_product_attention` (FlashAttention-2).
- **Distributed Ready**: First-class support for PyTorch FSDP (Fully Sharded Data Parallel).
- **Clean Code**: Type-hinted, dataclass-based configuration, <1000 lines of core code.

## 🛠️ Installation
```bash
git clone https://github.com/lwj2015/lightron.git
cd lightron
pip install -r requirements.txt
```

## 🏃 Quick Start
```bash
# Run on 4 GPUs with FSDP
torchrun --nproc_per_node=4 examples/train_llama.py
```

## citation

 If you use Lightron in your research or learning journey, please cite it as follows:
```bash
  @misc{lightron2024,
  author = {Wenjun Liu},
  title = {Lightron: A Modern Minimalist Distributed Training Framework},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lwj2015/lightron}}
}
```


