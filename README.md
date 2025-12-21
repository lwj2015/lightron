# Lightron

**Lightron** is a lightweight, educational, yet modern distributed training framework for LLMs. 
Lightron aims to bridge the gap between minimal implementations and modern production features such as **4-D Parallelism**, including **Tensor Parallelism, Pipeline Parallelism, Data Parallelism**, and **Context Parallelism**.

# Key Features

- **Distributed Ready**: Support 4-D Parallelism(TP, PP, DP, CP), EP and FSDP V2.
- **Modern Architecture**: RMSNorm, SwiGLU, Rotary Embeddings (RoPE), FlashAttention V2.
- **Clean Code**: Type-hinted, dataclass-based configuration, <1000 lines of core code.

# Installation
```bash
git clone https://github.com/lwj2015/lightron.git
cd lightron
pip install -r requirements.txt
```

# Quick Start
```bash
torchrun --nproc_per_node=8 --master_port=19501 trainer.py --config examples/config_tinystories.json
```

## Test All Reduce Communication on local device
```bash
torchrun --nproc_per_node=8 --master_port=12555 parallel/communication/all_reduce.py
```

## Test Ring Attention on local device
```bash
python parallel/communication/ring_attention.py 
```

# Citation

 If you use Lightron in your research or learning journey, please cite it as follows:
```bash
  @misc{lightron2025,
  author = {Wenjun Liu},
  title = {Lightron: A Modern Minimalist Distributed Training Framework},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lwj2015/lightron}}
}
```


