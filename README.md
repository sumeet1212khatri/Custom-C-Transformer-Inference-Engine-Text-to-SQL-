# Custom Transformer Inference Engine (C++ / CPU)

![Language](https://img.shields.io/badge/language-C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)


A framework-free, CPU-only transformer inference runtime written in C++, executing end-to-end autoregressive generation (weights → logits → tokens) without PyTorch, TensorFlow, or GPU dependencies.

The goal of this project is systems-level understanding of transformer inference, not model benchmarking.

---

## Why This Project Exists

Modern ML stacks abstract away inference details behind high-level frameworks and GPUs.

This project removes those abstractions and re-implements transformer inference at the systems level to understand:

- Memory layout and weight loading
- Numerical computation of attention and MLP blocks
- Autoregressive decoding
- Separation of compute and serving layers

---

## Architecture

 
```
Frontend (optional)
          ↓
FastAPI (transport only)
          ↓
C++ Inference Engine

   - mmap(model.bin)           
                         
   - Transformer forward pass  
                          
   - Autoregressive decoding   

``` 
### SLM tranning is done in google Colab colab link
```

https://colab.research.google.com/drive/1fkrAFCv32xtX1ralzYKAw522NoN0kzGn?usp=sharing

```
Data Set is from Hugging Face 
```

https://huggingface.co/datasets/PurpleAILAB/chatML_SQL_injection_dataset

```
This SLM GitHub Repo
```

https://github.com/sumeet1212khatri/SQL-Injection-GPT

```

Design principles:

- C++ owns all inference logic
- Python is strictly transport
- No external ML libraries at inference time
- No GPU dependency

---

## Model Configuration

- GPT-style Transformer
- 6 layers
- 6 attention heads
- 384 embedding dimension
- ~25M parameters
- Context length: 256
- Training dataset: ~78K text-to-SQL samples
- Training framework: PyTorch (training only)

The model performs **text-to-SQL translation**, not general-purpose question answering.

---

## Core Engineering Components

### 1. Custom Weight Serialization

- Model exported into a flat binary format (`model.bin`)
- Deterministic parameter layout
- FP32 storage
- Single-file deployment

### 2. Memory-Mapped Model Loading

This design enables constant-time startup independent of parameter parsing and avoids heap fragmentation from repeated allocations.

- Read-only `mmap` loading
- Avoids redundant memory copies
- Low startup overhead
- Clean separation between storage and compute


### 3. C++ Transformer Runtime

Attention computation follows scaled dot-product formulation with explicit causal masking implemented at the tensor level.

Implemented from scratch:

- Token & positional embeddings
- Multi-head self-attention
- LayerNorm
- GELU activation
- Feedforward (MLP) blocks
- Autoregressive decoding (greedy)

All numerical kernels implemented manually in C++.

No BLAS, no LibTorch, no Eigen.

---

## Correctness Validation

Inference correctness verified by:

- Deterministic greedy decoding
- Token-level comparison against PyTorch inference
- Accepting floating-point differences between CPU and GPU

Bit-wise equality is not expected.

---

## Repository Structure

```


.
├── training/
│ ├── train.py
│ ├── export_weights.py
│
├── inference/
│ ├── inference.cpp
│ ├── model.bin
│
├── api/
│ ├── server.py
│
└── README.md



```


---

## Build & Run

```bash
g++ -O2 inference.cpp -o inference
./inference "<PROMPT>"
```

