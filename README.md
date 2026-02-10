# Custom Transformer Inference Engine (C++ / CPU)

This project implements a **framework-free, CPU-only transformer inference engine in C++**, capable of executing end-to-end autoregressive inference (from model weights to token generation) without relying on PyTorch, TensorFlow, or external ML libraries.

The goal of this project is **systems-level understanding of transformer inference**, not model accuracy or dataset performance.

---

## Project Motivation

Modern ML workflows often hide inference internals behind high-level frameworks and GPUs.  
This project was built to:

- Understand how transformer inference works **at runtime**
- Control memory layout, numerical computation, and decoding logic
- Execute a trained transformer model **without Python or ML frameworks**
- Explore production-style concerns such as startup overhead and runtime isolation

---

## High-Level Architecture
```
            ┌──────────────┐
            │   Frontend   │
            │  (Optional)  │
            └──────┬───────┘
                   │
                   ▼
            ┌──────────────┐
            │   FastAPI    │
            │ (Transport) │
            └──────┬───────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│ C++ Inference Engine │
│ - mmap model.bin │
│ - Transformer forward pass │
│ - Autoregressive decoding │
└────────────────────────────────────────┘
```




- **C++** owns all inference logic
- **Python/FastAPI** is used only as a thin transport layer
- No GPU or ML framework is used during inference

---

## Model Overview

- Architecture: GPT-style Transformer
- Layers: 6
- Attention Heads: 6
- Embedding Dimension: 384
- Parameters: ~25M
- Context Length: 256 tokens
- Training Data: ~78K text-to-SQL samples
- Training Framework: PyTorch (training only)

The trained model learns **text-to-SQL translation**, not general-purpose question answering.

---

## Repository Structure
```

.
├── training/
│ ├── train.py # PyTorch training code
│ ├── export_weights.py # Exports model to custom binary format
│
├── inference/
│ ├── inference.cpp # Standalone C++ inference engine
│ ├── model.bin # Serialized model weights
│
├── api/
│ ├── server.py # Minimal FastAPI transport layer
│
├── README.md
```



---

## Weight Serialization

After training, model parameters are exported from PyTorch into a **custom binary format**.

Key design choices:
- Flat FP32 tensors
- Fixed layout for deterministic loading
- Single `model.bin` file
- Read-only **memory-mapped loading (mmap)** in C++

This avoids:
- Python dependencies at inference time
- Repeated memory copies
- Long model initialization delays

---

## C++ Inference Engine

The C++ runtime performs:

- Memory-mapped loading of model weights
- Token and positional embedding lookup
- Multi-head self-attention
- LayerNorm and GELU activations
- Feedforward (MLP) layers
- Autoregressive decoding loop (greedy decoding)

All numerical operations are implemented manually using standard C++.

No external math, ML, or GPU libraries are used.

---

## Correctness Validation

Inference correctness was validated by:

- Running greedy (deterministic) decoding
- Comparing generated token sequences against PyTorch inference
- Verifying **token-level equivalence**, accounting for floating-point differences between GPU and CPU

Bit-wise floating point equality is **not expected or required**.

---

## Running the C++ Inference Engine

### Build
```bash
g++ -O2 inference.cpp -o inference
./inference "<TOKENIZED_PROMPT_OR_TEXT>"
```


The program outputs generated token IDs or decoded text depending on configuration.

FastAPI Integration (Optional)

A minimal FastAPI server is provided to expose the C++ inference engine over HTTP.

Important design rule:

FastAPI handles transport only.
All inference remains inside the C++ process.

Python is intentionally kept out of the performance-critical path.

What This Project Is NOT

❌ A chatbot

❌ A production-ready LLM

❌ An accuracy-optimized ML benchmark

❌ A frontend-heavy application

What This Project Demonstrates

Systems-level understanding of transformer inference

Manual implementation of core neural network primitives

Memory-efficient model loading using mmap

Clear separation between compute and serving layers

Ability to translate ML theory into low-level engineering

Future Improvements

SIMD / AVX optimizations for matrix multiplication

Multi-threaded inference

FP16 weight support

Tokenizer implementation in C++

Batching support

Key Takeaway

The primary goal of this project is learning how transformers actually run, not building a high-level ML application.

This project intentionally prioritizes engineering depth over model complexity.



---





