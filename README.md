
# ⚡ Custom C++ Transformer Inference Engine (Full Stack)

![Language](https://img.shields.io/badge/language-C++%20%7C%20Python-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![Focus](https://img.shields.io/badge/focus-Systems%20Engineering-red?style=for-the-badge)

> **A framework-free, CPU-only Transformer runtime built from scratch.**
> Takes an English prompt from a Web UI, processes it via Python, and executes inference in pure C++.

---

## 📖 Project Overview

This project demonstrates a **Systems Engineering approach to AI**. Instead of relying on heavy frameworks like PyTorch or ONNX for inference, I built a custom inference engine in **C++** that reads raw binary weights and executes the Transformer forward pass manually.

Key Engineering Features:
* Zero-Copy Loading: Uses `mmap` to map 30M parameters into virtual memory instantly.
* Manual Kernels: Hand-written Matrix Multiplication, LayerNorm, Softmax, and GELU implementations.
* Full Stack Integration: A `FastAPI` wrapper bridges the high-performance C++ backend with a modern HTML frontend.

---

## 🏗️ Architecture

The system follows a strict separation of concerns: **Python for Transport, C++ for Compute.**

```
graph TD
    User[Frontend (HTML/JS)] -->|1. English Prompt| API[Python FastAPI Server]
    API -->|2. Tokenize| Tokens[Token IDs]
    
    subgraph "High-Performance Engine"
    Tokens -->|3. Subprocess Call| CPP{C++ Inference Engine}
    CPP -->|mmap| Weights[model.bin (30M Params)]
    CPP -->|Compute| Output[Generated SQL]
    end
    
    Output -->|4. Return Result| API
    API -->|5. Display| User

```
---
## 🧩 Model Specifications

- Property,Value
- Parameters,~30 Million
- Layers / Heads,6 Layers / 6 Heads
- Embedding Dim,384
- Context Window,256 Tokens
- Inference Speed,~15ms/token (CPU)
- Binary Size,~120 MB

---
## 📂 Project Structure

```

.
├── inference.cpp       # The Core Engine (Memory mgmt, MatMul, Attention)
├── server.py           # The Bridge (FastAPI, Tokenizer, Subprocess)
├── index.html          # The Frontend (Simple UI)
├── model.bin           # The Brain (Raw FP32 Weights)
├── tokenizer.bin       # The Dictionary (Vocabulary)
└── README.md           # Documentation

```



## ⚙️ Technical Deep Dive

1. Memory Management (mmap)


Standard file I/O (fread) copies data from disk to kernel space to user space. This engine uses mmap (Memory Mapping) to map the model.bin file directly into the process's virtual address space.
- Benefit: Instant startup time (<5ms) regardless of model size.

- OS Level: Leverages the OS page cache for efficient memory usage.

2. Custom Arithmetic Kernels
No BLAS or LAPACK libraries were used. All math is implemented from scratch to understand the low-level operations.
- MatMul: Naive implementation ($O(N^3)$) optimized with row-major weight layout.
- GELU: Approximate implementation using tanh.
- Softmax: Numerically stable implementation (subtracting max value).
---

## 🚀 How to Run (Local Setup)

### Prerequisites
- C++ Compiler: g++ or clang++

- Python 3: with fastapi, uvicorn, tiktoken


### Step 1: Compile the Engine
```
g++ -O3 inference.cpp -o inference
```


### Step 2: Start the Backend Server
```
pip install fastapi uvicorn tiktoken
uvicorn server:app --reload

```
### Step 3: Launch the Frontend
```
Simply double-click index.html to open it in your browser.

1. Type a query: "Get all users from Pune"

2. Click Generate

3. Watch the C++ Engine generate the SQL!

```

## 🔮 Future Improvements
- SIMD (AVX2): Vectorize the matrix multiplication loop for 4x-8x speedup.

- Quantization: Implement int8 weight quantization to reduce memory usage by 75%.

- Multithreading: Use OpenMP to parallelize Attention heads.

## 📜 License
MIT License. Built for educational and systems engineering demonstration.



