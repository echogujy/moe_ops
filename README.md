# triton-gmm-ops

Pure-Triton implementations of Mixture-of-Experts (MoE) operator primitives: **grouped GEMM (with SM90 TMA support)**, **token permute / unpermute**, and **fused top-k softmax**. No C++/CUDA extensions required — every kernel is a Triton `@triton.jit` kernel.

## Layout conventions

All ops share the stacked MoE layout:

| tensor      | shape                                           | meaning                                                                       |
| ----------- | ----------------------------------------------- | ----------------------------------------------------------------------------- |
| `A`       | `[total_tokens, K]`                           | activations, all experts stacked along M                                      |
| `B`       | `[E, N, K]` if `trans_b` else `[E, K, N]` | per-expert weights                                                            |
| `offsets` | `[E + 1]` or `[E]`                             | cumulative group ends (`offsets[e]` = end row of expert `e`; starts with 0 for `E + 1`) |
| `C`       | `[total_tokens, N]`                           | stacked output                                                                |

`bf16` and `fp16` are supported (bf16 is the MoE default).

## Operators

### Grouped GEMM (`grouped_gemm_ops.py`, `grouped_gemm_ops_sm90.py`)

- `grouped_gemm(A, B, offsets, trans_b=True) -> C` — Differentiable wrapper (`torch.autograd.Function`) supporting automatic architecture dispatch.
  - **SM90+ (Hopper)**: Dispatched to the TMA-accelerated grouped GEMM kernel using asynchronous bulk copy hardware (`tl.make_tensor_descriptor`) with built-in hardware zero-fill out-of-bounds boundary handling, avoiding CPU/GPU formatting overhead.
  - **SM80 (Ampere/Lovelace)**: Dispatched to the standard 3D Grid grouped GEMM kernel employing an optimized **3D Grid SM Layout** to avoid SM-CTA persistence bottlenecks.
  - Backward pass computes `grad_A` via parallel Triton GMM kernels (with TMA on Hopper) and `grad_B` via Triton GMM B-gradient kernel.

### Routing permute / unpermute (`permute_ops.py`, `unpermute_ops.py`)

- `permute(input, indices, num_out_tokens=0, E=None) -> (permuted, row_id_map, base)` — Differentiable wrapper employing a high-throughput **counting sort** algorithm (uses block-local hardware `tl.histogram` and exclusive prefix sums) to group token rows by expert.
- `unpermute(input, row_id_map, prob, num_tokens, num_topK, max_tokens=-1) -> out` — Recovers original token order using the inverted gather map and scales by routing probabilities.
- `permute_backward(grad_permuted, row_id_map, num_tokens, topK) -> grad_input` — Computes permute backward pass using scatter-add.

### Fused Top-K Softmax (`fused_topk_softmax_ops.py`)

- `fused_topk_softmax(logits, K, fp32_routing=False) -> (weights, indices)` — Fuses top-K selection and softmax normalization entirely in GPU registers/SRAM, avoiding intermediate global memory roundtrips. Backward pass computes exact softmax gradients and scatters them back to input gradients.
  - **`fp32_routing` (bool)**: If `True`, forces routing weight calculations and outputs to be done in high-precision `float32` regardless of activation dtype (e.g. `bfloat16`/`float16`). This prevents underflow/overflow training instability commonly observed in deep MoE models.

---

## Repository Layout

```
triton_gmm_ops/
├── triton_gmm_ops/              # importable package
│   ├── __init__.py               # Unified exposed public APIs
│   ├── grouped_gemm_ops.py       # Standard 3D Grid Grouped GEMM
│   ├── grouped_gemm_ops_sm90.py  # Hopper TMA Grouped GEMM & Dispatcher
│   ├── permute_ops.py            # Primary Counting-Sort Permute
│   ├── unpermute_ops.py          # Primary Unpermute
│   └── fused_topk_softmax_ops.py # Fused Top-K Softmax
├── tests/
│   ├── test_grouped_gemm.py      # Tests vs PyTorch Native
│   ├── test_permute.py           # Tests vs PyTorch Native
│   ├── test_unpermute.py         # Tests vs PyTorch Reference
│   └── test_topk_softmax.py      # Tests vs PyTorch Reference
└── develop/
    ├── experiment/               # TMA / Fused pipeline experiments
    └── deprecated/               # Argsort permute / Persistent-SM GMM
```

## Requirements

- Python >= 3.10
- PyTorch (CUDA build) + Triton
- A CUDA GPU (sm80+).

## Install

The package lives flat at the repo root (`triton_gmm_ops/`), so it installs with `pip` against the repo's own `pyproject.toml`.

```bash
cd moe_ops
pip install -e .
```

---

## Performance Benchmarks

Below is a performance comparison between the **NVIDIA Hopper GPU (GH200 480GB)** and the **NVIDIA Ada GPU (RTX 500 Ada Generation)**.

### 1. Hopper GPU (NVIDIA GH200 480GB, SM90)
*TMA path is enabled for bfloat16.*

#### Grouped GEMM Correctness & Performance (bfloat16):
- **Forward Path**:
  - `4096x4096x16x4096` : Triton GMM **0.278 ms** | PyTorch Native **0.273 ms** (0.98x)
  - `8192x4096x16x4096` : Triton GMM **0.471 ms** | PyTorch Native **0.466 ms** (0.99x)
  - `16384x4096x16x4096`: Triton GMM **0.861 ms** | PyTorch Native **0.869 ms** (1.01x)
  - `16384x8192x16x8192`: Triton GMM **3.685 ms** | PyTorch Native **3.318 ms** (0.90x)
- **Backward Path**:
  - `4096x4096x16x4096` : Triton GMM **0.605 ms** | PyTorch Native **0.657 ms** (1.09x)
  - `8192x4096x16x4096` : Triton GMM **0.982 ms** | PyTorch Native **1.028 ms** (1.05x)
  - `16384x4096x16x4096`: Triton GMM **1.772 ms** | PyTorch Native **1.710 ms** (0.97x)
  - `16384x8192x16x8192`: Triton GMM **7.423 ms** | PyTorch Native **6.674 ms** (0.90x)

#### Full MoE Layer Integration (`8192T d=4096 E=64 K=2 H=11008`):
- **Triton Forward**: **10.93 ms** (Breakdown: `router=0.096ms`, `permute=0.253ms`, `3xgemm=7.989ms`, `unpermute=1.202ms`)
- **Triton Backward**: **22.17 ms**
- **Overall Triton Fwd+Bwd**: **33.10 ms** (GEMM backward: `31.18ms` vs PyTorch Native `28.32ms`, 0.91x)

---

### 2. Ada GPU (NVIDIA RTX 500 Ada, Laptop GPU, SM89)
*Standard 3D grid path is active. Benchmarked against a sequential loop fallback for grouped GEMM.*

#### Grouped GEMM Correctness & Performance (bfloat16):
- **Forward Path**:
  - `4096x4096x16x4096` : Triton GMM **11.239 ms** | PyTorch Native (loop) **10.901 ms** (0.97x)
  - `8192x4096x16x4096` : Triton GMM **15.852 ms** | PyTorch Native (loop) **34.808 ms** (2.20x)
  - `16384x4096x16x4096`: Triton GMM **737.540 ms** | PyTorch Native (loop) **753.800 ms** (1.02x)
  - `16384x8192x16x8192`: Triton GMM **1535.994 ms** | PyTorch Native (loop) **2909.487 ms** (1.89x)
- **Backward Path**:
  - `4096x4096x16x4096` : Triton GMM **18.905 ms** | PyTorch Native (loop) **570.357 ms** (30.17x)
  - `8192x4096x16x4096` : Triton GMM **537.299 ms** | PyTorch Native (loop) **814.068 ms** (1.52x)
  - `16384x4096x16x4096`: Triton GMM **1766.246 ms** | PyTorch Native (loop) **1517.618 ms** (0.86x)
  - `16384x8192x16x8192`: Triton GMM **1698.232 ms** | PyTorch Native (loop) **22535.053 ms** (13.27x)

#### Full MoE Layer Integration (`4096T d=2048 E=16 K=2 H=5120`):
- **Triton Forward**: **36.39 ms** (Breakdown: `router=0.266ms`, `permute=1.072ms`, `3xgemm=22.949ms`, `unpermute=4.232ms`)
- **Triton Backward**: **69.97 ms**
- **Overall Triton Fwd+Bwd**: **106.36 ms** (GEMM backward: `90.13ms` vs PyTorch Native `2208.08ms`, **24.50x**)

---

## Running Tests

All tests verify correctness and benchmark performance against PyTorch Native/fallback:

```bash
python tests/test_grouped_gemm.py
python tests/test_permute.py
python tests/test_unpermute.py
python tests/test_topk_softmax.py
python tests/test_moe.py
```
