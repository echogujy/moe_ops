# triton-gmm-ops

Pure-Triton implementations of Mixture-of-Experts (MoE) operator primitives: **grouped GEMM**, **token permute / unpermute**, and **fused top-k softmax**. No C++/CUDA extensions required — every kernel is a Triton `@triton.jit` kernel.

## Layout conventions

All ops share the stacked MoE layout:

| tensor      | shape                                           | meaning                                                                       |
| ----------- | ----------------------------------------------- | ----------------------------------------------------------------------------- |
| `A`         | `[total_tokens, K]`                             | activations, all experts stacked along M                                      |
| `B`         | `[E, N, K]` if `trans_b` else `[E, K, N]`       | per-expert weights                                                            |
| `offsets`   | `[E]`                                           | cumulative group ends (`offsets[e]` = end row of expert `e`; starts at 0)     |
| `C`         | `[total_tokens, N]`                             | stacked output                                                                |

`bf16` and `fp16` are supported (bf16 is the MoE default).

## Operators

### Grouped GEMM (`grouped_gemm.py`)

- `grouped_gemm(A, B, offsets, trans_b=True) -> C` — Differentiable wrapper (`torch.autograd.Function`) employing an optimized **3D Grid SM Layout** to avoid SM-CTA persistence bottlenecks. Backward pass computes `grad_A` and `grad_B` via parallel Triton GMM kernels.

### Routing permute / unpermute (`permute.py`, `unpermute.py`)

- `permute(input, indices, num_out_tokens=0, E=None) -> (permuted, row_id_map, base)` — Differentiable wrapper employing a high-throughput **counting sort** algorithm (uses block-local hardware `tl.histogram` and exclusive prefix sums) to group token rows by expert.
- `unpermute(input, row_id_map, prob, num_tokens, num_topK, max_tokens=-1) -> out` — Recovers original token order using the inverted gather map and scales by routing probabilities.
- `permute_backward(grad_permuted, row_id_map, num_tokens, topK) -> grad_input` — Computes permute backward pass using scatter-add.

### Fused Top-K Softmax (`fused_topk_softmax.py`)

- `fused_topk_softmax(logits, K) -> (weights, indices)` — Fuses top-K selection and softmax normalization entirely in GPU registers/SRAM, avoiding intermediate global memory roundtrips. Backward pass computes exact softmax gradients and scatters them back to input gradients.

---

## Repository Layout

```
triton_gmm_ops/
├── triton_gmm_ops/          # importable package
│   ├── __init__.py           # Unified exposed public APIs
│   ├── grouped_gemm.py       # Primary 3D Grid Grouped GEMM
│   ├── permute.py            # Primary Counting-Sort Permute
│   ├── unpermute.py          # Primary Unpermute
│   └── fused_topk_softmax.py # Fused Top-K Softmax
├── tests/
│   ├── test_grouped_gemm.py  # Tests vs PyTorch Native
│   ├── test_permute.py       # Tests vs PyTorch Native
│   ├── test_unpermute.py    # Tests vs PyTorch Reference
│   └── test_topk_softmax.py  # Tests vs PyTorch Reference
└── develop/
    ├── experiment/           # TMA / Fused pipeline experiments
    └── deprecated/           # Argsort permute / Persistent-SM GMM
```

## Requirements

- Python >= 3.10
- PyTorch (CUDA build) + Triton (both come with `pip install torch` on Linux CUDA).
- A CUDA GPU (sm80+).

## Install

The package lives flat at the repo root (`triton_gmm_ops/`), so it installs with either `uv` or `pip` against the repo's own `pyproject.toml`.

```bash
cd triton-gmm-ops
pip install -e .
```

After install, `import triton_gmm_ops` works from anywhere.

## Running Tests

All tests verify correctness and benchmark performance against PyTorch Native:

```bash
python -m tests.test_grouped_gemm
python -m tests.test_permute
python -m tests.test_unpermute
python -m tests.test_topk_softmax
```
