"""Pure-Triton MoE operator library.

Stitched ops for Mixture-of-Experts routing + grouped GEMM, implemented
entirely in Triton (no C++/CUDA extension).

Public API
----------
Grouped GEMM (stacked, autograd-aware)
    group_gemm_fn(A, B, offsets, trans_b=True) -> C      # forward kernel (sm80+)
    grouped_gemm(A, B, offsets, trans_b=True)  -> C      # same, differentiable
    group_gemm_tma_fn(...)                             # TMA variant (sm90+)
    supports_tma()                                    # compute-capability gate

Routing permute / unpermute
    permute(input, indices, ...)                      # argsort path
    permute_countsort(input, indices, ...)            # counting-sort path (faster)
    permute_backward(grad_permuted, row_id_map, ...)  # == unpermute(prob=1)
    permute_autograd / permute_countsort_autograd     # differentiable wrappers
    unpermute(input, row_id_map, prob, ...)           # recover token order (diff)
    unpermute_forward / unpermute_backward            # raw kernels

Fused kernels
    fused_permute_gemm(X, W, base, sorted_id_map, ...)   # permute fused into GEMM
    fused_gemm_unpermute(X_perm, W, ..., prob, ...)      # GEMM fused into unpermute

Layout conventions (see each module's docstring for details)
    A:        [total_tokens, K]   (all experts stacked along M)
    B:        [E, N, K] if trans_b else [E, K, N]   (per-expert weights)
    offsets:  [E]  cumulative group ends (offsets[e] = end row of expert e)
    C:        [total_tokens, N]

bf16 and fp16 are supported; bf16 is the MoE default.
"""
from .grouped_gemm_ops import grouped_gemm
from .permute_ops import permute, permute_backward
from .unpermute_ops import unpermute, unpermute_forward, unpermute_backward
from .fused_topk_softmax_ops import fused_topk_softmax

__all__ = [
    "grouped_gemm",
    "permute",
    "permute_backward",
    "unpermute",
    "unpermute_forward",
    "unpermute_backward",
    "fused_topk_softmax",
]
