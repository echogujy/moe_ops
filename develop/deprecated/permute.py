"""Triton implementation of grouped_gemm's ``moe_permute_topK``.

Two specialized strategies are provided (no single default — pick per use case):

  * ``permute``          — argsort path. Groups tokens by expert via a non-stable
                           torch.argsort; one Triton kernel gathers the activations,
                           builds row_id_map, and records each expert's first/last
                           occurrence (-> counts/base, atomic-free on the offsets).
                           Order within an expert is irrelevant to the GEMM/unpermute.
                           Trade-off: pays for one extra host-side torch.argsort; on
                           A800 it is the SLOWER Triton path (~1.4-1.7x vs C++).
  * ``permute_countsort`` — counting sort. One fused kernel does the histogram
                           (block-local ``tl.histogram``) + exclusive prefix-sum
                           (``tl.cumsum``) together, then a second atomic-prefix-rank
                           scatter emits the grouped layout with no general sort. This
                           is the FASTER Triton path on A800 (~1.1-1.2x vs C++), and it
                           emits base (offset) as a byproduct of the fused setup kernel
                           (no counts atomic_add). Caveat: per-expert atomic_add
                           serializes and the scatter write (permuted[dest]) is
                           non-coalesced, so it still trails the C++ backend (the
                           fastest of the three).

Semantics (for both): given ``indices[t, k]`` = expert id of token t's k-th slot,
produce a per-expert grouped layout:
  permuted[dest[j]] = input[j // topK]        where j = t*topK + k
  row_id_map[k*num_tokens + t] = dest[j]       (inverse mapping for unpermute)

Both return ``(permuted, row_id_map, base)`` where
  base[e] = exclusive prefix sum of group sizes  (grouped-GEMM group offsets; the
           ``offs`` arg of ``F.grouped_mm`` and the atomic-counter seed for the
           countsort scatter). counts[e] is derivable as base[e] - base[e-1]
           (base[0] == 0), which is all the C++ grouped_gemm backend (counts consumer)
           needs.
Only base is emitted: the two are mutually redundant and base is what the countsort
scatter already consumes, so the setup kernel drops the counts atomic_add. For
``permute_countsort`` base comes from the fused histogram + prefix-sum setup kernel
(replacing torch.bincount/cumsum); for ``permute`` (argsort) it is derived on the host
from each expert's first/last occurrence recorded by the scatter kernel. ``E``
(max expert id + 1) may be passed in to skip the host-side max reduction; it must
satisfy max(indices) < E.

ponytail: supports the standard no-capacity-drop case (num_out_tokens == 0
→ num_tokens*topK). Capacity dropping (num_out_tokens < num_tokens*topK) is not
handled here because our MoE never uses it.
"""
import torch
import triton
import triton.language as tl

from .unpermute import unpermute_forward  # permute backward == unpermute(prob=1)


# ----------------------------------------------------------------------------
# Custom group histogram + exclusive prefix sum.
# Fused setup that replaces torch.bincount/cumsum/max, and emits the exact
# grouped-GEMM metadata (group sizes + offsets) consumed right after permute.
# ----------------------------------------------------------------------------

@triton.jit
def _group_offsets_kernel(flat_ptr, base_ptr, num_total: tl.constexpr,
                          E: tl.constexpr, BLOCK: tl.constexpr, stride_flat):
    """Fused histogram + exclusive prefix-sum in ONE kernel.

    Each block bins its BLOCK elements with the hardware histogram (tl.histogram,
    no atomics inside the block), turns the block-local histogram into its
    exclusive prefix sum via tl.cumsum, then atomically accumulates BOTH the
    block histogram (-> global counts[e]) and the block exclusive prefix sum
    (-> global base[e] = sum_{i<e} counts[i]) into the outputs.

    Requires E <= BLOCK so the [E] histogram fits one block vector.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    idx = pid * BLOCK + offs
    m = idx < num_total
    v = tl.load(flat_ptr + idx * stride_flat, mask=m).to(tl.int32)
    h = tl.histogram(v, E, mask=m)          # [E] block-local histogram
    inc = tl.cumsum(h, axis=0)              # [E] inclusive block-local prefix sum
    exc = inc - h                           # [E] exclusive block-local prefix sum
    bins = tl.arange(0, E)
    tl.atomic_add(base_ptr + bins, exc)     # -> global base[e] (exclusive prefix sum)


def _group_offsets(flat: torch.Tensor, E: int):
    """Custom-kernel (fused) histogram + prefix sum -> base (exclusive offsets)."""
    base = torch.zeros(E, dtype=torch.int32, device=flat.device)
    num_total = flat.shape[0]
    HBLOCK = 2048                           # ponytail: block size; also E <= HBLOCK
    _group_offsets_kernel[(num_total + HBLOCK - 1) // HBLOCK,](
        flat, base, num_total, E, HBLOCK, flat.stride(0)) # type: ignore
    return base


# ----------------------------------------------------------------------------
# Counting sort (faster Triton path; trails the C++ backend)
# ----------------------------------------------------------------------------

@triton.jit
def _permute_countsort_kernel(
    flat_ptr,                         # [num_total] int32 expert ids
    input_ptr,                       # [num_tokens, num_cols]
    out_ptr,                         # [num_out, num_cols]
    row_id_map_ptr,                  # [num_total] int32, init -1
    counter_ptr,                     # [E] int32 atomic buffer (init = base[e])
    num_tokens, num_cols, num_out, topK: tl.constexpr,
    stride_flat,
    stride_in_r, stride_in_c,
    stride_out_r, stride_out_c,
    BLOCK_C: tl.constexpr,
):
    j = tl.program_id(0)
    e = tl.load(flat_ptr + j * stride_flat).to(tl.int64)
    # atomic rank: returns base[e] + current_rank == destination output row.
    dest = tl.atomic_add(counter_ptr + e, 1).to(tl.int64)
    if dest < num_out:              # capacity drop -> skip (not used in our MoE)
        t = j // topK
        k = j % topK
        # scatter: permuted[dest] = input[t]
        for c0 in range(tl.cdiv(num_cols, BLOCK_C)):
            cols = c0 * BLOCK_C + tl.arange(0, BLOCK_C)
            mask = cols < num_cols
            vals = tl.load(input_ptr + t * stride_in_r + cols * stride_in_c, mask=mask)
            tl.store(out_ptr + dest * stride_out_r + cols * stride_out_c, vals, mask=mask)
        tl.store(row_id_map_ptr + (k * num_tokens + t), dest.to(tl.int32))


def permute_countsort(input: torch.Tensor, indices: torch.Tensor,
                      num_out_tokens: int = 0, E: int | None = None):
    num_tokens, num_cols = input.shape
    topK = indices.shape[1]
    flat = indices.reshape(-1).contiguous()       # [num_tokens*topK] int32 expert ids
    num_total = num_tokens * topK
    num_out = num_out_tokens if num_out_tokens > 0 else num_total
    if E is None:
        E = int(flat.max()) + 1                    # ponytail: pass E to skip this reduction

    base = _group_offsets(flat, E)
    counter = base.clone()                          # atomic buffer; base kept clean for output

    permuted = torch.empty(num_out, num_cols, device=input.device, dtype=input.dtype)
    row_id_map = torch.full((num_total,), -1, dtype=torch.int32, device=input.device)
    _permute_countsort_kernel[(num_total,)](
        flat, input, permuted, row_id_map, counter,
        num_tokens, num_cols, num_out, topK,
        flat.stride(0),
        input.stride(0), input.stride(1),
        permuted.stride(0), permuted.stride(1),
        BLOCK_C=1024,
    )
    return permuted, row_id_map, base


# ----------------------------------------------------------------------------
# Argsort path (slower Triton path; trails countsort and the C++ backend)
# ----------------------------------------------------------------------------

@triton.jit
def _permute_argsort_kernel(
    sorted_ptr, flat_ptr, input_ptr, out_ptr, row_id_map_ptr,
    start_ptr, end_ptr,                # first/last occurrence idx of each expert
    num_tokens, num_cols, topK, num_out: tl.constexpr,
    stride_flat,
    stride_in_r, stride_in_c,
    stride_out_r, stride_out_c,
    BLOCK_C: tl.constexpr,
):
    idx = tl.program_id(0)
    so = tl.load(sorted_ptr + idx).to(tl.int64)
    t = so // topK
    k = so % topK
    e = tl.load(flat_ptr + so * stride_flat).to(tl.int64)

    # first/last occurrence of expert e in the sorted order -> group bounds.
    # Each expert has exactly one first and one last, so start[e]/end[e] are
    # written by a single program each (race-free, no atomics).
    prev_valid = idx > 0
    prev_so = tl.load(sorted_ptr + (idx - 1), mask=prev_valid).to(tl.int64)
    prev_e = tl.load(flat_ptr + prev_so * stride_flat, mask=prev_valid).to(tl.int64)
    is_first = (idx == 0) | (prev_e != e)
    next_valid = idx < num_out - 1
    next_so = tl.load(sorted_ptr + (idx + 1), mask=next_valid).to(tl.int64)
    next_e = tl.load(flat_ptr + next_so * stride_flat, mask=next_valid).to(tl.int64)
    is_last = (idx == num_out - 1) | (next_e != e)
    if is_first:
        tl.store(start_ptr + e, idx.to(tl.int32))
    if is_last:
        tl.store(end_ptr + e, idx.to(tl.int32))

    for c0 in range(0, tl.cdiv(num_cols, BLOCK_C)):
        cols = c0 * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = cols < num_cols
        vals = tl.load(input_ptr + t * stride_in_r + cols * stride_in_c, mask=mask)
        tl.store(out_ptr + idx * stride_out_r + cols * stride_out_c, vals, mask=mask)
    tl.store(row_id_map_ptr + (k * num_tokens + t), idx.to(tl.int32))


def permute(input: torch.Tensor, indices: torch.Tensor,
            num_out_tokens: int = 0, E: int | None = None):
    num_tokens, num_cols = input.shape
    topK = indices.shape[1]
    flat = indices.reshape(-1).contiguous()
    num_out = num_out_tokens if num_out_tokens > 0 else num_tokens * topK
    if E is None:
        E = int(flat.max()) + 1
    num_total = flat.shape[0]

    # Non-stable sort is enough: order WITHIN an expert is irrelevant to the
    # grouped GEMM / unpermute, so we don't pay for stable ordering.
    sorted_order = torch.argsort(flat)               # [num_total] int64, sorted by expert

    permuted = torch.empty(num_out, num_cols, device=input.device, dtype=input.dtype)
    row_id_map = torch.full((num_tokens * topK,), -1, dtype=torch.int32, device=input.device)
    # start[e]/end[e] = first/last sorted position of expert e (init -1 = absent).
    start = torch.full((E,), -1, dtype=torch.int32, device=input.device)
    end = torch.full((E,), -1, dtype=torch.int32, device=input.device)
    _permute_argsort_kernel[(num_out,)](
        sorted_order, flat, input, permuted, row_id_map, start, end,
        num_tokens, num_cols, topK, num_out,
        flat.stride(0),
        input.stride(0), input.stride(1),
        permuted.stride(0), permuted.stride(1),
        BLOCK_C=1024,
    )
    # derive base (exclusive offsets) from the recorded bounds (atomic-free, gaps OK).
    # counts[e] is derivable as base[e] - base[e-1] (base[0] == 0) when needed.
    present = start >= 0
    sizes = torch.where(present, (end - start + 1).to(torch.int32), torch.zeros_like(start))
    base = torch.zeros(E, dtype=torch.int32, device=input.device)
    base[1:] = sizes.cumsum(0)[:-1].to(torch.int32)   # exclusive prefix sum
    return permuted, row_id_map, base


def permute_backward(grad_permuted: torch.Tensor, row_id_map: torch.Tensor,
                     num_tokens: int, topK: int) -> torch.Tensor:
    """Gradient of ``permute`` w.r.t. ``input`` — the only differentiable input
    (``indices`` is integer, so it has no gradient).

    ``permute`` is a pure gather: ``permuted[dest[j]] = input[t]`` with
    ``t = j // topK`` and ``dest[j] = row_id_map[j] = row_id_map[k*num_tokens + t]``.
    The reverse is a scatter-add of ``grad_permuted`` back onto each token row:
        grad_input[t] = Σ_k grad_permuted[row_id_map[k*num_tokens + t]]
    which is exactly ``unpermute_forward`` with ``prob = 1``. This mirrors the C++
    ``_PermuteFn.backward`` (grouped_gemm_custom_op.py), which reuses
    ``unpermute_raw`` with an empty prob (weight 1).
    """
    prob_ones = torch.ones(num_tokens, topK, device=grad_permuted.device, dtype=torch.float32)
    return unpermute_forward(grad_permuted, row_id_map, prob_ones, num_tokens, topK)


class PermuteAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices, num_out_tokens=0, E=None):
        ctx.num_tokens = input.shape[0]
        ctx.topK = indices.shape[1] if indices.ndim > 1 else 1
        permuted, row_id_map, base = permute(input, indices, num_out_tokens, E)
        ctx.save_for_backward(row_id_map)
        return permuted, row_id_map, base

    @staticmethod
    def backward(ctx, grad_permuted, grad_row_id_map, grad_base):
        (row_id_map,) = ctx.saved_tensors
        grad_input = permute_backward(grad_permuted, row_id_map, ctx.num_tokens, ctx.topK)
        return grad_input, None, None, None


class PermuteCountsortAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices, num_out_tokens=0, E=None):
        ctx.num_tokens = input.shape[0]
        ctx.topK = indices.shape[1] if indices.ndim > 1 else 1
        permuted, row_id_map, base = permute_countsort(input, indices, num_out_tokens, E)
        ctx.save_for_backward(row_id_map)
        return permuted, row_id_map, base

    @staticmethod
    def backward(ctx, grad_permuted, grad_row_id_map, grad_base):
        (row_id_map,) = ctx.saved_tensors
        grad_input = permute_backward(grad_permuted, row_id_map, ctx.num_tokens, ctx.topK)
        return grad_input, None, None, None


# Differentiable entry points (mirror `unpermute` in unpermute.py). The plain
# `permute` / `permute_countsort` functions above stay non-autograd so benchmarks
# measure the kernel alone, not autograd dispatch.
permute_autograd = PermuteAutograd.apply
permute_countsort_autograd = PermuteCountsortAutograd.apply

