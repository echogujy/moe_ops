
import torch
import triton
import triton.language as tl

_NS = "tg"

from .unpermute_ops import unpermute_forward  # permute backward == unpermute(prob=1)


# ----------------------------------------------------------------------------
# Custom group histogram + exclusive prefix sum.
# Fused setup that replaces torch.bincount/cumsum/max, and emits the exact
# grouped-GEMM metadata (group sizes + offsets) consumed right after permute.
# ----------------------------------------------------------------------------

@triton.jit
def _group_offsets_kernel(flat_ptr, offsets_ptr, num_total: tl.constexpr,
                          E: tl.constexpr, BLOCK: tl.constexpr, stride_flat):
    """Fused histogram + prefix-sum in ONE kernel.

    Accumulates the block exclusive prefix sum atomically into offsets_ptr + bins + 1.
    """
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    idx = pid * BLOCK + offs
    m = idx < num_total
    v = tl.load(flat_ptr + idx * stride_flat, mask=m).to(tl.int32)
    h = tl.histogram(v, E, mask=m)          # [E] block-local histogram
    inc = tl.cumsum(h, axis=0)              # [E] inclusive block-local prefix sum
    bins = tl.arange(0, E)
    # Store inclusive prefix sum into offsets[1:] to make it inclusive cumsum with offsets[0] = 0
    tl.atomic_add(offsets_ptr + bins + 1, inc)


def _group_offsets(flat: torch.Tensor, E: int):
    """Custom-kernel (fused) histogram + prefix sum -> offsets (size E+1, starting with 0)."""
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=flat.device)
    num_total = flat.shape[0]
    HBLOCK = 2048                           # ponytail: block size; also E <= HBLOCK
    _group_offsets_kernel[(num_total + HBLOCK - 1) // HBLOCK,](
        flat, offsets, num_total, E, HBLOCK, flat.stride(0)) # type: ignore
    return offsets


# ----------------------------------------------------------------------------
# Counting sort (scatter, atomic dest assignment)
# ----------------------------------------------------------------------------

@triton.jit
def _permute_countsort_kernel(
    flat_ptr,                        # [num_total] int32 expert ids
    input_ptr,                       # [num_tokens, num_cols]
    out_ptr,                         # [num_out, num_cols]
    row_id_map_ptr,                  # [num_total] int32
    counter_ptr,                     # [E] int32 atomic counter (init = offsets[e])
    num_tokens, num_cols, num_out, topK: tl.constexpr,
    stride_flat,
    stride_in_r, stride_in_c,
    stride_out_r, stride_out_c,
    BLOCK_C: tl.constexpr,
):
    j = tl.program_id(0)
    e = tl.load(flat_ptr + j * stride_flat).to(tl.int64)
    dest = tl.atomic_add(counter_ptr + e, 1).to(tl.int64)
    if dest < num_out:
        t = j // topK
        k = j % topK
        for c0 in range(tl.cdiv(num_cols, BLOCK_C)):
            cols = c0 * BLOCK_C + tl.arange(0, BLOCK_C)
            mask = cols < num_cols
            vals = tl.load(input_ptr + t * stride_in_r + cols * stride_in_c, mask=mask)
            tl.store(out_ptr + dest * stride_out_r + cols * stride_out_c, vals, mask=mask)
        tl.store(row_id_map_ptr + (k * num_tokens + t), dest.to(tl.int32))


# Two-layer pattern: the counting-sort (raw triton launches) is a custom_op so
# Dynamo/Inductor see it as an opaque, shape-known node. E is a python int
# (num_experts); callers under torch.compile always pass it, so the E<=0
# fallback's .item() only runs eagerly.

@torch.library.custom_op(f"{_NS}::permute_countsort", mutates_args=())
def permute_countsort(input: torch.Tensor, indices: torch.Tensor,
                      num_out_tokens: int = 0, E: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_cols = input.shape
    topK = indices.shape[1]
    flat = indices.reshape(-1).contiguous()
    num_total = num_tokens * topK
    num_out = num_out_tokens if num_out_tokens > 0 else num_total
    # offsets (from the full histogram) always sum to num_total, so permuted
    # must hold at least that many rows; a smaller capacity would make
    # grouped_gemm read OOB. (Capacity-dropping isn't implemented here.)
    num_out = max(num_out, num_total)
    if E <= 0:
        E = int(flat.max()) + 1

    offsets = _group_offsets(flat, E)
    counter = offsets[:-1].clone()

    permuted   = torch.empty(num_out, num_cols, device=input.device, dtype=input.dtype)
    row_id_map = torch.full((num_total,), -1, dtype=torch.int32, device=input.device)
    _permute_countsort_kernel[(num_total,)](
        flat, input, permuted, row_id_map, counter,
        num_tokens, num_cols, num_out, topK,
        flat.stride(0),
        input.stride(0), input.stride(1),
        permuted.stride(0), permuted.stride(1),
        BLOCK_C=1024,
    )
    return permuted, row_id_map, offsets


@permute_countsort.register_fake
def _(input: torch.Tensor, indices: torch.Tensor,
      num_out_tokens: int = 0, E: int = 0):
    num_tokens, num_cols = input.shape
    topK = indices.shape[1]
    num_total = num_tokens * topK
    num_out = max(num_out_tokens, num_total)
    # offsets length is unused for shape derivation downstream, so a conservative
    # bound keeps fake tracing valid when E isn't known at trace time.
    off_len = E + 1 if E > 0 else num_total + 1
    return (
        torch.empty(num_out, num_cols, device=input.device, dtype=input.dtype),
        torch.empty(num_total, device=input.device, dtype=torch.int32),
        torch.empty(off_len, device=input.device, dtype=torch.int32),
    )


def permute_backward(grad_permuted: torch.Tensor, row_id_map: torch.Tensor,
                     num_tokens: int, topK: int) -> torch.Tensor:
    prob_ones = torch.ones(num_tokens, topK, device=grad_permuted.device, dtype=torch.float32)
    return unpermute_forward(grad_permuted, row_id_map, prob_ones, num_tokens, topK)


class PermuteAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices, num_out_tokens=0, E=None):
        ctx.num_tokens = input.shape[0]
        ctx.topK = indices.shape[1] if indices.ndim > 1 else 1
        # E is forwarded to the custom_op; when None it becomes 0 and the real
        # function resolves num_experts at runtime (its .item() is never traced,
        # so no graph break under torch.compile). Compiled callers pass E.
        e = E if E is not None else 0
        permuted, row_id_map, offsets = torch.ops.tg.permute_countsort(
            input, indices, num_out_tokens, e)
        ctx.save_for_backward(row_id_map)
        return permuted, row_id_map, offsets

    @staticmethod
    def backward(ctx, grad_permuted, grad_row_id_map, grad_offsets):
        (row_id_map,) = ctx.saved_tensors
        grad_input = permute_backward(grad_permuted, row_id_map, ctx.num_tokens, ctx.topK)
        return grad_input, None, None, None


def permute(input, indices, num_out_tokens=0, E=None):
    """Differentiable Triton Permute (counting sort)."""
    return PermuteAutograd.apply(input, indices, num_out_tokens, E)
