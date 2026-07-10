
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


def permute_countsort(input: torch.Tensor, indices: torch.Tensor,
                      num_out_tokens: int = 0, E: int | None = None):
    num_tokens, num_cols = input.shape
    topK = indices.shape[1]
    flat = indices.reshape(-1).contiguous()
    num_total = num_tokens * topK
    num_out = num_out_tokens if num_out_tokens > 0 else num_total
    if E is None:
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


def permute_backward(grad_permuted: torch.Tensor, row_id_map: torch.Tensor,
                     num_tokens: int, topK: int) -> torch.Tensor:
    prob_ones = torch.ones(num_tokens, topK, device=grad_permuted.device, dtype=torch.float32)
    return unpermute_forward(grad_permuted, row_id_map, prob_ones, num_tokens, topK)


class PermuteAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices, num_out_tokens=0, E=None):
        ctx.num_tokens = input.shape[0]
        ctx.topK = indices.shape[1] if indices.ndim > 1 else 1
        permuted, row_id_map, offsets = permute_countsort(input, indices, num_out_tokens, E)
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
