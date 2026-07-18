
import torch
import triton
import triton.language as tl

_NS = "tg"


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_C": 512, "num_warps": 1}),
        triton.Config(kwargs={"BLOCK_C": 1024, "num_warps": 2}),
        triton.Config(kwargs={"BLOCK_C": 1024, "num_warps": 4}),
        triton.Config(kwargs={"BLOCK_C": 2048, "num_warps": 4}),
        triton.Config(kwargs={"BLOCK_C": 2048, "num_warps": 8}),
        triton.Config(kwargs={"BLOCK_C": 4096, "num_warps": 8}),
    ],
    key=["num_cols"],
)
@triton.jit
def _unpermute_kernel(
    input_ptr, row_id_map_ptr, prob_ptr, out_ptr,
    num_tokens, num_cols, topK: tl.constexpr,
    stride_in_r, stride_in_c,
    stride_prob_r, stride_prob_c,
    stride_out_r, stride_out_c,
    BLOCK_C: tl.constexpr,
):
    # Grid: one program per output token t (row of `out`).
    # Each program gathers its topK expert slots and writes out[t].
    t = tl.program_id(0)

    # Columns are processed in blocks of BLOCK_C. A token row can be far wider
    # than the register file (e.g. 8192 hidden dims), so we accumulate per block.
    # IMPORTANT: `acc` is reset every column block — it holds only the partial
    # sum for the columns in [c0*BLOCK_C, c0*BLOCK_C + BLOCK_C), never a running
    # total across blocks. (Forgetting to reset this is a silent correctness bug
    # that leaves every column block except the first at zero.)
    for c0 in range(0, tl.cdiv(num_cols, BLOCK_C)):
        cols = c0 * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = cols < num_cols
        acc = tl.zeros([BLOCK_C], dtype=tl.float32)
        for k in range(topK):
            # Source permuted row for token t's k-th slot.
            offset = k * num_tokens + t
            src = tl.load(row_id_map_ptr + offset.to(tl.int64)).to(tl.int64)
            # Routing weight for this slot; scalar load, bf16/f32 both fine.
            p = tl.load(prob_ptr + t * stride_prob_r + k * stride_prob_c)
            # Gather the whole source row (block of columns) and upcast to f32.
            vals = tl.load(input_ptr + src * stride_in_r + cols * stride_in_c,
                           mask=mask).to(tl.float32)
            acc = acc + vals * p
        # Write this column block of out[t]; cast accumulator back to input dtype.
        tl.store(out_ptr + t * stride_out_r + cols * stride_out_c,
                 acc.to(out_ptr.dtype.element_ty), mask=mask)


@torch.library.custom_op(f"{_NS}::unpermute_forward", mutates_args=())
def unpermute_forward(input: torch.Tensor, row_id_map: torch.Tensor, prob: torch.Tensor,
                      num_tokens: int, num_topK: int, max_tokens: int = -1) -> torch.Tensor:
    if max_tokens == -1:
        max_tokens = num_tokens
    num_cols = input.shape[1]
    out = torch.empty(max_tokens, num_cols, device=input.device, dtype=input.dtype)
    _unpermute_kernel[(max_tokens,)](
        input, row_id_map, prob, out,
        max_tokens, num_cols, num_topK,
        input.stride(0), input.stride(1),
        prob.stride(0), prob.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


@unpermute_forward.register_fake
def _(input, row_id_map, prob, num_tokens, num_topK, max_tokens=-1):
    if max_tokens == -1:
        max_tokens = num_tokens
    return torch.empty(max_tokens, input.shape[1], device=input.device, dtype=input.dtype)


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_C": 512, "num_warps": 1}),
        triton.Config(kwargs={"BLOCK_C": 1024, "num_warps": 2}),
        triton.Config(kwargs={"BLOCK_C": 1024, "num_warps": 4}),
        triton.Config(kwargs={"BLOCK_C": 2048, "num_warps": 4}),
        triton.Config(kwargs={"BLOCK_C": 2048, "num_warps": 8}),
        triton.Config(kwargs={"BLOCK_C": 4096, "num_warps": 8}),
    ],
    key=["num_cols"],
)
@triton.jit
def _unpermute_bwd_kernel(
    grad_out_ptr, input_ptr, row_id_map_ptr, prob_ptr,
    act_grad_ptr, prob_grad_ptr,
    num_tokens, num_cols, topK: tl.constexpr,
    stride_g_r, stride_g_c,
    stride_in_r, stride_in_c,
    stride_ag_r, stride_ag_c,
    stride_prob_r, stride_prob_c,
    stride_pg_r, stride_pg_c,
    BLOCK_C: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: one program per token t (NOT per (t,k)). Each program owns token t and
    # loops over its topK slots internally. This lets grad_out[t] be loaded ONCE
    # per column block and reused across all k, instead of being read topK times
    # by the topK separate programs in the old (t,k)-grid layout. For topK=2 this
    # roughly halves grad_out traffic -> the main backward speedup (B1).
    #
    # Merged outputs:
    #   act_grad[i, :] = prob[t,k] * grad_out[t, :]      (scatter of one row i)
    #   prob_grad[t,k]  = Σ_col grad_out[t,col]*input[i,col]  (reduction to scalar)
    # where i = row_id_map[k*num_tokens + t] is the permuted source row.
    # No write races: row_id_map is a bijection from valid (t,k) to permuted rows,
    # so each act_grad row i is written by exactly one program (the one owning t
    # and that specific k), and each prob_grad[t,k] by its own program.
    t = tl.program_id(0)
    kidx = tl.arange(0, BLOCK_K)

    # per-k prob_grad accumulator: a [BLOCK_K] vector padded to a power of
    # 2 (topK itself may be e.g. 6), reduced across all column blocks.
    # Only the first `topK` columns are real; padding stays 0 and is
    # dropped by the store mask below.
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)
    for c0 in range(0, tl.cdiv(num_cols, BLOCK_C)):
        cols = c0 * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = cols < num_cols
        # grad_out[t] loaded ONCE per block, reused for every k (B1 win).
        g = tl.load(grad_out_ptr + t * stride_g_r + cols * stride_g_c,
                    mask=mask).to(tl.float32)
        for k in range(topK):
            # row_id_map / prob are tiny scalar loads; reloading per block is
            # negligible traffic vs the row slices, so we keep them inside the k
            # loop to avoid dynamic tensor indexing (Triton forbids `tensor[k]`).
            i = tl.load(row_id_map_ptr + (k * num_tokens + t).to(tl.int64)).to(tl.int64)
            p = tl.load(prob_ptr + t * stride_prob_r + k * stride_prob_c).to(tl.float32)
            v = tl.load(input_ptr + i * stride_in_r + cols * stride_in_c,
                        mask=mask).to(tl.float32)
            # prob_grad: dot product over columns (sum -> one scalar per block).
            acc_k = tl.sum(g * v)
            acc = tl.where(kidx == k, acc + acc_k, acc)
            # act_grad: scatter row i of the permuted grad, weighted by p.
            # Cast back to the activation dtype (e.g. bf16) on store.
            tl.store(act_grad_ptr + i * stride_ag_r + cols * stride_ag_c,
                     (g * p).to(act_grad_ptr.dtype.element_ty), mask=mask)
    # prob_grad[t, :] = acc.  Only the real `topK` columns are written;
    # the padded columns (kidx >= topK) are dropped via the store mask.
    tl.store(prob_grad_ptr + t * stride_pg_r + kidx * stride_pg_c,
             acc.to(prob_grad_ptr.dtype.element_ty), mask=kidx < topK)


@torch.library.custom_op(f"{_NS}::unpermute_backward", mutates_args=())
def unpermute_backward(grad_out: torch.Tensor, input_fwd: torch.Tensor,
                       row_id_map: torch.Tensor, prob: torch.Tensor,
                       num_tokens: int, num_topK: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward: single merged kernel computing both act_grad and prob_grad."""
    num_cols = input_fwd.shape[1]
    act_grad = torch.empty_like(input_fwd)  # [num_out, num_cols]
    prob_grad = torch.empty(num_tokens, num_topK, device=prob.device, dtype=prob.dtype)
    BLOCK_K = triton.next_power_of_2(num_topK)
    _unpermute_bwd_kernel[(num_tokens,)](
        grad_out, input_fwd, row_id_map, prob, act_grad, prob_grad,
        num_tokens, num_cols, num_topK,
        grad_out.stride(0), grad_out.stride(1),
        input_fwd.stride(0), input_fwd.stride(1),
        act_grad.stride(0), act_grad.stride(1),
        prob.stride(0), prob.stride(1),
        prob_grad.stride(0), prob_grad.stride(1),
        BLOCK_K=BLOCK_K,
    )
    return act_grad, prob_grad


@unpermute_backward.register_fake
def _(grad_out, input_fwd, row_id_map, prob, num_tokens, num_topK):
    act_grad = torch.empty_like(input_fwd)
    prob_grad = torch.empty(num_tokens, num_topK, device=prob.device, dtype=prob.dtype)
    return act_grad, prob_grad


class UnpermuteAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, row_id_map, prob, num_tokens, num_topK, max_tokens=-1):
        if max_tokens == -1:
            max_tokens = num_tokens
        ctx.save_for_backward(input, row_id_map, prob)
        ctx.num_tokens = num_tokens
        ctx.num_topK = num_topK
        ctx.max_tokens = max_tokens
        return torch.ops.tg.unpermute_forward(input, row_id_map, prob, num_tokens, num_topK, max_tokens)

    @staticmethod
    def backward(ctx, grad_out):
        input, row_id_map, prob = ctx.saved_tensors
        act_grad, prob_grad = torch.ops.tg.unpermute_backward(
            grad_out, input, row_id_map, prob, ctx.num_tokens, ctx.num_topK)
        return act_grad, None, prob_grad, None, None, None


def unpermute(input, row_id_map, prob, num_tokens, num_topK, max_tokens: int = -1):
    """Differentiable Triton unpermute.

    Forward scatters weighted expert outputs back to original token positions;
    backward produces the gradient w.r.t. the permuted input (act_grad) and w.r.t.
    the routing probabilities (prob_grad).
    """
    return UnpermuteAutograd.apply(input, row_id_map, prob, num_tokens, num_topK, max_tokens)
