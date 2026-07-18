from typing import Optional
import torch
import triton
import triton.language as tl

# Import to ensure standard custom ops are registered before use
from .grouped_gemm_ops import gmm_forward_kernel, gmm_try_fn, gmm_try_gradB

# torch dtype -> tl.* counterpart for the constexpr kernel param
_TL_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32
}

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_tma(dtype: Optional[torch.dtype] = None):
    # TMA is hardware-designed and optimized for fp16/bf16.
    # Using TMA with fp32 often exceeds shared memory limits with typical tile sizes.
    # Therefore, we only enable TMA for half-precision types.
    if dtype is not None and dtype not in (torch.float16, torch.bfloat16):
        return False
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148

def _cdiv(a, b):
    return (a + b - 1) // b

tma_configs = [
    triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K': BK}, num_stages=s, num_warps=w)
    for BM in [128]
    for BN in [128, 256]
    for BK in [64, 128]
    for s in ([3, 4])
    for w in [4, 8]
]

@triton.autotune(
    tma_configs,
    key=['group_size'],
)
@triton.jit
def grouped_matmul_tma_kernel(
    # full (un-split) tensors — no per-group pointer arrays
    a_ptr,            # [total_tokens, K]
    b_ptr,            # [E, N, K] if TRANS_B else [E, K, N]
    c_ptr,            # [total_tokens, N]
    offsets,          # [E] cumulative group ends (offsets[e-1] is group e's start)
    group_size,       # E
    # geometry (uniform N, K across experts — MoE layout, already tile-aligned)
    N: tl.constexpr,
    K: tl.constexpr,
    lda: tl.constexpr,
    ldc: tl.constexpr,
    # number of persistent CTAs == physical SM count (passed, not autotuned)
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    # B is [E, N, K] (True) -> C = A @ B^T; [E, K, N] (False) -> C = A @ B
    TRANS_B: tl.constexpr,
    # element dtype (tl.float16 / tl.bfloat16)
    BLOCK_DTYPE: tl.constexpr,
):
    dtype = BLOCK_DTYPE
    tile_idx = tl.program_id(0).to(tl.int64)
    last_problem_end = tl.cast(0, tl.int64)
    # start_g is threaded: group 0 starts at row 0, every later group starts at
    # the previous group's end. Avoids a redundant `starts` array / double load.
    start_g = tl.cast(0, tl.int64)
    for g in range(group_size):
        g_i = tl.cast(g, tl.int64)
        end_g = tl.load(offsets + g).to(tl.int64)
        M_g = end_g - start_g
        num_m_tiles = tl.cdiv(M_g, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # static scheduling: this CTA drains every tile `tile_idx + k*NUM_SM`
        # that falls inside group g's tile range.
        if (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            a_base = a_ptr + start_g * lda
            b_base = b_ptr + g_i * N * K          # expert e sits at offset e*N*K
            c_base = c_ptr + start_g * ldc

            # TMA descriptors built from the full-tensor base pointers.
            a_desc = tl.make_tensor_descriptor(
                a_base,
                shape=[M_g, K],
                strides=[lda, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            )
            if TRANS_B:
                b_desc = tl.make_tensor_descriptor(
                    b_base,
                    shape=[N, K],
                    strides=[K, 1],
                    block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
                )
            else:
                b_desc = tl.make_tensor_descriptor(
                    b_base,
                    shape=[K, N],
                    strides=[N, 1],
                    block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
                )
            c_desc = tl.make_tensor_descriptor(
                c_base,
                shape=[M_g, N],
                strides=[ldc, 1],
                block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
            )

            while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
                tile_idx_in_gemm = tile_idx - last_problem_end
                tile_m_idx = tile_idx_in_gemm // num_n_tiles
                tile_n_idx = tile_idx_in_gemm % num_n_tiles

                # Cast offsets/block_shape coordinates to int32 for TMA descriptor load/store
                offs_am = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                offs_bn = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                    kk_k = tl.cast(kk * BLOCK_SIZE_K, tl.int32)
                    a = a_desc.load([offs_am, kk_k])
                    if TRANS_B:
                        b = b_desc.load([offs_bn, kk_k])
                        accumulator += tl.dot(a, b.T)
                    else:
                        b = b_desc.load([kk_k, offs_bn])
                        accumulator += tl.dot(a, b)

                c = accumulator.to(dtype)
                c_desc.store([offs_am, offs_bn], c)

                # go to the next tile by advancing NUM_SM
                tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        start_g = end_g     # becomes the next group's start row
        last_problem_end += num_tiles


def group_gemm_tma_fn(A, B, offsets, trans_b: bool = True):
    """Stacked grouped GEMM (TMA, sm90+) — same signature as group_gemm_fn.

    A:       [total_tokens, K]
    B:       [E, N, K] if trans_b else [E, K, N]
    offsets: [E+1] or [E]       cumulative group offsets
    trans_b: B layout flag (see group_gemm_fn).

    dtype is taken from A (and must match B); bf16 is the MoE default.
    Returns a single stacked C [total_tokens, N].

    Requires sm90+ (supports_tma()).
    """
    assert supports_tma(A.dtype)
    assert A.dtype == B.dtype, "A and B must share the same dtype"
    dtype = A.dtype
    total, K = A.shape
    E = B.shape[0]
    N = B.shape[1] if trans_b else B.shape[2]

    # Handle offsets format (can be size E or E+1)
    if offsets.numel() == E:
        ends = offsets
    else:
        ends = offsets[1:]

    C = torch.empty((total, N), device=A.device, dtype=dtype)

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    tl_dtype = _TL_DTYPE[dtype]
    nsm = num_sms()
    grid = lambda META: (nsm, )
    # Pass ends to the kernel, which represents cumulative ends [ends[0], ends[1], ...]
    grouped_matmul_tma_kernel[grid](
        A, B, C, ends, E, N, K,
        A.stride(0), C.stride(0),
        NUM_SM=nsm, TRANS_B=trans_b, BLOCK_DTYPE=tl_dtype,
    )
    return C


_NS = "tg"  # triton-gmm custom_op namespace

@torch.library.custom_op(f"{_NS}::gmm_tma_fwd_raw", mutates_args=())
def gmm_tma_fwd_raw(A: torch.Tensor, B: torch.Tensor, offsets: torch.Tensor,
                    trans_b: bool = True) -> torch.Tensor:
    return group_gemm_tma_fn(A, B, offsets, trans_b=trans_b)


@gmm_tma_fwd_raw.register_fake
def _(A: torch.Tensor, B: torch.Tensor, offsets: torch.Tensor,
      trans_b: bool = True):
    total, K = A.shape
    N = B.shape[1] if trans_b else B.shape[2]
    return torch.empty((total, N), device=A.device, dtype=A.dtype)


class GroupedGEMMAutogradSM90(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, offsets, trans_b=True):
        ctx.save_for_backward(A, B, offsets)
        ctx.trans_b = trans_b
        if supports_tma(A.dtype):
            return torch.ops.tg.gmm_tma_fwd_raw(A, B, offsets, trans_b=trans_b)
        else:
            return torch.ops.tg.gmm_fwd_raw(A, B, offsets, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad_output):
        A, B, offsets = ctx.saved_tensors
        trans_b = ctx.trans_b
        grad_A = grad_B = None
        if ctx.needs_input_grad[0]:
            Wb = (B if trans_b else B.transpose(1, 2)).contiguous()
            if supports_tma(A.dtype):
                grad_A = torch.ops.tg.gmm_tma_fwd_raw(grad_output, Wb, offsets, trans_b=False)
            else:
                grad_A = torch.ops.tg.gmm_fwd_raw(grad_output, Wb, offsets, trans_b=False)
        if ctx.needs_input_grad[1]:
            grad_B = torch.ops.tg.gmm_bwd_B_raw(A, B, offsets, grad_output, trans_b)
        return grad_A, grad_B, None, None


def grouped_gemm(A, B, offsets, trans_b: bool = True):
    """Differentiable Triton Grouped GEMM with automatic SM90 TMA dispatch."""
    return GroupedGEMMAutogradSM90.apply(A, B, offsets, trans_b)
