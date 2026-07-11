import torch
import triton
import triton.language as tl

_TL_DTYPE = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['N', 'K'],
)
@triton.jit
def gmm_forward_kernel(
    a_ptr, b_ptr, c_ptr, offsets,
    N: tl.constexpr, K: tl.constexpr,
    lda: tl.constexpr, ldc: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    TRANS_B: tl.constexpr, BLOCK_DTYPE: tl.constexpr,
):
    tile_m_idx = tl.program_id(0).to(tl.int64)
    tile_n_idx = tl.program_id(1).to(tl.int64)
    g = tl.program_id(2).to(tl.int64)
    
    # start row of expert g in A and C
    start_g = tl.load(offsets + g).to(tl.int64)
    end_g = tl.load(offsets + g + 1).to(tl.int64)
    M_g = end_g - start_g
    
    if tile_m_idx * BLOCK_SIZE_M >= M_g:
        return
    
    # Matmul tile calculation
    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_mask_m = offs_am[:, None] < M_g
    b_mask_n = offs_bn[None, :] < N
    
    a_base = a_ptr + start_g * lda
    b_base = b_ptr + g * N * K
    c_base = c_ptr + start_g * ldc
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        g_k = kk * BLOCK_SIZE_K + offs_k
        a_mask = a_mask_m & (g_k[None, :] < K)
        a = tl.load(a_base + offs_am[:, None] * lda + g_k[None, :], mask=a_mask, other=0.0)
        
        if TRANS_B:
            b_mask = (offs_bn[:, None] < N) & (g_k[None, :] < K)
            b_coalesced = tl.load(b_base + offs_bn[:, None] * K + g_k[None, :], mask=b_mask, other=0.0)
            b = tl.trans(b_coalesced)
        else:
            b_mask = b_mask_n & (g_k[:, None] < K)
            b = tl.load(b_base + g_k[:, None] * N + offs_bn[None, :], mask=b_mask, other=0.0)
            
        accumulator = tl.dot(a, b, acc=accumulator)
        
    c = accumulator.to(BLOCK_DTYPE)
    c_ptrs = c_base + offs_am[:, None] * ldc + offs_bn[None, :]
    tl.store(c_ptrs, c, mask=(offs_am[:, None] < M_g) & (offs_bn[None, :] < N))


_NS = "tg"  # triton-gmm custom_op namespace


@torch.library.custom_op(f"{_NS}::gmm_fwd_raw", mutates_args=())
def gmm_try_fn(A: torch.Tensor, B: torch.Tensor, offsets: torch.Tensor,
                trans_b: bool = True) -> torch.Tensor:
    assert A.dtype == B.dtype
    dtype = A.dtype
    total, K = A.shape
    E = B.shape[0]
    N = B.shape[1] if trans_b else B.shape[2]
    C = torch.empty((total, N), device=A.device, dtype=dtype)
    # M-grid sized by `total` (not the max expert size M_max). Computing M_max
    # needs `.item()`, which forces a CPU sync + unbacked symbol under
    # torch.compile -> graph break / compile crash. Oversizing the M-grid only
    # launches a few extra tiles that early-return per expert (see kernel guard),
    # so correctness is unchanged and the cost is negligible vs. a graph break.
    # Wrapped as a custom_op, the triton launch + this grid run at
    # inductor runtime (NOT traced), so no FakeTensor/.item() issue remains.
    grid = lambda META: (
        triton.cdiv(total, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
        E,
    )
    gmm_forward_kernel[grid](
        A, B, C, offsets, N, K,
        A.stride(0), C.stride(0),
        TRANS_B=trans_b, BLOCK_DTYPE=_TL_DTYPE[dtype]
    )
    return C


@gmm_try_fn.register_fake
def _(A: torch.Tensor, B: torch.Tensor, offsets: torch.Tensor,
      trans_b: bool = True):
    total, K = A.shape
    N = B.shape[1] if trans_b else B.shape[2]
    return torch.empty((total, N), device=A.device, dtype=A.dtype)


@triton.autotune(
    configs=[
        # small contraction (M_e ~512): big N/K tiles, small M tile, deep pipeline
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_warps=8, num_stages=3),
    ],
    key=['N', 'K'],
)
@triton.jit
def gmm_gradB_kernel(
    g_ptr, a_ptr, c_ptr, offsets,
    K: tl.constexpr, N: tl.constexpr,
    ldg: tl.constexpr, lda: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_DTYPE: tl.constexpr,
):
    # grad_B[g] = G_g^T @ A_g  ->  [N, K] = [N, M_e] @ [M_e, K], contraction over M_e (small).
    # Tile the large output (N, K); loop the small contraction M_e. Big N/K tiles +
    # deep num_stages is what small-contraction GEMMs need to keep tensor cores busy.
    tile_r = tl.program_id(0).to(tl.int64)   # N (output rows)
    tile_c = tl.program_id(1).to(tl.int64)   # K (output cols)
    g = tl.program_id(2).to(tl.int64)
    start_g = tl.load(offsets + g).to(tl.int64)
    end_g = tl.load(offsets + g + 1).to(tl.int64)
    M_e = end_g - start_g

    offs_r = tile_r * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)   # N indices
    offs_c = tile_c * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)   # K indices
    offs_m = tl.arange(0, BLOCK_SIZE_M)                          # token indices

    g_base = g_ptr + start_g * ldg
    a_base = a_ptr + start_g * lda
    c_base = c_ptr + g * N * K

    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    for mm in range(0, tl.cdiv(M_e, BLOCK_SIZE_M)):
        m_idx = mm * BLOCK_SIZE_M + offs_m
        # gg[m, n] = G_g[m, n]  ->  [M_tile, N_tile]
        gg = tl.load(g_base + m_idx[:, None] * ldg + offs_r[None, :],
                     mask=(m_idx[:, None] < M_e) & (offs_r[None, :] < N), other=0.0)
        # aa[m, k] = A_g[m, k]  ->  [M_tile, K_tile]
        aa = tl.load(a_base + m_idx[:, None] * lda + offs_c[None, :],
                     mask=(m_idx[:, None] < M_e) & (offs_c[None, :] < K), other=0.0)
        # grad_B[n, k] = sum_m gg[m,n] * aa[m,k] = tl.trans(gg) @ aa
        accumulator = tl.dot(tl.trans(gg), aa, acc=accumulator)

    c = accumulator.to(BLOCK_DTYPE)
    c_ptrs = c_base + offs_r[:, None] * K + offs_c[None, :]
    tl.store(c_ptrs, c, mask=(offs_r[:, None] < N) & (offs_c[None, :] < K))


# Two-layer pattern (mirrors grouped_gemm_custom_op.py): the raw triton
# launches are registered as custom_ops so Dynamo/Inductor see them as opaque,
# shape-known nodes. This is what stops the control_deps KeyError when a
# custom-op output (e.g. fused_topk_softmax's weights) feeds a functional
# collective like fc.all_reduce inside a compiled graph.

@torch.library.custom_op(f"{_NS}::gmm_bwd_B_raw", mutates_args=())
def gmm_try_gradB(A: torch.Tensor, B: torch.Tensor, offsets: torch.Tensor,
                  grad_output: torch.Tensor, trans_b: bool = True) -> torch.Tensor:
    E = B.shape[0]
    K = A.shape[1]
    N = B.shape[1] if trans_b else B.shape[2]
    assert trans_b, "grad_B only supports trans_b=True"
    base = torch.empty(E, N, K, device=A.device, dtype=A.dtype)
    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_SIZE_N']),
        triton.cdiv(K, META['BLOCK_SIZE_K']),
        E,
    )
    gmm_gradB_kernel[grid](
        grad_output, A, base, offsets,
        K, N, grad_output.stride(0), A.stride(0),
        BLOCK_DTYPE=_TL_DTYPE[A.dtype]
    )
    return base


@gmm_try_gradB.register_fake
def _(A: torch.Tensor, B: torch.Tensor, offsets: torch.Tensor,
      grad_output: torch.Tensor, trans_b: bool = True):
    E = B.shape[0]
    K = A.shape[1]
    N = B.shape[1] if trans_b else B.shape[2]
    return torch.empty(E, N, K, device=A.device, dtype=A.dtype)


class GroupedGEMMAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, offsets, trans_b=True):
        ctx.save_for_backward(A, B, offsets)
        ctx.trans_b = trans_b
        return torch.ops.tg.gmm_fwd_raw(A, B, offsets, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad_output):
        A, B, offsets = ctx.saved_tensors
        trans_b = ctx.trans_b
        grad_A = grad_B = None
        if ctx.needs_input_grad[0]:
            Wb = (B if trans_b else B.transpose(1, 2)).contiguous()
            grad_A = torch.ops.tg.gmm_fwd_raw(grad_output, Wb, offsets, trans_b=False)
        if ctx.needs_input_grad[1]:
            grad_B = torch.ops.tg.gmm_bwd_B_raw(A, B, offsets, grad_output, trans_b)
        return grad_A, grad_B, None, None


def grouped_gemm(A, B, offsets, trans_b: bool = True):
    """Differentiable Triton Grouped GEMM."""
    return GroupedGEMMAutograd.apply(A, B, offsets, trans_b)
