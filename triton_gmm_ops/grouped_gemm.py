import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


_TL_DTYPE = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


@triton.jit
def gmm_try_kernel(
    a_ptr, b_ptr, c_ptr, offsets, group_size,
    N: tl.constexpr, K: tl.constexpr,
    lda: tl.constexpr, ldc: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    TRANS_B: tl.constexpr, BLOCK_DTYPE: tl.constexpr,
):
    tile_m_idx = tl.program_id(0).to(tl.int64)
    tile_n_idx = tl.program_id(1).to(tl.int64)
    g = tl.program_id(2).to(tl.int64)
    
    # start row of expert g in A and C
    if g == 0:
        start_g = tl.cast(0, tl.int64)
    else:
        start_g = tl.load(offsets + (g - 1)).to(tl.int64)
        
    end_g = tl.load(offsets + g).to(tl.int64)
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
            
        accumulator += tl.dot(a, b)
        
    c = accumulator.to(BLOCK_DTYPE)
    c_ptrs = c_base + offs_am[:, None] * ldc + offs_bn[None, :]
    tl.store(c_ptrs, c, mask=(offs_am[:, None] < M_g) & (offs_bn[None, :] < N))


def gmm_try_fn(A, B, offsets, trans_b: bool = True):
    assert A.dtype == B.dtype
    dtype = A.dtype
    total, K = A.shape
    E = B.shape[0]
    N = B.shape[1] if trans_b else B.shape[2]
    
    # Use fallback block size if not autotuning or for initialization
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 64
    
    C = torch.empty((total, N), device=A.device, dtype=dtype)
    grid = (triton.cdiv(total, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), E)
    
    gmm_try_kernel[grid](
        A, B, C, offsets, E, N, K,
        A.stride(0), C.stride(0),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        TRANS_B=trans_b, BLOCK_DTYPE=_TL_DTYPE[dtype]
    )
    return C


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 32}),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 32}),
        triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 32}),
        triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 32}),
    ],
    key=['group_size'],
)
@triton.jit
def gmm_try_gradB_kernel(
    a_ptr, g_ptr, c_ptr, offsets, group_size,
    K: tl.constexpr, N: tl.constexpr,
    lda: tl.constexpr, ldg: tl.constexpr,
    ldc: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
    TRANS_B: tl.constexpr, BLOCK_DTYPE: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N)
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    
    GROUP_K = 8
    tiles_per_group = GROUP_K * num_n_tiles
    group_idx = tile // tiles_per_group
    first_tile_k = group_idx * GROUP_K
    current_group_size_k = tl.minimum(num_k_tiles - first_tile_k, GROUP_K)
    idx_in_group = tile % tiles_per_group
    tile_k = first_tile_k + (idx_in_group % current_group_size_k)
    tile_n = idx_in_group // current_group_size_k
    
    idx = tl.maximum(g - 1, 0)
    prev = tl.load(offsets + idx).to(tl.int64)
    start_g = tl.where(g == 0, tl.cast(0, tl.int64), prev)
    end_g = tl.load(offsets + g).to(tl.int64)
    M_e = end_g - start_g
    
    a_base = a_ptr + start_g * lda
    g_base = g_ptr + start_g * ldg
    c_base = c_ptr + g * ldc
    
    offs_k = tile_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    k_mask = offs_k[:, None] < K
    n_mask = offs_n[None, :] < N
    
    acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    for m in range(0, tl.cdiv(M_e, BLOCK_SIZE_M)):
        m_idx = m * BLOCK_SIZE_M + offs_m
        m_b = m_idx[:, None] < M_e
        a_coalesced = tl.load(a_base + m_idx[:, None] * lda + offs_k[None, :],
                              mask=(m_idx[:, None] < M_e) & (offs_k[None, :] < K),
                              other=0.0)
        a = tl.trans(a_coalesced)
        gg = tl.load(g_base + m_idx[:, None] * ldg + offs_n[None, :], mask=m_b & n_mask, other=0.0)
        acc += tl.dot(a, gg)
        
    c = acc.to(BLOCK_DTYPE)
    if TRANS_B:
        c_ptrs = c_base + offs_n[None, :] * K + offs_k[:, None]
        tl.store(c_ptrs, c, mask=k_mask & n_mask)
    else:
        c_ptrs = c_base + offs_k[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, c, mask=k_mask & n_mask)


def gmm_try_gradB(A, B, offsets, grad_output, trans_b):
    E = B.shape[0]
    K = A.shape[1]
    N = B.shape[1] if trans_b else B.shape[2]
    if trans_b:
        base = torch.empty(E, N, K, device=A.device, dtype=A.dtype)
    else:
        base = torch.empty(E, K, N, device=A.device, dtype=A.dtype)
        
    grid = lambda META: (
        E,
        triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(K, META['BLOCK_SIZE_K']),
    )
    gmm_try_gradB_kernel[grid](
        A, grad_output, base, offsets, E,
        K, N, A.stride(0), grad_output.stride(0), base.stride(0),
        TRANS_B=trans_b, BLOCK_DTYPE=_TL_DTYPE[A.dtype]
    )
    return base


class GroupedGEMMAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, offsets, trans_b=True):
        ctx.save_for_backward(A, B, offsets)
        ctx.trans_b = trans_b
        return gmm_try_fn(A, B, offsets, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad_output):
        A, B, offsets = ctx.saved_tensors
        trans_b = ctx.trans_b
        grad_A = grad_B = None
        if ctx.needs_input_grad[0]:
            Wb = (B if trans_b else B.transpose(1, 2)).contiguous()
            grad_A = gmm_try_fn(grad_output, Wb, offsets, trans_b=False)
        if ctx.needs_input_grad[1]:
            grad_B = gmm_try_gradB(A, B, offsets, grad_output, trans_b)
        return grad_A, grad_B, None, None


def grouped_gemm(A, B, offsets, trans_b: bool = True):
    """Differentiable Triton Grouped GEMM."""
    return GroupedGEMMAutograd.apply(A, B, offsets, trans_b)
