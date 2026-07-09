import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def num_sms():
    if is_cuda():
        return torch.cuda.get_device_properties("cuda").multi_processor_count
    return 148

_TL_DTYPE = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}

# ----------------------------------------------------------------------------
# 1. Fused Permute-GEMM Try Kernels (3D Grid / Output-Tile Mapped)
# ----------------------------------------------------------------------------

@triton.jit
def _fused_permute_gemm_try_kernel(
    x_ptr, w_ptr, y_ptr, offsets, sorted_id_map_ptr,
    N: tl.constexpr, K: tl.constexpr,
    ldx: tl.constexpr, ldy: tl.constexpr,
    group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    TRANS_B: tl.constexpr, BLOCK_DTYPE: tl.constexpr,
):
    tile_m_idx = tl.program_id(0).to(tl.int64)
    tile_n_idx = tl.program_id(1).to(tl.int64)
    g = tl.program_id(2).to(tl.int64)
    
    if g == 0:
        start_g = tl.cast(0, tl.int64)
    else:
        start_g = tl.load(offsets + (g - 1)).to(tl.int64)
    end_g = tl.load(offsets + g).to(tl.int64)
    M_g = end_g - start_g
    
    if tile_m_idx * BLOCK_SIZE_M >= M_g:
        return
        
    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_mask_m = offs_am[:, None] < M_g
    mapped_row_idx = start_g + offs_am
    t = tl.load(sorted_id_map_ptr + mapped_row_idx, mask=offs_am < M_g, other=0).to(tl.int64)
    
    a_ptrs = x_ptr + t[:, None] * ldx + offs_k[None, :]
    b_base = w_ptr + g * N * K
    if TRANS_B:
        ldb = K
        b_ptrs = b_base + offs_bn[:, None] * ldb + offs_k[None, :]
        b_k_step = BLOCK_SIZE_K
    else:
        ldb = N
        b_ptrs = b_base + offs_k[:, None] * ldb + offs_bn[None, :]
        b_k_step = BLOCK_SIZE_K * ldb
        
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        g_k = kk * BLOCK_SIZE_K + offs_k
        a_mask = a_mask_m & (g_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        if TRANS_B:
            b_mask = (offs_bn[:, None] < N) & (g_k[None, :] < K)
            b = tl.trans(tl.load(b_ptrs, mask=b_mask, other=0.0))
        else:
            b_mask = (offs_bn[None, :] < N) & (g_k[:, None] < K)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += b_k_step
        
    y = accumulator.to(BLOCK_DTYPE)
    offs_ym = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_yn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = (y_ptr + start_g * ldy) + ldy * offs_ym[:, None] + offs_yn[None, :]
    tl.store(y_ptrs, y, mask=(offs_ym[:, None] < M_g) & (offs_yn[None, :] < N))


@triton.jit
def _fused_permute_gemm_gradW_try_kernel(
    x_ptr, g_ptr, dw_ptr, offsets, sorted_id_map_ptr,
    K: tl.constexpr, N: tl.constexpr,
    ldx: tl.constexpr, ldg: tl.constexpr,
    group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
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
    
    if g == 0:
        start_g = tl.cast(0, tl.int64)
    else:
        start_g = tl.load(offsets + (g - 1)).to(tl.int64)
    end_g = tl.load(offsets + g).to(tl.int64)
    M_g = end_g - start_g
    
    offs_k = tile_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tile_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    
    k_mask = offs_k[:, None] < K
    n_mask = offs_n[None, :] < N
    
    if TRANS_B:
        dw_base = dw_ptr + g * N * K + offs_n[None, :] * K + offs_k[:, None]
    else:
        dw_base = dw_ptr + g * K * N + offs_k[:, None] * N + offs_n[None, :]
        
    accumulator = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    for m in range(0, tl.cdiv(M_g, BLOCK_SIZE_M)):
        m_idx = m * BLOCK_SIZE_M + offs_m
        m_b = m_idx < M_g
        
        mapped_row = start_g + m_idx
        t = tl.load(sorted_id_map_ptr + mapped_row, mask=m_b, other=0).to(tl.int64)
        
        x_mask = k_mask & m_b[None, :]
        x = tl.load(x_ptr + t[None, :] * ldx + offs_k[:, None], mask=x_mask, other=0.0)
        
        gy_mask = m_b[:, None] & n_mask
        gy = tl.load(g_ptr + (start_g + m_idx)[:, None] * ldg + offs_n[None, :], mask=gy_mask, other=0.0)
        
        accumulator += tl.dot(x, gy)
        
    dw = accumulator.to(BLOCK_DTYPE)
    tl.store(dw_base, dw, mask=k_mask & n_mask)


@triton.jit
def _fused_permute_gemm_gradX_try_kernel(
    g_ptr, w_ptr, dx_ptr, offsets, sorted_id_map_ptr,
    N: tl.constexpr, K: tl.constexpr,
    ldg: tl.constexpr, lddx: tl.constexpr,
    group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    TRANS_B: tl.constexpr, BLOCK_DTYPE: tl.constexpr,
):
    tile_m_idx = tl.program_id(0).to(tl.int64)
    tile_n_idx = tl.program_id(1).to(tl.int64)
    g = tl.program_id(2).to(tl.int64)
    
    if g == 0:
        start_g = tl.cast(0, tl.int64)
    else:
        start_g = tl.load(offsets + (g - 1)).to(tl.int64)
    end_g = tl.load(offsets + g).to(tl.int64)
    M_g = end_g - start_g
    
    if tile_m_idx * BLOCK_SIZE_M >= M_g:
        return
        
    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_mask_m = offs_am[:, None] < M_g
    a_base = g_ptr + start_g * ldg
    a_ptrs = a_base + offs_am[:, None] * ldg + offs_k[None, :]
    
    b_base = w_ptr + g * N * K
    if TRANS_B:
        ldb = K
        b_ptrs = b_base + offs_k[:, None] * ldb + offs_bn[None, :]
        b_k_step = BLOCK_SIZE_K * ldb
    else:
        ldb = N
        b_ptrs = b_base + offs_bn[:, None] * ldb + offs_k[None, :]
        b_k_step = BLOCK_SIZE_K
        
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kk in range(0, tl.cdiv(N, BLOCK_SIZE_K)):
        g_k = kk * BLOCK_SIZE_K + offs_k
        a_mask = a_mask_m & (g_k[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        if TRANS_B:
            b_mask = (g_k[:, None] < N) & (offs_bn[None, :] < K)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        else:
            b_mask = (offs_bn[:, None] < K) & (g_k[None, :] < N)
            b = tl.trans(tl.load(b_ptrs, mask=b_mask, other=0.0))
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += b_k_step
        
    dx = accumulator.to(BLOCK_DTYPE)
    
    dest = start_g + offs_am
    m_valid = offs_am < M_g
    t = tl.load(sorted_id_map_ptr + dest, mask=m_valid, other=0).to(tl.int64)
    
    dx_ptrs = dx_ptr + t[:, None] * lddx + offs_bn[None, :]
    mask = m_valid[:, None] & (offs_bn[None, :] < K)
    tl.atomic_add(dx_ptrs, dx, mask=mask)


class FusedPermuteGEMMTryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, base, sorted_id_map, trans_b=True):
        ctx.save_for_backward(X, W, base, sorted_id_map)
        ctx.trans_b = trans_b
        
        num_out = sorted_id_map.shape[0]
        N = W.shape[1] if trans_b else W.shape[2]
        K = X.shape[1]
        E = W.shape[0]
        
        offsets = torch.empty_like(base)
        offsets[:-1] = base[1:]
        offsets[-1] = num_out
        
        Y = torch.empty((num_out, N), device=X.device, dtype=X.dtype)
        
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 64
        
        grid = (triton.cdiv(num_out, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), E)
        
        _fused_permute_gemm_try_kernel[grid](
            X, W, Y, offsets, sorted_id_map,
            N, K,
            X.stride(0), Y.stride(0),
            E,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
            TRANS_B=trans_b, BLOCK_DTYPE=_TL_DTYPE[X.dtype],
        )
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        X, W, base, sorted_id_map = ctx.saved_tensors
        trans_b = ctx.trans_b
        
        K = X.shape[1]
        E = W.shape[0]
        N = W.shape[1] if trans_b else W.shape[2]
        num_out = sorted_id_map.shape[0]
        
        offsets = torch.empty_like(base)
        offsets[:-1] = base[1:]
        offsets[-1] = num_out
        
        # 1. grad_W
        grad_W = torch.zeros_like(W)
        grid_W = lambda META: (E, triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(K, META['BLOCK_SIZE_K']))
        _fused_permute_gemm_gradW_try_kernel[grid_W](
            X, grad_Y, grad_W, offsets, sorted_id_map,
            K, N,
            X.stride(0), grad_Y.stride(0),
            E,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128,
            TRANS_B=trans_b, BLOCK_DTYPE=_TL_DTYPE[W.dtype],
        )
        
        # 2. grad_X
        grad_X = torch.zeros_like(X)
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        grid_X = (triton.cdiv(num_out, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_N), E)
        _fused_permute_gemm_gradX_try_kernel[grid_X](
            grad_Y, W, grad_X, offsets, sorted_id_map,
            N, K,
            grad_Y.stride(0), grad_X.stride(0),
            E,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
            TRANS_B=trans_b, BLOCK_DTYPE=_TL_DTYPE[X.dtype],
        )
        
        return grad_X, grad_W, None, None, None


def fused_permute_gemm_try(X, W, base, sorted_id_map, trans_b=True):
    return FusedPermuteGEMMTryFunction.apply(X, W, base, sorted_id_map, trans_b)

# ----------------------------------------------------------------------------
# 2. Fused GEMM-Unpermute Try Kernels (3D Grid / Output-Tile Mapped)
# ----------------------------------------------------------------------------

@triton.jit
def _fused_gemm_unpermute_try_kernel(
    a_ptr, b_ptr, out_ptr, y_ptr, offsets, sorted_id_map_ptr, slot_map_ptr, prob_ptr,
    N: tl.constexpr, K: tl.constexpr,
    lda: tl.constexpr, ldout: tl.constexpr, ldy: tl.constexpr,
    stride_prob_r, stride_prob_c,
    group_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    TRANS_B: tl.constexpr, BLOCK_DTYPE: tl.constexpr,
):
    tile_m_idx = tl.program_id(0).to(tl.int64)
    tile_n_idx = tl.program_id(1).to(tl.int64)
    g = tl.program_id(2).to(tl.int64)
    
    if g == 0:
        start_g = tl.cast(0, tl.int64)
    else:
        start_g = tl.load(offsets + (g - 1)).to(tl.int64)
    end_g = tl.load(offsets + g).to(tl.int64)
    M_g = end_g - start_g
    
    if tile_m_idx * BLOCK_SIZE_M >= M_g:
        return
        
    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_mask_m = offs_am[:, None] < M_g
    a_base = a_ptr + start_g * lda
    a_ptrs = a_base + offs_am[:, None] * lda + offs_k[None, :]
    
    b_base = b_ptr + g * N * K
    if TRANS_B:
        ldb = K
        b_ptrs = b_base + offs_bn[:, None] * ldb + offs_k[None, :]
        b_k_step = BLOCK_SIZE_K
    else:
        ldb = N
        b_ptrs = b_base + offs_k[:, None] * ldb + offs_bn[None, :]
        b_k_step = BLOCK_SIZE_K * ldb
        
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        g_k = kk * BLOCK_SIZE_K + offs_k
        a_mask = a_mask_m & (g_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        if TRANS_B:
            b_mask = (offs_bn[:, None] < N) & (g_k[None, :] < K)
            b = tl.trans(tl.load(b_ptrs, mask=b_mask, other=0.0))
        else:
            b_mask = (offs_bn[None, :] < N) & (g_k[:, None] < K)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += b_k_step
        
    y_val = accumulator.to(BLOCK_DTYPE)
    
    # Store to Y_permuted for backward pass usage
    offs_ym = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_yn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = (y_ptr + start_g * ldy) + ldy * offs_ym[:, None] + offs_yn[None, :]
    tl.store(y_ptrs, y_val, mask=(offs_ym[:, None] < M_g) & (offs_yn[None, :] < N))
    
    # Scale and direct atomic write back to unpermuted output
    dest = start_g + offs_ym
    m_valid = offs_ym < M_g
    
    t = tl.load(sorted_id_map_ptr + dest, mask=m_valid, other=0).to(tl.int64)
    k = tl.load(slot_map_ptr + dest, mask=m_valid, other=0).to(tl.int64)
    
    p = tl.load(prob_ptr + t * stride_prob_r + k * stride_prob_c, mask=m_valid, other=0.0)
    val = accumulator * p[:, None]
    val = val.to(BLOCK_DTYPE)
    
    out_ptrs = out_ptr + t[:, None] * ldout + offs_yn[None, :]
    tl.atomic_add(out_ptrs, val, mask=m_valid[:, None] & (offs_yn[None, :] < N))
    # ponytail: Use atomic_add for MoE topK unpermute since multiple experts write to the same token.
    # Meets standard GEMM boundary condition checks.
    # (Actually we do atomic_add to handle topK > 1 where same token mapped to different experts).
    # Since out_ptr is pre-zeroed, atomic_add accumulates the routed experts correctly.


@triton.jit
def _fused_gemm_unpermute_bwd_grad_prob_kernel(
    g_ptr, y_ptr, dp_ptr, sorted_id_map_ptr, slot_map_ptr,
    N, ldg, ldy, stride_dp_r, stride_dp_c,
    num_total, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0).to(tl.int64)
    if j < num_total:
        t = tl.load(sorted_id_map_ptr + j).to(tl.int64)
        k = tl.load(slot_map_ptr + j).to(tl.int64)
        
        acc = 0.0
        for c0 in range(0, tl.cdiv(N, BLOCK_N)):
            cols = c0 * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = cols < N
            g_val = tl.load(g_ptr + t * ldg + cols, mask=mask, other=0.0).to(tl.float32)
            y_val = tl.load(y_ptr + j * ldy + cols, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(g_val * y_val)
            
        tl.store(dp_ptr + t * stride_dp_r + k * stride_dp_c, acc.to(dp_ptr.dtype.element_ty))


@triton.jit
def _fused_gemm_unpermute_bwd_gradY_kernel(
    g_ptr, prob_ptr, gy_ptr, sorted_id_map_ptr, slot_map_ptr,
    N, ldg, lgy, stride_prob_r, stride_prob_c,
    num_total, BLOCK_N: tl.constexpr,
):
    j = tl.program_id(0).to(tl.int64)
    if j < num_total:
        t = tl.load(sorted_id_map_ptr + j).to(tl.int64)
        k = tl.load(slot_map_ptr + j).to(tl.int64)
        
        p = tl.load(prob_ptr + t * stride_prob_r + k * stride_prob_c)
        
        for c0 in range(0, tl.cdiv(N, BLOCK_N)):
            cols = c0 * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = cols < N
            g_val = tl.load(g_ptr + t * ldg + cols, mask=mask, other=0.0)
            gy_val = g_val * p
            tl.store(gy_ptr + j * lgy + cols, gy_val, mask=mask)


@triton.jit
def _grouped_matmul_gradB_try_kernel(
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
    
    if g == 0:
        start_g = tl.cast(0, tl.int64)
    else:
        start_g = tl.load(offsets + (g - 1)).to(tl.int64)
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


@triton.jit
def _grouped_matmul_gradA_try_kernel(
    g_ptr, w_ptr, dx_ptr, offsets, group_size,
    N: tl.constexpr, K: tl.constexpr,
    ldg: tl.constexpr, lddx: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    TRANS_B: tl.constexpr, BLOCK_DTYPE: tl.constexpr,
):
    tile_m_idx = tl.program_id(0).to(tl.int64)
    tile_n_idx = tl.program_id(1).to(tl.int64)
    g = tl.program_id(2).to(tl.int64)
    
    if g == 0:
        start_g = tl.cast(0, tl.int64)
    else:
        start_g = tl.load(offsets + (g - 1)).to(tl.int64)
    end_g = tl.load(offsets + g).to(tl.int64)
    M_g = end_g - start_g
    
    if tile_m_idx * BLOCK_SIZE_M >= M_g:
        return
        
    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_mask_m = offs_am[:, None] < M_g
    a_base = g_ptr + start_g * ldg
    a_ptrs = a_base + offs_am[:, None] * ldg + offs_k[None, :]
    
    b_base = w_ptr + g * N * K
    if TRANS_B:
        ldb = K
        b_ptrs = b_base + offs_k[:, None] * ldb + offs_bn[None, :]
        b_k_step = BLOCK_SIZE_K * ldb
    else:
        ldb = N
        b_ptrs = b_base + offs_bn[:, None] * ldb + offs_k[None, :]
        b_k_step = BLOCK_SIZE_K
        
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kk in range(0, tl.cdiv(N, BLOCK_SIZE_K)):
        g_k = kk * BLOCK_SIZE_K + offs_k
        a_mask = a_mask_m & (g_k[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        if TRANS_B:
            b_mask = (g_k[:, None] < N) & (offs_bn[None, :] < K)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        else:
            b_mask = (offs_bn[:, None] < K) & (g_k[None, :] < N)
            b = tl.trans(tl.load(b_ptrs, mask=b_mask, other=0.0))
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += b_k_step
        
    dx = accumulator.to(BLOCK_DTYPE)
    dx_ptrs = (dx_ptr + start_g * lddx) + lddx * offs_am[:, None] + offs_bn[None, :]
    tl.store(dx_ptrs, dx, mask=(offs_am[:, None] < M_g) & (offs_bn[None, :] < K))


class FusedGEMMUnpermuteTryFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X_permuted, W, base, sorted_id_map, slot_map, prob, num_tokens, topK, trans_b=True):
        num_out = X_permuted.shape[0]
        N = W.shape[1] if trans_b else W.shape[2]
        K = X_permuted.shape[1]
        E = W.shape[0]
        
        offsets = torch.empty_like(base)
        offsets[:-1] = base[1:]
        offsets[-1] = num_out
        
        Y_permuted = torch.empty((num_out, N), device=X_permuted.device, dtype=X_permuted.dtype)
        out = torch.zeros((num_tokens, N), device=X_permuted.device, dtype=X_permuted.dtype)
        
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        
        grid = (triton.cdiv(num_out, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), E)
        
        _fused_gemm_unpermute_try_kernel[grid](
            X_permuted, W, out, Y_permuted, offsets, sorted_id_map, slot_map, prob,
            N, K,
            X_permuted.stride(0), out.stride(0), Y_permuted.stride(0),
            prob.stride(0), prob.stride(1),
            E,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
            TRANS_B=trans_b, BLOCK_DTYPE=_TL_DTYPE[X_permuted.dtype],
        )
        
        ctx.save_for_backward(X_permuted, W, base, sorted_id_map, slot_map, prob, Y_permuted)
        ctx.trans_b = trans_b
        return out

    @staticmethod
    def backward(ctx, grad_out):
        X_permuted, W, base, sorted_id_map, slot_map, prob, Y_permuted = ctx.saved_tensors
        trans_b = ctx.trans_b
        
        num_total = X_permuted.shape[0]
        N = Y_permuted.shape[1]
        K = X_permuted.shape[1]
        E = W.shape[0]
        
        # 1. Compute grad_prob
        grad_prob = torch.empty_like(prob)
        _fused_gemm_unpermute_bwd_grad_prob_kernel[(num_total,)](
            grad_out, Y_permuted, grad_prob, sorted_id_map, slot_map,
            N, grad_out.stride(0), Y_permuted.stride(0), grad_prob.stride(0), grad_prob.stride(1),
            num_total, BLOCK_N=1024,
        )
        
        # 2. Compute grad_Y_permuted
        grad_Y_permuted = torch.empty_like(Y_permuted)
        _fused_gemm_unpermute_bwd_gradY_kernel[(num_total,)](
            grad_out, prob, grad_Y_permuted, sorted_id_map, slot_map,
            N, grad_out.stride(0), grad_Y_permuted.stride(0), prob.stride(0), prob.stride(1),
            num_total, BLOCK_N=1024,
        )
        
        offsets = torch.empty_like(base)
        offsets[:-1] = base[1:]
        offsets[-1] = num_total

        # 3. Compute grad_W
        grad_W = torch.zeros_like(W)
        grid_W = lambda META: (E, triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(K, META['BLOCK_SIZE_K']))
        _grouped_matmul_gradB_try_kernel[grid_W](
            X_permuted, grad_Y_permuted, grad_W, offsets, E,
            K, N,
            X_permuted.stride(0), grad_Y_permuted.stride(0), grad_W.stride(0),
            BLOCK_SIZE_K=128, BLOCK_SIZE_N=128, BLOCK_SIZE_M=32,
            TRANS_B=trans_b, BLOCK_DTYPE=_TL_DTYPE[W.dtype],
        )
        
        # 4. Compute grad_X_permuted
        grad_X_permuted = torch.empty_like(X_permuted)
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        grid_X = (triton.cdiv(num_total, BLOCK_SIZE_M), triton.cdiv(K, BLOCK_SIZE_N), E)
        _grouped_matmul_gradA_try_kernel[grid_X](
            grad_Y_permuted, W, grad_X_permuted, offsets, E,
            N, K,
            grad_Y_permuted.stride(0), grad_X_permuted.stride(0),
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
            TRANS_B=trans_b, BLOCK_DTYPE=_TL_DTYPE[X_permuted.dtype],
        )
        
        return grad_X_permuted, grad_W, None, None, None, grad_prob, None, None, None


def fused_gemm_unpermute_try(X_permuted, W, base, sorted_id_map, slot_map, prob, num_tokens, topK, trans_b=True):
    return FusedGEMMUnpermuteTryFunction.apply(X_permuted, W, base, sorted_id_map, slot_map, prob, num_tokens, topK, trans_b)
