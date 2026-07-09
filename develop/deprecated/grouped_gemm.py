"""
Plain (non-TMA) grouped GEMM — Triton reference kernel (sm80+).

Splits the official Triton grouped-GEMM tutorial (NVIDIA, MIT) into a plain
variant (this file) and a TMA variant (grouped_gemm_tma.py). A fixed number of
CTAs tile a group of GEMMs with static, on-device scheduling. No TMA
dependency, so it runs on sm80+ (the TMA variant needs sm90).

The official kernel is fp16-only; we parametrize the element dtype via
BLOCK_DTYPE so it can also run in bf16 (the MoE precision on sm80/sm90). The
public API is stacked and aligned with the C++ grouped_gemm backend and
torch.nn.functional.grouped_mm:

    group_gemm_fn(A, B, offsets, trans_b=True) -> C

    A:       [total_tokens, K]             (all groups stacked along M)
    B:       [E, N, K] if trans_b else [E, K, N]
                                             (per-expert weights; trans_b=True
                                             matches the C++ `b` / MoE layout)
    offsets: [E]                            cumulative group offsets
    C:       [total_tokens, N]              (stacked output)

dtypes are taken from A (== B); bf16 is the MoE default.
Computed as C_e = A[offsets[e-1]:offsets[e]] @ (B[e].T if trans_b else B[e]).

Port of triton/course/.../09-grouped-gemm (NVIDIA, MIT).
"""
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


# torch dtype -> tl.* counterpart for the constexpr kernel param
_TL_DTYPE = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    # full (un-split) tensors — no per-group pointer arrays
    a_ptr,            # [total_tokens, K]
    b_ptr,            # [E, N, K] if TRANS_B else [E, K, N]
    c_ptr,            # [total_tokens, N]
    offsets,          # [E] cumulative group ends (offsets[e-1] is group e's start)
    group_size,       # E
    # geometry (uniform N, K across experts — MoE layout)
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
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            a_base = a_ptr + start_g * lda
            b_base = b_ptr + g_i * N * K          # expert e sits at offset e*N*K
            c_base = c_ptr + start_g * ldc

            tile_idx_in_gemm = tile_idx - last_problem_end
            GROUP_M = 8
            tiles_per_group = GROUP_M * num_n_tiles
            group_idx = tile_idx_in_gemm // tiles_per_group
            first_tile_m = group_idx * GROUP_M
            current_group_size_m = tl.minimum(num_m_tiles - first_tile_m, GROUP_M)
            idx_in_group = tile_idx_in_gemm % tiles_per_group
            tile_m_idx = first_tile_m + (idx_in_group % current_group_size_m)
            tile_n_idx = idx_in_group // current_group_size_m

            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            # Boundary masks: the kernel no longer assumes full tiles, so edge
            # tiles (M_g / N / K not multiples of the block size) are zero-padded
            # on load and skipped on store instead of reading/writing OOB. The
            # M/N parts are constant across the K loop; the K part must use the
            # GLOBAL k index (kk*BK + offs_k), not the local offs_k, or the last
            # K-tile would read out of bounds.
            a_mask_m = offs_am[:, None] < M_g
            b_mask_n = offs_bn[None, :] < N

            a_ptrs = a_base + offs_am[:, None] * lda + offs_k[None, :]
            if TRANS_B:
                # B[e] is [N, K]: read the [N, K] block coalesced as [BN, BK].
                # K is the stride-1 axis.
                ldb = K
                b_ptrs = b_base + offs_bn[:, None] * ldb + offs_k[None, :]
                b_k_step = BLOCK_SIZE_K
            else:
                # B[e] is [K, N]: read the [K, N] block directly. K is the
                # leading dim here, so its stride is ldb=N.
                ldb = N
                b_ptrs = b_base + offs_k[:, None] * ldb + offs_bn[None, :]
                b_k_step = BLOCK_SIZE_K * ldb

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                g_k = kk * BLOCK_SIZE_K + offs_k      # global K index
                a_mask = a_mask_m & (g_k[None, :] < K)
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                if TRANS_B:
                    b_mask = (offs_bn[:, None] < N) & (g_k[None, :] < K)
                    b_coalesced = tl.load(b_ptrs, mask=b_mask, other=0.0)
                    b = tl.trans(b_coalesced)
                else:
                    b_mask = b_mask_n & (g_k[:, None] < K)
                    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
                # a is [M, K], b is [K, N] -> A @ B (or A @ B^T when TRANS_B).
                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += b_k_step

            c = accumulator.to(BLOCK_DTYPE)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_base + ldc * offs_cm[:, None] + offs_cn[None, :]
            tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M_g) & (offs_cn[None, :] < N))

            tile_idx += NUM_SM

        start_g = end_g     # becomes the next group's start row
        last_problem_end += num_tiles


def group_gemm_fn(A, B, offsets, trans_b: bool = True):
    """Stacked grouped GEMM — aligned with the C++ / torch-native APIs.

    A:       [total_tokens, K]   (all groups stacked along M)
    B:       [E, N, K] if trans_b else [E, K, N]   (per-expert weights)
    offsets: [E]                 cumulative group offsets (offsets[e] = end
                                  row of group e in A; starts at 0)
    trans_b: B layout flag. True (default): B is [E, N, K] and the kernel
             computes C_e = A[...] @ B[e].T (MoE / C++ `gmm(trans_b=True)`).
             False: B is [E, K, N] and computes C_e = A[...] @ B[e]
             (torch.nn.functional.grouped_mm layout).

    dtype is taken from A (and must match B); bf16 is the MoE default.
    Returns a single stacked C [total_tokens, N].
    """
    assert A.dtype == B.dtype, "A and B must share the same dtype"
    dtype = A.dtype
    total, K = A.shape
    E = B.shape[0]
    N = B.shape[1] if trans_b else B.shape[2]
    assert offsets.numel() == E
    assert int(offsets[-1]) == total, "offsets[-1] must equal A.shape[0]"
    C = torch.empty((total, N), device=A.device, dtype=dtype)
    nsm = num_sms()
    grid = lambda META: (nsm, )
    grouped_matmul_kernel[grid](
        A, B, C, offsets, E, N, K,
        A.stride(0), C.stride(0),
        NUM_SM=nsm, TRANS_B=trans_b, BLOCK_DTYPE=_TL_DTYPE[dtype],
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
def grouped_matmul_gradB_kernel(
    a_ptr,            # [total, K]  activation A (grad feed for grad_B)
    g_ptr,            # [total, N]  upstream grad (grad_output)
    c_ptr,            # [E, K, N]  grad_B_base[e] = A_e^T @ grad_C_e
    offsets,          # [E] cumulative group ends
    group_size,       # E
    K: tl.constexpr,
    N: tl.constexpr,
    lda: tl.constexpr,
    ldg: tl.constexpr,
    ldc: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,   # contraction (M_e) tile
    BLOCK_DTYPE: tl.constexpr,
    TRANS_B: tl.constexpr,
):
    # 2D grid: (group, output_tile). Each CTA computes one [BK, BN] tile of
    # grad_B_base[g] = A_e^T @ grad_C_e (e = group g). The CONTRACTING dim
    # is the per-expert token count M_e (variable) - handled as the matmul's
    # K-loop. This is a SEPARATE kernel from the forward one (whose
    # contraction is the fixed K and which groups along the OUTPUT M). It is
    # the grouped/batched-GEMM form torch-native uses for its grad_weight:
    # group along the contraction dim, no padding waste.
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

    # group bounds (clamp the prev-offset index so g=0 never reads OOB)
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
    k_mask = offs_k[:, None] < K     # [BK, 1]  (A's K cols)
    n_mask = offs_n[None, :] < N      # [1, BN]

    acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    for m in range(0, tl.cdiv(M_e, BLOCK_SIZE_M)):
        m_idx = m * BLOCK_SIZE_M + offs_m
        m_b = m_idx[:, None] < M_e
        # Load A_e[m, k] coalesced with shape [BM, BK], then transpose to [BK, BM]
        a_coalesced = tl.load(a_base + m_idx[:, None] * lda + offs_k[None, :],
                              mask=(m_idx[:, None] < M_e) & (offs_k[None, :] < K),
                              other=0.0)
        a = tl.trans(a_coalesced)
        gg = tl.load(g_base + m_idx[:, None] * ldg + offs_n[None, :],
                    mask=m_b & n_mask, other=0.0)         # [BM, BN] = G_e
        acc += tl.dot(a, gg)                               # [BK, BN] = A_e^T @ G_e
    c = acc.to(BLOCK_DTYPE)
    if TRANS_B:
        # Store in transposed layout directly: base[g] is [N, K], element is base[g, n, k]
        # c has shape [BK, BN] representing [k, n], so we map it to stride K for n and stride 1 for k
        c_ptrs = c_base + offs_n[None, :] * K + offs_k[:, None]
        tl.store(c_ptrs, c, mask=k_mask & n_mask)
    else:
        # Store in normal layout directly: base[g] is [K, N], element is base[g, k, n]
        # c has shape [BK, BN] representing [k, n], so we map it to stride N for k and stride 1 for n
        c_ptrs = c_base + offs_k[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, c, mask=k_mask & n_mask)


def _grad_B(A, B, offsets, grad_output, trans_b):
    """grad w.r.t. B - a single contraction-grouped GEMM (one Triton launch).

    For C_e = A_e @ (B_e^T if trans_b else B_e):
        dC/dB_e = A_e^T @ grad_C_e            (trans_b=False; B_e is [K, N])
                = (A_e^T @ grad_C_e)^T        (trans_b=True;  B_e is [N, K])
    so the kernel computes grad_B_base[e] = A_e^T @ grad_C_e  ([K, N]) and we
    transpose to [N, K] only when trans_b=True. The contracting dim is the
    per-expert token count M_e (variable) - expressed as the matmul's
    K-loop, exactly the fused grad_weight form torch-native uses (group
    along the contraction dim, no padding waste). Replaces the old
    per-expert Python loop.
    """

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
    grouped_matmul_gradB_kernel[grid](
        A, grad_output, base, offsets, E,
        K, N, A.stride(0), grad_output.stride(0), base.stride(0),
        BLOCK_DTYPE=_TL_DTYPE[A.dtype],
        TRANS_B=trans_b,
    )
    return base


class GroupedGemmFunction(torch.autograd.Function):
    """Autograd wrapper turning group_gemm_fn into a complete PyTorch op.

    Forward dispatches to the TMA kernel on sm90+, else the plain kernel.
    Backward is backend-agnostic: grad_A is another grouped GEMM, grad_B is a
    contraction-grouped Triton kernel (see _grad_B).
    """

    @staticmethod
    def forward(ctx, A, B, offsets, trans_b):
        ctx.save_for_backward(A, B, offsets)
        ctx.trans_b = trans_b
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9:
            from .grouped_gemm_tma import group_gemm_tma_fn
            return group_gemm_tma_fn(A, B, offsets, trans_b=trans_b)
        return group_gemm_fn(A, B, offsets, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad_output):
        A, B, offsets = ctx.saved_tensors
        trans_b = ctx.trans_b
        grad_A = grad_B_total = None
        if ctx.needs_input_grad[0]:
            # grad_A = grad_C @ (B_e if trans_b else B_e^T); both are [N, K].
            # .contiguous() because the kernel assumes a dense [E, K_in, N_out]
            # layout (a transposed view would carry the wrong row stride).
            Wb = (B if trans_b else B.transpose(1, 2)).contiguous()
            grad_A = group_gemm_fn(grad_output, Wb, offsets, trans_b=False)
        if ctx.needs_input_grad[1]:
            grad_B_total = _grad_B(A, B, offsets, grad_output, trans_b)
        return grad_A, grad_B_total, None, None


def grouped_gemm(A, B, offsets, trans_b: bool = True):
    """Differentiable stacked grouped GEMM (autograd-aware).

    Same signature/layout as group_gemm_fn, but usable inside torch.autograd
    (e.g. as a MoE linear weight). Returns a single stacked C [total, N].
    """
    return GroupedGemmFunction.apply(A, B, offsets, trans_b)
