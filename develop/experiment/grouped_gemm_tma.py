"""
TMA grouped GEMM — Triton reference kernel (sm90+ only).

Uses tl.make_tensor_descriptor (TMA) for bulk async copies. Requires compute
capability >= 9 (supports_tma()); on sm80 it cannot run — use the plain variant
in grouped_gemm.py instead. The two modules share the same public shape:

    group_gemm_tma_fn(A, B, offsets, trans_b=True) -> C

    A:       [total_tokens, K]             (all groups stacked along M)
    B:       [E, N, K] if trans_b else [E, K, N]   (per-expert weights)
    offsets: [E]                            cumulative group offsets
    C:       [total_tokens, N]              (stacked output)

Same A+C improvements as the plain kernel:
  (A) the kernel consumes the full A/B/C tensors + offsets directly (no
      pre-split pointer arrays); start_g is threaded through the group loop;
      NUM_SM is pinned to the physical SM count (already outside the autotune
      space), so no SM idles.
  (C) partial tiles: TMA load/store cannot be masked, so group_gemm_tma_fn pads
      every dim to a multiple of the (max) block size, runs full tiles, then
      trims the padding back out of C. This is the TMA equivalent of the plain
      kernel's boundary masks. Ceiling: autotune block sizes are BM,BK<=128,
      BN<=256 — bump the PAD_* constants if tma_configs grows beyond that.

Port of triton/course/.../09-grouped-gemm (NVIDIA, MIT).
"""
from typing import Optional

import torch
import triton
import triton.language as tl

from .grouped_gemm import is_cuda, num_sms

# torch dtype -> tl.* counterpart for the constexpr kernel param
_TL_DTYPE = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def _cdiv(a, b):
    return (a + b - 1) // b


tma_configs = [
    triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, 'BLOCK_SIZE_K': BK}, num_stages=s, num_warps=w)  # noqa: E501
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

            # TMA descriptors built from the full-tensor base pointers. M_g / N
            # / K are already multiples of the block sizes (padded in the
            # wrapper), so every load/store block is fully in-bounds.
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

                offs_am = tile_m_idx * BLOCK_SIZE_M
                offs_bn = tile_n_idx * BLOCK_SIZE_N

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                    a = a_desc.load([offs_am, kk * BLOCK_SIZE_K])
                    b = b_desc.load([offs_bn, kk * BLOCK_SIZE_K])
                    # a is [M, K]; b is [N, K] (TRANS_B) -> b.T is [K, N],
                    # or b is [K, N] (not TRANS_B). Either way A @ B[e].
                    if TRANS_B:
                        accumulator += tl.dot(a, b.T)
                    else:
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
    offsets: [E]                 cumulative group offsets
    trans_b: B layout flag (see group_gemm_fn).

    dtype is taken from A (and must match B); bf16 is the MoE default.
    Returns a single stacked C [total_tokens, N].

    Requires sm90+ (supports_tma()).
    """
    assert supports_tma()
    assert A.dtype == B.dtype, "A and B must share the same dtype"
    dtype = A.dtype
    total, K = A.shape
    E = B.shape[0]
    N = B.shape[1] if trans_b else B.shape[2]
    assert offsets.numel() == E
    assert int(offsets[-1]) == total, "offsets[-1] must equal A.shape[0]"

    starts = torch.cat([torch.zeros(1, device=offsets.device, dtype=offsets.dtype),
                        offsets[:-1]])

    # (C) TMA load/store are unmasked, so partial tiles would read/write OOB.
    # Pad every dim to a multiple of the (max) block size; the kernel then runs
    # only full tiles and we trim the padding out of C afterward. Ceiling:
    # autotune block sizes are BM,BK<=128, BN<=256 — bump these if tma_configs
    # grows.
    PAD_M, PAD_N, PAD_K = 128, 256, 128
    Ms = [int(offsets[g]) - int(starts[g]) for g in range(E)]
    Mpads = [_cdiv(m, PAD_M) * PAD_M for m in Ms]
    Npad = _cdiv(N, PAD_N) * PAD_N
    Kpad = _cdiv(K, PAD_K) * PAD_K

    # Pad A per group (rows to Mpad, K to Kpad), then stack.
    A_parts = []
    for g in range(E):
        a_g = A[int(starts[g]):int(offsets[g])]            # [M_g, K]
        if Kpad > K:
            a_g = torch.cat(
                [a_g, torch.zeros(a_g.shape[0], Kpad - K, device=A.device, dtype=dtype)], 1)
        if Mpads[g] > a_g.shape[0]:
            a_g = torch.cat(
                [a_g, torch.zeros(Mpads[g] - a_g.shape[0], Kpad, device=A.device, dtype=dtype)], 0)
        A_parts.append(a_g)
    A2 = torch.cat(A_parts, 0)                             # [sum Mpad, Kpad]
    offsets2 = torch.cat([
        torch.zeros(1, device=offsets.device, dtype=offsets.dtype),
        torch.tensor(Mpads, device=offsets.device, dtype=offsets.dtype).cumsum(0),
    ]).to(offsets.dtype)

    # Pad B per expert (N and K).
    if trans_b:
        B2 = torch.zeros(E, Npad, Kpad, device=B.device, dtype=dtype)
        B2[:, :N, :K] = B
    else:
        B2 = torch.zeros(E, Kpad, Npad, device=B.device, dtype=dtype)
        B2[:, :K, :N] = B

    C2 = torch.zeros(int(offsets2[-1]), Npad, device=A.device, dtype=dtype)

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    tl_dtype = _TL_DTYPE[dtype]
    nsm = num_sms()
    grid = lambda META: (nsm, )
    grouped_matmul_tma_kernel[grid](
        A2, B2, C2, offsets2, E, Npad, Kpad,
        A2.stride(0), C2.stride(0),
        NUM_SM=nsm, TRANS_B=trans_b, BLOCK_DTYPE=tl_dtype,
    )

    # Trim padding: keep only the real [M_g, N] rows of each group.
    C = torch.empty(total, N, device=A.device, dtype=dtype)
    for g in range(E):
        cg_start = int(offsets2[g]) - Mpads[g]
        cg_end = int(offsets2[g])
        C[int(starts[g]):int(offsets[g])] = C2[cg_start:cg_end, :N]
    return C
