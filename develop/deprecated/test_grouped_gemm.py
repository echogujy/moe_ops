"""
Correctness + performance comparison of the Triton grouped GEMM.

All providers run in bfloat16 (the MoE precision, and the only dtype the C++
grouped_gemm backend supports). They share the SAME stacked input signature:

    A:       [total_tokens, K]   (all groups stacked along M)
    B:       [E, N, K]           (per-expert weights, last two dims transposed)
    offsets: [E]                 cumulative group offsets (offsets[e] = end row
                                 of group e in A; starts at 0)
    -> C:    [total_tokens, N]   (stacked output)

Compared:
  * Triton plain  — grouped_gemm.group_gemm_fn            (sm80+)
  * Triton TMA    — grouped_gemm_tma.group_gemm_tma_fn     (sm90+; skipped on sm80)
  * torch native  — torch.nn.functional.grouped_mm        (takes B as [E, K, N])
  * C++ library   — grouped_gemm_backend via grouped_gemm_custom_op.gmm
                    (gmm(A, B, batch_sizes, trans_b=True), batch_sizes = tokens/expert)

Reference per group:  C_e = A[offsets[e-1]:offsets[e]] @ B[e].T   (B[e] is [N, K]).

Run:
    source .venv/bin/activate
    python -m pytest tests/test_grouped_gemm.py -s   # or: python tests/test_grouped_gemm.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch

try:
    import grouped_gemm_backend  # noqa: F401  (the C++ backend; optional, not shipped)
    from step_mini.modules.grouped_gemm_custom_op import gmm as cpp_gmm
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

from triton_gmm_ops.grouped_gemm import group_gemm_fn, grouped_matmul_kernel, grouped_gemm, _TL_DTYPE
from triton_gmm_ops.grouped_gemm_tma import group_gemm_tma_fn, supports_tma

DEVICE = "cuda"
DTYPE = torch.bfloat16  # C++ backend is bf16-only; bf16 is the MoE precision


def _bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def _make_problem(E, total_tokens, N, K, multiple=128):
    """Deterministic per-expert token counts (multiples of `multiple`)."""
    per = max(multiple, (total_tokens // E // multiple) * multiple)
    ms = []
    for i in range(E):
        d = ((i * 37) % 8 - 4) * multiple
        ms.append(max(multiple, per + d))
    # pin N, K to multiples of the tile size so the unmasked kernel stays in-bounds
    N = (N // multiple) * multiple
    K = (K // multiple) * multiple
    return ms, N, K


def _build(E, ms, N, K):
    # B is [E, N, K] (last two dims transposed) — matches the C++ `gmm` layout
    B = torch.randn(E, N, K, device=DEVICE, dtype=DTYPE)
    A = torch.randn(sum(ms), K, device=DEVICE, dtype=DTYPE)  # stacked along M
    bs = torch.tensor(ms, device=DEVICE, dtype=torch.int32)
    offs = torch.cumsum(bs, 0).to(torch.int32)               # [E]
    starts = torch.cat([torch.zeros(1, device=DEVICE, dtype=torch.int32), offs[:-1]])
    # reference per group: C_e = A[start_e:end_e] @ B[e].T
    ref = [A[int(starts[e]):int(offs[e])] @ B[e].T for e in range(E)]
    # C++ takes batch_sizes; torch-native / Triton take the cumulative offsets
    b_cpp = B                                            # [E, N, K] for cpp_gmm trans_b=True
    w_native = B.transpose(1, 2).contiguous()            # [E, K, N] for F.grouped_mm
    return A, B, bs, offs, starts, ref, b_cpp, w_native


def _split(C, offs, starts):
    return [C[int(starts[e]):int(offs[e])] for e in range(offs.numel())]


def _check(name, out_stacked, offs, starts, ref, threshold=2.0):
    outs = _split(out_stacked, offs, starts)
    max_diff = 0.0
    for c, r in zip(outs, ref):
        d = (c.float() - r.float()).abs().max().item()
        max_diff = max(max_diff, d)
    ok = max_diff < threshold
    print(f"    [{name:>12}] max_diff={max_diff:.3f}  {'OK' if ok else 'FAIL'}")
    assert ok, f"{name} correctness failed (max_diff={max_diff:.3f})"


def test_correctness():
    print("=" * 64)
    print("Grouped GEMM - correctness (bf16, stacked API)")
    print("=" * 64)
    for (E, tot, N, K) in [(4, 2048, 1024, 512), (16, 4096, 2048, 1024), (64, 8192, 1024, 1024)]:
        ms, N, K = _make_problem(E, tot, N, K)
        A, B, bs, offs, starts, ref, b_cpp, w_native = _build(E, ms, N, K)
        print(f"  E={E} total={int(offs[-1])} N={N} K={K}")
        # Triton plain: trans_b=True (B is [E, N, K])
        tri_plain = group_gemm_fn(A, B, offs)
        _check("triton-plain (trans_b=T)", tri_plain, offs, starts, ref)
        # Triton plain: trans_b=False (B is [E, K, N])
        ref_nt = [A[int(starts[e]):int(offs[e])] @ w_native[e] for e in range(E)]
        tri_plain_f = group_gemm_fn(A, w_native, offs, trans_b=False)
        _check("triton-plain (trans_b=F)", tri_plain_f, offs, starts, ref_nt)
        # Triton TMA
        if supports_tma():
            tri_tma = group_gemm_tma_fn(A, B, offs)
            _check("triton-tma (trans_b=T)", tri_tma, offs, starts, ref)
            ref_nt = [A[int(starts[e]):int(offs[e])] @ w_native[e] for e in range(E)]
            tri_tma_f = group_gemm_tma_fn(A, w_native, offs, trans_b=False)
            _check("triton-tma (trans_b=F)", tri_tma_f, offs, starts, ref_nt)
        else:
            print("    [triton-tma ] SKIP (sm80, needs sm90)")
        # C++ library: gmm(A, B[E,N,K], batch_sizes, trans_b=True)
        if _HAS_CPP:
            cpp_out = cpp_gmm(A, b_cpp, bs, trans_b=True)
            _check("c++-gmm", cpp_out, offs, starts, ref)
        # torch native: grouped_mm(A, B[E,K,N], offs=offs)
        nat_out = torch.nn.functional.grouped_mm(A, w_native, offs=offs)
        _check("torch-native", nat_out, offs, starts, ref)


def test_performance():
    print("=" * 64)
    print("Grouped GEMM - performance (bf16, kernel-only where possible)")
    print("=" * 64)
    tl_dtype = _TL_DTYPE[DTYPE]
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    for (E, tot, N, K) in [
        (16, 8192, 4096, 4096),
        (64, 16384, 7168, 4096),
        (64, 32768, 4096, 4096),
    ]:
        ms, N, K = _make_problem(E, tot, N, K)
        A, B, bs, offs, starts, ref, b_cpp, w_native = _build(E, ms, N, K)
        gsz = E
        total = int(offs[-1])
        flops = 2 * total * N * K / 1e12  # TFLOP per group-set

        # ---- Triton: both variants consume the full A/B/C tensors directly. ----
        C = torch.empty(total, N, device=DEVICE, dtype=DTYPE)
        # Triton plain: full tensors, no pre-split (autotune). NUM_SM pinned
        # to the physical SM count so no SM idles.
        grid = lambda META: (nsm, )
        grouped_matmul_kernel[grid](A, B, C, offs, gsz, N, K,
                                     A.stride(0), C.stride(0),
                                     NUM_SM=nsm, BLOCK_DTYPE=tl_dtype, TRANS_B=True)
        t_tri = _bench(lambda: grouped_matmul_kernel[grid](
            A, B, C, offs, gsz, N, K,
            A.stride(0), C.stride(0),
            NUM_SM=nsm, BLOCK_DTYPE=tl_dtype, TRANS_B=True))

        row = f"  E={E} total={total} N={N} K={K}: Triton-plain {t_tri:7.3f} ms ({flops/t_tri*1e3:5.1f} TFLOP/s)"
        if supports_tma():
            group_gemm_tma_fn(A, B, offs)  # warm
            t_tma = _bench(lambda: group_gemm_tma_fn(A, B, offs))
            row += f" | Triton-TMA {t_tma:7.3f} ms ({flops/t_tma*1e3:5.1f} TFLOP/s)"
        if _HAS_CPP:
            cpp_gmm(A, b_cpp, bs, trans_b=True)  # warm
            t_c = _bench(lambda: torch.ops.gg.gmm_raw(A, b_cpp, bs, False, True))
            row += f" | C++ {t_c:7.3f} ms ({flops/t_c*1e3:5.1f} TFLOP/s)"
        torch.nn.functional.grouped_mm(A, w_native, offs=offs)
        t_n = _bench(lambda: torch.nn.functional.grouped_mm(A, w_native, offs=offs))
        row += f" | torch-native {t_n:7.3f} ms ({flops/t_n*1e3:5.1f} TFLOP/s)"
        print(row)


def test_performance_backward():
    """Backward perf: our grouped_gemm.backward() (grad_A = 1 grouped GEMM,
    grad_B = per-expert loop) vs torch-native / C++ autograd backends.

    grad_A and grad_B are each ~1 forward's FLOPs, so total backward ~= 2x
    forward; bw_flops below is 4*total*N*K.
    """
    print("=" * 64)
    print("Grouped GEMM - backward performance (bf16)")
    print("=" * 64)
    for (E, tot, N, K) in [
        (16, 8192, 4096, 4096),
        (64, 16384, 7168, 4096),
        (64, 32768, 4096, 4096),
    ]:
        ms, N, K = _make_problem(E, tot, N, K)
        A, B, bs, offs, starts, ref, b_cpp, w_native = _build(E, ms, N, K)
        total = int(offs[-1])
        grad_C = torch.randn(total, N, device=DEVICE, dtype=DTYPE)
        bw_flops = 4 * total * N * K / 1e12  # grad_A + grad_B

        # our differentiable op
        A_our = A.clone().requires_grad_(True)
        B_our = B.clone().requires_grad_(True)
        C_our = grouped_gemm(A_our, B_our, offs, trans_b=True)
        C_our.backward(grad_C, retain_graph=True)  # warm
        t_our = _bench(lambda: C_our.backward(grad_C, retain_graph=True))
        row = (f"  E={E} total={total} N={N} K={K}: "
               f"our {t_our:7.3f} ms ({bw_flops/t_our*1e3:5.1f} TFLOP/s)")
        if _HAS_CPP:
            A_c = A.clone().requires_grad_(True)
            B_c = b_cpp.clone().requires_grad_(True)
            C_c = cpp_gmm(A_c, B_c, bs, trans_b=True)
            C_c.backward(grad_C, retain_graph=True)  # warm
            t_c = _bench(lambda: C_c.backward(grad_C, retain_graph=True))
            row += f" | C++ {t_c:7.3f} ms ({bw_flops/t_c*1e3:5.1f} TFLOP/s)"
        A_n = A.clone().requires_grad_(True)
        B_n = w_native.clone().requires_grad_(True)
        C_n = torch.nn.functional.grouped_mm(A_n, B_n, offs=offs)
        C_n.backward(grad_C, retain_graph=True)  # warm
        t_n = _bench(lambda: C_n.backward(grad_C, retain_graph=True))
        row += f" | torch-native {t_n:7.3f} ms ({bw_flops/t_n*1e3:5.1f} TFLOP/s)"
        print(row)


def test_correctness_partial_tiles():
    """Edge-tile check: sizes NOT multiples of the tile block must still be
    correct. Exercises the boundary masks (zero-pad on load, skip on store);
    without them the kernel reads/writes out of bounds and corrupts neighbours.
    """
    print("=" * 64)
    print("Grouped GEMM - partial tiles (non-128 sizes, bf16)")
    print("=" * 64)
    torch.manual_seed(1)
    E = 4
    ms = [100, 250, 130, 300]          # not multiples of 128 -> partial M tiles
    N, K = 200, 130                    # not multiples of 128 -> partial N/K
    total = sum(ms)
    B = torch.randn(E, N, K, device=DEVICE, dtype=DTYPE)
    A = torch.randn(total, K, device=DEVICE, dtype=DTYPE)
    bs = torch.tensor(ms, device=DEVICE, dtype=torch.int32)
    offs = torch.cumsum(bs, 0).to(torch.int32)
    starts = torch.cat([torch.zeros(1, device=DEVICE, dtype=torch.int32), offs[:-1]])
    # trans_b = True (B is [E, N, K])
    ref = [A[int(starts[e]):int(offs[e])] @ B[e].T for e in range(E)]
    _check("partial (trans_b=T)", group_gemm_fn(A, B, offs), offs, starts, ref)
    # trans_b = False (B is [E, K, N])
    B_nt = B.transpose(1, 2).contiguous()
    ref_nt = [A[int(starts[e]):int(offs[e])] @ B_nt[e] for e in range(E)]
    _check("partial (trans_b=F)", group_gemm_fn(A, B_nt, offs, trans_b=False),
           offs, starts, ref_nt)


def test_backward():
    """Backward = analytic grad vs a torch-autograd reference built from
    per-expert GEMMs. Both paths are bf16 GEMMs on identical inputs, so the
    diff should be ~bf16 noise (not a correctness bug).
    """
    print("=" * 64)
    print("Grouped GEMM - backward (analytic vs torch ref, bf16)")
    print("=" * 64)
    torch.manual_seed(0)
    E, N, K = 4, 256, 128
    ms = [128, 256, 192, 320]          # not all multiples of 128 -> exercises masks
    total = sum(ms)
    offs = torch.cumsum(torch.tensor(ms, device=DEVICE, dtype=torch.int32), 0)
    starts = torch.cat([torch.zeros(1, device=DEVICE, dtype=torch.int32), offs[:-1]])
    grad_C = torch.randn(total, N, device=DEVICE, dtype=DTYPE)

    for trans_b in (True, False):
        A0 = torch.randn(total, K, device=DEVICE, dtype=DTYPE)
        if trans_b:
            B0 = torch.randn(E, N, K, device=DEVICE, dtype=DTYPE)
        else:
            B0 = torch.randn(E, K, N, device=DEVICE, dtype=DTYPE)

        # reference: per-expert torch autograd
        A_ref = A0.detach().clone().requires_grad_(True)
        B_ref = B0.detach().clone().requires_grad_(True)
        C_ref = torch.zeros(total, N, device=DEVICE, dtype=DTYPE)
        for e in range(E):
            a_e = A_ref[int(starts[e]):int(offs[e])]
            C_ref[int(starts[e]):int(offs[e])] = a_e @ B_ref[e].T if trans_b else a_e @ B_ref[e]
        C_ref.backward(grad_C)
        ref_grad_A, ref_grad_B = A_ref.grad, B_ref.grad

        # our differentiable op
        A_our = A0.detach().clone().requires_grad_(True)
        B_our = B0.detach().clone().requires_grad_(True)
        C_our = grouped_gemm(A_our, B_our, offs, trans_b=trans_b)
        C_our.backward(grad_C)

        da = (A_our.grad.float() - ref_grad_A.float()).abs().max().item()
        db = (B_our.grad.float() - ref_grad_B.float()).abs().max().item()
        name = f"trans_b={trans_b}"
        print(f"    [{name:>10}] grad_A max_diff={da:.3f}  grad_B max_diff={db:.3f}  "
              f"{'OK' if da < 2.0 and db < 2.0 else 'FAIL'}")
        assert da < 2.0 and db < 2.0, f"backward failed ({name})"


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return
    test_correctness()
    test_correctness_partial_tiles()
    test_backward()
    test_performance()
    test_performance_backward()
    print("=" * 64)
    print("ALL grouped GEMM checks passed.")
    print("=" * 64)


if __name__ == "__main__":
    main()
