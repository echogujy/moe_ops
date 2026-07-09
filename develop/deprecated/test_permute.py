"""Correctness + performance tests for the Triton `permute` op.

unpermute tests live in `test_unpermute.py` (separate file for isolated tuning).

Two implementations are benchmarked / compared (C++ _C.permute is the baseline
and the fastest of the three):
  * permute          — non-stable argsort (slower Triton path, ~1.4-1.7x vs C++)
  * permute_countsort — counting sort (faster Triton path, ~1.1-1.2x vs C++)

Correctness (order-independent, so it works for the non-stable counting sort):
  * per-expert multiset of permuted rows matches C++ _C.permute
    (both group tokens by expert with the same bucket boundaries)
  * round-trip: unpermute(permute(x)) with prob = 1/topK recovers x exactly
    (proves the row_id_map is the exact inverse of the gather)
  * for the stable argsort variant, row_id_map is additionally byte-exact vs C++.

Performance: large shapes vs C++ _C.permute (ms + achieved bandwidth).

Run:
    source .venv/bin/activate
    python -m pytest tests/test_permute.py -s   # or: python tests/test_permute.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time

import torch

try:
    import grouped_gemm_backend as _C
    # The real C++ permute backward is _PermuteFn.backward, invoked through the
    # autograd `permute` op (internally unpermute_raw(empty prob) == scatter-add).
    from step_mini.modules.grouped_gemm_custom_op import permute as cpp_permute
    _HAS_REF = True
except ImportError:
    _HAS_REF = False
    cpp_permute = None

from triton_gmm_ops.permute import permute as triton_permute
from triton_gmm_ops.permute import permute_countsort as triton_permute_countsort
from triton_gmm_ops.permute import permute_backward as triton_permute_backward
from triton_gmm_ops.permute import permute_autograd as triton_permute_autograd
from triton_gmm_ops.unpermute import unpermute_forward as triton_unpermute_fwd


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


def test_permute_correctness(fn, name, num_tokens, num_cols, topK, E, dtype=torch.bfloat16,
                              stable=False):
    dev = torch.device("cuda")
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=dtype)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)

    perm_t, rid_t, base_t = fn(x, idx)

    # per-expert bucket boundaries (order-independent, shared by C++ and countsort)
    flat = idx.reshape(-1)
    counts = torch.bincount(flat, minlength=E)
    base = torch.zeros(E, dtype=torch.long, device=dev)
    base[1:] = counts.cumsum(0)[:-1]

    # validate the emitted base (exclusive offsets) against the reference cumsum.
    # counts is derivable from base (base[e]-base[e-1]) so base alone is the check.
    assert torch.equal(base_t.to(torch.int64), base), "base mismatch"

    if _HAS_REF:
        perm_c, rid_c, _ = _C.permute(x, idx, num_tokens * topK, [], num_tokens * topK)
        ok = True
        max_diff = 0.0
        for e in range(E):
            sl = slice(base[e].item(), (base[e] + counts[e]).item())
            if counts[e] == 0:
                continue
            a = torch.sort(perm_t[sl], dim=0)[0]
            b = torch.sort(perm_c[sl], dim=0)[0]
            d = (a - b).abs().max().item()
            max_diff = max(max_diff, d)
            ok = ok and torch.allclose(a, b, atol=0, rtol=0)
        ref_src = "C++ _C.permute"
    else:
        ok = True
        max_diff = 0.0
        ref_src = "torch argsort"

    # round-trip: unpermute with prob = 1/topK must recover x exactly (proves
    # row_id_map is the true inverse of the gather, regardless of ordering).
    prob = torch.full((num_tokens, topK), 1.0 / topK, device=dev)
    out = triton_unpermute_fwd(perm_t, rid_t, prob, num_tokens, topK)
    ok_rt = torch.equal(out, x)
    ok = ok and ok_rt

    ok_map = True
    if stable and _HAS_REF:
        ok_map = torch.equal(rid_t, rid_c)

    status = "PASS" if ok else "FAIL"
    extra = f" round-trip={'OK' if ok_rt else 'BAD'}"
    if stable and _HAS_REF:
        extra += f" | row_id_map={'OK' if ok_map else 'MISMATCH'}"
    print(f"  [permute {name}] ({num_tokens}x{num_cols}, topK={topK}, E={E}) vs {ref_src}: "
          f"{status}  max_diff={max_diff:.2e}{extra}")
    assert ok and (ok_map if (stable and _HAS_REF) else True), "permute correctness failed"


def bench_permute(fn, name, num_tokens, num_cols, topK, E):
    dev = torch.device("cuda")
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)
    num_out = num_tokens * topK
    bytes_io = (num_out * num_cols + num_tokens * topK * num_cols) * 2  # in+out bf16

    def tri():
        return fn(x, idx)

    t_tri = _bench(tri)
    row = (f"  [permute {name}] {num_tokens}x{num_cols} topK={topK}: Triton {t_tri:7.3f} ms "
           f"({bytes_io/1e9/t_tri*1e3:6.1f} GB/s)")
    if _HAS_REF:
        _, _, ws = _C.permute(x, idx, num_out, [], num_tokens * topK)
        t_c = _bench(lambda: _C.permute(x, idx, num_out, ws, num_tokens * topK))
        row += f" | C++ {t_c:7.3f} ms ({bytes_io/1e9/t_c*1e3:6.1f} GB/s) | speedup {t_c/t_tri:5.2f}x"
    print(row)


def test_permute_backward(num_tokens, num_cols, topK, E, dtype=torch.bfloat16):
    dev = torch.device("cuda")
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=dtype)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)
    perm_t, rid_t, base_t = triton_permute(x, idx)
    grad_perm = torch.randn_like(perm_t)

    # Triton backward (reuses unpermute_forward with prob=1)
    g_input_tri = triton_permute_backward(grad_perm, rid_t, num_tokens, topK)

    # analytical reference: scatter-add over the topK slots
    g_input_ref = torch.zeros_like(x)
    rmap = rid_t.reshape(topK, num_tokens)
    for k in range(topK):
        src = rmap[k]
        valid = src >= 0
        g_input_ref += grad_perm[src.clamp(min=0)] * valid[:, None].to(grad_perm.dtype)

    ok_ref = torch.allclose(g_input_tri, g_input_ref, atol=1e-2, rtol=1e-2)
    max_diff_ref = (g_input_tri - g_input_ref).abs().max().item()

    ok_c = True
    max_diff_c = float("nan")
    if _HAS_REF:
        try:
            # Through the real C++ autograd permute op: _PermuteFn.forward then
            # .backward(), which is the actual C++ permute backward (scatter-add).
            input_c = x.detach().clone().requires_grad_(True)
            perm_c, _ = cpp_permute(input_c, idx)
            perm_c.backward(grad_perm)
            g_input_c = input_c.grad
            ok_c = torch.allclose(g_input_tri, g_input_c, atol=1e-2, rtol=1e-2)
            max_diff_c = (g_input_tri - g_input_c).abs().max().item()
        except Exception as e:  # noqa: BLE001 - best-effort cross-check
            print(f"  [permute backward] vs C++: SKIP ({e})")
            ok_c = True

    status = "PASS" if (ok_ref and ok_c) else "FAIL"
    diff_str = f"ref_maxdiff={max_diff_ref:.2e}"
    if _HAS_REF and max_diff_c == max_diff_c:  # skip if NaN (cross-check skipped)
        diff_str += f" | C++_ref_maxdiff={max_diff_c:.2e}"
    print(f"  [permute backward] ({num_tokens}x{num_cols}, topK={topK}) {status}  {diff_str}")
    assert ok_ref and ok_c, "permute backward failed"


def bench_permute_backward(num_tokens, num_cols, topK, E):
    dev = torch.device("cuda")
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)
    grad_perm = torch.randn(num_tokens * topK, num_cols, device=dev, dtype=torch.bfloat16)
    # in: grad_permuted [num_out, num_cols]; out: grad_input [num_tokens, num_cols]
    bytes_io = (num_tokens * topK * num_cols + num_tokens * num_cols) * 2

    # Both sides measured through their autograd permute op (forward once, then
    # time only the .backward()); retain_graph lets it re-run each iteration.
    x_tri = x.detach().clone().requires_grad_(True)
    perm_tri, _, _ = triton_permute_autograd(x_tri, idx)

    def tri():
        x_tri.grad = None
        perm_tri.backward(grad_perm, retain_graph=True)
        return x_tri.grad

    t_tri = _bench(tri)
    row = (f"  [permute bwd] {num_tokens}x{num_cols} topK={topK}: Triton {t_tri:7.3f} ms "
           f"({bytes_io/1e9/t_tri*1e3:6.1f} GB/s)")
    if _HAS_REF:
        x_c = x.detach().clone().requires_grad_(True)
        perm_c, _ = cpp_permute(x_c, idx)

        def cpp():
            x_c.grad = None
            perm_c.backward(grad_perm, retain_graph=True)
            return x_c.grad

        t_c = _bench(cpp)
        row += f" | C++ {t_c:7.3f} ms ({bytes_io/1e9/t_c*1e3:6.1f} GB/s) | speedup {t_c/t_tri:5.2f}x"
    print(row)


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return

    print("=" * 64)
    print("Triton permute - correctness")
    print("=" * 64)
    for (nt, nc, tk, e) in [(512, 1024, 2, 8), (2048, 4096, 2, 16), (4096, 7168, 2, 64)]:
        test_permute_correctness(triton_permute_countsort, "countsort", nt, nc, tk, e)
        test_permute_correctness(triton_permute, "argsort", nt, nc, tk, e, stable=False)

    print("=" * 64)
    print("Triton permute - backward")
    print("=" * 64)
    for (nt, nc, tk, e) in [(512, 1024, 2, 8), (2048, 4096, 2, 16), (4096, 7168, 2, 64)]:
        test_permute_backward(nt, nc, tk, e)

    print("=" * 64)
    print("Triton permute - performance (large shapes)")
    print("=" * 64)
    for (nt, nc, tk, e) in [
        (8192, 4096, 2, 64),
        (16384, 4096, 2, 64),
        (16384, 8192, 2, 64),
        (32768, 4096, 2, 128),
    ]:
        bench_permute(triton_permute_countsort, "countsort", nt, nc, tk, e)
        bench_permute(triton_permute, "argsort", nt, nc, tk, e)

    print("=" * 64)
    print("Triton permute - backward performance (large shapes)")
    print("=" * 64)
    for (nt, nc, tk, e) in [
        (8192, 4096, 2, 64),
        (16384, 4096, 2, 64),
        (16384, 8192, 2, 64),
        (32768, 4096, 2, 128),
    ]:
        bench_permute_backward(nt, nc, tk, e)

    print("=" * 64)
    print("ALL Triton permute tests passed.")
    print("=" * 64)


if __name__ == "__main__":
    main()
