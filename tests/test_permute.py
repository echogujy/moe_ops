import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import torch
from triton_gmm_ops.permute import permute, permute_backward, permute_countsort
from triton_gmm_ops.unpermute import unpermute_forward

try:
    import grouped_gemm_backend as _C
    from step_mini.modules.grouped_gemm_custom_op import permute as cpp_permute
    _HAS_REF = True
except ImportError:
    _HAS_REF = False
    cpp_permute = None


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


def test_permute_correctness(num_tokens, num_cols, topK, E, dtype=torch.bfloat16):
    dev = torch.device("cuda")
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=dtype)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)

    # Call the simplified permute autograd function
    perm_t, rid_t, base_t = permute(x, idx)

    # Compute expected offsets
    flat = idx.reshape(-1)
    counts = torch.bincount(flat, minlength=E)
    base = torch.zeros(E, dtype=torch.long, device=dev)
    base[1:] = counts.cumsum(0)[:-1]

    # Validate offsets
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
            ok = ok and torch.allclose(a, b, atol=1e-4, rtol=1e-4)
        ref_src = "C++ _C.permute"
        assert ok, f"permute mismatch against {ref_src}, max_diff={max_diff:.2e}"
    else:
        ref_src = "pure check"

    # Round-trip check
    prob = torch.full((num_tokens, topK), 1.0 / topK, device=dev)
    out = unpermute_forward(perm_t, rid_t, prob, num_tokens, topK)
    assert torch.allclose(out, x, atol=1e-4, rtol=1e-4), "round-trip check failed"
    print(f"  [permute correctness] ({num_tokens}x{num_cols}, topK={topK}, E={E}) vs {ref_src}: PASS")


def test_permute_backward_correctness(num_tokens, num_cols, topK, E, dtype=torch.bfloat16):
    dev = torch.device("cuda")
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=dtype)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)
    perm_t, rid_t, base_t = permute_countsort(x, idx)
    grad_perm = torch.randn_like(perm_t)

    # Triton backward
    g_input_tri = permute_backward(grad_perm, rid_t, num_tokens, topK)

    # Analytical reference
    g_input_ref = torch.zeros_like(x)
    rmap = rid_t.reshape(topK, num_tokens)
    for k in range(topK):
        src = rmap[k]
        valid = src >= 0
        g_input_ref += grad_perm[src.clamp(min=0)] * valid[:, None].to(grad_perm.dtype)

    assert torch.allclose(g_input_tri, g_input_ref, atol=1e-2, rtol=1e-2), "backward analytical check failed"
    print(f"  [permute backward correctness] ({num_tokens}x{num_cols}, topK={topK}, E={E}): PASS")


def bench_permute_op(num_tokens, num_cols, topK, E):
    dev = torch.device("cuda")
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)
    num_out = num_tokens * topK
    bytes_io = (num_out * num_cols + num_tokens * topK * num_cols) * 2

    t_tri = _bench(lambda: permute(x, idx))
    row = f"  [bench permute] {num_tokens}x{num_cols} topK={topK}: Triton {t_tri:7.3f} ms ({bytes_io/1e9/t_tri*1e3:6.1f} GB/s)"

    if _HAS_REF:
        _, _, ws = _C.permute(x, idx, num_out, [], num_tokens * topK)
        t_c = _bench(lambda: _C.permute(x, idx, num_out, ws, num_tokens * topK))
        row += f" | C++ {t_c:7.3f} ms ({bytes_io/1e9/t_c*1e3:6.1f} GB/s) | speedup {t_c/t_tri:5.2f}x"
    print(row)


if __name__ == "__main__":
    print("================================================================")
    print("Correctness Check - Permute:")
    print("================================================================")
    test_permute_correctness(4096, 4096, 2, 16)
    test_permute_correctness(8192, 4096, 2, 16)
    test_permute_backward_correctness(4096, 4096, 2, 16)
    test_permute_backward_correctness(8192, 4096, 2, 16)
    print("================================================================")
    print("\n================================================================")
    print("Benchmark - Permute:")
    print("================================================================")
    bench_permute_op(4096, 4096, 2, 16)
    bench_permute_op(8192, 4096, 2, 16)
    bench_permute_op(16384, 4096, 2, 16)
    print("================================================================")
