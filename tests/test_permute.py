import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
# permute_countsort lives in the permute submodule but isn't re-exported
# at the package top level, so import it from there.
from triton_gmm_ops.permute_ops import permute, permute_backward, permute_countsort


# ---------------------------------------------------------------------------
# PyTorch-native permute reference
# ---------------------------------------------------------------------------

def native_permute(input: torch.Tensor, indices: torch.Tensor):
    """Pure-PyTorch reference for permute_countsort."""
    num_tokens, num_cols = input.shape
    topK = indices.shape[1]
    flat = indices.reshape(-1)
    E = int(flat.max().item()) + 1

    order = torch.argsort(flat.to(torch.int64), stable=True)   # group by expert

    permuted = input[order // topK]

    counts = torch.bincount(flat, minlength=E)
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=input.device)
    offsets[1:] = counts.cumsum(0).to(torch.int32)

    row_id_map = torch.full((num_tokens * topK,), -1, dtype=torch.int32, device=input.device)
    t_idx = order // topK
    k_idx = order % topK
    dest = torch.arange(num_tokens * topK, device=input.device, dtype=torch.int32)
    row_id_map[k_idx * num_tokens + t_idx] = dest

    return permuted, row_id_map, offsets


def native_permute_backward(grad_permuted: torch.Tensor, row_id_map: torch.Tensor,
                            num_tokens: int, topK: int) -> torch.Tensor:
    """Pure-PyTorch reference for permute_backward.
    grad_input[t] = sum_k grad_permuted[row_id_map[k*T + t]]
    """
    g = torch.zeros(num_tokens, grad_permuted.shape[1],
                    device=grad_permuted.device, dtype=grad_permuted.dtype)
    rmap = row_id_map.reshape(topK, num_tokens)   # [K, T]
    for k in range(topK):
        src = rmap[k]                              # [T] dest rows in grad_permuted
        valid = src >= 0                           # [T]
        gathered = grad_permuted[src.clamp(min=0)] * valid[:, None].to(grad_permuted.dtype)
        g += gathered
    return g


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_permute_correctness(num_tokens, num_cols, topK, E, dtype=torch.bfloat16):
    dev = torch.device("cuda")
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=dtype)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)

    perm_tri, rid_tri, offsets_tri = permute(x, idx)
    perm_ref, rid_ref, offsets_ref = native_permute(x, idx)

    assert torch.equal(offsets_tri.to(torch.int64), offsets_ref.to(torch.int64)), \
        f"offsets mismatch: tri={offsets_tri} ref={offsets_ref}"

    flat = idx.reshape(-1)
    counts = torch.bincount(flat, minlength=E)
    ok, max_diff = True, 0.0
    for e in range(E):
        s, end = int(offsets_ref[e]), int(offsets_ref[e + 1])
        if counts[e] == 0:
            continue
        a = torch.sort(perm_tri[s:end], dim=0)[0]
        b = torch.sort(perm_ref[s:end], dim=0)[0]
        d = (a.float() - b.float()).abs().max().item()
        max_diff = max(max_diff, d)
        ok = ok and (d == 0.0)
    assert ok, f"permuted rows mismatch, max_diff={max_diff:.2e}"
    print(f"  [permute correctness] ({num_tokens}x{num_cols}, topK={topK}, E={E}): PASS (max_diff={max_diff:.2e})")

    # Round-trip: permute_backward(perm, prob=ones) / topK should recover x
    # permute_backward uses prob=ones internally, so divide by topK
    out = permute_backward(perm_tri, rid_tri, num_tokens, topK) / topK
    rt_diff = (out.float() - x.float()).abs().max().item()
    print(f"  [permute round-trip]  max_diff={rt_diff:.2e}")
    assert torch.allclose(out, x, atol=1e-3, rtol=1e-3), f"round-trip failed, max_diff={rt_diff:.2e}"


def test_permute_backward_correctness(num_tokens, num_cols, topK, E, dtype=torch.bfloat16):
    dev = torch.device("cuda")
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=dtype)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)
    perm_t, rid_t, _ = permute_countsort(x, idx)
    grad_perm = torch.randn_like(perm_t)

    g_tri = permute_backward(grad_perm, rid_t, num_tokens, topK)
    g_ref = native_permute_backward(grad_perm, rid_t, num_tokens, topK)

    max_diff = (g_tri.float() - g_ref.float()).abs().max().item()
    print(f"  [permute backward] ({num_tokens}x{num_cols}, topK={topK}, E={E}): PASS (max_diff={max_diff:.2e})")
    assert torch.allclose(g_tri, g_ref, atol=1e-2, rtol=1e-2), \
        f"backward mismatch vs native, max_diff={max_diff:.2e}"


# ---------------------------------------------------------------------------
# Benchmark: Triton vs PyTorch native
# ---------------------------------------------------------------------------

def bench_permute_op(num_tokens, num_cols, topK, E):
    dev = torch.device("cuda")
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)
    num_out = num_tokens * topK
    bytes_io = (num_out * num_cols + num_tokens * topK * num_cols) * 2  # bf16

    t_tri = _bench(lambda: permute(x, idx))
    t_nat = _bench(lambda: native_permute(x, idx))
    bw_tri = bytes_io / 1e9 / t_tri * 1e3
    bw_nat = bytes_io / 1e9 / t_nat * 1e3
    print(f"  [fwd bench] {num_tokens}x{num_cols} topK={topK} E={E}: "
          f"Triton {t_tri:6.3f} ms ({bw_tri:6.1f} GB/s) | "
          f"Native {t_nat:6.3f} ms ({bw_nat:6.1f} GB/s) | "
          f"speedup {t_nat/t_tri:.2f}x")


def bench_permute_backward(num_tokens, num_cols, topK, E):
    dev = torch.device("cuda")
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)
    _, rid, _ = permute_countsort(x, idx)
    num_out = num_tokens * topK
    grad_perm = torch.randn(num_out, num_cols, device=dev, dtype=torch.bfloat16)
    # bytes: read grad_permuted (T*K rows) + write grad_input (T rows, accumulated topK times)
    bytes_io = (num_out * num_cols + num_tokens * num_cols) * 2  # bf16

    t_tri = _bench(lambda: permute_backward(grad_perm, rid, num_tokens, topK))
    t_nat = _bench(lambda: native_permute_backward(grad_perm, rid, num_tokens, topK))
    bw_tri = bytes_io / 1e9 / t_tri * 1e3
    bw_nat = bytes_io / 1e9 / t_nat * 1e3
    print(f"  [bwd bench] {num_tokens}x{num_cols} topK={topK} E={E}: "
          f"Triton {t_tri:6.3f} ms ({bw_tri:6.1f} GB/s) | "
          f"Native {t_nat:6.3f} ms ({bw_nat:6.1f} GB/s) | "
          f"speedup {t_nat/t_tri:.2f}x")


if __name__ == "__main__":
    print("=" * 64)
    print("Correctness Check - Permute:")
    print("=" * 64)
    test_permute_correctness(4096,  4096, 2, 16)
    test_permute_correctness(8192,  4096, 2, 16)
    test_permute_correctness(4096,  4096, 4, 64)
    test_permute_backward_correctness(4096, 4096, 2, 16)
    test_permute_backward_correctness(8192, 4096, 2, 16)

    print("\n" + "=" * 64)
    print("Benchmark - Permute Forward (Triton vs PyTorch Native):")
    print("=" * 64)
    bench_permute_op(4096,  4096, 2, 16)
    bench_permute_op(8192,  4096, 2, 16)
    bench_permute_op(16384, 4096, 2, 16)
    bench_permute_op(16384, 4096, 4, 64)

    print("\n" + "=" * 64)
    print("Benchmark - Permute Backward (Triton vs PyTorch Native):")
    print("=" * 64)
    bench_permute_backward(4096,  4096, 2, 16)
    bench_permute_backward(8192,  4096, 2, 16)
    bench_permute_backward(16384, 4096, 2, 16)
    bench_permute_backward(16384, 4096, 4, 64)
    print("=" * 64)
