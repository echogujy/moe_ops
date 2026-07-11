import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import torch
from triton_gmm_ops import permute
from triton_gmm_ops import unpermute, unpermute_forward, unpermute_backward

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


def _torch_unpermute_ref(input, row_id_map, prob, num_tokens, topK):
    """Independent pure-torch reference for unpermute."""
    num_cols = input.shape[1]
    rowmap = row_id_map.reshape(topK, num_tokens)  # [topK, num_tokens]
    out = torch.zeros(num_tokens, num_cols, dtype=torch.float32, device=input.device)
    for k in range(topK):
        src = rowmap[k]
        valid = src >= 0
        gathered = input[src.clamp(min=0)] * valid[:, None].to(input.dtype)
        out += prob[:, k:k + 1] * gathered.to(torch.float32)
    return out.to(input.dtype)


def test_unpermute_correctness(num_tokens, num_cols, topK, E, dtype=torch.bfloat16):
    dev = torch.device("cuda")
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=dtype)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)
    perm_t, rid_t, _ = permute(x, idx)
    prob = torch.softmax(torch.randn(num_tokens, topK, device=dev), dim=-1).float()

    out_t = unpermute(perm_t, rid_t, prob, num_tokens, topK)

    # independent torch reference
    out_ref = _torch_unpermute_ref(perm_t, rid_t, prob, num_tokens, topK)
    max_diff_ref = (out_t - out_ref).abs().max()
    ok_ref = torch.allclose(out_t, out_ref, atol=1e-2, rtol=1e-2)

    print(f"  [unpermute] ({num_tokens}x{num_cols}, topK={topK}) PASS  torch_ref_maxdiff={max_diff_ref.item():.2e}")
    assert ok_ref, "unpermute correctness failed"


def test_unpermute_bf16_prob(num_tokens, num_cols, topK, E):
    dev = torch.device("cuda")
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)
    perm_t, rid_t, _ = permute(x, idx)
    prob_bf16 = torch.softmax(torch.randn(num_tokens, topK, device=dev), dim=-1).bfloat16()
    prob_f32 = prob_bf16.float()

    out_bf16 = unpermute_forward(perm_t, rid_t, prob_bf16, num_tokens, topK)
    out_f32 = unpermute_forward(perm_t, rid_t, prob_f32, num_tokens, topK)
    ok = torch.allclose(out_bf16, out_f32, atol=1e-2, rtol=1e-2)
    print(f"  [unpermute bf16-prob] ({num_tokens}x{num_cols}) PASS max_diff={(out_bf16 - out_f32).abs().max().item():.2e}")
    assert ok, "unpermute with bf16 prob diverged from float32 prob"


def test_unpermute_backward(num_tokens, num_cols, topK, E, dtype=torch.float32):
    dev = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=dtype)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)
    perm_t, rid_t, _ = permute(x, idx)
    prob = torch.softmax(torch.randn(num_tokens, topK, device=dev), dim=-1).to(dtype)

    a = perm_t.detach().requires_grad_(True)
    b = prob.detach().requires_grad_(True)
    unpermute(a, rid_t, b, num_tokens, topK).sum().backward()
    ag_analytic, pg_analytic = a.grad.clone(), b.grad.clone()

    # FD check - using float32 for clean check
    eps = 1e-3
    num_rows = a.shape[0]
    ag_num = torch.zeros(num_rows, device=dev, dtype=torch.float32)
    for i in range(num_rows):
        ap = a.detach().clone(); ap[i] += eps
        an = a.detach().clone(); an[i] -= eps
        fp = unpermute_forward(ap, rid_t, b.detach(), num_tokens, topK).sum().item()
        fn = unpermute_forward(an, rid_t, b.detach(), num_tokens, topK).sum().item()
        ag_num[i] = (fp - fn) / (2 * eps)
    
    ag_diff = (ag_analytic.sum(dim=1).float() - ag_num).abs().max().item()
    ok_a = ag_diff < 5e-2

    pg_num = torch.zeros_like(b)
    for flat in range(b.numel()):
        t, k = divmod(flat, topK)
        bp = b.detach().clone(); bp[t, k] += eps
        bn = b.detach().clone(); bn[t, k] -= eps
        fp = unpermute_forward(a.detach(), rid_t, bp, num_tokens, topK).sum().item()
        fn = unpermute_forward(a.detach(), rid_t, bn, num_tokens, topK).sum().item()
        pg_num[t, k] = (fp - fn) / (2 * eps)
    
    pg_diff = (pg_analytic.float() - pg_num).abs().max().item()
    ok_b = pg_diff < 5e-2
    print(f"  [unpermute backward] ({num_tokens}x{num_cols}, topK={topK}) FD check: act_grad={'PASS' if ok_a else 'FAIL'} (max_diff={ag_diff:.2e}) | prob_grad={'PASS' if ok_b else 'FAIL'} (max_diff={pg_diff:.2e})")
    assert ok_a and ok_b, f"unpermute backward FD check failed, ag_diff={ag_diff:.2e}, pg_diff={pg_diff:.2e}"


def bench_unpermute(num_tokens, num_cols, topK, E):
    dev = torch.device("cuda")
    x = torch.randn(num_tokens, num_cols, device=dev, dtype=torch.bfloat16)
    idx = torch.randint(0, E, (num_tokens, topK), device=dev, dtype=torch.int32)
    perm_t, rid_t, _ = permute(x, idx)
    prob = torch.softmax(torch.randn(num_tokens, topK, device=dev), dim=-1).float()
    bytes_io = (num_tokens * topK * num_cols + num_tokens * num_cols) * 2

    t_tri = _bench(lambda: unpermute_forward(perm_t, rid_t, prob, num_tokens, topK))
    t_ref = _bench(lambda: _torch_unpermute_ref(perm_t, rid_t, prob, num_tokens, topK))
    print(f"  [bench unpermute] {num_tokens}x{num_cols} topK={topK}: Triton {t_tri:7.3f} ms ({bytes_io/1e9/t_tri*1e3:6.1f} GB/s) | Native {t_ref:7.3f} ms | speedup {t_ref/t_tri:5.2f}x")


if __name__ == "__main__":
    print("=" * 64)
    print("Triton unpermute - correctness")
    print("=" * 64)
    test_unpermute_correctness(512, 1024, 2, 8)
    test_unpermute_correctness(2048, 4096, 2, 16)
    test_unpermute_correctness(2048, 4096, 1, 16)
    test_unpermute_bf16_prob(2048, 4096, 2, 16)
    test_unpermute_backward(32, 64, 2, 8)
    test_unpermute_backward(64, 128, 2, 16)

    print("\n" + "=" * 64)
    print("Triton unpermute - performance (large shapes)")
    print("=" * 64)
    for (nt, nc, tk, e) in [
        (8192, 4096, 2, 64),
        (16384, 4096, 2, 64),
        (16384, 8192, 2, 64),
    ]:
        bench_unpermute(nt, nc, tk, e)
