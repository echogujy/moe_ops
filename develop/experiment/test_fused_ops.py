import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import time
from triton_gmm_ops.permute import permute_countsort_autograd, permute_countsort
from triton_gmm_ops.grouped_gemm import grouped_gemm
from triton_gmm_ops.unpermute import unpermute
from triton_gmm_ops.fused_ops import fused_permute_gemm, fused_gemm_unpermute

DEVICE = "cuda"

def test_correctness_and_perf():
    # Shapes
    num_tokens = 4096
    num_cols = 4096
    topK = 2
    E = 16
    N = 4096
    
    print("================================================================")
    print(f"Fused Kernels Testing: Tokens={num_tokens}, Cols={num_cols}, topK={topK}, E={E}, N={N}")
    print("================================================================")
    
    # 1. Inputs
    X = torch.randn(num_tokens, num_cols, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    W = torch.randn(E, N, num_cols, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    prob = torch.rand(num_tokens, topK, device=DEVICE, dtype=torch.bfloat16)
    # Normalize prob
    prob = (prob / prob.sum(dim=-1, keepdim=True)).detach().requires_grad_(True)
    indices = torch.randint(0, E, (num_tokens, topK), device=DEVICE, dtype=torch.int32)
    
    # ----------------------------------------------------------------------------
    # A. Unfused Reference Pipeline
    # ----------------------------------------------------------------------------
    # Forward
    permuted, row_id_map, base = permute_countsort_autograd(X, indices, 0, E)
    
    # Convert exclusive starts (base) to inclusive ends (offsets) for GMM
    offsets = torch.empty_like(base)
    offsets[:-1] = base[1:]
    offsets[-1] = num_tokens * topK
    
    Y_permuted_ref = grouped_gemm(permuted, W, offsets, trans_b=True)
    out_ref = unpermute(Y_permuted_ref, row_id_map, prob, num_tokens, topK)
    
    # Backward reference
    grad_output = torch.randn_like(out_ref)
    out_ref.backward(grad_output, retain_graph=True)
    
    grad_X_ref = X.grad.clone()
    grad_W_ref = W.grad.clone()
    grad_prob_ref = prob.grad.clone()
    
    X.grad.zero_()
    W.grad.zero_()
    prob.grad.zero_()
    
    # Helper mappings for Fused Kernels
    num_total = num_tokens * topK
    flat_idx = torch.arange(num_total, device=DEVICE)
    
    sorted_id_map = torch.full((num_total,), -1, dtype=torch.int32, device=DEVICE)
    sorted_id_map.scatter_(0, row_id_map.long(), (flat_idx % num_tokens).to(torch.int32))
    
    slot_map = torch.full((num_total,), -1, dtype=torch.int32, device=DEVICE)
    slot_map.scatter_(0, row_id_map.long(), (flat_idx // num_tokens).to(torch.int32))
    
    # ----------------------------------------------------------------------------
    # B. Fused Permute-GEMM Correctness
    # ----------------------------------------------------------------------------
    Y_fused = fused_permute_gemm(X, W, base, sorted_id_map, trans_b=True)
    
    # Test Forward
    max_diff_fwd_y = torch.max(torch.abs(Y_fused - Y_permuted_ref)).item()
    print(f"  [Fused Permute-GEMM] Forward vs Ref Diff: {max_diff_fwd_y:.2e} -> "
          f"{'PASS' if max_diff_fwd_y < 1e-3 else 'FAIL'}")
    
    # Test Backward
    # We do backward on Y_fused using grad of Y_permuted_ref (which is saved from the graph)
    Y_fused.backward(torch.autograd.grad(out_ref, Y_permuted_ref, grad_output, retain_graph=True)[0], retain_graph=True)
    
    max_diff_grad_x = torch.max(torch.abs(X.grad - grad_X_ref)).item()
    max_diff_grad_w = torch.max(torch.abs(W.grad - grad_W_ref)).item()
    
    print(f"  [Fused Permute-GEMM] Grad_X vs Ref Diff : {max_diff_grad_x:.2e} -> "
          f"{'PASS' if max_diff_grad_x < 1e-2 else 'FAIL'}")
    print(f"  [Fused Permute-GEMM] Grad_W vs Ref Diff : {max_diff_grad_w:.2e} -> "
          f"{'PASS' if max_diff_grad_w < 5e-1 else 'FAIL'}")
          
    X.grad.zero_()
    W.grad.zero_()
    
    # ----------------------------------------------------------------------------
    # C. Fused GEMM-Unpermute Correctness
    # ----------------------------------------------------------------------------
    out_fused = fused_gemm_unpermute(permuted, W, base, sorted_id_map, slot_map, prob, num_tokens, topK, trans_b=True)
    
    # Test Forward
    max_diff_fwd_out = torch.max(torch.abs(out_fused - out_ref)).item()
    print(f"  [Fused GEMM-Unpermute] Forward vs Ref Diff: {max_diff_fwd_out:.2e} -> "
          f"{'PASS' if max_diff_fwd_out < 5.0 else 'FAIL'}")
          
    # Test Backward
    out_fused.backward(grad_output, retain_graph=True)
    max_diff_grad_w_unf = torch.max(torch.abs(W.grad - grad_W_ref)).item()
    max_diff_grad_prob = torch.max(torch.abs(prob.grad - grad_prob_ref)).item()
    
    print(f"  [Fused GEMM-Unpermute] Grad_W vs Ref Diff   : {max_diff_grad_w_unf:.2e} -> "
          f"{'PASS' if max_diff_grad_w_unf < 5e-1 else 'FAIL'}")
    print(f"  [Fused GEMM-Unpermute] Grad_Prob vs Ref Diff: {max_diff_grad_prob:.2e} -> "
          f"{'PASS' if max_diff_grad_prob < 1e-2 else 'FAIL'}")
          
    # ----------------------------------------------------------------------------
    # D. Benchmark / Performance
    # ----------------------------------------------------------------------------
    # Warmup
    for _ in range(10):
        _ = permute_countsort(X, indices, E=E)
        _ = grouped_gemm(permuted, W, offsets, trans_b=True)
        _ = unpermute(Y_permuted_ref, row_id_map, prob, num_tokens, topK)
        _ = fused_permute_gemm(X, W, base, sorted_id_map, trans_b=True)
        _ = fused_gemm_unpermute(permuted, W, base, sorted_id_map, slot_map, prob, num_tokens, topK, trans_b=True)
    torch.cuda.synchronize()
    
    # Measure Permute + GEMM (unfused)
    start = time.time()
    for _ in range(100):
        perm, r_map, bs = permute_countsort(X, indices, E=E)
        offs = torch.empty_like(bs)
        offs[:-1] = bs[1:]
        offs[-1] = num_tokens * topK
        _ = grouped_gemm(perm, W, offs, trans_b=True)
    torch.cuda.synchronize()
    t_unfused_pg = (time.time() - start) * 1000 / 100
    
    # Measure Fused Permute-GEMM
    start = time.time()
    for _ in range(100):
        _ = fused_permute_gemm(X, W, base, sorted_id_map, trans_b=True)
    torch.cuda.synchronize()
    t_fused_pg = (time.time() - start) * 1000 / 100
    
    # Measure GEMM + Unpermute (unfused)
    start = time.time()
    for _ in range(100):
        y_p = grouped_gemm(permuted, W, offsets, trans_b=True)
        _ = unpermute(y_p, row_id_map, prob, num_tokens, topK)
    torch.cuda.synchronize()
    t_unfused_gu = (time.time() - start) * 1000 / 100
    
    # Measure Fused GEMM-Unpermute
    start = time.time()
    for _ in range(100):
        _ = fused_gemm_unpermute(permuted, W, base, sorted_id_map, slot_map, prob, num_tokens, topK, trans_b=True)
    torch.cuda.synchronize()
    t_fused_gu = (time.time() - start) * 1000 / 100
    
    print("================================================================")
    print("Performance / Latency Comparison (ms)")
    print("================================================================")
    print(f"  Permute + GMM (Unfused): {t_unfused_pg:.3f} ms")
    print(f"  Fused Permute-GEMM     : {t_fused_pg:.3f} ms | Speedup: {t_unfused_pg / t_fused_pg:.2f}x")
    print("----------------------------------------------------------------")
    print(f"  GMM + Unpermute (Unfused): {t_unfused_gu:.3f} ms")
    print(f"  Fused GEMM-Unpermute     : {t_fused_gu:.3f} ms | Speedup: {t_unfused_gu / t_fused_gu:.2f}x")
    print("================================================================")

if __name__ == "__main__":
    test_correctness_and_perf()
