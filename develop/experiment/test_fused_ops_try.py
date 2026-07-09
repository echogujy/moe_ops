import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import time

try:
    import grouped_gemm_backend
    from step_mini.modules.grouped_gemm_custom_op import gmm as cpp_gmm
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

from triton_gmm_ops.permute import permute_countsort_autograd, permute_countsort
from triton_gmm_ops.gmm_try import gmm_try
from triton_gmm_ops.unpermute import unpermute
from triton_gmm_ops.fused_ops import fused_permute_gemm, fused_gemm_unpermute
from triton_gmm_ops.fused_ops_try import fused_permute_gemm_try, fused_gemm_unpermute_try

DEVICE = "cuda"

def _bench(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) * 1000 / iters


def test_correctness():
    # Run correctness check in float32 to avoid non-deterministic bfloat16 atomic accumulation noise
    num_tokens = 8192
    num_cols = 4096
    topK = 2
    E = 16
    N = 4096
    
    X = torch.randn(num_tokens, num_cols, device=DEVICE, dtype=torch.float32, requires_grad=True)
    W1 = torch.randn(E, N, num_cols, device=DEVICE, dtype=torch.float32, requires_grad=True)
    W2 = torch.randn(E, num_cols, N, device=DEVICE, dtype=torch.float32, requires_grad=True)
    prob = torch.rand(num_tokens, topK, device=DEVICE, dtype=torch.float32)
    prob = (prob / prob.sum(dim=-1, keepdim=True)).detach().requires_grad_(True)
    indices = torch.randint(0, E, (num_tokens, topK), device=DEVICE, dtype=torch.int32)
    
    # Unfused Reference Pipeline (2 GEMMs)
    permuted, row_id_map, base = permute_countsort_autograd(X, indices, 0, E)
    offsets = torch.empty_like(base)
    offsets[:-1] = base[1:]
    offsets[-1] = num_tokens * topK
    
    Y1_ref = gmm_try(permuted, W1, offsets, trans_b=True)
    Y2_ref = gmm_try(Y1_ref, W2, offsets, trans_b=True)
    out_ref = unpermute(Y2_ref, row_id_map, prob, num_tokens, topK)
    
    grad_output = torch.randn_like(out_ref)
    out_ref.backward(grad_output, retain_graph=True)
    
    grad_X_ref = X.grad.clone()
    grad_W1_ref = W1.grad.clone()
    grad_W2_ref = W2.grad.clone()
    grad_prob_ref = prob.grad.clone()
    
    X.grad.zero_()
    W1.grad.zero_()
    W2.grad.zero_()
    prob.grad.zero_()
    
    num_total = num_tokens * topK
    flat_idx = torch.arange(num_total, device=DEVICE)
    sorted_id_map = torch.full((num_total,), -1, dtype=torch.int32, device=DEVICE)
    sorted_id_map.scatter_(0, row_id_map.long(), (flat_idx % num_tokens).to(torch.int32))
    slot_map = torch.full((num_total,), -1, dtype=torch.int32, device=DEVICE)
    slot_map.scatter_(0, row_id_map.long(), (flat_idx // num_tokens).to(torch.int32))
    
    # Fused Try Pipeline (2 GEMMs)
    Y1_fused = fused_permute_gemm_try(X, W1, base, sorted_id_map, trans_b=True)
    out_fused = fused_gemm_unpermute_try(Y1_fused, W2, base, sorted_id_map, slot_map, prob, num_tokens, topK, trans_b=True)
    
    max_diff_fwd = torch.max(torch.abs(out_fused - out_ref)).item()
    out_fused.backward(grad_output, retain_graph=True)
    
    max_diff_grad_x = torch.max(torch.abs(X.grad - grad_X_ref)).item()
    max_diff_grad_w1 = torch.max(torch.abs(W1.grad - grad_W1_ref)).item()
    max_diff_grad_w2 = torch.max(torch.abs(W2.grad - grad_W2_ref)).item()
    max_diff_grad_prob = torch.max(torch.abs(prob.grad - grad_prob_ref)).item()
    
    print("================================================================")
    print("Correctness Check (Full 2-GEMM MoE Block, float32 precision):")
    print("================================================================")
    print(f"  Forward vs Ref Diff: {max_diff_fwd:.2e} -> {'PASS' if max_diff_fwd < 5e-3 else 'FAIL'}")
    print(f"  Grad_X vs Ref Diff : {max_diff_grad_x:.2e} -> {'PASS' if max_diff_grad_x < 5e-3 else 'FAIL'}")
    print(f"  Grad_W1 vs Ref Diff: {max_diff_grad_w1:.2e} -> {'PASS' if max_diff_grad_w1 < 5e-3 else 'FAIL'}")
    print(f"  Grad_W2 vs Ref Diff: {max_diff_grad_w2:.2e} -> {'PASS' if max_diff_grad_w2 < 5e-3 else 'FAIL'}")
    print(f"  Grad_Prob vs Ref   : {max_diff_grad_prob:.2e} -> {'PASS' if max_diff_grad_prob < 2e-1 else 'FAIL'}")
    print("================================================================")


def run_benchmark():
    shapes = [
        (4096, 4096, 2, 16, 4096),
        (8192, 4096, 2, 16, 4096),
        (16384, 4096, 2, 16, 4096),
        (16384, 8192, 2, 16, 8192),
    ]
    
    # ----------------------------------------------------------------------------
    # 1. FORWARD Pass Benchmark
    # ----------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"{'FULL MOE 2-GEMM BLOCK - FORWARD LATENCY (ms)':^115}")
    print("=" * 115)
    print(f"{'Shape (Tokens x Cols x topK x E x N)':<32} | {'Unfused Native':<18} | {'Unfused C++':<18} | {'Fused Orig':<18} | {'Fused Try':<18} | {'Speedup (vs Orig)':<15}")
    print("-" * 115)
    
    for tokens, cols, topK, E, N in shapes:
        X = torch.randn(tokens, cols, device=DEVICE, dtype=torch.bfloat16)
        W1 = torch.randn(E, N, cols, device=DEVICE, dtype=torch.bfloat16)
        W2 = torch.randn(E, cols, N, device=DEVICE, dtype=torch.bfloat16)
        
        w1_native = W1.transpose(1, 2).contiguous()
        w2_native = W2.transpose(1, 2).contiguous()
        
        prob = torch.rand(tokens, topK, device=DEVICE, dtype=torch.bfloat16)
        prob = (prob / prob.sum(dim=-1, keepdim=True))
        indices = torch.randint(0, E, (tokens, topK), device=DEVICE, dtype=torch.int32)
        
        permuted, row_id_map, base = permute_countsort(X, indices, E=E)
        offsets = torch.empty_like(base)
        offsets[:-1] = base[1:]
        offsets[-1] = tokens * topK
        
        sizes = torch.zeros(E, dtype=torch.int32, device=DEVICE)
        sizes[0] = offsets[0]
        sizes[1:] = offsets[1:] - offsets[:-1]
        
        num_total = tokens * topK
        flat_idx = torch.arange(num_total, device=DEVICE)
        sorted_id_map = torch.full((num_total,), -1, dtype=torch.int32, device=DEVICE)
        sorted_id_map.scatter_(0, row_id_map.long(), (flat_idx % tokens).to(torch.int32))
        slot_map = torch.full((num_total,), -1, dtype=torch.int32, device=DEVICE)
        slot_map.scatter_(0, row_id_map.long(), (flat_idx // tokens).to(torch.int32))
        
        # Unfused Native Forward (2 GEMMs)
        def unfused_native():
            perm, r_map, bs = permute_countsort(X, indices, E=E)
            offs = torch.empty_like(bs)
            offs[:-1] = bs[1:]
            offs[-1] = tokens * topK
            y1 = torch.nn.functional.grouped_mm(perm, w1_native, offs=offs)
            y2 = torch.nn.functional.grouped_mm(y1, w2_native, offs=offs)
            return unpermute(y2, r_map, prob, tokens, topK)
            
        # Unfused C++ Forward (2 GEMMs)
        def unfused_cpp():
            perm, r_map, bs = permute_countsort(X, indices, E=E)
            sz = torch.zeros(E, dtype=torch.int32, device=DEVICE)
            sz[0] = bs[0]
            sz[1:] = bs[1:] - bs[:-1]
            y1 = cpp_gmm(perm, W1, sz, trans_b=True)
            y2 = cpp_gmm(y1, W2, sz, trans_b=True)
            return unpermute(y2, r_map, prob, tokens, topK)
            
        # Fused Original Forward (2 GEMMs)
        def fused_orig():
            y1 = fused_permute_gemm(X, W1, base, sorted_id_map, trans_b=True)
            return fused_gemm_unpermute(y1, W2, base, sorted_id_map, slot_map, prob, tokens, topK, trans_b=True)
            
        # Fused Try Forward (2 GEMMs)
        def fused_try():
            y1 = fused_permute_gemm_try(X, W1, base, sorted_id_map, trans_b=True)
            return fused_gemm_unpermute_try(y1, W2, base, sorted_id_map, slot_map, prob, tokens, topK, trans_b=True)
            
        t_nat = _bench(unfused_native)
        t_cpp = _bench(unfused_cpp) if _HAS_CPP else float('nan')
        t_orig = _bench(fused_orig)
        t_try = _bench(fused_try)
        
        cpp_str = f"{t_cpp:12.3f} ms" if _HAS_CPP else "N/A"
        shape_str = f"{tokens}x{cols}x{topK}x{E}x{N}"
        
        print(f"{shape_str:<32} | {t_nat:12.3f} ms | {cpp_str:<18} | {t_orig:12.3f} ms | {t_try:12.3f} ms | {t_orig / t_try:.2f}x")
    print("=" * 115)

    # ----------------------------------------------------------------------------
    # 2. BACKWARD Pass Benchmark
    # ----------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print(f"{'FULL MOE 2-GEMM BLOCK - BACKWARD LATENCY (ms)':^115}")
    print("=" * 115)
    print(f"{'Shape (Tokens x Cols x topK x E x N)':<32} | {'Unfused Native':<18} | {'Unfused C++':<18} | {'Fused Orig':<18} | {'Fused Try':<18} | {'Speedup (vs Orig)':<15}")
    print("-" * 115)
    
    for tokens, cols, topK, E, N in shapes:
        X = torch.randn(tokens, cols, device=DEVICE, dtype=torch.bfloat16)
        W1 = torch.randn(E, N, cols, device=DEVICE, dtype=torch.bfloat16)
        W2 = torch.randn(E, cols, N, device=DEVICE, dtype=torch.bfloat16)
        w1_native = W1.transpose(1, 2).contiguous()
        w2_native = W2.transpose(1, 2).contiguous()
        
        prob = torch.rand(tokens, topK, device=DEVICE, dtype=torch.bfloat16)
        prob = (prob / prob.sum(dim=-1, keepdim=True))
        indices = torch.randint(0, E, (tokens, topK), device=DEVICE, dtype=torch.int32)
        
        permuted, row_id_map, base = permute_countsort(X, indices, E=E)
        offsets = torch.empty_like(base)
        offsets[:-1] = base[1:]
        offsets[-1] = tokens * topK
        
        sizes = torch.zeros(E, dtype=torch.int32, device=DEVICE)
        sizes[0] = offsets[0]
        sizes[1:] = offsets[1:] - offsets[:-1]
        
        num_total = tokens * topK
        flat_idx = torch.arange(num_total, device=DEVICE)
        sorted_id_map = torch.full((num_total,), -1, dtype=torch.int32, device=DEVICE)
        sorted_id_map.scatter_(0, row_id_map.long(), (flat_idx % tokens).to(torch.int32))
        slot_map = torch.full((num_total,), -1, dtype=torch.int32, device=DEVICE)
        slot_map.scatter_(0, row_id_map.long(), (flat_idx // tokens).to(torch.int32))
        
        grad_out = torch.randn(tokens, cols, device=DEVICE, dtype=torch.bfloat16)
        
        # Unfused Native Backward Setup
        X_nat = X.clone().requires_grad_(True)
        W1_nat = w1_native.clone().requires_grad_(True)
        W2_nat = w2_native.clone().requires_grad_(True)
        prob_nat = prob.clone().requires_grad_(True)
        perm_nat, r_map_nat, bs_nat = permute_countsort_autograd(X_nat, indices, 0, E)
        offs_nat = torch.empty_like(bs_nat)
        offs_nat[:-1] = bs_nat[1:]
        offs_nat[-1] = tokens * topK
        y1_nat = torch.nn.functional.grouped_mm(perm_nat, W1_nat, offs=offs_nat)
        y2_nat = torch.nn.functional.grouped_mm(y1_nat, W2_nat, offs=offs_nat)
        out_nat = unpermute(y2_nat, r_map_nat, prob_nat, tokens, topK)
        t_nat = _bench(lambda: out_nat.backward(grad_out, retain_graph=True))
        
        # Unfused C++ Backward Setup
        t_cpp = float('nan')
        if _HAS_CPP:
            X_cpp = X.clone().requires_grad_(True)
            W1_cpp = W1.clone().requires_grad_(True)
            W2_cpp = W2.clone().requires_grad_(True)
            prob_cpp = prob.clone().requires_grad_(True)
            perm_cpp, r_map_cpp, bs_cpp = permute_countsort_autograd(X_cpp, indices, 0, E)
            sz_cpp = torch.zeros(E, dtype=torch.int32, device=DEVICE)
            sz_cpp[0] = bs_cpp[0]
            sz_cpp[1:] = bs_cpp[1:] - bs_cpp[:-1]
            y1_cpp = cpp_gmm(perm_cpp, W1_cpp, sz_cpp, trans_b=True)
            y2_cpp = cpp_gmm(y1_cpp, W2_cpp, sz_cpp, trans_b=True)
            out_cpp = unpermute(y2_cpp, r_map_cpp, prob_cpp, tokens, topK)
            t_cpp = _bench(lambda: out_cpp.backward(grad_out, retain_graph=True))
            
        # Fused Orig Backward Setup
        X_orig = X.clone().requires_grad_(True)
        W1_orig = W1.clone().requires_grad_(True)
        W2_orig = W2.clone().requires_grad_(True)
        prob_orig = prob.clone().requires_grad_(True)
        y1_orig = fused_permute_gemm(X_orig, W1_orig, base, sorted_id_map, trans_b=True)
        out_orig = fused_gemm_unpermute(y1_orig, W2_orig, base, sorted_id_map, slot_map, prob_orig, tokens, topK, trans_b=True)
        t_orig = _bench(lambda: out_orig.backward(grad_out, retain_graph=True))
        
        # Fused Try Backward Setup
        X_try = X.clone().requires_grad_(True)
        W1_try = W1.clone().requires_grad_(True)
        W2_try = W2.clone().requires_grad_(True)
        prob_try = prob.clone().requires_grad_(True)
        y1_try = fused_permute_gemm_try(X_try, W1_try, base, sorted_id_map, trans_b=True)
        out_try = fused_gemm_unpermute_try(y1_try, W2_try, base, sorted_id_map, slot_map, prob_try, tokens, topK, trans_b=True)
        t_try = _bench(lambda: out_try.backward(grad_out, retain_graph=True))
        
        cpp_str = f"{t_cpp:12.3f} ms" if _HAS_CPP else "N/A"
        shape_str = f"{tokens}x{cols}x{topK}x{E}x{N}"
        
        print(f"{shape_str:<32} | {t_nat:12.3f} ms | {cpp_str:<18} | {t_orig:12.3f} ms | {t_try:12.3f} ms | {t_orig / t_try:.2f}x")
    print("=" * 115)


if __name__ == "__main__":
    test_correctness()
    run_benchmark()
