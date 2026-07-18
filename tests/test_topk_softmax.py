import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import time
from triton_gmm_ops import fused_topk_softmax

DEVICE = "cuda"

def _bench(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) * 1000 / iters


def test_correctness():
    configs = [
        (16, 2),   # E=16, K=2
        (64, 4),   # E=64, K=4
        (256, 16), # E=256, K=16
    ]
    
    num_tokens = 8192
    print("================================================================")
    print("Correctness Check - Fused Top-K Softmax (Forward & Backward in BF16):")
    print("================================================================")
    
    for E, K in configs:
        # Generate non-overlapping columns in float32
        logits_raw = torch.randn(num_tokens, E, device=DEVICE, dtype=torch.float32) * 0.05
        cols = torch.arange(E, device=DEVICE).float() * 0.5
        
        logits_bf16 = (logits_raw + cols.unsqueeze(0)).to(torch.bfloat16).requires_grad_(True)
        logits_fp32 = (logits_raw + cols.unsqueeze(0)).to(torch.float32).requires_grad_(True)
        
        # 1. Reference (PyTorch Native BF16)
        w_ref, idx_ref = torch.topk(logits_bf16, K, dim=-1)
        w_ref_soft = torch.nn.functional.softmax(w_ref.float(), dim=-1).to(torch.bfloat16)
        
        grad_out = torch.randn_like(w_ref_soft)
        w_ref_soft.backward(grad_out, retain_graph=True)
        grad_logits_ref = logits_bf16.grad.clone()
        
        # 1b. Reference (PyTorch Native FP32)
        w_ref_fp32, idx_ref_fp32 = torch.topk(logits_fp32, K, dim=-1)
        w_ref_soft_fp32 = torch.nn.functional.softmax(w_ref_fp32, dim=-1)
        grad_out_fp32 = grad_out.float()
        w_ref_soft_fp32.backward(grad_out_fp32, retain_graph=True)
        grad_logits_ref_fp32 = logits_fp32.grad.clone()
        
        # 2. Triton Fused (FP16/BF16 Routing)
        logits_bf16.grad.zero_()
        w_fused, idx_fused = fused_topk_softmax(logits_bf16, K)
        w_fused.backward(grad_out, retain_graph=True)
        grad_logits_fused = logits_bf16.grad.clone()
        
        # 3. Triton Fused (FP32 Routing on FP32 Logits)
        logits_fp32.grad.zero_()
        w_fused_fp32, idx_fused_fp32 = fused_topk_softmax(logits_fp32, K, fp32_routing=True)
        assert w_fused_fp32.dtype == torch.float32, "weights must be float32 when fp32_routing is True"
        w_fused_fp32.backward(grad_out_fp32, retain_graph=True)
        grad_logits_fused_fp32 = logits_fp32.grad.clone()
        
        # Check diffs for BF16 path
        max_diff_fwd = torch.max(torch.abs(w_ref_soft.float() - w_fused.float())).item()
        max_diff_idx = torch.max(torch.abs(idx_ref.int() - idx_fused)).item()
        max_diff_bwd = torch.max(torch.abs(grad_logits_ref.float() - grad_logits_fused.float())).item()
        
        # Check diffs for FP32 path
        max_diff_fwd_fp32 = torch.max(torch.abs(w_ref_soft_fp32 - w_fused_fp32)).item()
        max_diff_bwd_fp32 = torch.max(torch.abs(grad_logits_ref_fp32 - grad_logits_fused_fp32)).item()
        
        # bf16 tolerances are typically around 1e-2 to 1e-3, fp32 around 1e-5
        fwd_ok = max_diff_fwd < 1e-2
        idx_ok = max_diff_idx == 0
        bwd_ok = max_diff_bwd < 1e-2
        fwd_fp32_ok = max_diff_fwd_fp32 < 1e-5
        bwd_fp32_ok = max_diff_bwd_fp32 < 1e-5
        
        print(f"Config E={E}, K={K}:")
        print(f"  Forward Diff:  {max_diff_fwd:.2e} -> {'PASS' if fwd_ok else 'FAIL'}")
        print(f"  Indices Diff:  {max_diff_idx:.2e} -> {'PASS' if idx_ok else 'FAIL'}")
        print(f"  Backward Diff: {max_diff_bwd:.2e} -> {'PASS' if bwd_ok else 'FAIL'}")
        print(f"  Forward Diff (FP32 Routing):  {max_diff_fwd_fp32:.2e} -> {'PASS' if fwd_fp32_ok else 'FAIL'}")
        print(f"  Backward Diff (FP32 Routing): {max_diff_bwd_fp32:.2e} -> {'PASS' if bwd_fp32_ok else 'FAIL'}")
        
        assert fwd_ok, f"Forward mismatch: {max_diff_fwd:.2e}"
        assert idx_ok, f"Indices mismatch: {max_diff_idx:.2e}"
        assert bwd_ok, f"Backward mismatch: {max_diff_bwd:.2e}"
        assert fwd_fp32_ok, f"FP32 Forward mismatch: {max_diff_fwd_fp32:.2e}"
        assert bwd_fp32_ok, f"FP32 Backward mismatch: {max_diff_bwd_fp32:.2e}"
    print("================================================================")


def run_benchmark():
    shapes = [8192, 16384, 32768]
    configs = [
        (16, 2),   # E=16, K=2
        (64, 4),   # E=64, K=4
        (256, 16), # E=256, K=16
    ]
    
    for E, K in configs:
        print("\n" + "=" * 80)
        print(f"{f'ROUTER TOP-K SOFTMAX BENCHMARK: E={E}, K={K}':^80}")
        print("=" * 80)
        print(f"{'Num Tokens':<12} | {'FWD Native':<12} | {'FWD Triton':<12} | {'BWD Native':<12} | {'BWD Triton':<12} | {'Speedup F/B':<12}")
        print("-" * 80)
        
        for tokens in shapes:
            logits = torch.randn(tokens, E, device=DEVICE, dtype=torch.bfloat16)
            
            # Setup tensors with grad for backward bench
            logits_nat = logits.clone().requires_grad_(True)
            w_nat, idx_nat = torch.topk(logits_nat, K, dim=-1)
            out_nat = torch.nn.functional.softmax(w_nat.float(), dim=-1).to(torch.bfloat16)
            grad_out = torch.randn_like(out_nat)
            
            logits_tri = logits.clone().requires_grad_(True)
            out_tri, idx_tri = fused_topk_softmax(logits_tri, K)
            
            # Forward Functions
            def fwd_native():
                w, idx = torch.topk(logits, K, dim=-1)
                return torch.nn.functional.softmax(w.float(), dim=-1).to(torch.bfloat16), idx
                
            def fwd_triton():
                return fused_topk_softmax(logits, K)
                
            # Backward Functions
            def bwd_native():
                out_nat.backward(grad_out, retain_graph=True)
                
            def bwd_triton():
                out_tri.backward(grad_out, retain_graph=True)
                
            t_fwd_nat = _bench(fwd_native)
            t_fwd_tri = _bench(fwd_triton)
            
            t_bwd_nat = _bench(bwd_native)
            t_bwd_tri = _bench(bwd_triton)
            
            speedup_str = f"{t_fwd_nat/t_fwd_tri:.2f}x / {t_bwd_nat/t_bwd_tri:.2f}x"
            print(f"{tokens:<12} | {t_fwd_nat:9.3f} ms | {t_fwd_tri:9.3f} ms | {t_bwd_nat:9.3f} ms | {t_bwd_tri:9.3f} ms | {speedup_str:<12}")
        print("=" * 80)


if __name__ == "__main__":
    test_correctness()
    run_benchmark()
