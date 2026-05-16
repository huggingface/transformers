"""Microbenchmark the new gguf_mul_mat_vec_multi_f32 vs 3 separate matvec calls.

Loads the locally-built kernel .so directly (no Hub round-trip) and runs at
Qwen-attention dimensions: hidden=2048, q_heads*d=2048, kv_heads*d=256.
The multi op should be faster by the 3 → 1 dispatcher saving (~50 µs / call).
"""
import time, torch, numpy as np

# Load the local build.
import sys, os
SO = "/Users/arthurzucker/Work/gguf-dequant-kernels/result-bundle/torch210-metal-aarch64-darwin/_gguf_dequant_83dbe84_dirty.abi3.so"
torch.ops.load_library(SO)
ns = torch.ops._gguf_dequant_83dbe84_dirty

mul_mat_vec_q4_K = ns.mul_mat_vec_q4_K_f32.default
mul_mat_vec_multi = ns.gguf_mul_mat_vec_multi_f32.default

# Qwen1.5-MoE-A2.7B attention shape: hidden=2048, q=2048, k=v=256 (GQA)
K = 2048
Mq = 2048
Mk = 256
Mv = 256
np.random.seed(0)
def fake_qw(M, K):
    nbytes = M * (K // 256) * 144  # Q4_K
    return torch.from_numpy(np.random.randint(0, 256, size=nbytes, dtype=np.uint8).copy()).to('mps')

qw_q = fake_qw(Mq, K)
qw_k = fake_qw(Mk, K)
qw_v = fake_qw(Mv, K)
x = (torch.randn(K, dtype=torch.float32, device='mps') * 0.1).contiguous()

# Warmup
for _ in range(20):
    yq = torch.empty(Mq, dtype=torch.float32, device='mps')
    yk = torch.empty(Mk, dtype=torch.float32, device='mps')
    yv = torch.empty(Mv, dtype=torch.float32, device='mps')
    mul_mat_vec_q4_K(qw_q, x, yq)
    mul_mat_vec_q4_K(qw_k, x, yk)
    mul_mat_vec_q4_K(qw_v, x, yv)
torch.mps.synchronize()

# Path A: 3 separate matvec calls (current behaviour)
for ITERS in (256, 1024):
    yq = torch.empty(Mq, dtype=torch.float32, device='mps')
    yk = torch.empty(Mk, dtype=torch.float32, device='mps')
    yv = torch.empty(Mv, dtype=torch.float32, device='mps')
    t0 = time.perf_counter()
    for _ in range(ITERS):
        mul_mat_vec_q4_K(qw_q, x, yq)
        mul_mat_vec_q4_K(qw_k, x, yk)
        mul_mat_vec_q4_K(qw_v, x, yv)
    torch.mps.synchronize()
    a = (time.perf_counter() - t0) / ITERS * 1e6
    yq2 = torch.empty(Mq, dtype=torch.float32, device='mps')
    yk2 = torch.empty(Mk, dtype=torch.float32, device='mps')
    yv2 = torch.empty(Mv, dtype=torch.float32, device='mps')
    fmt_codes = [4, 4, 4]  # q4_K
    qw_list = [qw_q, qw_k, qw_v]
    y_list = [yq2, yk2, yv2]
    t0 = time.perf_counter()
    for _ in range(ITERS):
        mul_mat_vec_multi(qw_list, x, y_list, fmt_codes)
    torch.mps.synchronize()
    b = (time.perf_counter() - t0) / ITERS * 1e6
    print(f"ITERS={ITERS:>4}  per-proj 3-call: {a:7.1f} µs/qkv   multi 1-call: {b:7.1f} µs/qkv   saved: {a-b:+6.1f} µs ({(a-b)/a*100:+.0f}%)")

# Sanity: result equality
yq_ref = torch.empty(Mq, dtype=torch.float32, device='mps')
yk_ref = torch.empty(Mk, dtype=torch.float32, device='mps')
yv_ref = torch.empty(Mv, dtype=torch.float32, device='mps')
mul_mat_vec_q4_K(qw_q, x, yq_ref)
mul_mat_vec_q4_K(qw_k, x, yk_ref)
mul_mat_vec_q4_K(qw_v, x, yv_ref)
yq_m = torch.empty(Mq, dtype=torch.float32, device='mps')
yk_m = torch.empty(Mk, dtype=torch.float32, device='mps')
yv_m = torch.empty(Mv, dtype=torch.float32, device='mps')
mul_mat_vec_multi([qw_q, qw_k, qw_v], x, [yq_m, yk_m, yv_m], [4, 4, 4])
torch.mps.synchronize()
print(f"q-max diff: {(yq_ref - yq_m).abs().max().item():.6e}")
print(f"k-max diff: {(yk_ref - yk_m).abs().max().item():.6e}")
print(f"v-max diff: {(yv_ref - yv_m).abs().max().item():.6e}")
