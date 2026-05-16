"""Pure-GPU-time microbenchmark of mul_mat_vec_q4_K_f32.

Measures the kernel work in isolation by submitting many back-to-back calls
between two synchronizations. Subtracts launch overhead by varying the
number of in-flight kernels in one batch.
"""
import time, torch, numpy as np, gguf
import shutil
shutil.rmtree('/Users/arthurzucker/.cache/huggingface/hub/models--ArthurZ--gguf-kernels', ignore_errors=True)
from transformers.integrations.gguf_linear import _ensure_metal_kernels
m = _ensure_metal_kernels()
ns = m._ops
mul_mat_vec = ns.mul_mat_vec_q4_K_f32

# Mimic Qwen1.5-MoE attention shape: M=2048, K=2048
M, K = 2048, 2048
qt = gguf.GGMLQuantizationType.Q4_K
np.random.seed(0)
# Fake Q4_K bytes — gguf-py doesn't have a Python Q4_K quantizer, so we just
# stuff random bytes (correctness doesn't matter for timing).
nbytes = M * (K // 256) * 144
qw = torch.from_numpy(np.random.randint(0, 256, size=nbytes, dtype=np.uint8).copy()).to('mps')
x = torch.randn(K, dtype=torch.float32, device='mps') * 0.1

# Warmup
for _ in range(20):
    y = torch.empty(M, dtype=torch.float32, device='mps')
    mul_mat_vec(qw, x, y)
torch.mps.synchronize()

# Time many calls in batches, varying N. If per-call cost is constant,
# the slope of (total time vs N) tells us per-call time.
for N in (1, 4, 16, 64, 256):
    t0 = time.perf_counter()
    ys = []
    for _ in range(N):
        y = torch.empty(M, dtype=torch.float32, device='mps')
        mul_mat_vec(qw, x, y)
        ys.append(y)
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    print(f"  N={N:3d}: {dt*1000:7.2f} ms total  → {dt/N*1e6:7.2f} µs / call")

# Now without torch.empty in the loop (pre-allocated outputs)
print("\nPre-allocated outputs:")
y_pre = torch.empty(M, dtype=torch.float32, device='mps')
for N in (1, 4, 16, 64, 256):
    t0 = time.perf_counter()
    for _ in range(N):
        mul_mat_vec(qw, x, y_pre)
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    print(f"  N={N:3d}: {dt*1000:7.2f} ms total  → {dt/N*1e6:7.2f} µs / call")
