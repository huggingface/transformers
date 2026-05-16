"""Time how long Cache.update takes on MPS at decode shape."""
import time, torch
from transformers.cache_utils import StaticCache
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained('Qwen/Qwen1.5-MoE-A2.7B')
H = cfg.num_attention_heads * (cfg.hidden_size // cfg.num_attention_heads)
KVH = cfg.num_key_value_heads
HD = cfg.hidden_size // cfg.num_attention_heads

max_kv = 512
cache = StaticCache(config=cfg, max_batch_size=1, max_cache_len=max_kv,
                    device='mps', dtype=torch.float32)

# Decode shape: 1 token's worth of K/V per layer
k = torch.randn(1, KVH, 1, HD, dtype=torch.float32, device='mps')
v = torch.randn(1, KVH, 1, HD, dtype=torch.float32, device='mps')

# Warm
for layer_idx in range(4):
    cache.update(k, v, layer_idx, cache_kwargs={"cache_position": torch.tensor([0], device='mps')})
torch.mps.synchronize()

# Pre-allocate cache_positions once; index updates change but tensor doesn't realloc
cp = torch.zeros(1, dtype=torch.long, device='mps')
ITERS = 1000

# Variant A: sync between calls (worst case, no pipelining)
t0 = time.perf_counter()
for i in range(ITERS):
    cp.fill_(i % (max_kv - 1))
    cache.update(k, v, 0, cache_kwargs={"cache_position": cp})
torch.mps.synchronize()
per_call_a = (time.perf_counter() - t0) / ITERS * 1e6
print(f"Cache.update [sync each]    : {per_call_a:7.1f} µs/call")

# Variant B: cycle through 24 layers per "token", sync once per "token"
t0 = time.perf_counter()
for tok in range(ITERS):
    cp.fill_(tok % (max_kv - 1))
    for layer in range(24):
        cache.update(k, v, layer, cache_kwargs={"cache_position": cp})
torch.mps.synchronize()
per_token = (time.perf_counter() - t0) / ITERS * 1e3
print(f"Cache.update across 24 layers/token (pipelined): {per_token:.2f} ms/token")

# Variant C: just the indexed assignment (raw, no Python wrapper overhead)
cache_buf_k = torch.zeros(1, KVH, max_kv, HD, dtype=torch.float32, device='mps')
t0 = time.perf_counter()
for i in range(ITERS):
    cp.fill_(i % (max_kv - 1))
    cache_buf_k[:, :, cp] = k
torch.mps.synchronize()
per_raw = (time.perf_counter() - t0) / ITERS * 1e6
print(f"Raw indexed assignment       : {per_raw:7.1f} µs/call")
