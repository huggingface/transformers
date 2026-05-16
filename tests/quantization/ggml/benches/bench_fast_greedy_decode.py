"""Bench the production fast_greedy_decode API."""
import os, time, torch, warnings
warnings.filterwarnings('ignore')
import torch._dynamo
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.allow_unspec_int_on_nn_module = True
os.environ['TRANSFORMERS_GGUF_METAL_KERNELS_SO'] = '/Users/arthurzucker/Work/gguf-dequant-kernels/result-bundle/torch210-metal-aarch64-darwin/_gguf_dequant_7a428ca_dirty.abi3.so'
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.gguf_linear import setup_for_compile, fast_greedy_decode

repo = '/Users/arthurzucker/.cache/huggingface/hub/models--gdax--Qwen1.5-MoE-A2.7B_gguf/snapshots/9f02e3e589316464a4ecce048d35358d54c60298'
tok = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-MoE-A2.7B')
m = AutoModelForCausalLM.from_pretrained(repo, gguf_file='Qwen1.5-MoE-A2.7B_q4_k_m.gguf',
                                          gguf_linear=True, dtype=torch.float32).to('mps').eval()
setup_for_compile(m)

ids = tok('The quick brown fox', return_tensors='pt').input_ids.to('mps')
import sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64

# Warm
_ = fast_greedy_decode(m, ids, max_new_tokens=8)
torch.mps.synchronize()

# Timed
for trial in range(3):
    t0 = time.perf_counter()
    _ = fast_greedy_decode(m, ids, max_new_tokens=N)
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    print(f"trial {trial+1}: {N} tokens in {dt*1000:.1f}ms = {N/dt:.1f} tok/s ({dt/N*1000:.2f} ms/tok)")
