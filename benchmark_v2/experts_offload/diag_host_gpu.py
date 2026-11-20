"""Attribution only (tok/s are measured with bench_experts_offload.py): per decode step, host time until forward
returns (MPS runs asynchronously) vs total time after the device sync, in 128-token windows up to 1024 tokens."""

import sys
import time

import torch

from dense_quant import quantize_dense
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


sys.argv += [] if len(sys.argv) > 2 else ["ArthurZ/Qwen3-30B-A3B-metal-experts-4bit", "4"]
repo, spec = sys.argv[1], sys.argv[2]

tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="mps")
if spec != "bf16":
    quantize_dense(model, spec)
ids = tok.apply_chat_template(
    [{"role": "user", "content": "Write a long story about a lighthouse keeper."}],
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    enable_thinking=False,
).input_ids.to("mps")
with torch.inference_mode():
    cache = DynamicCache(config=model.config)
    token = model(ids, past_key_values=cache).logits[:, -1:].argmax(-1)
    host, total = [], []
    for _ in range(1024):
        torch.mps.synchronize()
        start = time.perf_counter()
        logits = model(token, past_key_values=cache).logits
        host.append(time.perf_counter() - start)
        token = logits[:, -1:].argmax(-1)
        torch.mps.synchronize()
        total.append(time.perf_counter() - start)
for w in range(0, 1024, 128):
    h, t = sum(host[w : w + 128]) / 128 * 1e3, sum(total[w : w + 128]) / 128 * 1e3
    print(
        f"tokens {w:4d}-{w + 127:4d}: total {t:5.2f} ms ({1e3 / t:5.1f} tok/s), host until return {h:5.2f} ms, device tail {t - h:5.2f} ms"
    )
