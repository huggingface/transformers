"""Replicate PR #45974's bench: Qwen2.5-0.5B-Instruct fp16, generate_batch,
compare sdpa vs kernels-community/metal-flash-sdpa on MPS."""
import os, sys, time, torch, warnings
warnings.filterwarnings("ignore")
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, GenerationConfig, ContinuousBatchingConfig,
)

ATTN = sys.argv[1] if len(sys.argv) > 1 else "sdpa"
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 64

PROMPTS = [
    "The quick brown fox jumps over",
    "Artificial intelligence is",
    "Climate change requires",
    "The best way to learn programming is",
    "Once upon a time in a",
    "Quantum computing will",
    "The capital of France is",
    "Machine learning models",
] * 4  # 32 prompts so CB has work to schedule

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
t = time.perf_counter()
kwargs = {} if ATTN == "default" else {"attn_implementation": ATTN}
m = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float16, **kwargs,
).to("mps").eval()
print(f"loaded in {time.perf_counter()-t:.1f}s  attn={ATTN!r}")

inputs = [tok(p, return_tensors=None)["input_ids"] for p in PROMPTS]
gen_cfg = GenerationConfig(max_new_tokens=MAX_NEW, do_sample=False,
                           pad_token_id=tok.eos_token_id or 0)
cb_cfg = ContinuousBatchingConfig(num_blocks=256, max_batch_tokens=512, max_queue_size=64)

_ = m.generate_batch(inputs=inputs[:1], generation_config=gen_cfg,
                     continuous_batching_config=cb_cfg, progress_bar=False, warmup=True)
torch.mps.synchronize()
t0 = time.perf_counter()
res = m.generate_batch(inputs=inputs, generation_config=gen_cfg,
                       continuous_batching_config=cb_cfg, progress_bar=False)
torch.mps.synchronize()
dt = time.perf_counter() - t0
total_new = sum(len(r.generated_tokens) for r in res.values())
print(f"{len(PROMPTS)} prompts × ≤{MAX_NEW} tok — {total_new} actual gen tokens in {dt:.2f}s = {total_new/dt:.1f} tok/s aggregate")
print(f"  sample req-0 first 80 chars:", repr(tok.decode(res[list(res.keys())[0]].generated_tokens)[:80]))
