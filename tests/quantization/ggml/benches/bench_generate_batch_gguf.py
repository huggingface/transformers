"""Bench generate_batch on Qwen1.5-MoE-A2.7B Q4_K_M (continuous batching).

Compares the default attn_implementation against
``kernels-community/metal-flash-sdpa`` per huggingface/transformers#45974.
"""
import os, sys, time, torch, warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_GGUF_METAL_KERNELS_SO"] = (
    "/Users/arthurzucker/Work/gguf-dequant-kernels/result-bundle/"
    "torch210-metal-aarch64-darwin/_gguf_dequant_7a428ca_dirty.abi3.so"
)
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
    ContinuousBatchingConfig,
)

REPO = "/Users/arthurzucker/.cache/huggingface/hub/models--gdax--Qwen1.5-MoE-A2.7B_gguf/snapshots/9f02e3e589316464a4ecce048d35358d54c60298"
GGUF = "Qwen1.5-MoE-A2.7B_q4_k_m.gguf"

PROMPTS = [
    "The quick brown fox jumps over",
    "Artificial intelligence is",
    "Climate change requires",
    "The best way to learn programming is",
    "Once upon a time in a",
    "Quantum computing will",
    "The capital of France is",
    "Machine learning models",
]
MAX_NEW = int(sys.argv[1]) if len(sys.argv) > 1 else 64
ATTN = sys.argv[2] if len(sys.argv) > 2 else "sdpa"


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")
    t = time.perf_counter()
    kwargs = {}
    if ATTN != "default":
        kwargs["attn_implementation"] = ATTN
    model = AutoModelForCausalLM.from_pretrained(
        os.path.dirname(REPO + "/"), gguf_file=GGUF, gguf_linear=True,
        dtype=torch.float32, **kwargs,
    ).to("mps").eval()
    print(f"loaded in {time.perf_counter()-t:.1f}s  attn={ATTN!r}")

    inputs = [tok(p, return_tensors=None)["input_ids"] for p in PROMPTS]
    gen_cfg = GenerationConfig(max_new_tokens=MAX_NEW, do_sample=False,
                               pad_token_id=tok.eos_token_id or 0)

    # Memory budget tuned for the 10 GB MPS recommendedMaxMemory left over
    # after the model weights load. block_size defaults to 32 → 64 blocks ×
    # 32 = 2048 KV positions across all active requests.
    # fp32 KV cache → halve num_blocks again to fit in the ~10 GB MPS budget
    # left after weights.
    cb_cfg = ContinuousBatchingConfig(
        num_blocks=24,
        max_batch_tokens=256,
        max_queue_size=16,
    )

    # Warmup
    try:
        _ = model.generate_batch(
            inputs=inputs[:1], generation_config=gen_cfg,
            continuous_batching_config=cb_cfg, progress_bar=False, warmup=True,
        )
    except Exception as e:
        print(f"warmup error: {type(e).__name__}: {str(e)[:200]}")
        return
    torch.mps.synchronize()

    t0 = time.perf_counter()
    res = model.generate_batch(
        inputs=inputs, generation_config=gen_cfg,
        continuous_batching_config=cb_cfg, progress_bar=False,
    )
    torch.mps.synchronize()
    dt = time.perf_counter() - t0
    total_new = sum(len(r.generated_tokens) for r in res.values())
    print(f"{len(PROMPTS)} prompts × {MAX_NEW} new tokens (target) — "
          f"{total_new} actual gen tokens in {dt:.2f}s = {total_new/dt:.1f} tok/s aggregate")
    print(f"  ({total_new/len(PROMPTS)/dt:.1f} tok/s/request average)")
    print(f"  sample req-0 first 60 chars:", tok.decode(res[list(res.keys())[0]].generated_tokens)[:60])


if __name__ == "__main__":
    main()
