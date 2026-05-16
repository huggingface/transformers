"""Benchmark Qwen1.5-MoE-A2.7B Q4_K_M under transformers (eager + compile)."""
import argparse, os, time, warnings
warnings.filterwarnings("ignore")
import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.recompile_limit = 512
# Stop dynamo from specialising on integer attributes of nn.Module instances
# (most importantly the DynamicCache layer_idx, which would otherwise cause
# one recompile per layer per generate() call — 24 useless recompiles).
torch._dynamo.config.allow_unspec_int_on_nn_module = True
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def time_decode(model, input_ids, n_new, *, label):
    model.eval()
    with torch.inference_mode():
        # Warmup (also primes the KV cache shape for compile)
        _ = model.generate(input_ids, max_new_tokens=4, do_sample=False, use_cache=True,
                           pad_token_id=model.config.eos_token_id or 0)
        if input_ids.device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        out = model.generate(input_ids, max_new_tokens=n_new, do_sample=False, use_cache=True,
                             pad_token_id=model.config.eos_token_id or 0)
        if input_ids.device.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
    new_tokens = out.shape[1] - input_ids.shape[1]
    tok_s = new_tokens / (t1 - t0)
    print(f"[{label}] {new_tokens} new tokens in {t1 - t0:.2f}s → {tok_s:.2f} tok/s")
    return tok_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--repo", required=True, help="HF repo id (for config/tokenizer)")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    repo_path = os.path.dirname(args.gguf)
    gguf_file = os.path.basename(args.gguf)

    t = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(repo_path, gguf_file=gguf_file, gguf_linear=True,
                                                 dtype=torch.float32)
    model = model.to(args.device)
    # Static KV cache so the compiled graph's shapes are constant during
    # decode — without this, the cache length grows each token and dynamo
    # recompiles on every step.
    model.generation_config.cache_implementation = "static"
    print(f"loaded in {time.perf_counter() - t:.1f}s")

    moe_count = sum(1 for m in model.modules() if m.__class__.__name__ == "GgufQwen2MoeExperts")
    lin_count = sum(1 for m in model.modules() if m.__class__.__name__ == "GgufLinear")
    print(f"GgufLinear: {lin_count}, GgufQwen2MoeExperts: {moe_count}")

    ids = tok("The quick brown fox jumps over", return_tensors="pt").input_ids.to(args.device)
    print(f"prompt: {ids.shape}")

    # Default is gguf_bmm. Bench it first.
    bmm_tok_s = time_decode(model, ids, args.n, label="gguf_bmm")

    # Toggle to eager (one-at-a-time per-expert dispatch) and bench.
    from transformers.integrations.gguf_linear import GgufQwen2MoeExperts
    for m in model.modules():
        if isinstance(m, GgufQwen2MoeExperts):
            m._experts_implementation = "eager"
    eager_tok_s = time_decode(model, ids, args.n, label="eager(per-expert)")
    for m in model.modules():
        if isinstance(m, GgufQwen2MoeExperts):
            m._experts_implementation = "gguf_bmm"
    print(f"speedup bmm vs eager: {bmm_tok_s / eager_tok_s:.2f}x")

    if args.compile:
        # Fuse Q/K/V into a single multi-matvec op call where possible.
        if os.environ.get("BENCH_NO_QKV_FUSION", "0") not in ("1", "true", "True"):
            from transformers.integrations.gguf_linear import apply_fused_qkv, apply_fused_kv_update
            n_fused = apply_fused_qkv(model)
            n_kv = apply_fused_kv_update(model)
            print(f"fused QKV on {n_fused}, KV-update patched: {n_kv}")

        # dynamic=False so the static-cache fast path stays specialised on
        # decode shape (1 query token, fixed kv length).
        compile_mode = os.environ.get("BENCH_COMPILE_MODE", "reduce-overhead")
        scope = os.environ.get("BENCH_COMPILE_SCOPE", "forward")
        if scope == "layers":
            # Compile each decoder layer independently. Smaller graphs trace
            # faster + recompile of one layer doesn't invalidate the rest.
            for layer in model.model.layers:
                layer.forward = torch.compile(layer.forward, mode=compile_mode, dynamic=False)
        elif scope == "model":
            # Compile the inner LM (no lm_head). The head is a single matvec
            # that compile can't accelerate much.
            model.model.forward = torch.compile(
                model.model.forward, mode=compile_mode, dynamic=False
            )
        else:
            model.forward = torch.compile(
                model.forward, mode=compile_mode, dynamic=False, fullgraph=False
            )
        print(f"compile mode: {compile_mode} scope: {scope}")
        # Compile takes a few iters to warm up
        time_decode(model, ids, 8, label="compile warmup")
        compile_tok_s = time_decode(model, ids, args.n, label="bmm+compile")
        print(f"bmm={bmm_tok_s:.2f}  eager={eager_tok_s:.2f}  compile={compile_tok_s:.2f}")


if __name__ == "__main__":
    main()
