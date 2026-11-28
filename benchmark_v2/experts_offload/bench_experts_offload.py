# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "accelerate", "huggingface_hub[hf_xet]", "safetensors", "kernels"]
# ///
"""Benchmark streamed MoE experts: prefill / decode tok/s and where a decode token's time goes.

Configs (comma separated, `kind[:fraction of the expert bytes kept on the device]`):
  full            every weight on the device (the compute ceiling, needs the memory)
  ours:0.25       `device_map={"": dev, "experts": "disk"}` (this branch), cache = 25% of the experts
  ours-la0:0.25   same with `lookahead=0` (no next-layer guesses)
  ours-lfu:0.25   same with the approximate LFU policy
  poc:0.25        `experts_offload={"cache_bytes": ...}` (#50717)
  accelerate:0.25 `device_map="auto"` with the same device budget, the rest on cpu
  ours-pin2:0.25  the first 2 layers keep all their experts
  <config>+compile  e.g. ours:0.25+compile: decode through torch.compile (reduce-overhead) and a StaticCache

  uv run bench_experts_offload.py --configs ours:0.25,ours:0.5
  hf jobs uv run --flavor a100-large -s HF_TOKEN bench_experts_offload.py   (see the PR for the overlay mount)
"""

import argparse
import gc
import glob
import inspect
import json
import os
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
parser.add_argument("--configs", default="full,ours:0.5,ours:0.25,accelerate:0.25")
parser.add_argument("--prompt-tokens", type=int, default=1024)
parser.add_argument("--new-tokens", type=int, default=128)
parser.add_argument("--lookahead", type=int, default=1)
parser.add_argument("--experts-implementation", default="grouped_mm")
parser.add_argument(
    "--sustained", type=int, default=0, help="also generate this many tokens, report the 2nd half's mean"
)
parser.add_argument("--attn", default=None, help="attn_implementation, e.g. kernels-community/metal-flash-sdpa")
parser.add_argument("--use-kernels", action="store_true")
parser.add_argument(
    "--stream-extra", default="", help="comma-separated module patterns also streamed from disk, e.g. ngram_embedding"
)
parser.add_argument(
    "--quantize-dense",
    default="",
    help="experiment: Metal affine quantize every nn.Linear but the router after load",
)
parser.add_argument("--cold", action="store_true", help="drop the checkpoint from the page cache before each config")
parser.add_argument("--out", default="bench_results.jsonl")
parser.add_argument(
    "--router-skew",
    type=float,
    default=0.0,
    help="synthetic weights: per-layer expert popularity on top of unit-std router logits (0.6: the best 25%% of "
    "experts take ~88%% of picks), simulating real routing locality",
)
args = parser.parse_args()

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from safetensors import safe_open  # noqa: E402

import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def log(msg):
    print(msg, flush=True)


device = "cuda" if torch.cuda.is_available() else "mps"
device_key = 0 if device == "cuda" else "mps"
sync = torch.cuda.synchronize if device == "cuda" else torch.mps.synchronize
log(
    f"transformers {transformers.__version__} from {os.path.dirname(transformers.__file__)} | torch {torch.__version__}"
)
log(f"device {device}: {torch.cuda.get_device_name() if device == 'cuda' else 'Apple GPU'}")


start = time.perf_counter()
path = os.path.expanduser(args.model)
if not os.path.isdir(path):
    path = snapshot_download(
        args.model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "tokenizer*"]
    )
files = sorted(glob.glob(f"{path}/*.safetensors"))
expert_bytes = rest_bytes = 0
for file in files:
    with safe_open(file, "pt") as f:
        for key in f.keys():
            s = f.get_slice(key)
            size = torch.Size(s.get_shape()).numel() * {
                "BF16": 2,
                "F16": 2,
                "F32": 4,
                "U32": 4,
                "I32": 4,
                "I64": 8,
            }.get(s.get_dtype(), 1)
            if ".experts." in key:
                expert_bytes += size
            elif any(f".{p}." in key for p in args.stream_extra.split(",") if p):
                pass  # streamed rows: not resident
            else:
                rest_bytes += size
log(
    f"{args.model}: experts {expert_bytes / 2**30:.1f} GiB, rest {rest_bytes / 2**30:.1f} GiB ({time.perf_counter() - start:.0f}s)"
)

tokenizer = AutoTokenizer.from_pretrained(path)
text = inspect.getsource(torch.nn.modules.module)  # real text, so routing is not random
prefill_ids = tokenizer(text, return_tensors="pt").input_ids[:, : args.prompt_tokens]
# tarekziade/moe-expert-streaming scripts/bench.py: different prompts, each once, the cache carries over (repeating a
# prompt replays its routing and overstates the hit rate)
DECODE_PROMPTS = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "Explain the causes of the French Revolution to a high school student.",
    "Translate into French and explain the grammar: 'The children would have gone if it had not rained.'",
]


def encode(prompt):
    if not tokenizer.chat_template:  # tiny test checkpoints
        return tokenizer(prompt, return_tensors="pt").input_ids
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).input_ids


chat_ids = [encode(prompt) for prompt in DECODE_PROMPTS]


def drop_page_cache():
    if not hasattr(os, "posix_fadvise"):
        log("  (no posix_fadvise: page cache kept)")
        return
    for file in files:
        fd = os.open(file, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def load(kind, fraction):
    common = {"dtype": torch.bfloat16, "experts_implementation": args.experts_implementation}
    if args.attn:
        common["attn_implementation"] = args.attn
    budget = int(rest_bytes + fraction * expert_bytes)
    if kind == "full":
        return AutoModelForCausalLM.from_pretrained(path, device_map=device, **common)
    if kind.startswith("ours"):
        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map={"": device, "experts": "disk", **{p: "disk" for p in args.stream_extra.split(",") if p}},
            max_memory={device_key: budget},
            **common,
        )
        store = model._experts_offload
        store.lookahead = int(kind[len("ours-la") :]) if kind.startswith("ours-la") else args.lookahead
        if kind == "ours-lfu":
            store.policy = "lfu"
        if kind.startswith("ours-pin"):
            store.pin_layers = int(kind[len("ours-pin") :])
        return model
    if kind == "poc":
        return AutoModelForCausalLM.from_pretrained(
            path, device_map=device, experts_offload={"cache_bytes": int(fraction * expert_bytes)}, **common
        )
    if kind == "accelerate":
        return AutoModelForCausalLM.from_pretrained(
            path, device_map="auto", max_memory={device_key: budget, "cpu": "400GiB"}, **common
        )
    raise ValueError(kind)


def store_stats(model):
    store = getattr(model, "_experts_offload", None)
    if store is not None:
        return store.stats()
    store = next((m.offload_store for m in model.modules() if hasattr(m, "offload_store")), None)  # POC
    if store is not None:
        stats = store.stats()
        return {"hits": stats["hits"], "misses": stats["misses"], "reads": stats["read_calls"]}
    return {}


def reset_stats(model):
    store = getattr(model, "_experts_offload", None)
    if store is not None:
        store.reset_stats()
    for m in model.modules():
        if hasattr(m, "offload_store"):
            m.offload_store.hits = m.offload_store.misses = 0
            break


class _Timer:
    """`generate` streamer: the time between consecutive tokens."""

    def __init__(self):
        self.stamps, self.tokens = [], []

    def put(self, value):
        self.stamps.append(time.perf_counter())
        if len(self.stamps) > 1:  # the first call is the prompt
            self.tokens.append(int(value.reshape(-1)[-1]))

    def end(self):
        pass


@torch.inference_mode()
def decode(model, ids, new_tokens, dev):
    timer = _Timer()
    kwargs = {"cache_implementation": "static"} if getattr(model, "compile_decode", False) else {}
    model.generate(
        ids.to(dev),
        max_new_tokens=new_tokens + 1,
        min_new_tokens=new_tokens + 1,
        do_sample=False,
        streamer=timer,
        **kwargs,
    )
    times = [b - a for a, b in zip(timer.stamps[1:], timer.stamps[2:])]
    return timer.tokens[1:], times


@torch.inference_mode()
def run(model, new_tokens):
    dev = model.get_input_embeddings().weight.device
    model(chat_ids[0][:, :8].to(dev))  # warm up kernels
    sync()
    reset_stats(model)
    start = time.perf_counter()
    model(prefill_ids.to(dev))
    sync()
    prefill_s = time.perf_counter() - start
    prefill = store_stats(model)

    reset_stats(model)
    tokens, per_prompt, times = [], [], []
    for ids in chat_ids:
        prompt_tokens, prompt_times = decode(model, ids, new_tokens, dev)
        tokens.append(prompt_tokens)
        prompt_times = sorted(prompt_times)
        per_prompt.append(round(1 / prompt_times[len(prompt_times) // 2], 2))
        times += prompt_times
        log(f"    prompt {len(per_prompt)}/{len(chat_ids)}: {per_prompt[-1]} tok/s")
    stats = store_stats(model)  # includes the 4 prompts' own (short) prefills
    sustained = None
    if args.sustained:
        _, long_times = decode(model, chat_ids[0], args.sustained, dev)
        tail = long_times[len(long_times) // 2 :]
        sustained = len(tail) / sum(tail)
        log(f"    sustained: {sustained:.2f} tok/s over the last {len(tail)} of {args.sustained} tokens")
    times = sorted(times)
    median_ms = 1e3 * times[len(times) // 2]
    result = {
        "prefill_tok_s": prefill_ids.shape[1] / prefill_s,
        "decode_tok_s": 1e3 / median_ms,
        "decode_tok_s_per_prompt": per_prompt,
        "decode_ms_median": median_ms,
        "sustained_tok_s": sustained,
        "tokens": tokens,
        "prefill_stats": prefill,
    }
    if stats:
        n = len(times)
        lookups = stats.get("hits", 0) + stats.get("misses", 0)
        result["per_token"] = {
            "misses": stats.get("misses", 0) / n,
            "hit_rate": stats.get("hits", 0) / max(lookups, 1),
            "reads": stats.get("reads", 0) / n,
            "covered": stats.get("covered", 0) / max(stats.get("misses", 0), 1),  # share of misses read ahead
            "sync_ms": 1e3 * stats.get("sync_seconds", 0) / n,
            "wait_ms": 1e3 * stats.get("wait_seconds", 0) / n,
        }
        result["slots"] = stats.get("slots")
    return result


configs = args.configs.split(",")
reference = None
started = time.perf_counter()
for index, config in enumerate(configs, 1):
    name = config
    compile_ = name.endswith("+compile")
    kind, _, fraction = name.removesuffix("+compile").partition(":")
    fraction = float(fraction or 1)
    elapsed = time.perf_counter() - started
    eta = f"eta ~{elapsed / (index - 1) * (len(configs) - index + 1) / 60:.0f}m" if index > 1 else "eta ?"
    log(f"[{index}/{len(configs)}] {name} | elapsed {elapsed / 60:.1f}m | {eta}")
    if args.cold:
        drop_page_cache()
    try:
        start = time.perf_counter()
        model = load(kind, fraction)
        if args.use_kernels:
            model.set_use_kernels(True)
        if args.quantize_dense:
            from dense_quant import quantize_dense  # this folder (on HF Jobs: mounted, on PYTHONPATH)

            quantize_dense(model, args.quantize_dense)
        if args.router_skew:
            from sim_routing import skew_routers  # this folder

            log(f"  router skew {args.router_skew} on {skew_routers(model, args.router_skew)} routers")
        if compile_:  # `generate` compiles the decode step with a static cache (CUDA graphs: reduce-overhead)
            torch._inductor.config.graph_partition = True  # CUDA graphs split around `cudagraph_unsafe` ops
            model.compile_decode = True
        load_s = time.perf_counter() - start
        log(f"  loaded in {load_s:.0f}s")
        result = run(model, min(args.new_tokens, 16) if kind == "accelerate" else args.new_tokens)
        result.update(config=name, load_s=load_s, device=device)
        if device == "cuda":
            result["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        if reference is None:
            reference = result["tokens"]
        ref, got = reference[0], result["tokens"][0]
        n = min(len(ref), len(got))
        result["same_tokens_as_first"] = next((i for i in range(n) if ref[i] != got[i]), n)
        summary = {k: v for k, v in result.items() if k != "tokens"}
        log(f"  RESULT {json.dumps(summary)}")
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
    except Exception as error:  # keep going: one broken config should not lose the others
        log(f"  FAILED {name}: {type(error).__name__}: {error}")
        import traceback

        log("".join(traceback.format_exc().splitlines(True)[-14:]))
    model = None  # free it (and its CUDA graphs) before the next config
    torch._dynamo.reset()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        torch.mps.empty_cache()
log(f"done in {(time.perf_counter() - started) / 60:.1f}m")
sys.exit(0)
