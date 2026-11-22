"""Where the current best decode step's time goes: the compiled step re-measured with one component replaced by a
no-op at a time (outputs are garbage; the timing is the point). Same loading as hc_bench (4-bit dense plan).

    PYTHONPATH=src:benchmark_v2/inkling2_metal python benchmark_v2/inkling2_metal/hillclimb/hc_ablate.py
"""

import json
import os
import statistics
import time

import inductor_mps_fixes
import torch

from transformers import AutoModelForImageTextToText, AutoTokenizer, CompileConfig, MetalConfig
from transformers.integrations.metal_quantization import MetalFp4Experts, MetalLinear
from transformers.models.inkling2_loop import modeling_inkling2_loop as M


HERE = os.path.dirname(os.path.abspath(__file__))
inductor_mps_fixes.apply()
repo = "thinkingmachines/Inkling-2-Mini"
metal = MetalConfig(bits=4, group_size=64)
plan = {"model.visual.*": None, "*_message_gate": None, "lm_head": metal, "*": metal}
model = AutoModelForImageTextToText.from_pretrained(
    repo, device_map="mps", dtype=torch.bfloat16, quantization_config=plan
)
model.eval()
for module in model.modules():
    if type(module).__name__.endswith("RelativePositionBias"):
        module.proj_t = module.proj.detach().t().to(torch.bfloat16).contiguous()
tokenizer = AutoTokenizer.from_pretrained(repo)
prompt = [{"role": "user", "content": "Write a short story about a lighthouse keeper."}]
ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = ids["input_ids"].to("mps")
compile_config = CompileConfig()
compile_config._compile_all_devices = True
gen = {"cache_implementation": "static", "compile_config": compile_config, "do_sample": False}

originals = {}


def patch(owner, name, fn):
    originals[(owner, name)] = getattr(owner, name)
    setattr(owner, name, fn)


def restore():
    for (owner, name), fn in originals.items():
        setattr(owner, name, fn)
    originals.clear()


def router_noop(hidden_states, module):
    tokens = hidden_states.numel() // module.hidden_dim
    weights = hidden_states.new_ones(tokens, module.top_k, dtype=torch.float32)
    indices = torch.arange(module.top_k, device=hidden_states.device).expand(tokens, -1)
    return hidden_states.new_zeros(tokens, module.num_experts, dtype=torch.float32), weights, indices


def attn_res_noop(block_states, partial, query, module):
    n_queries = query.reshape(-1, partial.shape[-1]).shape[0]
    return partial.float().unsqueeze(-2).expand(*partial.shape[:-1], n_queries, -1)


def metal_linear_noop(self, x):
    return x.new_zeros(*x.shape[:-1], self.out_features)


ABLATIONS = {
    "none": lambda: None,
    "experts": lambda: patch(MetalFp4Experts, "forward", lambda self, h, i, w: torch.zeros_like(h)),
    "router": lambda: patch(M, "inkling2_loop_router", router_noop),
    "attention": lambda: patch(M, "inkling2_loop_attention", lambda q, k, v, *a, **kw: (q.transpose(1, 2), None)),
    "attn_res": lambda: patch(M, "inkling2_loop_attention_residual", attn_res_noop),
    "message_gate": lambda: patch(M, "inkling2_loop_message_gate", lambda h, m, mod: h.float()),
    "short_conv": lambda: patch(M, "inkling2_loop_short_convolution_step", lambda h, s, w, k: h),
    "dense_4bit": lambda: patch(MetalLinear, "forward", metal_linear_noop),
}


def step_ms(tokens=96, samples=3):
    stamps, meds = [], []

    class Stamps:
        def put(self, _):
            stamps.append(time.perf_counter())

        def end(self):
            pass

    with torch.inference_mode():
        for _ in range(2):
            model.generate(ids, max_new_tokens=8, min_new_tokens=8, **gen)
        for _ in range(samples):
            stamps.clear()
            model.generate(ids, max_new_tokens=tokens, min_new_tokens=tokens, streamer=Stamps(), **gen)
            steps = sorted(b - a for a, b in zip(stamps[2:], stamps[3:]))
            meds.append(steps[len(steps) // 2] * 1e3)
    return statistics.median(meds)


start, base, results = time.perf_counter(), None, {}
for i, (name, apply) in enumerate(ABLATIONS.items()):
    torch._dynamo.reset()
    apply()
    try:
        ms = step_ms()
        base = ms if base is None else base
        results[name] = ms
        print(f"[{i + 1}/{len(ABLATIONS)}] without {name:13s} {ms:6.2f} ms/token (saves {base - ms:+6.2f}) "
              f"| elapsed {time.perf_counter() - start:.0f}s", flush=True)  # fmt: skip
    except Exception as error:  # noqa: BLE001
        print(
            f"[{i + 1}/{len(ABLATIONS)}] without {name}: failed ({type(error).__name__}: {str(error)[:120]})",
            flush=True,
        )
    restore()
with open(os.path.join(HERE, "ledger.jsonl"), "a") as f:
    f.write(
        json.dumps({"config": "ablation-4bit", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "ms_per_token": results})
        + "\n"
    )
