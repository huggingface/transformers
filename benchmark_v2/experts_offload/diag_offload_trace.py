"""Where a streamed decode step's time goes (synthetic 512-expert Qwen3.8, compiled): the store's trace marks per layer,
aggregated over 32 tokens: GPU wait for the routing, host miss handling, pass-2 issue, and the gap between layers
(issuing the next graph segment).

    PYTHONPATH=src:. python diag_offload_trace.py [cache_fraction] [router_skew]
"""

import collections
import glob
import json
import os
import struct
import sys

import torch
from dense_quant import quantize_dense
from sim_routing import skew_routers

from transformers import AutoModelForCausalLM, CompileConfig


fraction = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
skew = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
path = os.path.expanduser("~/.cache/qwen38-synthetic-512e")
expert_bytes = rest_bytes = 0
for file in glob.glob(f"{path}/*.safetensors"):
    with open(file, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    for key, value in header.items():
        if key != "__metadata__":
            size = value["data_offsets"][1] - value["data_offsets"][0]
            expert_bytes, rest_bytes = (
                (expert_bytes + size, rest_bytes) if ".mlp.experts." in key else (expert_bytes, rest_bytes + size)
            )
model = AutoModelForCausalLM.from_pretrained(
    path, dtype=torch.bfloat16, device_map={"": "mps", "experts": "disk"},
    max_memory={"mps": int(rest_bytes + fraction * expert_bytes)},
)  # fmt: skip
quantize_dense(model, "4g32")
skew_routers(model, skew)
store = model._experts_offload
store.lookahead = int(sys.argv[3]) if len(sys.argv) > 3 else store.lookahead
cc = CompileConfig()
cc._compile_all_devices = True
kwargs = {"cache_implementation": "static", "compile_config": cc, "do_sample": False}
ids = torch.randint(0, 200_000, (1, 64), device="mps")
with torch.inference_mode():
    model.generate(ids, max_new_tokens=40, min_new_tokens=40, **kwargs)  # compile + warm the pool
    store.trace = []
    model.generate(ids, max_new_tokens=40, min_new_tokens=40, **kwargs)
main = collections.Counter(t for _, t, *_ in store.trace).most_common(1)[0][0]
events = [(t, phase, layer) for t, thread, phase, layer, _ in store.trace if thread == main]
spans = collections.defaultdict(float)
for (t0, p0, l0), (t1, p1, l1) in zip(events, events[1:]):
    spans[f"{p0} -> {p1}"] += t1 - t0
tokens = 39
total = events[-1][0] - events[0][0]
print(f"cache {fraction}, skew {skew}: {total / tokens * 1e3:.1f} ms/token traced, store stats {store.stats()}")
for span, seconds in sorted(spans.items(), key=lambda kv: -kv[1])[:12]:
    print(f"{seconds / tokens * 1e3:8.2f} ms/token  {span}")
print("mps host router active:", store.mps_router is not None)
