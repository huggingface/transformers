"""Per-kernel GPU time of a Qwen3.8 decode step on Metal (synthetic weights): eager decode steps with every Metal
command waited on (`torch.mps.profiler(wait_until_completed=True)`), so an op's CPU time ~ its GPU time + a fixed sync
cost, measured on a trivial op and subtracted. Matmuls / custom Metal kernels are the same as in the compiled step;
eager elementwise ops are unfused (their compiled cost is the difference to the compiled step time)."""

import collections

import torch
from bench_qwen38_synthetic import build


model, text = build()
ids = torch.randint(0, 200_000, (1, 64), device="mps")
steps = 4
with torch.inference_mode():
    out = model(ids, use_cache=True)
    cache, token = out.past_key_values, out.logits[:, -1:].argmax(-1)
    for _ in range(3):
        out = model(token, past_key_values=cache, use_cache=True)
        token = out.logits[:, -1:].argmax(-1)
    probe = torch.zeros(16, device="mps")
    with (
        torch.mps.profiler.profile(mode="interval", wait_until_completed=True),
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof,
    ):
        for _ in range(200):
            probe.add_(1)  # the floor: a trivial op's sync cost
        for _ in range(steps):
            out = model(token, past_key_values=cache, use_cache=True)
            token = out.logits[:, -1:].argmax(-1)
events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CPU]
floor_events = [e.cpu_time_total for e in events if e.name == "aten::add_"][:200]
floor = sorted(floor_events)[len(floor_events) // 2]
# Ops that dispatch Metal work: custom / hub kernels (whole op, children skipped) and leaf aten ops
def custom(e):
    return not e.name.startswith("aten::") and "::" in e.name


skip = set()
for e in events:
    if custom(e):
        stack = list(e.cpu_children)
        while stack:
            child = stack.pop()
            skip.add(id(child))
            stack.extend(child.cpu_children)
rows = collections.defaultdict(lambda: [0.0, 0])
for e in events:
    if id(e) in skip or (e.cpu_children and not custom(e)) or (e.name == "aten::add_" and e.input_shapes == [[16], []]):
        continue
    if not custom(e) and not e.name.startswith("aten::"):
        continue
    shapes = "x".join(str(s) for s in (e.input_shapes or [])[:3])[:70]
    row = rows[(e.name, shapes)]
    row[0] += max(e.cpu_time_total - floor, 0.0)
    row[1] += 1
total = sum(r[0] for r in rows.values())
print(f"sync floor {floor:.1f} us/op; attributed GPU time {total / steps / 1e3:.2f} ms/step over {sum(r[1] for r in rows.values()) // steps} ops/step")
for (name, shapes), (us, n) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:30]:
    print(f"{us / steps / 1e3:6.3f} ms {n // steps:4d}/step {us / n:7.1f} us  {name:40s} {shapes}")
