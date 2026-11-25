"""Per-component breakdown of Qwen3.8 decode time on MPS.

Answers: where does the ~25 ms/token of non-stall time actually go?
Prints store.stats() deltas (hits/misses/reads/covered/sync/wait), the host router
thread's spin/work seconds, and the residual (wall - host) which must be GPU work.

Run through .bench/mem_bounded_run.py so the same loader patches apply:
  python .bench/mem_bounded_run.py profile_decode.py <ckpt> 4g32c 16
"""

import sys
import time

import torch
from dense_quant import quantize_dense

from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig

repo = sys.argv[1] if len(sys.argv) > 1 else "ArthurZ/Qwen3-30B-A3B-metal-experts-4bit"
spec = sys.argv[2] if len(sys.argv) > 2 else "4g32c"
cache_gib = float(sys.argv[3]) if len(sys.argv) > 3 else 0
N = int(sys.argv[4]) if len(sys.argv) > 4 else 256

tok = AutoTokenizer.from_pretrained(repo)
if cache_gib:
    import glob
    import os

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = repo if os.path.isdir(repo) else snapshot_download(repo)
    resident = 0
    for file in glob.glob(f"{path}/*.safetensors"):
        with safe_open(file, "pt") as f:
            for key in f.keys():
                if ".experts." not in key and ".ngram_embedding." not in key:
                    meta = f.get_tensor_meta(key) if hasattr(f, "get_tensor_meta") else None
                    shape = meta.shape if meta else f.get_slice(key).get_shape()
                    resident += torch.Size(shape).numel() * 2
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        dtype=torch.bfloat16,
        device_map={"": "mps", "experts": "disk", "ngram_embedding": "disk"},
        max_memory={"mps": int(resident + cache_gib * 2**30)},
    )
else:
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="mps")
quantize_dense(model, spec)
compile_config = CompileConfig()
compile_config._compile_all_devices = True
compiled = {"cache_implementation": "static"}
if os.environ.get("NO_COMPILE") != "1":
    compiled["compile_config"] = compile_config

prompt = "Write a short story about a lighthouse keeper who finds a message in a bottle."


def ids(p):
    return tok.apply_chat_template(
        [{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=False,
    ).input_ids.to("mps")  # fmt: skip


# Find every offload store attached to the model.
stores = []
seen = set()
for name, module in model.named_modules():
    store = getattr(module, "offload_store", None)
    if store is not None and id(store) not in seen:
        seen.add(id(store))
        stores.append((name, store))
print(f"offload stores: {[n for n, _ in stores]}", flush=True)
if not stores:
    raise SystemExit("no offload store found: is this a streamed checkpoint?")

def host_routers():
    # `mps_router` is created by `setup()` on the first forward, so this must be lazy.
    out = []
    for _, store in stores:
        for attr in ("mps_router", "host_router", "_host_router", "router_thread"):
            r = getattr(store, attr, None)
            if r is not None:
                out.append(r)
    return out


class Stamps:
    def __init__(self):
        self.t = []

    def put(self, _):
        self.t.append(time.perf_counter())

    def end(self):
        pass


def snapshot():
    out = {}
    for _, store in stores:
        s = store.stats()
        for k, v in s.items():
            out[k] = out.get(k, 0) + v
    for r in host_routers():
        for k in ("spin_seconds", "work_seconds", "ahead_reads", "jobs_done"):
            if hasattr(r, k):
                out[k] = out.get(k, 0) + getattr(r, k)
    return out


def reset():
    for _, store in stores:
        store.reset_stats()
    for r in host_routers():
        for k in ("spin_seconds", "work_seconds", "ahead_reads", "jobs_done"):
            if hasattr(r, k):
                setattr(r, k, 0)
        for k in ("latency", "pending_guess", "guess_layers", "guess_exact", "guess_held", "guess_wanted",
                  "guess_no_router", "guess_not_decode", "guess_no_ahead", "guess_calls"):
            if hasattr(r, k):
                setattr(r, k, type(getattr(r, k))() if k in ("latency",) else ({} if k == "pending_guess" else 0))


def generate(n):
    stamps = Stamps()
    with torch.inference_mode():
        model.generate(ids(prompt), max_new_tokens=n, min_new_tokens=n, do_sample=False, streamer=stamps, **compiled)
    steps = [b - a for a, b in zip(stamps.t[2:], stamps.t[3:])]
    return steps


generate(32)  # compile warmup
torch.mps.synchronize()

# Per-layer trace: (time, thread, phase, layer, info). Enabled only for the measured window.
for _, store in stores:
    store.trace = []

if os.environ.get("TRACE_ITEM") == "1":
    import collections, threading, traceback

    _seen = collections.Counter()
    _cost = collections.Counter()
    _orig_item, _orig_int, _orig_float = torch.Tensor.item, torch.Tensor.__int__, torch.Tensor.__float__

    def _timed(fn, self, tag=""):
        t = time.perf_counter()
        try:
            return fn(self)
        finally:
            _cost[tag + _key()] += time.perf_counter() - t

    def _key():
        st = traceback.extract_stack(limit=16)
        chain = " <- ".join(f"{f.name}:{f.lineno}" for f in st[-9:-3] if "profile_decode" not in f.filename)
        return f"[{threading.current_thread().name}] {chain}"

    def _item(self):
        _seen[_key()] += 1
        t = time.perf_counter()
        try:
            return _orig_item(self)
        finally:
            _cost[_key()] += time.perf_counter() - t

    def _int(self):
        _seen[_key()] += 1
        return _orig_int(self)

    def _float(self):
        _seen[_key()] += 1
        return _orig_float(self)

    _orig_bool, _orig_tolist, _orig_len = torch.Tensor.__bool__, torch.Tensor.tolist, torch.Tensor.__len__

    def _bool(self):
        _seen["BOOL " + _key()] += 1
        t = time.perf_counter()
        try:
            return _orig_bool(self)
        finally:
            _cost["BOOL " + _key()] += time.perf_counter() - t

    def _tolist(self, *a, **k):
        _seen["TOLIST " + _key()] += 1
        t = time.perf_counter()
        try:
            return _orig_tolist(self, *a, **k)
        finally:
            _cost["TOLIST " + _key()] += time.perf_counter() - t

    def _len(self):
        _seen["LEN " + _key()] += 1
        return _orig_len(self)

    torch.Tensor.item, torch.Tensor.__int__, torch.Tensor.__float__ = _item, _int, _float
    torch.Tensor.__bool__, torch.Tensor.tolist, torch.Tensor.__len__ = _bool, _tolist, _len
    print("TRACE_ITEM on: every scalar sync is attributed to its callers at exit", flush=True)
    import atexit

    def _dump():
        print("\n=== scalar sync callers (count, total ms, ms/call) ===", flush=True)
        for key, sec in _cost.most_common(12):
            n = _seen.get(key, 0) or 1
            print(f"{n:6d}  {sec * 1000:9.1f} ms  {sec / n * 1000:7.3f} ms/call  {key}", flush=True)
    atexit.register(_dump)

if os.environ.get("TRACE_OPS"):
    import collections, threading, time as _t, traceback

    _ops_seen = collections.Counter()
    _ops_cost = collections.Counter()

    def _wrap(name):
        orig = getattr(torch.Tensor, name)

        def inner(self, *a, **k):
            st = traceback.extract_stack(limit=14)
            key = name + " [" + threading.current_thread().name + "] " + " <- ".join(
                f"{f.name}:{f.lineno}" for f in st[-8:-3] if "profile_decode" not in f.filename
            )
            t = _t.perf_counter()
            try:
                return orig(self, *a, **k)
            finally:
                _ops_seen[key] += 1
                _ops_cost[key] += _t.perf_counter() - t

        setattr(torch.Tensor, name, inner)

    for _n in [x.strip() for x in os.environ["TRACE_OPS"].split(",") if x.strip()]:
        _wrap(_n)
    print(f"TRACE_OPS on: {os.environ['TRACE_OPS']}", flush=True)
    import atexit

    def _dump_ops():
        print("\n=== op callers (count, total ms, ms/call) ===", flush=True)
        for key, sec in _ops_cost.most_common(15):
            n = _ops_seen.get(key, 0) or 1
            print(f"{n:7d}  {sec * 1000:9.1f} ms  {sec / n * 1000:7.3f} ms/call  {key}", flush=True)
    atexit.register(_dump_ops)

mark = os.environ.get("DECODE_MARK")
if mark:  # let a tracer attach to the measured window instead of the checkpoint load
    open(mark, "w").write(str(os.getpid()))
    print(f"decode window marker: {mark} pid {os.getpid()}", flush=True)
reset()
# Main-thread CPU (issue) time vs wall: wall - cpu = blocked on the GPU.
_t0, _c0 = time.perf_counter(), time.thread_time()
ev_start, ev_end = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
ev_start.record()
steps = generate(N)
ev_end.record()
torch.mps.synchronize()
gpu_elapsed = ev_start.elapsed_time(ev_end) / 1000.0  # seconds
cpu_issue = time.thread_time() - _c0
wall = sum(steps)
per_tok = wall / len(steps)
snap = snapshot()

host_sync = snap.get("sync_seconds", 0.0)
host_wait = snap.get("wait_seconds", 0.0)
host_spin = snap.get("spin_seconds", 0.0)
host_work = snap.get("work_seconds", 0.0)
host_total = host_sync + host_wait + host_spin + host_work
residual = wall - host_total
hits = snap.get("hits", 0)
misses = snap.get("misses", 0)
reads = snap.get("reads", 0)
covered = snap.get("covered", 0)

print("", flush=True)
print(f"tokens measured           {len(steps)}", flush=True)
print(f"wall                      {per_tok * 1000:8.2f} ms/token   ({1 / per_tok:.2f} tok/s)", flush=True)
print(f"  host sync (routing copy){host_sync / len(steps) * 1000:8.2f} ms/token  {host_sync / wall * 100:5.1f}%", flush=True)
print(f"  host wait (expert read) {host_wait / len(steps) * 1000:8.2f} ms/token  {host_wait / wall * 100:5.1f}%", flush=True)
print(f"  host spin (slot resolve){host_spin / len(steps) * 1000:8.2f} ms/token  {host_spin / wall * 100:5.1f}%", flush=True)
print(f"  host work (install/evict){host_work / len(steps) * 1000:8.2f} ms/token  {host_work / wall * 100:5.1f}%", flush=True)
print(f"  residual (GPU + python) {residual / len(steps) * 1000:8.2f} ms/token  {residual / wall * 100:5.1f}%", flush=True)
print("", flush=True)
print(f"main-thread cpu (issue)   {cpu_issue / len(steps) * 1000:8.2f} ms/token  {cpu_issue / wall * 100:5.1f}%  of wall", flush=True)
print(f"blocked on gpu (wall-cpu) {(wall - cpu_issue) / len(steps) * 1000:8.2f} ms/token  {(wall - cpu_issue) / wall * 100:5.1f}%  of wall", flush=True)
print(f"mps-event gpu span        {gpu_elapsed / len(steps) * 1000:8.2f} ms/token  {gpu_elapsed / wall * 100:5.1f}%  of wall", flush=True)
print("", flush=True)
print(f"hits {hits}  misses {misses}  reads {reads}  covered {covered}", flush=True)
if hits + misses:
    print(f"hit rate                  {hits / (hits + misses) * 100:6.2f}%", flush=True)
if snap.get("jobs_done"):
    print(f"router jobs               {snap['jobs_done']}  ahead_reads {snap.get('ahead_reads', 0)}", flush=True)

# Per-layer aggregation of the events-off path: sync_start..sync_end and wait_start..wait_end.
from collections import defaultdict

per_layer = defaultdict(lambda: defaultdict(float))
open_ev = {}
for store in [st for _, st in stores]:
    trace = store.trace or []
    for ts, _tid, phase, layer, _info in trace:
        if phase in ("sync_start", "wait_start"):
            open_ev[(layer, phase[: -len("_start")])] = ts
        elif phase in ("sync_end", "wait_end"):
            base = phase[: -len("_end")]
            if (layer, base) in open_ev:
                per_layer[layer][base] += ts - open_ev.pop((layer, base))

if per_layer:
    n = len(steps)
    rows = sorted(per_layer.items(), key=lambda kv: -(kv[1].get("sync", 0) + kv[1].get("wait", 0)))
    print("", flush=True)
    print(f"{'layer':22s} {'sync ms':>9s} {'wait ms':>9s} {'total ms':>9s}  (per token)", flush=True)
    tot_s = tot_w = 0.0
    for layer, d in rows:
        sy, wa = d.get("sync", 0.0) / n * 1000, d.get("wait", 0.0) / n * 1000
        tot_s += sy
        tot_w += wa
        print(f"{str(layer):22s} {sy:9.2f} {wa:9.2f} {sy + wa:9.2f}", flush=True)
    print(f"{'SUM':22s} {tot_s:9.2f} {tot_w:9.2f} {tot_s + tot_w:9.2f}", flush=True)

for r in host_routers():
    rt = getattr(r, "rt_times", None)
    if rt and rt.get("calls"):
        n_c = rt["calls"]
        print("route() cpu breakdown (ms/token): " + "  ".join(
            f"{k} {v / len(steps) * 1000:.2f}" for k, v in rt.items() if k != "calls"), flush=True)
print(f"router thread spin/work   {snap.get('spin_seconds', 0.0) / len(steps) * 1000:8.2f} / {snap.get('work_seconds', 0.0) / len(steps) * 1000:.2f} ms/token (runs beside the GPU)", flush=True)


# Handshake and prediction instrumentation (events-on path): how long the submit -> routing-landed round trip
# takes, how much of that is the miss install, and how well the ahead-read prediction does.
def _pct(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] * 1000


for r in host_routers():
    lat = getattr(r, "latency", None)
    if not lat:
        continue
    n_steps = max(len(steps), 1)
    lat_ms = [x[1] for x in lat]
    inst_ms = [x[2] for x in lat]
    layers = len({x[0] for x in lat})
    print("", flush=True)
    print(f"handshake submit->landed  p50 {_pct(lat_ms, 0.5):6.3f}  p90 {_pct(lat_ms, 0.9):6.3f}  "
          f"p99 {_pct(lat_ms, 0.99):6.3f} ms   (mean {sum(lat_ms) / len(lat_ms) * 1000:.3f})", flush=True)
    print(f"handshake install (miss)  p50 {_pct(inst_ms, 0.5):6.3f}  p90 {_pct(inst_ms, 0.9):6.3f}  "
          f"p99 {_pct(inst_ms, 0.99):6.3f} ms", flush=True)
    print(f"handshake calls {len(lat)}  over {layers} layers, {len(lat) / n_steps:.1f}/token   "
          f"sum {sum(lat_ms) / n_steps * 1000:.2f} ms/token", flush=True)
    held, wanted, exact, gl = getattr(r, "guess_held", 0), getattr(r, "guess_wanted", 0), getattr(r, "guess_exact", 0), getattr(r, "guess_layers", 0)
    if gl:
        print(f"guess layers {gl}  exact-layer {exact / gl * 100:5.1f}%  "
              f"recall {held and min(held, wanted) / max(wanted, 1) * 100:5.1f}%  "
              f"(predicted {held / gl:.1f} experts/layer, used {wanted / gl:.1f})", flush=True)
    else:
        print("guess layers 0: no prediction was ever compared", flush=True)
    print(f"guess calls {getattr(r, 'guess_calls', 0)}  no router {getattr(r, 'guess_no_router', 0)}  "
          f"not decode {getattr(r, 'guess_not_decode', 0)}  no next layer {getattr(r, 'guess_no_ahead', 0)}", flush=True)
