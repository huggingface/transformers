"""Where prefill time goes: TTFT at 512 / 2048 prompt tokens with one component replaced by a no-op (garbage output)."""
import statistics, time, torch
import inductor_mps_fixes
from transformers import AutoModelForImageTextToText, MetalConfig
from transformers.integrations.metal_quantization import MetalFp4Experts, MetalLinear
from transformers.models.inkling2_loop import modeling_inkling2_loop as M

inductor_mps_fixes.apply()
metal = MetalConfig(bits=4, group_size=64)
plan = {"model.visual.*": None, "*_message_gate": None, "lm_head": metal, "*": metal}
model = AutoModelForImageTextToText.from_pretrained("thinkingmachines/Inkling-2-Mini", device_map="mps",
                                                    dtype=torch.bfloat16, quantization_config=plan).eval()  # fmt: skip
saved = {}
def patch(o, n, f): saved[(o, n)] = getattr(o, n); setattr(o, n, f)
def restore():
    for (o, n), f in saved.items(): setattr(o, n, f)
    saved.clear()
ABL = {
    "none": lambda: None,
    "experts": lambda: patch(MetalFp4Experts, "forward", lambda self, h, i, w: torch.zeros_like(h)),
    "attention": lambda: patch(M, "inkling2_loop_attention", lambda q, k, v, *a, **kw: (q.transpose(1, 2), None)),
    "dense_4bit": lambda: patch(MetalLinear, "forward", lambda self, x: x.new_zeros(*x.shape[:-1], self.out_features)),
    "attn_res": lambda: patch(M, "inkling2_loop_attention_residual",
                              lambda b, p, q, m: p.float().unsqueeze(-2).expand(*p.shape[:-1], q.reshape(-1, p.shape[-1]).shape[0], -1)),
    "message_gate": lambda: patch(M, "inkling2_loop_message_gate", lambda h, m, mod: h.float()),
}
start = time.perf_counter()
with torch.inference_mode():
    for n in (512, 2048):
        ids = torch.randint(1000, 50000, (1, n), device="mps")
        base = None
        for i, (name, apply) in enumerate(ABL.items()):
            apply()
            try:
                model.generate(ids, max_new_tokens=1, do_sample=False)
                ts = []
                for _ in range(3):
                    torch.mps.synchronize(); t = time.perf_counter()
                    model.generate(ids, max_new_tokens=1, do_sample=False)
                    torch.mps.synchronize(); ts.append(time.perf_counter() - t)
                ms = statistics.median(ts) * 1e3; base = ms if base is None else base
                print(f"[{n} {i + 1}/{len(ABL)}] without {name:12s} {ms:8.0f} ms (saves {base - ms:+7.0f}) | elapsed {time.perf_counter() - start:.0f}s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[{n} {i + 1}/{len(ABL)}] without {name}: failed {type(e).__name__}: {str(e)[:100]}", flush=True)
            restore()
