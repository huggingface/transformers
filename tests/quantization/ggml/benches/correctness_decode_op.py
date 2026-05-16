"""Compare logits with vs without the decode_op fast path — must match within
quant precision so we know the all-in-one op is correct end-to-end."""
import os, torch, warnings
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "/Users/arthurzucker/.cache/huggingface/hub/models--gdax--Qwen1.5-MoE-A2.7B_gguf/snapshots/9f02e3e589316464a4ecce048d35358d54c60298"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")

def run(use_decode):
    if use_decode:
        os.environ["TRANSFORMERS_GGUF_MOE_DECODE"] = "1"
    else:
        os.environ["TRANSFORMERS_GGUF_MOE_DECODE"] = "0"
    # Reload the module so env var takes effect
    import importlib, transformers.integrations.gguf_linear as g, transformers.integrations.moe as m
    importlib.reload(g); importlib.reload(m)
    from transformers import AutoModelForCausalLM as Auto
    model = Auto.from_pretrained(repo, gguf_file="Qwen1.5-MoE-A2.7B_q4_k_m.gguf",
                                  gguf_linear=True, dtype=torch.float32).to("mps").eval()
    model.generation_config.cache_implementation = "static"
    ids = tok("The quick brown fox jumps over the", return_tensors="pt").input_ids.to("mps")
    with torch.inference_mode():
        out = model(ids)
    return out.logits[0, -1].cpu().clone()

l1 = run(use_decode=False)
l2 = run(use_decode=True)
top5_a = torch.topk(l1, 5)
top5_b = torch.topk(l2, 5)
print(f"max diff: {(l1 - l2).abs().max().item():.4e}")
print(f"rmse:     {(l1 - l2).pow(2).mean().sqrt().item():.4e}")
print(f"top-1 match: {top5_a.indices[0] == top5_b.indices[0]} ({top5_a.indices[0].item()} vs {top5_b.indices[0].item()})")
print(f"top-5 sets match: {set(top5_a.indices.tolist()) == set(top5_b.indices.tolist())}")
