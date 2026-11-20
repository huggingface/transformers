"""Quality of quantizing the always-on layers (attention, lm_head) on top of 4-bit Metal experts: perplexity on fixed
text and greedy agreement with the bf16 always-on layers.

  python quality_check.py <repo> <spec> [<spec> ...]     spec: "<bits>[g<group>][:<lm_head bits>]", e.g. 4 4g32 8:4
"""

import inspect
import sys

import torch

from dense_quant import quantize_dense
from transformers import AutoModelForCausalLM, AutoTokenizer


repo = sys.argv[1] if len(sys.argv) > 1 else "ArthurZ/Qwen3-30B-A3B-metal-experts-4bit"
tok = AutoTokenizer.from_pretrained(repo)
text = inspect.getsource(torch.nn.modules.module)
ids = tok(text, return_tensors="pt").input_ids[:, 2048:3072].to("mps")  # 1024 tokens, past the license header
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Write a Python function that parses an ISO 8601 date string without datetime."}],
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    enable_thinking=False,
).input_ids.to("mps")

reference = None
for spec in [""] + sys.argv[2:]:
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="mps")
    if spec:
        quantize_dense(model, spec)
    with torch.inference_mode():
        logits = model(ids).logits.float()
        nll = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])
        out = model.generate(prompt, max_new_tokens=128, do_sample=False)[0, prompt.shape[1] :].tolist()
    reference = reference or out
    same = next((i for i, (a, b) in enumerate(zip(reference, out)) if a != b), len(out))
    print(f"always-on={spec or 'bf16'}: ppl {nll.exp().item():.4f}, greedy = bf16 for {same}/128", flush=True)
    del model
    torch.mps.empty_cache()
