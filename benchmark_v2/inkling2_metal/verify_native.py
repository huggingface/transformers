"""Inkling-2-Mini on MPS: experts kept in the checkpoint's NVFP4 / MXFP8 (quantization plan read from the safetensors
dtypes, `MetalFp4Experts`) vs the bf16-dequantized load. Same prompt, next-token logits compared.

    PYTHONPATH=src python benchmark_v2/inkling2_metal/verify_native.py [native|dequantized|both]
"""

import sys
import time

import torch

from transformers import AutoModelForImageTextToText, AutoTokenizer


REPO = "thinkingmachines/Inkling-2-Mini"
OUT = "/tmp/inkling2_logits_{}.pt"


def run(mode):
    kwargs = {"device_map": "mps", "dtype": torch.bfloat16}
    if mode == "dequantized":  # run on the base Inkling2 Mini PR, which dequantizes the experts to bf16 when loading
        pass
    start = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(REPO, **kwargs).eval()
    experts = {type(m).__name__ for n, m in model.named_modules() if n.endswith("mlp.experts")}
    print(f"[{mode}] loaded in {time.perf_counter() - start:.0f}s, experts: {experts}, "
          f"allocated {torch.mps.current_allocated_memory() / 2**30:.1f} GiB", flush=True)  # fmt: skip
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    messages = [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    ids = ids["input_ids"].to("mps")
    with torch.inference_mode():
        logits = model(ids).logits[0, -1].float().cpu()
        out = model.generate(ids, max_new_tokens=24, do_sample=False)
    print(f"[{mode}] {tokenizer.decode(out[0, ids.shape[1] :])!r}", flush=True)
    torch.save(logits, OUT.format(mode))
    return logits


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "native"
    if mode in ("native", "dequantized"):
        run(mode)
    else:
        native, dequantized = torch.load(OUT.format("native")), torch.load(OUT.format("dequantized"))
        cos = torch.nn.functional.cosine_similarity(native, dequantized, dim=0).item()
        top = (native.topk(10).indices == dequantized.topk(10).indices).float().mean().item()
        print(f"cosine {cos:.5f}, top-10 agreement {top:.0%}, argmax equal {native.argmax() == dequantized.argmax()}")
