import copy
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, KernelConfig
from transformers.integrations import unfuse_modules


model_id = "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random"
tokenizer = AutoTokenizer.from_pretrained(model_id)

kernel_config = KernelConfig({
    (
        ("RMSNorm", "model.layers.*.post_attention_layernorm"),
        ("MLP",     "model.layers.*.mlp"),
    ): "michaelbenayoun/dummy-rmsnorm-mlp:RMSNormMLP",
})

model = AutoModelForCausalLM.from_pretrained(model_id, use_kernels=True, kernel_config=kernel_config, device_map="cuda")

input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.to(model.device)

original_model = copy.deepcopy(model)
unfuse_modules(original_model)
original_model.eval()

with torch.no_grad():
    fused_out = model(input_ids).logits
    original_out = original_model(input_ids).logits

print("Max diff fused vs original:", (fused_out - original_out).abs().max().item())
