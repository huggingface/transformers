import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, KernelConfig


model_id = "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random"
tokenizer = AutoTokenizer.from_pretrained(model_id)
inputs = tokenizer("Hello, how are you?", return_tensors="pt")

# --- baseline: plain model, no fusion ---
print("=" * 60)
print("Loading baseline model (no fusion)...")
baseline = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")
baseline.eval()
inputs = {k: v.to(baseline.device) for k, v in inputs.items()}

with torch.no_grad():
    baseline_out = baseline(**inputs).logits
print("Baseline output shape:", baseline_out.shape)
# del baseline

# --- fused model ---
print("=" * 60)
print("Loading fused model...")
kernel_config = KernelConfig(
    {
        (
            ("RMSNorm", "model.layers.*.post_attention_layernorm"),
            ("MLP",     "model.layers.*.mlp"),
        ): "michaelbenayoun/dummy-rmsnorm-mlp:RMSNormMLP",
    },
)

fused_model = AutoModelForCausalLM.from_pretrained(
    model_id, use_kernels=True, kernel_config=kernel_config, device_map="cuda"
)
fused_model.eval()
print(fused_model)

with torch.no_grad():
    fused_out = fused_model(**inputs).logits
print("Fused output shape:", fused_out.shape)

# --- compare ---
print("=" * 60)
print("Max diff fused vs baseline:", (fused_out - baseline_out).abs().max().item())
