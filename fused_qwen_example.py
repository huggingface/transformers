import copy
import torch
import torch.nn as nn

from kernels import Mode, register_kernel_mapping

from transformers import AutoModelForCausalLM, AutoTokenizer, KernelConfig
from transformers.module_fusion import unfuse_modules


model_id = "michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random"
tokenizer = AutoTokenizer.from_pretrained(model_id)


class FakeRMSNormMLP(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        print("Using fake RMSNormMLP kernel")
        hidden_states = self.RMSNorm(hidden_states)
        hidden_states = self.MLP(hidden_states)
        return hidden_states


class _InMemoryRepo:
    """Minimal fake repository that returns a local class instead of downloading from the hub."""

    def __init__(self, layer_cls: type):
        self.layer_name = layer_cls.__name__
        self._layer_cls = layer_cls

    def load(self) -> type:
        return self._layer_cls

    def __hash__(self):
        return hash(self._layer_cls)

    def __eq__(self, other):
        return isinstance(other, _InMemoryRepo) and self._layer_cls is other._layer_cls


# In production this would be a real hub repo string e.g. "kernels-community/rmsnorm-mlp:RMSNormMLP".
# For testing we pre-register a fake in-memory kernel so no hub download is needed.
register_kernel_mapping({
    "RMSNormMLP": {
        "cuda": {Mode.INFERENCE: _InMemoryRepo(FakeRMSNormMLP)},
    }
})

kernel_config = KernelConfig({
    ("RMSNorm", "MLP"): "fake/repo:RMSNormMLP",
})

model = AutoModelForCausalLM.from_pretrained(model_id, use_kernels=True, kernel_config=kernel_config, device_map="cuda")
model.eval()

input_ids = tokenizer("Hello, how are you?", return_tensors="pt").input_ids.to(model.device)

original_model = copy.deepcopy(model)
unfuse_modules(original_model)
original_model.eval()

with torch.no_grad():
    fused_out = model(input_ids).logits
    original_out = original_model(input_ids).logits

print("Max diff fused vs original:", (fused_out - original_out).abs().max().item())
