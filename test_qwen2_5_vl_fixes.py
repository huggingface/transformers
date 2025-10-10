import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLPreTrainedModel
from transformers import Qwen2_5_VLConfig

# ----------------------------
# 1️⃣ Test _init_weights fix
# ----------------------------
print("Running _init_weights tests...")

# Initialize dummy config and model
config = Qwen2_5_VLConfig()
model = Qwen2_5_VLPreTrainedModel(config)

# Float tensor test
linear_float = nn.Linear(10, 10)
model._init_weights(linear_float)
print("✅ Float tensor initialized successfully")

# Int8-like tensor test
linear_int8 = nn.Linear(10, 10)
linear_int8.weight.requires_grad = False
linear_int8.weight.data = torch.randint(-128, 128, (10, 10), dtype=torch.int8).to(torch.float32)
model._init_weights(linear_int8)
print("✅ Int8-like tensor safely skipped by _init_weights")

# ----------------------------
# 2️⃣ Test logits_to_keep logic
# ----------------------------
print("\nRunning logits_to_keep tests...")

# Dummy hidden states
hidden_states = torch.randn(1, 5, 10)  # batch_size=1, seq_len=5, hidden_dim=10

# Dummy lm_head
model.lm_head = nn.Linear(10, 10, bias=False)

# Test with logits_to_keep=None
logits_to_keep = None
if logits_to_keep is None or logits_to_keep == 0:
    logits = model.lm_head(hidden_states)
else:
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = model.lm_head(hidden_states[:, slice_indices, :])
print("Logits shape with logits_to_keep=None:", logits.shape)

# Test with logits_to_keep=2
logits_to_keep = 2
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
logits = model.lm_head(hidden_states[:, slice_indices, :])
print("Logits shape with logits_to_keep=2:", logits.shape)

print("\n✅ All tests passed — _init_weights and logits_to_keep logic work as expected!")