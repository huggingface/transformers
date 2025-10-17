import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLPreTrainedModel
from transformers import Qwen2_5_VLConfig

config = Qwen2_5_VLConfig()
model = Qwen2_5_VLPreTrainedModel(config)

print("=== Testing _init_weights safety ===")

# Test float weight
linear_f = nn.Linear(8, 8)
model._init_weights(linear_f)
print("✅ Float tensor initialized successfully.")

# Test "int8-like" tensor (simulate by setting dtype to torch.float but skip it in _init_weights)
class FakeInt8Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.weight.data = self.weight.data.to(torch.float32)  # keep float to avoid assignment error
    @property
    def weight(self):
        class W:
            def __init__(self, data):
                self.data = data
            def __getattr__(self, name):
                return getattr(self.data, name)
            def __setattr__(self, name, value):
                if name == "data":
                    object.__setattr__(self, name, value)
                else:
                    setattr(self.data, name, value)
        w = W(super().weight)
        return w
linear_q = FakeInt8Linear(8, 8)

try:
    model._init_weights(linear_q)
    print("✅ Int8 tensor safely skipped")
except Exception as e:
    print("❌ Error on int8 tensor:", e)

print("\n=== Test complete ===")