# Reviewer-Enforced Standards

These standards are derived from actual reviewer feedback on model PRs (primarily vasqu's review of PR #44320, SAM3-LiteText). Violations will be flagged and changes requested.

## Modeling Code

### nn.ModuleList over nn.Sequential

Bad:
```python
self.layers = nn.Sequential(
    ConvLayer(hidden_size),
    BatchNorm(hidden_size),
)
```

Good:
```python
self.conv = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, bias=False)
self.norm = nn.BatchNorm2d(hidden_size)
```

Or for variable-length layer lists:
```python
self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])
```

### nn.Linear for projections

Bad:
```python
self.projection = nn.Parameter(torch.empty(config.hidden_size, config.projection_dim))
# manual matmul in forward: output = hidden @ self.projection
```

Good:
```python
self.projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)
```

### Reuse existing components

Bad — rewriting MLP from scratch:
```python
class MyModelMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
```

Good — inherit from existing:
```python
from ..clip.modeling_clip import CLIPMLP

class MyModelMLP(CLIPMLP):
    pass  # or override only what differs
```

### Configurable constants

Bad:
```python
self.layer_scale = nn.Parameter(1e-5 * torch.ones((hidden_size,)), requires_grad=True)
```

Good:
```python
# In config:
layer_scale_init: float = 1e-5

# In model:
self.layer_scale = nn.Parameter(
    config.layer_scale_init * torch.ones((config.hidden_size,)), requires_grad=True
)
```

### Clean naming

Bad — keeping opaque names from original codebase:
```python
self.rbr_skip = nn.BatchNorm2d(hidden_size)
self.rbr_conv = nn.ModuleList([...])
```

Good — descriptive names:
```python
self.skip_norm = nn.BatchNorm2d(hidden_size)
self.conv_branches = nn.ModuleList([...])
```

If you must keep a name, document what it means.

### Minimal PreTrainedModel overrides

Bad — overriding attributes to their default values:
```python
class MyPreTrainedModel(PreTrainedModel):
    config_class = MyConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    _supports_flex_attn = True
    _supports_attention_backend = True
    _no_split_modules = [...]
    _skip_keys_device_placement = [...]
    # ... 10 more attributes that are already the default
```

Good — only override what differs:
```python
class MyPreTrainedModel(PreTrainedModel):
    config_class = MyConfig
    main_input_name = "pixel_values"
    input_modalities = ["image", "text"]
    _no_split_modules = ["MyEncoderLayer", "MyDecoderLayer"]
```

### Data transforms inside layers

Bad — permutations in parent forward loop:
```python
# In the model's forward:
for idx, layer in enumerate(self.layers):
    if idx in self.special_indices:
        hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)
        hidden_states = layer(hidden_states)
        hidden_states = hidden_states.squeeze(2).permute(0, 2, 1)
    else:
        hidden_states = layer(hidden_states)
```

Good — each layer handles its own format:
```python
# In the special layer's forward:
def forward(self, hidden_states):
    hidden_states = hidden_states.permute(0, 2, 1).unsqueeze(2)
    # ... do computation ...
    return hidden_states.squeeze(2).permute(0, 2, 1)

# In the model's forward (clean):
for layer in self.layers:
    hidden_states = layer(hidden_states)
```

### Conditional layers with nn.Identity

Bad:
```python
def forward(self, hidden_states):
    if self.config.use_special_mixer:
        hidden_states = self.special_mixer(hidden_states)
    # else: pass through
```

Good:
```python
def __init__(self, config):
    self.token_mixer = SpecialMixer(config) if config.use_special_mixer else nn.Identity()

def forward(self, hidden_states):
    hidden_states = self.token_mixer(hidden_states)
```

### Attention support flags

Bad — skipping tests:
```python
# In test file:
@unittest.skip("Flash attention not compatible with float masks")
def test_flash_attn_2_inference_equivalence(self):
    pass
```

Good — setting flags in model:
```python
# In model file:
class MyPreTrainedModel(PreTrainedModel):
    _supports_flash_attn = False  # float attention masks incompatible
```

### @capture_outputs decorator

Bad:
```python
@capture_outputs(tie_last_hidden_states=False)
def forward(self, ...):
```

Good (unless the parameter is truly needed for backward compatibility):
```python
@capture_outputs
def forward(self, ...):
```

## Meta-observation

Always apply a simplification pass:

- Remove redundant abstractions
- Flatten unnecessary nesting
- Replace verbose patterns with existing library utilities
- Question every attribute override and magic number
