<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Model structure rules

Transformers enforces a set of static rules on every `modeling_*.py`, `modular_*.py`, and `configuration_*.py` file. The [mlinter](https://github.com/huggingface/transformers/tree/main/utils/mlinter) tool checks them as part of `make typing` and blocks CircleCI if violations are found.

These are the expected model conventions for adding or changing modeling code. They keep the codebase consistent and ensure compatibility with features like pipeline parallelism, device maps, and weight tying.

## Running the checker

`make typing` runs `mlinter` alongside the `ty` type checker. Run `mlinter` on its own with the following commands.

```bash
python -m utils.mlinter                  # check all modeling files
python -m utils.mlinter --changed-only   # check only files changed vs origin/main
python -m utils.mlinter --list-rules     # list all rules and their enabled status
python -m utils.mlinter --rule TRF001    # show built-in docs for a specific rule
```

The `--changed-only` flag is the fastest option during development. It only checks the files you've modified relative to the main branch.

## Fixing a violation

When a rule violation is detected, the error looks like this:

```
src/transformers/models/acme/modeling_acme.py:18: TRF013: AcmeModel.__init__ does not call self.post_init().
```

Use the rule ID to look up the fix in the [rules reference](#rules-reference). TRF013 is triggered when a [`PreTrainedModel`] subclass doesn't call `self.post_init()`. That method performs essential finalization steps, and omitting it causes runtime bugs.

```diff
 class AcmeModel(AcmePreTrainedModel):
     def __init__(self, config):
         super().__init__(config)
         self.layers = nn.ModuleList(
             [AcmeDecoderLayer(config) for _ in range(config.num_hidden_layers)]
         )
+        self.post_init()
```

## Rules reference

Each rule below lists what it enforces and a diff showing the fix.

### TRF001

`config_class` on an `AcmePreTrainedModel` must reference `AcmeConfig`. The name prefix must match. A mismatch breaks [Auto](./model_doc/auto) classes and model loading.

```diff
 class AcmePreTrainedModel(PreTrainedModel):
-    config_class = NanoConfig  # wrong family; should be AcmeConfig
+    config_class = AcmeConfig
```

### TRF002

`base_model_prefix` must be a non-empty string with no whitespace. Invalid values break weight-loading key mapping and base model access patterns.

```diff
 class AcmePreTrainedModel(PreTrainedModel):
-    base_model_prefix = ""
+    base_model_prefix = "model"
```

### TRF003

`forward()` must use the `@can_return_tuple` decorator rather than manual `if not return_dict: return (x,)` branching. The old pattern is error-prone and verbose.

```diff
- def forward(self, x, return_dict=None):
-     if not return_dict:
-         return (x,)
-     return AcmeModelOutput(last_hidden_state=x)
+ @can_return_tuple
+ def forward(self, x):
+     return AcmeModelOutput(last_hidden_state=x)
```

### TRF004

Models must not override `tie_weights()`. Overriding it breaks weight loading, `device_map` computation, and saving. Declare tied weights using the `_tied_weights_keys` class attribute instead.

```diff
- def tie_weights(self):
-     self.lm_head.weight = self.emb.weight
+ class AcmeForCausalLM(AcmePreTrainedModel):
+     _tied_weights_keys = ["lm_head.weight"]
```

### TRF005

`_no_split_modules`, when defined, must be a list or tuple of non-empty strings. Bad values break device-map partitioning and sharding.

```diff
- _no_split_modules = [SomeLayerClass, ""]
+ _no_split_modules = ["AcmeDecoderLayer", "AcmeAttention"]
```

### TRF006

`forward()` must reference every cache argument (`past_key_values`, `use_cache`) declared in its signature. Unused cache arguments indicate incomplete caching support and produce an inconsistent API.

```diff
 def forward(self, x, past_key_values=None, use_cache=False):
+    if use_cache:
+        ...
     return x
```

### TRF007

`self.post_init()` must be the final call in `__init__`. Attribute assignments after it bypass initialization and finalization logic.

```diff
 def __init__(self, config):
     ...
-    self.post_init()
-    self.proj = nn.Linear(...)
+    self.proj = nn.Linear(...)
+    self.post_init()
```

### TRF008

`@add_start_docstrings` must not be called with an empty string. Empty usage produces incomplete API documentation.

```diff
- @add_start_docstrings("")
+ @add_start_docstrings("The Acme model.")
  class AcmeModel(AcmePreTrainedModel):
     ...
```

### TRF009

Each model must be self-contained in a single file. Importing implementation code from another model package makes behavior harder to inspect and maintain.

```diff
- from transformers.models.llama.modeling_llama import LlamaAttention
+ # Keep implementation local to this file.
+ # If reusing code from another model, copy it with a # Copied from comment.
```

### TRF010

Direct [`PreTrainedConfig`] subclasses in `configuration_*.py` and `modular_*.py` should use the `@strict(accept_kwargs=True)` decorator. Without it, the config class misses the runtime type-validation contract and drifts from the dataclass-based config standard.

```diff
+ @strict(accept_kwargs=True)
  class AcmeConfig(PreTrainedConfig):
      ...
```

### TRF011

`forward()` must not access non-`nn.Module` attributes on submodules. Pipeline parallelism can replace any submodule with `torch.nn.Identity`, so accessing custom attributes raises `AttributeError` at runtime. Read per-layer metadata from `self.config` instead.

```diff
 def forward(self, ...):
-    for decoder_layer in self.layers:
+    for i, decoder_layer in enumerate(self.layers):
         hidden_states = decoder_layer(
             hidden_states,
-            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
+            attention_mask=causal_mask_mapping[self.config.layer_types[i]],
         )
```

### TRF012

`_init_weights(self, module)` must not call in-place operations like `.normal_()` or `.zero_()` directly on module weights. Transformers tracks initialization state with internal flags, and in-place ops bypass that mechanism. Use `transformers.initialization` primitives instead.

```diff
+ from transformers import initialization as init
+ 
  def _init_weights(self, module):
-     module.weight.normal_(mean=0.0, std=0.02)
+     init.normal_(module.weight, mean=0.0, std=0.02)
```

### TRF013

Every [`PreTrainedModel`] subclass with an `__init__` method must call `self.post_init()`. In modular files, `super().__init__()` is also accepted, and it propagates `post_init()` from the parent. Omitting it skips essential finalization and causes runtime bugs.

```diff
  class AcmeModel(AcmePreTrainedModel):
      def __init__(self, config):
          super().__init__(config)
          self.layers = nn.ModuleList(...)
+         self.post_init()
```

### TRF014

`trust_remote_code` should never be passed or used in native model integration files. It allows arbitrary code loading which native integrations should never depend on.

```diff
  class AcmeModel(AcmePreTrainedModel):
      def __init__(self, config):
          super().__init__(config)
-         self.model = AutoModel.from_pretrained(..., trust_remote_code=True)
+         self.model = AutoModel.from_pretrained(...)
```

## Suppressing violations

If you need to suppress a rule violation, use one of the two options below.

### Inline suppression

Add a `# trf-ignore: RULE_ID` comment on the violating line. Include an explanation so reviewers understand why the suppression is justified.

```py
# trf-ignore: TRF011 — mask is derived from self.config, not the layer
hidden_states = layer(hidden_states, attention_mask=mask_from_config)
```

Don't use `trf-ignore` to silence violations that should be fixed in the code.

### `allowlist_models`

For models with legacy code that can't be fixed immediately, add the model's directory name to the relevant rule's `allowlist_models` list in `utils/mlinter/rules.toml`.

```toml
[rules.TRF004]
allowlist_models = ["existing_model", "your_model_name"]
```
