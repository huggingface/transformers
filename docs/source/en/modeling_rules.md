<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Model structure rules

Transformers enforces a set of static rules on every `modeling_*.py`, `modular_*.py`, and `configuration_*.py` file. The [mlinter](https://github.com/huggingface/transformers/tree/main/utils/mlinter) tool checks them as part of `make typing` and errors out if violations are found.

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

Each rule below lists what it enforces and a diff showing the fix. Run `python -m utils.mlinter --rule TRF001` to see the built-in docs for any rule.

<!-- BEGIN RULES REFERENCE -->

### TRF001

Checks naming consistency between <Model>PreTrainedModel and config_class. Mismatched config_class can break loading, auto classes, and developer expectations.

```diff
class AcmePreTrainedModel(PreTrainedModel):
-    config_class = WileConfig
+    config_class = AcmeConfig
```

### TRF002

Checks that base_model_prefix, when set, is a non-empty, whitespace-free string literal. Invalid prefixes can break weight loading key mapping and base model access patterns.

```diff
class AcmePreTrainedModel(PreTrainedModel):
-    base_model_prefix = ""
+    base_model_prefix = "model"
```

### TRF003

Detects forward methods that use the old 'if not return_dict: return (x,)' pattern. The old return_dict branching pattern is error-prone and verbose. Use the capture_output or can_return_tuple decorators instead.

```diff
-def forward(self, x, return_dict=None):
-    if not return_dict:
-        return (x,)
-    return AcmeModelOutput(last_hidden_state=x)
+@can_return_tuple
+def forward(self, x):
+    return AcmeModelOutput(last_hidden_state=x)
```

### TRF004

Checks that no model class defines a tie_weights method. Overriding tie_weights leads to bad consequences for loading, device_map computation, and saving. Use _tied_weights_keys class attribute to declare tied weights instead.

```diff
-def tie_weights(self):
-    self.lm_head.weight = self.emb.weight
+class AcmeForCausalLM(AcmePreTrainedModel):
+    _tied_weights_keys = ["lm_head.weight"]
```

### TRF005

Checks the shape of _no_split_modules when present. Malformed values can break device-map partitioning and sharding behavior.

```diff
-_no_split_modules = [SomeLayerClass, ""]
+_no_split_modules = ["AcmeDecoderLayer", "AcmeAttention"]
```

### TRF006

Checks forward signatures that expose cache arguments for usage of those arguments in method body. Unused cache arguments can indicate incomplete caching support and inconsistent API behavior.

```diff
def forward(self, x, past_key_values=None, use_cache=False):
+    if use_cache:
+        ...
     return x
```

### TRF007

Checks for self attribute assignments after self.post_init() in __init__. Mutating model structure after post_init can bypass intended initialization/finalization logic.

```diff
def __init__(self, config):
     ...
-    self.post_init()
-    self.proj = nn.Linear(...)
+    self.proj = nn.Linear(...)
+    self.post_init()
```

### TRF008

Checks add_start_docstrings usage on model classes for non-empty docstring arguments. Empty decorator usage produces unclear docs and weakens generated API documentation quality.

```diff
-@add_start_docstrings("")
+@add_start_docstrings("The Acme model.")
 class AcmeModel(AcmePreTrainedModel):
     ...
```

### TRF009

Checks modeling files for cross-model imports such as transformers.models.other_model.* or from ..other_model.* imports. Cross-model implementation imports violate the single-file policy and make model behavior harder to inspect and maintain.

```diff
-from transformers.models.llama.modeling_llama import LlamaAttention
+# Keep implementation local to this file.
+# If reusing code, copy it with a # Copied from comment.
```

### TRF010

Checks direct PreTrainedConfig/PretrainedConfig subclasses in configuration_*.py and modular_*.py for an explicit @strict(accept_kwargs=True) decorator. Without strict, new config classes miss the repo's runtime type-validation contract and drift from the dataclass-based config standard.

```diff
+@strict(accept_kwargs=True)
 class AcmeConfig(PreTrainedConfig):
     ...
```

### TRF011

In forward() methods of PreTrainedModel subclasses, checks for attribute accesses on submodules that would not exist on torch.nn.Identity. This includes attribute accesses on loop variables iterating over self.layers, and self.<submodule>.<attr> chains where <attr> is not a standard nn.Module attribute. Pipeline parallelism may replace any submodule with torch.nn.Identity. Accessing custom attributes (e.g. decoder_layer.attention_type) on a replaced module raises AttributeError at runtime. Per-layer metadata should be read from self.config instead.

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

Checks that _init_weights(self, module) does not use in-place operations (e.g. .normal_(), .zero_()) directly on module weights. We rely on internal flags set on parameters to track whether they need re-initialization. In-place ops bypass this mechanism. Use the `init` primitives instead.

```diff
+from transformers import initialization as init
+
 def _init_weights(self, module):
-    module.weight.normal_(mean=0.0, std=0.02)
+    init.normal_(module.weight, mean=0.0, std=0.02)
```

### TRF013

Checks that every PreTrainedModel subclass with an __init__ method calls self.post_init(). In modular files, calling super().__init__() is also accepted since it propagates post_init from the parent. post_init performs essential finalization (weight initialization, gradient checkpointing setup, etc.). Omitting it causes subtle runtime bugs.

```diff
class AcmeModel(AcmePreTrainedModel):
     def __init__(self, config):
         super().__init__(config)
         self.layers = nn.ModuleList(...)
+        self.post_init()
```

### TRF014

Checks whether `trust_remote_code` is passed or used in code (e.g. as kwarg) within native model integration files. `trust_remote_code` allows arbitrary loading, including binaries, which should only be a power feature for users, not a standard use-case. Native integrations must not depend on it, as remote code cannot be reviewed or maintained within transformers.

```diff
class AcmeModel(AcmePreTrainedModel):
     def __init__(self, config):
         super().__init__(config)
-        self.model = AutoModel.from_pretrained(..., trust_remote_code=True)
+        self.model = AutoModel.from_pretrained(...)
```

<!-- END RULES REFERENCE -->

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
