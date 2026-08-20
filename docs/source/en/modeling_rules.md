<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Model structure rules

Transformers enforces a set of static rules on every `modeling_*.py`, `modular_*.py`, and `configuration_*.py` file. The [mlinter](https://github.com/huggingface/transformers-mlinter) package provides the checker engine, and the repository keeps its active rule set in `utils/rules.toml`. That local TOML lets us enable, disable, or tweak rules quickly without waiting for a new `transformers-mlinter` release.

These are the expected model conventions for adding or changing modeling code. They keep the codebase consistent and ensure compatibility with features like pipeline parallelism, device maps, and weight tying.

## Running the checker

`make typing` runs `mlinter` alongside the `ty` type checker through the repo wrapper, so it picks up `utils/rules.toml`. Run the same wrapper directly with the following commands.

```bash
python utils/check_modeling_structure.py                 # check all modeling files
python utils/check_modeling_structure.py --changed-only  # check only files changed vs origin/main
python utils/check_modeling_structure.py --list-rules    # list all rules and their enabled status
python utils/check_modeling_structure.py --rule TRF001   # show built-in docs for a specific rule
```

The `--changed-only` flag is the fastest option during development. It only checks the files you've modified relative to the main branch. If you invoke `mlinter` directly instead of the wrapper, pass `--rules-toml utils/rules.toml` so local overrides are applied.

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

Each rule below lists what it enforces and a diff showing the fix. Run `python utils/check_modeling_structure.py --rule TRF001` to see the built-in docs for any rule with the repo's current rule set.

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

### TRF015

When a PreTrainedModel subclass defines _tied_weights_keys as a non-empty collection, checks that the corresponding configuration file declares a tie_word_embeddings field. Without tie_word_embeddings in the config, users cannot control weight tying behavior. The model ties weights unconditionally, breaking serialization round-trips and preventing fine-tuning with untied heads.

```diff
# configuration_foo.py
 @strict(accept_kwargs=True)
 class FooConfig(PreTrainedConfig):
     hidden_size: int = 768
+    tie_word_embeddings: bool = True
```

### TRF016

When an image_processing_*.py or video_processing_*.py class declares boolean do_* attributes (e.g. do_resize, do_rescale, do_normalize, do_convert_rgb) and overrides preprocess() or _preprocess(), checks that each declared flag is still consumed along the override path. That can be a direct reference in the override body, delegating back to the base implementation via super().preprocess(..., **kwargs) or super()._preprocess(..., **kwargs), or, for image processors, forwarding do_convert_rgb into the shared image-preparation path via _preprocess_image_like_inputs(...) or _prepare_image_like_inputs(...). The allowlist of base-handled flags (do_sample_frames) is exempted because the base preprocess() consumes them before _preprocess() runs. A do_X attribute that is not referenced by the override is a dead flag: setting do_X=False at construction or call time has no effect, and the underlying operation runs unconditionally. This silently breaks user expectations and makes per-call overrides ineffective.

```diff
class AcmeVideoProcessor(BaseVideoProcessor):
     do_resize = True
     do_normalize = True

     def _preprocess(
         self,
         videos,
+        do_resize: bool,
+        do_normalize: bool,
         size,
         image_mean,
         image_std,
         **kwargs,
     ):
         for video in videos:
-            video = self.resize(video, size=size)
-            video = self.normalize(video, image_mean, image_std)
+            if do_resize:
+                video = self.resize(video, size=size)
+            if do_normalize:
+                video = self.normalize(video, image_mean, image_std)
```

### TRF017

Checks classes decorated with both @auto_docstring and @dataclass for source ordering: @auto_docstring must appear above @dataclass. Decorators are applied bottom-up. When @dataclass is listed above @auto_docstring, @auto_docstring runs first on a class that has no synthesized __init__ yet and ends up modifying the parent class's __init__.__doc__ instead of the subclass's.

```diff
-@dataclass
 @auto_docstring(
     custom_intro="""
     Output type of [`AcmeForPreTraining`].
     """
 )
+@dataclass
 class AcmeForPreTrainingOutput(ModelOutput):
     ...
```

### TRF018

Checks that every PreTrainedModel subclass that overrides `_init_weights(self, module, ...)` chains the call up via `super()._init_weights(...)`. In modular files, `PreTrainedModel._init_weights(self, module)` and `raise AttributeError(...)` are accepted because they are modularization sentinels. If a model intentionally fully overrides initialization, suppress with `# trf-ignore: TRF018` on the line above the method. The base `_init_weights` covers standard module types (Linear, Embedding, LayerNorm, RotaryEmbedding, ...). Skipping `super()._init_weights(...)` silently leaves submodules unhandled by the override uninitialized, which can pass tests and surface much later as subtle weight-init bugs (cf. https://github.com/huggingface/transformers/pull/45597).

```diff
from ... import initialization as init

 def _init_weights(self, module):
+    super()._init_weights(module)
     if isinstance(module, AcmeCustomLayer):
-        module.gate.data.zero_()
+        init.zeros_(module.gate)
```

### TRF019

Checks that `*ProcessorKwargs` TypedDict classes in `processing_*.py` files do not set a non-empty `_defaults` dict. Old models released before cutoff date are not checked against the rule for backwards compatibility; new models must not hardcode defaults in Python. Hardcoding defaults in `_defaults` scatters processor configuration across Python source files, makes it unintuitive when it comes to overriding defaults via config, and bloats up the code. The canonical home for processor defaults is `processor_config.json` on the hub, which is shipped with the checkpoint and can be updated without touching code.

```diff
class Gemma4ProcessorKwargs(ProcessingKwargs, total=False):
-    _defaults = {
-        "text_kwargs": {"padding": False},
-        "images_kwargs": {"return_tensors": "pt"},
-    }
     images_kwargs: Gemma4ImageProcessorKwargs
```

### TRF020

In model directories whose configuration declares `kv_lora_rank` (Multi-head Latent Attention), checks the attention class that owns the KV LoRA expansion projection (conventionally `kv_b_proj`, or any `nn.Linear(config.kv_lora_rank, ...)`). The expansion must not be applied inside `forward()`; it must live in a dedicated method (e.g. `expand_kv`) that `forward()` calls. External backends (vLLM/SGLang) override the KV LoRA expansion so they can store and consume the compressed KV cache directly instead of the materialized key/value states. When the expansion is inlined in `forward()`, there is no single method to override and the backend is forced to materialize the full cache, losing the memory savings that MLA exists to provide.

```diff
+    def expand_kv(self, k_nope, k_pe):
+        key_shape = (*k_nope.shape[:-1], -1, self.qk_nope_head_dim + self.v_head_dim)
+        k_nope = self.kv_b_proj(k_nope).view(key_shape).transpose(1, 2)
+        k_nope, value_states = torch.split(k_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
+        k_pe = k_pe.expand(*k_nope.shape[:-1], -1)
+        key_states = torch.cat((k_nope, k_pe), dim=-1)
+        return key_states, value_states
+
     def forward(self, hidden_states, ...):
         ...
-        k_nope = self.kv_b_proj(k_pass).view(key_shape).transpose(1, 2)
-        k_nope, value_states = torch.split(k_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
-        k_pe = k_rot.expand(*k_nope.shape[:-1], -1)
-        key_states = torch.cat((k_nope, k_pe), dim=-1)
+        key_states, value_states = self.expand_kv(k_pass, k_rot)
```

### TRF021

In modeling_*.py and modular_*.py, checks calls to torch.tensor(<value>, ..., device=<non-cpu>) whose <value> provably evaluates to a Python scalar. Scalar-ness is resolved statically: numeric literals, arithmetic over them, torch.finfo/torch.iinfo fields, scalar-returning builtins and math.* calls, locals bound exactly once, self.<attr> assigned in the class body, and self.config.<field>/config.<field> whose annotation in the companion configuration file is int/float/bool (following attribute_map aliases). Fields that may also be sequences, such as `eos_token_id: int | list[int] | None`, and any expression that cannot be resolved are left alone. Construction-time methods (__init__, _init_weights, __post_init__, post_init) are exempt because they never run inside a capture region. torch.tensor(<python scalar>, device=<accelerator>) materialises the value on the host and then issues a host-to-device copy. CUDA graph capture forbids that copy, so the model cannot be captured. torch.full((), <value>, dtype=..., device=...) fills the same 0-d tensor directly on-device with a capturable kernel and no synchronisation.

```diff
def get_placeholder_mask(self, input_ids, inputs_embeds):
     special_image_mask = (
         inputs_embeds
         == self.get_input_embeddings()(
-            torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
+            torch.full((), self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
         )
     ).all(-1)
```

### TRF022

Checks that every string in a `_no_split_modules` list on a class in a `modeling_*.py` or `modular_*.py` file names a class that is defined in that file, imported into it, or defined by a sibling module of the same model directory. Complements TRF005, which only validates the shape of the value. `device_map` resolves `_no_split_modules` by comparing the strings to `module.__class__.__name__` at runtime. A stale or misspelled name matches nothing and is silently ignored, so the module it was meant to keep together can still be split across devices. Entries naming another model's classes are also redundant and must be dropped rather than corrected: `post_init` already collects `_no_split_modules` from child submodels, so a submodel's layers are registered automatically.

```diff
class VideoLlavaPreTrainedModel(PreTrainedModel):
-    _no_split_modules = ["VideoLlavaVisionAttention"]
```

### TRF023

In configuration_*.py and modular_*.py, checks classes whose name ends in Config for fields declared under an upstream paper's abbreviation instead of the library's canonical name: d_model/n_embd (hidden_size), d_ff/d_inner/ffn_dim/ffn_hidden_size/expansion_ratio (intermediate_size), d_head (head_dim), n_head/n_heads (num_attention_heads), n_layer/n_layers/num_blocks (num_hidden_layers). Fields are collected from the class body and from __init__/__post_init__ assignments and signature defaults. Ambiguous names that are still idiomatic in parts of the library (num_heads, num_layers, embed_dim, mlp_ratio) are deliberately not flagged. Models contributed before cutoff_date keep their existing names. Every generic that reads a model's shape — device_map planning, tensor/pipeline parallel plans, quantization, PEFT, attention-backend selection, `attribute_map` consumers — looks up the canonical names. A config that spells the same quantity `d_model` silently opts out of all of it, and the mismatch has to be rediscovered by a reviewer on every single new model. Derive the checkpoint's own spelling in the conversion script instead.

```diff
@strict(accept_kwargs=True)
 class AcmeConfig(PreTrainedConfig):
-    d_model: int = 1024
-    d_ff: int = 4096
-    n_heads: int = 16
-    n_layers: int = 24
+    hidden_size: int = 1024
+    intermediate_size: int = 4096
+    num_attention_heads: int = 16
+    num_hidden_layers: int = 24
```

### TRF024

In modeling_*.py and modular_*.py, checks torch.nn layer constructors (Linear, Embedding, LayerNorm, RMSNorm, GroupNorm, BatchNorm*, InstanceNorm*, Conv*d, ConvTranspose*d, Bilinear, MultiheadAttention) for an integer literal greater than 8 in a dimension position, whether passed positionally or by keyword (in_features, out_features, in_channels, out_channels, num_embeddings, embedding_dim, embed_dim, normalized_shape, num_channels, hidden_size). Operator-shape arguments such as kernel_size, stride, padding and num_groups are ignored, and literals up to 8 are allowed so scalar heads, binary classifiers and RGB channel counts stay clean. Models contributed before cutoff_date are exempt. A hardcoded width silently pins the module to one checkpoint size: the same architecture at another scale loads with a shape mismatch, and `from_pretrained` cannot report which value is wrong because there is no config field to point at. It also splits the source of truth, so a config that is edited no longer describes the model that gets built.

```diff
class AcmeAtomEmbedding(nn.Module):
     def __init__(self, config):
         super().__init__()
-        self.proj = nn.Linear(768, 3072, bias=False)
-        self.norm = nn.LayerNorm(3072)
+        self.proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
+        self.norm = nn.LayerNorm(config.intermediate_size)
```

### TRF025

In modeling_*.py and modular_*.py, checks calls to a mask factory (the masking_utils entry points create_causal_mask, create_bidirectional_mask, create_sliding_window_causal_mask, create_chunked_causal_mask, create_masks_for_generate, and any create_*_mask helper) that occur inside a class which does not inherit from PreTrainedModel. Top-level models and sub-models are where masks are meant to be created, so only plain nn.Module blocks — layers, attention modules, encoders — are in scope. Mask construction is O(sequence length squared) work that does not vary per layer. Building it inside the layer repeats that cost once per layer, and because each layer then owns its own mask, the attention backends can no longer be given a single prepared mask — which is how manual padding fixups and per-layer mask divergence get introduced. Create it once in the model and pass it down.

```diff
class AcmeLayer(nn.Module):
     def forward(self, hidden_states, attention_mask=None, **kwargs):
-        attention_mask = create_causal_mask(
-            config=self.config, input_embeds=hidden_states, attention_mask=attention_mask, ...
-        )
         return self.self_attn(hidden_states, attention_mask, **kwargs)

 class AcmeModel(AcmePreTrainedModel):
     def forward(self, input_ids=None, attention_mask=None, **kwargs):
+        causal_mask = create_causal_mask(
+            config=self.config, input_embeds=inputs_embeds, attention_mask=attention_mask, ...
+        )
         for layer in self.layers:
-            hidden_states = layer(hidden_states, attention_mask, **kwargs)
+            hidden_states = layer(hidden_states, causal_mask, **kwargs)
```

### TRF026

In modeling_*.py and modular_*.py, flags a class that is not a PreTrainedModel subclass, defines only __init__ and forward, assigns exactly one self.<attr> in __init__, and whose forward body is exactly `return self.<attr>(...)` for that same attribute. A leading docstring is ignored. Classes with any other method, any additional attribute, or any statement before the return are left alone because they do work of their own. The wrapper adds a level to every weight name and to _no_split_modules, tensor-parallel and pipeline plans, and to every conversion mapping, while contributing no computation. Readers then have to open one more class to discover that nothing happens in it, which is the single most common structural review comment on new models. PreTrainedModel subclasses are exempt: those exist for from_pretrained and the auto classes even when the forward only delegates.

```diff
-class AcmeAtomTransformer(nn.Module):
-    def __init__(self, config):
-        super().__init__()
-        self.encoder = AcmeEncoder(config)
-
-    def forward(self, hidden_states, **kwargs):
-        return self.encoder(hidden_states, **kwargs)
-
 class AcmeModel(AcmePreTrainedModel):
     def __init__(self, config):
         super().__init__(config)
-        self.atom_transformer = AcmeAtomTransformer(config)
+        self.encoder = AcmeEncoder(config)
```

### TRF027

Flags any `assert` statement in modeling_*.py, modular_*.py and configuration_*.py. `python -O` strips assert statements, so a shape or config check written as an assert silently disappears in optimised runs. An assert also gives the user a bare AssertionError with no guidance, where a ValueError can name the offending value and what to do about it.

```diff
def forward(self, hidden_states):
-    assert hidden_states.dim() == 3
+    if hidden_states.dim() != 3:
+        raise ValueError(f"Expected a 3D tensor, got shape {tuple(hidden_states.shape)}.")
```

### TRF028

Checks the first 25 lines of modeling_*.py, modular_*.py, configuration_*.py, processing_*.py, image_processing_*.py and video_processing_*.py for a `Licensed under the <name> License` line followed by every clause of the standard warranty paragraph, from `You may obtain a copy of the License at` through `limitations under the License.`. The lines are flattened and lowercased before matching, so wrapping and comment style do not matter. Neither the license name nor the copyright line is checked: BLIP is BSD-3-clause and Sapiens2 carries Meta's own license, and the copyright year and attribution vary per model. A new model file that ships without the header leaves its provenance ambiguous, and adding it later means touching a file that has already been released. Matching only the words `Apache License` would accept a header truncated mid-paragraph or mangled by a bad search-and-replace, which is what every header defect in the library actually looks like: bitnet loses the closing `limitations under the License.`, tvp and bridgetower have a stray `=` before every comma, and minimax_m3_vl stops after the URL.

```diff
+# Copyright 2026 The HuggingFace Team. All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# ...
 """PyTorch Acme model."""
```

### TRF029

In modeling_*.py and modular_*.py, flags an `__init__` that accepts `config` alongside an argument whose name is unambiguously a config field (hidden_size, num_attention_heads, intermediate_size, head_dim, num_hidden_layers, embed_dim, dropout, eps, patch_size, rope_theta, ...). kosmos2 is allowlisted because its doc page (kosmos-2.md) cannot be derived from the directory name, so the cutoff cannot grandfather it. The same number now has two sources of truth and the caller decides which one wins, so editing the config no longer changes the model that gets built. It also makes every call site carry architecture knowledge that belongs inside the module, which is why reviewers ask for it on new models over and over.

```diff
class AcmeAttention(nn.Module):
-    def __init__(self, config, embed_dim, num_heads, dropout):
+    def __init__(self, config, layer_idx=None):
         super().__init__()
-        self.embed_dim = embed_dim
-        self.num_heads = num_heads
+        self.embed_dim = config.hidden_size
+        self.num_heads = config.num_attention_heads
```

### TRF030

In modeling_*.py and modular_*.py, flags attribute chains rooted at `config` or `self.config` that go three or more levels deep. `config.hidden_size` (one hop) and `config.text_config.hidden_size` (two hops, the normal sub-config access) are fine; one violation is reported per line. A module that walks `config.diffusion_config.atom_encoder_config.hidden_size` is coupled to the whole config hierarchy rather than to its own slice of it, so it cannot be reused, tested or given a different sub-config. Pass the relevant sub-config down and the chain collapses to one hop.

```diff
class AcmeAtomEncoder(nn.Module):
     def __init__(self, config):
         super().__init__()
-        self.norm = AcmeLayerNorm(config.diffusion_config.atom_encoder_config.hidden_size)
+        self.norm = AcmeLayerNorm(config.hidden_size)
```

### TRF031

In modeling_*.py and modular_*.py, flags a top-level `@dataclass` class whose bases do not include something ending in `Output`. A plain dataclass does not index like a tuple, does not survive `return_dict=False`, and is invisible to @auto_docstring, so its fields never reach the generated API docs. Inheriting ModelOutput gets all three for free.

```diff
@auto_docstring
 @dataclass
-class AcmeStructureOutput:
+class AcmeStructureOutput(ModelOutput):
     positions: torch.Tensor
     confidence: torch.Tensor
```

### TRF032

In modeling_*.py and modular_*.py, flags masked_fill, masked_fill_, full, full_like and new_full called with a negated numeric literal of magnitude 1e3 or more. A hardcoded -1e9 overflows to -inf in float16 and is not nearly the smallest value in float32, so the same mask behaves differently per dtype and can produce NaNs after softmax. `torch.finfo(dtype).min` is the smallest representable value in whatever dtype is actually running.

```diff
-attention_scores = attention_scores.masked_fill(~mask, -1e9)
+attention_scores = attention_scores.masked_fill(~mask, torch.finfo(attention_scores.dtype).min)
```

### TRF033

In modeling_*.py and modular_*.py, flags methods whose name starts with `set_`, except the PreTrainedModel contract methods set_input_embeddings, set_output_embeddings, set_decoder, set_encoder, set_attn_implementation and set_default_language. A setter makes the model's behaviour depend on call order: the value is not in the config, so it is not saved, not restored by from_pretrained, and not visible to anything planning device maps or parallelism. Users then have to know to call it, and forgetting is silent.

```diff
class AcmeTriangleAttention(nn.Module):
-    def set_chunk_size(self, chunk_size):
-        self.chunk_size = chunk_size
+    def __init__(self, config):
+        super().__init__()
+        self.chunk_size = config.chunk_size
```

### TRF034

In modeling_*.py and modular_*.py, flags a locally-defined class whose name ends in `Layer` or `Block`, instantiated inside an `nn.ModuleList(...)`, that does not reach `GradientCheckpointingLayer` through its local base chain. One violation per layer class. ModuleLists of projections, heads or experts are out of scope because they are not checkpointing boundaries. `gradient_checkpointing_enable()` wraps layers by asking each one whether it is a GradientCheckpointingLayer. A plain nn.Module in the stack is skipped silently, so training appears to use checkpointing and still allocates full activations for those layers, and the OOM shows up far from the cause.

```diff
-class AcmeDecoderLayer(nn.Module):
+class AcmeDecoderLayer(GradientCheckpointingLayer):
     def __init__(self, config, layer_idx):
         super().__init__()
```

### TRF035

Flags `# noqa` comments, with or without explicit codes, in modeling_*.py, modular_*.py and configuration_*.py. Model files are ordinary code held to the repo's lint rules; a suppression there means the underlying issue was left in place, and blanket `# noqa` also hides every future violation on that line. Reviewers spend threads deleting these on new models, and they cluster in machine-generated code where the generator silenced the linter rather than satisfying it.

```diff
-from ...modeling_utils import PreTrainedModel  # noqa: F401
+from ...modeling_utils import PreTrainedModel
```

### TRF036

In modeling_*.py and modular_*.py, flags any `nn.Sequential(...)` construction. Sequential names its children by position, so weights land at `mlp.0.weight` and `mlp.2.weight`; the conversion mapping, _tied_weights_keys and every parallelism plan then have to reference indices, and inserting a layer renames everything after it. It also hides the forward, so the dtype casts and residuals between the steps are not visible where they happen.

```diff
-        self.mlp = nn.Sequential(
-            nn.Linear(config.hidden_size, config.intermediate_size),
-            nn.GELU(),
-            nn.Linear(config.intermediate_size, config.hidden_size),
-        )
+        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
+        self.act = ACT2FN[config.hidden_act]
+        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
```

### TRF037

In modeling_*.py and modular_*.py, flags calls to einsum and reports the equation string when it is a literal. Disabled by default: einsum is occasionally the clearest way to express a contraction, so this is opt-in rather than a hard convention. An einsum equation encodes the shapes in a notation the reader has to decode, and when the equation is built dynamically the reader cannot tell which contraction runs at all. Reviewers ask for einsums to be expanded on almost every model that introduces them, which is why it is worth having the check available even though it is not enforced.

```diff
-        pair_bias = torch.einsum("bqhc,bkhc->bhqk", query_states, key_states)
+        pair_bias = query_states.permute(0, 2, 1, 3) @ key_states.permute(0, 2, 3, 1)
```

### TRF038

For each modeling_*.py, processing_*.py, image_processing_*.py, video_processing_*.py and feature_extraction_*.py file, checks that a corresponding tests/models/<model>/test_*.py file exists (e.g. modeling_acme.py -> tests/models/acme/test_modeling_acme.py). configuration_*.py is exempt since config classes are conventionally covered by ConfigTester inside the companion test_modeling_*.py file. modular_*.py files are handled separately: since a single modular file can define modeling, processing, image/video-processor and config classes together, the classes it defines are classified by name suffix (XxxModel/XxxPreTrainedModel/XxxForCausalLM -> modeling, XxxImageProcessor(Fast) -> image processing, XxxProcessor -> processing, XxxVideoProcessor -> video processing, XxxFeatureExtractor -> feature extraction, XxxConfig -> skipped), and one violation is reported per missing test file, not just one per source file. There is no `# trf-ignore: TRF038` support for this rule: use `allowlist_models` if a model genuinely cannot ship a test yet, so the exemption is visible in review. A source file with no test file has no regression coverage: a broken forward pass, a bad conversion mapping, or a processor bug can land and stay broken indefinitely. Every model can be exercised with a dummy config and randomly initialized weights, so 'nothing to test' is not a valid reason to skip this.

```diff
src/transformers/models/acme/modeling_acme.py
+tests/models/acme/test_modeling_acme.py
```

### TRF039

Finds `if is_*_available(): import ...` blocks (including combinations like `is_vision_available() and is_torch_available()`) and checks whether the imported name is referenced anywhere else in the file, including inside string type hints and __all__. Flags the import if it is not. ruff's unused-import check does not clean these up, because the import is reachable and 'used' as far as static analysis of the block alone is concerned. When code is refactored to no longer need PIL.Image, torch, etc., the guarded import is easy to forget and lingers as dead weight and a misleading signal about the file's real dependencies.

```diff
if is_vision_available():
-    from PIL import Image
```

### TRF040

In modeling_*.py and modular_*.py, checks methods decorated with both @capture_outputs and @can_return_tuple. Complements TRF003, which covers manual return_dict branching in forward(). @capture_outputs and @can_return_tuple both pop return_dict, thus only the outermost decorator sees the true value. @capture_outputs already handles to_tuple conversion which makes @can_return_tuple redundant.

```diff
-@can_return_tuple
 @merge_with_config_defaults
 @capture_outputs
 @auto_docstring
 def forward(self, x):
     return AcmeModelOutput(last_hidden_state=x)
```

### TRF041

In modeling_*.py and modular_*.py, flags every `if`/`elif` statement and every conditional expression whose condition reads a `config.*` or `self.config.*` attribute and that does not carry a `# CODEPATH:` comment. The comment is accepted on the branch line itself or anywhere in the contiguous comment block directly above it, so it can head a multi-line explanation. Deliberately broad: any config attribute in the condition counts, not only boolean feature flags, because a branch on a numeric or optional config field forks the graph exactly as much as a branch on a flag does. One shape is exempt outright, by structure rather than by name: `X if X is not None else fallback`, where the field under test is itself one of the two results. That yields the field when set and a default when not, which is `getattr(config, x, default)` spelled long, so it cannot fork the graph and there is no path to name. Merely mentioning None does not qualify — `config.vision_config is not None` selects a whole extra tower and still owes a note. A field that gates no checkpoint divergence at all, such as `problem_type` selecting a loss or `hidden_act` looking up an activation, can be exempted for a whole file with a module-level `# trf-ignore: TRF041 config.problem_type, config.hidden_act` directive at column 0, naming the fields comma- or space-separated; `self.config.x`, `config.x` and `x` all name the same field. The directive has to name at least one field, so a bare `# trf-ignore: TRF041` keeps its per-line meaning instead of muting the file. Exemption is per field, not per branch: a condition reading several config fields is skipped only when every one of them is exempt, so `if config.problem_type and config.use_cache` still has to explain itself when only `problem_type` is named. Every config-gated branch is a second architecture living in the same file, and the reader cannot tell from the code whether both halves are reachable. That is why reviewers ask "is this ever used?", "are they all needed?" and "why are there so many cases?" on almost every new model, and why dead experimental branches survive for releases. This rule does not forbid the branch; it borrows Rust's `// SAFETY:` discipline and makes the author write down the justification next to it. A branch nobody can name a checkpoint for is a branch to delete, and the note makes that obvious at review time instead of three rounds later.

```diff
+        # CODEPATH: ESMC-6B ships pre-normalised embeddings, the 300M/600M checkpoints do not.
         if config.use_embedding_norm:
             hidden_states = self.embedding_norm(hidden_states)

-        if config.msa_encoder_enabled:
-            hidden_states = self.msa_encoder(hidden_states)
+        # no released checkpoint sets msa_encoder_enabled -> branch removed
```

### TRF042

In tests/models/*/test_tokenization_*.py, checks that the file defines a test class inheriting TokenizerTesterMixin. Only classes the test runner collects are considered — a `TestCase` base, or the `*Test` naming convention when the base is another model's test class — so files whose only classes are helpers are skipped, and a helper mixing in the suite does not satisfy the rule on a real test class's behalf. The violation is reported on the first test class that does not run the suite. Inheritance is followed through base classes defined in the same file and into another model's tokenizer test imported by name, so a class deriving from one that already carries the mixin is satisfied; a base the tests tree cannot resolve never counts as carrying it. `auto` is allowlisted because test_tokenization_auto.py tests AutoTokenizer resolution rather than one model's tokenizer. Ships with the file discovery widened to tests/models/**/test_tokenization_*.py, which affects which files the linter walks for every rule -- existing rules all gate on the file-name prefix, so none of them see these files. TokenizerTesterMixin is where encode/decode round-tripping, padding and truncation, special-token handling, added-token persistence, and save/load equivalence are actually checked. A tokenizer test that only asserts a couple of hand-written token id lists passes while the tokenizer is broken in every one of those dimensions, and the gap is invisible in review because the file looks like it has tests. Reviewers ask for the mixin by name on new tokenizers; five of the six tokenizer tests missing it predate 2026.

```diff
-class AcmeTokenizationTest(unittest.TestCase):
+class AcmeTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
+    tokenizer_class = AcmeTokenizer
+    test_slow_tokenizer = True
```

### TRF043

Checks forward signatures of classes whose name ends in Attention for a declared position_ids parameter. position_ids is consumed downstream by flash-attention padding-free training and must flow through **kwargs. An attention class that names it in the signature swallows it before the attention interface can read it; the llama standard passes position_embeddings plus **kwargs.

```diff
class AcmeAttention(nn.Module):
     def forward(
         self,
         hidden_states,
         position_embeddings,
         attention_mask=None,
-        position_ids=None,
         **kwargs: Unpack[TransformersKwargs],
     ):
```

### TRF044

Checks every function in modeling_*.py and modular_*.py for a parameter named cache_position. cache_position was removed from all models in v5. Code that reintroduces it (usually copied from pre-v5 sources) threads a dead argument through every layer; the cache update call is past_key_values.update(key_states, value_states, self.layer_idx) with no position threading.

```diff
def forward(
     self,
     hidden_states,
     past_key_values=None,
-    cache_position=None,
     **kwargs,
 ):
-    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_position)
+    key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
```

### TRF045

Checks forward signatures in modeling_*.py and modular_*.py for the legacy output_attentions, output_hidden_states, and return_dict parameters. Models contributed before the cutoff date are exempt. The decorator stack owns output control: @capture_outputs resolves output_* flags against the config and records the tensors via _can_record_outputs, and @can_return_tuple handles return_dict. Declaring them in the signature reintroduces manual flag threading that drifts from the decorator behavior.

```diff
+@capture_outputs
 def forward(
     self,
     input_ids,
-    output_attentions=None,
-    output_hidden_states=None,
-    return_dict=None,
     **kwargs: Unpack[TransformersKwargs],
 ):
```

### TRF046

Checks forward methods in modeling_*.py and modular_*.py for assignments to self attributes. Hidden state written during forward breaks batching, torch.compile, and reasoning about the module. Carried state is passed explicitly (cache objects, the generate loop); values that depend only on config or static shapes belong in __init__.

```diff
def forward(self, hidden_states):
-    self.sequence_length = hidden_states.shape[1]
-    embeddings = self.compute_embeddings(self.sequence_length)
+    embeddings = self.compute_embeddings(hidden_states.shape[1])
```

### TRF047

Checks preprocess, _preprocess, __call__, and post_process* methods in image_processing_*.py and video_processing_*.py for assignments to self attributes. A processor that carries state between calls breaks preprocess-many-then-postprocess batching: the second preprocess overwrites the state the first postprocess needs. Return the value or pass it through the method chain.

```diff
def _preprocess(self, images, **kwargs):
-    self.original_sizes = [image.shape[-2:] for image in images]
+    original_sizes = [image.shape[-2:] for image in images]
     ...
+    return BatchFeature(data={"pixel_values": pixel_values, "original_sizes": original_sizes})
```

### TRF048

Checks class-level _tied_weights_keys declarations for list/tuple/set literals. v5 changed _tied_weights_keys to a dict mapping the tied target parameter to its source. The list form no longer tells the loading code which parameter is the source, so tying, device_map computation, and saving misbehave silently.

```diff
class AcmeForCausalLM(AcmePreTrainedModel):
-    _tied_weights_keys = ["lm_head.weight"]
+    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
```

### TRF049

Checks __init__ methods in modeling_*.py and modular_*.py for init calls: nn.init.* / init.* primitives and in-place initializers on own parameters (self.weight.data.normal_()). Models instantiate on the meta device, so tensor values written in __init__ are discarded before loading; a parameter initialized only there has random content when fine-tuning from scratch or after a meta-device reload. Allocate with torch.empty in __init__ and initialize in _init_weights.

```diff
class AcmeEmbeddings(nn.Module):
     def __init__(self, config):
         super().__init__()
         self.position_embedding = nn.Parameter(torch.empty(config.num_positions, config.hidden_size))
-        nn.init.trunc_normal_(self.position_embedding, std=config.initializer_range)

 class AcmePreTrainedModel(PreTrainedModel):
     def _init_weights(self, module):
         super()._init_weights(module)
+        if isinstance(module, AcmeEmbeddings):
+            init.trunc_normal_(module.position_embedding, std=self.config.initializer_range)
```

### TRF050

Checks __init__ methods of classes whose name ends in Attention for calls to a *RotaryEmbedding class. The Model owns a single rotary_emb, builds inv_freq once, and passes cos/sin down as position_embeddings. A rotary module per attention layer duplicates buffers, recomputes frequencies per layer, and diverges from the interface contract that attention receives position_embeddings.

```diff
class AcmeAttention(nn.Module):
     def __init__(self, config, layer_idx):
         super().__init__()
-        self.rotary_emb = AcmeRotaryEmbedding(config)

 class AcmeModel(AcmePreTrainedModel):
     def __init__(self, config):
         super().__init__(config)
+        self.rotary_emb = AcmeRotaryEmbedding(config)
```

### TRF051

Checks modeling_*.py and modular_*.py for comparisons against a _attn_implementation attribute. Backend dispatch belongs to ALL_ATTENTION_FUNCTIONS.get_interface, and backend-conditional tensor munging (padding, reshaping) belongs in the shared wrappers under integrations/. Inline branching keeps the model body kernel-aware and breaks when new backends register.

```diff
-if self.config._attn_implementation == "flash_attention_2":
-    attn_output = flash_path(query_states, key_states, value_states)
-else:
-    attn_output = eager_path(query_states, key_states, value_states)
+attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
+attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, ...)
```

### TRF052

Checks modeling_*.py and modular_*.py for module-level assignments to names ending in _ATTENTION_CLASSES. Per-backend attention classes selected from a dict are the pre-interface idiom: three near-identical classes drift apart, and hub attention kernels registered into ALL_ATTENTION_FUNCTIONS never reach them. One attention class dispatching through the interface replaces the dict; do not propagate it from a legacy parent.

```diff
-ACME_ATTENTION_CLASSES = {
-    "eager": AcmeAttention,
-    "flash_attention_2": AcmeFlashAttention2,
-    "sdpa": AcmeSdpaAttention,
-}
+class AcmeAttention(nn.Module):
+    def forward(self, ...):
+        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
```

### TRF053

Checks modeling_*.py and modular_*.py for assignments that build shift_logits/shift_labels (and shifted_ variants) by slicing, as in labels[..., 1:]. Receiving already-shifted labels (shift_labels = kwargs.pop("shift_labels", labels)) is the correct idiom and is not flagged. self.loss_function shifts labels itself, so modeling code that pre-shifts trains on doubly-shifted targets or forces a bespoke loss path. Decoder-only models pass the raw labels and let the loss shift them. Encoder-decoder models are the mirror case: their labels are already shifted because the decoder input gets the decoder start token prepended, so they must pass shift_labels=labels to stop the loss from shifting again. Double-shift is the recurring training-loss bug (Git/Florence2/Moonshine family).

```diff
if labels is not None:
-    shift_logits = logits[..., :-1, :].contiguous()
-    shift_labels = labels[..., 1:].contiguous()
-    loss = nn.functional.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
+    # decoder-only: labels are unshifted, the loss shifts them
+    loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)
+    # encoder-decoder: labels are already shifted, hand them over as shift_labels
+    loss = self.loss_function(logits=logits, labels=labels, shift_labels=labels, vocab_size=self.config.vocab_size)
```

### TRF055

Checks that PreTrainedModel subclasses in modeling_*.py and modular_*.py do not assign `config = SomeConfig` as a class attribute. `PreTrainedModel.__init_subclass__` derives `config_class` by looking for a `config` **annotation** via `inspect.get_annotations(cls)`. An assignment (`config = SomeConfig`) is invisible to this mechanism: it creates a stray class attribute but `inspect.get_annotations` returns `None` for it, so the subclass falls back to inheriting the parent's `config_class` instead of picking up the intended config. A pure annotation (`config: SomeConfig`) has no runtime value and does not create an attribute, so it is correctly detected by `inspect.get_annotations` and sets `config_class` to the right class.

```diff
class Gemma4VisionModel(Gemma4PreTrainedModel):
     """The Gemma 4 Vision Encoder."""
-    config = Gemma4VisionConfig
+    config: Gemma4VisionConfig
```

### TRF056

In modeling_*.py and modular_*.py, flags `.item()` and `.tolist()` calls inside any `forward`. A `.tolist()` whose result is the split-size argument of `split(...)`, is exempt as torch.split needs Python ints. Both calls read a tensor back to the host, so dynamo cannot trace them resulting in a graph break.

```diff
-        for grid, item in zip(grid_thw.tolist(), split_items):
-            _, height, width = grid
-            merged.append(self.patch_merger(item, size=(height, width)))
+        for grid, item in zip(grid_thw, split_items):
+            merged.append(self.patch_merger(item, size=(grid[1], grid[2])))
```

### TRF057

Checks `@auto_docstring` on the classes that need it: public `PreTrainedModel` subclasses (`<Model>PreTrainedModel`, `<Model>Model`, `<Model>For<Task>`, backbones), `PreTrainedConfig` subclasses, `ModelOutput` subclasses, image processors and `ProcessorMixin` subclasses, and on their public methods: `forward`, `get_image_features`, `get_video_features`, `get_audio_features`, `get_text_features`, `preprocess` and `__call__`. A class or method in a `modular_*.py` file is checked against the files generated from it. Without the decorator, a class ships with no intro and no parameter documentation, and a method with no argument documentation, no `Returns` section and no usage example, so the standard descriptions in `auto_docstring.py` have to be hand-written per model instead.

```diff
+@auto_docstring
 @dataclass
 class AcmeModelOutputWithPast(ModelOutput):
     logits: torch.FloatTensor | None = None

+@auto_docstring
 class AcmeForConditionalGeneration(AcmePreTrainedModel):
+    @auto_docstring
     def forward(self, input_ids, pixel_values=None, **kwargs):
         ...
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

For models with legacy code that can't be fixed immediately, add the model's directory name to the relevant rule's `allowlist_models` list in the [mlinter rules.toml](https://github.com/huggingface/transformers-mlinter/blob/main/mlinter/rules.toml).

```toml
[rules.TRF004]
allowlist_models = ["existing_model", "your_model_name"]
```
