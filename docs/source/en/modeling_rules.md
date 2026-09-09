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

Checks that <Model>PreTrainedModel's config_class is named <Model>Config. A mismatch can break loading, auto classes and developer expectations.

```diff
class AcmePreTrainedModel(PreTrainedModel):
-    config_class = WileConfig
+    config_class = AcmeConfig
```

### TRF002

Checks base_model_prefix is a non-empty, whitespace-free string literal. Invalid prefixes can break weight-loading key mapping and base-model access.

```diff
class AcmePreTrainedModel(PreTrainedModel):
-    base_model_prefix = ""
+    base_model_prefix = "model"
```

### TRF003

Flags the old `if not return_dict: return (x,)` pattern in forward. Manual return_dict branching is verbose and easy to get wrong. Let @capture_output or @can_return_tuple do it.

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

Checks that no model class defines a tie_weights method. Overriding tie_weights breaks loading, device_map computation and saving. Declare tied weights in the _tied_weights_keys class attribute.

```diff
-def tie_weights(self):
-    self.lm_head.weight = self.emb.weight
+class AcmeForCausalLM(AcmePreTrainedModel):
+    _tied_weights_keys = ["lm_head.weight"]
```

### TRF005

Checks the shape of _no_split_modules when present. Malformed values can break device-map partitioning and sharding.

```diff
-_no_split_modules = [SomeLayerClass, ""]
+_no_split_modules = ["AcmeDecoderLayer", "AcmeAttention"]
```

### TRF006

Checks that cache arguments in a forward signature are used in the body. An unused cache argument suggests incomplete caching support and inconsistent API behavior.

```diff
def forward(self, x, past_key_values=None, use_cache=False):
+    if use_cache:
+        ...
     return x
```

### TRF007

Checks for self attribute assignments after self.post_init() in __init__. Mutating model structure after post_init bypasses its initialization and finalization work.

```diff
def __init__(self, config):
     ...
-    self.post_init()
-    self.proj = nn.Linear(...)
+    self.proj = nn.Linear(...)
+    self.post_init()
```

### TRF008

Checks that add_start_docstrings on a model class gets a non-empty argument. An empty argument produces unclear, low-quality generated API docs.

```diff
-@add_start_docstrings("")
+@add_start_docstrings("The Acme model.")
 class AcmeModel(AcmePreTrainedModel):
     ...
```

### TRF009

Flags imports into another model's package from a model's shipped implementation files: modeling_*, configuration_*, processing_*, image_processing_*, video_processing_*, feature_extraction_*, tokenization_* and generation_*.py. Three forms count: package path (`from transformers.models.other.modeling_other import X`), relative path (`from ..other.configuration_other import X`) and public API (`from transformers import OtherModel`) -- for the last the owning directory is inferred from the class name and confirmed against the classes it defines, so an unresolvable name is left alone. Out of scope: `modular_*.py` (building on another model is its purpose, and the converter flattens those imports away), `convert_*.py`, `__init__.py` and files under `auto`. Allowed targets: the model's own directory, `auto` and `timm_wrapper`. One model, one definition: model A's behavior lives in its own files, and a change to model B must not silently change model A -- for every file kind, not just modeling. Reach sub-configs through AutoConfig and CONFIG_MAPPING, and use a modular file to build on another model's code.

```diff
-from transformers.models.llama.modeling_llama import LlamaAttention
-from transformers import CLIPTextModelWithProjection
+# Keep implementation local to this model's own files.
+# To build on another model, write modular_acme.py; to reuse a snippet,
+# copy it with a # Copied from comment.
```

### TRF010

Checks that direct PreTrainedConfig/PretrainedConfig subclasses in configuration_*.py and modular_*.py carry @strict(accept_kwargs=True). Without it a new config misses the repo's runtime type-validation contract and drifts from the dataclass-based config standard.

```diff
+@strict(accept_kwargs=True)
 class AcmeConfig(PreTrainedConfig):
     ...
```

### TRF011

In forward() of PreTrainedModel subclasses, flags submodule attribute accesses torch.nn.Identity would not have: on loop variables over self.layers, and self.<submodule>.<attr> where <attr> is not a standard nn.Module attribute. Pipeline parallelism may replace any submodule with torch.nn.Identity, so reading a custom attribute (e.g. decoder_layer.attention_type) off it raises AttributeError at runtime. Read per-layer metadata from self.config.

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

Flags in-place ops (.normal_(), .zero_(), ...) on module weights inside _init_weights. Parameters carry internal flags tracking whether they still need re-initialization; in-place ops bypass them. Use the `init` primitives.

```diff
+from transformers import initialization as init
+
 def _init_weights(self, module):
-    module.weight.normal_(mean=0.0, std=0.02)
+    init.normal_(module.weight, mean=0.0, std=0.02)
```

### TRF013

Checks that every PreTrainedModel subclass defining __init__ calls self.post_init(). In modular files super().__init__() counts, since it propagates the parent's post_init. post_init does essential finalization (weight init, gradient checkpointing setup, ...); skipping it causes subtle runtime bugs.

```diff
class AcmeModel(AcmePreTrainedModel):
     def __init__(self, config):
         super().__init__(config)
         self.layers = nn.ModuleList(...)
+        self.post_init()
```

### TRF014

Flags `trust_remote_code` used or passed (e.g. as a kwarg) in native model integration files. `trust_remote_code` loads arbitrary code, including binaries -- a power feature for users, not something a native integration may depend on, since remote code cannot be reviewed or maintained in transformers.

```diff
class AcmeModel(AcmePreTrainedModel):
     def __init__(self, config):
         super().__init__(config)
-        self.model = AutoModel.from_pretrained(..., trust_remote_code=True)
+        self.model = AutoModel.from_pretrained(...)
```

### TRF015

When a PreTrainedModel subclass sets a non-empty _tied_weights_keys, checks the companion configuration file for a tie_word_embeddings field. Without it users cannot control weight tying: the model ties unconditionally, breaking serialization round-trips and fine-tuning with untied heads.

```diff
# configuration_foo.py
 @strict(accept_kwargs=True)
 class FooConfig(PreTrainedConfig):
     hidden_size: int = 768
+    tie_word_embeddings: bool = True
```

### TRF016

When an image_processing_*.py or video_processing_*.py class declares boolean do_* attributes (do_resize, do_rescale, do_normalize, do_convert_rgb, ...) and overrides preprocess() or _preprocess(), checks each flag is still consumed there: referenced directly, delegated via super().preprocess/_preprocess(..., **kwargs), or -- image processors only -- forwarded through _preprocess_image_like_inputs/_prepare_image_like_inputs. do_sample_frames is exempt: the base preprocess() consumes it before _preprocess() runs. A do_X the override never references is dead: setting do_X=False has no effect, the operation runs anyway, and per-call overrides silently break.

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

On classes carrying both @auto_docstring and @dataclass, checks @auto_docstring comes first. Decorators apply bottom-up, so @dataclass on top runs @auto_docstring first, on a class with no synthesized __init__ yet: it then modifies the parent's __init__.__doc__ instead of the subclass's.

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

Checks that every PreTrainedModel subclass overriding `_init_weights(self, module, ...)` chains up via `super()._init_weights(...)`. Modular files also accept the sentinels `PreTrainedModel._init_weights(self, module)`, `PreTrainedModel._init_weights(module)` and `raise AttributeError(...)`. For a deliberate full override, suppress with `# trf-ignore: TRF018` above the method. The base `_init_weights` covers the standard module types (Linear, Embedding, LayerNorm, RotaryEmbedding, ...). Skipping it leaves every submodule the override misses uninitialized -- which passes tests and surfaces much later as a subtle weight-init bug (cf. https://github.com/huggingface/transformers/pull/45597).

```diff
from ... import initialization as init

 def _init_weights(self, module):
+    super()._init_weights(module)
     if isinstance(module, AcmeCustomLayer):
-        module.gate.data.zero_()
+        init.zeros_(module.gate)
```

### TRF019

Flags a non-empty `_defaults` on `*ProcessorKwargs` TypedDicts in `processing_*.py`. Models released before the cutoff date are grandfathered. Hardcoded `_defaults` scatter processor configuration across Python source, are awkward to override from config, and bloat the code. In `processor_config.json` on the hub they travel with the checkpoint and can be updated without a code change.

```diff
class Gemma4ProcessorKwargs(ProcessingKwargs, total=False):
-    _defaults = {
-        "text_kwargs": {"padding": False},
-        "images_kwargs": {"return_tensors": "pt"},
-    }
     images_kwargs: Gemma4ImageProcessorKwargs
```

### TRF020

In model directories whose configuration declares `kv_lora_rank` (Multi-head Latent Attention), checks the attention class owning the KV LoRA expansion projection (`kv_b_proj`, or any `nn.Linear(config.kv_lora_rank, ...)`): the expansion must live in a dedicated method (e.g. `expand_kv`) that `forward()` calls, not inline. In modular files a method inherited from an imported base counts. External backends (vLLM/SGLang) override the expansion to consume the compressed KV cache directly. Inlined in `forward()` there is nothing to override, so the backend must materialize the full cache -- losing the memory savings MLA exists for.

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

In modeling_*.py and modular_*.py, flags `torch.tensor(<value>, ..., device=<non-cpu>)` where `<value>` provably resolves to a Python scalar -- from numeric literals and arithmetic on them, torch.finfo/iinfo fields, scalar-returning builtins and math.* calls, locals bound exactly once, self.<attr> assigned in the class body, and config fields annotated int/float/bool in the companion configuration file (following attribute_map). Anything that may also be a sequence (`eos_token_id: int | list[int] | None`) or that cannot be resolved is left alone. __init__, _init_weights, __post_init__ and post_init are exempt: they never run inside a capture region. torch.tensor(<python scalar>, device=<accelerator>) materialises the value on the host then copies it to the device; CUDA graph capture forbids that copy, so the model cannot be captured. torch.full((), <value>, dtype=..., device=...) fills the same 0-d tensor on-device with a capturable kernel and no synchronisation.

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

Checks that every string in a `_no_split_modules` list in modeling_*.py or modular_*.py names a class defined in that file, imported into it, or defined by a sibling module of the same model directory. Complements TRF005, which only checks the value's shape. `device_map` matches these strings against `module.__class__.__name__` at runtime, so a stale or misspelled name matches nothing and is silently ignored -- the module it should keep together can still be split across devices. Delete entries naming another model's classes rather than correcting them: `post_init` already collects `_no_split_modules` from child submodels.

```diff
class VideoLlavaPreTrainedModel(PreTrainedModel):
-    _no_split_modules = ["VideoLlavaVisionAttention"]
```

### TRF023

In configuration_*.py and modular_*.py, flags `*Config` fields named after an upstream paper's abbreviation instead of the canonical name: d_model/n_embd -> hidden_size, d_ff/d_inner/ffn_dim/ffn_hidden_size/expansion_ratio -> intermediate_size, d_head -> head_dim, n_head/n_heads -> num_attention_heads, n_layer/n_layers/num_blocks -> num_hidden_layers. Fields are read from the class body and from __init__/__post_init__ assignments and defaults. Ambiguous but idiomatic names (num_heads, num_layers, embed_dim, mlp_ratio) are not flagged, and models added before cutoff_date keep theirs. Everything that reads a model's shape -- device_map planning, tensor/pipeline parallel plans, quantization, PEFT, attention-backend selection, `attribute_map` consumers -- looks up the canonical names, so a config spelling the same quantity `d_model` silently opts out of all of them. Map the checkpoint's own spelling in the conversion script.

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

In modeling_*.py and modular_*.py, flags an integer literal greater than 8 in a dimension argument of a torch.nn constructor (Linear, Embedding, LayerNorm, RMSNorm, GroupNorm, BatchNorm*, InstanceNorm*, Conv*d, ConvTranspose*d, Bilinear, MultiheadAttention), positional or by keyword (in_features, out_features, in_channels, out_channels, num_embeddings, embedding_dim, embed_dim, normalized_shape, num_channels, hidden_size). Operator-shape arguments (kernel_size, stride, padding, num_groups) are ignored; literals up to 8 keep scalar heads, binary classifiers and RGB channel counts clean. Models added before cutoff_date are exempt. A hardcoded width pins the module to one checkpoint size: the same architecture at another scale loads with a shape mismatch, and `from_pretrained` cannot say which value is wrong because no config field points at it. It also splits the source of truth, so editing the config no longer changes the model that gets built.

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

In modeling_*.py and modular_*.py, flags calls to a mask factory (the masking_utils entry points create_causal_mask, create_bidirectional_mask, create_sliding_window_causal_mask, create_chunked_causal_mask, create_masks_for_generate, and any create_*_mask helper) inside a class that does not inherit from PreTrainedModel -- plain nn.Module blocks such as layers, attention modules and encoders. Mask construction is O(sequence length squared) work that does not vary per layer, so building it in the layer pays that cost once per layer, and each layer then owns its own mask, so the attention backends can no longer be handed a single prepared one. Build it once in the model and pass it down.

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

In modeling_*.py and modular_*.py, flags a non-PreTrainedModel class that defines only __init__ and forward, assigns exactly one self.<attr> in __init__, and whose forward body is exactly `return self.<attr>(...)` for it (a leading docstring is ignored). Any other method, extra attribute, statement before the return, or `super()` call in forward means the class does work of its own. The wrapper adds a level to every weight name, to _no_split_modules, parallelism plans and every conversion mapping, while computing nothing -- and readers have to open one more class to find that out. PreTrainedModel subclasses are exempt: they exist for from_pretrained and the auto classes even when forward only delegates.

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

Flags any `assert` statement in modeling_*.py, modular_*.py and configuration_*.py. `python -O` strips asserts, so a shape or config check written as one silently disappears in optimised runs. An assert also gives a bare AssertionError, where a ValueError can name the offending value and say what to do about it.

```diff
def forward(self, hidden_states):
-    assert hidden_states.dim() == 3
+    if hidden_states.dim() != 3:
+        raise ValueError(f"Expected a 3D tensor, got shape {tuple(hidden_states.shape)}.")
```

### TRF028

Checks the first 25 lines of modeling_*.py, modular_*.py, configuration_*.py, processing_*.py, image_processing_*.py and video_processing_*.py for a `Licensed under the <name> License` line followed by every clause of the standard warranty paragraph, from `You may obtain a copy of the License at` through `limitations under the License.`. Lines are flattened and lowercased first, so wrapping and comment style do not matter. The license name and copyright line are not checked: they vary per model. A file shipped without the header leaves its provenance ambiguous, and adding it later means touching an already-released file. Matching only `Apache License` would let through the defects that actually occur: a paragraph truncated mid-way, a header that stops after the URL, or one mangled by a bad search-and-replace.

```diff
+# Copyright 2026 The HuggingFace Team. All rights reserved.
+#
+# Licensed under the Apache License, Version 2.0 (the "License");
+# ...
 """PyTorch Acme model."""
```

### TRF029

In modeling_*.py and modular_*.py, flags an `__init__` taking `config` alongside an argument whose name is unambiguously a config field (hidden_size, num_attention_heads, intermediate_size, head_dim, num_hidden_layers, embed_dim, dropout, eps, patch_size, rope_theta, ...). A parameter optional with a `None` default is exempt: that is an override, not a second source of truth, and it is how one MLP class serves both the dense and the expert width of a MoE model. A hardcoded default such as `hidden_size: int = 1024` is not -- it wins over the config whenever the caller passes nothing. kosmos2 is allowlisted: its doc page is not derivable from the directory name, so the cutoff cannot grandfather it. The same number now has two sources of truth and the caller picks the winner, so editing the config no longer changes the model that gets built. It also pushes architecture knowledge out to every call site, where it does not belong.

```diff
class AcmeAttention(nn.Module):
-    def __init__(self, config, embed_dim, num_heads, dropout):
+    def __init__(self, config, layer_idx=None):
         super().__init__()
-        self.embed_dim = embed_dim
-        self.num_heads = num_heads
+        self.embed_dim = config.hidden_size
+        self.num_heads = config.num_attention_heads

 class AcmeMLP(nn.Module):
     # an optional override is fine: omitting it reads the config
     def __init__(self, config, intermediate_size=None):
         super().__init__()
         self.intermediate_size = intermediate_size or config.intermediate_size
```

### TRF030

In modeling_*.py and modular_*.py, flags attribute chains rooted at `config` or `self.config` three or more levels deep. `config.hidden_size` (one hop) and `config.text_config.hidden_size` (two hops, the normal sub-config access) are fine. One violation per line. A module that walks `config.diffusion_config.atom_encoder_config.hidden_size` is coupled to the whole config hierarchy rather than its own slice, so it cannot be reused, tested or given a different sub-config. Pass the relevant sub-config down and the chain collapses to one hop.

```diff
class AcmeAtomEncoder(nn.Module):
     def __init__(self, config):
         super().__init__()
-        self.norm = AcmeLayerNorm(config.diffusion_config.atom_encoder_config.hidden_size)
+        self.norm = AcmeLayerNorm(config.hidden_size)
```

### TRF031

In modeling_*.py and modular_*.py, flags a top-level `@dataclass` whose bases include nothing ending in `Output`, unless it has two or more mandatory fields -- those are internal argument bundles, which `ModelOutput` rejects at runtime. A plain output dataclass does not index like a tuple, does not survive `return_dict=False`, and is invisible to @auto_docstring, so its fields never reach the generated API docs. ModelOutput gets all three for free.

```diff
@auto_docstring
 @dataclass
-class AcmeStructureOutput:
+class AcmeStructureOutput(ModelOutput):
     positions: torch.Tensor
     confidence: Optional[torch.Tensor] = None
```

### TRF032

In modeling_*.py and modular_*.py, flags masked_fill, masked_fill_, full, full_like and new_full called with a negated numeric literal of magnitude 1e3 or more. A hardcoded -1e9 overflows to -inf in float16 and is nowhere near the smallest value in float32, so the same mask behaves differently per dtype and can produce NaNs after softmax. `torch.finfo(dtype).min` is the smallest representable value in whatever dtype is actually running.

```diff
-attention_scores = attention_scores.masked_fill(~mask, -1e9)
+attention_scores = attention_scores.masked_fill(~mask, torch.finfo(attention_scores.dtype).min)
```

### TRF033

In modeling_*.py and modular_*.py, flags methods whose name starts with `set_`, except the PreTrainedModel contract methods set_input_embeddings, set_output_embeddings, set_decoder, set_encoder, set_attn_implementation and set_default_language. A setter makes behaviour depend on call order: the value is not in the config, so it is not saved, not restored by from_pretrained, and not visible to device-map or parallelism planning. Users have to know to call it, and forgetting is silent.

```diff
class AcmeTriangleAttention(nn.Module):
-    def set_chunk_size(self, chunk_size):
-        self.chunk_size = chunk_size
+    def __init__(self, config):
+        super().__init__()
+        self.chunk_size = config.chunk_size
```

### TRF034

In modeling_*.py and modular_*.py, flags a locally-defined `*Layer`/`*Block` class instantiated in an `nn.ModuleList(...)` that does not reach `GradientCheckpointingLayer` through its base chain; modular files follow relative imports into sibling models, and unresolved chains are inconclusive. Out of scope: a model that never sets `supports_gradient_checkpointing = True`, which raises from `gradient_checkpointing_enable()` instead of skipping a layer; a layer holding `nn.BatchNorm*`/`nn.InstanceNorm*`, whose statistics would be recomputed twice; and a stack that is not the model's token mixer, shown by an attention, modulation, mixer or SSM module assigned as `self.x = Y(...)`. `gradient_checkpointing_enable()` wraps a layer only if it is a GradientCheckpointingLayer. A plain nn.Module on the trunk is skipped silently, so training looks checkpointed while still allocating full activations, and the OOM surfaces far from the cause. Elsewhere -- a conv backbone, a decode head -- the trade is the author's call, so the rule stays out.

```diff
-class AcmeDecoderLayer(nn.Module):
+class AcmeDecoderLayer(GradientCheckpointingLayer):
     def __init__(self, config, layer_idx):
         super().__init__()
```

### TRF035

Flags `# noqa` comments, with or without codes, in modeling_*.py, modular_*.py and configuration_*.py. In a modular file `F401`, `F821` and `F822` are accepted: it is a generation source that deliberately does not define every name it uses, so ruff's undefined-name family fires on correct code -- `__all__` entries the converter fills in, classes living in the parent model, imports kept to be re-exported. A `# noqa` naming only those codes is skipped; one naming anything else is reported on the codes that are left. A bare `# noqa` is always reported; in a modular file the message asks for the code rather than a rewrite. Model files are ordinary code held to the repo's lint rules, so a suppression means the underlying issue was left in place -- and a bare `# noqa` also hides every future violation on that line. Fix the code instead.

```diff
-from ...modeling_utils import PreTrainedModel  # noqa: F401
+from ...modeling_utils import PreTrainedModel
```

### TRF036

In modeling_*.py and modular_*.py, flags any `nn.Sequential(...)` construction. Sequential names its children by position, so weights land at `mlp.0.weight` and `mlp.2.weight`: the conversion mapping, _tied_weights_keys and every parallelism plan then reference indices, and inserting a layer renames everything after it. It also hides the forward, so the dtype casts and residuals between steps are invisible where they happen.

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

In modeling_*.py and modular_*.py, flags calls to einsum, reporting the equation when it is a literal. Disabled by default: einsum is occasionally the clearest way to express a contraction, so this is opt-in rather than a hard convention. An einsum equation encodes the shapes in a notation the reader has to decode, and a dynamically built equation hides which contraction runs at all. Expand it into explicit matmul/transpose operations.

```diff
-        pair_bias = torch.einsum("bqhc,bkhc->bhqk", query_states, key_states)
+        pair_bias = query_states.permute(0, 2, 1, 3) @ key_states.permute(0, 2, 3, 1)
```

### TRF038

Checks that every modeling_*.py, processing_*.py, image_processing_*.py, video_processing_*.py, feature_extraction_*.py and tokenization_*.py file has a matching tests/models/<model>/test_*.py (modeling_acme.py -> tests/models/acme/test_modeling_acme.py). Exempt: configuration_*.py, covered by ConfigTester in the companion test_modeling_*.py, and tokenization_utils*.py, a helper covered through its tokenizer; tokenization_<name>_fast.py maps to its slow counterpart's test file. A modular_*.py can define several families at once, so its classes are routed by name suffix -- XxxModel/XxxPreTrainedModel/XxxFor<Task> modeling, XxxImageProcessor(Fast) image processing, XxxProcessor processing, XxxVideoProcessor video processing, XxxFeatureExtractor feature extraction, XxxTokenizer(Fast) tokenization, XxxConfig skipped -- and one violation is reported per missing test file. `# trf-ignore: TRF038` is not supported; use `allowlist_models` so an exemption is visible in review. A source file with no test file has no regression coverage: a broken forward pass, a bad conversion mapping, or a tokenizer that drops a special token can land and stay broken indefinitely. A dummy config with random weights, or a small hand-written vocabulary, is enough to exercise any of them.

```diff
src/transformers/models/acme/modeling_acme.py
+tests/models/acme/test_modeling_acme.py
 src/transformers/models/acme/tokenization_acme.py
+tests/models/acme/test_tokenization_acme.py
```

### TRF039

Finds `if is_*_available(): import ...` blocks (including combinations such as `is_vision_available() and is_torch_available()`) and flags the import when the name is referenced nowhere else in the file, including inside string type hints and __all__. ruff's unused-import check does not clean these up: the import is reachable, so the block looks fine on its own. Once a refactor no longer needs PIL.Image, torch, etc., the guarded import lingers as dead weight and a misleading signal about the file's real dependencies.

```diff
if is_vision_available():
-    from PIL import Image
```

### TRF040

In modeling_*.py and modular_*.py, flags methods decorated with both @capture_outputs and @can_return_tuple. Complements TRF003, which covers manual return_dict branching in forward(). Both decorators pop return_dict, so only the outermost one sees the true value. @capture_outputs already handles the to_tuple conversion, making @can_return_tuple redundant.

```diff
-@can_return_tuple
 @merge_with_config_defaults
 @capture_outputs
 @auto_docstring
 def forward(self, x):
     return AcmeModelOutput(last_hidden_state=x)
```

### TRF041

In modeling_*.py and modular_*.py, flags every `if`/`elif` and conditional expression whose condition reads a `config.*` or `self.config.*` attribute without a `# CODEPATH:` comment, accepted on the branch line or in the contiguous comment block directly above it. Any config attribute counts, not just boolean flags. Exempt by structure: `X if X is not None else fallback`, where the tested field is itself one of the results -- `getattr(config, x, default)` spelled long, which cannot fork the graph; and a guard, an `if` with no `else` whose body only raises or only warns/logs, since one side aborts and nothing diverges past it. Merely mentioning None does not qualify: `config.vision_config is not None` still owes a note. Exempt by field: framework plumbing that gates no checkpoint divergence -- `problem_type` selecting a loss, `hidden_act` looking up an activation, `num_labels`, `use_cache`, `is_decoder`, the special token ids, the `summary_*` head settings; the full list is `DEFAULT_EXEMPT_ATTRIBUTES` in `mlinter/trf041.py`, extended per project by `ignored_attributes = [...]` on the rule table. A model exempts one of its own fields file-wide with a module-level `# trf-ignore: TRF041 config.scale_embedding, config.auxiliary_loss` at column 0 (`self.config.x`, `config.x` and `x` are the same field). It must name at least one field -- a bare `# trf-ignore: TRF041` keeps its per-line meaning -- and a condition is skipped only when every field it reads is exempt. A config-gated branch is a second architecture in the same file, and the code cannot say whether both halves are still reachable -- which is how dead experimental branches survive for releases. The rule does not forbid the branch: like Rust's `// SAFETY:`, it asks for the checkpoints taking each side to be written down next to it. A branch nobody can name one for is a branch to delete.

```diff
+        # CODEPATH: ESMC-6B ships pre-normalised embeddings, the 300M/600M checkpoints do not.
         if config.use_embedding_norm:
             hidden_states = self.embedding_norm(hidden_states)

-        if config.msa_encoder_enabled:
-            hidden_states = self.msa_encoder(hidden_states)
+        # no released checkpoint sets msa_encoder_enabled -> branch removed
```

### TRF042

In tests/models/*/test_tokenization_*.py, checks the file defines a collected test class inheriting TokenizerTesterMixin. Only classes the runner collects count -- a `TestCase` base, or the `*Test` naming convention when the base is another model's test class -- so helper-only files are skipped, and a helper mixing in the suite does not satisfy the rule for a real test class. Inheritance is followed through bases in the same file and into another model's tokenizer test imported by name; an unresolvable base never counts. Reported on the first test class that does not run the suite. `auto` is allowlisted: test_tokenization_auto.py tests AutoTokenizer resolution, not a tokenizer. TokenizerTesterMixin is where encode/decode round-tripping, padding and truncation, special-token handling, added-token persistence and save/load equivalence are actually checked. A test that only asserts a few hand-written token id lists passes while the tokenizer is broken in every one of them, and still looks tested in review.

```diff
-class AcmeTokenizationTest(unittest.TestCase):
+class AcmeTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
+    tokenizer_class = AcmeTokenizer
+    test_slow_tokenizer = True
```

### TRF043

Flags a declared position_ids parameter in the forward signature of classes whose name ends in Attention. position_ids is consumed downstream by flash-attention padding-free training and must flow through **kwargs. Naming it in the signature swallows it before the attention interface can read it; the llama standard passes position_embeddings plus **kwargs.

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

Flags a parameter named cache_position on any function in modeling_*.py and modular_*.py. cache_position was removed from all models in v5. Reintroducing it (usually copied from pre-v5 sources) threads a dead argument through every layer; the cache update is past_key_values.update(key_states, value_states, self.layer_idx), with no position threading.

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

Flags the legacy output_attentions, output_hidden_states and return_dict parameters in forward signatures in modeling_*.py and modular_*.py. Models contributed before the cutoff date are exempt. The decorator stack owns output control: @capture_outputs resolves the output_* flags against the config and records tensors via _can_record_outputs, and @can_return_tuple handles return_dict. Declaring them in the signature reintroduces manual flag threading that drifts from the decorators.

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

Flags assignments to self attributes in forward methods in modeling_*.py and modular_*.py. State written during forward breaks batching, torch.compile, and reasoning about the module. Carried state is passed explicitly (cache objects, the generate loop); values that depend only on config or static shapes belong in __init__.

```diff
def forward(self, hidden_states):
-    self.sequence_length = hidden_states.shape[1]
-    embeddings = self.compute_embeddings(self.sequence_length)
+    embeddings = self.compute_embeddings(hidden_states.shape[1])
```

### TRF047

Flags assignments to self attributes in preprocess, _preprocess, __call__ and post_process* methods in image_processing_*.py and video_processing_*.py. A processor carrying state between calls breaks preprocess-many-then-postprocess batching: the second preprocess overwrites the state the first postprocess needs. Return the value or pass it through the method chain.

```diff
def _preprocess(self, images, **kwargs):
-    self.original_sizes = [image.shape[-2:] for image in images]
+    original_sizes = [image.shape[-2:] for image in images]
     ...
+    return BatchFeature(data={"pixel_values": pixel_values, "original_sizes": original_sizes})
```

### TRF048

Flags list/tuple/set literals in class-level _tied_weights_keys declarations. v5 changed _tied_weights_keys to a dict mapping each tied target parameter to its source. The list form no longer says which parameter is the source, so tying, device_map computation and saving misbehave silently.

```diff
class AcmeForCausalLM(AcmePreTrainedModel):
-    _tied_weights_keys = ["lm_head.weight"]
+    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
```

### TRF049

Flags init calls in __init__ methods in modeling_*.py and modular_*.py: nn.init.* / init.* primitives and in-place initializers on own parameters (self.weight.data.normal_()). Models instantiate on the meta device, so tensor values written in __init__ are discarded before loading: a parameter initialized only there holds random content when fine-tuning from scratch or after a meta-device reload. Allocate with torch.empty in __init__, initialize in _init_weights.

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

Flags calls to a *RotaryEmbedding class in the __init__ of classes whose name ends in Attention. The Model owns a single rotary_emb, builds inv_freq once, and passes cos/sin down as position_embeddings. One rotary module per attention layer duplicates buffers, recomputes frequencies per layer, and diverges from the contract that attention receives position_embeddings.

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

Flags comparisons against a _attn_implementation attribute in modeling_*.py and modular_*.py. Backend dispatch belongs to ALL_ATTENTION_FUNCTIONS.get_interface, and backend-conditional tensor munging (padding, reshaping) belongs in the shared wrappers under integrations/. Inline branching keeps the model body kernel-aware and breaks when new backends register.

```diff
-if self.config._attn_implementation == "flash_attention_2":
-    attn_output = flash_path(query_states, key_states, value_states)
-else:
-    attn_output = eager_path(query_states, key_states, value_states)
+attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
+attn_output, attn_weights = attention_interface(self, query_states, key_states, value_states, ...)
```

### TRF052

Flags module-level assignments to names ending in _ATTENTION_CLASSES in modeling_*.py and modular_*.py. Per-backend attention classes picked from a dict are the pre-interface idiom: near-identical classes drift apart, and hub attention kernels registered into ALL_ATTENTION_FUNCTIONS never reach them. One attention class dispatching through the interface replaces the dict; do not propagate it from a legacy parent.

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

In modeling_*.py and modular_*.py, flags assignments that build shift_logits/shift_labels (and shifted_ variants) by slicing, as in labels[..., 1:]. Receiving already-shifted labels (shift_labels = kwargs.pop("shift_labels", labels)) is the correct idiom and is not flagged. self.loss_function shifts labels itself, so pre-shifting in modeling code trains on doubly-shifted targets or forces a bespoke loss path. Decoder-only models pass raw labels and let the loss shift them. Encoder-decoder models are the mirror case: their labels are already shifted by the prepended decoder start token, so they pass shift_labels=labels to stop the loss shifting again.

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

Flags `config = SomeConfig` as a class attribute on PreTrainedModel subclasses in modeling_*.py and modular_*.py. `PreTrainedModel.__init_subclass__` derives `config_class` from a `config` **annotation**, via `inspect.get_annotations(cls)`. An assignment creates a stray class attribute that call does not report, so the subclass silently keeps the parent's `config_class`. A bare annotation has no runtime value, creates no attribute, and is picked up correctly.

```diff
class Gemma4VisionModel(Gemma4PreTrainedModel):
     """The Gemma 4 Vision Encoder."""
-    config = Gemma4VisionConfig
+    config: Gemma4VisionConfig
```

### TRF056

In modeling_*.py and modular_*.py, flags `.item()` and `.tolist()` calls inside any `forward`. A `.tolist()` feeding the split-size argument of `split(...)` is exempt: torch.split needs Python ints. Both calls read a tensor back to the host, which dynamo cannot trace, so the graph breaks.

```diff
-        for grid, item in zip(grid_thw.tolist(), split_items):
-            _, height, width = grid
-            merged.append(self.patch_merger(item, size=(height, width)))
+        for grid, item in zip(grid_thw, split_items):
+            merged.append(self.patch_merger(item, size=(grid[1], grid[2])))
```

### TRF057

Checks `@auto_docstring` on public `PreTrainedModel` subclasses (`<Model>PreTrainedModel`, `<Model>Model`, `<Model>For<Task>`, backbones), `PreTrainedConfig` subclasses, `ModelOutput` subclasses, image processors and `ProcessorMixin` subclasses, and on their public methods `forward`, `get_image_features`, `get_video_features`, `get_audio_features`, `get_text_features`, `preprocess` and `__call__`. A `modular_*.py` is checked against the files generated from it. Without it a class ships with no intro and no parameter documentation, and a method with no argument documentation, no `Returns` section and no usage example -- all of which then have to be hand-written per model instead of coming from `auto_docstring.py`.

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

### TRF058

In modeling_*.py and modular_*.py, flags `register_buffer("<name>", ...)` calls whose buffer name is a string literal, on any receiver (`self`, or another module such as `layer.mamba`). A computed name -- a variable or f-string, e.g. one buffer per layer inside a loop -- has no attribute-assignment equivalent and is exempt. Since torch>=2.5 `nn.Buffer` registers a buffer through plain attribute assignment, like `nn.Parameter`. A buffer created by a method call only exists as a side effect of running `__init__`, so a modular file that wants to tweak one has to redefine the whole `__init__`. Assigned as an attribute, it can be inherited and overridden on its own.

```diff
-        self.register_buffer("inv_freq", inv_freq, persistent=False)
-        self.register_buffer(
-            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
-        )
+        self.inv_freq = nn.Buffer(inv_freq, persistent=False)
+        self.position_ids = nn.Buffer(torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False)
```

### TRF059

For model directories whose tensor-parallel plan assigns `moe_tp_experts`, checks that routed `*Experts` classes take hidden states, top-k indices and top-k routing weights as the first three positional `forward` arguments. Aliases such as `selected_experts` and `routing_weights` are accepted. `MoeExpertsParallel` applies a gradient transform to positional argument 3. A different signature silently applies it to the wrong tensor or skips it.

```diff
class AcmeExperts(nn.Module):
-    def forward(self, hidden_states):
+    def forward(self, hidden_states, top_k_index, top_k_weights):
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
