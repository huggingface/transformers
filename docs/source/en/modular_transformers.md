<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
-->

# Add a model with modular transformers

Modular transformers reduces the code needed to add a model by allowing imports and inheritance, in contrast to the [single model, single file](https://huggingface.co/blog/transformers-design-philosophy) policy. Instead of repeating model components across files, add a *modular* file to your model folder and inherit from existing classes.

A converter generates standalone files from the modular file. Users get the same single-file interface they already know.

> [!NOTE]
> Modular transformers isn't meant to replace the [legacy modeling code](./add_new_model). If your model isn't based on an existing model, add a `modeling.py` file manually. The same applies to configuration, tokenization, or processing files that can't cleanly inherit from a similar file.
>
> There's no single right order either. Some contributors write the modular file first and generate from it. Others start with a hand-written `modeling.py` and refactor it into a modular file later. Both approaches work.

## Implementing a modular file

Start by finding a model in Transformers similar to yours. Good starting points are [Mistral](./model_doc/mistral), [Qwen2](./model_doc/qwen2), [Cohere](./model_doc/cohere) and [Cohere2](./model_doc/cohere2), and [Llama](./model_doc/llama). The table below maps common components to models you can inherit from.

| Component | Model |
|---|---|
| Mixture of experts | Mixtral or Qwen2-MoE |
| Interleaved (and/or partial) rotary embedding | GLM, Phi |
| State space models | Jamba, Bamba, Zamba, Mamba2 |
| Recurrent hidden states | Gemma2 |
| Sliding window attention/full attention patterns per layer | Gemma2, Cohere2 |
| QKV clipping | Olmo |
| QK normalization | Olmo2, Cohere |
| Fused QKV (not recommended) | Phi3 |

> [!TIP]
> Use the [modular-detector-v2](https://huggingface.co/spaces/Molbap/modular-detector-v2) tool to find existing implementations to inherit from. Paste a code snippet and it returns the most similar methods already in Transformers, so you can identify the best parent class before you start writing.

Don't modify an existing model just to make inheritance work for your new one. If renaming or subclassing a parent class is too awkward, copy the relevant code directly instead.

Create `src/transformers/models/<name>/modular_<name>.py`, where `<name>` matches the snake_case model directory name. This section walks you through implementing [Olmo2](./model_doc/olmo2) from [Olmo](./model_doc/olmo) with the modular approach (refer to the original [modular_olmo2.py](../../../src/transformers/models/olmo2/modular_olmo2) file).

### Config

There are two points where [`Olmo2Config`] differs from [`OlmoConfig`].

1. There is a new argument, `rms_norm_eps`.
2. The `clip_qkv` argument is no longer used.

Declare new arguments as class-level type annotations with a default value. For removed arguments, assign `AttributeError()` to suppress the inherited attribute in the generated file (see [Removing attributes](#removing-attributes)).

```diff
- @auto_docstring(checkpoint="allenai/OLMo-7B-hf")
+ @auto_docstring(checkpoint="allenai/Olmo2-7B-1124-hf")
+ @strict
- class OlmoConfig(PreTrainedConfig):
+ class Olmo2Config(OlmoConfig):
      ...
-     model_type = "olmo"
+     model_type = "olmo2"
      ...
+     rms_norm_eps: float = 1e-5
-     clip_qkv: float | None = None
+     clip_qkv = AttributeError()
```

`@auto_docstring` generates standard argument docs automatically (see the [@auto_docstring](./auto_docstring) guide). `@strict` rejects unknown kwargs at instantiation time, catching typos and stale arguments early. Add both to every config class because the decorators aren't inherited from the parent. Declare them explicitly even if the parent config already has them.

To set a derived attribute or handle backward-compatibility logic, use `__post_init__` instead of `__init__`. For example, Cohere2 computes `head_dim` and derives `layer_types` at init time.

```py
def __post_init__(self, **kwargs):
    if self.num_key_value_heads is None:
        self.num_key_value_heads = self.num_attention_heads
    self.head_dim = self.hidden_size // self.num_attention_heads
    super().__post_init__(**kwargs)
```

For models with tensor or pipeline parallelism support, define `base_model_tp_plan` and `base_model_pp_plan` as class-level dictionaries on the config. Both dictionaries define how to shard the model across devices. See existing configs like [Olmo2](../../../src/transformers/models/olmo2/modular_olmo2) or [Cohere2](../../../src/transformers/models/cohere2/modular_cohere2) for examples.

```py
class MyNewModelConfig(PreTrainedConfig):
    model_type = "my_new_model"

    # Tensor parallelism: maps layer name patterns to sharding strategies.
    # Use "colwise" / "rowwise" for standard sharding, or the "gather_output" /
    # "split_input" variants when an extra op (e.g. a QK norm) prevents fusing.
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    # Pipeline parallelism: maps submodule names to their (input, output) tensor names.
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
```

### Norm

To copy a parent class without changes, inherit with `pass`. The linter copies the parent's content and renames all references to match the new model.

```py
from ..olmo.modeling_olmo import OlmoRotaryEmbedding

class Olmo2RotaryEmbedding(OlmoRotaryEmbedding):
    pass
```

To change specific behavior, inherit and override only what differs. [`Olmo2RMSNorm`] differs from [`LlamaRMSNorm`] on one line. The multiply happens *before* casting back to the input dtype, not after.

```diff
  from ..llama.modeling_llama import LlamaRMSNorm

  class Olmo2RMSNorm(LlamaRMSNorm):
      def forward(self, hidden_states):
          input_dtype = hidden_states.dtype
          hidden_states = hidden_states.to(torch.float32)
          variance = hidden_states.pow(2).mean(-1, keepdim=True)
          hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
-         return self.weight * hidden_states.to(input_dtype)
+         return (self.weight * hidden_states).to(input_dtype)
```

### Attention

Olmo2's attention is identical to Olmo's except it applies [`RMSNorm`] to the queries and keys, and removes qkv clipping. `super().__init__(...)` copies the parent body and appends the two new norm lines. The `forward` is fully redefined because queries and keys now pass through norms before projection. The linter also pulls in any imported functions into the generated file, including `apply_rotary_pos_emb`, `eager_attention_forward`, and their dependencies.

```diff
  class Olmo2Attention(OlmoAttention):
      def __init__(self, config: Olmo2Config, layer_idx: int | None = None):
          super().__init__(config, layer_idx=layer_idx)
+         self.q_norm = Olmo2RMSNorm(config.num_attention_heads * self.head_dim, config.rms_norm_eps)
+         self.k_norm = Olmo2RMSNorm(config.num_key_value_heads * self.head_dim, config.rms_norm_eps)

      def forward(self, ...):
          ...
-         query_states = self.q_proj(hidden_states)
-         key_states = self.k_proj(hidden_states)
+         query_states = self.q_norm(self.q_proj(hidden_states))
+         key_states = self.k_norm(self.k_proj(hidden_states))
          value_states = self.v_proj(hidden_states)

-         if self.config.clip_qkv is not None:
-             query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
-             key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
-             value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
-
          ...
```

### DecoderLayer

After `super().__init__(...)`, overwrite the norm attributes with `Olmo2RMSNorm` instances and reassign `self.self_attn` to the new `Olmo2Attention` class. The `del self.input_layernorm` removes the parent's `input_layernorm` assignment since Olmo2 applies the norm *after* attention rather than before. See [Removing attributes](#removing-attributes) for details on what `del` does and doesn't remove.

The `forward` is rewritten to reflect the post-attention norm placement. A `forward` rewrite is only needed when an attribute is renamed, not when only its type changes.

```diff
  class Olmo2DecoderLayer(OlmoDecoderLayer):
      def __init__(self, config: Olmo2Config, layer_idx: int):
          super().__init__(config, layer_idx=layer_idx)
-         self.self_attn = OlmoAttention(config=config, layer_idx=layer_idx)
-         self.input_layernorm = OlmoLayerNorm(config.hidden_size)
-         self.post_attention_layernorm = OlmoLayerNorm(config.hidden_size)
+         self.self_attn = Olmo2Attention(config=config, layer_idx=layer_idx)
+         del self.input_layernorm
+         self.post_attention_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
+         self.post_feedforward_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

      def forward(self, ...):
          residual = hidden_states
-         hidden_states = self.input_layernorm(hidden_states)
          # Self Attention
          hidden_states, _ = self.self_attn(...)
-         hidden_states = residual + hidden_states
+         hidden_states = self.post_attention_layernorm(hidden_states)
+         hidden_states = residual + hidden_states

          # Fully Connected
          residual = hidden_states
-         hidden_states = self.post_attention_layernorm(hidden_states)
          hidden_states = self.mlp(hidden_states)
-         hidden_states = residual + hidden_states
+         hidden_states = self.post_feedforward_layernorm(hidden_states)
+         hidden_states = residual + hidden_states
          return hidden_states
```

### Model

Only the type of `self.norm` changes here. The `forward` method is identical to the parent's, so the linter carries it over automatically.

```diff
  class Olmo2Model(OlmoModel):
      def __init__(self, config: Olmo2Config):
          super().__init__(config)
-         self.layers = nn.ModuleList(
-             [OlmoDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
-         )
-         self.norm = OlmoLayerNorm(config.hidden_size)
+         self.norm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
+         self.layers = nn.ModuleList(
+             [Olmo2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
+         )
```

### Model head

The logic is identical to [`OlmoForCausalLM`], so no changes are needed.

```py
from ..olmo.modeling_olmo import OlmoForCausalLM

class Olmo2ForCausalLM(OlmoForCausalLM):
    pass
```

### Other classes

The [modeling_olmo2.py](../../../src/transformers/models/olmo2/modeling_olmo2) generated by the linter also contains classes ([`Olmo2MLP`], [`Olmo2RotaryEmbedding`], [`Olmo2PreTrainedModel`]) that weren't explicitly defined in `modular_olmo2.py`.

The linter pulls in any class an inherited class depends on, unless you explicitly redefine it. Imported functions like `apply_rotary_pos_emb` follow the same rule.

For example, [`OlmoDecoderLayer`] has `self.mlp = OlmoMLP(config)`. [`Olmo2MLP`] was never defined in the modular file, so the linter creates it automatically, equivalent to using `pass`.

```py
from ..olmo.modeling_olmo import OlmoMLP

class Olmo2MLP(OlmoMLP):
    pass
```

If you want [`Olmo2MLP`] to inherit from a different model instead, be explicit.

```py
# switch to Mistral definition
from ..mistral.modeling_mistral import MistralMLP

class Olmo2MLP(MistralMLP):
    pass
```

### Finishing the file

Every modular file must declare a `logger` and an `__all__` list at the module level.

```py
logger = logging.get_logger(__name__)

__all__ = [
    "Olmo2Config",
    "Olmo2ForCausalLM",
    "Olmo2Model",
    "Olmo2PreTrainedModel",
]
```

`__all__` must list every public class in the file. The converter and downstream imports depend on it. A class missing from `__all__` won't be exported correctly.

## Generate the modeling files

The `modular_model_converter.py` script generates standalone `modeling.py`, `configuration.py`, and other files from your modular file. For each inherited class, it copies the parent body into the child, renames all references to match the new model, and pulls in any helper functions or classes those parents depend on.

The output files contain no cross-model imports and no inheritance from other model directories. The linter flattens inheritance to a single level. If [`Olmo2Attention`] inherits from [`OlmoAttention`], the generated `Olmo2Attention` is fully self-contained. But if `OlmoAttention` itself inherited from something else, the linter doesn't inline that grandparent.

Run the command below to generate files from a modular file.

```bash
python utils/modular_model_converter.py your_model
```

Never edit the generated files directly because any changes will be overwritten on the next run.

## Patterns for modular files

The sections below document common usage patterns, such as removing attributes or overriding decorated methods, when working with a modular file.

### Removing attributes

Removing an inherited attribute depends on whether you're working with a config class or an `nn.Module` subclass.

For a config class, assign `AttributeError()` to the attribute at the class level.

```py
class MyNewConfig(ParentConfig):
    removed_attr = AttributeError()
```

The linter removes the attribute declaration from the generated config file entirely. Config classes use a dataclass-style layout with no `__init__`, so assigning `AttributeError()` at the class level is the correct approach.

For an `nn.Module` subclass, use `del self.attribute` after `super().__init__(...)`.

```py
class MyNewModel(ParentModel):
    def __init__(self, config: MyNewConfig):
        super().__init__(config)
        del self.attribute
```

`del self.attribute` removes only the `self.attribute = ...` assignment line from the copied parent body. It doesn't remove any other lines that reference `self.attribute`. If the parent's `forward` or other methods also reference the attribute, override those methods too.

```py
class DummyModel(nn.Module):
    def __init__(self, config: DummyConfig):
        super().__init__()
        self.attribute = config.attribute
        if self.attribute:
            # do more stuff with `self.attribute` here
            ...

class MyNewDummyModel(DummyModel):
    def __init__(self, config: MyNewDummyConfig):
        super().__init__(config)
        del self.attribute
        # 'self.attribute = config.attribute' is removed, but the 'if self.attribute:' block remains.
        # Override forward() or any other method that references self.attribute.
```

### Working with `super()`

`super().__init__(config)` tells the converter to copy the parent body into the child. Two patterns let you override this behavior.

- Call a specific parent class directly when you need the generated output to call a grandparent (`nn.Module.__init__`) rather than the modular parent.
- Use `**super_kwargs` to inherit a parent method's full signature while adding a custom docstring or swapping a decorator.

#### Call a grandparent class directly

Be explicit about which class you're calling when you need `super()` to target the generated class parent rather than the modular parent. The example below calls `nn.Module.__init__(self)` directly. `DummyModule` is itself an `nn.Module`, so the converter writes it as `super().__init__()` in the generated `MyNewDummyModule`.

```py
class MyNewDummyModule(DummyModule):                   |     class MyNewDummyModule(nn.Module):
                                                       |
  def __init__(self):                                  |       def __init__(self):
    nn.Module.__init__(self)                           |         super().__init__()
    self.foo = config.foo                              |         self.foo = config.foo
    ...                                                |         ...
```

#### super_kwargs

Use `**super_kwargs` to inherit a parent method's full signature while adding a custom docstring or swapping a decorator. In the overridden signature, it tells the linter to expand all parent arguments in the generated output.

The most common use is adding a model-specific docstring, like documenting the `labels` argument, without rewriting the full signature.

```py
# modular_gemma.py
class GemmaForCausalLM(LlamaForCausalLM):
    def forward(**super_kwargs):
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, GemmaForCausalLM
        >>> model = GemmaForCausalLM.from_pretrained("google/gemma-7b")
        ...
        ```"""
        return super().forward(**super_kwargs)
```

The generated `GemmaForCausalLM.forward` has the full `LlamaForCausalLM` signature with no manual copying needed.

`**super_kwargs` is a shortcut for niche cases. If you're changing behavior, write the full signature instead.

### Deleting unused methods

Remove a parent method by overriding it with a `raise AttributeError("")` statement. The linter removes the method from the generated file.

```py
class GemmaTokenizer(LlamaTokenizer):
    ...

    def get_spm_processor(self):
        raise AttributeError("Not needed for Gemma")

    def unk_token_length(self):
        raise AttributeError("Not needed for Gemma")
```

### Overriding decorated methods

When you override a decorated parent method, the parent's decorator carries over automatically. If you add your own decorator, it replaces the parent's.

Two decorators appear throughout the library, one for [capturing model intermediate outputs](./model_output_tracing) and one for [auto-generating docstrings](./auto_docstring).

In the example below, a subclass overrides a decorated parent method without adding its own decorator. The parent's decorator carries over.

```py
class NewModel(DummyModel):       |   class NewModel(nn.Module):
  ...                             |     ...
                                  |
  def forward(...):               |     @decorator(...)
    ...                           |     def forward(...):
                                  |       ...
```

If you add a new decorator, your decorator replaces the parent's.

```py
class NewModel(DummyModel):       |   class NewModel(nn.Module):
  ...                             |     ...
                                  |
  @my_new_decorator(...)          |     @my_new_decorator(...)
  def forward(...):               |     def forward(...):
    ...                           |       ...
```

### Special naming

The linter automatically renames everything when inheriting from a class. Use the same class name prefix across all classes in the same file.

Avoid mixing prefixes like in the example below. `MyModelIncredibleMLP` breaks naming conventions, and the linter won't know whether to use `MyModelIncredible` or `MyModel` when renaming higher-order dependencies.

```py
class MyModelIncredibleMLP(LlamaMLP):
    ...

class MyModelDecoderLayer(LlamaDecoderLayer):
    ...
```

With no [implicit dependencies](#other-classes), you can rename a single class locally. Explicitly redefine every other mention of that class with the new name pattern. Otherwise, the linter adds an unwanted `MyModelMLP` class alongside `MyModelIncredibleMLP`.

The linter raises a warning when it detects an ambiguous prefix.

```text
We detected multiple prefix names when inheriting from transformers.models.llama.modeling_llama: ('Emu3Text', 'Emu3'). We will only use the most used 'Emu3' prefix when grabbing args and dependencies. Make sure to subclass the intermediate classes with the prefix you want (if different from 'Emu3') or use a single prefix in all the modular (best).
```

Ambiguous prefixes are most common in multimodal models where class names include a modality qualifier like `Text`. To give a dependency a specific prefix, explicitly rename it with a `pass`.

```py
class Emu3TextMLP(LlamaMLP):
    pass
```

### Config docstrings

The linter doesn't support partial docstring inheritance yet. When adding or removing config attributes, add the full docstring directly in the modular file under the class definition.

## Checkpoint conversion

Once you've generated your modeling files, verify that real weights load correctly. Write a conversion script to translate the upstream checkpoint format into a Transformers-compatible one, then save it to the Hub.

### Write a conversion script

Add a `convert_<model>_to_hf.py` file to `src/transformers/models/<model>/`. The script loads the upstream weights, renames and reshapes keys to match your module's parameter names, and saves the result with [`~PreTrainedModel.save_pretrained`].

> [!TIP]
> Look for an existing script to copy and adapt. Models under `src/transformers/models/` include a `convert_*_to_hf.py` you can use as a starting point.

After running the script, load the saved checkpoint with [`~PreTrainedModel.from_pretrained`] and confirm every expected weight loaded correctly. Unused checkpoint keys indicate mismatched names, so print them to catch problems early.

```py
model = YourModelForTask.from_pretrained("path/to/output/")
```

Check shape and name matches when iterating over keys. Shape mismatches typically mean a parameter in your config is wrong, the architecture differs from the original, or a weight needs to be transposed.

```py
for key, tensor in original_state_dict.items():
    hf_tensor = hf_model.state_dict().get(mapped_key)
    assert hf_tensor.shape == tensor.shape, (
        f"Shape mismatch for {key}: expected {tensor.shape}, got {hf_tensor.shape}"
    )
```

Fix any issues by iterating between your modular file, the generated modeling file, and the conversion script until all weights load cleanly.

Once the checkpoint loads cleanly, push it to the Hub using [`~PreTrainedModel.push_to_hub`]. Refer to the [model sharing](./model_sharing) guide for more details.

```py
model.push_to_hub("username/your-model-name")
```

### Runtime conversion mapping

Add a runtime mapping to `src/transformers/conversion_mapping.py` when the published weights don't match your module's parameter layout. Common cases include fused weights stored separately and MoE expert tensors that need stacking. The mapping lets [`~PreTrainedModel.from_pretrained`] load the Hub checkpoint without a separate export step.

Refer to the [dynamic weight loading](./weightconverter) guide for how to write [`WeightRenaming`] and [`WeightConverter`] rules and register them for your `model_type`.

## Next steps

- [Model structure rules](./modeling_rules) are static rules enforced on all `modeling_*.py`, `modular_*.py`, and `configuration_*.py` files. Run `make typing` to check them before opening a PR.
- [Add vision processing components](./add_vision_processing_components) walks through adding an image processor, video processor, and processor for a multimodal model.
- [Auto-generating docstrings](./auto_docstring) shows how to use `@auto_docstring` so you don't have to hand-write argument docs for shared model APIs.
- [Writing model tests](./testing) covers how to write integration tests for your new model and run it locally.
- [Pull request checks](./pr_checks) explains the CI checks your PR has to pass before it can be merged, and how to reproduce and fix them locally.
