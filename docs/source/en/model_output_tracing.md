# Tracing model intermediate outputs

Every model's `forward()` method used to manually resolve `None` flags like `output_attentions` from config defaults, accumulate per-layer attention weights and hidden states into tuples, and convert [`ModelOutput`] dataclasses to plain tuples when `return_dict=False`. Two decorators replace all of that boilerplate.

- `@capture_outputs` resolves output flags, collects intermediate values, and handles `return_dict` conversion.
- `@merge_with_config_defaults` resolves `use_cache` from config. Omit it for models that don't cache, like [`CLIPModel`].

You'll mostly encounter these decorators when integrating a new model. See [adding a model to 🤗 Transformers]./modular_transformers) for a step-by-step guide.

## Declare which submodules to capture

Apply `@capture_outputs` to the base model's `forward()` method. It attaches forward hooks that:

- Intercept outputs from specified submodule classes during the forward pass, without those submodules needing to know they're being observed.
- Collect per-layer attention weights and hidden states into tuples.
- Inject collected values into the returned [`ModelOutput`] dataclass.
- Convert the dataclass to a plain tuple when `return_dict=False`.
- Resolve `output_attentions` and `output_hidden_states` from kwargs or `self.config` when `None`.

## Map output fields to submodules

`@capture_outputs` needs to know which submodule produces which output. Declare a `_can_record_outputs` class-level dictionary on your `PreTrainedModel` subclass. Each key is an output field name (`"hidden_states"`, `"attentions"`, `"cross_attentions"`), and each value is a module class or an `OutputRecorder` instance.

## Fine-grained control with OutputRecorder

`OutputRecorder` accepts a `target_class` (an `nn.Module` subclass whose outputs to collect) and an optional `index` to select which element of the module's output tuple to get. Pass `layer_name` to attach hooks only to modules with a specific attribute name. Use `layer_name` when two layers share the same class, for example self-attention vs. cross-attention.

The exampple below shows how these decoratos are used in practice, including different levels output control. See [LlamaModel](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) for real world usage in codebase.

```python
from ...processing_utils import Unpack
from ...utils import TransformersKwargs
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs, OutputRecorder

class MyPreTrainedModel(PreTrainedModel):
    _can_record_outputs = {
        # Capture hidden_states: hook fires after each MyDecoderBlock forward,
        # grabbing its first output (index 0 by default).
        "hidden_states": MyDecoderBlock,

        # Capture self-attention weights: hook fires after each MyAttention
        # forward, grabbing its second output (index=1 by default).
        "attentions": MyAttention,

        # Capture cross-attention weights: same class, different submodule.
        # layer_name targets the attribute `self.crossattention` inside the block.
        # Captures second output as requested (index=1)
        "cross_attentions": OutputRecorder(
            MyAttention, layer_name="crossattention", index=1
        ),
    }

# Now in base model's forward we need the decorators and `Unpack` `kwargs`
class MyModel(MyPreTrainedModel):

    @merge_with_config_defaults # ← resolves use_cache
    @capture_outputs            # ← handles output collection + return_dict
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:

        # No manual collection needed. Just run your layers normally.
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)

        # Return the primary outputs. The decorator will fill in
        # hidden_states/attentions/cross_attentions automatically.
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values
        )
```

## Advanced usage: swapping intermediate capturing layers at load time

The output-tracing system described above depends on `_can_record_outputs` pointing at the *exact* classes that a model's layers instantiate. When you swap out a layer implementation — for example, to use a custom attention kernel, a quantised expert layer, or an architecture variant — those pointers need to stay in sync. The patching API provides a clean, global registry for this, and automatically keeps `_can_record_outputs` consistent.

First let's see how you can swap layers and use custom layers with `register_patch_mapping`. Keys to `register_patch_mapping` can be exact class names or regex patterns — exact matches take priority, then patterns are tested with `re.search()`. Trying to register the same key twice raises `ValueError` unless you pass `overwrite=True`. The replacement must be a subclass of `nn.Module`; anything else raises an error.


```python
from transformers.monkey_patching import register_patch_mapping, unregister_patch_mapping

# Exact name – replaces only Qwen2MoeExperts
register_patch_mapping({"Qwen2MoeExperts": SequentialExperts})

# Regex – replaces every class whose name ends in "Attention"
register_patch_mapping({".*Attention$": FusedAttention})

# Anchored version – only matches Llama2Attention, Llama3Attention, …
register_patch_mapping({"^Llama\\d+Attention$": CustomLlamaAttention})

# Same way, custom keys can be removed from the registry by passing the name that was registered
unregister_patch_mapping(["Qwen2MoeExperts", ".*Attention$"])
```

Now that the layers have been swapped, we need to update the `OutputRecorder` to point at the new target class. `patch_output_recorders` fixes this by walking every submodule of an already-instantiated model and updating each `OutputRecorder.target_class` to the registered replacement:

```python
# Built manually, outside from_pretrained
model = Qwen2MoeModel(config)

# Without this, _can_record_outputs still points at the original Qwen2MoeExperts class
# and hooks will never fire on CustomExperts instances.
patch_output_recorders(model)
```
