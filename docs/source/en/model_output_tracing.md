# Tracing Model Intermediate Outputs with Decorators

Every Transformers model's `forward()` method has historically needed to handle three chores that is same across all model classes and acrhitectures:

1. **Resolve `None` flags from config defaults** — If the caller passes `output_attentions=None`, fall back to `self.config.output_attentions`. Same for `output_hidden_states`, `use_cache`, and `return_dict`.
2. **Manually collect intermediate outputs** — Loop over layers, appending each layer's attention weights to an accumulator tuple when `output_attentions=True`; doing the same for hidden states.
3. **Convert `ModelOutput` → tuple on the way out** — When `return_dict=False`, unpack the dataclass into a flat tuple.

This boilerplate was copy-pasted across every one of the 240+ models in the library. Two new decorators, `@capture_outputs` and `@merge_with_config_defaults` can replace all of it.


## Collect intermediate outputs with `@capture_outputs` and `@merge_with_config_defaults`

`@capture_outputs` is applied to the base model's `forward()` method and attached forward hooks to:

- Automatically intercept outputs from specified submodule classes during the forward pass, without those submodules needing to know they're being observed.
- Collect per-layer attention weights and hidden states into tuples.
- Inject the collected values into the `ModelOutput` dataclass that the method returns.
- Handle the `return_dict=False` conversion (producing a plain tuple instead of a dataclass).
- Resolve `output_attentions` and `output_hidden_states` from kwargs or from `self.config` when `None`.

For `@capture_outputs` to know *which* submodule produces *which* output, the `ModelPreTrainedModel` class must declare a `_can_record_outputs` class-level dictionary. Each key is an output field name (`"hidden_states"`, `"attentions"`, `"cross_attentions"`), and each value is either a module class or an `OutputRecorder` instance.

The `OutputRecorder` expects a `module_class` which is an `nn.Module` instance whose outputs we want to collect and optionally an `index` to indicate which element of the module's output tuple to grab. If `layer_name` is also provided, hooks will be attached only modules with the given layer name. This is useful when two layers share the same class as in self-attention vs. cross-attention layers.

Another required decorator, `@merge_with_config_defaults`, resolves the `use_cache` argument to config's default value. It can be ommited in models that do not require caching such as `CLIPModel`


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
    @capture_outputs          # ← handles output collection + return_dict
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
