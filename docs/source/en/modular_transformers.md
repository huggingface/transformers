# Contributing a new model to Transformers

Modular Transformers lowers the bar for contributing models and significantly reduces the code required to add a model by allowing imports and inheritance.

One of Transformers' core design feature is the [single model, single file](https://huggingface.co/blog/transformers-design-philosophy) policy. Model components - such as attention layers - are repeated across many files and any independent implementations tend to diverge as fixes and changes are applied to specific parts of the code.

The [`# Copied from`](./pr_checks#check-copies) statements prevents the code from diverging, and it is enforced by our continuous integration tests and local commands. The downside is that this approach is tedious and adds significantly more lines of code, most of which is boilerplate.

## Motivation

Modular Transformers addresses these issues by adding a *modular* file to a model folder. The modular file can import code from other models and inherit code from other classes unlike traditional modeling and processing files.

> [!TIP]
> Modular Transformers isn't meant to replace the modeling code, and if your model isn't based on an existing model, you'll need to add a `modeling.py` file manually. Likewise, if a configuration, tokenization or processing file can't easily inherit from a similar file, you can add that file directly.

A modular file contains model, processor, and configuration class code that would otherwise be in separate files under the single model, single file policy.

Model users still import and use the single-file interface they've grown familiar with. In doing so, we hope to enable simpler contributions while sticking to our philosophy.

## Create a modeling.py file

A linter "unravels" the modular file into a `modeling.py` file to preserve the single model, single file directory structure (modeling, processor, etc.). Inheritance is flattened to only a **single** level.

Run the command below to automatically generate a `modeling.py` file from a modular file (assuming the snake lowercase name of the model you want to convert is `your_model`).

```bash
python utils/modular_model_converter.py  your_model
```

For example:

- If a configuration class inherits from another class, but adds and deletes an argument, the generated file directly references it if an argument is added or completely removes it if an argument is deleted.
- If a class inherits from another, like `GemmaModel(LlamaModel)`, the dependencies are automatically inferred. All submodules are also automatically inferred from the superclass.
- If a new function is defined in the modular file and used inside classes, the linter automatically infers these as well.

You should be able to write everything (tokenizer, image processor, model, config, etc.) in a modular and their corresponding single-files are generated.

The example below demonstrates how a model can be added with significantly fewer lines of code with Modular Transformers.

### BERT and RoBERTa

BERT and RoBERTa, two very similar models, differ solely in how the embedding layer is implemented.

Instead of redefining the model entirely, consider the `modular_roberta.py` file shown below for the modeling and configuration classes (the tokenizer isn't shown in this example).

```py
from torch import nn
from ..bert.configuration_bert import BertConfig
from ..bert.modeling_bert import (
    BertModel,
    BertEmbeddings,
    BertForMaskedLM
)

# RoBERTa and BERT config is identical
class RobertaConfig(BertConfig):
  model_type = 'roberta'

# Redefine the embeddings to highlight the padding id difference, and redefine the position embeddings
class RobertaEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config())

        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

# RoBERTa and BERT model is identical except for the embedding layer, which is defined above, so no need for additional changes here
class RobertaModel(BertModel):
  def __init__(self, config):
    super().__init__(config)
    self.embeddings = RobertaEmbeddings(config)


# The model heads now only need to redefine the model inside to `RobertaModel`
class RobertaForMaskedLM(BertForMaskedLM):
  def __init__(self, config):
    super().__init__(config)
    self.model = RobertaModel(config)
```

If you don't use the defined dependency, you'll receive the following error.

```
ValueError: You defined `RobertaEmbeddings` in the modular_roberta.py, it should be used when you define `BertModel`, as it is one of it's direct dependencies. Make sure you use it in the `__init__` function.
```

## Implementing a modular file

The easiest way to start is by browsing Transformers for a model similar to yours in order to inherit from it. Some good starting points are [Mistral](./model_doc/mistral), [Qwen2](./model_doc/qwen2), [Cohere](./model_doc/cohere) and [Cohere2](./model_doc/cohere2), and [Llama](./model_doc/llama). Refer to the table below for components your model might be using and where you can inherit from.

| Component | Model |
|---|---|
| Mixture of expert | SwitchTransformers or Mixtral |
| Interleaved (and/or partial) rotary embedding | GLM, Phi |
| State space models | Jamba, Bamba, Zamba, Mamba2 |
| Recurrent hidden states | Gemma2 |
| Sliding window attention/full attention patterns per layer | Gemma2, Cohere2 |
| QKV clipping | Olmo |
| QK normalization | Olmo2, Cohere |
| Fused QKV (not recommended) | Phi3 |

This section will walk you through how to implement [Olmo2](./model_doc/olmo2) from [Olmo](./model_doc/olmo) with modular Transformers (you can refer to the original [modeling.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modular_olmo2.py) file).

### Config

The modular `Olmo2Config` is shown below.

```py
from ..olmo.configuration_olmo import OlmoConfig

class Olmo2Config(OlmoConfig):
    r"""
    This is the configuration class to store the configuration of a [Olmo2Model](/docs/transformers/main/en/model_doc/olmo2#transformers.Olmo2Model).
    """

    def __init__(
        self,
        vocab_size=50304,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            **kwargs,
        )

        self.rms_norm_eps = rms_norm_eps
        del self.clip_qkv
```

There are three points where the `Olmo2Config` is different from the original `OlmoConfig`.

1. The default value of most arguments have changed.
2. There is a new argument, `rms_norm_eps`.
3. The `clip_qkv` argument isn't used anymore.

For the new default values and argument, overwrite the `__init__` function with the new default values and add `rms_norm_eps`. Assign `rms_norm_eps` to `self` in the body of `__init__`. For the `clip_qkv` argument, use `del self.clip_qkv` to remove the assignment of this attribute in the unraveled code (post-linter conversion).

Notice how the `super().__init__(...)` is used. Typically, it calls the parent `__init__`.

But in modular Transformers, if there is a call like `super().my_function(...)`, the linter takes the body of `my_function` in the parent and unravels it where the call to `super().my_function(...)` occurred. The `del self.clip_qkv` statement removes the reference to `self.clip_qkv` in the unraveled body.

`del self.` and `super().my_function(..)` work together, and it should always be placed after `super().my_function(...)`. You can add whatever you want *before* calling `super()`, and it is placed before the parents body.

### Norm

```py
from ..llama.modeling_llama import LlamaRMSNorm

class Olmo2RMSNorm(LlamaRMSNorm):
    pass
```

Nothing needs to be modified in `LlamaRMSNorm`. The linter unravels the exact content of `LlamaRMSNorm` into `Olmo2RMSNorm`. References to Llama in the docstrings, type hints, and comments are also changed to Olmo2.

### Attention

The modular `Olmo2Attention` is shown below.

```py
from ..llama.modeling_llama import eager_attention_forward
from ..olmo.modeling_olmo import OlmoAttention, apply_rotary_pos_emb


# Olmo2 attention is identical to OLMo attention except:
# - Norm is applied to attention queries and keys.
# - No qkv clipping.
class Olmo2Attention(OlmoAttention):
    def __init__(self, config: Olmo2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx=layer_idx)
        self.q_norm = Olmo2RMSNorm(config.num_attention_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = Olmo2RMSNorm(config.num_key_value_heads * self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

The `super().__init__(...)` copies the parent definition and adds 2 new layers from `Olmo2RMSNorm`. The forward pass needs to be overwritten to use these 2 new layers. A pass with the norm layers is added before projecting with `q_proj` and `k_proj`. To make it easier, the `eager_attention_forward` function is directly imported from Llama and the `apply_rotary_pos_emb` is imported from Olmo.

The linter automatically adds these imported functions in the final `modeling_olmo2.py` file by copying their definitions from the source files. The `rotate_half` and `repeat_kv` functions are also added because they are used inside `apply_rotary_pos_emb` and `eager_attention_forward`.

The `Attention` class had to be redefined because there weren't any existing models with an `Attention` layer that included a `RMSNorm` layer.

### DecoderLayer

The modular `DecoderLayer` is shown below.

```py
from ..olmo.modeling_olmo import OlmoDecoderLayer

# The OLMo2 layers are identical to those of the OLMo model except:
# - RMSNorm is used instead of standard layer norm.
# - Norm is applied after attention/feedforward rather than before.
class Olmo2DecoderLayer(OlmoDecoderLayer):
    def __init__(self, config: Olmo2Config, layer_idx: int):
        super().__init__(config, layer_idx=layer_idx)
        self.post_attention_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Olmo2Attention(config=config, layer_idx=layer_idx)
        del self.input_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
```

The norm type is switched in `__init__` by overwriting `self.post_attention_layernorm` after the call to `super().__init__(...)`. Delete the `self.input_layernorm` attributed and replace it with `self.post_feedforward_layernorm` because it is applied after in Olmo2. The forward method is overwritten to reflect this change.

If you only switched `self.post_feedforward_layernorm` and `self.input_layernorm` from `LayerNorm` to `RMSNorm` without also changing the name and logic of `self.input_layernorm`, then you wouldn't have to rewrite the forward method.

### Model

The modular `Olmo2Model` class is shown below.

```py
from ..olmo.modeling_olmo import OlmoModel

# The OLMo2 model is identical to the OLMo model, except RMSNorm is used instead of
# standard layer norm for the output norm.
class Olmo2Model(OlmoModel):
    def __init__(self, config: Olmo2Config):
        super().__init__(config)
        self.norm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [Olmo2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
```

You only need to change the *type* of the `self.norm` attribute to use `RMSNorm` instead of `LayerNorm`. This change doesn't affect the logic in the forward method (layer name and usage is identical to the parent class), so you don't need to overwrite it. The linter automatically unravels it.

### Model head

The modular causal modeling head is shown below.

```py
from ..olmo.modeling_olmo import OlmoForCausalLM

class Olmo2ForCausalLM(OlmoForCausalLM):
    pass
```

The logic is identical to `OlmoForCausalLM` which means you don't need to make any changes here.

### Other classes

The [modeling_olmo2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmo2/modeling_olmo2.py) generated by the linter also contains some classes (`Olmo2MLP`, `Olmo2RotaryEmbedding`, `Olmo2PreTrainedModel`) that weren't explicitly defined in `modular_olmo2.py`.

Classes that are a dependency of an inherited class but aren't explicitly defined are automatically added as a part of dependency tracing. This is similar to how some functions were added to the `Attention` class without directly importing them.

For example, `OlmoDecoderLayer` has an attribute defined as `self.mlp = OlmoMLP(config)`. This class was never explicitly redefined in `Olmo2MLP`, so the linter automatically created a `Olmo2MLP` class similar to `OlmoMLP`. It is identical to the code below if it was explicitly written in `modular_olmo2.py`.

```py
from ..olmo.modeling_olmo import OlmoMLP

class Olmo2MLP(OlmoMLP):
    pass
```

However, it was necessary to rewrite `Olmo2RMSNorm` because the layer norm needed to be redefined in the `Attention` and `DecoderLayer` classes. Similarly, this is why you didn't need to create the `Olmo2PreTrainedModel` and `Olmo2RotaryEmbedding` classes.

Classes that aren't rewritten are copied from the file where the inherited module first uses them. This means if you wanted `Olmo2MLP` to inherit from `MistralMLP` instead, you would need to be more explicit as shown below.

```py
# switch to mistral definition
from ..mistral.modeling_mistral import MistralMLP

class Olmo2MLP(MistralMLP):
    pass
```

## Removing attributes

You can `del` to remove attributes defined in the parent after using `super().__init__()`. However, this doesn't work if the attribute is also used somewhere else as shown below. It only suppresses the assignment. The `self.attribute = config.attribute` line is removed, but the `if` statement remains and references the attribute.

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
```

## Calling parent methods without unravelling their definition

If you want to inherit from a module `DummyModule` and want to call `super()` WITHOUT unravelling the parent's code (that is, you want to call `super()` on the *generated* class parent), be explicit about which class' `super()` you're calling. The example below shows how to call the `super()` of `nn.Module` (unraveled code shown on the right). In this example, as `DummyModule` is itself a `nn.Module`, it makes sense to call `nn.Module.__init__(self)` as it's what was the initial intention. It's then unravelled as `super()` in `MyNewDummyModule` to follow Python's best-practices.

```py
class MyNewDummyModule(DummyModule):                   |     class MyNewDummyModule(nn.Module):
                                                       |
  def __init__(self):                                  |       def __init__(self):
    nn.Module.__init__(self)                           |         super().__init__()
    self.foo = config.foo                              |         self.foo = config.foo
    ...                                                |         ...
```

## Deleting unused methods

Remove an attribute by overwriting it with a `raise AttributeError("")` statement to mimic the behavior you want when you remove a parent function in Python. The example below removes the methods in the unraveled code.

```py
class GemmaTokenizer(LlamaTokenizer):
    ...

    def get_spm_processor(self):
        raise AttributeError("Not needed for Gemma")

    def unk_token_length(self):
        raise AttributeError("Not needed for Gemma")
```

## Defining new functions

By default, if you inherit from a class and override a method with one or more decorators in the parent method, the decorators are also added to the unraveled code *only if you don't add any yourself*. Otherwise, the redefined decorator is used.

For example, if you had a parent class shown below and you overwrite it, the parent decorator is kept.

```py
class DummyModel(nn.Module):
  ...

  @decorator(...)
  def forward(...)
    # do stuff here
```

Modular code is shown on the left, and the unraveled code is shown on the right.

```py
class NewModel(DummyModel):       |   class NewModel(nn.Module):
  ...                             |     ...
                                  |
  def forward(...):               |     @decorator(...)
    ...                           |     def forward(...):
                                  |       ...
```

But if you add a new decorator, your new decorator is used instead.

```py
class NewModel(DummyModel):       |   class NewModel(nn.Module):
  ...                             |     ...
                                  |
  @my_new_decorator(...)          |     @my_new_decorator(...)
  def forward(...):               |     def forward(...):
    ...                           |       ...
```

## super_kwargs

In scenarios where a forward method is really long and you want to switch decorators, you don't need to redefine everything and copy/paste the function. You can use `super().forward(...)` to unravel the parent body. When there are a lot of arguments in the function signature, use the special `**super_kwargs` syntax in the overwritten signature.

This syntax indicates to the linter to unravel all the parent signature arguments here. An example signature in a [`AutoModelForCausalLM`] model is shown below, with lots of arguments.

```py
class LlamaForCausalLM(nn.Module):
  ...

  @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
  @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
  def forward(
      self,
      input_ids: torch.LongTensor = None,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.LongTensor] = None,
      past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = None,
      inputs_embeds: Optional[torch.FloatTensor] = None,
      labels: Optional[torch.LongTensor] = None,
      use_cache: Optional[bool] = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
      cache_position: Optional[torch.LongTensor] = None,
      num_logits_to_keep: int = 0,
      **kwargs: Unpack[KwargsForCausalLM],
  ) -> Union[Tuple, CausalLMOutputWithPast]:
    ...
```

Instead of rewriting and copying/pasting all of those arguments, use the `super().forward(**super_kwargs)` statement (modular code shown on the left, unraveled code on the right).

```py
class NewModelForCausalLM(LlamaForCausalLM):    |    class LlamaForCausalLM(nn.Module):
  ...                                           |      ...
                                                |
  @my_new_decorator                             |     @my_new_decorator
  def forward(self, **super_kwargs):            |     def forward(
    super().forward(**super_kwargs)             |         self,
                                                |         input_ids: torch.LongTensor = None,
                                                |         attention_mask: Optional[torch.Tensor] = None,
                                                |         position_ids: Optional[torch.LongTensor] = None,
                                                |         past_key_values: Optional[Union[Cache, list[torch.FloatTensor]]] = |None,
                                                |         inputs_embeds: Optional[torch.FloatTensor] = None,
                                                |         labels: Optional[torch.LongTensor] = None,
                                                |         use_cache: Optional[bool] = None,
                                                |         output_attentions: Optional[bool] = None,
                                                |         output_hidden_states: Optional[bool] = None,
                                                |         return_dict: Optional[bool] = None,
                                                |         cache_position: Optional[torch.LongTensor] = None,
                                                |         num_logits_to_keep: int = 0,
                                                |         **kwargs: Unpack[KwargsForCausalLM],
                                                |     ) -> Union[Tuple, CausalLMOutputWithPast]:
                                                |       ...
```

This makes it very easy to switch decorators and makes it explicit that the only change you want to apply is the decorator.

`**super_kwargs` should not be used to avoid being explicit when redefining methods though. If you overwrite a method, you should explicitly write the signature as you normally would. The `**super_kwargs` syntax is a shortcut for switching decorators and a few other niche cases.

## Docstring variables

> [!TIP]
> Refer to the [Documeting a model](./auto_docstring) guide for more information about how you can use the `@auto_docstring` decorator to help automatically generate consistent docstring arguments.

If an object defined in both the modular and modeling file from which it inherits, the modular definition has precedence unless for assignments containing the pattern `DOCSTRING`. These variables are typically used in `MODEL_START_DOCSTRING` and `MODEL_INPUT_DOCSTRING` in the modeling files. They are big blocks of docstrings and the linter rewrites the names everywhere. For this reason, assignments containing the `DOCSTRING` variable can use the definition found in the source file without copying the whole docstring, by simply setting the variable to `None` in the modular file.

This is very useful if you need the variable reference somewhere but you don't want to clutter the modular file with docstrings which are always the same. The example code below allows you to automatically use the same docstrings from [Mistral](./model_doc/mistral) in [Starcoder2](./model_doc/starcoder2).

```py
STARCODER2_INPUTS_DOCSTRING = None  # will be automatically redefined

class Starcoder2Model(MistralModel):
    ...

    @add_start_docstrings_to_model_forward(STARCODER2_INPUTS_DOCSTRING)
    def forward(...)
        ...
```

Setting the variable to anything other than `None` will override the docstring, so that you can customize the docstrings if needed.

## Special naming

The linter automatically renames everything when inheriting from a class. For consistency, you should always use the same class name prefix when inheriting from different classes from the same file.

The example below is not recommended. It breaks standards in the library, `MyModelIncredibleMLP` instead of `LlamaMLP`, and because the linter doesn't know how to rename potential higher-order dependencies (`MyModelIncredible` or just `MyModel`).

```py
class MyModelIncredibleMLP(LlamaMLP):
    ...

class MyModelDecoderLayer(LlamaDecoderLayer):
    ...
```

However, if there aren't any [implicit dependencies](#other-classes), then you can locally rename a single class. Make sure you still explicitly redefine every other mention of the class with the new name pattern though. For example, all mentions of `LlamaMLP` should be renamed to `MyModelIncredibleMLP` otherwise the linter may add a new and unwanted `MyModelMLP` class.

The linter raises a warning if an ambiguous case is detected. It explains what is happening and which prefix is used by default for getting the dependencies. These warning and renaming pattern complications usually only come up when defining multimodal models. For example, adding `Text` to class names in a multimodal model to make it clear which modality it refers to.

```py
We detected multiple prefix names when inheriting from transformers.models.llama.modeling_llama: ('Emu3Text', 'Emu3'). We will only use the most used 'Emu3' prefix when grabbing args and dependencies. Make sure to subclass the intermediate classes with the prefix you want (if different from 'Emu3') or use a single prefix in all the modular (best).
```

If there are automatic dependencies with a prefix, but you want another one, explicitly rename the classes locally with a `pass` class as shown in the following.

```py
class Emu3TextMLP(LlamaMLP):
    pass
```

## Config docstrings

When inheriting a `Config` class or adding and deleting attributes, you may want to only redefine the new attributes in the docstring. However, the linter doesn't support this yet. You need to directly add the while docstring directly in the modular file under the class definition.
