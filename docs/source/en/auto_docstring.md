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

# Auto-generating docstrings

The `@auto_docstring` decorator generates consistent docstrings for model classes and methods. It pulls in standard argument descriptions automatically, so you only write documentation for new or custom arguments. When [adding a new model](./modular_transformers), skip the boilerplate and focus on what's new.

## @auto_docstring

Import the decorator in your `modular_model.py` file (or `modeling_model.py` for older models).

```python
from ...utils import auto_docstring
```

If your model inherits from another library model in a modular file, `@auto_docstring` is already applied in the parent. `make fix-repo` copies it into the generated `modeling_model.py` file for you. Only apply the decorator explicitly to customize its behavior (standalone models, custom intros, or overridden arguments).

> [!WARNING]
> When overriding any decorator in a modular file, include **all** decorators from the parent function or class. If you only override some, the rest won't appear in the generated modeling file.

The decorator accepts the following optional arguments:

| argument | description |
|---|---|
| `custom_intro` | A description of the class or method, inserted before the Args section. Required for classes that don't end with a [recognized suffix](#how-it-works) like `ForCausalLM` or `ForTokenClassification`. |
| `custom_args` | Docstring text for specific parameters. Useful when the same custom arguments appear in several places in the modeling file. |
| `checkpoint` | A model checkpoint identifier (`"org/my-model"`) used to generate usage examples. Overrides the checkpoint auto-inferred from the config class. Typically set on config classes. |

## Usage

How `@auto_docstring` works depends on what you're decorating. Model classes pull parameter docs from `__init__`, config classes pull from class-level annotations, processor classes auto-generate intros from their components, and methods like `forward` get return types and usage examples.

### Model classes

Place `@auto_docstring` directly above the class definition. The decorator derives parameter descriptions from the `__init__` method's signature and docstring.

```python
from transformers.modeling_utils import PreTrainedModel
from ...utils import auto_docstring

@auto_docstring
class MyAwesomeModel(PreTrainedModel):
    def __init__(self, config, custom_parameter: int = 10, another_custom_arg: str = "default"):
        r"""
        custom_parameter (`int`, *optional*, defaults to 10):
            Description of the custom_parameter for MyAwesomeModel.
        another_custom_arg (`str`, *optional*, defaults to "default"):
            Documentation for another unique argument.
        """
        super().__init__(config)
        self.custom_parameter = custom_parameter
        self.another_custom_arg = another_custom_arg
        # ... rest of your init

    # ... other methods
```

Pass `custom_intro` and `custom_args` for more control. Custom arguments can go in `custom_args` or in the `__init__` docstring. Use `custom_args` when the same arguments repeat across several methods.

```python
@auto_docstring(
    custom_intro="""This model performs specific synergistic operations.
    It builds upon the standard Transformer architecture with unique modifications.""",
    custom_args="""
    custom_parameter (`type`, *optional*, defaults to `default_value`):
        A concise description for custom_parameter if not defined or overriding the description in `auto_docstring.py`.
    internal_helper_arg (`type`, *optional*, defaults to `default_value`):
        A concise description for internal_helper_arg if not defined or overriding the description in `auto_docstring.py`.
    """
)
class MySpecialModel(PreTrainedModel):
    def __init__(self, config: ConfigType, custom_parameter: "type" = "default_value", internal_helper_arg=None):
        # ...
```

Also apply `@auto_docstring` to classes that inherit from [`~utils.ModelOutput`].

```python
@auto_docstring(
    custom_intro="""
    Custom model outputs with additional fields.
    """
)
@dataclass
class MyModelOutput(ImageClassifierOutput):
    r"""
    loss (`torch.FloatTensor`, *optional*):
        The loss of the model.
    custom_field (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
        A custom output field specific to this model.
    """

    # Standard fields (hidden_states, logits, attentions, etc.) are documented automatically when
    # the description matches the standard text. Loss typically varies per model, so document it above.
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    # Custom fields need to be documented in the docstring above
    custom_field: Optional[torch.FloatTensor] = None
```

### Config classes

Place `@auto_docstring` directly above a [`PreTrainedConfig`] subclass, alongside the `@strict` decorator. `@strict` adds runtime type validation and turns the class into a validated dataclass. Config parameters are *class-level annotations* (not `__init__` arguments), and `@auto_docstring` reads them from the class body to generate docs.

[`ConfigArgs`] provides standard parameters like `vocab_size`, `hidden_size`, and `num_hidden_layers`, so they don't need a description unless the behavior differs. [`PreTrainedConfig`] base parameters are excluded automatically. The `checkpoint` argument generates the usage example.

```python
from huggingface_hub.dataclasses import strict
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring

@strict
@auto_docstring(checkpoint="org/my-model-checkpoint")
class MyModelConfig(PreTrainedConfig):
    r"""
    custom_param (`int`, *optional*, defaults to 64):
        Description of a parameter specific to this model.
    another_param (`str`, *optional*, defaults to `"gelu"`):
        Description of another model-specific parameter.

    ```python
    >>> from transformers import MyModelConfig, MyModel

    >>> configuration = MyModelConfig()
    >>> model = MyModel(configuration)
    >>> configuration = model.config
    ```
    """

    model_type = "my_model"

    # Standard params (vocab_size, hidden_size, etc.) are auto-documented from ConfigArgs.
    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    # Model-specific params must be documented in the class docstring above.
    custom_param: int = 64
    another_param: str = "gelu"
```

### Processor classes

Multimodal processors ([`ProcessorMixin`] subclasses, `processing_*.py`) always use the bare `@auto_docstring`. The class intro is auto-generated. Document only `__init__` parameters not already covered by [`ProcessorArgs`] (`image_processor`, `tokenizer`, `chat_template`, and others).

If every parameter is standard, omit the docstring. Decorate `__call__` with `@auto_docstring` too. Its body docstring holds only a `Returns:` section plus any extra model-specific call arguments. `return_tensors` is appended automatically.

```python
from ...processing_utils import ProcessorMixin, ProcessingKwargs, Unpack
from ...utils import auto_docstring

class MyModelProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {"text_kwargs": {"padding": False}}

@auto_docstring
class MyModelProcessor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, custom_param: int = 4, **kwargs):
        r"""
        custom_param (`int`, *optional*, defaults to 4):
            A parameter specific to this processor not covered by the standard ProcessorArgs.
        """
        super().__init__(image_processor, tokenizer)
        self.custom_param = custom_param

    @auto_docstring
    def __call__(self, images=None, text=None, **kwargs: Unpack[MyModelProcessorKwargs]):
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- Token ids to be fed to the model.
            - **pixel_values** -- Pixel values to be fed to the model.
        """
        # ...
```

#### Image and video processors

Image and video processors (`BaseImageProcessor` subclasses, `image_processing_*.py`) follow one of two patterns.

If the processor has model-specific parameters, define a `XxxImageProcessorKwargs(ImagesKwargs, total=False)` TypedDict with a docstring for those parameters, set `valid_kwargs` on the class, and use the bare `@auto_docstring`. The `__init__` has no docstring.

```python
class MyModelImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    custom_threshold (`float`, *optional*, defaults to `self.custom_threshold`):
        A parameter specific to this image processor.
    """
    custom_threshold: float | None

@auto_docstring
class MyModelImageProcessor(TorchvisionBackend):
    valid_kwargs = MyModelImageProcessorKwargs
    custom_threshold: float = 0.5

    def __init__(self, **kwargs: Unpack[MyModelImageProcessorKwargs]):
        super().__init__(**kwargs)
```

If the class only sets standard class-level attributes (`size`, `resample`, `image_mean`, etc.) with no custom kwargs, use `@auto_docstring(custom_intro="Constructs a MyModel image processor.")` instead.

```python
@auto_docstring(custom_intro="Constructs a MyModel image processor.")
class MyModelImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 224, "width": 224}
```

When overriding `preprocess`, decorate it with `@auto_docstring` and document only arguments not in [`ImageProcessorArgs`]. Standard arguments and `return_tensors` are included automatically.

### Functions

Place `@auto_docstring` directly above the function definition. The decorator derives parameter descriptions from the function signature.

The decorator generates return-value text from the [`ModelOutput`] class docstring.

```python
class MyModel(PreTrainedModel):
    # ...
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        new_custom_argument: Optional[torch.Tensor] = None,
        # ... other arguments
    ) -> Union[Tuple, ModelOutput]:
        r"""
        new_custom_argument (`torch.Tensor`, *optional*):
            Description of this new custom argument and its expected shape or type.
        """
        # ...
```

Pass `custom_intro` and `custom_args` for more control. Use `custom_args` to define shared argument docs once when the same parameters appear in several methods.

```python
MODEL_COMMON_CUSTOM_ARGS = r"""
    common_arg_1 (`torch.Tensor`, *optional*, defaults to `default_value`):
        Description of common_arg_1
    common_arg_2 (`torch.Tensor`, *optional*, defaults to `default_value`):
        Description of common_arg_2
"""

class MyModel(PreTrainedModel):
    # ...
    @auto_docstring(
        custom_intro="""This is a custom introduction for the function.""",
        custom_args=MODEL_COMMON_CUSTOM_ARGS
    )
    def forward(self, input_ids=None, common_arg_1=None, common_arg_2=None) -> ModelOutput:
        r"""method-specific args go here"""
        # ...
```

Write `Returns` and `Examples` sections manually in the docstring to override the auto-generated versions.

```python
    def forward(self, input_ids=None) -> torch.Tensor:
        r"""
        Returns:
            `torch.Tensor`: A custom Returns section for non-ModelOutput return types.

        Example:

        ```python
        >>> model = MyModel.from_pretrained("org/my-model")
        >>> output = model(input_ids)
        ```
        """
        # ...
```

### Documenting arguments

Follow these rules when documenting different argument types.

- `auto_docstring.py` defines standard arguments (`input_ids`, `attention_mask`, `pixel_values`, etc.) and includes them automatically. Don't redefine them locally unless the argument behaves differently in your model.

    If a standard argument behaves differently in your model, override it locally in a `r""" """` block. The local definition takes priority. The `labels` argument, for instance, is commonly customized per model and often needs an override.

- Standard config arguments (`vocab_size`, `hidden_size`, `num_hidden_layers`, etc.) follow the same principle but come from [`ConfigArgs`]. Standard processor arguments (`image_processor`, `tokenizer`, `do_resize`, `return_tensors`, etc.) come from [`ProcessorArgs`] and [`ImageProcessorArgs`]. Only document a parameter if it is model-specific or behaves differently from the standard description.

- Document new or custom arguments in an `r""" """` block. Place them after the signature for functions, in the `__init__` docstring for model or processor classes, in the class body docstring for config classes, or in the `XxxImageProcessorKwargs` TypedDict body for image processors.

    ```py
    argument_name (`type`, *optional*, defaults to `X`):
        Description of the argument.
        Explain its purpose, expected shape/type if complex, and default behavior.
        This can span multiple lines.
    ```

  * Include `type` in backticks.
  * Add *optional* if the argument is not required or has a default value.
  * Add "defaults to X" if it has a default value. You don't need to add "defaults to `None`" if the default value is `None`.
  * Pass the same block into `custom_args` when the same arguments repeat across several methods (see the [Functions example above](#functions)).

- The decorator extracts types from function signatures automatically. If a parameter has a type annotation, you don't need to repeat the type in the docstring format string. When both are present, the signature type takes precedence. The docstring type acts as a fallback for unannotated parameters.

## Checking the docstrings

A utility script validates docstrings when you open a pull request. CI runs the script and checks the following.

> [!TIP]
> If you see an `[ERROR]` in the output, add the parameter's description to the docstring or the appropriate Args class in `auto_docstring.py`.

* Checks that `@auto_docstring` is applied to relevant model classes and public methods.
* Validates argument completeness and consistency: documented arguments must exist in the signature, and types and default values must match. Unknown arguments without a local description are flagged.
* Flags incomplete placeholders like `<fill_type>` and `<fill_docstring>`.
* Verifies docstrings follow the expected formatting style.

Run the check locally before committing.

```bash
make fix-repo
```

`make fix-repo` runs several other checks too. To run only the docstring and auto-docstring checks, use the command below.

```bash
# to only check files included in the diff without fixing them
python utils/check_docstrings.py
# to fix and overwrite the files in the diff
# python utils/check_docstrings.py --fix_and_overwrite
# to fix and overwrite all files
# python utils/check_docstrings.py --fix_and_overwrite --check_all
```

## Quick-reference checklist

| Do | Don't |
|---|---|
| Apply `@auto_docstring` to model, config, and processor classes and their primary methods (`forward`, `__call__`, `preprocess`). | Add `@auto_docstring` to inherited models in modular files because it carries over automatically. |
| Document only new or model-specific arguments. | Redefine standard arguments (`input_ids`, `attention_mask`, `vocab_size`, etc.) that behave the same as their default descriptions. |
| Put config parameters in the class body docstring as class-level annotations. | Put config parameters in `__init__`. |
| Put image processor parameters in a `XxxImageProcessorKwargs` TypedDict. | Put image processor parameters in `__init__`. |
| Run `python utils/check_docstrings.py --fix_and_overwrite` before committing. | Ignore `[ERROR]` output because it means a parameter is undocumented. |

## How it works

The `@auto_docstring` decorator generates docstrings through the following steps.

1. The decorator inspects the signature to read arguments, types, and defaults from the decorated class's `__init__` or the decorated function. For config classes, it walks class-level annotations up the inheritance chain and stops before [`PreTrainedConfig`], excluding base class fields.

    It automatically filters out parameters like `self`, `kwargs`, `args`, `deprecated_arguments`, and `_`-prefixed names. A few private parameters are renamed to their public equivalents (`_out_features` → `out_features` for backbone models).

2. Common argument descriptions come from `auto_docstring.py`: [`ModelArgs`] (model inputs), [`ModelOutputArgs`] (output fields like `hidden_states`), [`ImageProcessorArgs`] (image preprocessing), [`ProcessorArgs`] (multimodal processor components), and [`ConfigArgs`] (config hyperparameters).

3. Each parameter's description follows this priority chain:

    - A manual docstring (`r""" """` block or `custom_args`) takes priority.
    - The predefined source dict ([`ModelArgs`], [`ConfigArgs`], [`ImageProcessorArgs`], [`ProcessorArgs`], [`ModelOutputArgs`]) is the fallback.
    - If neither source has a description, the parameter is flagged with `[ERROR]` in the build output.

4. For model classes with standard names like `ModelForCausalLM`, or classes that map to a pipeline, `@auto_docstring` generates the intro. For multimodal processors, the intro lists which components (tokenizer, image processor, and so on) the class wraps. See [ClassDocstring](https://github.com/huggingface/transformers/blob/main/src/transformers/utils/auto_docstring.py#L2437) for the full list.

    If the class name isn't in `ClassDocstring`, set `custom_intro`.

5. Predefined docstrings can reference dynamic values from Transformers' [auto_modules](https://github.com/huggingface/transformers/tree/main/src/transformers/models/auto), such as `{processor_class}`, `{image_processor_class}`, and `{config_class}`. The placeholders resolve automatically.

6. The decorator picks usage examples based on the model's task or pipeline compatibility. It reads checkpoint metadata from the configuration class so examples use real model IDs. The `checkpoint` argument overrides the checkpoint inferred from the config class's docstring. Set `checkpoint` on config classes, or when checkpoint inference fails. If you see an error like `"Config not found for <model_name>"`, add an entry to `HARDCODED_CONFIG_FOR_MODELS` in `auto_docstring.py`.

7. For methods like `forward`, the decorator writes the `Returns` section from the method's return type. When the return type is a [`~transformers.utils.ModelOutput`] subclass, `@auto_docstring` pulls field descriptions from that class's docstring. A custom `Returns` block in the function's docstring takes precedence.

8. For methods in `UNROLL_KWARGS_METHODS` and classes in `UNROLL_KWARGS_CLASSES`, the decorator expands `**kwargs` typed with `Unpack[KwargsTypedDict]`. Each key from the `TypedDict` becomes a documented parameter.

    The same expansion applies to `__call__` and `preprocess` methods on [`BaseImageProcessor`] and [`ProcessorMixin`] subclasses. Generic base types (`TextKwargs`, `ImagesKwargs`, `VideosKwargs`, `AudioKwargs`) are skipped. Only model-specific subclasses are unrolled.
