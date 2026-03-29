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

# Documenting a model

The `@auto_docstring` decorator generates consistent docstrings for model classes and methods. It pulls in standard argument descriptions automatically, so you only write documentation for new or custom arguments. When [adding a new model](./modular_transformers), skip the boilerplate and focus on what's unique.

This page covers how to use `@auto_docstring` and how it works under the hood.

## @auto_docstring

Import the decorator in your `modular_model.py` file (or `modeling_model.py` for older models).

```python
from ...utils import auto_docstring
```

If your model inherits from another library model in a modular file, `@auto_docstring` is already applied in the parent. `make fix-repo` copies it into the generated `modeling_model.py` file for you. Only apply the decorator explicitly to customize its behavior (standalone models, custom intros, or overridden arguments).

The decorator accepts the following optional arguments:

| argument | description |
|---|---|
| `custom_intro` | A description of the class or method, inserted before the Args section. Required for classes whose name isn't in `ClassDocstring`. |
| `custom_args` | Docstring text for specific parameters. Useful when the same custom arguments appear in several places in the modeling file. |
| `checkpoint` | A model checkpoint identifier (e.g. `"org/my-model"`) used to generate usage examples. Overrides the checkpoint auto-inferred from the config class. Typically set on config classes. |

## Usage

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

Pass `custom_intro` and `custom_args` for more control.

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

Or use only `custom_intro` and define the custom arguments in the class docstring instead.

```python
@auto_docstring(
    custom_intro="""This model performs specific synergistic operations.
    It builds upon the standard Transformer architecture with unique modifications.""",
)
class MySpecialModel(PreTrainedModel):
    def __init__(self, config: ConfigType, custom_parameter: "type" = "default_value", internal_helper_arg=None):
        r"""
        custom_parameter (`type`, *optional*, defaults to `default_value`):
            A concise description for custom_parameter if not defined or overriding the description in `auto_docstring.py`.
        internal_helper_arg (`type`, *optional*, defaults to `default_value`):
            A concise description for internal_helper_arg if not defined or overriding the description in `auto_docstring.py`.
        """
        # ...
```

Also apply `@auto_docstring` to classes that inherit from [`~utils.ModelOutput`].

```python
@dataclass
@auto_docstring(
    custom_intro="""
    Custom model outputs with additional fields.
    """
)
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

Place `@auto_docstring` directly above a [`PreTrainedConfig`] subclass, alongside `@strict` from `huggingface_hub.dataclasses`. Config parameters are *class-level annotations* (not `__init__` arguments), following the `@strict` dataclass pattern used throughout Transformers. The class docstring documents model-specific parameters and optionally a usage example.

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

`ConfigArgs` provides standard parameters like `vocab_size`, `hidden_size`, and `num_hidden_layers`, so they don't need a description unless the behavior differs. `PreTrainedConfig` base parameters are excluded automatically. The `checkpoint` argument generates the usage example.

### Processor classes

Multimodal processors ([`ProcessorMixin`] subclasses, `processing_*.py`) always use bare `@auto_docstring`. The class intro is auto-generated. Document only `__init__` parameters not already covered by [`ProcessorArgs`] (`image_processor`, `tokenizer`, `chat_template`, and others). If every parameter is standard, omit the docstring. Decorate `__call__` with `@auto_docstring` too. Its body docstring holds only a `Returns:` section plus any extra model-specific call arguments. `return_tensors` is appended automatically.

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

Image and video processors (`BaseImageProcessor` subclasses, `image_processing_*.py`) follow one of two patterns.

If the processor has model-specific parameters, define a `XxxImageProcessorKwargs(ImagesKwargs, total=False)` TypedDict with a docstring for those parameters, set `valid_kwargs` on the class, and use bare `@auto_docstring`. The `__init__` has no docstring.

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

When overriding `preprocess`, decorate it with `@auto_docstring` and document only arguments not in `ImageProcessorArgs`. Standard arguments and `return_tensors` are included automatically.

### Functions

Place `@auto_docstring` directly above the method definition. The decorator derives parameter descriptions from the function signature.

Return-value text is generated from the [`ModelOutput`] class docstring.

```python
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        new_custom_argument: Optional[torch.Tensor] = None,
        arg_documented_in_args_doc: Optional[torch.Tensor] = None,
        # ... other arguments
    ) -> Union[Tuple, ModelOutput]:
        r"""
        new_custom_argument (`torch.Tensor`, *optional*):
            Description of this new custom argument and its expected shape or type.
        """
        # ...
```

Pass `custom_intro` and `custom_args` for more control. `Returns` and `Examples` sections can be written manually.

```python
MODEL_COMMON_CUSTOM_ARGS = r"""
    common_arg_1 (`torch.Tensor`, *optional*, defaults to `default_value`):
        Description of common_arg_1
    common_arg_2 (`torch.Tensor`, *optional*, defaults to `default_value`):
        Description of common_arg_2
    ...
"""

class MyModel(PreTrainedModel):
    # ...
    @auto_docstring(
        custom_intro="""
        This is a custom introduction for the function.
        """,
        custom_args=MODEL_COMMON_CUSTOM_ARGS
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        common_arg_1: Optional[torch.Tensor] = None,
        common_arg_2: Optional[torch.Tensor] = None,
        #...
        function_specific_argument: Optional[torch.Tensor] = None,
        # ... other arguments
    ) -> torch.Tensor:
        r"""
        function_specific_argument (`torch.Tensor`, *optional*):
            Description of an argument specific to this function

        Returns:
            `torch.Tensor`: For a function returning a generic type, a custom "Returns" section can be specified.

        Example:

        (To override the default example with a custom one or to add an example for a model class that does not have a pipeline)

        ```python
        ...
        ```
        """
        # ...
```

> [!WARNING]
> When overriding any decorator in a modular file, include **all** decorators from the parent function or class. If you only override some, the rest won't appear in the generated modeling file.

## Documenting arguments

Follow these rules when documenting different argument types.

- The decorator resolves each parameter through a priority chain (see [How it works](#how-it-works)).

- Standard arguments (`input_ids`, `attention_mask`, `pixel_values`, etc.) are defined in `auto_docstring.py` (the single source of truth for standard arguments) and included automatically. Don't redefine them locally unless the argument behaves differently in your model.

    If a standard argument behaves differently in your model, override it locally in a `r""" """` block. The local definition takes priority. The `labels` argument, for instance, is commonly customized per model and often needs an override.

- Standard config arguments (`vocab_size`, `hidden_size`, `num_hidden_layers`, etc.) follow the same principle but come from [`ConfigArgs`]. Standard processor arguments (`image_processor`, `tokenizer`, `do_resize`, `return_tensors`, etc.) come from [`ProcessorArgs`] and [`ImageProcessorArgs`]. Only document a parameter if it is model-specific or behaves differently from the standard description.

- Document new or custom arguments in an `r""" """` block after the signature for functions, in the `__init__` docstring for model or processor classes, in the class body docstring for config classes, or in the `XxxImageProcessorKwargs` TypedDict body for image processors.

    ```py
    argument_name (`type`, *optional*, defaults to `X`):
        Description of the argument.
        Explain its purpose, expected shape/type if complex, and default behavior.
        This can span multiple lines.
    ```

  * Include `type` in backticks.
  * Add *optional* if the argument is not required or has a default value.
  * Add "defaults to X" if it has a default value. You don't need to add "defaults to `None`" if the default value is `None`.

    Pass the same block into `@auto_docstring` as `custom_args` when the same arguments repeat across several places in the modeling file.

    ```py
    class MyModel(PreTrainedModel):
        # ...
        @auto_docstring(
            custom_intro="""
            This is a custom introduction for the function.
            """,
            custom_args=r"""
            common_arg_1 (`torch.Tensor`, *optional*, defaults to `default_value`):
                Description of common_arg_1
            """
        )
    ```

- Types are extracted from function signatures automatically. If a parameter has a type annotation, you don't need to repeat the type in the docstring format string. When both are present, the signature type takes precedence. The docstring type acts as a fallback for parameters that aren't annotated.

## Checking the docstrings

A utility script validates docstrings when you open a Pull Request. CI runs the script and checks the following criteria.

> [!TIP]
> If you see an `[ERROR]` in the output, add the parameter's description to the docstring or the appropriate Args class in `auto_docstring.py`.

* Ensures `@auto_docstring` is applied to relevant model classes and public methods.
* Ensures arguments are complete and consistent. It checks that documented arguments exist in the signature and that types and default values match. Unknown arguments without a local description are flagged.
* Reminds you to complete placeholders like `<fill_type>` and `<fill_docstring>`.
* Ensures docstrings are formatted according to the expected docstring style.

Run the check locally before committing.

```bash
make fix-repo
```

`make fix-repo` runs several other checks too. To run only the docstring and auto-docstring checks, use the command below.

```bash
python utils/check_docstrings.py # to only check files included in the diff without fixing them
# python utils/check_docstrings.py --fix_and_overwrite # to fix and overwrite the files in the diff
# python utils/check_docstrings.py --fix_and_overwrite --check_all # to fix and overwrite all files
```

## Quick-reference checklist

| Do | Don't |
|---|---|
| Apply `@auto_docstring` to model, config, and processor classes and their primary methods (`forward`, `__call__`, `preprocess`). | Add `@auto_docstring` in a modular file that inherits from a parent that already has it since it carries over automatically. |
| Document only new or model-specific arguments. | Redefine standard arguments (`input_ids`, `attention_mask`, `vocab_size`, etc.) unless the behavior differs. |
| Put config parameters in the class body docstring as class-level annotations. | Document config parameters in `__init__`. |
| Put image processor parameters in a `XxxImageProcessorKwargs` TypedDict. | Document image processor parameters in `__init__`. |
| Run `python utils/check_docstrings.py --fix_and_overwrite` before committing. | Ignore `[ERROR]` output because it means a parameter is undocumented. |

## How it works

The `@auto_docstring` decorator generates docstrings through the following steps.

1. It inspects the signature to read the arguments, types, and defaults from the decorated class's `__init__` or the decorated function. For config classes, it walks class-level annotations up the MRO and stops before [`PreTrainedConfig`], so base class fields are excluded.

    Parameters like `self`, `kwargs`, `args`, `deprecated_arguments`, and `_`-prefixed parameters are automatically filtered out. A few private parameters are renamed to their public equivalents (e.g., `_out_features` → `out_features` for backbone models).

2. Common argument descriptions come from `auto_docstring.py`: [`ModelArgs`] (model inputs), [`ModelOutputArgs`], [`ImageProcessorArgs`] (image preprocessing), [`ProcessorArgs`] (multimodal processor components), and [`ConfigArgs`] (config hyperparameters).

3. Each parameter resolves in this priority chain:

    - A manual docstring (`r""" """` block or `custom_args`) takes priority.
    - The predefined source dict ([`ModelArgs`], [`ConfigArgs`], [`ImageProcessorArgs`], [`ProcessorArgs`], [`ModelOutputArgs`]) is the fallback.
    - If neither source has a description, the parameter is flagged with `[ERROR]` in the build output.

4. For model classes with standard names like `ModelForCausalLM`, or classes that map to a pipeline, `@auto_docstring` generates the intro. For multimodal processors, the intro lists which components (tokenizer, image processor, and so on) the class wraps. See [ClassDocstring](https://github.com/huggingface/transformers/blob/1f553bdc1703c78e272656ab8fb86d6494593e18/src/transformers/utils/auto_docstring.py#L2437) for the full list.

    If the class name isn't in `ClassDocstring`, set `custom_intro`.

5. Predefined docstrings can reference dynamic values from Transformers' [auto_modules](https://github.com/huggingface/transformers/tree/main/src/transformers/models/auto), such as `{processor_class}`, `{image_processor_class}`, and `{config_class}`. These placeholders resolve automatically.

6. The decorator picks usage examples based on the model's task or pipeline compatibility. It reads checkpoint metadata from the configuration class so examples use real model IDs. The `checkpoint` argument overrides the auto-inferred checkpoint (normally from the config class's docstring). Use it for config classes, or when checkpoint inference fails. If you see an error like "Config not found for <model_name>", add an entry to `HARDCODED_CONFIG_FOR_MODELS` in `auto_docstring.py`.

7. For methods like `forward`, the decorator writes the `Returns` section from the method's return type. When the return type is a [`~transformers.utils.ModelOutput`] subclass, `@auto_docstring` pulls field descriptions from that class's docstring. A custom `Returns` block in the function's docstring takes precedence.

8. For methods in `UNROLL_KWARGS_METHODS` and classes in `UNROLL_KWARGS_CLASSES`, `**kwargs` typed with `Unpack[KwargsTypedDict]` are expanded. Each key from the `TypedDict` becomes a documented parameter.

    Supported typed kwargs include `TextKwargs`, `ImagesKwargs`, `VideosKwargs`, `AudioKwargs`, and `ProcessingKwargs`. It also supports the [`BaseImageProcessor`] and [`ProcessorMixin`] subclasses.
