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

The `@auto_docstring` decorator in Transformers generates consistent docstrings for model classes and their methods. It reduces boilerplate by automatically including standard argument descriptions while also allowing overrides to add new or custom arguments. [Contributing a new model](./modular_transformers) is easier because you don't need to manually add the standard docstrings, and only focus on documenting new arguments.

This guide describes how to use the `@auto_docstring` decorator and how it works.

## @auto_docstring

Start by importing the decorator in the modeling file (`modular_model.py` or `modeling_model.py`).

```python
from ...utils import auto_docstring
```

Select whether you'd like to apply `@auto_docstring` to a class or function below to see how to use it.

<hfoptions id="type">
<hfoption id="model classes">

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

Arguments can also be passed directly to `@auto_docstring` for more control. Use the `custom_intro` parameter to describe the argument and the `custom_args` parameter to describe the arguments.

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

You can also choose to only use `custom_intro` and define the custom arguments directly in the class.

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

You should also use the `@auto_docstring` decorator for classes that inherit from [`~utils.ModelOutput`].

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

    # Standard fields like hidden_states, logits, attentions etc. can be automatically documented if the description is the same as the standard arguments.
    # However, given that the loss docstring is often different per model, you should document it in the docstring above.
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    # Custom fields need to be documented in the docstring above
    custom_field: Optional[torch.FloatTensor] = None
```

</hfoption>
<hfoption id="config classes">

Place `@auto_docstring` directly above a `PreTrainedConfig` subclass, alongside `@strict` from `huggingface_hub.dataclasses`. Config parameters are declared as **class-level annotations** (not as `__init__` arguments) — the `@strict` dataclass pattern used throughout Transformers. The class docstring documents model-specific parameters and optionally a usage example.

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

Standard parameters like `vocab_size`, `hidden_size`, and `num_hidden_layers` are sourced from `ConfigArgs` and don't need a description unless the behavior differs. `PreTrainedConfig` base parameters are automatically excluded. The `checkpoint` argument is used to generate the usage example.

</hfoption>
<hfoption id="processor classes">

**Multimodal processors** (`ProcessorMixin` subclasses, `processing_*.py`) always use bare `@auto_docstring`. The class intro is auto-generated. Only document `__init__` parameters not already covered by `ProcessorArgs` (`image_processor`, `tokenizer`, `chat_template`, etc.) — omit the docstring entirely if all parameters are standard. Decorate `__call__` with `@auto_docstring` as well; its body docstring contains only a `Returns:` section plus any extra model-specific call arguments. `return_tensors` is automatically appended.

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

---

**Image and video processors** (`BaseImageProcessor` subclasses, `image_processing_*.py`) follow one of two patterns.

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

</hfoption>
<hfoption id="functions">

Place `@auto_docstring` directly above the method definition. The decorator derives parameter descriptions from the function signature.

```python
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        new_custom_argument: Optional[torch.Tensor] = None,
        arg_documented_in_args_doc: Optional[torch.Tensor] = None,
        # ... other arguments
    ) -> Union[Tuple, ModelOutput]: # The description of the return value will automatically be generated from the ModelOutput class docstring.
        r"""
        new_custom_argument (`torch.Tensor`, *optional*):
            Description of this new custom argument and its expected shape or type.
        """
        # ...
```

Arguments can also be passed directly to `@auto_docstring` for more control. Use the `custom_intro` parameter to describe the argument and the `custom_args` parameter to describe the arguments.

The `Returns` and `Examples` parts of the docstring can also be manually specified.

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
        """
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

</hfoption>
</hfoptions>

## Documenting arguments

There are some rules for documenting different types of arguments and they're listed below.

- Standard arguments (`input_ids`, `attention_mask`, `pixel_values`, etc.) are defined and retrieved from `auto_docstring.py`. It is the single source of truth for standard arguments and should not be redefined locally if an argument's description and shape is the same as an argument in `auto_docstring.py`.

    If a standard argument behaves differently in your model, then you can override it locally in a `r""" """` block. This local definition has a higher priority. For example, the `labels` argument is often customized per model and typically requires overriding.

- Standard **config** arguments (`vocab_size`, `hidden_size`, `num_hidden_layers`, etc.) follow the same principle but are sourced from `ConfigArgs`. Standard **processor** arguments (`image_processor`, `tokenizer`, `do_resize`, `return_tensors`, etc.) are sourced from `ProcessorArgs` and `ImageProcessorArgs`. Only document a parameter if it is model-specific or behaves differently from the standard description.

- New or custom arguments should be documented within an `r""" """` block after the signature if it is a function, in the `__init__` method's docstring if it is a model or processor class, in the **class body docstring** if it is a config class, or in the `XxxImageProcessorKwargs` TypedDict body if it is an image processor.

    ```py
    argument_name (`type`, *optional*, defaults to `X`):
        Description of the argument.
        Explain its purpose, expected shape/type if complex, and default behavior.
        This can span multiple lines.
    ```

  * Include `type` in backticks.
  * Add *optional* if the argument is not required or has a default value.
  * Add "defaults to X" if it has a default value. You don't need to add "defaults to `None`" if the default value is `None`.

    These arguments can also be passed to `@auto_docstring` as a `custom_args` argument. It is used to define the docstring block for new arguments once if they are repeated in multiple places in the modeling file.

    ```py
    class MyModel(PreTrainedModel):
    # ...
    @auto_docstring(
        custom_intro="""
        This is a custom introduction for the function.
        """
        custom_args=r"""
        common_arg_1 (`torch.Tensor`, *optional*, defaults to `default_value`):
            Description of common_arg_1
        """
    )
    ```

## Checking the docstrings

Transformers includes a utility script to validate the docstrings when you open a Pull Request which triggers CI (continuous integration) checks. The script checks for the following criteria.

* Ensures `@auto_docstring` is applied to relevant mode classes and public methods.
* Ensures arguments are complete and consistent. It checks that documented arguments exist in the signature and verifies whether the types and default values in the docstring match the signature. Arguments that aren't known standard arguments or if they lack a local description are flagged.
* Reminds you to complete placeholders like `<fill_type>` and `<fill_docstring>`.
* Ensures docstrings are formatted according to the expected docstring style.

You can run this check locally - before committing - by running the following command.

```bash
make fix-repo
```

`make fix-repo` runs several other checks as well. If you don't need those checks, run the command below to only perform docstring and auto-docstring checks.

```bash
python utils/check_docstrings.py # to only check files included in the diff without fixing them
# python utils/check_docstrings.py --fix_and_overwrite # to fix and overwrite the files in the diff
# python utils/check_docstrings.py --fix_and_overwrite --check_all # to fix and overwrite all files
```

## modular_model.py files

When working with modular files (`modular_model.py`), follow the guidelines below for applying `@auto_docstring`.

- For standalone models in modular files, apply `@auto_docstring` like you would in a `modeling_model.py` file.
- For models that inherit from other library models, `@auto_docstring` is automatically carried over to the generated modeling file. You don't need to add `@auto_docstring` in your modular file.

    If you need to modify the `@auto_docstring` behavior, apply the customized decorator in your modular file. Make sure to **include all other decorators** that are present in the original function or class.

> [!WARNING]
> When overriding any decorator in a modular file, you must include **all** decorators that were applied to that function or class in the parent model. If you only override some decorators, the others won't be included in the generated modeling file.

## How it works

The `@auto_docstring` decorator automatically generates docstrings by:

1. Inspecting the signature (arguments, types, defaults) of the decorated class' `__init__` method or the decorated function. For config classes, parameters are collected from class-level annotations up the MRO, stopping before `PreTrainedConfig`, so base-class fields are automatically excluded.
2. Retrieving predefined docstrings for common arguments from `auto_docstring.py`: [`ModelArgs`] (model inputs), [`ImageProcessorArgs`] (image preprocessing), [`ProcessorArgs`] (multimodal processor components), and [`ConfigArgs`] (config hyperparameters).
3. Adding argument descriptions in one of two ways as shown below.

    | method | description | usage |
    |---|---|---|
    | `r""" """` | add custom docstring content directly to a method signature or within the `__init__` docstring | document new arguments or override standard descriptions |
    | `custom_args` | add custom docstrings for specific arguments directly in `@auto_docstring` | define docstring for new arguments once if they're repeated in multiple places in the modeling file |

4. Adding class and function descriptions. For model classes with standard naming patterns like `ModelForCausalLM`, or if the class belongs to a pipeline, `@auto_docstring` automatically generates the appropriate description. For multimodal processor classes, the intro is auto-generated describing which components (tokenizer, image processor, etc.) it wraps. `@auto_docstring` also accepts `custom_intro` to override or supply a description.

5. Using a templating system to allow predefined docstrings to include dynamic information from Transformers' [auto_modules](https://github.com/huggingface/transformers/tree/main/src/transformers/models/auto) such as `{{processor_class}}` and `{{config_class}}`.

6. Finding appropriate usage examples based on the model's task or pipeline compatibility. It extracts checkpoint information form the model's configuration class to provide concrete examples with real model identifiers.

7. Adding return values to the docstring. For methods like `forward`, the decorator automatically generates the `Returns` field in the docstring based on the method's return type annotation.

    For example, if a method returns a [`~transformers.utils.ModelOutput`] subclass, `@auto_docstring` extracts the field descriptions from the class' docstring to create a comprehensive return value description. You can also manually specify a custom `Returns` field in a functions docstring.

8. Unrolling kwargs typed with the unpack operator. For specific methods (defined in `UNROLL_KWARGS_METHODS`) or classes (defined in `UNROLL_KWARGS_CLASSES`), the decorator processes `**kwargs` parameters that are typed with `Unpack[KwargsTypedDict]`. It extracts the documentations from the `TypedDict` and adds each parameter to the function's docstring.

    Currently only supported for [`ImagesKwargs`].

## Best practices

Follow the best practices below to help maintain consistent and informative documentation for Transformers!

* Use `@auto_docstring` for [`PreTrainedModel`] subclasses and their primary methods (`forward`, `get_text_features`, etc.), for [`PreTrainedConfig`] subclasses (alongside `@strict`), and for processor classes ([`ProcessorMixin`] and [`BaseImageProcessor`] subclasses) and their primary methods (`__call__`, `preprocess`).
* Do not redefine standard arguments unless their behavior differs in your model.
* For config classes, document parameters in the class body docstring (class-level annotations, not `__init__`). For image processors, document custom parameters in a `XxxImageProcessorKwargs` TypedDict rather than in `__init__`.
* Run `check_docstrings` locally and iteratively.
