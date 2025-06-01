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

# Utilizing the @auto_docstring Decorator

The `@auto_docstring` decorator in the Hugging Face Transformers library helps generate docstrings for model classes and their methods, which will be used to build the documentation for the library. It aims to improve consistency and reduce boilerplate by automatically including standard argument descriptions and allowing for targeted overrides and additions.

---

## 📜 How it Works

The `@auto_docstring` decorator constructs docstrings by:

1.  **Signature Inspection:** It inspects the signature (arguments, types, defaults) of the decorated class's `__init__` method or the decorated function.
2.  **Centralized Docstring Fetching:** It retrieves predefined docstrings for common arguments (e.g., `input_ids`, `attention_mask`) from internal library sources (like `ModelArgs` or `ImageProcessorArgs` in `utils/args_doc.py`).
3.  **Overriding or Adding Arguments Descriptions:**
    * **Direct Docstring Block:** It incorporates custom docstring content from an `r""" """` (or `""" """`) block below the method signature or within the `__init__` docstring. This is for documenting new arguments or overriding standard descriptions.
    * **Decorator Arguments (`custom_args`):** A `custom_args` docstring block can be passed to the decorator to provide docstrings for specific arguments directly in the decorator call. This can be used to define the docstring block for new arguments once if they are repeated in multiple places in the modeling file.
4.  **Adding Classes and Functions Introduction:**
    * **`custom_intro` argument:** Allows prepending a custom introductory paragraph to a class or function docstring.
    * **Automatic Introduction Generation:** For model classes with standard naming patterns (like `ModelForCausalLM`) or belonging to a pipeline, the decorator automatically generates an appropriate introductory paragraph using `ClassDocstring` in `utils/args_doc.py` as the source.
5.  **Templating:** The decorator uses a templating system, allowing predefined docstrings to include dynamic information deduced from the `auto_modules` of the library, such as `{{processor_class}}` or `{{config_class}}`.
6.  **Deducing Relevant Examples:** The decorator attempts to find appropriate usage examples based on the model's task or pipeline compatibility. It extracts checkpoint information from the model's configuration class to provide concrete examples with real model identifiers.
7.  **Adding Return Value Documentation:** For methods like `forward`, the decorator can automatically generate the "Returns" section based on the method's return type annotation. For example, for a method returning a `ModelOutput` subclass, it will extracts field descriptions from that class's docstring to create a comprehensive return value description. A custom `Returns` section can also be manually specified in the function docstring block.
8.  **Unrolling Kwargs Typed With Unpack Operator:** For specific methods (defined in `UNROLL_KWARGS_METHODS`) or classes (defined in `UNROLL_KWARGS_CLASSES`), the decorator processes `**kwargs` parameters that are typed with `Unpack[KwargsTypedDict]`. It extracts the documentation from the TypedDict and adds each parameter to the function's docstring. Currently, this functionality is only supported for `FastImageProcessorKwargs`.


---

## 🚀 How to Use @auto_docstring

### 1. Importing the Decorator
Import the decorator into your modeling file:

```python
from ...utils import auto_docstring
```

### 2. Applying to Classes
Place `@auto_docstring` directly above the class definition. It uses the `__init__` method's signature and its docstring for parameter descriptions.

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

#### Advanced Class Decoration:

Arguments can be passed directly to `@auto_docstring` for more control:

```python
@auto_docstring(
    custom_intro="""This model performs specific synergistic operations.
    It builds upon the standard Transformer architecture with unique modifications.""",
    custom_args="""
    custom_parameter (`type`, *optional*, defaults to `default_value`):
        A concise description for custom_parameter if not defined or overriding the description in `args_doc.py`.
    internal_helper_arg (`type`, *optional*, defaults to `default_value`):
        A concise description for internal_helper_arg if not defined or overriding the description in `args_doc.py`.
    """
)
class MySpecialModel(PreTrainedModel):
    def __init__(self, config: ConfigType, custom_parameter: "type" = "default_value", internal_helper_arg=None):
        # ...
```

Or:

```python
@auto_docstring(
    custom_intro="""This model performs specific synergistic operations.
    It builds upon the standard Transformer architecture with unique modifications.""",
)
class MySpecialModel(PreTrainedModel):
    def __init__(self, config: ConfigType, custom_parameter: "type" = "default_value", internal_helper_arg=None):
        r"""
        custom_parameter (`type`, *optional*, defaults to `default_value`):
            A concise description for custom_parameter if not defined or overriding the description in `args_doc.py`.
        internal_helper_arg (`type`, *optional*, defaults to `default_value`):
            A concise description for internal_helper_arg if not defined or overriding the description in `args_doc.py`.
        """
        # ...
```

### 3. Applying to Functions (e.g., `forward` method)
Apply the decorator above method definitions, such as the `forward` method.

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

#### Advanced Function Decoration:

Arguments can be passed directly to `@auto_docstring` for more control. `Returns` and `Examples` sections can also be manually specified:

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

---

### ✍️ Documenting Arguments: Approach & Priority

1.  **Standard Arguments (e.g., `input_ids`, `attention_mask`, `pixel_values`, `encoder_hidden_states` etc.):**
    * `@auto_docstring` retrieves descriptions from a central source. Do not redefine these locally if their description and shape are the same as in `args_doc.py`.

2.  **New or Custom Arguments:**
    * **Primary Method:** Document these within an `r""" """` docstring block following the signature (for functions) or in the `__init__` method's docstring (for class parameters).
    * **Format:**
        ```
        argument_name (`type`, *optional*, defaults to `X`):
            Description of the argument.
            Explain its purpose, expected shape/type if complex, and default behavior.
            This can span multiple lines.
        ```
    * Include `type` in backticks.
    * Add "*optional*" if the argument is not required (has a default value).
    * Add "defaults to `X`" if it has a default value (no need to specify "defaults to `None`" if the default value is `None`).

3.  **Overriding Standard Arguments:**
    * If a standard argument behaves differently (e.g., different expected shape, model-specific behavior), provide its complete description in the local `r""" """` docstring. This local definition takes precedence.
    * The `labels` argument is often customized per model and typically requires a specific docstring.

4.  **Using Decorator Arguments for Overrides or New Arguments (`custom_args`):**
    * New or custom arguments docstrings can also be passed to `@auto_docstring` as a `custom_args` argument. This can be used to define the docstring block for new arguments once if they are repeated in multiple places in the modeling file.

---

### Usage with [modular files](./modular_transformers)

When working with modular files, follow these guidelines for applying the `@auto_docstring` decorator:

- **For standalone models in modular files:**
  Apply the `@auto_docstring` decorator just as you would in regular modeling files.

- **For models inheriting from other library models:**
  - When inheriting from a parent model, decorators (including `@auto_docstring`) are automatically carried over to the generated modeling file without needing to add them in your modular file.
  - If you need to modify the `@auto_docstring` behavior, apply the customized decorator in your modular file, making sure to *include all other decorators* that were present on the original function/class.

  > **Warning**: When overriding any decorator in a modular file, you must include ALL decorators that were applied to that function/class in the parent model. If you only override some decorators, the others won't be included in the generated modeling file.


**Note**: The `check_auto_docstrings` tool doesn't check modular files directly, but it will check (and modify when using `--fix_and_overwrite`) the generated modeling files. If issues are found in the generated files, you'll need to update your modular files accordingly.

---

## ✅ Checking Your Docstrings with `check_auto_docstrings`

The library includes a utility script to validate docstrings. This check is typically run during Continuous Integration (CI).

#### What it Checks:

* **Decorator Presence:** Ensures `@auto_docstring` is applied to relevant model classes and public methods. (TODO)
* **Argument Completeness & Consistency:**
    * Flags arguments in the signature that are not known standard arguments and lack a local description.
    * Ensures documented arguments exist in the signature. (TODO)
    * Verifies that types and default values in the docstring match the signature. (TODO)
* **Placeholder Detection:** Reminds you to complete placeholders like `<fill_type>` or `<fill_docstring>`.
* **Formatting:** Adherence to the expected docstring style.

#### Running the Check Locally:

Run this check locally before committing. The common command is:

```bash
make fix-copies
```

Alternatively, to only perform docstrings and auto-docstring checks, you can use:

```bash
python utils/check_docstrings.py # to only check files included in the diff without fixing them
# Or: python utils/check_docstrings.py --fix_and_overwrite # to fix and overwrite the files in the diff
# Or: python utils/check_docstrings.py --fix_and_overwrite --check_all # to fix and overwrite all files
```

#### Workflow with the Checker:

1.  Add `@auto_docstring(...)` to the class or method.
2.  For new, custom, or overridden arguments, add descriptions in an `r""" """` block.
3.  Run `make fix-copies` (or the `check_docstrings.py` utility).
    * For unrecognized arguments lacking documentation, the utility will create placeholder entries.
4.  Manually edit these placeholders with accurate types and descriptions.
5.  Re-run the check to ensure all issues are resolved.

---

## 🔑 Key Takeaways & Best Practices

* Use `@auto_docstring` for new PyTorch model classes (`PreTrainedModel` subclasses) and their primary for methods (e.g., `forward`, `get_text_features` etc.).
* For classes, the `__init__` method's docstring is the main source for parameter descriptions when using `@auto_docstring` on the class.
* Rely on standard docstrings; do not redefine common arguments unless their behavior is different in your specific model.
* Document new or custom arguments clearly.
* Run `check_docstrings` locally and iteratively.

By following these guidelines, you help maintain consistent and informative documentation for the Hugging Face Transformers library 🤗.
