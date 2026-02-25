# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for auto_docstring decorator and check_auto_docstrings function.
"""

import importlib
import os
import statistics
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

import torch

from transformers.configuration_utils import PretrainedConfig
from transformers.image_processing_utils import BatchFeature
from transformers.image_processing_utils_fast import BaseImageProcessorFast
from transformers.image_utils import ImageInput
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from transformers.testing_utils import require_torch
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils.auto_docstring import (
    auto_docstring,
)
from transformers.utils.import_utils import is_torch_available


if is_torch_available():
    import torch

_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "utils"))

from check_docstrings import (  # noqa: E402
    _build_ast_indexes,
    _find_typed_dict_classes,
    find_files_with_auto_docstring,
    update_file_with_new_docstrings,
)


class TestCheckDocstrings(unittest.TestCase):
    """Test check_auto_docstrings static analysis tool for detecting and fixing docstring issues."""

    def test_missing_args_detection_and_placeholder_generation(self):
        """Test that missing custom args are detected and placeholders generated while preserving Examples and code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "model.py")

            original = textwrap.dedent("""
                from transformers.utils.auto_docstring import auto_docstring

                @auto_docstring
                def forward(self, input_ids, custom_temperature: float = 1.0):
                    '''
                    Example:
                    ```python
                    >>> model.forward(input_ids, custom_temperature=0.7)
                    ```
                    '''
                    result = input_ids * custom_temperature
                    return result
            """)

            with open(test_file, "w") as f:
                f.write(original)

            with open(test_file, "r") as f:
                content = f.read()

            items = _build_ast_indexes(content)
            lines = content.split("\n")

            # Test detection (overwrite=False) - should detect missing arg
            missing, fill, redundant = update_file_with_new_docstrings(
                test_file, lines, items, content, overwrite=False
            )
            self.assertTrue(any("custom_temperature" in msg for msg in missing))

            # Generate placeholders (overwrite=True)
            update_file_with_new_docstrings(test_file, lines, items, content, overwrite=True)

            with open(test_file, "r") as f:
                updated = f.read()

            # Verify results
            self.assertIn("custom_temperature", updated)
            self.assertIn("<fill_docstring>", updated)  # Placeholder added
            self.assertIn("input_ids", updated)  # Standard arg from ModelArgs
            self.assertIn("Example:", updated)  # Example preserved
            self.assertIn("result = input_ids * custom_temperature", updated)  # Code preserved

    def test_multi_item_file_processing(self):
        """Test processing files with multiple @auto_docstring decorators (class + method) in a single pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "modeling.py")

            original = textwrap.dedent("""
                from transformers.utils.auto_docstring import auto_docstring
                from transformers.modeling_utils import PreTrainedModel

                @auto_docstring
                class MyModel(PreTrainedModel):
                    def __init__(self, config):
                        super().__init__(config)
                        self.layer = None

                    @auto_docstring
                    def forward(self, input_ids, scale_factor: float = 1.0):
                        '''
                        Example:
                        ```python
                        >>> outputs = model.forward(input_ids, scale_factor=2.0)
                        ```
                        '''
                        return self.layer(input_ids) * scale_factor
            """)

            with open(test_file, "w") as f:
                f.write(original)

            with open(test_file, "r") as f:
                content = f.read()

            items = _build_ast_indexes(content)

            # Should find 2 decorated items
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0].kind, "class")
            self.assertEqual(items[1].kind, "function")

            lines = content.split("\n")

            # Detect issues
            missing, fill, redundant = update_file_with_new_docstrings(
                test_file, lines, items, content, overwrite=False
            )

            # Should detect missing scale_factor in forward method
            self.assertTrue(any("scale_factor" in msg for msg in missing))

            # Update file
            update_file_with_new_docstrings(test_file, lines, items, content, overwrite=True)

            with open(test_file, "r") as f:
                updated = f.read()

            # Verify updates and preservation
            self.assertIn("scale_factor", updated)  # Custom arg added with placeholder
            self.assertIn("<fill_docstring>", updated)  # Placeholder present
            self.assertIn("Example:", updated)  # Example preserved
            self.assertIn("self.layer = None", updated)  # __init__ code preserved
            self.assertIn("return self.layer(input_ids) * scale_factor", updated)  # forward code preserved

    def test_typed_dict_field_detection(self):
        """Test that _find_typed_dict_classes correctly identifies custom fields vs standard inherited fields."""
        content = textwrap.dedent("""
            from typing import TypedDict
            from transformers.processing_utils import ImagesKwargs

            class CustomImageKwargs(ImagesKwargs, total=False):
                '''
                custom_mode (`str`):
                    Custom processing mode.
                '''
                # Standard field from ImagesKwargs - should be in all_fields but not fields
                do_resize: bool
                # Custom fields - should be in both all_fields and fields
                custom_mode: str
                undocumented_custom: int
        """)

        typed_dicts = _find_typed_dict_classes(content)

        # Should find the TypedDict
        self.assertEqual(len(typed_dicts), 1)
        self.assertEqual(typed_dicts[0]["name"], "CustomImageKwargs")

        # all_fields includes everything
        self.assertIn("do_resize", typed_dicts[0]["all_fields"])
        self.assertIn("custom_mode", typed_dicts[0]["all_fields"])
        self.assertIn("undocumented_custom", typed_dicts[0]["all_fields"])

        # fields only includes custom fields (not standard args like do_resize)
        # Both documented and undocumented custom fields are included
        self.assertIn("custom_mode", typed_dicts[0]["fields"])
        self.assertIn("undocumented_custom", typed_dicts[0]["fields"])
        self.assertNotIn("do_resize", typed_dicts[0]["fields"])  # Standard arg excluded

    def test_file_discovery_finds_decorated_files(self):
        """Test that check_auto_docstrings can discover files containing @auto_docstring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            has_decorator = os.path.join(tmpdir, "modeling.py")
            no_decorator = os.path.join(tmpdir, "utils.py")

            with open(has_decorator, "w") as f:
                f.write("@auto_docstring\ndef forward(self): pass")

            with open(no_decorator, "w") as f:
                f.write("def helper(): pass")

            found = find_files_with_auto_docstring([has_decorator, no_decorator])

            self.assertEqual(len(found), 1)
            self.assertEqual(found[0], has_decorator)


class DummyConfig(PretrainedConfig):
    model_type = "dummy_test"

    def __init__(self, vocab_size=1000, hidden_size=768, num_attention_heads=12, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads


@auto_docstring
class DummyForTestModel(PreTrainedModel):
    config_class = DummyConfig

    def __init__(self, config: DummyConfig):
        super().__init__(config)

    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        temperature: float = 1.0,
        custom_dict: dict[str, int | float] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> CausalLMOutputWithPast:
        r"""
        temperature (`float`, *optional*, defaults to 1.0):
            Temperature value for scaling logits during generation.
        custom_dict (`dict[str, Union[int, float]]`, *optional*):
            Custom dictionary parameter with string keys and numeric values.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DummyForTestModel
        >>> import torch

        >>> model = DummyForTestModel.from_pretrained("dummy-model")
        >>> tokenizer = AutoTokenizer.from_pretrained("dummy-model")
        >>> inputs = tokenizer("Hello world", return_tensors="pt")
        >>> outputs = model.forward(**inputs, temperature=0.7)
        >>> logits = outputs.logits
        ```
        """
        pass


class ComplexProcessorKwargs(ProcessingKwargs, total=False):
    r"""
    custom_processing_mode (`str`, *optional*, defaults to `"standard"`):
        Custom processing mode for advanced text/image processing. Can be 'standard', 'enhanced', or 'experimental'.
    enable_advanced_features (`bool`, *optional*, defaults to `False`):
        Whether to enable advanced processing features like custom tokenization strategies.
    custom_threshold (`float`, *optional*, defaults to 0.5):
        Custom threshold value for filtering or processing decisions.
    output_format (`str`, *optional*, defaults to `"default"`):
        Output format specification. Can be 'default', 'extended', or 'minimal'.
    """

    custom_processing_mode: str
    enable_advanced_features: bool
    custom_threshold: float
    output_format: str


@auto_docstring
class DummyProcessorForTest(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        custom_processing_mode="standard",
        enable_advanced_features=False,
        custom_threshold=0.5,
        output_format="default",
        **kwargs,
    ):
        r"""
        custom_processing_mode (`str`, *optional*, defaults to `"standard"`):
            Custom processing mode for advanced text/image processing. Can be 'standard', 'enhanced', or 'experimental'.
        enable_advanced_features (`bool`, *optional*, defaults to `False`):
            Whether to enable advanced processing features like custom tokenization strategies.
        custom_threshold (`float`, *optional*, defaults to 0.5):
            Custom threshold value for filtering or processing decisions.
        output_format (`str`, *optional*, defaults to `"default"`):
            Output format specification. Can be 'default', 'extended', or 'minimal'.
        """
        pass

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[ComplexProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Example:

        ```python
        >>> from transformers import DummyProcessorForTest
        >>> processor = DummyProcessorForTest.from_pretrained("dummy-processor")
        >>> inputs = processor(text="Hello world", images=["image.jpg"], return_tensors="pt")
        ```
        """
        pass


class DummyImageProcessorKwargs(ImagesKwargs, total=False):
    r"""
    image_grid_pinpoints (`list[list[int]]`, *optional*):
        A list of possible resolutions to use for processing high resolution images. The best resolution is selected
        based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
        method.
    custom_scale (`float`, *optional*, defaults to 255.0):
        Custom scale factor for preprocessing pipelines.
    """

    image_grid_pinpoints: list[list[int]]
    custom_scale: float


@auto_docstring(
    custom_intro="""
    Constructs a fast DummyForTest image processor.
    """
)
class DummyForTestImageProcessorFast(BaseImageProcessorFast):
    model_input_names = ["pixel_values"]
    valid_kwargs = DummyImageProcessorKwargs

    def __init__(self, **kwargs: Unpack[DummyImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        **kwargs: Unpack[DummyImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Example:

        ```python
        >>> from transformers import DummyForTestImageProcessorFast
        >>> from PIL import Image
        >>> import requests

        >>> processor = DummyForTestImageProcessorFast.from_pretrained("dummy-processor")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor.preprocess(images=image, return_tensors="pt")
        ```
        """
        pass


@require_torch
class TestFullDocstringGeneration(unittest.TestCase):
    """
    End-to-end tests for @auto_docstring runtime docstring generation.

    Tests validate complete docstrings with single assertEqual assertions to ensure structure,
    formatting, standard args, custom params, and TypedDict unrolling work correctly.
    """

    def test_dummy_model_complete_docstring(self):
        self.maxDiff = None
        """Test complete class and forward method docstrings for PreTrainedModel with ModelArgs and custom parameters."""
        actual_class_docstring = DummyForTestModel.__doc__
        expected_class_docstring = """
This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

Parameters:
    config ([`DummyConfig`]):
        Model configuration class with all the parameters of the model. Initializing with a config file does not
        load the weights associated with the model, only the configuration. Check out the
        [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
        self.assertEqual(actual_class_docstring, expected_class_docstring)

        actual_docstring = DummyForTestModel.forward.__doc__

        expected_docstring = """        The [`DummyForTestModel`] forward method, overrides the `__call__` special method.

        <Tip>

        Although the recipe for forward pass needs to be defined within this function, one should call the [`Module`]
        instance afterwards instead of this since the former takes care of running the pre and post processing steps while
        the latter silently ignores them.

        </Tip>

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.

                [What are position IDs?](../glossary#position-ids)
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
                is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
                model's internal embedding lookup matrix.
            temperature (`float`, *optional*, defaults to 1.0):
                Temperature value for scaling logits during generation.
            custom_dict (`dict[str, Union[int, float]]`, *optional*):
                Custom dictionary parameter with string keys and numeric values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:
            [`~modeling_outputs.CausalLMOutputWithPast`] or `tuple(torch.FloatTensor)`: A [`~modeling_outputs.CausalLMOutputWithPast`] or a tuple of
            `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
            elements depending on the configuration ([`None`]) and inputs.

        - **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
        - **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        - **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

          Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
          `past_key_values` input) to speed up sequential decoding.
        - **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
          one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

          Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        - **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
          sequence_length)`.

          Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
          heads.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DummyForTestModel
        >>> import torch

        >>> model = DummyForTestModel.from_pretrained("dummy-model")
        >>> tokenizer = AutoTokenizer.from_pretrained("dummy-model")
        >>> inputs = tokenizer("Hello world", return_tensors="pt")
        >>> outputs = model.forward(**inputs, temperature=0.7)
        >>> logits = outputs.logits
        ```
"""

        self.assertEqual(actual_docstring, expected_docstring)

    def test_dummy_processor_complete_docstring(self):
        self.maxDiff = None
        """Test complete class and __call__ docstrings for ProcessorMixin with complex TypedDict kwargs unrolling."""

        actual_docstring = DummyProcessorForTest.__call__.__doc__

        expected_docstring = """        Args:
            images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`, *optional*):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            text (`Union[str, list[str], list[list[str]]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.
            custom_processing_mode (`str`, *kwargs*, *optional*, defaults to `"standard"`):
                Custom processing mode for advanced text/image processing. Can be 'standard', 'enhanced', or 'experimental'.
            enable_advanced_features (`bool`, *kwargs*, *optional*, defaults to `False`):
                Whether to enable advanced processing features like custom tokenization strategies.
            custom_threshold (`float`, *kwargs*, *optional*, defaults to 0.5):
                Custom threshold value for filtering or processing decisions.
            output_format (`str`, *kwargs*, *optional*, defaults to `"default"`):
                Output format specification. Can be 'default', 'extended', or 'minimal'.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
            **kwargs ([`ProcessingKwargs`], *optional*):
                Additional processing options for each modality (text, images, videos, audio). Model-specific parameters
                are listed above; see the TypedDict class for the complete list of supported arguments.

        Returns:
            `~image_processing_base.BatchFeature`:
            - **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
            - **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
              initialization.

        Example:

        ```python
        >>> from transformers import DummyProcessorForTest
        >>> processor = DummyProcessorForTest.from_pretrained("dummy-processor")
        >>> inputs = processor(text="Hello world", images=["image.jpg"], return_tensors="pt")
        ```
"""

        self.assertEqual(actual_docstring, expected_docstring)

        actual_class_docstring = DummyProcessorForTest.__doc__

        expected_class_docstring = """Constructs a DummyProcessorForTest which wraps a image processor and a tokenizer into a single processor.

[`DummyProcessorForTest`] offers all the functionalities of [`image_processor_class`] and [`tokenizer_class`]. See the
[`~image_processor_class`] and [`~tokenizer_class`] for more information.
Parameters:
    image_processor (`image_processor_class`):
        The image processor is a required input.
    tokenizer (`tokenizer_class`):
        The tokenizer is a required input.
    custom_processing_mode (`str`, *optional*, defaults to `"standard"`):
        Custom processing mode for advanced text/image processing. Can be 'standard', 'enhanced', or 'experimental'.
    enable_advanced_features (`bool`, *optional*, defaults to `False`):
        Whether to enable advanced processing features like custom tokenization strategies.
    custom_threshold (`float`, *optional*, defaults to 0.5):
        Custom threshold value for filtering or processing decisions.
    output_format (`str`, *optional*, defaults to `"default"`):
        Output format specification. Can be 'default', 'extended', or 'minimal'.
"""

        self.assertEqual(actual_class_docstring, expected_class_docstring)

    def test_dummy_image_processor_complete_docstring(self):
        self.maxDiff = None
        """Test complete class and preprocess docstrings for BaseImageProcessorFast with custom ImagesKwargs and custom_intro."""

        actual_preprocess_docstring = DummyForTestImageProcessorFast.preprocess.__doc__

        expected_preprocess_docstring = """        Args:
            images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            image_grid_pinpoints (`list[list[int]]`, *kwargs*, *optional*):
                A list of possible resolutions to use for processing high resolution images. The best resolution is selected
                based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
                method.
            custom_scale (`float`, *kwargs*, *optional*, defaults to 255.0):
                Custom scale factor for preprocessing pipelines.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.
            **kwargs ([`ImagesKwargs`], *optional*):
                Additional image preprocessing options. Model-specific kwargs are listed above; see the TypedDict class
                for the complete list of supported arguments.

        Returns:
            `~image_processing_base.BatchFeature`:
            - **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
            - **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
              initialization.

        Example:

        ```python
        >>> from transformers import DummyForTestImageProcessorFast
        >>> from PIL import Image
        >>> import requests

        >>> processor = DummyForTestImageProcessorFast.from_pretrained("dummy-processor")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor.preprocess(images=image, return_tensors="pt")
        ```
"""

        self.assertEqual(actual_preprocess_docstring, expected_preprocess_docstring)

        actual_class_docstring = DummyForTestImageProcessorFast.__doc__

        expected_class_docstring = """
Constructs a fast DummyForTest image processor.

Args:
    image_grid_pinpoints (`list[list[int]]`, *kwargs*, *optional*):
        A list of possible resolutions to use for processing high resolution images. The best resolution is selected
        based on the original size of the image. Can be overridden by `image_grid_pinpoints` in the `preprocess`
        method.
    custom_scale (`float`, *kwargs*, *optional*, defaults to 255.0):
        Custom scale factor for preprocessing pipelines.
    **kwargs ([`ImagesKwargs`], *optional*):
        Additional image preprocessing options. Model-specific kwargs are listed above; see the TypedDict class
        for the complete list of supported arguments.
"""

        self.assertEqual(actual_class_docstring, expected_class_docstring)


# ---------------------------------------------------------------------------
# Performance tests for auto_docstring
# ---------------------------------------------------------------------------


class TestAutoDocstringPerformance:
    """
    Performance tests for auto_docstring.

    The decorator runs at *class-definition / import time*, so with hundreds of
    models in the library the cumulative cost matters even though each individual
    call looks cheap.  These tests assert an upper bound to catch regressions.
    """

    # Upper bound (%) of total import time that auto_docstring overhead may take.
    # Relative metric; robust across CI vs local. Catches serious regressions.
    AUTO_DOCSTRING_COST_PCT_UPPER_BOUND = 70.0

    def test_auto_docstring_import_time_upper_bound(self):
        """
        Asserts that auto_docstring overhead stays below a percentage of total
        import time.

        Method
        ------
        1. Collect ``modeling_*.py``, ``image_processing_*.py``, ``processing_*.py``
           under ``transformers/models``, then sample every 10th for speed.
        2. Warmup: import the sampled modules once so Python's bytecode cache is hot.
        3. Measure WITH auto_docstring: clear cache, re-import, median over 5 runs.
        4. Measure WITHOUT auto_docstring: noop-patch, clear cache, re-import, median.
        5. cost_pct = (real - noop) / real * 100; assert cost_pct < upper bound.
        """
        if "transformers.utils" not in sys.modules:
            importlib.import_module("transformers.utils")
        _utils_module = sys.modules["transformers.utils"]

        src_root = Path(__file__).resolve().parent.parent.parent / "src"
        models_dir = src_root / "transformers" / "models"
        all_modules: list[str] = []
        for pattern in ("modeling_*.py", "image_processing_*.py", "processing_*.py"):
            for f in sorted(models_dir.rglob(pattern)):
                rel = f.with_suffix("").relative_to(src_root)
                all_modules.append(".".join(rel.parts))
        model_modules = all_modules[::10]

        def _clear():
            for key in [k for k in sys.modules if k.startswith("transformers.models")]:
                del sys.modules[key]

        def _import_all():
            for mod in model_modules:
                try:
                    importlib.import_module(mod)
                except Exception:
                    continue

        _import_all()  # warmup

        # With auto_docstring (real)
        times_real: list[float] = []
        for _ in range(5):
            _clear()
            t0 = time.perf_counter()
            _import_all()
            times_real.append(time.perf_counter() - t0)

        # Without auto_docstring (noop patch)
        _orig = _utils_module.auto_docstring
        _noop = lambda x=None, **kw: (lambda f: f) if x is None else x  # noqa: E731
        times_noop: list[float] = []
        for _ in range(5):
            _utils_module.auto_docstring = _noop
            try:
                _clear()
                t0 = time.perf_counter()
                _import_all()
                times_noop.append(time.perf_counter() - t0)
            finally:
                _utils_module.auto_docstring = _orig

        median_real = statistics.median(times_real)
        median_noop = statistics.median(times_noop)
        cost_pct = (median_real - median_noop) / median_real * 100 if median_real > 0 else 0.0
        print(f"Cost percentage: {cost_pct:.1f}%")
        assert cost_pct < self.AUTO_DOCSTRING_COST_PCT_UPPER_BOUND, (
            f"auto_docstring cost {cost_pct:.1f}% of import time exceeds upper bound "
            f"{self.AUTO_DOCSTRING_COST_PCT_UPPER_BOUND}% "
            f"({len(model_modules)} of {len(all_modules)} modules)"
        )
