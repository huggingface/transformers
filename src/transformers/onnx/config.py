# Copyright 2021 The HuggingFace Team. All rights reserved.
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
import copy
import dataclasses
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
from packaging import version

from ..utils import TensorType, is_torch_available, is_vision_available, logging
from .utils import ParameterFormat, compute_effective_axis_dimension, compute_serialized_parameters_size


if TYPE_CHECKING:
    from ..configuration_utils import PretrainedConfig
    from ..feature_extraction_utils import FeatureExtractionMixin
    from ..tokenization_utils_base import PreTrainedTokenizerBase


if is_vision_available():
    from PIL import Image

logger = logging.get_logger(__name__)


DEFAULT_ONNX_OPSET = 11

# 2 Gb
EXTERNAL_DATA_FORMAT_SIZE_LIMIT = 2 * 1024 * 1024 * 1024


@dataclasses.dataclass
class PatchingSpec:
    """
    Data class that holds patching specifications.

    Args:
        o: Module / object where the op to patch is located
        name: Name of the op to monkey patch
        custom_op: Custom op that patches the original op
        orig_op: Original op that is being patched
        op_wrapper: Wrapper (optional) that wraps both the original and custom ops.
            It is useful for ops that are class or static methods for instance.
    """

    o: Any
    name: str
    custom_op: Callable
    orig_op: Optional[Callable] = None
    op_wrapper: Optional[Callable] = None


class OnnxConfig(ABC):
    """
    Base class for ONNX exportable model describing metadata on how to export the model through the ONNX format.
    """

    default_fixed_batch = 2
    default_fixed_sequence = 8
    default_fixed_num_choices = 4
    torch_onnx_minimum_version = version.parse("1.8")
    _tasks_to_common_outputs = {
        "default": OrderedDict({"last_hidden_state": {0: "batch", 1: "sequence"}}),
        "masked-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "causal-lm": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "seq2seq-lm": OrderedDict({"logits": {0: "batch", 1: "decoder_sequence"}}),
        "sequence-classification": OrderedDict({"logits": {0: "batch"}}),
        "token-classification": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
        "multiple-choice": OrderedDict({"logits": {0: "batch"}}),
        "question-answering": OrderedDict(
            {
                "start_logits": {0: "batch", 1: "sequence"},
                "end_logits": {0: "batch", 1: "sequence"},
            }
        ),
        "image-classification": OrderedDict({"logits": {0: "batch", 1: "sequence"}}),
    }

    def __init__(self, config: "PretrainedConfig", task: str = "default", patching_specs: List[PatchingSpec] = None):
        self._config = config

        if task not in self._tasks_to_common_outputs:
            raise ValueError(
                f"{task} is not a supported task, supported tasks: {self._tasks_to_common_outputs.keys()}"
            )
        self.task = task

        self._patching_specs = []
        for spec in patching_specs if patching_specs is not None else []:
            final_spec = spec
            if spec.orig_op is None:
                final_spec = dataclasses.replace(spec, orig_op=getattr(spec.o, spec.name))
            self._patching_specs.append(final_spec)

    @classmethod
    def from_model_config(cls, config: "PretrainedConfig", task: str = "default") -> "OnnxConfig":
        """
        Instantiate a OnnxConfig for a specific model

        Args:
            config: The model's configuration to use when exporting to ONNX

        Returns:
            OnnxConfig for this model
        """
        return cls(config, task=task)

    @property
    @abstractmethod
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the input tensors to provide to the model

        Returns:
            For each input: its name associated to the axes symbolic name and the axis position within the tensor
        """
        raise NotImplementedError()

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        """
        Mapping containing the axis definition of the output tensors to provide to the model

        Returns:
            For each output: its name associated to the axes symbolic name and the axis position within the tensor
        """
        common_outputs = self._tasks_to_common_outputs[self.task]
        return copy.deepcopy(common_outputs)

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        """
        Dictionary of keys to override in the model's config before exporting

        Returns:
            Dictionary with the keys (and their corresponding values) to override
        """
        if hasattr(self._config, "use_cache"):
            return {"use_cache": False}

        return None

    @property
    def default_batch_size(self) -> int:
        """
        The default batch size to use if no other indication

        Returns:
            Integer > 0
        """
        # Using 2 avoid ONNX making assumption about single sample batch
        return OnnxConfig.default_fixed_batch

    @property
    def default_sequence_length(self) -> int:
        """
        The default sequence length to use if no other indication

        Returns:
            Integer > 0
        """
        return OnnxConfig.default_fixed_sequence

    @property
    def default_num_choices(self) -> int:
        """
        The default number of choices to use if no other indication

        Returns:
            Integer > 0
        """
        return OnnxConfig.default_fixed_num_choices

    @property
    def default_onnx_opset(self) -> int:
        """
        Which onnx opset to use when exporting the model

        Returns:
            Integer ONNX Opset version
        """
        return DEFAULT_ONNX_OPSET

    @property
    def atol_for_validation(self) -> float:
        """
        What absolute tolerance value to use during model conversion validation.

        Returns:
            Float absolute tolerance value.
        """
        return 1e-5

    @property
    def is_torch_support_available(self) -> bool:
        """
        The minimum PyTorch version required to export the model.

        Returns:
            `bool`: Whether the installed version of PyTorch is compatible with the model.
        """
        if is_torch_available():
            from transformers.utils import torch_version

            return torch_version >= self.torch_onnx_minimum_version
        else:
            return False

    @staticmethod
    def use_external_data_format(num_parameters: int) -> bool:
        """
        Flag indicating if the model requires using external data format

        Args:
            num_parameters: Number of parameter on the model

        Returns:
            True if model.num_parameters() * size_of(float32) >= 2Gb False otherwise
        """

        return (
            compute_serialized_parameters_size(num_parameters, ParameterFormat.Float)
            >= EXTERNAL_DATA_FORMAT_SIZE_LIMIT
        )

    def _generate_dummy_images(
        self, batch_size: int = 2, num_channels: int = 3, image_height: int = 40, image_width: int = 40
    ):
        images = []
        for _ in range(batch_size):
            data = np.random.rand(image_height, image_width, num_channels) * 255
            images.append(Image.fromarray(data.astype("uint8")).convert("RGB"))
        return images

    def generate_dummy_inputs(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin"],
        batch_size: int = -1,
        seq_length: int = -1,
        num_choices: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
        num_channels: int = 3,
        image_width: int = 40,
        image_height: int = 40,
        tokenizer: "PreTrainedTokenizerBase" = None,
    ) -> Mapping[str, Any]:
        """
        Generate inputs to provide to the ONNX exporter for the specific framework

        Args:
            preprocessor: ([`PreTrainedTokenizerBase`] or [`FeatureExtractionMixin`]):
                The preprocessor associated with this model configuration.
            batch_size (`int`, *optional*, defaults to -1):
                The batch size to export the model for (-1 means dynamic axis).
            num_choices (`int`, *optional*, defaults to -1):
                The number of candidate answers provided for multiple choice task (-1 means dynamic axis).
            seq_length (`int`, *optional*, defaults to -1):
                The sequence length to export the model for (-1 means dynamic axis).
            is_pair (`bool`, *optional*, defaults to `False`):
                Indicate if the input is a pair (sentence 1, sentence 2)
            framework (`TensorType`, *optional*, defaults to `None`):
                The framework (PyTorch or TensorFlow) that the tokenizer will generate tensors for.
            num_channels (`int`, *optional*, defaults to 3):
                The number of channels of the generated images.
            image_width (`int`, *optional*, defaults to 40):
                The width of the generated images.
            image_height (`int`, *optional*, defaults to 40):
                The height of the generated images.

        Returns:
            Mapping[str, Tensor] holding the kwargs to provide to the model's forward function
        """
        from ..feature_extraction_utils import FeatureExtractionMixin
        from ..tokenization_utils_base import PreTrainedTokenizerBase

        if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
            raise ValueError("You cannot provide both a tokenizer and a preprocessor to generate dummy inputs.")
        if tokenizer is not None:
            warnings.warn(
                "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use `preprocessor` instead.",
                FutureWarning,
            )
            logger.warning("Overwriting the `preprocessor` argument with `tokenizer` to generate dummmy inputs.")
            preprocessor = tokenizer
        if isinstance(preprocessor, PreTrainedTokenizerBase):
            # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
            batch_size = compute_effective_axis_dimension(
                batch_size, fixed_dimension=OnnxConfig.default_fixed_batch, num_token_to_add=0
            )
            # If dynamic axis (-1) we forward with a fixed dimension of 8 tokens to avoid optimizations made by ONNX
            token_to_add = preprocessor.num_special_tokens_to_add(is_pair)
            seq_length = compute_effective_axis_dimension(
                seq_length, fixed_dimension=OnnxConfig.default_fixed_sequence, num_token_to_add=token_to_add
            )
            # Generate dummy inputs according to compute batch and sequence
            dummy_input = [" ".join([preprocessor.unk_token]) * seq_length] * batch_size
            if self.task == "multiple-choice":
                # If dynamic axis (-1) we forward with a fixed dimension of 4 candidate answers to avoid optimizations
                # made by ONNX
                num_choices = compute_effective_axis_dimension(
                    num_choices, fixed_dimension=OnnxConfig.default_fixed_num_choices, num_token_to_add=0
                )
                dummy_input = dummy_input * num_choices
                # The shape of the tokenized inputs values is [batch_size * num_choices, seq_length]
                tokenized_input = preprocessor(dummy_input, text_pair=dummy_input)
                # Unflatten the tokenized inputs values expanding it to the shape [batch_size, num_choices, seq_length]
                for k, v in tokenized_input.items():
                    tokenized_input[k] = [v[i : i + num_choices] for i in range(0, len(v), num_choices)]
                return dict(tokenized_input.convert_to_tensors(tensor_type=framework))
            return dict(preprocessor(dummy_input, return_tensors=framework))
        elif isinstance(preprocessor, FeatureExtractionMixin) and preprocessor.model_input_names[0] == "pixel_values":
            # If dynamic axis (-1) we forward with a fixed dimension of 2 samples to avoid optimizations made by ONNX
            batch_size = compute_effective_axis_dimension(batch_size, fixed_dimension=OnnxConfig.default_fixed_batch)
            dummy_input = self._generate_dummy_images(batch_size, num_channels, image_height, image_width)
            return dict(preprocessor(images=dummy_input, return_tensors=framework))
        else:
            raise ValueError(
                "Unable to generate dummy inputs for the model. Please provide a tokenizer or a preprocessor."
            )

    def patch_ops(self):
        for spec in self._patching_specs:
            custom_op = spec.custom_op if spec.op_wrapper is None else spec.op_wrapper(spec.custom_op)
            setattr(spec.o, spec.name, custom_op)

    def restore_ops(self):
        for spec in self._patching_specs:
            orig_op = spec.orig_op if spec.op_wrapper is None else spec.op_wrapper(spec.orig_op)
            setattr(spec.o, spec.name, orig_op)

    @classmethod
    def flatten_output_collection_property(cls, name: str, field: Iterable[Any]) -> Dict[str, Any]:
        """
        Flatten any potential nested structure expanding the name of the field with the index of the element within the
        structure.

        Args:
            name: The name of the nested structure
            field: The structure to, potentially, be flattened

        Returns:
            (Dict[str, Any]): Outputs with flattened structure and key mapping this new structure.

        """
        from itertools import chain

        return {f"{name}.{idx}": item for idx, item in enumerate(chain.from_iterable(field))}


class OnnxConfigWithPast(OnnxConfig, ABC):
    def __init__(
        self,
        config: "PretrainedConfig",
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        super().__init__(config, task=task, patching_specs=patching_specs)
        self.use_past = use_past

    @classmethod
    def with_past(cls, config: "PretrainedConfig", task: str = "default") -> "OnnxConfigWithPast":
        """
        Instantiate a OnnxConfig with `use_past` attribute set to True

        Args:
            config: The underlying model's config to use when exporting to ONNX

        Returns:
            OnnxConfig with `.use_past = True`
        """
        return cls(config, task=task, use_past=True)

    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = super().outputs
        if self.use_past:
            self.fill_with_past_key_values_(common_outputs, direction="outputs")

        return common_outputs

    @property
    def values_override(self) -> Optional[Mapping[str, Any]]:
        if hasattr(self._config, "use_cache"):
            return {"use_cache": self.use_past}

        return None

    @property
    def num_layers(self) -> int:
        """
        The number of layers attribute retrieved from the model config. Override this for model configs where the
        number of layers attribute is not called `num_layers`.
        """
        if not hasattr(self._config, "num_layers"):
            raise AttributeError(
                "could not find the number of layers attribute in the model configuration, override the num_layers property of the model OnnxConfig to solve this"
            )
        return self._config.num_layers

    @property
    def num_attention_heads(self) -> int:
        """
        The number of attention heads attribute retrieved from the model config. Override this for model configs where
        the number of attention heads attribute is not called `num_attention_heads`.
        """
        if not hasattr(self._config, "num_attention_heads"):
            raise AttributeError(
                "could not find the number of attention heads attribute in the model configuration, override the num_attention_heads property of the model OnnxConfig to solve this"
            )
        return self._config.num_attention_heads

    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:

        # TODO: should we set seq_length = 1 when self.use_past = True?
        common_inputs = super().generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

            batch, seqlen = common_inputs["input_ids"].shape
            # Not using the same length for past_key_values
            past_key_values_length = seqlen + 2
            shape = (
                batch,
                self.num_attention_heads,
                past_key_values_length,
                self._config.hidden_size // self.num_attention_heads,
            )

            if "attention_mask" in common_inputs:
                common_inputs["attention_mask"] = torch.cat(
                    [common_inputs["attention_mask"], torch.ones(batch, past_key_values_length)], dim=1
                )

            common_inputs["past_key_values"] = []
            for _ in range(self.num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))

        return common_inputs

    def fill_with_past_key_values_(self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str):
        """
        Fill the input_or_ouputs mapping with past_key_values dynamic axes considering.

        Args:
            inputs_or_outputs: The mapping to fill.
            direction: either "inputs" or "outputs", it specifies whether input_or_outputs is the input mapping or the
                output mapping, this is important for axes naming.

        """
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        name = "past_key_values" if direction == "inputs" else "present"
        for i in range(self.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch", 2: "past_sequence + sequence"}
            inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch", 2: "past_sequence + sequence"}

    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        flattened_output[f"{name}.{idx}.key"] = t[0]
        flattened_output[f"{name}.{idx}.value"] = t[1]

    def flatten_output_collection_property(self, name: str, field: Iterable[Any]) -> Dict[str, Any]:
        flattened_output = {}
        if name in ["present", "past_key_values"]:
            for idx, t in enumerate(field):
                self._flatten_past_key_values_(flattened_output, name, idx, t)
        else:
            flattened_output = super().flatten_output_collection_property(name, field)

        return flattened_output


class OnnxSeq2SeqConfigWithPast(OnnxConfigWithPast):
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        common_outputs = super(OnnxConfigWithPast, self).outputs
        # Renaming the outputs axes properly.
        for name, axes_names in common_outputs.items():
            sequence_name = "encoder_sequence" if "encoder" in name else "decoder_sequence"
            for axis_idx, name in axes_names.items():
                if "sequence" in name:
                    axes_names[axis_idx] = sequence_name
                # We reset the value as the order in common_outputs (OrderedDict) is lost otherwise
                else:
                    axes_names[axis_idx] = name
        if self.use_past:
            self.fill_with_past_key_values_(common_outputs, direction="outputs")

        return common_outputs

    @property
    def num_layers(self) -> Tuple[int]:
        try:
            num_layers = super().num_layers
            num_layers = (num_layers, num_layers)
        except AttributeError:
            if hasattr(self._config, "encoder_layers") and hasattr(self._config, "decoder_layers"):
                num_layers = (self._config.encoder_layers, self._config.decoder_layers)
            else:
                raise AttributeError(
                    "could not find the number of encoder and decoder layers attributes in the model configuration, override the num_layers property of the model OnnxConfig to solve this"
                )

        return num_layers

    @property
    def num_attention_heads(self) -> Tuple[int]:
        try:
            num_attention_heads = super().num_attention_heads
            num_attention_heads = (num_attention_heads, num_attention_heads)
        except AttributeError:
            if hasattr(self._config, "encoder_attention_heads") and hasattr(self._config, "decoder_attention_heads"):
                num_attention_heads = (self._config.encoder_attention_heads, self._config.decoder_attention_heads)
            else:
                raise AttributeError(
                    "could not find the number of attention heads for the encoder and the decoder attributes in the model configuration, override the num_attention_heads property of the model OnnxConfig to solve this"
                )
        return num_attention_heads

    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:

        encoder_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # Generate decoder inputs
        decoder_seq_length = seq_length if not self.use_past else 1
        decoder_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=decoder_seq_length, is_pair=is_pair, framework=framework
        )
        decoder_inputs = {f"decoder_{name}": tensor for name, tensor in decoder_inputs.items()}
        common_inputs = dict(**encoder_inputs, **decoder_inputs)

        if self.use_past:
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch
            batch = common_inputs["input_ids"].shape[0]
            encoder_seq_length = common_inputs["input_ids"].shape[1]
            decoder_seq_length = common_inputs["decoder_input_ids"].shape[1]
            num_encoder_attention_heads, num_decoder_attention_heads = self.num_attention_heads
            encoder_shape = (
                batch,
                num_encoder_attention_heads,
                encoder_seq_length,
                self._config.hidden_size // num_encoder_attention_heads,
            )
            decoder_shape = (
                batch,
                num_decoder_attention_heads,
                # Not using the same length for past_key_values
                decoder_seq_length + 3,
                self._config.hidden_size // num_decoder_attention_heads,
            )

            common_inputs["past_key_values"] = []
            # If the number of encoder and decoder layers are present in the model configuration, both are considered
            num_encoder_layers, num_decoder_layers = self.num_layers
            min_num_layers = min(num_encoder_layers, num_decoder_layers)
            max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
            remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"

            for _ in range(min_num_layers):
                # For encoder-decoder models, past_key_values contains pre-computed values for both the encoder and the
                # decoder layers, hence a tuple of 4 tensors instead of 2
                common_inputs["past_key_values"].append(
                    (
                        torch.zeros(decoder_shape),
                        torch.zeros(decoder_shape),
                        torch.zeros(encoder_shape),
                        torch.zeros(encoder_shape),
                    )
                )

            # TODO: test this.
            shape = encoder_shape if remaining_side_name == "encoder" else decoder_shape
            for _ in range(min_num_layers, max_num_layers):
                common_inputs["past_key_values"].append((torch.zeros(shape), torch.zeros(shape)))

        return common_inputs

    def fill_with_past_key_values_(self, inputs_or_outputs: Mapping[str, Mapping[int, str]], direction: str):
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        name = "past_key_values" if direction == "inputs" else "present"

        # If the number of encoder and decoder layers are present in the model configuration, both are considered
        num_encoder_layers, num_decoder_layers = self.num_layers
        min_num_layers = min(num_encoder_layers, num_decoder_layers)
        max_num_layers = max(num_encoder_layers, num_decoder_layers) - min_num_layers
        remaining_side_name = "encoder" if num_encoder_layers > num_decoder_layers else "decoder"

        encoder_sequence = "past_encoder_sequence"
        decoder_sequence = "past_decoder_sequence" if direction == "inputs" else "past_decoder_sequence + sequence"

        for i in range(min_num_layers):
            inputs_or_outputs[f"{name}.{i}.decoder.key"] = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.decoder.value"] = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.encoder.key"] = {0: "batch", 2: encoder_sequence}
            inputs_or_outputs[f"{name}.{i}.encoder.value"] = {0: "batch", 2: encoder_sequence}

        for i in range(min_num_layers, max_num_layers):
            if remaining_side_name == "encoder":
                axes_info = {0: "batch", 2: encoder_sequence}
            else:
                axes_info = {0: "batch", 2: decoder_sequence}
            inputs_or_outputs[f"{name}.{i}.{remaining_side_name}.key"] = axes_info

    def _flatten_past_key_values_(self, flattened_output, name, idx, t):
        flattened_output[f"{name}.{idx}.decoder.key"] = t[0]
        flattened_output[f"{name}.{idx}.decoder.value"] = t[1]
        flattened_output[f"{name}.{idx}.encoder.key"] = t[2]
        flattened_output[f"{name}.{idx}.encoder.value"] = t[3]
