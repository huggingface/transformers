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

import warnings
from inspect import signature
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Tuple, Union

import numpy as np
from packaging.version import Version, parse

from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import (
    TensorType,
    is_tf_available,
    is_torch_available,
    logging,
)
from .config import OnnxConfig


if is_torch_available():
    from ..modeling_utils import PreTrainedModel

if is_tf_available():
    from ..modeling_tf_utils import TFPreTrainedModel

if TYPE_CHECKING:
    from ..feature_extraction_utils import FeatureExtractionMixin
    from ..processing_utils import ProcessorMixin
    from ..tokenization_utils import PreTrainedTokenizer


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# This is the minimal required version to support some ONNX Runtime features
ORT_QUANTIZE_MINIMUM_VERSION = parse("1.4.0")


def check_onnxruntime_requirements(minimum_version: Version):
    """
    Check onnxruntime is installed and if the installed version match is recent enough

    Raises:
        ImportError: If onnxruntime is not installed or too old version is found
    """
    try:
        import onnxruntime

        # Parse the version of the installed onnxruntime
        ort_version = parse(onnxruntime.__version__)

        # We require 1.4.0 minimum
        if ort_version < ORT_QUANTIZE_MINIMUM_VERSION:
            raise ImportError(
                f"We found an older version of onnxruntime ({onnxruntime.__version__}) "
                f"but we require onnxruntime to be >= {minimum_version} to enable all the conversions options.\n"
                "Please update onnxruntime by running `pip install --upgrade onnxruntime`"
            )

    except ImportError:
        raise ImportError(
            "onnxruntime doesn't seem to be currently installed. "
            "Please install the onnxruntime by running `pip install onnxruntime`"
            " and relaunch the conversion."
        )


def export_pytorch(
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin", "ProcessorMixin"],
    model: "PreTrainedModel",
    config: OnnxConfig,
    opset: int,
    output: Path,
    tokenizer: "PreTrainedTokenizer" = None,
    device: str = "cpu",
) -> Tuple[List[str], List[str]]:
    """
    Export a PyTorch model to an ONNX Intermediate Representation (IR)

    Args:
        preprocessor: ([`PreTrainedTokenizer`], [`FeatureExtractionMixin`] or [`ProcessorMixin`]):
            The preprocessor used for encoding the data.
        model ([`PreTrainedModel`]):
            The model to export.
        config ([`~onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.
        opset (`int`):
            The version of the ONNX operator set to use.
        output (`Path`):
            Directory to store the exported ONNX model.
        device (`str`, *optional*, defaults to `cpu`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the ONNX configuration.
    """

    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        raise ValueError("You cannot provide both a tokenizer and a preprocessor to export the model.")
    if tokenizer is not None:
        warnings.warn(
            "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use"
            " `preprocessor` instead.",
            FutureWarning,
        )
        logger.info("Overwriting the `preprocessor` argument with `tokenizer` to generate dummmy inputs.")
        preprocessor = tokenizer

    if issubclass(type(model), PreTrainedModel):
        import torch
        from torch.onnx import export as onnx_export

        logger.info(f"Using framework PyTorch: {torch.__version__}")
        with torch.no_grad():
            model.config.return_dict = True
            model.eval()

            # Check if we need to override certain configuration item
            if config.values_override is not None:
                logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
                for override_config_key, override_config_value in config.values_override.items():
                    logger.info(f"\t- {override_config_key} -> {override_config_value}")
                    setattr(model.config, override_config_key, override_config_value)

            # Ensure inputs match
            # TODO: Check when exporting QA we provide "is_pair=True"
            model_inputs = config.generate_dummy_inputs(preprocessor, framework=TensorType.PYTORCH)
            device = torch.device(device)
            if device.type == "cuda" and torch.cuda.is_available():
                model.to(device)
                model_inputs_device = {}
                for k, v in model_inputs.items():
                    if isinstance(v, Tuple):
                        model_inputs_device[k] = tuple(
                            x.to(device) if isinstance(x, torch.Tensor) else None for x in v
                        )
                    elif isinstance(v, List):
                        model_inputs_device[k] = [
                            tuple(x.to(device) if isinstance(x, torch.Tensor) else None for x in t) for t in v
                        ]
                    else:
                        model_inputs_device[k] = v.to(device)

                model_inputs = model_inputs_device

            inputs_match, matched_inputs = ensure_model_and_config_inputs_match(model, model_inputs.keys())
            onnx_outputs = list(config.outputs.keys())

            if not inputs_match:
                raise ValueError("Model and config inputs doesn't match")

            config.patch_ops()

            onnx_export(
                model,
                (model_inputs,),
                f=output.as_posix(),
                input_names=list(config.inputs.keys()),
                output_names=onnx_outputs,
                dynamic_axes=dict(chain(config.inputs.items(), config.outputs.items())),
                do_constant_folding=True,
                opset_version=opset,
            )

            config.restore_ops()

    return matched_inputs, onnx_outputs


def export_tensorflow(
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin"],
    model: "TFPreTrainedModel",
    config: OnnxConfig,
    opset: int,
    output: Path,
    tokenizer: "PreTrainedTokenizer" = None,
) -> Tuple[List[str], List[str]]:
    """
    Export a TensorFlow model to an ONNX Intermediate Representation (IR)

    Args:
        preprocessor: ([`PreTrainedTokenizer`] or [`FeatureExtractionMixin`]):
            The preprocessor used for encoding the data.
        model ([`TFPreTrainedModel`]):
            The model to export.
        config ([`~onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.
        opset (`int`):
            The version of the ONNX operator set to use.
        output (`Path`):
            Directory to store the exported ONNX model.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the ONNX configuration.
    """
    import onnx
    import tensorflow as tf
    import tf2onnx

    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        raise ValueError("You cannot provide both a tokenizer and preprocessor to export the model.")
    if tokenizer is not None:
        warnings.warn(
            "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use"
            " `preprocessor` instead.",
            FutureWarning,
        )
        logger.info("Overwriting the `preprocessor` argument with `tokenizer` to generate dummmy inputs.")
        preprocessor = tokenizer

    model.config.return_dict = True

    # Check if we need to override certain configuration item
    if config.values_override is not None:
        logger.info(f"Overriding {len(config.values_override)} configuration item(s)")
        for override_config_key, override_config_value in config.values_override.items():
            logger.info(f"\t- {override_config_key} -> {override_config_value}")
            setattr(model.config, override_config_key, override_config_value)

    # Ensure inputs match
    model_inputs = config.generate_dummy_inputs(preprocessor, framework=TensorType.TENSORFLOW)
    inputs_match, matched_inputs = ensure_model_and_config_inputs_match(model, model_inputs.keys())
    onnx_outputs = list(config.outputs.keys())

    input_signature = [
        tf.TensorSpec([None] * tensor.ndim, dtype=tensor.dtype, name=key) for key, tensor in model_inputs.items()
    ]
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=opset)
    onnx.save(onnx_model, output.as_posix())
    config.restore_ops()

    return matched_inputs, onnx_outputs


def export(
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin", "ProcessorMixin"],
    model: Union["PreTrainedModel", "TFPreTrainedModel"],
    config: OnnxConfig,
    opset: int,
    output: Path,
    tokenizer: "PreTrainedTokenizer" = None,
    device: str = "cpu",
) -> Tuple[List[str], List[str]]:
    """
    Export a Pytorch or TensorFlow model to an ONNX Intermediate Representation (IR)

    Args:
        preprocessor: ([`PreTrainedTokenizer`], [`FeatureExtractionMixin`] or [`ProcessorMixin`]):
            The preprocessor used for encoding the data.
        model ([`PreTrainedModel`] or [`TFPreTrainedModel`]):
            The model to export.
        config ([`~onnx.config.OnnxConfig`]):
            The ONNX configuration associated with the exported model.
        opset (`int`):
            The version of the ONNX operator set to use.
        output (`Path`):
            Directory to store the exported ONNX model.
        device (`str`, *optional*, defaults to `cpu`):
            The device on which the ONNX model will be exported. Either `cpu` or `cuda`. Only PyTorch is supported for
            export on CUDA devices.

    Returns:
        `Tuple[List[str], List[str]]`: A tuple with an ordered list of the model's inputs, and the named inputs from
        the ONNX configuration.
    """
    if not (is_torch_available() or is_tf_available()):
        raise ImportError(
            "Cannot convert because neither PyTorch nor TensorFlow are not installed. "
            "Please install torch or tensorflow first."
        )

    if is_tf_available() and isinstance(model, TFPreTrainedModel) and device == "cuda":
        raise RuntimeError("`tf2onnx` does not support export on CUDA device.")

    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        raise ValueError("You cannot provide both a tokenizer and a preprocessor to export the model.")
    if tokenizer is not None:
        warnings.warn(
            "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use"
            " `preprocessor` instead.",
            FutureWarning,
        )
        logger.info("Overwriting the `preprocessor` argument with `tokenizer` to generate dummmy inputs.")
        preprocessor = tokenizer

    if is_torch_available():
        from ..utils import get_torch_version

        if not config.is_torch_support_available:
            logger.warning(
                f"Unsupported PyTorch version for this model. Minimum required is {config.torch_onnx_minimum_version},"
                f" got: {get_torch_version()}"
            )

    if is_torch_available() and issubclass(type(model), PreTrainedModel):
        return export_pytorch(preprocessor, model, config, opset, output, tokenizer=tokenizer, device=device)
    elif is_tf_available() and issubclass(type(model), TFPreTrainedModel):
        return export_tensorflow(preprocessor, model, config, opset, output, tokenizer=tokenizer)


def validate_model_outputs(
    config: OnnxConfig,
    preprocessor: Union["PreTrainedTokenizer", "FeatureExtractionMixin", "ProcessorMixin"],
    reference_model: Union["PreTrainedModel", "TFPreTrainedModel"],
    onnx_model: Path,
    onnx_named_outputs: List[str],
    atol: float,
    tokenizer: "PreTrainedTokenizer" = None,
):
    from onnxruntime import InferenceSession, SessionOptions

    logger.info("Validating ONNX model...")

    if isinstance(preprocessor, PreTrainedTokenizerBase) and tokenizer is not None:
        raise ValueError("You cannot provide both a tokenizer and a preprocessor to validate the model outputs.")
    if tokenizer is not None:
        warnings.warn(
            "The `tokenizer` argument is deprecated and will be removed in version 5 of Transformers. Use"
            " `preprocessor` instead.",
            FutureWarning,
        )
        logger.info("Overwriting the `preprocessor` argument with `tokenizer` to generate dummmy inputs.")
        preprocessor = tokenizer

    # generate inputs with a different batch_size and seq_len that was used for conversion to properly test
    # dynamic input shapes.
    if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
        reference_model_inputs = config.generate_dummy_inputs(
            preprocessor,
            batch_size=config.default_fixed_batch + 1,
            seq_length=config.default_fixed_sequence + 1,
            framework=TensorType.PYTORCH,
        )
    else:
        reference_model_inputs = config.generate_dummy_inputs(
            preprocessor,
            batch_size=config.default_fixed_batch + 1,
            seq_length=config.default_fixed_sequence + 1,
            framework=TensorType.TENSORFLOW,
        )

    # Create ONNX Runtime session
    options = SessionOptions()
    session = InferenceSession(onnx_model.as_posix(), options, providers=["CPUExecutionProvider"])

    # Compute outputs from the reference model
    if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
        reference_model.to("cpu")
    ref_outputs = reference_model(**reference_model_inputs)
    ref_outputs_dict = {}

    # We flatten potential collection of outputs (i.e. past_keys) to a flat structure
    for name, value in ref_outputs.items():
        # Overwriting the output name as "present" since it is the name used for the ONNX outputs
        # ("past_key_values" being taken for the ONNX inputs)
        if name == "past_key_values":
            name = "present"
        if isinstance(value, (list, tuple)):
            value = config.flatten_output_collection_property(name, value)
            ref_outputs_dict.update(value)
        else:
            ref_outputs_dict[name] = value

    # Create onnxruntime inputs from the reference model inputs
    reference_model_inputs_onnxruntime = config.generate_dummy_inputs_onnxruntime(reference_model_inputs)

    # We flatten potential collection of inputs (i.e. past_keys)
    onnx_inputs = {}
    for name, value in reference_model_inputs_onnxruntime.items():
        if isinstance(value, (list, tuple)):
            value = config.flatten_output_collection_property(name, value)
            onnx_inputs.update({tensor_name: pt_tensor.numpy() for tensor_name, pt_tensor in value.items()})
        else:
            onnx_inputs[name] = value.numpy()

    # Compute outputs from the ONNX model
    onnx_outputs = session.run(onnx_named_outputs, onnx_inputs)

    # Check we have a subset of the keys into onnx_outputs against ref_outputs
    ref_outputs_set, onnx_outputs_set = set(ref_outputs_dict.keys()), set(onnx_named_outputs)
    if not onnx_outputs_set.issubset(ref_outputs_set):
        logger.info(
            f"\t-[x] ONNX model output names {onnx_outputs_set} do not match reference model {ref_outputs_set}"
        )

        raise ValueError(
            "Outputs doesn't match between reference model and ONNX exported model: "
            f"{onnx_outputs_set.difference(ref_outputs_set)}"
        )
    else:
        logger.info(f"\t-[✓] ONNX model output names match reference model ({onnx_outputs_set})")

    # Check the shape and values match
    for name, ort_value in zip(onnx_named_outputs, onnx_outputs):
        if is_torch_available() and issubclass(type(reference_model), PreTrainedModel):
            ref_value = ref_outputs_dict[name].detach().numpy()
        else:
            ref_value = ref_outputs_dict[name].numpy()
        logger.info(f'\t- Validating ONNX Model output "{name}":')

        # Shape
        if not ort_value.shape == ref_value.shape:
            logger.info(f"\t\t-[x] shape {ort_value.shape} doesn't match {ref_value.shape}")
            raise ValueError(
                "Outputs shape doesn't match between reference model and ONNX exported model: "
                f"Got {ref_value.shape} (reference) and {ort_value.shape} (ONNX)"
            )
        else:
            logger.info(f"\t\t-[✓] {ort_value.shape} matches {ref_value.shape}")

        # Values
        if not np.allclose(ref_value, ort_value, atol=atol):
            bad_indices = np.logical_not(np.isclose(ref_value, ort_value, atol=atol))
            logger.info(f"\t\t-[x] values not close enough (atol: {atol})")
            raise ValueError(
                "Outputs values doesn't match between reference model and ONNX exported model: "
                f"Got max absolute difference of: {np.amax(np.abs(ref_value - ort_value))} for "
                f"{ref_value[bad_indices]} vs {ort_value[bad_indices]}"
            )
        else:
            logger.info(f"\t\t-[✓] all values close (atol: {atol})")


def ensure_model_and_config_inputs_match(
    model: Union["PreTrainedModel", "TFPreTrainedModel"], model_inputs: Iterable[str]
) -> Tuple[bool, List[str]]:
    """

    :param model_inputs: :param config_inputs: :return:
    """
    if is_torch_available() and issubclass(type(model), PreTrainedModel):
        forward_parameters = signature(model.forward).parameters
    else:
        forward_parameters = signature(model.call).parameters
    model_inputs_set = set(model_inputs)

    # We are fine if config_inputs has more keys than model_inputs
    forward_inputs_set = set(forward_parameters.keys())
    is_ok = model_inputs_set.issubset(forward_inputs_set)

    # Make sure the input order match (VERY IMPORTANT !!!!)
    matching_inputs = forward_inputs_set.intersection(model_inputs_set)
    ordered_inputs = [parameter for parameter in forward_parameters.keys() if parameter in matching_inputs]
    return is_ok, ordered_inputs
