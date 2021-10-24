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

from inspect import signature
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
from packaging.version import Version, parse

from .. import PreTrainedModel, PreTrainedTokenizer, TensorType, TFPreTrainedModel, is_torch_available
from ..file_utils import is_torch_onnx_dict_inputs_support_available
from ..utils import logging
from .config import OnnxConfig


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
                f"Please update onnxruntime by running `pip install --upgrade onnxruntime`"
            )

    except ImportError:
        raise ImportError(
            "onnxruntime doesn't seem to be currently installed. "
            "Please install the onnxruntime by running `pip install onnxruntime`"
            " and relaunch the conversion."
        )


def export(
    tokenizer: PreTrainedTokenizer, model: PreTrainedModel, config: OnnxConfig, opset: int, output: Path
) -> Tuple[List[str], List[str]]:
    """
    Export a PyTorch backed pipeline to ONNX Intermediate Representation (IR

    Args:
        tokenizer:
        model:
        config:
        opset:
        output:

    Returns:

    """
    if not is_torch_available():
        raise ImportError("Cannot convert because PyTorch is not installed. Please install torch first.")

    import torch
    from torch.onnx import export

    from ..file_utils import torch_version

    if not is_torch_onnx_dict_inputs_support_available():
        raise AssertionError(f"Unsupported PyTorch version, minimum required is 1.8.0, got: {torch_version}")

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
        model_inputs = config.generate_dummy_inputs(tokenizer, framework=TensorType.PYTORCH)
        inputs_match, matched_inputs = ensure_model_and_config_inputs_match(model, model_inputs.keys())
        onnx_outputs = list(config.outputs.keys())

        if not inputs_match:
            raise ValueError("Model and config inputs doesn't match")

        config.patch_ops()

        # export can works with named args but the dict containing named args as to be last element of the args tuple
        export(
            model,
            (model_inputs,),
            f=output.as_posix(),
            input_names=list(config.inputs.keys()),
            output_names=onnx_outputs,
            dynamic_axes={name: axes for name, axes in chain(config.inputs.items(), config.outputs.items())},
            do_constant_folding=True,
            use_external_data_format=config.use_external_data_format(model.num_parameters()),
            enable_onnx_checker=True,
            opset_version=opset,
        )

        config.restore_ops()

    return matched_inputs, onnx_outputs


def validate_model_outputs(
    config: OnnxConfig,
    tokenizer: PreTrainedTokenizer,
    reference_model: Union[PreTrainedModel, TFPreTrainedModel],
    onnx_model: Path,
    onnx_named_outputs: List[str],
    atol: float,
):
    from onnxruntime import InferenceSession, SessionOptions

    logger.info("Validating ONNX model...")

    # TODO: generate inputs with a different batch_size and seq_len that was used for conversion to properly test
    # dynamic input shapes.
    reference_model_inputs = config.generate_dummy_inputs(tokenizer, framework=TensorType.PYTORCH)

    # Create ONNX Runtime session
    options = SessionOptions()
    session = InferenceSession(onnx_model.as_posix(), options)

    # Compute outputs from the reference model
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

    # We flatten potential collection of inputs (i.e. past_keys)
    onnx_inputs = {}
    for name, value in reference_model_inputs.items():
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
            f"\t-[x] ONNX model outputs' name {onnx_outputs_set} doesn't match reference model {ref_outputs_set}"
        )

        raise ValueError(
            "Outputs doesn't match between reference model and ONNX exported model: "
            f"{onnx_outputs_set.difference(ref_outputs_set)}"
        )
    else:
        logger.info(f"\t-[✓] ONNX model outputs' name match reference model ({onnx_outputs_set}")

    # Check the shape and values match
    for name, ort_value in zip(onnx_named_outputs, onnx_outputs):
        ref_value = ref_outputs_dict[name].detach().numpy()
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
            logger.info(f"\t\t-[x] values not close enough (atol: {atol})")
            raise ValueError(
                "Outputs values doesn't match between reference model and ONNX exported model: "
                f"Got max absolute difference of: {np.amax(np.abs(ref_value - ort_value))}"
            )
        else:
            logger.info(f"\t\t-[✓] all values close (atol: {atol})")


def ensure_model_and_config_inputs_match(
    model: Union[PreTrainedModel, TFPreTrainedModel], model_inputs: Iterable[str]
) -> Tuple[bool, List[str]]:
    """

    :param model_inputs:
    :param config_inputs:
    :return:
    """
    forward_parameters = signature(model.forward).parameters
    model_inputs_set = set(model_inputs)

    # We are fine if config_inputs has more keys than model_inputs
    forward_inputs_set = set(forward_parameters.keys())
    is_ok = model_inputs_set.issubset(forward_inputs_set)

    # Make sure the input order match (VERY IMPORTANT !!!!)
    matching_inputs = forward_inputs_set.intersection(model_inputs_set)
    ordered_inputs = [parameter for parameter in forward_parameters.keys() if parameter in matching_inputs]
    return is_ok, ordered_inputs
