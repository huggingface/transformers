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

from itertools import chain
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
from packaging.version import Version, parse

from onnxruntime import GraphOptimizationLevel

from .. import PreTrainedModel, PreTrainedTokenizer, TensorType, TFPreTrainedModel, is_torch_available
from .config import OnnxConfig, OnnxVariable
from .interpolate import (
    evaluate_expr_to_int,
    expand_repeated_onnx_variables,
    insert_additional_onnx_value_within_inputs,
    interpolate_expression,
)
from .utils import flatten_output_collection_property


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


def convert_pytorch(
    tokenizer: PreTrainedTokenizer, model: PreTrainedModel, config: OnnxConfig, opset: int, output: Path
) -> Tuple[List[OnnxVariable], List[OnnxVariable]]:
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
        raise Exception("Cannot convert because PyTorch is not installed. Please install torch first.")

    import torch
    from torch.onnx import export

    print(f"Using framework PyTorch: {torch.__version__}")
    torch.set_grad_enabled(False)
    model.config.return_dict = True

    onnx_inputs = expand_repeated_onnx_variables(model, config.inputs)
    onnx_outputs = expand_repeated_onnx_variables(model, config.outputs)

    # Ensure inputs match
    # TODO: Sequence length = 4 hard coded, provide this value through CLI would be better
    model_inputs = tokenizer([tokenizer.unk_token] * 4, is_split_into_words=True, return_tensors=TensorType.PYTORCH)
    model_inputs = insert_additional_onnx_value_within_inputs(model_inputs, onnx_inputs, TensorType.PYTORCH)

    inputs_match, ordered_onnx_inputs = ensure_model_and_config_inputs_match(model_inputs.keys(), onnx_inputs)

    if not inputs_match:
        raise ValueError("Model and config inputs doesn't match")

    # export can works with named args but the dict containing named args as to be last element of the args tuple
    export(
        model,
        (dict(model_inputs),),
        f=output.as_posix(),
        input_names=[var.name for var in ordered_onnx_inputs],
        output_names=[var.name for var in onnx_outputs],
        dynamic_axes={var.name: var.axes for var in chain(config.inputs, onnx_outputs)},
        do_constant_folding=True,
        use_external_data_format=config.use_external_data_format,
        enable_onnx_checker=True,
        opset_version=opset,
    )

    return onnx_inputs, onnx_outputs


def optimize(
    onnx_model_path: Path,
    model: Union[PreTrainedModel, TFPreTrainedModel],
    onnx_config: OnnxConfig,
    optimization_level: GraphOptimizationLevel,
    use_gpu: bool,
    output: Path,
):
    from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
    from onnxruntime.transformers.optimizer import optimize_by_onnxruntime, optimize_model

    # If we have an optimizer in the config, let's optimize offline
    if onnx_config.optimizer is not None:
        print(f"Optimizing model through dedicated '{onnx_config.optimizer}' tool (the tool's name might not match):")

        # Interpolate addition_args if needed
        if onnx_config.optimizer_additional_args is not None:
            additional_args = {
                arg_name: evaluate_expr_to_int(interpolate_expression(arg_value, model))
                for arg_name, arg_value in onnx_config.optimizer_additional_args.items()
            }
        else:
            additional_args = {}

        optimizer_options = BertOptimizationOptions(onnx_config.optimizer)
        if onnx_config.optimizer_features is not None:
            for feature_name, feature_value in onnx_config.optimizer_features.items():
                print(f"\t- {feature_name} = {feature_value}")
                setattr(optimizer_options, feature_name, feature_value)

        # Optimize
        optimizer = optimize_model(
            input=onnx_model_path.as_posix(),
            model_type=onnx_config.optimizer,
            optimization_options=optimizer_options,
            opt_level=int(optimization_level),
            use_gpu=use_gpu,
            **additional_args,
        )

        print(f"Optimization statistics: {optimizer.get_fused_operator_statistics()}")
        optimizer.save_model_to_file(output.as_posix())

    # Else use online ONNX Runtime optimization
    else:
        from os import replace

        temp_output_path = Path(optimize_by_onnxruntime(onnx_model_path.as_posix()))
        replace(temp_output_path, output)


def validate_model_outputs(
    tokenizer: PreTrainedTokenizer,
    reference_model: Union[PreTrainedModel, TFPreTrainedModel],
    onnx_model: Path,
    onnx_inputs: List[OnnxVariable],
    onnx_named_outputs: List[OnnxVariable],
    atol: float,
):
    from onnxruntime import InferenceSession, SessionOptions

    print("Validating ONNX model...")

    # TODO: Sequence length = 4 hard coded, provide this value through CLI would be better
    reference_tensor_type = (
        TensorType.PYTORCH if isinstance(reference_model, PreTrainedModel) else TensorType.TENSORFLOW
    )
    reference_model_inputs = tokenizer(
        [tokenizer.unk_token] * 4, is_split_into_words=True, return_tensors=reference_tensor_type
    )
    onnx_model_inputs = tokenizer([tokenizer.unk_token] * 4, is_split_into_words=True, return_tensors=TensorType.NUMPY)

    # Check if we need to introduce some more variables
    reference_model_inputs = insert_additional_onnx_value_within_inputs(
        reference_model_inputs, onnx_inputs, TensorType.PYTORCH
    )
    onnx_model_inputs = insert_additional_onnx_value_within_inputs(onnx_model_inputs, onnx_inputs, TensorType.NUMPY)

    # Create ONNX Runtime session
    options = SessionOptions()
    session = InferenceSession(onnx_model.as_posix(), options)

    # Compute outputs from the reference model
    ref_outputs = reference_model(**reference_model_inputs)
    ref_outputs_dict = {}

    # We flatten potential collection of outputs (i.e. past_keys) to a flat structure
    for name, value in ref_outputs.items():
        if isinstance(value, (list, tuple)):
            value = flatten_output_collection_property(name, value)
            ref_outputs_dict.update(value)
        else:
            ref_outputs_dict[name] = value

    # Compute outputs from the ONNX model
    onnx_outputs_name = [var.name for var in onnx_named_outputs]
    onnx_outputs = session.run(onnx_outputs_name, dict(onnx_model_inputs))

    # Check we have a subset of the keys into onnx_outputs against ref_outputs
    onnx_named_outputs = zip(onnx_outputs_name, onnx_outputs)
    ref_outputs_set, onnx_outputs_set = set(ref_outputs_dict.keys()), set(onnx_outputs_name)
    if not onnx_outputs_set.issubset(ref_outputs_set):
        raise ValueError(
            "Outputs doesn't match between reference model and ONNX exported model: "
            f"{onnx_outputs_set.difference(ref_outputs_set)}"
        )

    # Check the shape and values match
    for name, ort_value in onnx_named_outputs:
        ref_value = ref_outputs_dict[name].numpy()

        # Shape
        if not ref_value.shape == ort_value.shape:
            raise ValueError(
                "Outputs shape doesn't match between reference model and ONNX exported model: "
                f"Got {ref_value.shape} (reference) and {ort_value.shape} (ONNX)"
            )

        # Values
        if not np.allclose(ref_value, ort_value, atol=atol):
            raise ValueError(
                "Outputs values doesn't match between reference model and ONNX exported model: "
                f"Got max absolute difference of: {np.amax(np.abs(ref_value - ort_value))}"
            )


def ensure_model_and_config_inputs_match(
    model_inputs: Iterable[str], config_inputs: Iterable[OnnxVariable]
) -> Tuple[bool, List[OnnxVariable]]:
    """

    :param model_inputs:
    :param config_inputs:
    :return:
    """
    model_inputs_set, config_inputs_set = set(model_inputs), set(map(lambda var: var.name, config_inputs))

    # We are fine if config_inputs has more keys than model_inputs
    is_ok = model_inputs_set.issubset(config_inputs_set)

    # Make sure the input order match
    matching_inputs = config_inputs_set.intersection(model_inputs_set)
    ordered_matching_inputs = [input for input in config_inputs if input.name in matching_inputs]
    return is_ok, ordered_matching_inputs
