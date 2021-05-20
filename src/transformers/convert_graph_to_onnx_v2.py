from argparse import ArgumentParser
from itertools import chain
from pathlib import Path

from logging import getLogger, basicConfig
from typing import Set, Tuple, List, Iterable, Union

import functools
import numpy as np
from packaging.version import parse, Version

# This is the minimal required version to
# support some ONNX Runtime features
from torch.nn import Module

from transformers import AutoModel, TFAutoModel, PreTrainedModel, is_torch_available, TFPreTrainedModel
from transformers.configuration_utils import OnnxConfig, OnnxVariable
from transformers.models.bart.configuration_bart import BartOnnxConfig

ORT_QUANTIZE_MINIMUM_VERSION = parse("1.4.0")

# Set of frameworks we can export from
FRAMEWORK_NAME_PT = "pytorch"
FRAMEWORK_NAME_TF = "tensorflow"
FRAMEWORK_CHOICES = {FRAMEWORK_NAME_PT, FRAMEWORK_NAME_PT}

# Set of model topologies we support
SUPPORTED_MODEL_KIND = {
    "BartModel": BartOnnxConfig,
    "TFBartModel": BartOnnxConfig
}


def generate_identified_filename(filename: Path, identifier: str) -> Path:
    """
    Append a string-identifier at the end (before the extension, if any) to the provided filepath

    Args:
        filename: pathlib.Path The actual path object we would like to add an identifier suffix
        identifier: The suffix to add

    Returns: String with concatenated identifier at the end of the filename
    """
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(filename.suffix)


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


def check_supported_model_or_raise(model: Union[PreTrainedModel, TFPreTrainedModel]) -> Tuple[str, OnnxConfig]:
    model_kind = type(model).__name__
    if model_kind not in SUPPORTED_MODEL_KIND:
        raise KeyError(
            f"{model_kind} ({args.model}) is not supported yet. "
            f"Only {SUPPORTED_MODEL_KIND} are supported. "
            f"If you want to support ({model_kind}) please propose a PR or open up an issue."
        )

    return model_kind, SUPPORTED_MODEL_KIND[model_kind]


def expand_repeated_onnx_variables(model: Union[PreTrainedModel, TFPreTrainedModel], variables: List[OnnxVariable]) -> List[OnnxVariable]:
    def rgetattr(obj, attr):
        """
        Get an attribute recursively (i.e. multiple nested object obj.property_x.propert_y)
        :param obj:
        :param attr:
        :return:
        """
        def _getattr(_obj, _attr):
            return getattr(_obj, _attr)
        return functools.reduce(_getattr, [obj] + attr.split('.'))

    def evaluate_expr_to_int(expression: str) -> int:
        """
        From an expression like 4 * 5 - 1, evaluate
        :param expression:
        :return:
        """

        def _reval(node):
            if isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.BinOp):
                return SUPPORTED_OPERATORS[type(node.op)](_reval(node.left), _reval(node.right))
            elif isinstance(node, ast.UnaryOp):
                return SUPPORTED_OPERATORS[type(node.op)](_reval(node.operand))
            else:
                raise TypeError(node)

        # Compute the abstract syntax tree and evaluate
        return int(_reval(ast.parse(expression, mode="eval").body))

    interpolation_re = regex.compile(r"\$([^\s]+)")
    onnx_variables = []

    # Iterate over all the variables
    for variable in variables:

        # If the repeated is a string, we need to process to get an int
        if isinstance(variable.repeated, str):

            # We need to interpolate some variables, iteratively until there is no more interpolation key
            if "$" in variable.repeated:

                # Look for the first interpolation key
                interpolation_key = interpolation_re.search(variable.repeated)
                while interpolation_key is not None:

                    # Get the interpolation key's value
                    interpolation_value = rgetattr(model, interpolation_key.group(1))

                    # Regenerate the variable with the interpolated value
                    variable = OnnxVariable(
                        variable.name,
                        variable.axes,
                        variable.repeated.replace(f"${interpolation_key.group(1)}", str(interpolation_value))
                    )

                    # Check if anything remains (another key)
                    interpolation_key = interpolation_re.search(variable.repeated)

            # str values purpose is to refer to dynamic values from within the model, evaluated at runtime
            # better to use just an int otherwise if the field is constant
            else:
                raise ValueError(f"Invalid value {variable.repeated} not starting with $")

            # Evaluate the expression to an int
            repeated = evaluate_expr_to_int(variable.repeated)


        # We accept only string and int, so else branch should always be int
        elif isinstance(variable.repeated, int):
            repeated = variable.repeated
        else:
            raise ValueError(
                f"Invalid type for repeated property, should be either int or str, "
                f"got {type(variable.repeated)}"
            )

        # Generate the variables by suffixing the name with ".{index}" and setting repeated = 1
        if repeated > 1:
            repeated_vars = [
                OnnxVariable(name=f"{variable.name}.{index}", axes=variable.axes, repeated=1)
                for index in range(repeated)
            ]

            # Append all the variable to
            onnx_variables += repeated_vars
        else:
            onnx_variables.append(variable)

    return onnx_variables


def ensure_model_and_config_inputs_match(model_inputs: Iterable[str], config_inputs: Iterable[OnnxVariable]) -> Tuple[bool, List[OnnxVariable]]:
    """

    :param model_inputs:
    :param config_inputs:
    :return:
    """
    model_inputs_set, config_inputs_set = set(model_inputs), set(map(lambda var: var.name, config_inputs))

    # We are fine if config_inputs has less keys than model_inputs
    is_ok = config_inputs_set.issubset(model_inputs_set)

    # Make sure the input order match
    matching_inputs = config_inputs_set.intersection(model_inputs_set)
    ordered_matching_inputs = [input for input in config_inputs if input.name in matching_inputs]
    return is_ok, ordered_matching_inputs

def convert_pytorch(model: PreTrainedModel, config: OnnxConfig, opset: int, output: Path):
    """
    Export a PyTorch backed pipeline to ONNX Intermediate Representation (IR

    Args:
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
    inputs_match, ordered_onnx_inputs = ensure_model_and_config_inputs_match(
        model.dummy_inputs.keys(),
        onnx_inputs
    )

    if not inputs_match:
        raise ValueError("Model and config inputs doesn't match")

    # export can works with named args but the dict containing named args as to be last element of the args tuple
    export(
        model,
        (model.dummy_inputs, ),
        f=output.as_posix(),
        input_names=[var.name for var in ordered_onnx_inputs],
        output_names=[var.name for var in onnx_outputs],
        dynamic_axes={var.name: var.axes for var in chain(config.inputs, config.outputs)},
        do_constant_folding=True,
        use_external_data_format=config.use_external_data_format,
        enable_onnx_checker=True,
        opset_version=opset,
    )


def validate_model_outputs(reference_model: Union[PreTrainedModel, TFPreTrainedModel], onnx_model: Path, onnx_outputs: List[OnnxVariable], atol: float):
    from onnxruntime import InferenceSession, SessionOptions

    print("Validating ONNX model...")

    options = SessionOptions()
    session = InferenceSession(onnx_model.as_posix(), options)

    # Compute outputs from the reference model
    ref_outputs = reference_model(**reference_model.dummy_inputs)

    # Compute outputs from the ONNX model
    onnx_inputs = {name: tensor.numpy() for name, tensor in reference_model.dummy_inputs.items()}
    onnx_outputs_name = [var.name for var in onnx_outputs]
    onnx_outputs = zip(onnx_outputs_name, session.run(onnx_outputs_name, onnx_inputs))

    # Check we have a subset of the keys into onnx_outputs against ref_outputs
    ref_outputs_set, onnx_outputs_set = set(ref_outputs.keys()), set(onnx_outputs_name)
    if not onnx_outputs_set.issubset(ref_outputs_set):
        raise ValueError(
            "Outputs doesn't match between reference model and ONNX exported model: "
            f"{onnx_outputs_set.difference(ref_outputs_set)}"
        )

    # Check the shape and values match
    for name, ort_value in onnx_outputs:
        ref_value = ref_outputs[name].numpy()

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


if __name__ == '__main__':
    parser = ArgumentParser("Hugging Face ONNX Exporter tool")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model's name of path on disk to load.")
    parser.add_argument("-f", "--framework", choices=FRAMEWORK_CHOICES, required=True, help=f"Framework to use when exporting. Possible values are: {FRAMEWORK_CHOICES}")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version to export the model with (default 12).")
    parser.add_argument("--atol", type=float, default=1e-5, help="Absolute difference tolerence when validating the model,")
    parser.add_argument("output", type=Path, help="Path indicating where to store generated ONNX model.")

    # Retrieve CLI arguments
    args = parser.parse_args()

    print(f"About to export model: {args.model} using framework: {args.framework}")

    # Allocate the model
    model = (AutoModel if args.framework == FRAMEWORK_NAME_PT else TFAutoModel).from_pretrained(args.model)
    model_kind, onnx_config = check_supported_model_or_raise(model, features=args.features)

    # Override model's config if needed
    if onnx_config.runtime_config_overrides is not None:
        print("Overriding model's config values:")
        for config_key, config_value in onnx_config.runtime_config_overrides.items():
            print(f"\t- {config_key} => {config_value}")
            setattr(model.config, config_key, config_value)

    # Ensure the requested opset is sufficient
    if args.opset < onnx_config.minimum_required_onnx_opset:
        raise ValueError(
            f"Opset {args.opset} is not sufficient to export {model_kind}. "
            f"At least  {onnx_config.minimum_required_onnx_opset} is required."
        )

    if args.framework == FRAMEWORK_NAME_PT:
        convert_pytorch(model, onnx_config, args.opset, args.output.joinpath("model.onnx"))
    else:
        raise NotImplementedError()

    # TODO: store output path correctly
    validate_model_outputs(model, args.output.joinpath("model.onnx"), onnx_config.outputs, args.atol)
    print()

