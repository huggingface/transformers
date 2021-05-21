from argparse import ArgumentParser
from itertools import chain
from pathlib import Path

from typing import Tuple, List, Iterable, Union, Any, Dict

import ast
import functools
import numpy as np
import operator as op
import regex
from packaging.version import parse, Version

from transformers import AutoModel, TFAutoModel, PreTrainedModel, is_torch_available, TFPreTrainedModel, AutoTokenizer, \
    PreTrainedTokenizer, TensorType, BatchEncoding
from transformers.configuration_utils import OnnxConfig, OnnxVariable
from transformers.models.albert import ALBERT_ONNX_CONFIG
from transformers.models.bart import BART_ONNX_CONFIG, BART_ONNX_CONFIG_WITH_PAST
from transformers.models.bert import BERT_ONNX_CONFIG
from transformers.models.distilbert import DISTILBERT_ONNX_CONFIG
from transformers.models.gpt2 import GPT2_ONNX_CONFIG, GPT2_ONNX_CONFIG_WITH_PAST
from transformers.models.roberta import ROBERTA_ONNX_CONFIG
from transformers.models.xlm_roberta import XLM_ROBERTA_ONNX_CONFIG

from onnxruntime import GraphOptimizationLevel


# Regular expression used to match interpolation keys
INTERPOLATION_RE = regex.compile(r"\$([^\s]+)")


# This is the minimal required version to
# support some ONNX Runtime features
ORT_QUANTIZE_MINIMUM_VERSION = parse("1.4.0")

# Set of frameworks we can export from
FRAMEWORK_NAME_PT = "pytorch"
FRAMEWORK_NAME_TF = "tensorflow"
FRAMEWORK_CHOICES = {FRAMEWORK_NAME_PT, FRAMEWORK_NAME_PT}

# Set of model topologies we support
SUPPORTED_MODEL_KIND = {
    "albert": {
        "default": ALBERT_ONNX_CONFIG
    },
    "bart": {
        "default": BART_ONNX_CONFIG,
        "with_past": BART_ONNX_CONFIG_WITH_PAST
    },
    "bert": {
        "default": BERT_ONNX_CONFIG
    },
    "distilbert": {
        "default": DISTILBERT_ONNX_CONFIG,
    },
    "gpt2": {
        "default": GPT2_ONNX_CONFIG,
        "with_past": GPT2_ONNX_CONFIG_WITH_PAST
    },
    "longformer": {
        "default": LONGFORMER_ONNX_CONFIG,
    },
    "roberta": {
        "default": ROBERTA_ONNX_CONFIG,
    },
    "t5": {
        "default": T5_ONNX_CONFIG,
    },
    "xlm-roberta": {
        "default": XLM_ROBERTA_ONNX_CONFIG
    }
}

# Supported operators when parsing OnnxVariable repeated field (supports +, -, *, //)
SUPPORTED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}

# ONNX Runtime optimization levels for humans
ONNX_OPTIMIZATION_LEVELS = {
    "disabled": GraphOptimizationLevel.ORT_DISABLE_ALL,
    "default": GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": GraphOptimizationLevel.ORT_ENABLE_ALL
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


def check_supported_model_or_raise(model: Union[PreTrainedModel, TFPreTrainedModel], features: str = "default") -> Tuple[str, OnnxConfig]:
    model_kind = type(model).__name__
    if model_kind not in SUPPORTED_MODEL_KIND:
        raise KeyError(
            f"{model_kind} ({args.model}) is not supported yet. "
            f"Only {SUPPORTED_MODEL_KIND} are supported. "
            f"If you want to support ({model_kind}) please propose a PR or open up an issue."
        )

    # Look for the features
    model_features = SUPPORTED_MODEL_KIND[model_kind]
    if features not in model_features:
        raise ValueError(
            f"{model_kind} doesn't support features {features}. "
            f"Supported values are: {list(model_features.keys())}"
        )

    return model_kind, SUPPORTED_MODEL_KIND[model_kind][features]


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


def interpolate_expression(expression: str, model: Union[PreTrainedModel, TFPreTrainedModel]) -> str:
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

    # Copy the expression string to operate on it
    interpolated_expr = expression

    # Look for the first interpolation key
    interpolation_key = INTERPOLATION_RE.search(expression)
    while interpolation_key is not None:

        # Get the interpolation key's value
        interpolation_value = rgetattr(model, interpolation_key.group(1))

        # Regenerate the expression with the interpolated value
        interpolated_expr = interpolated_expr.replace(
            f"${interpolation_key.group(1)}",
            str(interpolation_value)
        )

        # Check if anything remains (another key)
        interpolation_key = INTERPOLATION_RE.search(interpolated_expr)

    return interpolated_expr


def expand_repeated_onnx_variables(model: Union[PreTrainedModel, TFPreTrainedModel], variables: List[OnnxVariable]) -> List[OnnxVariable]:
    onnx_variables = []

    # Iterate over all the variables
    for variable in variables:

        # If the repeated is a string, we need to process to get an int
        if isinstance(variable.repeated, str):

            # We need to interpolate some variables, iteratively until there is no more interpolation key
            if "$" in variable.repeated:
                variable = OnnxVariable(
                    variable.name,
                    variable.axes,
                    interpolate_expression(variable.repeated, model),  # Interpolate from model instance
                    variable.value
                )

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


def flatten_output_collection_property(name: str, field: Iterable[Any]) -> Dict[str, Any]:
    from itertools import chain
    return {
        f"{name}.{idx}": item
        for idx, item in enumerate(chain.from_iterable(field))
    }


def ensure_model_and_config_inputs_match(model_inputs: Iterable[str], config_inputs: Iterable[OnnxVariable]) -> Tuple[bool, List[OnnxVariable]]:
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


def insert_additional_onnx_value_within_inputs(inputs: Union[BatchEncoding, Dict[str, Any]], onnx_variables: List[OnnxVariable], tensor_type: TensorType) -> Dict[str, Any]:
    for onnx_var in onnx_variables:
        if onnx_var.name not in inputs and onnx_var.value is not None:
            encoding = BatchEncoding(
                {onnx_var.name: onnx_var.value}
            ).convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=True)

            inputs[onnx_var.name] = encoding[onnx_var.name]

    return inputs


def convert_pytorch(tokenizer: PreTrainedTokenizer, model: PreTrainedModel, config: OnnxConfig, opset: int, output: Path) -> Tuple[List[OnnxVariable], List[OnnxVariable]]:
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

    inputs_match, ordered_onnx_inputs = ensure_model_and_config_inputs_match(
        model_inputs.keys(),
        onnx_inputs
    )

    if not inputs_match:
        raise ValueError("Model and config inputs doesn't match")

    # export can works with named args but the dict containing named args as to be last element of the args tuple
    export(
        model,
        (dict(model_inputs), ),
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


def optimize(onnx_model_path: Path, model: Union[PreTrainedModel, TFPreTrainedModel], onnx_config: OnnxConfig, optimization_level: GraphOptimizationLevel, use_gpu: bool, output: Path):
    from onnxruntime.transformers.optimizer import optimize_model, optimize_by_onnxruntime
    from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions

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
            **additional_args
        )

        print(f"Optimization statistics: {optimizer.get_fused_operator_statistics()}")
        optimizer.save_model_to_file(output.as_posix())

    # Else use online ONNX Runtime optimization
    else:
        from os import replace
        temp_output_path = Path(optimize_by_onnxruntime(onnx_model_path.as_posix()))
        replace(temp_output_path, output)


def validate_model_outputs(tokenizer: PreTrainedTokenizer, reference_model: Union[PreTrainedModel, TFPreTrainedModel], onnx_model: Path, onnx_inputs: List[OnnxVariable], onnx_named_outputs: List[OnnxVariable], atol: float):
    from onnxruntime import InferenceSession, SessionOptions

    print("Validating ONNX model...")

    # TODO: Sequence length = 4 hard coded, provide this value through CLI would be better
    reference_tensor_type = TensorType.PYTORCH if isinstance(reference_model, PreTrainedModel) else TensorType.TENSORFLOW
    reference_model_inputs = tokenizer([tokenizer.unk_token] * 4, is_split_into_words=True, return_tensors=reference_tensor_type)
    onnx_model_inputs = tokenizer([tokenizer.unk_token] * 4, is_split_into_words=True, return_tensors=TensorType.NUMPY)

    # Check if we need to introduce some more variables
    reference_model_inputs = insert_additional_onnx_value_within_inputs(reference_model_inputs, onnx_inputs, TensorType.PYTORCH)
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


if __name__ == '__main__':
    parser = ArgumentParser("Hugging Face ONNX Exporter tool")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model's name of path on disk to load.")
    parser.add_argument("-f", "--framework", choices=FRAMEWORK_CHOICES, required=True, help=f"Framework to use when exporting. Possible values are: {FRAMEWORK_CHOICES}")
    parser.add_argument("--features", choices=["default", "with_past"], default="default", help="Export the model with some additional features.")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version to export the model with (default 12).")
    parser.add_argument("--optimize", action="store_true", help="Flag indicating if we should try to optimize the model.")
    parser.add_argument("--use-gpu", action="store_true", help="Flag indicating if we should try to optimize the model for GPU inference.")
    parser.add_argument("--optimization-level", choices=ONNX_OPTIMIZATION_LEVELS.keys(), default="disabled", help="Flag indicating if we should try to optimize the model.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute difference tolerence when validating the model,")
    parser.add_argument("output", type=Path, help="Path indicating where to store generated ONNX model.")

    # Retrieve CLI arguments
    args = parser.parse_args()
    args.output = args.output if args.output.is_file() else args.output.joinpath("model.onnx")

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    print(f"About to export model: {args.model} using framework: {args.framework}")

    # Allocate the model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
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
        onnx_inputs, onnx_outputs = convert_pytorch(tokenizer, model, onnx_config, args.opset, args.output)
    else:
        raise NotImplementedError()

    validate_model_outputs(tokenizer, model, args.output, onnx_inputs, onnx_outputs, args.atol)
    print(f"All good, model saved at: {args.output.as_posix()}")

    if args.optimize and args.optimization_level != "disabled":
        print(f"About to optimize model with optimization_level: {args.optimization_level}")

        args.opt_model_output = generate_identified_filename(args.output, f"_optimized_{args.optimization_level}")
        args.optimization_level = ONNX_OPTIMIZATION_LEVELS[args.optimization_level]
        optimize(args.output, model, onnx_config, args.optimization_level, args.use_gpu, args.opt_model_output)

        if not args.use_gpu:
            validate_model_outputs(tokenizer, model, args.opt_model_output, onnx_inputs, onnx_outputs, args.atol)
        else:
            print(
                "Validating model targeting GPU is not supported yet. "
                "Please, fill an issue or submit a PR if it's something you need."
            )

        print(f"Optimized model saved at: {args.opt_model_output.as_posix()}")
