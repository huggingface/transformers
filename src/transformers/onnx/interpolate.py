import ast
import functools
import operator as op
from typing import Any, Dict, List, Union

import regex

from .. import BatchEncoding, PreTrainedModel, TensorType, TFPreTrainedModel
from .config import OnnxVariable


# Regular expression used to match interpolation keys
INTERPOLATION_RE = regex.compile(r"\$([^\s]+)")

# Supported operators when parsing OnnxVariable repeated field (supports +, -, *, //)
# Set of supported operator while evaluating mathematical expression
SUPPORTED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
}


def evaluate_expr_to_int(expression: str) -> int:
    """
    From an expression like 4 * 5 - 1, evaluate the following using Python's AST (abstract syntaxic tree)

    Args:
        expression: A string representation of a "simple" mathematical expression

    Raises:
        TypeError if expression contains invalid or not allowed operators

    Returns:
        Evaluated final int value
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
    """
    Interpolate the string expression with the properties from the model. Expressions can be of different forms:

        - Simple mathematical
        - Mono variable from model's property: $config.hidden_size
        - Multiple variables from model's property: $config.hidden_size * $config.num_heads
        - Mono variable from model's property with constant: $config.hidden_size + 1

    The variable present within the expression are replaced at runtime with the value from the model properties such
    that: `$config.hidden_size * 4` would be replaced by (for instance) `768 * 4`

    Args:
        expression: A string expression to be evaluated (i.e. interpolated)
        model: The model from which property will be taken

    Returns:
        str with interpolated (i.e. replaced) value(s) from model's properties
    """

    def rgetattr(obj, attr):
        """
        Get an attribute recursively (i.e. multiple nested object obj.property_x.propert_y) :param obj: :param attr:
        :return:
        """

        def _getattr(_obj, _attr):
            return getattr(_obj, _attr)

        return functools.reduce(_getattr, [obj] + attr.split("."))

    # Copy the expression string to operate on it
    interpolated_expr = expression

    # Look for the first interpolation key
    interpolation_key = INTERPOLATION_RE.search(expression)
    while interpolation_key is not None:

        # Get the interpolation key's value
        interpolation_value = rgetattr(model, interpolation_key.group(1))

        # Regenerate the expression with the interpolated value
        interpolated_expr = interpolated_expr.replace(f"${interpolation_key.group(1)}", str(interpolation_value))

        # Check if anything remains (another key)
        interpolation_key = INTERPOLATION_RE.search(interpolated_expr)

    return interpolated_expr


def expand_repeated_onnx_variables(
    model: Union["PreTrainedModel", "TFPreTrainedModel"], variables: List["OnnxVariable"]
) -> List["OnnxVariable"]:
    """
    From a list of OnnxVariable, which may or may not have repeated field, this method will expand (i.e repeat)
    individual variables the number of time specified by the field `repeat`.

    As an example, the following list of OnnxVariable: ```python in_variables = [ OnnxVariable(name="input_ids", ...,
    "repeat"=1), OnnxVariable(name="past_keys", ..., "repeat"="$config.num_encoder_layers"), ] ```

    would be expanded as the following: ```python out_variables = [ OnnxVariable(name="input_ids", ..., "repeat"=1),
    OnnxVariable(name="past_keys.0", ..., "repeat"=1), OnnxVariable(name="past_keys.1", ..., "repeat"=1), # ... n
    times, with n = $config.num_encoder_layers OnnxVariable(name="past_keys.(n - 1)", ..., "repeat"=1), ] ```

    Args:
        model: Model to be used a reference to interpolate potential expressions from
        variables: The list of `OnnxVariable` with repeated value >= 1

    Returns:
        List of OnnxVariable with potential individual OnnxVariable which might have been repeated

    """
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
                    variable.value,
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
                f"Invalid type for repeated property, should be either int or str, " f"got {type(variable.repeated)}"
            )

        # Generate the variables by suffixing the name with ".{index}" and setting repeated = 1
        if repeated > 1:
            repeated_vars = [
                OnnxVariable(name=f"{variable.name}.{index}", axes=variable.axes, repeated=1, value=variable.value)
                for index in range(repeated)
            ]

            # Append all the variable to
            onnx_variables += repeated_vars
        else:
            onnx_variables.append(variable)

    return onnx_variables


def insert_additional_onnx_value_within_inputs(
    inputs: Union[BatchEncoding, Dict[str, Any]], onnx_variables: List["OnnxVariable"], tensor_type: TensorType
) -> Dict[str, Any]:
    """
    Introduce constant inputs for a model from a non None `value` of `OnnxVariable`. For instance, the taking the
    following list of OnnxVariable: ```python in_variables = [ OnnxVariable(name="input_ids", ..., repeat=1,
    value=None), OnnxVariable(name="decoder_input_ids", repeat=1, value=[1, 2, 3]) ] ```

    would generate the following model's inputs:

    ```python out_variables = { "input_ids": Tensor([...]), "decoder_input_ids": Tensor([1, 2, 3]) } ```

    Args:
        inputs: (BatchEncoding or Dict[str, Any]) List of inputs as generated by the model's tokenizer
        onnx_variables: (List[OnnxVariable]) List of OnnxVariable input to be compared against model's tokenizer input
        tensor_type: The type of tensor the constant value will be created for

    Returns:
        (Dict[str, Any]) Inputs of the model with potential `OnnxVariable` appended
    """
    for onnx_var in onnx_variables:
        if onnx_var.name not in inputs and onnx_var.value is not None:
            encoding = BatchEncoding({onnx_var.name: onnx_var.value}).convert_to_tensors(
                tensor_type=tensor_type, prepend_batch_axis=True
            )

            inputs[onnx_var.name] = encoding[onnx_var.name]

    return inputs
