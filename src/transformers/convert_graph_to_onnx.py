from argparse import ArgumentParser, ArgumentTypeError
from os import listdir, makedirs
from os.path import abspath, dirname, exists
from typing import Dict, List, Optional, Tuple

from transformers import AutoModel, AutoTokenizer, is_tf_available, is_torch_available
from transformers.tokenization_utils import BatchEncoding


class OnnxConverterArgumentParser(ArgumentParser):
    """
    Wraps all the script arguments supported to export transformers models to ONNX IR
    """

    def __init__(self):
        super(OnnxConverterArgumentParser, self).__init__("ONNX Converter")

        self.add_argument("--model", type=str, required=True, help="Model's id or path (ex: bert-base-cased)")
        self.add_argument("--tokenizer", type=str, help="Tokenizer's id or path (ex: bert-base-cased)")
        self.add_argument("--framework", type=str, choices=["pt", "tf"], help="Framework for loading the model")
        self.add_argument("--opset", type=int, default=11, help="ONNX opset to use")
        self.add_argument("--check-loading", action="store_true", help="Check ONNX is able to load the model")
        self.add_argument("--use-external-format", action="store_true", help="Allow exporting model >= than 2Gb")
        self.add_argument("output", type=abspath)


def get_tokenizer_and_model(tokenizer_name, model_name):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def ensure_onnx_requirements_met(framework, use_external_format):
    # Check the wanted framework is available
    if framework == "pt" and not is_torch_available():
        raise Exception("Cannot convert because PyTorch is not installed. Please install torch first.")
        import torch

        if use_external_format and (torch.__version__ < "1.5.0"):
            raise ArgumentTypeError(
                f"use-external-format requires torch>=1.5.0\n"
                f"Update your pytorch version, or if your model is <2Gb set '--use-external-format false'"
            )
    if framework == "tf" and not is_tf_available():
        raise Exception("Cannot convert because TF is not installed. Please install tensorflow first.")


def ensure_valid_input(model, tokens, input_names):
    """
    Ensure input are presented in the correct order, without any None
    Args:
        model: The model used to forward the input data
        tokens: BatchEncoding holding the input data
        input_names: The name of the inputs

    Returns: Tuple

    """
    print("Ensuring inputs are in correct order")
    model_args_name = model.forward.__code__.co_varnames
    model_args, ordered_input_names = [], []

    for arg_name in model_args_name[1:]:  # start at index 1 to skip "self" argument
        if arg_name in input_names:
            ordered_input_names.append(arg_name)
            model_args.append(tokens[arg_name])
        else:
            print("{} is not present in the generated input list.".format(arg_name))
            break

    print("Generated inputs order: {}".format(ordered_input_names))
    return ordered_input_names, tuple(model_args)


def infer_shapes(tokenizer, model, framework: str) -> Tuple[List[str], List[str], Dict, BatchEncoding]:
    def build_shape_dict(name: str, tensor, is_input: bool, seq_len: int):
        if isinstance(tensor, (tuple, list)):
            return [build_shape_dict(name, t, is_input, seq_len) for t in tensor]

        else:
            # Let's assume batch is the first axis with only 1 element (~~ might not be always true ...)
            axes = {[axis for axis, numel in enumerate(tensor.shape) if numel == 1][0]: "batch"}
            if is_input:
                if len(tensor.shape) == 2:
                    axes[1] = "sequence"
                else:
                    raise ValueError("Unable to infer tensor axes ({})".format(len(tensor.shape)))
            else:
                seq_axes = [dim for dim, shape in enumerate(tensor.shape) if shape == seq_len]
                axes.update({dim: "sequence" for dim in seq_axes})

        print("Found {} {} with shape: {}".format("input" if is_input else "output", name, axes))
        return axes

    tokens = tokenizer("This is a sample output", return_tensors=framework)
    seq_len = tokens.input_ids.shape[-1]
    outputs = model(**tokens) if framework == "pt" else model(tokens)

    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)

    # Generate input names & axes
    input_vars = list(tokens.keys())
    input_dynamic_axes = {k: build_shape_dict(k, v, True, seq_len) for k, v in tokens.items()}

    # flatten potentially grouped outputs (past for gpt2, attentions)
    outputs_flat = []
    for output in outputs:
        if isinstance(output, (tuple, list)):
            outputs_flat.extend(output)
        else:
            outputs_flat.append(output)

    # Generate output names & axes
    output_names = ["output_{}".format(i) for i in range(len(outputs_flat))]
    output_dynamic_axes = {k: build_shape_dict(k, v, False, seq_len) for k, v in zip(output_names, outputs_flat)}

    # Create the aggregated axes representation
    dynamic_axes = dict(input_dynamic_axes, **output_dynamic_axes)
    return input_vars, output_names, dynamic_axes, tokens


def convert_pytorch(tokenizer, model, model_name: str, opset: int, output: str, use_external_format: bool):
    import torch
    from torch.onnx import export

    print("Using framework PyTorch: {}".format(torch.__version__))

    with torch.no_grad():
        input_names, output_names, dynamic_axes, tokens = infer_shapes(tokenizer, model, "pt")
        ordered_input_names, model_args = ensure_valid_input(model, tokens, input_names)
        kwargs = {
            "f": output,
            "input_names": ordered_input_names,
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
            "do_constant_folding": True,
            "enable_onnx_checker": True,
            "opset_version": opset,
        }

        # Older versions of pytorch do not have the use_external_data_format option
        if use_external_format:
            kwargs["use_external_data_format"] = True

        export(model, model_args, **kwargs)


def convert_tensorflow(tokenizer, model, model_name: str, opset: int, output: str):
    print("/!\\ Please note TensorFlow doesn't support exporting model > 2Gb /!\\")

    try:
        import tensorflow as tf
        from keras2onnx import convert_keras, save_model, __version__ as k2ov

        print("Using framework TensorFlow: {}, keras2onnx: {}".format(tf.version.VERSION, k2ov))

        # Build
        input_names, output_names, dynamic_axes, tokens = infer_shapes(tokenizer, model, "tf")

        # Forward
        model.predict(tokens.data)
        onnx_model = convert_keras(model, model_name, target_opset=opset)
        save_model(onnx_model, output)

    except ImportError as e:
        raise Exception(
            "Cannot import {} required to convert TF model to ONNX. Please install {} first.".format(e.name, e.name)
        )


def convert(
    framework: str,
    model_name: str,
    output: str,
    opset: int,
    tokenizer_name: Optional[str] = None,
    use_external_format: bool = False,
):
    print("ONNX opset version set to: {}".format(opset))
    ensure_onnx_requirements_met(framework, use_external_format)

    if tokenizer_name is None:
        tokenizer_name = model_name

    # Getting models
    tokenizer, model = get_tokenizer_and_model(tokenizer_name, model_name)

    # Generating file structure
    parent = dirname(output)
    if not exists(parent):
        print("Creating folder {}".format(parent))
        makedirs(parent)
    elif len(listdir(parent)) > 0:
        raise Exception("Folder {} is not empty, aborting conversion".format(parent))

    # Export the graph
    if framework == "pt":
        convert_pytorch(tokenizer, model, model_name, opset, output, use_external_format)
    else:
        convert_tensorflow(tokenizer, model, model_name, opset, output)


def verify(path: str):
    from onnxruntime import InferenceSession, SessionOptions
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException

    print("Checking ONNX model loading from: {}".format(path))
    try:
        onnx_options = SessionOptions()
        _ = InferenceSession(path, onnx_options, providers=["CPUExecutionProvider"])
        print("Model correctly loaded")
    except RuntimeException as re:
        print("Error while loading the model: {}".format(re))


if __name__ == "__main__":
    parser = OnnxConverterArgumentParser()
    args = parser.parse_args()

    try:
        # Convert
        convert(
            args.framework, args.model, args.output, args.opset, args.tokenizer, args.use_external_format,
        )

        # And verify
        if args.check_loading:
            verify(args.output)
    except Exception as e:
        print("Error while converting the model: {}".format(e))
        exit(1)
