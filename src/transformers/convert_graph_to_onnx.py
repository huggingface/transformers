from argparse import ArgumentParser, Namespace
from os import listdir, makedirs
from os.path import abspath, dirname, exists
from typing import Dict, List, Tuple, Optional

from transformers import is_tf_available, is_torch_available
from transformers.pipelines import SUPPORTED_TASKS, Pipeline, pipeline
from transformers.tokenization_utils import BatchEncoding


class OnnxConverterArgumentParser(ArgumentParser):
    """
    Wraps all the script arguments supported to export transformers models to ONNX IR
    """

    def __init__(self):
        super(OnnxConverterArgumentParser, self).__init__("ONNX Converter")

        self.add_argument("--model", type=str, required=True, help="Model's id or path (ex: bert-base-cased)")
        self.add_argument("--tokenizer", type=str, help="Tokenizer's id or path (ex: bert-base-cased)")
        self.add_argument("--task", type=str, default=None, choices=list(SUPPORTED_TASKS.keys()), help="Model's task")
        self.add_argument("--framework", type=str, choices=["pt", "tf"], help="Framework for loading the model")
        self.add_argument("--opset", type=int, default=-1, help="ONNX opset to use (-1 = latest)")
        self.add_argument("--check-loading", action="store_true", help="Check ONNX is able to load the model")
        self.add_argument("output")


def infer_shapes(nlp: Pipeline, framework: str) -> Tuple[List[str], List[str], Dict, BatchEncoding]:
    def build_shape_dict(tensor, is_input: bool, seq_len: int):
        axes = {0: "batch"}
        if is_input:
            if len(tensor.shape) == 2:
                axes[1] = "sequence"
            else:
                raise ValueError("Unable to infer tensor axes ({})".format(len(tensor.shape)))
        else:
            seq_axes = [dim for dim, shape in enumerate(tensor.shape) if shape == seq_len]
            axes.update({dim: "sequence" for dim in seq_axes})

        return axes

    tokens = nlp.tokenizer.encode_plus("This is a sample output", return_tensors=framework)
    seq_len = tokens.input_ids.shape[-1]
    outputs = nlp.model(**tokens) if args.framework == "pt" else nlp.model(tokens)

    if not isinstance(outputs, (list, tuple)):
        outputs = (outputs,)

    # Generate names
    output_names = ["output_{}".format(i) for i in range(len(outputs))]
    input_vars = list(tokens.keys())

    # Define dynamic axes
    input_dynamic_axes = {k: build_shape_dict(v, True, seq_len) for k, v in tokens.items()}
    output_dynamic_axes = {k: build_shape_dict(v, False, seq_len) for k, v in zip(output_names, outputs)}
    dynamic_axes = dict(input_dynamic_axes, **output_dynamic_axes)
    return input_vars, output_names, dynamic_axes, tokens


def load_graph_from_args(task: str, framework: str, model: str, tokenizer: Optional[str] = None) -> Pipeline:
    # If no tokenizer provided
    if tokenizer is None:
        tokenizer = args.model

    print("Loading pipeline (task: {}, model: {}, tokenizer: {})".format(task, model, tokenizer))

    # Allocate tokenizer and model
    return pipeline(task, model=model, framework=framework)


def convert_pytorch(nlp: Pipeline, opset: int, output: str):
    if not is_torch_available():
        print("Cannot convert because PyTorch is not installed. Please install torch first.")
        exit(1)

    import torch
    from torch.onnx import export

    print("PyTorch: {}".format(torch.__version__))

    with torch.no_grad():
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, "pt")
        tokens = tuple(tokens[key] for key in input_names)  # Need to be ordered
        export(
            nlp.model,
            tokens,
            f=output,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
            use_external_data_format=True,
            enable_onnx_checker=True,
            opset_version=opset,
        )


def convert_tensorflow(nlp: Pipeline, opset: int, output: str):
    if not is_tf_available():
        print("Cannot convert {} because TF is not installed. Please install torch first.".format(args.model))
        exit(1)

    print("Please note TensorFlow doesn't support exporting model > 2Gb")

    try:
        import tensorflow as tf
        from keras2onnx import convert_keras, save_model, __version__ as k2ov

        print("TensorFlow: {}, keras2onnx: {}".format(tf.version.VERSION, k2ov))

        # Build
        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, "tf")

        # Forward
        nlp.model.predict(list(tokens.data.values()))
        onnx_model = convert_keras(nlp.model, nlp.model.name, target_opset=opset)
        save_model(onnx_model, output)

    except ImportError as e:
        print("Cannot import {} required to convert TF model to ONNX. Please install {} first.".format(e.name, e.name))
        exit(1)


def convert(task: str, framework: str, model: str, tokenizer: Optional[str], opset: int, output: str):
    if opset == -1:
        from onnx.defs import onnx_opset_version

        print("Setting ONNX opset version to: {}".format(onnx_opset_version()))
        opset = onnx_opset_version()

    # Load the pipeline
    nlp = load_graph_from_args(task, framework, model, tokenizer)

    parent = dirname(output)
    if not exists(parent):
        print("Creating folder {}".format(parent))
        makedirs(parent)
    elif len(listdir(parent)) > 0:
        print("Folder {} is not empty, aborting conversion".format(parent))
        exit(1)

    # Export the graph
    if args.framework == "pt":
        convert_pytorch(nlp, opset, output)
    else:
        convert_tensorflow(nlp, opset, output)


def verify(path: str):
    from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
    from onnxruntime.capi.onnxruntime_pybind11_state import RuntimeException

    print("Checking ONNX model loading from: {}".format(path))
    try:
        onnx_options = SessionOptions()
        onnx_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        _ = InferenceSession(path, onnx_options, providers=["CPUExecutionProvider"])
        print("Model correctly loaded")
    except RuntimeException as re:
        print("Error while loading the model: {}".format(re))


if __name__ == "__main__":
    parser = OnnxConverterArgumentParser()
    args = parser.parse_args()

    # Make sure output is absolute path
    args.output = abspath(args.output)

    # Convert
    convert(args.task, args.framework, args.model, args.tokenizer, args.opset, args.output)

    # And verify
    if args.check_loading:
        verify(args.output)
