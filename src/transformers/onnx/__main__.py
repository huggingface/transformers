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

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Tuple, Union

from onnxruntime import GraphOptimizationLevel
from transformers.file_utils import is_coloredlogs_available, is_sympy_available
from transformers.models.albert import AlbertOnnxConfig
from transformers.models.auto import AutoTokenizer
from transformers.models.bart import BartOnnxConfig
from transformers.models.bert import BertOnnxConfig
from transformers.models.distilbert import DistilBertOnnxConfig
from transformers.models.gpt2 import GPT2OnnxConfig
from transformers.models.roberta import RobertaOnnxConfig
from transformers.models.t5 import T5OnnxConfig
from transformers.models.xlm_roberta import XLMRobertaOnnxConfig

from .. import is_tf_available, is_torch_available
from ..utils import logging
from .convert import convert_pytorch, optimize, validate_model_outputs
from .utils import generate_identified_filename


# Set of frameworks we can export from
FRAMEWORK_NAME_PT = "pytorch"
FRAMEWORK_NAME_TF = "tensorflow"
FRAMEWORK_CHOICES = {FRAMEWORK_NAME_PT, FRAMEWORK_NAME_PT}

if is_torch_available():
    from transformers import AutoModel, AutoModelForTokenClassification, PreTrainedModel

    FEATURES_TO_AUTOMODELS = {
        "default": AutoModel,
        "with_past": AutoModel,
        "token_classification": AutoModelForTokenClassification,
    }

if is_tf_available():
    from transformers import TFAutoModel, TFAutoModelForTokenClassification, TFPreTrainedModel

    FEATURES_TO_TF_AUTOMODELS = {
        "default": TFAutoModel,
        "with_path": TFAutoModel,
        "token_classification": TFAutoModelForTokenClassification,
    }

# Set of model topologies we support associated to the features supported by each topology and the factory
SUPPORTED_MODEL_KIND = {
    "albert": {"default": AlbertOnnxConfig.default},
    "bart": {"default": BartOnnxConfig.default, "with_past": BartOnnxConfig.with_past},
    "bert": {"default": BertOnnxConfig.default},
    "distilbert": {"default": DistilBertOnnxConfig.default},
    "gpt2": {"default": GPT2OnnxConfig.default, "with_past": GPT2OnnxConfig.with_past},
    "roberta": {"default": RobertaOnnxConfig},
    "t5": {"default": T5OnnxConfig.default, "with_past": T5OnnxConfig.with_past},
    "xlm-roberta": {"default": XLMRobertaOnnxConfig.default},
}

# ONNX Runtime optimization levels for humans
ONNX_OPTIMIZATION_LEVELS = {
    "disabled": GraphOptimizationLevel.ORT_DISABLE_ALL,
    "default": GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def get_model_from_framework_and_features(framework: str, features: str, model: str):
    """
    Attempt to retrieve a model from a model's name and the features to be enabled.

    Args:
        framework: The framework we are targeting
        features: The features required
        model: The name of the model to export

    Returns:

    """
    if framework == FRAMEWORK_NAME_PT:
        if features not in FEATURES_TO_AUTOMODELS:
            raise KeyError(
                f"Unknown feature: {features}." f"Possible values are {list(FEATURES_TO_AUTOMODELS.values())}"
            )

        return FEATURES_TO_AUTOMODELS[features].from_pretrained(model)
    elif framework == FRAMEWORK_NAME_TF:
        if features not in FEATURES_TO_TF_AUTOMODELS:
            raise KeyError(
                f"Unknown feature: {features}." f"Possible values are {list(FEATURES_TO_AUTOMODELS.values())}"
            )
        return FEATURES_TO_TF_AUTOMODELS[features].from_pretrained(model)
    else:
        raise ValueError(f"Unknown framework: {framework}")


def check_supported_model_or_raise(
    model: Union[PreTrainedModel, TFPreTrainedModel], features: str = "default"
) -> Tuple[str, Callable]:
    """
    Check whether or not the model has the requested features

    Args:
        model: The model to export
        features: The name of the features to check if they are avaiable

    Returns:
        (str) The type of the model (OnnxConfig) The OnnxConfig instance holding the model export properties

    """
    if model.config.model_type not in SUPPORTED_MODEL_KIND:
        raise KeyError(
            f"{model.config.model_type} ({model.name}) is not supported yet. "
            f"Only {SUPPORTED_MODEL_KIND} are supported. "
            f"If you want to support ({model.config.model_type}) please propose a PR or open up an issue."
        )

    # Look for the features
    model_features = SUPPORTED_MODEL_KIND[model.config.model_type]
    if features not in model_features:
        raise ValueError(
            f"{model.config.model_type} doesn't support features {features}. "
            f"Supported values are: {list(model_features.keys())}"
        )

    return model.config.model_type, SUPPORTED_MODEL_KIND[model.config.model_type][features]


def main():
    parser = ArgumentParser("Hugging Face ONNX Exporter tool")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model's name of path on disk to load.")
    parser.add_argument(
        "-f",
        "--framework",
        choices=FRAMEWORK_CHOICES,
        required=True,
        help=f"Framework to use when exporting. Possible values are: {FRAMEWORK_CHOICES}",
    )
    parser.add_argument(
        "--features",
        choices=["default", "with_past", "token_classification"],
        default="default",
        help="Export the model with some additional features.",
    )
    parser.add_argument(
        "--opset", type=int, default=12, help="ONNX opset version to export the model with (default 12)."
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Flag indicating if we should try to optimize the model for GPU inference.",
    )
    parser.add_argument(
        "--optimization-level",
        choices=ONNX_OPTIMIZATION_LEVELS.keys(),
        default="disabled",
        help="Flag indicating if we should try to optimize the model.",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-4, help="Absolute difference tolerence when validating the model."
    )
    parser.add_argument("output", type=Path, help="Path indicating where to store generated ONNX model.")

    # Retrieve CLI arguments
    args = parser.parse_args()
    args.output = args.output if args.output.is_file() else args.output.joinpath("model.onnx")

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    if args.optimization_level != "disabled":
        if not is_sympy_available() or not is_coloredlogs_available():
            raise EnvironmentError(
                "SymPy and coloredlogs must be installed in order to optimize ONNX models: "
                "pip install sympy coloredlogs"
            )

    logger.info(f"About to export model: {args.model} using framework: {args.framework}")

    # Allocate the model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = get_model_from_framework_and_features(args.framework, args.features, args.model)
    model_kind, model_onnx_config = check_supported_model_or_raise(model, features=args.features)
    onnx_config = model_onnx_config(model.config)

    # Ensure the requested opset is sufficient
    if args.opset < onnx_config.default_onnx_opset:
        raise ValueError(
            f"Opset {args.opset} is not sufficient to export {model_kind}. "
            f"At least  {onnx_config.default_onnx_opset} is required."
        )

    if args.framework == FRAMEWORK_NAME_PT:
        onnx_inputs, onnx_outputs = convert_pytorch(tokenizer, model, onnx_config, args.opset, args.output)
    else:
        raise NotImplementedError()

    validate_model_outputs(onnx_config, tokenizer, model, args.output, onnx_outputs, args.atol)
    logger.info(f"All good, model saved at: {args.output.as_posix()}")

    if args.optimization_level != "disabled":
        logger.info(f"About to optimize model with optimization_level: {args.optimization_level}")

        args.opt_model_output = generate_identified_filename(args.output, f"_optimized_{args.optimization_level}")
        args.optimization_level = ONNX_OPTIMIZATION_LEVELS[args.optimization_level]

        optimize(args.output, onnx_config, args.optimization_level, args.use_gpu, args.opt_model_output)

        if not args.use_gpu:
            validate_model_outputs(onnx_config, tokenizer, model, args.opt_model_output, onnx_outputs, args.atol)
        else:
            logger.info(
                "Validating model targeting GPU is not supported yet. "
                "Please, fill an issue or submit a PR if it's something you need."
            )

        logger.info(f"Optimized model saved at: {args.opt_model_output.as_posix()}")


if __name__ == "__main__":
    logger = logging.get_logger("transformers.onnx")  # pylint: disable=invalid-name
    logger.setLevel(logging.INFO)
    main()
