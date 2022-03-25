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

from ..models.auto import AutoConfig, AutoFeatureExtractor, AutoTokenizer
from ..models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING_NAMES
from ..models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from ..utils import logging
from .convert import export, validate_model_outputs
from .features import FeaturesManager


def main():
    parser = ArgumentParser("Hugging Face Transformers ONNX exporter")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    parser.add_argument(
        "--feature",
        choices=list(FeaturesManager.AVAILABLE_FEATURES),
        default="default",
        help="The type of features to export the model with.",
    )
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version to export the model with.")
    parser.add_argument(
        "--atol", type=float, default=None, help="Absolute difference tolerence when validating the model."
    )
    parser.add_argument(
        "--framework", type=str, choices=["pt", "tf"], default="pt", help="The framework to use for the ONNX export."
    )
    parser.add_argument("output", type=Path, help="Path indicating where to store generated ONNX model.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path indicating where to store cache.")

    # Retrieve CLI arguments
    args = parser.parse_args()
    args.output = args.output if args.output.is_file() else args.output.joinpath("model.onnx")

    if not args.output.parent.exists():
        args.output.parent.mkdir(parents=True)

    # Check the modality of the inputs and instantiate the appropriate preprocessor
    # TODO(lewtun): Refactor this as a function if we need to check modalities elsewhere as well
    config = AutoConfig.from_pretrained(args.model)
    if config.model_type in TOKENIZER_MAPPING_NAMES:
        preprocessor = AutoTokenizer.from_pretrained(args.model)
    elif config.model_type in FEATURE_EXTRACTOR_MAPPING_NAMES:
        preprocessor = AutoFeatureExtractor.from_pretrained(args.model)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    # Allocate the model
    model = FeaturesManager.get_model_from_feature(
        args.feature, args.model, framework=args.framework, cache_dir=args.cache_dir
    )
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=args.feature)
    onnx_config = model_onnx_config(model.config)

    # Ensure the requested opset is sufficient
    if args.opset is None:
        args.opset = onnx_config.default_onnx_opset

    if args.opset < onnx_config.default_onnx_opset:
        raise ValueError(
            f"Opset {args.opset} is not sufficient to export {model_kind}. "
            f"At least  {onnx_config.default_onnx_opset} is required."
        )

    onnx_inputs, onnx_outputs = export(
        preprocessor,
        model,
        onnx_config,
        args.opset,
        args.output,
    )

    if args.atol is None:
        args.atol = onnx_config.atol_for_validation

    validate_model_outputs(onnx_config, preprocessor, model, args.output, onnx_outputs, args.atol)
    logger.info(f"All good, model saved at: {args.output.as_posix()}")


if __name__ == "__main__":
    logger = logging.get_logger("transformers.onnx")  # pylint: disable=invalid-name
    logger.setLevel(logging.INFO)
    main()
