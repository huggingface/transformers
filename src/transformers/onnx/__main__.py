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

from transformers.models.auto import AutoTokenizer

from ..utils import logging
from .convert import export, validate_model_outputs
from .features import FeaturesManager


def main():
    parser = ArgumentParser("Hugging Face ONNX Exporter tool")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model's name of path on disk to load.")
    parser.add_argument(
        "--feature",
        choices=list(FeaturesManager.AVAILABLE_FEATURES),
        default="default",
        help="Export the model with some additional feature.",
    )
    parser.add_argument(
        "--opset", type=int, default=12, help="ONNX opset version to export the model with (default 12)."
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

    # Allocate the model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = FeaturesManager.get_model_from_feature(args.feature, args.model)
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=args.feature)
    onnx_config = model_onnx_config(model.config)

    # Ensure the requested opset is sufficient
    if args.opset < onnx_config.default_onnx_opset:
        raise ValueError(
            f"Opset {args.opset} is not sufficient to export {model_kind}. "
            f"At least  {onnx_config.default_onnx_opset} is required."
        )

    onnx_inputs, onnx_outputs = export(tokenizer, model, onnx_config, args.opset, args.output)

    validate_model_outputs(onnx_config, tokenizer, model, args.output, onnx_outputs, args.atol)
    logger.info(f"All good, model saved at: {args.output.as_posix()}")


if __name__ == "__main__":
    logger = logging.get_logger("transformers.onnx")  # pylint: disable=invalid-name
    logger.setLevel(logging.INFO)
    main()
