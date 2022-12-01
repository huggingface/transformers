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
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

from packaging import version

from ..utils import logging
from ..utils.import_utils import is_optimum_available


MIN_OPTIMUM_VERSION = "1.5.0"

ENCODER_DECODER_MODELS = ["vision-encoder-decoder"]


def main():
    parser = ArgumentParser("Hugging Face Transformers ONNX exporter")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Model ID on huggingface.co or path on disk to load model from."
    )
    parser.add_argument(
        "--feature",
        default="default",
        help="The type of features to export the model with.",
    )
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version to export the model with.")
    parser.add_argument(
        "--atol", type=float, default=None, help="Absolute difference tolerance when validating the model."
    )
    parser.add_argument(
        "--framework",
        type=str,
        choices=["pt", "tf"],
        default=None,
        help=(
            "The framework to use for the ONNX export."
            " If not provided, will attempt to use the local checkpoint's original framework"
            " or what is available in the environment."
        ),
    )
    parser.add_argument("output", type=Path, help="Path indicating where to store generated ONNX model.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path indicating where to store cache.")
    parser.add_argument(
        "--preprocessor",
        type=str,
        choices=["auto", "tokenizer", "feature_extractor", "processor"],
        default="auto",
        help="Which type of preprocessor to use. 'auto' tries to automatically detect it.",
    )
    args = parser.parse_args()

    cmd_line = [
        sys.executable,
        "-m",
        "optimum.exporters.onnx",
        f"--model {args.model}",
        f"--task {args.feature}",
        f"--framework {args.framework}" if args.framework is not None else "",
        f"{args.output}",
    ]
    print(" ".join(cmd_line))
    proc = subprocess.Popen(" ".join(cmd_line), stdout=subprocess.PIPE, shell=True)
    proc.wait()

    logger.info(
        "The export was done by optimum.exporters.onnx, it is suggested to use this tool directly for the future, more"
        " information here: https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model"
    )


if __name__ == "__main__":
    if is_optimum_available():
        from optimum.version import __version__ as optimum_version

        parsed_optimum_version = version.parse(optimum_version)
        if parsed_optimum_version < version.parse(MIN_OPTIMUM_VERSION):
            raise RuntimeError(
                f"transformers.onnx requires optimum >= {MIN_OPTIMUM_VERSION} but {optimum_version} is installed. You "
                "can upgrade optimum by running: pip install -U optimum"
            )
    else:
        raise RuntimeError(
            "transformers.onnx requires optimum to run, you can install the library by running: pip install optimum"
        )
    logger = logging.get_logger("transformers.onnx")  # pylint: disable=invalid-name
    logger.setLevel(logging.INFO)
    main()
