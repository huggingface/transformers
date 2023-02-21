# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

import argparse

from transformers.models.udop import UdopDualForConditionalGeneration, UdopUnimodelForConditionalGeneration


def convert_udop_checkpoint(pytorch_model_path, udop_type, pytorch_dump_path):
    if udop_type == "uni":
        model = UdopUnimodelForConditionalGeneration.from_pretrained(pytorch_model_path)
    else:
        model = UdopDualForConditionalGeneration.from_pretrained(pytorch_model_path)

    model.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pytorch_model_path", default=None, type=str, required=True, help="Path to udop pytorch checkpoint."
    )
    parser.add_argument(
        "--udop_type",
        default=None,
        type=str,
        required=True,
        help="The json file for UDOP model config.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_udop_checkpoint(args.pytorch_model_path, args.udop_type, args.pytorch_dump_path)
