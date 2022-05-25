# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import os

import torch

from transformers.utils import WEIGHTS_NAME


DIALOGPT_MODELS = ["small", "medium", "large"]

OLD_KEY = "lm_head.decoder.weight"
NEW_KEY = "lm_head.weight"


def convert_dialogpt_checkpoint(checkpoint_path: str, pytorch_dump_folder_path: str):
    d = torch.load(checkpoint_path)
    d[NEW_KEY] = d.pop(OLD_KEY)
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    torch.save(d, os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogpt_path", default=".", type=str)
    args = parser.parse_args()
    for MODEL in DIALOGPT_MODELS:
        checkpoint_path = os.path.join(args.dialogpt_path, f"{MODEL}_ft.pkl")
        pytorch_dump_folder_path = f"./DialoGPT-{MODEL}"
        convert_dialogpt_checkpoint(
            checkpoint_path,
            pytorch_dump_folder_path,
        )
