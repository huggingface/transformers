# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""
Convert SAM checkpoints from the original repository.

URL: https://github.com/salesforce/LAVIS/tree/main/projects/blip2
"""

import argparse

import requests
import torch

from PIL import Image

from transformers import (
    AutoTokenizer,
    SamConfig,
    SamForImageSegmentation,
    SamProcessor,
    SamVisionConfig,
    SamProcessor,
    ViTImageProcessor,
    T5Tokenizer,
    OPTConfig,
    T5Config,
)
from huggingface_hub import hf_hub_download

def replace_keys(state_dict):
    return state_dict

def convert_sam_checkpoint(model_name, pytorch_dump_folder, push_to_hub):
    checkpoint_path = hf_hub_download("ybelkada/segment-anything", f"checkpoints/{model_name}.pth")

    if "sam_vit_b" in model_name:
        config = SamConfig()
    elif "sam_vit_l" in model_name:
        vision_config = SamVisionConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            global_attn_indexes=[5, 11, 17, 23],
        )

        config = SamConfig(
            vision_config=vision_config,
        )
    elif "sam_vit_h" in model_name:
        vision_config = SamVisionConfig(
            hidden_size=1280,
            num_hidden_layers=32,
            num_attention_heads=16,
            global_attn_indexes=[7, 15, 23, 31],
        )

        config = SamConfig(
            vision_config=vision_config,
        )


    state_dict = torch.load(checkpoint_path, map_location="cpu")
    state_dict = replace_keys(state_dict)

    image_processor = ViTImageProcessor()
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    processor = SamProcessor(image_processor=image_processor, tokenizer=tokenizer)
    hf_model = SamForImageSegmentation(config)
    hf_model.load_state_dict(state_dict)

    hf_model.save_pretrained(pytorch_dump_folder)
    processor.save_pretrained(pytorch_dump_folder)

    if push_to_hub:
        hf_model.push_to_hub()
        processor.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = [
        "sam_vit_b_01ec64",
        "sam_vit_h_4b8939",
        "sam_vit_l_0b3195"
    ]
    parser.add_argument(
        "--model_name",
        default="sam_vit_h_4b8939",
        choices=choices,
        type=str,
        help="Path to hf config.json of model to convert",
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )

    args = parser.parse_args()

    convert_sam_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
