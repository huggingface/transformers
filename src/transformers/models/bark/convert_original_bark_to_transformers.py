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

import argparse

import torch
from bark.generation import load_model

from transformers.models.bark import BarkConfig, BarkForTextToSpeech


@torch.no_grad()
def convert_bark_checkpoint(use_gpu=False, use_small=True, pytorch_dump_folder_path=None):
    transformers_config = BarkConfig()

    text_model = load_model(use_gpu=use_gpu, use_small=use_small, model_type="text")
    tokenizer = text_model["tokenizer"]
    text_model = text_model["model"]
    text_state_dict = text_model.state_dict()

    coarse_model = load_model(use_gpu=use_gpu, use_small=use_small, model_type="coarse")
    coarse_model_state_dict = coarse_model.state_dict()

    fine_model = load_model(use_gpu=use_gpu, use_small=use_small, model_type="fine")
    fine_model_state_dict = fine_model.state_dict()

    model = BarkForTextToSpeech(transformers_config)
    model.text_model.load_state_dict(text_state_dict)
    model.coarse_model.load_state_dict(coarse_model_state_dict)
    model.fine_model.load_state_dict(fine_model_state_dict)

    # tokenizer.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    args = parser.parse_args()

    convert_bark_checkpoint(pytorch_dump_folder_path=args.pytorch_dump_folder_path)
