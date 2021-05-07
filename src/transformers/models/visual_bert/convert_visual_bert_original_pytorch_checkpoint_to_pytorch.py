# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert VisualBert checkpoint."""


import argparse
from pathlib import Path

import torch
import numpy as np
import random

from collections import OrderedDict

from transformers import (
    VisualBertConfig,
    VisualBertForMultipleChoice,
    VisualBertForNLVR,
    # VisualBertForFlickr,
    VisualBertForVQA,
    VisualBertForVQAAdvanced,
    VisualBertForPreTraining,
    # VisualBertModel,
)

from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

rename_keys_prefix = [
    ('bert.bert', 'visual_bert'),
    ('bert.cls', 'cls'),
    ('bert.classifier', 'cls'),
    ('token_type_embeddings_visual', 'visual_token_type_embeddings'),
    ('position_embeddings_visual', 'visual_position_embeddings'),
    ('projection', 'visual_projection')
]

ACCEPTABLE_CHECKPOINTS = [
    'nlvr2_coco_pre_trained.th',
    'nlvr2_fine_tuned.th',
    'nlvr2_pre_trained.th',
    'vcr_coco_pre_train.th',
    'vcr_fine_tune.th',
    'vcr_pre_train.th',
    'vqa_coco_pre_trained.th',
    'vqa_fine_tuned.th',
    'vqa_pre_trained.th',
]


def seed(value=42):
    """Set random seed for everything.
    Args:
        value (int): Seed
    """
    np.random.seed(value)
    torch.manual_seed(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(value)


def generate_inputs(config_name, visual_embedding_dim):
    seed()
    if 'vcr' in config_name and 'pre' not in config_name:
        input_ids = torch.tensor([[[1, 2, 3, 4, 5, 6] for i in range(4)]])
        attention_mask = torch.ones_like(input_ids)
        token_type_ids = torch.ones_like(input_ids)

        visual_embeds = torch.randn(size=(1, 4, 10, visual_embedding_dim), dtype=torch.float32)
        visual_token_type_ids = torch.ones(size=(1, 4, 10), dtype=torch.int32)
        visual_attention_mask = torch.ones_like(visual_token_type_ids)
    else:
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6]).reshape(1, -1)
        token_type_ids = torch.tensor([0, 0, 0, 1, 1, 1]).reshape(1, -1)
        visual_embeds = torch.randn(size=(1, 10, visual_embedding_dim), dtype=torch.float32)
        visual_token_type_ids = torch.ones(size=(1, 10), dtype=torch.int32)
        attention_mask = torch.tensor([1] * 6).reshape(1, -1)
        visual_attention_mask = torch.tensor([1] * 10).reshape(1, -1)

    return input_ids, token_type_ids, attention_mask, visual_embeds, visual_token_type_ids, visual_attention_mask


def load_state_dict(checkpoint_path):
    sd = torch.load(checkpoint_path, map_location="cpu")
    return sd


def get_new_dict(d, rename_keys_prefix=rename_keys_prefix):
    new_d = OrderedDict()
    # detector_d = OrderedDict()
    for key in d:
        if 'detector' in key:
            # detector_d[key.replace('detector.','')] = d[key]
            continue
        new_key = key
        for name_pair in rename_keys_prefix:
            new_key = new_key.replace(name_pair[0], name_pair[1])
        new_d[new_key] = d[key]
        if key == 'bert.cls.predictions.decoder.weight':
            # Old bert code didn't have `decoder.bias`, but was added separately
            new_d['cls.predictions.decoder.bias'] = new_d['cls.predictions.bias']
    return new_d


@ torch.no_grad()
def convert_visual_bert_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our VisualBERT structure.
    """

    assert checkpoint_path.split('/')[-1] in ACCEPTABLE_CHECKPOINTS, f"The checkpoint provided must be in {ACCEPTABLE_CHECKPOINTS}."

    # Get Config
    if 'pre' in checkpoint_path:
        model_type = 'pretraining'
        if 'vcr' in checkpoint_path:
            config_params = {
                'visual_embedding_dim': 512
            }
        elif 'vqa_advanced' in checkpoint_path:
            config_params = {
                'visual_embedding_dim': 2048
            }
        elif 'vqa' in checkpoint_path:
            config_params = {
                'visual_embedding_dim': 2048
            }
        elif 'nlvr' in checkpoint_path:
            config_params = {
                'visual_embedding_dim': 1024
            }
        else:
            raise NotImplementedError(f"No implementation found for `{checkpoint_path}`.")
    else:
        if 'vcr' in checkpoint_path:
            config_params = {
                'visual_embedding_dim': 512
            }
            model_type = 'multichoice'
        elif 'vqa_advanced' in checkpoint_path:
            config_params = {
                'visual_embedding_dim': 2048
            }
            model_type = 'vqa_advanced'
        elif 'vqa' in checkpoint_path:
            config_params = {
                'visual_embedding_dim': 2048,
                'num_labels': 3129
            }
            model_type = 'vqa'
        elif 'nlvr' in checkpoint_path:
            config_params = {
                'visual_embedding_dim': 1024,
                'num_labels': 2,
            }
            model_type = 'nlvr'

    config = VisualBertConfig(**config_params)

    # Load State Dict
    state_dict = load_state_dict(checkpoint_path)
    new_state_dict = get_new_dict(state_dict)

    if model_type == 'pretraining':
        model = VisualBertForPreTraining(config)
    elif model_type == 'vqa':
        model = VisualBertForVQA(config)
    elif model_type == 'vqa_advanced':
        model = VisualBertForVQAAdvanced(config)
    elif model_type == 'nlvr':
        model = VisualBertForNLVR(config)
    elif model_type == 'multichoice':
        model = VisualBertForMultipleChoice(config)

    model.load_state_dict(new_state_dict)
    # Save Checkpoints
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "orig_checkpoint_path", type=str, help="A path to .th on local filesystem."
    )
    parser.add_argument("pytorch_dump_folder_path", type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_visual_bert_checkpoint(args.orig_checkpoint_path, args.pytorch_dump_folder_path)
