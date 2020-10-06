# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert RoBERTa checkpoint."""


import argparse

import torch

from transformers import logging
from transformers.configuration_prophetnet import ProphetNetConfig
from transformers.modeling_prophetnet import ProphetNetForConditionalGeneration

# transformers_old should correspond to branch `save_old_prophetnet_model_structure` here
from transformers_old.modeling_prophetnet import (
    ProphetNetForConditionalGeneration as ProphetNetForConditionalGenerationOld,
)


logger = logging.get_logger(__name__)
logging.set_verbosity_info()


def convert_prophetnet_checkpoint_to_pytorch(prophetnet_checkpoint_path: str, pytorch_dump_folder_path: str):
    """
    Copy/paste/tweak prohpetnet's weights to our prophetnet structure.
    """
    prophet_old = ProphetNetForConditionalGenerationOld.from_pretrained(prophetnet_checkpoint_path)
    prophet_old.eval()
    prophet_config_old = prophet_old.config

    prophet, loading_info = ProphetNetForConditionalGeneration.from_pretrained(
        prophetnet_checkpoint_path, output_loading_info=True
    )

    mapping = {
        "ngram_self_attn_layer_norm": "self_attn_layer_norm",
        "feed_forward": "",
        "intermediate": "fc1",
        "output": "fc2",
    }

    for key in loading_info["missing_keys"]:
        model = prophet
        old_model = prophet_old
        attributes = key.split(".")
        is_key_init = False
        for attribute in attributes:
            if attribute == "weight":
                assert old_model.weight.shape == model.weight.shape, "Shapes have to match!"
                model.weight = old_model.weight
                logger.info(f"{attribute} is initialized.")
                is_key_init = True
                break
            elif attribute == "bias":
                assert old_model.bias.shape == model.bias.shape, "Shapes have to match!"
                model.bias = old_model.bias
                logger.info(f"{attribute} is initialized")
                is_key_init = True
                break

            if attribute in mapping:
                old_attribute = mapping[attribute]
            else:
                old_attribute = attribute

            if attribute.isdigit():
                model = model[int(attribute)]
                old_model = old_model[int(old_attribute)]
            else:
                model = getattr(model, attribute)

                if old_attribute == "":
                    old_model = old_model
                else:
                    old_model = getattr(old_model, old_attribute)

        if not is_key_init:
            raise ValueError(f"{key} was not correctly initialized!")

    print(f"Saving model to {pytorch_dump_folder_path}")
    prophet.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--prophetnet_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_prophetnet_checkpoint_to_pytorch(args.prophetnet_checkpoint_path, args.pytorch_dump_folder_path)
