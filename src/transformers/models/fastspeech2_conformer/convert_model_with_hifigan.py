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
"""Convert FastSpeech2Conformer checkpoint."""

import argparse

import torch

from transformers import (
    FastSpeech2ConformerConfig,
    FastSpeech2ConformerHifiGan,
    FastSpeech2ConformerHifiGanConfig,
    FastSpeech2ConformerModel,
    FastSpeech2ConformerWithHifiGan,
    FastSpeech2ConformerWithHifiGanConfig,
    logging,
)

from .convert_fastspeech2_conformer_original_pytorch_checkpoint_to_pytorch import (
    convert_espnet_state_dict_to_hf,
    remap_model_yaml_config,
)
from .convert_hifigan import load_weights, remap_hifigan_yaml_config


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.FastSpeech2Conformer")


def convert_FastSpeech2ConformerWithHifiGan_checkpoint(
    checkpoint_path,
    yaml_config_path,
    pytorch_dump_folder_path,
    repo_id=None,
):
    # Prepare the model
    model_params, *_ = remap_model_yaml_config(yaml_config_path)
    model_config = FastSpeech2ConformerConfig(**model_params)

    model = FastSpeech2ConformerModel(model_config)

    espnet_checkpoint = torch.load(checkpoint_path, weights_only=True)
    hf_compatible_state_dict = convert_espnet_state_dict_to_hf(espnet_checkpoint)
    model.load_state_dict(hf_compatible_state_dict)

    # Prepare the vocoder
    config_kwargs = remap_hifigan_yaml_config(yaml_config_path)
    vocoder_config = FastSpeech2ConformerHifiGanConfig(**config_kwargs)

    vocoder = FastSpeech2ConformerHifiGan(vocoder_config)
    load_weights(espnet_checkpoint, vocoder, vocoder_config)

    # Prepare the model + vocoder
    config = FastSpeech2ConformerWithHifiGanConfig.from_sub_model_configs(model_config, vocoder_config)
    with_hifigan_model = FastSpeech2ConformerWithHifiGan(config)
    with_hifigan_model.model = model
    with_hifigan_model.vocoder = vocoder

    with_hifigan_model.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        with_hifigan_model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, default=None, type=str, help="Path to original checkpoint")
    parser.add_argument(
        "--yaml_config_path", required=True, default=None, type=str, help="Path to config.yaml of model to convert"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,
        default=None,
        type=str,
        help="Path to the output `FastSpeech2ConformerModel` PyTorch model.",
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()

    convert_FastSpeech2ConformerWithHifiGan_checkpoint(
        args.checkpoint_path,
        args.yaml_config_path,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
