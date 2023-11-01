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
"""Convert VITS discriminator checkpoint and add it to an already converted VITS checkpoint."""

import argparse

import torch

from transformers import VitsConfig, logging
from transformers.models.vits.feature_extraction_vits import VitsFeatureExtractor

# TODO: change once added
from transformers.models.vits.modeling_vits import VitsDiscriminator, VitsModel, VitsModelForPreTraining
from transformers.models.vits.tokenization_vits import VitsTokenizer


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.vits")


MAPPING = {
    "conv_post": "final_conv",
}
TOP_LEVEL_KEYS = []
IGNORE_KEYS = []


@torch.no_grad()
def convert_checkpoint(
    pytorch_dump_folder_path,
    checkpoint_path=None,
    generator_checkpoint_path=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    config = VitsConfig.from_pretrained(generator_checkpoint_path)
    generator = VitsModel.from_pretrained(generator_checkpoint_path)

    discriminator = VitsDiscriminator(config)

    for disc in discriminator.discriminators:
        disc.apply_weight_norm()

    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # load weights

    state_dict = checkpoint["model"]

    for k, v in list(state_dict.items()):
        for old_layer_name in MAPPING:
            new_k = k.replace(old_layer_name, MAPPING[old_layer_name])

        state_dict[new_k] = state_dict.pop(k)

    extra_keys = set(state_dict.keys()) - set(discriminator.state_dict().keys())
    extra_keys = {k for k in extra_keys if not k.endswith(".attn.bias")}
    missing_keys = set(discriminator.state_dict().keys()) - set(state_dict.keys())
    missing_keys = {k for k in missing_keys if not k.endswith(".attn.bias")}
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    discriminator.load_state_dict(state_dict, strict=False)
    n_params = discriminator.num_parameters(exclude_embeddings=True)
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params")

    for disc in discriminator.discriminators:
        disc.remove_weight_norm()

    model = VitsModelForPreTraining(config)

    # load weights
    model.text_encoder = generator.text_encoder
    model.flow = generator.flow
    model.decoder = generator.decoder
    model.duration_predictor = generator.duration_predictor
    model.posterior_encoder = generator.posterior_encoder

    if config.num_speakers > 1:
        model.embed_speaker = generator.embed_speaker

    model.discriminator = discriminator
    tokenizer = VitsTokenizer.from_pretrained(generator_checkpoint_path, verbose=False)
    feature_extractor = VitsFeatureExtractor(sampling_rate=model.config.sampling_rate, feature_size=80)

    model.save_pretrained(pytorch_dump_folder_path)
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        feature_extractor.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Local path to original discriminator checkpoint"
    )
    parser.add_argument(
        "--generator_checkpoint_path", default=None, type=str, help="Path to the 🤗 generator (VitsModel)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.pytorch_dump_folder_path,
        args.checkpoint_path,
        args.generator_checkpoint_path,
        args.push_to_hub,
    )
