# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert Hubert checkpoint."""

import argparse

import torch

from transformers import HubertConfig, HubertForSequenceClassification, Wav2Vec2FeatureExtractor, logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SUPPORTED_MODELS = ["UtteranceLevel"]


@torch.no_grad()
def convert_s3prl_checkpoint(base_model_name, config_path, checkpoint_path, model_dump_path):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint["Config"]["downstream_expert"]["modelrc"]["select"] not in SUPPORTED_MODELS:
        raise NotImplementedError(f"The supported s3prl models are {SUPPORTED_MODELS}")

    downstream_dict = checkpoint["Downstream"]

    hf_congfig = HubertConfig.from_pretrained(config_path)
    hf_model = HubertForSequenceClassification.from_pretrained(base_model_name, config=hf_congfig)
    hf_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        base_model_name, return_attention_mask=True, do_normalize=False
    )

    if hf_congfig.use_weighted_layer_sum:
        hf_model.layer_weights.data = checkpoint["Featurizer"]["weights"]

    hf_model.projector.weight.data = downstream_dict["projector.weight"]
    hf_model.projector.bias.data = downstream_dict["projector.bias"]
    hf_model.classifier.weight.data = downstream_dict["model.post_net.linear.weight"]
    hf_model.classifier.bias.data = downstream_dict["model.post_net.linear.bias"]

    hf_feature_extractor.save_pretrained(model_dump_path)
    hf_model.save_pretrained(model_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name", default=None, type=str, help="Name of the huggingface pretrained base model."
    )
    parser.add_argument("--config_path", default=None, type=str, help="Path to the huggingface classifier config.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to the s3prl checkpoint.")
    parser.add_argument("--model_dump_path", default=None, type=str, help="Path to the final converted model.")
    args = parser.parse_args()
    convert_s3prl_checkpoint(args.base_model_name, args.config_path, args.checkpoint_path, args.model_dump_path)
