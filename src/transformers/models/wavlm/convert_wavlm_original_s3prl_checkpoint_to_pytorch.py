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

from transformers import (
    Wav2Vec2FeatureExtractor,
    WavLMConfig,
    WavLMForAudioFrameClassification,
    WavLMForSequenceClassification,
    WavLMForXVector,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def convert_classification(base_model_name, hf_config, downstream_dict):
    model = WavLMForSequenceClassification.from_pretrained(base_model_name, config=hf_config)
    model.projector.weight.data = downstream_dict["projector.weight"]
    model.projector.bias.data = downstream_dict["projector.bias"]
    model.classifier.weight.data = downstream_dict["model.post_net.linear.weight"]
    model.classifier.bias.data = downstream_dict["model.post_net.linear.bias"]
    return model


def convert_diarization(base_model_name, hf_config, downstream_dict):
    model = WavLMForAudioFrameClassification.from_pretrained(base_model_name, config=hf_config)
    model.classifier.weight.data = downstream_dict["model.linear.weight"]
    model.classifier.bias.data = downstream_dict["model.linear.bias"]
    return model


def convert_xvector(base_model_name, hf_config, downstream_dict):
    model = WavLMForXVector.from_pretrained(base_model_name, config=hf_config)
    model.projector.weight.data = downstream_dict["connector.weight"]
    model.projector.bias.data = downstream_dict["connector.bias"]
    for i, kernel_size in enumerate(hf_config.tdnn_kernel):
        model.tdnn[i].kernel.weight.data = downstream_dict[
            f"model.framelevel_feature_extractor.module.{i}.kernel.weight"
        ]
        model.tdnn[i].kernel.bias.data = downstream_dict[f"model.framelevel_feature_extractor.module.{i}.kernel.bias"]

    model.feature_extractor.weight.data = downstream_dict["model.utterancelevel_feature_extractor.linear1.weight"]
    model.feature_extractor.bias.data = downstream_dict["model.utterancelevel_feature_extractor.linear1.bias"]
    model.classifier.weight.data = downstream_dict["model.utterancelevel_feature_extractor.linear2.weight"]
    model.classifier.bias.data = downstream_dict["model.utterancelevel_feature_extractor.linear2.bias"]
    model.objective.weight.data = downstream_dict["objective.W"]
    return model


@torch.no_grad()
def convert_s3prl_checkpoint(base_model_name, config_path, checkpoint_path, model_dump_path):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    downstream_dict = checkpoint["Downstream"]

    hf_config = WavLMConfig.from_pretrained(config_path)
    hf_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        base_model_name, return_attention_mask=True, do_normalize=False
    )

    arch = hf_config.architectures[0]
    if arch.endswith("ForSequenceClassification"):
        hf_model = convert_classification(base_model_name, hf_config, downstream_dict)
    elif arch.endswith("ForAudioFrameClassification"):
        hf_model = convert_diarization(base_model_name, hf_config, downstream_dict)
    elif arch.endswith("ForXVector"):
        hf_model = convert_xvector(base_model_name, hf_config, downstream_dict)
    else:
        raise NotImplementedError(f"S3PRL weights conversion is not supported for {arch}")

    if hf_config.use_weighted_layer_sum:
        hf_model.layer_weights.data = checkpoint["Featurizer"]["weights"]

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
