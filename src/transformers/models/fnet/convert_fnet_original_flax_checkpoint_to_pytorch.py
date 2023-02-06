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
"""Convert FNet checkpoint."""


import argparse

import torch
from flax.training.checkpoints import restore_checkpoint

from transformers import FNetConfig, FNetForPreTraining
from transformers.utils import logging


logging.set_verbosity_info()


def convert_flax_checkpoint_to_pytorch(flax_checkpoint_path, fnet_config_file, save_path):
    # Initialise PyTorch model
    config = FNetConfig.from_json_file(fnet_config_file)
    print(f"Building PyTorch model from configuration: {config}")
    fnet_pretraining_model = FNetForPreTraining(config)

    checkpoint_dict = restore_checkpoint(flax_checkpoint_path, None)
    pretrained_model_params = checkpoint_dict["target"]

    # Embeddings
    # Position IDs
    state_dict = fnet_pretraining_model.state_dict()

    position_ids = state_dict["fnet.embeddings.position_ids"]
    new_state_dict = {"fnet.embeddings.position_ids": position_ids}
    # Embedding Layers
    new_state_dict["fnet.embeddings.word_embeddings.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["word"]["embedding"]
    )
    new_state_dict["fnet.embeddings.position_embeddings.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["position"]["embedding"][0]
    )
    new_state_dict["fnet.embeddings.token_type_embeddings.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["type"]["embedding"]
    )
    new_state_dict["fnet.embeddings.projection.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["hidden_mapping_in"]["kernel"]
    ).T
    new_state_dict["fnet.embeddings.projection.bias"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["hidden_mapping_in"]["bias"]
    )
    new_state_dict["fnet.embeddings.LayerNorm.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["layer_norm"]["scale"]
    )
    new_state_dict["fnet.embeddings.LayerNorm.bias"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["layer_norm"]["bias"]
    )

    # Encoder Layers
    for layer in range(config.num_hidden_layers):
        new_state_dict[f"fnet.encoder.layer.{layer}.fourier.output.LayerNorm.weight"] = torch.tensor(
            pretrained_model_params["encoder"][f"encoder_{layer}"]["mixing_layer_norm"]["scale"]
        )
        new_state_dict[f"fnet.encoder.layer.{layer}.fourier.output.LayerNorm.bias"] = torch.tensor(
            pretrained_model_params["encoder"][f"encoder_{layer}"]["mixing_layer_norm"]["bias"]
        )

        new_state_dict[f"fnet.encoder.layer.{layer}.intermediate.dense.weight"] = torch.tensor(
            pretrained_model_params["encoder"][f"feed_forward_{layer}"]["intermediate"]["kernel"]
        ).T
        new_state_dict[f"fnet.encoder.layer.{layer}.intermediate.dense.bias"] = torch.tensor(
            pretrained_model_params["encoder"][f"feed_forward_{layer}"]["intermediate"]["bias"]
        )

        new_state_dict[f"fnet.encoder.layer.{layer}.output.dense.weight"] = torch.tensor(
            pretrained_model_params["encoder"][f"feed_forward_{layer}"]["output"]["kernel"]
        ).T
        new_state_dict[f"fnet.encoder.layer.{layer}.output.dense.bias"] = torch.tensor(
            pretrained_model_params["encoder"][f"feed_forward_{layer}"]["output"]["bias"]
        )

        new_state_dict[f"fnet.encoder.layer.{layer}.output.LayerNorm.weight"] = torch.tensor(
            pretrained_model_params["encoder"][f"encoder_{layer}"]["output_layer_norm"]["scale"]
        )
        new_state_dict[f"fnet.encoder.layer.{layer}.output.LayerNorm.bias"] = torch.tensor(
            pretrained_model_params["encoder"][f"encoder_{layer}"]["output_layer_norm"]["bias"]
        )

    # Pooler Layers
    new_state_dict["fnet.pooler.dense.weight"] = torch.tensor(pretrained_model_params["encoder"]["pooler"]["kernel"]).T
    new_state_dict["fnet.pooler.dense.bias"] = torch.tensor(pretrained_model_params["encoder"]["pooler"]["bias"])

    # Masked LM Layers
    new_state_dict["cls.predictions.transform.dense.weight"] = torch.tensor(
        pretrained_model_params["predictions_dense"]["kernel"]
    ).T
    new_state_dict["cls.predictions.transform.dense.bias"] = torch.tensor(
        pretrained_model_params["predictions_dense"]["bias"]
    )
    new_state_dict["cls.predictions.transform.LayerNorm.weight"] = torch.tensor(
        pretrained_model_params["predictions_layer_norm"]["scale"]
    )
    new_state_dict["cls.predictions.transform.LayerNorm.bias"] = torch.tensor(
        pretrained_model_params["predictions_layer_norm"]["bias"]
    )
    new_state_dict["cls.predictions.decoder.weight"] = torch.tensor(
        pretrained_model_params["encoder"]["embedder"]["word"]["embedding"]
    )
    new_state_dict["cls.predictions.decoder.bias"] = torch.tensor(
        pretrained_model_params["predictions_output"]["output_bias"]
    )
    new_state_dict["cls.predictions.bias"] = torch.tensor(pretrained_model_params["predictions_output"]["output_bias"])

    # Seq Relationship Layers
    new_state_dict["cls.seq_relationship.weight"] = torch.tensor(
        pretrained_model_params["classification"]["output_kernel"]
    )
    new_state_dict["cls.seq_relationship.bias"] = torch.tensor(
        pretrained_model_params["classification"]["output_bias"]
    )

    # Load State Dict
    fnet_pretraining_model.load_state_dict(new_state_dict)

    # Save PreTrained
    print(f"Saving pretrained model to {save_path}")
    fnet_pretraining_model.save_pretrained(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--flax_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--fnet_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained FNet model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument("--save_path", default=None, type=str, required=True, help="Path to the output model.")
    args = parser.parse_args()
    convert_flax_checkpoint_to_pytorch(args.flax_checkpoint_path, args.fnet_config_file, args.save_path)
