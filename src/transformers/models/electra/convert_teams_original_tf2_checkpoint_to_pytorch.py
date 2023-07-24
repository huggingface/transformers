# Copyright 2023 The HuggingFace Team. All rights reserved.
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
This script converts a checkpoint from the "Training ELECTRA Augmented with Multi-word Selection" (TEAMS)
implementation into a PyTorch-compatible ELECTRA model. The official implementation of "TEAMS" can be found in the
TensorFlow Models Garden repository:

https://github.com/tensorflow/models/tree/v2.9.2/official/projects/teams
"""
import argparse

import tensorflow as tf
import torch

from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining
from transformers.utils import logging


logging.set_verbosity_info()


def convert_checkpoint_to_pytorch(
    tf_checkpoint_path: str, config_path: str, pytorch_dump_path: str, discriminator_or_generator: str
):
    def get_array(name: str):
        name += "/.ATTRIBUTES/VARIABLE_VALUE"
        array = tf.train.load_variable(tf_checkpoint_path, name)

        if "kernel" in name:
            array = array.transpose()

        return torch.from_numpy(array)

    def get_array_with_reshaping(name: str, orginal_shape):
        name += "/.ATTRIBUTES/VARIABLE_VALUE"
        array = tf.train.load_variable(tf_checkpoint_path, name)
        array = array.reshape(orginal_shape)

        if "kernel" in name:
            array = array.transpose()

        return torch.from_numpy(array)

    config = ElectraConfig.from_pretrained(config_path)

    if discriminator_or_generator == "generator":
        model = ElectraForMaskedLM(config)
    elif discriminator_or_generator == "discriminator":
        model = ElectraForPreTraining(config)
    else:
        raise ValueError("The discriminator_or_generator argument should be either 'discriminator' or 'generator'")

    model.electra.embeddings.word_embeddings.weight.data = get_array("model/masked_lm/embedding_table")
    model.electra.embeddings.position_embeddings.weight.data = get_array(
        "encoder/layer_with_weights-0/layer_with_weights-1/embeddings"
    )
    model.electra.embeddings.token_type_embeddings.weight.data = get_array(
        "encoder/layer_with_weights-0/layer_with_weights-2/embeddings"
    )
    model.electra.embeddings.LayerNorm.weight.data = get_array(
        "encoder/layer_with_weights-0/layer_with_weights-3/gamma"
    )
    model.electra.embeddings.LayerNorm.bias.data = get_array("encoder/layer_with_weights-0/layer_with_weights-3/beta")

    if discriminator_or_generator == "generator":
        model.generator_predictions.LayerNorm.weight.data = get_array("model/masked_lm/layer_norm/gamma")
        model.generator_predictions.LayerNorm.bias.data = get_array("model/masked_lm/layer_norm/beta")

        model.generator_predictions.dense.weight.data = get_array("model/masked_lm/dense/kernel")
        model.generator_predictions.dense.bias.data = get_array("model/masked_lm/dense/bias")

        model.generator_lm_head.weight.data = get_array("model/masked_lm/embedding_table")
        model.generator_lm_head.bias.data = get_array("model/masked_lm/output_bias.Sbias")
    else:
        model.discriminator_predictions.dense.bias.data = get_array("model/discriminator_rtd_head/dense/bias")
        model.discriminator_predictions.dense.weight.data = get_array("model/discriminator_rtd_head/dense/kernel")

        model.discriminator_predictions.dense_prediction.bias.data = get_array(
            "model/discriminator_rtd_head/rtd_head/bias"
        )
        model.discriminator_predictions.dense_prediction.weight.data = get_array(
            "model/discriminator_rtd_head/rtd_head/kernel"
        )

    encoder_layer_end = config.num_hidden_layers

    if discriminator_or_generator == "generator":
        # Generator layers are shared in TEAMS model:
        encoder_layer_end = config.num_hidden_layers // 2

    for i in range(0, encoder_layer_end):
        layer = model.electra.encoder.layer[i]

        self_attn = layer.attention.self

        self_attn.query.weight.data = get_array_with_reshaping(
            f"encoder/layer_with_weights-{i + 1}/_attention_layer/_query_dense/kernel",
            self_attn.query.weight.data.shape,
        )
        self_attn.query.bias.data = get_array_with_reshaping(
            f"encoder/layer_with_weights-{i + 1}/_attention_layer/_query_dense/bias", self_attn.query.bias.data.shape
        )

        self_attn.key.weight.data = get_array_with_reshaping(
            f"encoder/layer_with_weights-{i + 1}/_attention_layer/_key_dense/kernel", self_attn.key.weight.data.shape
        )
        self_attn.key.bias.data = get_array_with_reshaping(
            f"encoder/layer_with_weights-{i + 1}/_attention_layer/_key_dense/bias", self_attn.key.bias.data.shape
        )

        self_attn.value.weight.data = get_array_with_reshaping(
            f"encoder/layer_with_weights-{i + 1}/_attention_layer/_value_dense/kernel",
            self_attn.value.weight.data.shape,
        )
        self_attn.value.bias.data = get_array_with_reshaping(
            f"encoder/layer_with_weights-{i + 1}/_attention_layer/_value_dense/bias", self_attn.value.bias.data.shape
        )

        self_output = layer.attention.output

        self_output.dense.weight.data = get_array_with_reshaping(
            f"encoder/layer_with_weights-{i + 1}/_attention_layer/_output_dense/kernel",
            self_output.dense.weight.data.shape,
        )
        self_output.dense.bias.data = get_array_with_reshaping(
            f"encoder/layer_with_weights-{i + 1}/_attention_layer/_output_dense/bias",
            self_output.dense.bias.data.shape,
        )

        self_output.LayerNorm.weight.data = get_array_with_reshaping(
            f"encoder/layer_with_weights-{i + 1}/_attention_layer_norm/gamma", self_output.LayerNorm.weight.data.shape
        )
        self_output.LayerNorm.bias.data = get_array_with_reshaping(
            f"encoder/layer_with_weights-{i + 1}/_attention_layer_norm/beta", self_output.LayerNorm.bias.data.shape
        )

        intermediate = layer.intermediate

        intermediate.dense.weight.data = get_array(f"encoder/layer_with_weights-{i + 1}/_intermediate_dense/kernel")
        intermediate.dense.bias.data = get_array(f"encoder/layer_with_weights-{i + 1}/_intermediate_dense/bias")

        electra_output = layer.output

        electra_output.dense.weight.data = get_array(
            f"encoder/layer_with_weights-{i + 1}/_output_dense/kernel"
        )  # , electra_output.dense.weight.data.shape)
        electra_output.dense.bias.data = get_array(
            f"encoder/layer_with_weights-{i + 1}/_output_dense/bias"
        )  # , electra_output.dense.bias.data.shape)

        electra_output.LayerNorm.weight.data = get_array_with_reshaping(
            f"encoder/layer_with_weights-{i + 1}/_output_layer_norm/gamma", electra_output.LayerNorm.weight.data.shape
        )
        electra_output.LayerNorm.bias.data = get_array_with_reshaping(
            f"encoder/layer_with_weights-{i + 1}/_output_layer_norm/beta", electra_output.LayerNorm.bias.data.shape
        )

    if discriminator_or_generator == "generator":
        for i in range(encoder_layer_end, config.num_hidden_layers):
            layer = model.electra.encoder.layer[i]

            self_attn = layer.attention.self

            self_attn.query.weight.data = get_array_with_reshaping(
                f"model/generator_network/layer_with_weights-{i + 1}/_attention_layer/_query_dense/kernel",
                self_attn.query.weight.data.shape,
            )
            self_attn.query.bias.data = get_array_with_reshaping(
                f"model/generator_network/layer_with_weights-{i + 1}/_attention_layer/_query_dense/bias",
                self_attn.query.bias.data.shape,
            )

            self_attn.key.weight.data = get_array_with_reshaping(
                f"model/generator_network/layer_with_weights-{i + 1}/_attention_layer/_key_dense/kernel",
                self_attn.key.weight.data.shape,
            )
            self_attn.key.bias.data = get_array_with_reshaping(
                f"model/generator_network/layer_with_weights-{i + 1}/_attention_layer/_key_dense/bias",
                self_attn.key.bias.data.shape,
            )

            self_attn.value.weight.data = get_array_with_reshaping(
                f"model/generator_network/layer_with_weights-{i + 1}/_attention_layer/_value_dense/kernel",
                self_attn.value.weight.data.shape,
            )
            self_attn.value.bias.data = get_array_with_reshaping(
                f"model/generator_network/layer_with_weights-{i + 1}/_attention_layer/_value_dense/bias",
                self_attn.value.bias.data.shape,
            )

            self_output = layer.attention.output

            self_output.dense.weight.data = get_array_with_reshaping(
                f"model/generator_network/layer_with_weights-{i + 1}/_attention_layer/_output_dense/kernel",
                self_output.dense.weight.data.shape,
            )
            self_output.dense.bias.data = get_array_with_reshaping(
                f"model/generator_network/layer_with_weights-{i + 1}/_attention_layer/_output_dense/bias",
                self_output.dense.bias.data.shape,
            )

            self_output.LayerNorm.weight.data = get_array_with_reshaping(
                f"model/generator_network/layer_with_weights-{i + 1}/_attention_layer_norm/gamma",
                self_output.LayerNorm.weight.data.shape,
            )
            self_output.LayerNorm.bias.data = get_array_with_reshaping(
                f"model/generator_network/layer_with_weights-{i + 1}/_attention_layer_norm/beta",
                self_output.LayerNorm.bias.data.shape,
            )

            intermediate = layer.intermediate

            intermediate.dense.weight.data = get_array(
                f"model/generator_network/layer_with_weights-{i + 1}/_intermediate_dense/kernel"
            )
            intermediate.dense.bias.data = get_array(
                f"model/generator_network/layer_with_weights-{i + 1}/_intermediate_dense/bias"
            )

            electra_output = layer.output

            electra_output.dense.weight.data = get_array(
                f"model/generator_network/layer_with_weights-{i + 1}/_output_dense/kernel"
            )
            electra_output.dense.bias.data = get_array(
                f"model/generator_network/layer_with_weights-{i + 1}/_output_dense/bias"
            )

            electra_output.LayerNorm.weight.data = get_array_with_reshaping(
                f"model/generator_network/layer_with_weights-{i + 1}/_output_layer_norm/gamma",
                electra_output.LayerNorm.weight.data.shape,
            )
            electra_output.LayerNorm.bias.data = get_array_with_reshaping(
                f"model/generator_network/layer_with_weights-{i + 1}/_output_layer_norm/beta",
                electra_output.LayerNorm.bias.data.shape,
            )

    # Export final model
    model.save_pretrained(pytorch_dump_path)

    # Integration test - should load without any errors ;)
    if discriminator_or_generator == "generator":
        loaded_model = ElectraForMaskedLM.from_pretrained(pytorch_dump_path)
    else:
        loaded_model = ElectraForPreTraining.from_pretrained(pytorch_dump_path)

    print(loaded_model.eval())
    print("Model conversion was done successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_checkpoint_path", type=str, required=True, help="Path to the TensorFlow TEAMS checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="The config json file corresponding to the TEAMS model. This specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model.",
    )
    parser.add_argument(
        "--discriminator_or_generator",
        default=None,
        type=str,
        required=True,
        help=(
            "Whether to export the generator or the discriminator. Should be a string, either 'discriminator' or "
            "'generator'."
        ),
    )
    args = parser.parse_args()
    convert_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path, args.discriminator_or_generator
    )
