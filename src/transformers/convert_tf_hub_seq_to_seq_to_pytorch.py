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
"""Convert Seq2Seq TF Hub checkpoint."""


import argparse
import logging

import torch
import tensorflow as tf  # has to be tf 1.15
import tensorflow_text  # noqa: F401 has to be tf 1.15
import tensorflow_hub as hub  # has to be tf 1.15
import numpy as np

from transformers import RobertaConfig, EncoderDecoderConfig, EncoderDecoderModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tf_weights_in_roberta_seq_to_seq(pytorch_model, tf_hub_path, model_class):
    model = hub.Module(tf_hub_path)
    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        init.run()
        all_variables = model.variable_map
        keep_track_variables = all_variables.copy()
        for key in reversed(list(all_variables.keys())):
            if "global" in key:
                logger.info(f"Skipping {key}...")
                continue
            pytorch_model_pointer = getattr(pytorch_model.decoder, model_class)
            is_embedding = False
            logger.info(f"Trying to match {key}...")
            # remove start_string = "module/bert/"
            sub_layers = key.split("/")[2:]
            for i, sub_layer in enumerate(sub_layers):
                if sub_layer == "embeddings":
                    is_embedding = True
                elif sub_layer == "LayerNorm":
                    is_embedding = False
                if "layer" in sub_layer:
                    pytorch_model_pointer = pytorch_model_pointer.layer[int(sub_layer.split("_")[-1])]
                elif sub_layer in ["kernel", "gamma"]:
                    pytorch_model_pointer = pytorch_model_pointer.weight
                elif sub_layer == "beta":
                    pytorch_model_pointer = pytorch_model_pointer.bias
                elif sub_layer == "encdec":
                    pytorch_model_pointer = pytorch_model_pointer.crossattention.self
                elif sub_layer == "encdec_output":
                    pytorch_model_pointer = pytorch_model_pointer.crossattention.output
                else:
                    if sub_layer == "attention" and "encdec" in sub_layers[i+1]:
                        continue
                    try:
                        pytorch_model_pointer = getattr(pytorch_model_pointer, sub_layer)
                    except AttributeError:
                        logger.info(f"Skipping to initialize {key} at {sub_layer}...")

            array = np.asarray(sess.run(all_variables[key]))
            if not is_embedding:
                logger.info("Transposing numpy weight of shape {} for {}".format(array.shape, key))
                array = np.transpose(array)
            else:
                pytorch_model_pointer = pytorch_model_pointer.weight

            try:
                assert (
                    pytorch_model_pointer.shape == array.shape
                ), f"Pointer shape {pytorch_model_pointer.shape} and array shape {array.shape} mismatched"
            except AssertionError as e:
                e.args += (pytorch_model_pointer.shape, array.shape)
                raise
            logger.info(f"Initialize PyTorch weight {key}")

            pytorch_model_pointer.data = torch.from_numpy(array.astype(np.float32))
            keep_track_variables.pop(key, None)

        logger.info("Weights not copied to PyTorch model: {}".format(", ".join(keep_track_variables.keys())))
        return pytorch_model


def convert_tf_checkpoint_to_pytorch(tf_hub_path, pytorch_dump_path):
    # Initialise PyTorch model
    encoder_config = RobertaConfig.from_pretrained("roberta-large", vocab_size=50358, max_position_embeddings=512)
    decoder_config = RobertaConfig.from_pretrained("roberta-large", vocab_size=50358, is_decoder=True, add_cross_attention=True, max_position_embeddings=512)
    config = EncoderDecoderConfig(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict())
    model = EncoderDecoderModel(config)
    print("Building PyTorch model from configuration: {}".format(str(config)))

    # Load weights from tf checkpoint
    load_tf_weights_in_roberta_seq_to_seq(model, tf_hub_path, model_class="roberta")

    # Save pytorch-model
    print("Save PyTorch model and config to {}".format(pytorch_dump_path))
    model.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_hub_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_hub_path, args.pytorch_dump_path)
