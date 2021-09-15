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
"""Convert Perceiver checkpoints originally implemented in Haiku."""


import argparse
import pickle
from pathlib import Path
import json

import numpy as np
import torch
from huggingface_hub import cached_download, hf_hub_url

import haiku as hk
from transformers import PerceiverConfig, PerceiverForImageClassification, PerceiverForMaskedLM, PerceiverTokenizer
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_keys(state_dict):
    for name in list(state_dict):
        param = state_dict.pop(name)

        print("Processing name:", name)
        
        ## PREPROCESSORS ##
        
        # rename text preprocessor embeddings (for MLM model)
        name = name.replace("embed/embeddings", "input_preprocessor.embeddings.weight")
        if name.startswith("trainable_position_encoding/pos_embs"):
            name = name.replace("trainable_position_encoding/pos_embs", "input_preprocessor.position_embeddings.weight")
        
        # rename image preprocessor embeddings (for image classification model)
        name = name.replace("image_preprocessor/~/conv2_d/w", "input_preprocessor.convnet_1x1.weight")
        name = name.replace("image_preprocessor/~/conv2_d/b", "input_preprocessor.convnet_1x1.bias")
        name = name.replace("image_preprocessor/~_build_network_inputs/trainable_position_encoding/pos_embs", "input_preprocessor.position_embeddings.weight")
        name = name.replace("image_preprocessor/~_build_network_inputs/position_encoding_projector/linear/w", "input_preprocessor.positions_projection.weight")
        name = name.replace("image_preprocessor/~_build_network_inputs/position_encoding_projector/linear/b", "input_preprocessor.positions_projection.bias")
        
        ## DECODERS ## 
        
        # rename prefix of decoders
        name = name.replace("classification_decoder/~/basic_decoder/cross_attention/", "decoder.decoder.decoding_cross_attention.")
        name = name.replace("classification_decoder/~/basic_decoder/output/b", "decoder.decoder.final_layer.bias")
        name = name.replace("classification_decoder/~/basic_decoder/output/w", "decoder.decoder.final_layer.weight")
        name = name = name.replace("classification_decoder/~/basic_decoder/~/", "decoder.decoder.") 
        name = name.replace("basic_decoder/cross_attention/", "decoder.decoding_cross_attention.")
        name = name.replace("basic_decoder/~/", "decoder.")
        if "decoder" in name:
            name = name.replace("trainable_position_encoding/pos_embs", "output_position_encodings.weight")
        
        # rename embedding decoder bias (for MLM model)
        name = name.replace("embedding_decoder/bias", "output_postprocessor.bias")
        
        ## PERCEIVER MODEL ##
        
        # rename latent embeddings
        name = name.replace("perceiver_encoder/~/trainable_position_encoding/pos_embs", "embeddings.latents")
        
        # rename prefixes
        if name.startswith("perceiver_encoder/~/"):
            if "self_attention" in name:
                suffix = "self_attends."
            else:
                suffix = ""
            name = name.replace("perceiver_encoder/~/", "encoder." + suffix)
        # rename layernorm parameters
        if "offset" in name:
            name = name.replace("offset", "bias")
        if "scale" in name:
            name = name.replace("scale", "weight")
        # in HuggingFace, the layernorm in between attention + MLP is just called "layernorm"
        # rename layernorm in between attention + MLP of cross-attention
        if "cross_attention" in name and "layer_norm_2" in name:
            name = name.replace("layer_norm_2", "layernorm")
        # rename layernorm in between attention + MLP of self-attention
        if "self_attention" in name and "layer_norm_1" in name:
            name = name.replace("layer_norm_1", "layernorm")

        # in HuggingFace, the layernorms for queries + keys are called "layernorm1" and "layernorm2"
        if "cross_attention" in name and "layer_norm_1" in name:
            name = name.replace("layer_norm_1", "attention.self.layernorm2")
        if "cross_attention" in name and "layer_norm" in name:
            name = name.replace("layer_norm", "attention.self.layernorm1")
        if "self_attention" in name and "layer_norm" in name:
            name = name.replace("layer_norm", "attention.self.layernorm1")

        # rename special characters by dots
        name = name.replace("-", ".")
        name = name.replace("/", ".")
        # rename keys, queries, values and output of attention layers
        if ("cross_attention" in name or "self_attention" in name) and "mlp" not in name:
            if "linear.b" in name:
                name = name.replace("linear.b", "self.query.bias")
            if "linear.w" in name:
                name = name.replace("linear.w", "self.query.weight")
            if "linear_1.b" in name:
                name = name.replace("linear_1.b", "self.key.bias")
            if "linear_1.w" in name:
                name = name.replace("linear_1.w", "self.key.weight")
            if "linear_2.b" in name:
                name = name.replace("linear_2.b", "self.value.bias")
            if "linear_2.w" in name:
                name = name.replace("linear_2.w", "self.value.weight")
            if "linear_3.b" in name:
                name = name.replace("linear_3.b", "output.dense.bias")
            if "linear_3.w" in name:
                name = name.replace("linear_3.w", "output.dense.weight")
        if "self_attention_" in name:
            name = name.replace("self_attention_", "")
        if "self_attention" in name:
            name = name.replace("self_attention", "0")
        # rename dense layers of 2-layer MLP
        if "mlp" in name:
            if "linear.b" in name:
                name = name.replace("linear.b", "dense1.bias")
            if "linear.w" in name:
                name = name.replace("linear.w", "dense1.weight")
            if "linear_1.b" in name:
                name = name.replace("linear_1.b", "dense2.bias")
            if "linear_1.w" in name:
                name = name.replace("linear_1.w", "dense2.weight")

        # finally, TRANSPOSE if kernel and not embedding layer, and set value
        if name[-6:] == "weight" and "input_preprocessor" not in name and "output_position_encodings" not in name:
            param = np.transpose(param)

        # if conv2d, then we need to permute the axes
        if name.endswith("convnet_1x1.weight"):
            param = np.transpose(param)
        
        state_dict["perceiver." + name] = torch.from_numpy(param)


@torch.no_grad()
def convert_perceiver_checkpoint(pickle_file, pytorch_dump_folder_path, task="MLM"):
    """
    Copy/paste/tweak model's weights to our Perceiver structure.
    """

    # load parameters as FlatMapping data structure
    with open(pickle_file, "rb") as f:
        checkpoint = pickle.loads(f.read())

    if isinstance(checkpoint, dict):
        assert task == "image_classification", "Make sure to set task to image classification"
        # the image classification checkpoint with conv_preprocessing also has batchnorm state
        params = checkpoint['params']
        state = checkpoint['state']
    else:
        params = checkpoint
    
    # turn into initial state dict
    state_dict = dict()
    for scope_name, parameters in hk.data_structures.to_mutable_dict(params).items():
        for param_name, param in parameters.items():
            state_dict[scope_name + "/" + param_name] = param

    # rename keys
    rename_keys(state_dict)

    # load HuggingFace model
    config = PerceiverConfig()
    if task == "MLM":
        config.qk_channels = 8 * 32
        config.v_channels = 1280
        model = PerceiverForMaskedLM(config)
    elif task == "image_classification":
        config.num_latents = 512
        config.d_latents = 1024
        config.d_model = 512
        config.num_blocks = 8
        config.num_self_attends_per_block = 6
        config.num_cross_attention_heads = 1
        config.num_self_attention_heads = 8
        config.qk_channels = None
        config.v_channels = None
        config.num_labels = 1000
        repo_id = "datasets/huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename)), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        model = PerceiverForImageClassification(config)
    model.eval()

    # load weights
    model.load_state_dict(state_dict)

    # prepare dummy input
    tokenizer = PerceiverTokenizer.from_pretrained("/Users/NielsRogge/Documents/Perceiver/Tokenizer files")
    text = "This is an incomplete sentence where some words are missing."
    encoding = tokenizer(text, padding="max_length", return_tensors="pt")
    # mask " missing.". Note that the model performs much better if the masked chunk starts with a space.
    encoding.input_ids[0, 51:60] = tokenizer.mask_token_id

    # forward pass
    outputs = model(inputs=encoding.input_ids, attention_mask=encoding.attention_mask)
    logits = outputs.logits

    # verify logits
    print("Shape of logits:", logits.shape)
    print("First elements of logits:", logits[0, :3, :3])

    masked_tokens_predictions = logits[0, 51:60].argmax(dim=-1)
    print("Greedy predictions:")
    print(masked_tokens_predictions)
    print()
    print("Predicted string:")
    print(tokenizer.decode(masked_tokens_predictions))

    # Finally, save files
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pickle_file",
        type=str,
        default=None,
        required=True,
        help="Path to local pickle file of a Perceiver checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory, provided as a string.",
    )
    parser.add_argument(
        "--task",
        default="MLM",
        type=str,
        help="Task, provided as a string. One of 'MLM', 'image_classification'.",
    )

    args = parser.parse_args()
    convert_perceiver_checkpoint(args.pickle_file, args.pytorch_dump_folder_path, args.task)
