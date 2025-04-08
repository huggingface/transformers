# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert ALIGN checkpoints from the original repository."""

import argparse
import os

import align
import numpy as np
import requests
import tensorflow as tf
import torch
from PIL import Image
from tokenizer import Tokenizer

from transformers import (
    AlignConfig,
    AlignModel,
    AlignProcessor,
    BertConfig,
    BertTokenizer,
    EfficientNetConfig,
    EfficientNetImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def preprocess(image):
    image = tf.image.resize(image, (346, 346))
    image = tf.image.crop_to_bounding_box(image, (346 - 289) // 2, (346 - 289) // 2, 289, 289)
    return image


def get_align_config():
    vision_config = EfficientNetConfig.from_pretrained("google/efficientnet-b7")
    vision_config.image_size = 289
    vision_config.hidden_dim = 640
    vision_config.id2label = {"0": "LABEL_0", "1": "LABEL_1"}
    vision_config.label2id = {"LABEL_0": 0, "LABEL_1": 1}
    vision_config.depthwise_padding = []

    text_config = BertConfig()
    config = AlignConfig.from_text_vision_configs(
        text_config=text_config, vision_config=vision_config, projection_dim=640
    )
    return config


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


def get_processor():
    image_processor = EfficientNetImageProcessor(
        do_center_crop=True,
        rescale_factor=1 / 127.5,
        rescale_offset=True,
        do_normalize=False,
        include_top=False,
        resample=Image.BILINEAR,
    )
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tokenizer.model_max_length = 64
    processor = AlignProcessor(image_processor=image_processor, tokenizer=tokenizer)
    return processor


# here we list all keys to be renamed (original name on the left, our name on the right)
def rename_keys(original_param_names):
    # EfficientNet image encoder
    block_names = [v.split("_")[0].split("block")[1] for v in original_param_names if v.startswith("block")]
    block_names = list(set(block_names))
    block_names = sorted(block_names)
    num_blocks = len(block_names)
    block_name_mapping = {b: str(i) for b, i in zip(block_names, range(num_blocks))}

    rename_keys = []
    rename_keys.append(("stem_conv/kernel:0", "embeddings.convolution.weight"))
    rename_keys.append(("stem_bn/gamma:0", "embeddings.batchnorm.weight"))
    rename_keys.append(("stem_bn/beta:0", "embeddings.batchnorm.bias"))
    rename_keys.append(("stem_bn/moving_mean:0", "embeddings.batchnorm.running_mean"))
    rename_keys.append(("stem_bn/moving_variance:0", "embeddings.batchnorm.running_var"))

    for b in block_names:
        hf_b = block_name_mapping[b]
        rename_keys.append((f"block{b}_expand_conv/kernel:0", f"encoder.blocks.{hf_b}.expansion.expand_conv.weight"))
        rename_keys.append((f"block{b}_expand_bn/gamma:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.weight"))
        rename_keys.append((f"block{b}_expand_bn/beta:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.bias"))
        rename_keys.append(
            (f"block{b}_expand_bn/moving_mean:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.running_mean")
        )
        rename_keys.append(
            (f"block{b}_expand_bn/moving_variance:0", f"encoder.blocks.{hf_b}.expansion.expand_bn.running_var")
        )
        rename_keys.append(
            (f"block{b}_dwconv/depthwise_kernel:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_conv.weight")
        )
        rename_keys.append((f"block{b}_bn/gamma:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.weight"))
        rename_keys.append((f"block{b}_bn/beta:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.bias"))
        rename_keys.append(
            (f"block{b}_bn/moving_mean:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.running_mean")
        )
        rename_keys.append(
            (f"block{b}_bn/moving_variance:0", f"encoder.blocks.{hf_b}.depthwise_conv.depthwise_norm.running_var")
        )

        rename_keys.append((f"block{b}_se_reduce/kernel:0", f"encoder.blocks.{hf_b}.squeeze_excite.reduce.weight"))
        rename_keys.append((f"block{b}_se_reduce/bias:0", f"encoder.blocks.{hf_b}.squeeze_excite.reduce.bias"))
        rename_keys.append((f"block{b}_se_expand/kernel:0", f"encoder.blocks.{hf_b}.squeeze_excite.expand.weight"))
        rename_keys.append((f"block{b}_se_expand/bias:0", f"encoder.blocks.{hf_b}.squeeze_excite.expand.bias"))
        rename_keys.append(
            (f"block{b}_project_conv/kernel:0", f"encoder.blocks.{hf_b}.projection.project_conv.weight")
        )
        rename_keys.append((f"block{b}_project_bn/gamma:0", f"encoder.blocks.{hf_b}.projection.project_bn.weight"))
        rename_keys.append((f"block{b}_project_bn/beta:0", f"encoder.blocks.{hf_b}.projection.project_bn.bias"))
        rename_keys.append(
            (f"block{b}_project_bn/moving_mean:0", f"encoder.blocks.{hf_b}.projection.project_bn.running_mean")
        )
        rename_keys.append(
            (f"block{b}_project_bn/moving_variance:0", f"encoder.blocks.{hf_b}.projection.project_bn.running_var")
        )

    key_mapping = {}
    for item in rename_keys:
        if item[0] in original_param_names:
            key_mapping[item[0]] = "vision_model." + item[1]

    # BERT text encoder
    rename_keys = []
    old = "tf_bert_model/bert"
    new = "text_model"
    for i in range(12):
        rename_keys.append(
            (
                f"{old}/encoder/layer_._{i}/attention/self/query/kernel:0",
                f"{new}.encoder.layer.{i}.attention.self.query.weight",
            )
        )
        rename_keys.append(
            (
                f"{old}/encoder/layer_._{i}/attention/self/query/bias:0",
                f"{new}.encoder.layer.{i}.attention.self.query.bias",
            )
        )
        rename_keys.append(
            (
                f"{old}/encoder/layer_._{i}/attention/self/key/kernel:0",
                f"{new}.encoder.layer.{i}.attention.self.key.weight",
            )
        )
        rename_keys.append(
            (
                f"{old}/encoder/layer_._{i}/attention/self/key/bias:0",
                f"{new}.encoder.layer.{i}.attention.self.key.bias",
            )
        )
        rename_keys.append(
            (
                f"{old}/encoder/layer_._{i}/attention/self/value/kernel:0",
                f"{new}.encoder.layer.{i}.attention.self.value.weight",
            )
        )
        rename_keys.append(
            (
                f"{old}/encoder/layer_._{i}/attention/self/value/bias:0",
                f"{new}.encoder.layer.{i}.attention.self.value.bias",
            )
        )
        rename_keys.append(
            (
                f"{old}/encoder/layer_._{i}/attention/output/dense/kernel:0",
                f"{new}.encoder.layer.{i}.attention.output.dense.weight",
            )
        )
        rename_keys.append(
            (
                f"{old}/encoder/layer_._{i}/attention/output/dense/bias:0",
                f"{new}.encoder.layer.{i}.attention.output.dense.bias",
            )
        )
        rename_keys.append(
            (
                f"{old}/encoder/layer_._{i}/attention/output/LayerNorm/gamma:0",
                f"{new}.encoder.layer.{i}.attention.output.LayerNorm.weight",
            )
        )
        rename_keys.append(
            (
                f"{old}/encoder/layer_._{i}/attention/output/LayerNorm/beta:0",
                f"{new}.encoder.layer.{i}.attention.output.LayerNorm.bias",
            )
        )
        rename_keys.append(
            (
                f"{old}/encoder/layer_._{i}/intermediate/dense/kernel:0",
                f"{new}.encoder.layer.{i}.intermediate.dense.weight",
            )
        )
        rename_keys.append(
            (
                f"{old}/encoder/layer_._{i}/intermediate/dense/bias:0",
                f"{new}.encoder.layer.{i}.intermediate.dense.bias",
            )
        )
        rename_keys.append(
            (f"{old}/encoder/layer_._{i}/output/dense/kernel:0", f"{new}.encoder.layer.{i}.output.dense.weight")
        )
        rename_keys.append(
            (f"{old}/encoder/layer_._{i}/output/dense/bias:0", f"{new}.encoder.layer.{i}.output.dense.bias")
        )
        rename_keys.append(
            (f"{old}/encoder/layer_._{i}/output/LayerNorm/gamma:0", f"{new}.encoder.layer.{i}.output.LayerNorm.weight")
        )
        rename_keys.append(
            (f"{old}/encoder/layer_._{i}/output/LayerNorm/beta:0", f"{new}.encoder.layer.{i}.output.LayerNorm.bias")
        )

    rename_keys.append((f"{old}/embeddings/word_embeddings/weight:0", f"{new}.embeddings.word_embeddings.weight"))
    rename_keys.append(
        (f"{old}/embeddings/position_embeddings/embeddings:0", f"{new}.embeddings.position_embeddings.weight")
    )
    rename_keys.append(
        (f"{old}/embeddings/token_type_embeddings/embeddings:0", f"{new}.embeddings.token_type_embeddings.weight")
    )
    rename_keys.append((f"{old}/embeddings/LayerNorm/gamma:0", f"{new}.embeddings.LayerNorm.weight"))
    rename_keys.append((f"{old}/embeddings/LayerNorm/beta:0", f"{new}.embeddings.LayerNorm.bias"))

    rename_keys.append((f"{old}/pooler/dense/kernel:0", f"{new}.pooler.dense.weight"))
    rename_keys.append((f"{old}/pooler/dense/bias:0", f"{new}.pooler.dense.bias"))
    rename_keys.append(("dense/kernel:0", "text_projection.weight"))
    rename_keys.append(("dense/bias:0", "text_projection.bias"))
    rename_keys.append(("dense/bias:0", "text_projection.bias"))
    rename_keys.append(("temperature:0", "temperature"))

    for item in rename_keys:
        if item[0] in original_param_names:
            key_mapping[item[0]] = item[1]
    return key_mapping


def replace_params(hf_params, tf_params, key_mapping):
    list(hf_params.keys())

    for key, value in tf_params.items():
        if key not in key_mapping:
            continue

        hf_key = key_mapping[key]
        if "_conv" in key and "kernel" in key:
            new_hf_value = torch.from_numpy(value).permute(3, 2, 0, 1)
        elif "embeddings" in key:
            new_hf_value = torch.from_numpy(value)
        elif "depthwise_kernel" in key:
            new_hf_value = torch.from_numpy(value).permute(2, 3, 0, 1)
        elif "kernel" in key:
            new_hf_value = torch.from_numpy(np.transpose(value))
        elif "temperature" in key:
            new_hf_value = value
        elif "bn/gamma" or "bn/beta" in key:
            new_hf_value = torch.from_numpy(np.transpose(value)).squeeze()
        else:
            new_hf_value = torch.from_numpy(value)

        # Replace HF parameters with original TF model parameters
        hf_params[hf_key].copy_(new_hf_value)


@torch.no_grad()
def convert_align_checkpoint(checkpoint_path, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    Copy/paste/tweak model's weights to our ALIGN structure.
    """
    # Load original model
    seq_length = 64
    tok = Tokenizer(seq_length)
    original_model = align.Align("efficientnet-b7", "bert-base", 640, seq_length, tok.get_vocab_size())
    original_model.compile()
    original_model.load_weights(checkpoint_path)

    tf_params = original_model.trainable_variables
    tf_non_train_params = original_model.non_trainable_variables
    tf_params = {param.name: param.numpy() for param in tf_params}
    for param in tf_non_train_params:
        tf_params[param.name] = param.numpy()
    tf_param_names = list(tf_params.keys())

    # Load HuggingFace model
    config = get_align_config()
    hf_model = AlignModel(config).eval()
    hf_params = hf_model.state_dict()

    # Create src-to-dst parameter name mapping dictionary
    print("Converting parameters...")
    key_mapping = rename_keys(tf_param_names)
    replace_params(hf_params, tf_params, key_mapping)

    # Initialize processor
    processor = get_processor()
    inputs = processor(
        images=prepare_img(), text="A picture of a cat", padding="max_length", max_length=64, return_tensors="pt"
    )

    # HF model inference
    hf_model.eval()
    with torch.no_grad():
        outputs = hf_model(**inputs)

    hf_image_features = outputs.image_embeds.detach().numpy()
    hf_text_features = outputs.text_embeds.detach().numpy()

    # Original model inference
    original_model.trainable = False
    tf_image_processor = EfficientNetImageProcessor(
        do_center_crop=True,
        do_rescale=False,
        do_normalize=False,
        include_top=False,
        resample=Image.BILINEAR,
    )
    image = tf_image_processor(images=prepare_img(), return_tensors="tf", data_format="channels_last")["pixel_values"]
    text = tok(tf.constant(["A picture of a cat"]))

    image_features = original_model.image_encoder(image, training=False)
    text_features = original_model.text_encoder(text, training=False)

    image_features = tf.nn.l2_normalize(image_features, axis=-1)
    text_features = tf.nn.l2_normalize(text_features, axis=-1)

    # Check whether original and HF model outputs match  -> np.allclose
    if not np.allclose(image_features, hf_image_features, atol=1e-3):
        raise ValueError("The predicted image features are not the same.")
    if not np.allclose(text_features, hf_text_features, atol=1e-3):
        raise ValueError("The predicted text features are not the same.")
    print("Model outputs match!")

    if save_model:
        # Create folder to save model
        if not os.path.isdir(pytorch_dump_folder_path):
            os.mkdir(pytorch_dump_folder_path)
        # Save converted model and image processor
        hf_model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # Push model and image processor to hub
        print("Pushing converted ALIGN to the hub...")
        processor.push_to_hub("align-base")
        hf_model.push_to_hub("align-base")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_path",
        default="./weights/model-weights",
        type=str,
        help="Path to the pretrained TF ALIGN checkpoint.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="hf_model",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image processor to the hub")

    args = parser.parse_args()
    convert_align_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.save_model, args.push_to_hub)
