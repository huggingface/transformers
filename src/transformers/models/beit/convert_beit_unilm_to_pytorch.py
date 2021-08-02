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
"""Convert BEiT checkpoints from the unilm repository."""


import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

import requests
from transformers import BEiTConfig, BEiTFeatureExtractor, BEiTForImageClassification, BEiTForMaskedImageModeling
from transformers.utils import logging
from transformers.utils.imagenet_classes import id2label


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, base_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"blocks.{i}.norm1.weight", f"beit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"beit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"beit.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"beit.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"beit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"beit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"beit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"beit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"beit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"beit.encoder.layer.{i}.output.dense.bias"))

    # projection layer + position embeddings
    rename_keys.extend(
        [
            ("cls_token", "beit.embeddings.cls_token"),
            ("patch_embed.proj.weight", "beit.embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.proj.bias", "beit.embeddings.patch_embeddings.projection.bias"),
        ]
    )

    if base_model:
        # layernorm + pooler
        rename_keys.extend(
            [
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
                ("pre_logits.fc.weight", "pooler.dense.weight"),
                ("pre_logits.fc.bias", "pooler.dense.bias"),
            ]
        )

        # if just the base model, we should remove "beit" from all keys that start with "beit"
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("beit") else pair for pair in rename_keys]
    else:
        # layernorm + classification head
        rename_keys.extend(
            [
                ("fc_norm.weight", "beit.pooler.layernorm.weight"),
                ("fc_norm.bias", "beit.pooler.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, base_model=False):
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "beit."
        # queries, keys and values
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"blocks.{i}.attn.v_bias")

        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[:config.hidden_size,:]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = v_bias

        # gamma_1 and gamma_2 
        # we call them lambda because otherwise they are renamed when using .from_pretrained
        gamma_1 = state_dict.pop(f"blocks.{i}.gamma_1")
        gamma_2 = state_dict.pop(f"blocks.{i}.gamma_2")

        state_dict[f"{prefix}encoder.layer.{i}.lambda_1"] = gamma_1
        state_dict[f"{prefix}encoder.layer.{i}.lambda_2"] = gamma_2
        
        # relative_position bias table + index
        table = state_dict.pop(f"blocks.{i}.attn.relative_position_bias_table")
        index = state_dict.pop(f"blocks.{i}.attn.relative_position_index")

        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_bias_table"] = table
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_index"] = index


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_beit_checkpoint(checkpoint_path, pytorch_dump_folder_path, task="MIM", size="base", image_size=224):
    """
    Copy/paste/tweak model's weights to our BEiT structure.
    """

    # define default BEiT configuration
    config = BEiTConfig()
    # task 
    base_model = False
    if task == "MIM":
        # masked image modeling
        config.use_shared_relative_position_bias = True
        base_model = True
    elif task == "IFT":
        # intermediate fine-tuning on ImageNet-22k
        config.use_relative_position_bias = True
        config.num_labels = 21841
    elif task == "FT":
        # fine-tuning on ImageNet-1k
        config.use_relative_position_bias = True
        config.num_labels = 1000
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        config.image_size = image_size
    else:
        raise ValueError(f"Task {task} not supported")
        
    # size of the architecture
    if size == "base":
        pass
    elif size == "large":
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
    else:
        raise ValueError(f"Size {size} not found in model name, should be either 'base' or 'large'")

    # load state_dict of original model, remove and rename some keys
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))['model']
    rename_keys = create_rename_keys(config, base_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model)

    # load HuggingFace model
    if task == "MIM":
        model = BEiTForMaskedImageModeling(config)
    else:
        model = BEiTForImageClassification(config)
    model.eval()
    model.load_state_dict(state_dict)
    
    # Check outputs on an image
    feature_extractor = BEiTFeatureExtractor(size=224, resample=Image.BILINEAR, do_center_crop=False)
    encoding = feature_extractor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    
    outputs = model(pixel_values)
    logits = outputs.logits

    print("Sum of logits:", logits.sum().item())
    print("Shape of logits:", logits.shape)
    print("Predicted class idx:", logits.argmax(-1).item())

    assert logits.shape == torch.Size([1, 1000])
    print("Logits:", logits[0, :3])
    assert torch.allclose(logits[0,:3], torch.tensor([-1.2385, -1.0987, -1.0108]), atol=1e-3)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the original PyTorch checkpoint (.pth file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    parser.add_argument(
        "--task",
        default="MIM",
        type=str,
        help="Task on which the model was trained. Can be either 'MIM' (masked image modeling), 'IFT' (intermediate fine-tuning) or 'FT' (fine-tuning)."
    )
    parser.add_argument("--size", default="base", type=str, help="Model size (base or large)")
    parser.add_argument("--image_size", default=224, type=int, help="Image size")
    args = parser.parse_args()
    convert_beit_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.task, args.size, args.image_size)
