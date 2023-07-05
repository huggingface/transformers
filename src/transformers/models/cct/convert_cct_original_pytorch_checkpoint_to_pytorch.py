# # coding=utf-8
# # Copyright 2023 The HuggingFace Inc. team.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# """Convert CCT checkpoints from the original repository.

# URL: https://github.com/microsoft/CCT"""


import argparse
import json
from pathlib import Path
from collections import OrderedDict

import torch
from huggingface_hub import cached_download, hf_hub_url

from transformers import CctConfig, CctForImageClassification, ConvNextFeatureExtractor
from PIL import Image
import requests


def embeddings(idx):
    """
    The function helps in renaming embedding layer weights.

    Args:
        idx: stage number in original model
    """
    embed = {}
    embed[f"tokenizer.conv_layers.{idx}.0.weight"] = f"cct.embedder.embedding_layers.{3*idx}.weight"

    return embed


def transformer(idx):
    """
    The function helps in renaming transformer block weights.

    """
    transformer_weights = {}

    if idx == 0:
        transformer_weights[f"classifier.positional_emb"] = f"cct.encoder.positional_emb"
        transformer_weights[f"classifier.attention_pool.weight"] = f"cct.encoder.attention_pool.weight"
        transformer_weights[f"classifier.attention_pool.bias"] = f"cct.encoder.attention_pool.bias"

    transformer_weights[f"classifier.blocks.{idx}.pre_norm.weight"] = f"cct.encoder.blocks.{idx}.pre_norm.weight"
    transformer_weights[f"classifier.blocks.{idx}.pre_norm.bias"] = f"cct.encoder.blocks.{idx}.pre_norm.bias"
    transformer_weights[f"classifier.blocks.{idx}.linear1.weight"] = f"cct.encoder.blocks.{idx}.linear1.weight"
    transformer_weights[f"classifier.blocks.{idx}.linear1.bias"] = f"cct.encoder.blocks.{idx}.linear1.bias"
    transformer_weights[f"classifier.blocks.{idx}.norm1.weight"] = f"cct.encoder.blocks.{idx}.norm1.weight"
    transformer_weights[f"classifier.blocks.{idx}.norm1.bias"] = f"cct.encoder.blocks.{idx}.norm1.bias"
    transformer_weights[f"classifier.blocks.{idx}.linear2.weight"] = f"cct.encoder.blocks.{idx}.linear2.weight"
    transformer_weights[f"classifier.blocks.{idx}.linear2.bias"] = f"cct.encoder.blocks.{idx}.linear2.bias"
    transformer_weights[f"classifier.blocks.{idx}.self_attn.qkv.weight"] = f"cct.encoder.blocks.{idx}.self_attn.qkv.weight"
    transformer_weights[f"classifier.blocks.{idx}.self_attn.proj.weight"] = f"cct.encoder.blocks.{idx}.self_attn.proj.weight"
    transformer_weights[f"classifier.blocks.{idx}.self_attn.proj.bias"] = f"cct.encoder.blocks.{idx}.self_attn.proj.bias"

    return transformer_weights

def final():
    """
    Function helps in renaming final normalization and classification layer
    """
    head = {}
    head["classifier.norm.weight"] = "cct.encoder.norm.weight"
    head["classifier.norm.bias"] = "cct.encoder.norm.bias"
    head["classifier.fc.weight"] = "classifier.weight"
    head["classifier.fc.bias"] = "classifier.bias"

    return head


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@torch.no_grad()
def convert_cct_checkpoint(model_name, pytorch_dump_folder):
    """
    Fucntion to convert the cct checkpoint to huggingface checkpoint
    """

    # define CCT configuration based on URL
    checkpoint_url_dict = {
        'cct_14_7x2_224':
            'https://shi-labs.com/projects/cct/checkpoints/pretrained/cct_14_7x2_224_imagenet.pth',
        'cct_14_7x2_384':
            'https://shi-labs.com/projects/cct/checkpoints/finetuned/cct_14_7x2_384_imagenet.pth',
    }

    checkpoint_url = checkpoint_url_dict[model_name]
    num_labels = 1000
    img_labels_file = "imagenet-1k-id2label.json"
    repo_id = "huggingface/label-files"
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, img_labels_file, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    if model_name.split('_')[3] == '224':
        size = 224
    else:
        size = 384

    config = CctConfig()
    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    config.img_size = size

    # Description of transformer layers
    config.num_transformer_layers = 14
    config.num_heads = 6
    config.mlp_ratio = 3
    config.embed_dim = 384

    config.conv_kernel_size = 7
    config.num_conv_layers = 2
    config.conv_stride = 2
    config.conv_padding = 3
    config.out_channels = [64, 384]

    model = CctForImageClassification(config)

    # load original state_dict from URL
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)
    
    for idx in range(config.num_conv_layers):
        new_keys = embeddings(idx)
        for key, n_key in new_keys.items():
            value = state_dict.pop(key)

            assert value.shape == model.get_parameter(n_key).data.shape, f"Shape of weights doesn't match for {key} in original model and {n_key} in hf model: Expected{model.get_parameter(n_key).data.shape}, got {value.shape}"
            print(f"Initialized weight {n_key} from {key}, with shape {value.shape}")
            state_dict[n_key] = value

    for idx in range(config.num_transformer_layers):
        new_keys = transformer(idx)
        for key, n_key in new_keys.items():
            value = state_dict.pop(key)

            assert value.shape == model.get_parameter(n_key).data.shape, f"Shape of weights doesn't match for {key} in original model and {n_key} in hf model: Expected{model.get_parameter(n_key).data.shape}, got {value.shape}"
            print(f"Initialized weight {n_key} from {key}, with shape {value.shape}")
            state_dict[n_key] = value

    new_keys = final()
    for key, n_key in new_keys.items():
        value = state_dict.pop(key)

        assert value.shape == model.get_parameter(n_key).data.shape, f"Shape of weights doesn't match for {key} in original model and {n_key} in hf model: Expected{model.get_parameter(n_key).data.shape}, got {value.shape}"
        print(f"Initialized weight {n_key} from {key}, with shape {value.shape}")
        state_dict[n_key] = value

    # load HuggingFace model
    model.load_state_dict(state_dict)
    print(f"Successfully loaded state dict")
    model.eval()

    # Check outputs on an image, prepared by ConvNextFeatureExtractor
    feature_extractor = ConvNextFeatureExtractor(size=size)
    pixel_values = feature_extractor(images=prepare_img(), return_tensors="pt").pixel_values

    logits = model(pixel_values).logits

    if model_name == 'cct_14_7x2_224':
        expected_logits = torch.tensor([0.4862, 0.6512, 0.2444, 0.2685, 0.4327])
    if model_name == 'cct_14_7x2_384':
        expected_logits = torch.tensor([0.1484, 1.1873, 0.3872, 0.1801, 0.5467])
        
    assert torch.allclose(logits[0, :5], expected_logits, atol=1e-3)
    assert logits.shape == (1, num_labels)

    Path(pytorch_dump_folder).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder}")
    model.save_pretrained(pytorch_dump_folder)
    print(f"Saving feature extractor to {pytorch_dump_folder}")
    feature_extractor.save_pretrained(pytorch_dump_folder)

    print("Pushing model to the hub...")
    model.push_to_hub(
        repo_id=f"rishabbala/{model_name}",
        commit_message="Add model CCT",
        use_temp_dir=True
    )
    feature_extractor.push_to_hub(
        repo_id=f"rishabbala/{model_name}",
        commit_message="Add feature exteractor CCT",
        use_temp_dir=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="cct_14_7x2_384",
        type=str,
        help="Name of the cct model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_cct_checkpoint(args.model_name, args.pytorch_dump_folder_path)
