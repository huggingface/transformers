# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Convert LLaVa 1.6 checkpoints from the original repository.

URL: https://github.com/haotian-liu/LLaVA/tree/main.
"""

import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open

from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaImageProcessor,
    LlavaProcessor,
)


KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
    "language_model.model.image_newline": "image_newline",
}


def load_original_state_dict():
    # note: this is currently defined for liuhaotian/llava-v1.6-mistral-7b
    filenames = [
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    ]
    filepaths = [
        hf_hub_download(repo_id="liuhaotian/llava-v1.6-mistral-7b", filename=file, repo_type="model")
        for file in filenames
    ]

    original_state_dict = {}
    for path in filepaths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                original_state_dict[key] = f.get_tensor(key)

    return original_state_dict


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value
    return new_state_dict


def load_image():
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


def convert_llava_to_hf(model_id, pytorch_dump_folder_path, push_to_hub=False):
    # load original config
    filepath = hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model")
    # read json
    with open(filepath) as f:
        data = json.load(f)
        print(data)

    text_model_id = data["_name_or_path"]
    vision_model_id = data["mm_vision_tower"]

    torch.set_default_dtype(torch.float16)
    text_config = AutoConfig.from_pretrained(text_model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    # tokenizer.add_special_tokens({"pad_token": "<pad>"})

    image_processor = LlavaImageProcessor.from_pretrained(vision_model_id)
    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)

    config = LlavaConfig(text_config=text_config, mm_patch_merge_type="spatial_unpad")
    config.pad_token_id = 32001

    with torch.device("meta"):
        model = LlavaForConditionalGeneration(config)

    # load original state dict
    state_dict = load_original_state_dict()
    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, assign=True)

    # TODO verify input_ids
    image = load_image()
    text = "hello world"
    inputs = processor(images=image, text=text, return_tensors="pt")

    for k, v in inputs.items():
        print(k, v.shape)

    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="llava_1_6_input_ids.pt", repo_type="dataset")
    original_input_ids = torch.load(filepath, map_location="cpu")
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="llava_1_6_pixel_values.pt", repo_type="dataset")
    original_pixel_values = torch.load(filepath, map_location="cpu")

    assert torch.allclose(original_pixel_values, inputs.pixel_values.half())

    print(tokenizer.decode([id for id in original_input_ids.tolist()[0] if id != -200]))

    # TODO test single forward pass
    device = "cuda:1"
    model.to(device)

    print("Single forward pass")
    with torch.inference_mode():
        inputs = inputs.to(device)
        outputs = model(**inputs)
        print("Shape of logits:", outputs.logits.shape)

    # TODO test generation
    # pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    # mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    # n = pre_expansion_embeddings.size()[0]
    # sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    # dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # # We add an image token so we resize the model
    # # Pad to 64 for performance reasons
    # pad_shape = 64
    # model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
    # model.language_model.model.embed_tokens.weight.data[32000:] = torch.stack(
    #     tuple((dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[32000:].shape[0]))),
    #     dim=0,
    # )
    # model.language_model.lm_head.weight.data[32000:] = torch.stack(
    #     tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[32000:].shape[0]))),
    #     dim=0,
    # )

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor for {model_id} to {pytorch_dump_folder_path}")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        repo_id = model_id.split("/")[-1]
        model.push_to_hub(f"nielsr/{repo_id}-hf")
        processor.push_to_hub(f"nielsr/{repo_id}-hf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        help="Hub location of the model to convert",
        default="liuhaotian/llava-v1.6-mistral-7b",
        required=False,
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()

    convert_llava_to_hf(args.model_id, args.pytorch_dump_folder_path, args.push_to_hub)
