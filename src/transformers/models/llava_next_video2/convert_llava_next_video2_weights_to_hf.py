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

"""Convert LlavaNextVideo2 checkpoints from the original repository.

URL: https://github.com/LLaVA-VL/LLaVA-NeXT/tree/inference
"""

import argparse
import glob
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlavaNextImageProcessor,
    LlavaNextVideo2Config,
    LlavaNextVideo2ForConditionalGeneration,
    LlavaNextVideoImageProcessor,
    LlavaNextVideoProcessor,
    SiglipVisionConfig,
)


KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",
    ".vision_resampler": "",  # all lmms-lab models do avg pooling, so no vision_resampler
    "model.mm_projector": "multi_modal_projector",
    "model": "model.model",
    "vision_model.model": "vision_model",
    "lm_head": "language_model.lm_head",
    "model.model": "language_model.model",
    "multi_modal_projector.0": "multi_modal_projector.linear_1",
    "multi_modal_projector.2": "multi_modal_projector.linear_2",
    "language_model.model.image_newline": "image_newline",
}


chat_qwen = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>' }}"
    "{% endfor %}"
    "{# Render all videos first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'video') %}"
    "{{ '<video>' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ '\n' + content['text'] }}"
    "{% endfor %}"
    "{% else %}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{% generation %}"
    "{{ '\n' + content['text'] }}"
    "{% endgeneration %}"
    "{% endfor %}"
    "{% endif %}"
    "{{'<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

model2template = {
    "lmms-lab/LLaVA-Video-7B-Qwen2": chat_qwen,
    "lmms-lab/LLaVA-Video-32B-Qwen": chat_qwen,
    "lmms-lab/LLaVA-Video-72B-Qwen2": chat_qwen,
}


def load_original_state_dict(model_id):
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
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

        new_state_dict[key] = value.to(torch.bfloat16)
    return new_state_dict


def convert_llava_to_hf(model_id, pytorch_dump_folder_path, push_to_hub=False):
    # load original config
    filepath = hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model")
    with open(filepath) as f:
        data = json.load(f)
        print(data)

    if model_id == "lmms-lab/LLaVA-Video-7B-Qwen2":
        text_model_id = "Qwen/Qwen2-7B-Instruct"
    elif model_id == "lmms-lab/LLaVA-Video-32B-Qwen":
        text_model_id = "Qwen/Qwen1.5-32B-Chat"
    elif model_id == "lmms-lab/LLaVA-Video-72B-Qwen2":
        text_model_id = "Qwen/Qwen2-72B-Instruct"
    else:
        raise ValueError("Incorrect checkpoint referenced. Text model-id not identified!")

    image_token_index = 151646
    video_token_index = 151647
    vision_model_id = data["mm_vision_tower"]

    if vision_model_id == "google/siglip-so400m-patch14-384":
        vision_config = SiglipVisionConfig(
            hidden_size=1152,
            image_size=384,
            intermediate_size=4304,
            num_attention_heads=16,
            num_hidden_layers=26,  # drop the last layer
            patch_size=14,
            vision_use_head=False,  # no head
        ).to_dict()
        vision_feature_select_strategy = "full"
        vision_feature_layer = -1
    else:
        vision_config = None  # fallback to default
        vision_feature_select_strategy = "default"
        vision_feature_layer = -2

    torch.set_default_dtype(torch.bfloat16)
    text_config = AutoConfig.from_pretrained(text_model_id)
    text_config = text_config.to_dict()

    tokenizer = AutoTokenizer.from_pretrained(
        text_model_id,
        use_fast=True,
        padding_side="left",
        extra_special_tokens={"image_token": "<image>", "video_token": "<video>"},
    )

    image_processor = LlavaNextImageProcessor.from_pretrained(
        vision_model_id, image_grid_pinpoints=data["image_grid_pinpoints"]
    )
    if vision_model_id == "google/siglip-so400m-patch14-384":
        image_processor.do_center_crop = False  # otherwise it will default to True in `LlavaNextImageProcessor`
        image_processor.crop_size = None

    video_processor = LlavaNextVideoImageProcessor.from_pretrained(vision_model_id)
    processor = LlavaNextVideoProcessor(
        tokenizer=tokenizer,
        video_processor=video_processor,
        image_processor=image_processor,
        chat_template=model2template[model_id],
        vision_feature_select_strategy=vision_feature_select_strategy,
        patch_size=14,
    )

    config = LlavaNextVideo2Config(
        text_config=text_config,
        vision_config=vision_config,
        image_grid_pinpoints=image_processor.image_grid_pinpoints,
        use_image_newline_parameter=True,
        spatial_pool_mode=data.get("mm_spatial_pool_mode", "average"),
        video_token_index=video_token_index,
        image_token_index=image_token_index,
        vision_feature_select_strategy=vision_feature_select_strategy,
        vision_feature_layer=vision_feature_layer,
    )

    with init_empty_weights():
        model = LlavaNextVideo2ForConditionalGeneration(config)

    # load original state dict
    state_dict = load_original_state_dict(model_id)
    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, assign=True, strict=True)

    # See https://nlp.stanford.edu/~johnhew/vocab-expansion.html for why we get mean/stdev this way to expand embeddings
    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model
    # Pad to 64 for performance reasons
    pad_shape = 64
    vocab_size = config.text_config.vocab_size

    # this one has 2 additional tokens, namely <image>, <video> and <pad>
    num_tokens = vocab_size + 3
    model.resize_token_embeddings(num_tokens, pad_to_multiple_of=pad_shape)
    model.language_model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
        tuple(
            (dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[vocab_size:].shape[0]))
        ),
        dim=0,
    )
    model.language_model.lm_head.weight.data[vocab_size:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[vocab_size:].shape[0]))),
        dim=0,
    )

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor for {model_id} to {pytorch_dump_folder_path}")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        repo_id = model_id.split("/")[-1]
        repo_id = repo_id.replace("LLaVA-Video", "LLaVA-Next-Video")
        print(f"Pushing model to hub repo: {repo_id}")
        model.push_to_hub(f"llava-hf/{repo_id}-hf")
        processor.push_to_hub(f"llava-hf/{repo_id}-hf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        help="Hub location of the model to convert",
        default="lmms-lab/LLaVA-Video-7B-Qwen2",
        choices=[
            "lmms-lab/LLaVA-Video-7B-Qwen2",
            "lmms-lab/LLaVA-Video-32B-Qwen",
            "lmms-lab/LLaVA-Video-72B-Qwen2",
        ],
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
