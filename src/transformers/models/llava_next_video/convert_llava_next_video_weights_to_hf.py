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

"""Convert LLaVa-NeXT-Video checkpoints from the original repository.

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
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    LlavaNextImageProcessor,
    LlavaNextVideoConfig,
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoImageProcessor,
    LlavaNextVideoProcessor,
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

# {{SYSTEM_PROMPT}} USER: <image>\n{{PROMPT}} ASSISTANT:" assistant end with "</s> "
chat_vicuna = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ message['content'][0]['text'] }}"
    "{% else %}"
    "{{ message['role'].upper() + ': '}}"
    "{% endif %}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] + ' '}}"
    "{% endfor %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'ASSISTANT:' }}"
    "{% endif %}"
)

# "[INST] <image>\nWhat is shown in this image? [/INST]" assistant end with "</s> "
chat_mistral = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '[INST] ' }}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] }}"
    "{% endfor %}"
    "{{' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    r"{{ ' ' + message['content'][0]['text'] + '<\s> '}}"
    "{% else %}"
    "{{ raise_exception('Only user and assistant roles are supported!') }}"
    "{% endif %}"
    "{% endfor %}"
)

# "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"
chat_yi = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{# Render all images first #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}"
    "{{ '<image>\n' }}"
    "{% endfor %}"
    "{# Render all text next #}"
    "{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}"
    "{{ content['text'] }}"
    "{% endfor %}"
    "{{'<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

model2template = {
    "lmms-lab/LLaVA-NeXT-Video-7B-32K": chat_mistral,
    "lmms-lab/LLaVA-NeXT-Video-7B": chat_vicuna,
    "lmms-lab/LLaVA-NeXT-Video-7B-DPO": chat_vicuna,
    "lmms-lab/LLaVA-NeXT-Video-34B": chat_yi,
    "lmms-lab/LLaVA-NeXT-Video-34B-DPO": chat_yi,
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

    if model_id == "lmms-lab/LLaVA-NeXT-Video-7B-32K":
        text_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        video_token_index = 32000
        image_token_index = 32001
        overwrite_text_config = {}
    elif model_id in ["lmms-lab/LLaVA-NeXT-Video-7B", "lmms-lab/LLaVA-NeXT-Video-7B-DPO"]:
        text_model_id = "lmsys/vicuna-7b-v1.5"
        video_token_index = 32000
        image_token_index = 32001
        overwrite_text_config = {"factor": 2.0, "type": "linear"}
    elif model_id in ["lmms-lab/LLaVA-NeXT-Video-34B", "lmms-lab/LLaVA-NeXT-Video-34B-DPO"]:
        text_model_id = "NousResearch/Nous-Hermes-2-Yi-34B"
        video_token_index = 64000
        image_token_index = 64001
        overwrite_text_config = {}
    else:
        raise ValueError("Incorrect checkpoint referenced. Text model-id not identified!")

    vision_model_id = data["mm_vision_tower"]

    torch.set_default_dtype(torch.bfloat16)
    text_config = AutoConfig.from_pretrained(text_model_id)
    text_config = text_config.to_dict()
    text_config.update(overwrite_text_config)

    tokenizer = AutoTokenizer.from_pretrained(text_model_id, use_fast=True, padding_side="left")
    tokenizer.add_tokens(AddedToken("<video>", special=True, normalized=False), special_tokens=True)
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)

    image_processor = LlavaNextImageProcessor.from_pretrained(vision_model_id)
    video_processor = LlavaNextVideoImageProcessor.from_pretrained(vision_model_id)
    processor = LlavaNextVideoProcessor(
        tokenizer=tokenizer,
        video_processor=video_processor,
        image_processor=image_processor,
        chat_template=model2template[model_id],
    )

    config = LlavaNextVideoConfig(
        text_config=text_config,
        image_grid_pinpoints=image_processor.image_grid_pinpoints,
        use_image_newline_parameter=True,
        video_token_index=video_token_index,
        image_token_index=image_token_index,
    )

    with init_empty_weights():
        model = LlavaNextVideoForConditionalGeneration(config)

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
        print(f"Pushing model to hub repo: {repo_id}")
        model.push_to_hub(f"llava-hf/{repo_id}-hf")
        processor.push_to_hub(f"llava-hf/{repo_id}-hf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        help="Hub location of the model to convert",
        default="lmms-lab/LLaVA-NeXT-Video-7B",
        choices=[
            "lmms-lab/LLaVA-NeXT-Video-7B",
            "lmms-lab/LLaVA-NeXT-Video-7B-DPO",
            "lmms-lab/LLaVA-NeXT-Video-7B-32K",
            "lmms-lab/LLaVA-NeXT-Video-34B",
            "lmms-lab/LLaVA-NeXT-Video-34B-DPO",
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
