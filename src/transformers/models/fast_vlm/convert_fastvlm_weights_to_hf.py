# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import argparse
import glob
import os
import re

import requests
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors import safe_open

from transformers import (
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    CLIPImageProcessor,
    FastVlmConfig,
    FastVlmForConditionalGeneration,
    LlavaProcessor,
)


os.environ["TIMM_FUSED_ATTN"] = (
    "0"  # to avoid logits diverging, needed because the original implementation uses regular (not fused) atteniton
)

KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.vision_tower.model": "model.vision_tower.timm_model",
    "patch_embed": "stem",
    "layers": "language_model.layers",
    "embed_tokens": "language_model.embed_tokens",
    "layer_scale_1": "layer_scale_1.gamma",
    "layer_scale_2": "layer_scale_2.gamma",
    "mm_projector.0": "multi_modal_projector.linear_1",
    "mm_projector.2": "multi_modal_projector.linear_2",
    "conv_exp": "final_conv",
    "se.reduce": "se.fc1",
    "se.expand": "se.fc2",
    "convffn": "mlp",
    "lkb_reparam": "reparam_conv",
}


def map_to_stage(number):
    number = int(number)
    if number == 0:
        return 0
    if number in {1, 2}:
        return 1
    if number in {3, 4}:
        return 2
    if number in {5, 6, 7}:
        return 3
    if number in {8, 9, 10}:
        return 4


def load_original_state_dict(model_id):
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    if "model.vision_tower.vision_tower.model.head.proj" in original_state_dict:
        del original_state_dict["model.vision_tower.vision_tower.model.head.proj"]
    return original_state_dict


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}

    single_pattern = r"network\.(\d{1,2})"
    double_pattern = r"network\.(\d{1,2})\.(\d{1,2})"
    pos_embedding_pattern = r"stages\.(\d{1,2})\.reparam_conv"

    for key, value in state_dict.items():
        if key.endswith("layer_scale"):
            key = key.replace("layer_scale", "layer_scale.gamma")
        if key.startswith("model.norm"):
            key = key.replace("model.norm", "model.language_model.norm")
        if "token_mixer" not in key:
            key = key.replace(".proj.", ".downsample.proj.")

        matches = re.findall(double_pattern, key)
        if len(matches) == 1:
            match = matches[0]
            key = key.replace(f"network.{match[0]}.{match[1]}", f"stages.{map_to_stage(match[0])}.blocks.{match[1]}")

        matches = re.findall(single_pattern, key)
        if len(matches) == 1:
            match = matches[0]
            key = key.replace(f"network.{match[0]}", f"stages.{map_to_stage(match[0])}")

        matches = re.findall(pos_embedding_pattern, key)
        if len(matches) == 1:
            match = matches[0]
            key = key.replace(f"stages.{match[0]}", f"stages.{match[0]}.pos_emb")

        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value
    return new_state_dict


def convert_fastvlm_to_hf(text_model_id, vision_model_id, output_hub_path, old_state_dict_id):
    torch.set_default_dtype(torch.bfloat16)

    text_config = AutoConfig.from_pretrained(text_model_id)
    vision_config = AutoConfig.from_pretrained(vision_model_id)
    vision_config.model_args = {"inference_mode": True}
    vision_config.hidden_size = vision_config.num_features
    vision_config.label2id = {}
    vision_config.id2label = {}
    config = FastVlmConfig(
        text_config=text_config,
        vision_config=vision_config,
    )
    config.vision_feature_select_strategy = "full"
    config.vision_feature_layer = -1
    config.image_token_index = 151646
    config.image_seq_length = 256

    tokenizer = AutoTokenizer.from_pretrained(
        text_model_id,
        chat_template="{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n'}}{# Render all images first #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>' }}{% endfor %}{# Render all text next #}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ '\n' + content['text'] }}{% endfor %}{{'<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}",
    )

    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)
    image_processor = CLIPImageProcessor(
        crop_size={"height": 1024, "width": 1024},
        image_mean=[0.0, 0.0, 0.0],
        image_std=[1.0, 1.0, 1.0],
        size={"shortest_edge": 1024},
    )

    processor = LlavaProcessor(tokenizer=tokenizer, image_processor=image_processor)
    processor.patch_size = 64  # effective patch size (2^6)

    model = FastVlmForConditionalGeneration(config)

    state_dict = load_original_state_dict(old_state_dict_id)
    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, strict=True, assign=True)

    pre_expansion_embeddings = model.language_model.embed_tokens.weight.data
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # We add an image token so we resize the model and pad to 64 for performance reasons
    pad_shape = 64
    vocab_size = config.text_config.vocab_size
    model.resize_token_embeddings(config.text_config.vocab_size + 1, pad_shape)
    model.language_model.embed_tokens.weight.data[vocab_size:] = torch.stack(
        tuple(dist.sample() for _ in range(model.language_model.embed_tokens.weight.data[vocab_size:].shape[0])),
        dim=0,
    )
    model.lm_head.weight.data[vocab_size:] = torch.stack(
        tuple(dist.sample() for _ in range(model.lm_head.weight.data[vocab_size:].shape[0])),
        dim=0,
    )

    conversation = [{"role": "user", "content": [{"type": "text", "text": "What are these?"}, {"type": "image"}]}]
    prompt = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
    inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to("cuda")
    inputs = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v) for k, v in inputs.items()}

    model = model.cuda()
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits

    # in order to get the same logits as in the Apple repo, we need to manually replace the original (Apple) LayerNorm2D with Timm's LayerNorm2D or vice versa
    # otherwise numerical errors accumulate
    if output_hub_path == "KamilaMila/FastVLM-0.5B":
        expected_shape = torch.Size([1, 280, 152000])
        expected_slice = torch.tensor([4.1250, 9.6875, 11.1875], device="cuda")
    elif output_hub_path == "KamilaMila/FastVLM-1.5B":
        expected_shape = torch.Size([1, 280, 152000])
        expected_slice = torch.tensor([3.3750, 11.5000, 11.8125], device="cuda")
    elif output_hub_path == "KamilaMila/FastVLM-7B":
        expected_shape = torch.Size([1, 280, 152128])
        expected_slice = torch.tensor([3.8281, 9.0625, 7.9062], device="cuda")

    logits_slice = logits[0, -1, :3]
    assert torch.allclose(expected_slice, logits_slice, atol=1e-8)
    assert logits.shape == expected_shape

    model.push_to_hub(output_hub_path)
    processor.push_to_hub(output_hub_path)
    print("Successfully pushed to hub!")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--text_model_id",
        default="Qwen/Qwen2-1.5B",
        help="Hub location of the text model",
    )
    parser.add_argument(
        "--vision_model_id",
        default="timm/fastvit_mci3.apple_mclip2_dfndr2b",
        help="Hub location of the vision model",
    )
    parser.add_argument(
        "--output_hub_path",
        default="KamilaMila/FastVLM-1.5B",
        help="Location on the hub of the converted model",
    )
    parser.add_argument(
        "--old_state_dict_id",
        default="apple/FastVLM-1.5B",
        help="Location on the hub of the raw state dict of the original model.",
    )
    args = parser.parse_args()
    convert_fastvlm_to_hf(args.text_model_id, args.vision_model_id, args.output_hub_path, args.old_state_dict_id)


if __name__ == "__main__":
    main()
