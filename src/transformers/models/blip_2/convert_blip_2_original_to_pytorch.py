# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import re

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

import requests

# pip3 install salesforce-lavis
from lavis.models import load_model_and_preprocess
from transformers import (
    BertTokenizer,
    Blip2Config,
    Blip2ForConditionalGeneration,
)


def load_demo_image(image_size, device):
    img_url = "https://storage.googleapis.com/sfr-vision-language-research/blip2/demo.jpg"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]
    )
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def rename_key(key):
    if "visual_encoder" in key:
        key = re.sub("visual_encoder*", "vision_model.encoder", key)
    if "blocks" in key:
        key = re.sub(r"blocks", "layers", key)
    if "attn" in key:
        key = re.sub(r"attn", "self_attn", key)
    if "norm1" in key:
        key = re.sub(r"norm1", "layer_norm1", key)
    if "norm2" in key:
        key = re.sub(r"norm2", "layer_norm2", key)
    if "encoder.norm" in key:
        key = re.sub(r"encoder.norm", "post_layernorm", key)
    if "encoder.patch_embed.proj" in key:
        key = re.sub(r"encoder.patch_embed.proj", "embeddings.patch_embedding", key)

    if "encoder.pos_embed" in key:
        key = re.sub(r"encoder.pos_embed", "embeddings.position_embedding", key)
    if "encoder.cls_token" in key:
        key = re.sub(r"encoder.cls_token", "embeddings.class_embedding", key)

    if "self_attn" in key:
        key = re.sub(r"self_attn.proj", "self_attn.projection", key)

    return key


def get_blip2_config(model_name):
    return Blip2Config()


@torch.no_grad()
def convert_blip2_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to Transformers design.
    """
    config = get_blip2_config(model_name)

    hf_model = Blip2ForConditionalGeneration(config).eval()

    # load original model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
    )
    original_model.eval()

    # TODO modify state dict
    state_dict = original_model.state_dict()

    hf_model.load_state_dict(state_dict)

    image_size = 384
    image = load_demo_image(image_size=image_size, device="cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    input_ids = tokenizer(["a picture of"]).input_ids

    out = hf_model.generate(image, input_ids)

    # TODO assert values
    # out_itm = hf_itm_model(question_input_ids, image, use_itm_head=True)
    # out = hf_itm_model(question_input_ids, image, use_itm_head=False)

    # assert out[0].item() == 0.2110687494277954
    # assert torch.nn.functional.softmax(out_itm[0], dim=1)[:, 1].item() == 0.45698845386505127

    if pytorch_dump_folder_path is not None:
        hf_model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        hf_model.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model and processor to the hub after converting")

    args = parser.parse_args()

    convert_blip2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
