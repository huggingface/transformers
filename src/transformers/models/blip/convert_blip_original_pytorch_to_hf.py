# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

# git clone https://github.com/salesforce/BLIP.git
from models.blip import blip_decoder
from models.blip_itm import blip_itm
from models.blip_vqa import blip_vqa
from transformers import (
    BertTokenizer,
    BlipConfig,
    BlipForConditionalGeneration,
    BlipForImageTextRetrieval,
    BlipForQuestionAnswering,
)


def load_demo_image(image_size, device):
    img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
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


@torch.no_grad()
def convert_blip_checkpoint(pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = BlipConfig.from_pretrained(config_path)
    else:
        config = BlipConfig(projection_dim=512, text_config={}, vision_config={})

    hf_model = BlipForConditionalGeneration(config).eval()

    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"

    pt_model = blip_decoder(pretrained=model_url, image_size=384, vit="base")
    pt_model = pt_model.eval()

    modified_state_dict = pt_model.state_dict()
    for key in modified_state_dict.copy():
        value = modified_state_dict.pop(key)
        renamed_key = rename_key(key)
        modified_state_dict[renamed_key] = value

    hf_model.load_state_dict(modified_state_dict)

    image_size = 384
    image = load_demo_image(image_size=image_size, device="cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    input_ids = tokenizer(["a picture of"]).input_ids

    out = hf_model.generate(image, input_ids)

    assert out[0].tolist() == [30522, 1037, 3861, 1997, 1037, 2450, 3564, 2006, 1996, 3509, 2007, 2014, 3899, 102]

    out = hf_model.generate(image)

    assert out[0].tolist() == [30522, 1037, 2450, 3564, 2006, 1996, 3509, 2007, 2014, 3899, 102]

    if pytorch_dump_folder_path is not None:
        hf_model.save_pretrained(pytorch_dump_folder_path)

    # model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_vqa.pth'
    model_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
    )

    vqa_model = blip_vqa(pretrained=model_url, image_size=image_size, vit="base")
    vqa_model.eval()

    modified_state_dict = vqa_model.state_dict()
    for key in modified_state_dict.copy():
        value = modified_state_dict.pop(key)
        renamed_key = rename_key(key)
        modified_state_dict[renamed_key] = value

    hf_vqa_model = BlipForQuestionAnswering(config)

    hf_vqa_model.load_state_dict(modified_state_dict)

    question = ["How many dogs are in this image?"]
    question_input_ids = tokenizer(question, return_tensors="pt").input_ids

    answer = hf_vqa_model.generate(question_input_ids, image)
    print(tokenizer.decode(answer[0]))

    assert tokenizer.decode(answer[0]) == "[UNK] 1 [SEP]"
    if pytorch_dump_folder_path is not None:
        hf_vqa_model.save_pretrained(pytorch_dump_folder_path + "_vqa")

    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth"

    itm_model = blip_itm(pretrained=model_url, image_size=image_size, vit="base")
    itm_model.eval()

    modified_state_dict = itm_model.state_dict()
    for key in modified_state_dict.copy():
        value = modified_state_dict.pop(key)
        renamed_key = rename_key(key)
        modified_state_dict[renamed_key] = value

    hf_itm_model = BlipForImageTextRetrieval(config)

    question = ["A picture of a woman with a dog sitting in a beach"]
    question_input_ids = tokenizer(
        question,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=35,
    ).input_ids

    hf_itm_model.load_state_dict(modified_state_dict)
    hf_itm_model.eval()

    out_itm = hf_itm_model(question_input_ids, image, use_itm_head=True)
    out = hf_itm_model(question_input_ids, image, use_itm_head=False)

    assert out[0].item() == 0.2110687494277954
    assert torch.nn.functional.softmax(out_itm[0], dim=1)[:, 1].item() == 0.45698845386505127

    if pytorch_dump_folder_path is not None:
        hf_itm_model.save_pretrained(pytorch_dump_folder_path + "_itm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    args = parser.parse_args()

    convert_blip_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
