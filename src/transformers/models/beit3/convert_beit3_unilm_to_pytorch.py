# coding=utf-8
# Copyright 2024 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
import copy
import json

import numpy as np
import pandas as pd
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    Beit3Config,
    Beit3ForCaptioning,
    Beit3ForImageClassification,
    Beit3ForImagesAndTextClassification,
    Beit3ForImageTextRetrieval,
    Beit3ForQuestionAnswering,
    Beit3Processor,
    BeitImageProcessor,
    XLMRobertaTokenizer,
)
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


model_type_to_class_mapping = {
    "image_classification": Beit3ForImageClassification,
    "vqa": Beit3ForQuestionAnswering,
    "visual_reasoning": Beit3ForImagesAndTextClassification,
    "image_captioning": Beit3ForCaptioning,
    "image_text_retrieval": Beit3ForImageTextRetrieval,
}

rename_key_mappings = {
    "head": "classifier",
    "text_embed": "text_embedding",
    "vision_embed": "vision_embedding",
    "k_proj": "key_proj",
    "q_proj": "query_proj",
    "v_proj": "value_proj",
    "A": "image",
    "B": "text",
    "layer_norm": "fc_norm",
    "self_attn_fc_norm": "self_attn_layer_norm",
    "final_fc_norm": "final_layer_norm",
    "first": "first",
    "vision_embedding.proj": "vision_embedding.projection",
}


def get_id2label_for_imagenet_1k():
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def get_id2label_for_vqa():
    beit3_vqa_labels = "https://huggingface.co/datasets/Raghavan/beit3_vqa_answer2label.txt/raw/main/answer2label.txt"
    data = pd.read_json(beit3_vqa_labels, lines=True)
    id2label = dict(data[["label", "answer"]].values.tolist())
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def get_base_config_image_classification():
    id2label, label2id = get_id2label_for_imagenet_1k()
    return Beit3Config(
        hidden_size=768, num_labels=1000, id2label=id2label, label2id=label2id, intermediate_size=768 * 4
    )


def get_large_config_image_classification():
    id2label, label2id = get_id2label_for_imagenet_1k()
    return Beit3Config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=1024 * 4,
        num_labels=1000,
        id2label=id2label,
        label2id=label2id,
    )


def get_base_config_vqa(img_size):
    id2label, label2id = get_id2label_for_vqa()
    return Beit3Config(
        intermediate_size=768 * 4, num_labels=3129, image_size=img_size, id2label=id2label, label2id=label2id
    )


def get_large_config_vqa(img_size):
    id2label, label2id = get_id2label_for_vqa()
    return Beit3Config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=1024 * 4,
        num_labels=3129,
        image_size=img_size,
        id2label=id2label,
        label2id=label2id,
    )


def get_base_config_visual_reasoning(img_size):
    id2label = {0: "False", 1: "True"}
    label2id = {v: k for k, v in id2label.items()}

    return Beit3Config(
        intermediate_size=768 * 4,
        num_labels=2,
        image_size=img_size,
        normalize_before=True,
        encoder_normalize_before=True,
        id2label=id2label,
        label2id=label2id,
    )


def get_large_config_visual_reasoning(img_size):
    id2label = {0: "False", 1: "True"}
    label2id = {v: k for k, v in id2label.items()}
    return Beit3Config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=1024 * 4,
        num_labels=2,
        image_size=img_size,
        normalize_before=True,
        encoder_normalize_before=True,
        id2label=id2label,
        label2id=label2id,
    )


def get_base_config_captioning(img_size):
    return Beit3Config(
        intermediate_size=768 * 4,
        num_labels=2,
        image_size=img_size,
        normalize_before=True,
        encoder_normalize_before=True,
    )


def get_large_config_captioning(img_size):
    return Beit3Config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=1024 * 4,
        num_labels=2,
        image_size=img_size,
        normalize_before=True,
        encoder_normalize_before=True,
    )


def get_base_config_image_text_retrieval(img_size):
    return Beit3Config(
        intermediate_size=768 * 4,
        num_labels=2,
        image_size=img_size,
        normalize_before=True,
        encoder_normalize_before=True,
    )


def get_large_config_image_text_retrieval(img_size):
    return Beit3Config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=1024 * 4,
        num_labels=2,
        image_size=img_size,
        normalize_before=True,
        encoder_normalize_before=True,
    )


def rename_keys(model_state_dict, beit3_model_type):
    new_state_dict = copy.deepcopy(model_state_dict)
    rename_key_names = rename_key_mappings.keys()
    for key in model_state_dict:
        current_key = copy.deepcopy(key)
        for rename_key in rename_key_names:
            if rename_key in current_key:
                val = new_state_dict.pop(current_key)
                new_state_dict[current_key.replace(rename_key, rename_key_mappings[rename_key])] = val
                current_key = current_key.replace(rename_key, rename_key_mappings[rename_key])

    if beit3_model_type == "vqa":
        keys_to_be_renamed_for_vqa = [
            "pooler.norm.weight",
            "pooler.norm.bias",
            "pooler.dense.weight",
            "pooler.dense.bias",
        ]
        for key in keys_to_be_renamed_for_vqa:
            new_state_dict[f"beit3.{key}"] = new_state_dict[key]
            del new_state_dict[key]
    return new_state_dict


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


def get_tokenizer():
    return XLMRobertaTokenizer.from_pretrained(
        "https://conversationhub.blob.core.windows.net/beit-share-public/beit3/sentencepiece/beit3.spm?sv=2021-10-04"
        + "&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r"
        + "&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D"
    )


def convert_beit3_checkpoint(checkpoint_url, pytorch_dump_folder_path, beit3_model_type, validate_logits):
    img_size = 224
    if "480" in checkpoint_url:
        img_size = 480
    elif "768" in checkpoint_url:
        img_size = 768
    elif "384" in checkpoint_url:
        img_size = 384

    config = Beit3Config()
    if beit3_model_type == "image_classification":
        if "base" in checkpoint_url:
            config = get_base_config_image_classification()
        else:
            config = get_large_config_image_classification()
    elif beit3_model_type == "vqa":
        if "base" in checkpoint_url:
            config = get_base_config_vqa(img_size)
        else:
            config = get_large_config_vqa(img_size)
    elif beit3_model_type == "visual_reasoning":
        if "base" in checkpoint_url:
            config = get_base_config_visual_reasoning(img_size)
        else:
            config = get_large_config_visual_reasoning(img_size)
    elif beit3_model_type == "image_captioning":
        if "base" in checkpoint_url:
            config = get_base_config_captioning(img_size)
        else:
            config = get_large_config_captioning(img_size)
    elif beit3_model_type == "image_text_retrieval":
        if "base" in checkpoint_url:
            config = get_base_config_image_text_retrieval(img_size)
        else:
            config = get_large_config_image_text_retrieval(img_size)

    ulilm_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", check_hash=True)

    model = model_type_to_class_mapping[beit3_model_type](config)

    model_state_dict = ulilm_state_dict["model"]
    model_state_dict = rename_keys(model_state_dict, beit3_model_type)
    model.load_state_dict(model_state_dict)

    image_processor = BeitImageProcessor(
        do_resize=True, size=img_size, do_center_crop=False, image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD
    )
    tokenizer = get_tokenizer()
    beit3_processor = Beit3Processor(image_processor, tokenizer)
    image = prepare_img()
    model.eval()
    input = beit3_processor(text=["This is photo of a cat"], images=image)
    if "beit3_base_patch16_224_in1k" in checkpoint_url:
        output = model(pixel_values=torch.tensor(input["pixel_values"]))
        if validate_logits:
            assert output.logits.shape == torch.Size([1, 1000])
            np.testing.assert_allclose(
                output.logits.detach().numpy()[:, :3], torch.tensor([[-0.260473, -0.420061, -0.492118]]), rtol=1e-05
            )
    elif "beit3_base_patch16_480_vqa" in checkpoint_url:
        output = model(
            input_ids=torch.tensor(input["input_ids"]),
            pixel_values=torch.tensor(input["pixel_values"]),
            attention_mask=torch.ones(input["input_ids"].shape),
        )
        if validate_logits:
            assert output.logits.shape == torch.Size([1, 3129])
            np.testing.assert_allclose(
                output.logits.detach().numpy()[:, :3], torch.tensor([[-10.862484, -12.388088, -7.6599636]]), rtol=1e-05
            )
    elif "beit3_base_patch16_224_nlvr2" in checkpoint_url:
        pixel_values = torch.cat(
            (torch.tensor(input["pixel_values"]).unsqueeze(1), torch.tensor(input["pixel_values"]).unsqueeze(1)), dim=1
        )
        output = model(
            input_ids=torch.tensor(input["input_ids"]),
            pixel_values=pixel_values,
            attention_mask=torch.ones(input["input_ids"].shape),
        )
        if validate_logits:
            assert output.logits.shape == torch.Size([1, 2])
            np.testing.assert_allclose(
                output.logits.detach().numpy(), torch.tensor([[6.593818, -6.582055]]), rtol=1e-05
            )
    elif "beit3_base_patch16_480_coco_captioning" in checkpoint_url:
        language_masked_pos = torch.zeros(input["input_ids"].shape)
        to_fill = list(range(0, input["input_ids"].shape[1], 3))
        language_masked_pos[:, to_fill] = 1
        output = model(
            input_ids=torch.tensor(input["input_ids"]),
            pixel_values=torch.tensor(input["pixel_values"]),
            attention_mask=torch.ones(input["input_ids"].shape),
            language_masked_pos=language_masked_pos,
        )
        if validate_logits:
            assert output.logits.shape == torch.Size([3, 64010])
            np.testing.assert_allclose(
                output.logits.detach().numpy()[0, :3], torch.tensor([-19.36987, -19.369905, -17.022049]), rtol=1e-05
            )
    elif "beit3_base_patch16_384_coco_retrieval" in checkpoint_url:
        another_input_ids = beit3_processor(text=["This is photo of a dog"], images=image)["input_ids"]
        output = model(
            input_ids=torch.tensor([input["input_ids"][0], another_input_ids[0]]),
            pixel_values=torch.tensor([input["pixel_values"][0], input["pixel_values"][0]]),
        )
        if validate_logits:
            assert round(float(output.loss.detach().numpy()), 4) == 1.8435

    print(f"Saving model to {pytorch_dump_folder_path}")
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    model.save_pretrained(pytorch_dump_folder_path)
    # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    # feature_extractor.save_pretrained(pytorch_dump_folder_path)
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    parser.add_argument(
        "--beit3_model_type",
        default=None,
        type=str,
        help="Beit3 model type, it has to be one of image_classification, vqa, visual_reasoning,"
        "image_captioning,image_text_retrieval",
    )
    parser.add_argument(
        "--validate_logits",
        action="store_false",
        help="whether to assert logits outputs",
    )
    args = parser.parse_args()

    if args.beit3_model_type not in model_type_to_class_mapping.keys():
        raise Exception(
            "beit3_model_type should be one of image_classification, vqa,visual_reasoning,"
            "image_captioning,image_text_retrieval"
        )

    args = parser.parse_args()
    convert_beit3_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.beit3_model_type, args.validate_logits
    )
