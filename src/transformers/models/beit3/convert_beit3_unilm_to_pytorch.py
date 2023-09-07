# coding=utf-8
# Copyright 2023 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from transformers import (
    Beit3Config,
    Beit3ForCaptioning,
    Beit3ForImageClassification,
    Beit3ForImageTextRetrieval,
    Beit3ForVisualQuestionAnswering,
    Beit3ForVisualReasoning,
    Beit3ImageProcessor,
    Beit3Processor,
    XLMRobertaTokenizer,
)


model_type_to_class_mapping = {
    "image_classification": Beit3ForImageClassification,
    "vqa": Beit3ForVisualQuestionAnswering,
    "visual_reasoning": Beit3ForVisualReasoning,
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
    "A": "first",
    "B": "second",
    "layer_norm": "fc_norm",
    "self_attn_fc_norm": "self_attn_layer_norm",
    "final_fc_norm": "final_layer_norm",
}


# SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


def get_base_config_image_classification():
    return Beit3Config(hidden_size=768 * 4, num_labels=1000)


def get_large_config_image_classification():
    return Beit3Config(embed_dim=1024, layers=24, num_attention_heads=16, hidden_size=1024 * 4, num_labels=1000)


def get_base_config_vqa(img_size):
    return Beit3Config(hidden_size=768 * 4, num_labels=3129, img_size=img_size)


def get_large_config_vqa(img_size):
    return Beit3Config(
        embed_dim=1024, layers=24, num_attention_heads=16, hidden_size=1024 * 4, num_labels=3129, img_size=img_size
    )


def get_base_config_visual_reasoning(img_size):
    return Beit3Config(
        hidden_size=768 * 4, num_labels=2, img_size=img_size, normalize_before=True, encoder_normalize_before=True
    )


def get_large_config_visual_reasoning(img_size):
    return Beit3Config(
        embed_dim=1024,
        layers=24,
        num_attention_heads=16,
        hidden_size=1024 * 4,
        num_labels=2,
        img_size=img_size,
        normalize_before=True,
        encoder_normalize_before=True,
    )


def get_base_config_captioning(img_size):
    return Beit3Config(
        hidden_size=768 * 4, num_labels=2, img_size=img_size, normalize_before=True, encoder_normalize_before=True
    )


def get_large_config_captioning(img_size):
    return Beit3Config(
        embed_dim=1024,
        layers=24,
        num_attention_heads=16,
        hidden_size=1024 * 4,
        num_labels=2,
        img_size=img_size,
        normalize_before=True,
        encoder_normalize_before=True,
    )


def get_base_config_image_text_retrieval(img_size):
    return Beit3Config(
        hidden_size=768 * 4, num_labels=2, img_size=img_size, normalize_before=True, encoder_normalize_before=True
    )


def get_large_config_image_text_retrieval(img_size):
    return Beit3Config(
        embed_dim=1024,
        layers=24,
        num_attention_heads=16,
        hidden_size=1024 * 4,
        num_labels=2,
        img_size=img_size,
        normalize_before=True,
        encoder_normalize_before=True,
    )


def rename_keys(model_state_dict):
    new_state_dict = copy.deepcopy(model_state_dict)
    rename_key_names = rename_key_mappings.keys()
    for key in model_state_dict:
        current_key = copy.deepcopy(key)
        for rename_key in rename_key_names:
            if rename_key in current_key:
                val = new_state_dict.pop(current_key)
                new_state_dict[current_key.replace(rename_key, rename_key_mappings[rename_key])] = val
                current_key = current_key.replace(rename_key, rename_key_mappings[rename_key])

    return new_state_dict


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


def get_tokenizer():
    return XLMRobertaTokenizer.from_pretrained(
        "https://huggingface.co/xlm-roberta-base/resolve/main/sentencepiece.bpe.model"
    )


def convert_beit3_checkpoint(checkpoint_url, pytorch_dump_folder_path, beit3_model_type):
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
    model_state_dict = rename_keys(model_state_dict)
    model.load_state_dict(model_state_dict)

    image_processor = Beit3ImageProcessor(do_resize=True, size=img_size)
    tokenizer = get_tokenizer()
    beit3_processor = Beit3Processor(image_processor, tokenizer)
    image = prepare_img()
    model.eval()
    input = beit3_processor(text=["This is photo of a cat"], images=image)
    if "beit3_base_patch16_224_in1k" in checkpoint_url:
        output = model(pixel_values=torch.tensor(input["pixel_values"]))
        assert output.logits.shape == torch.Size([1, 1000])
        np.testing.assert_allclose(
            output.logits.detach().numpy()[:, :3], torch.tensor([[-0.260473, -0.420061, -0.492118]]), rtol=1e-05
        )
    elif "beit3_base_patch16_480_vqa" in checkpoint_url:
        output = model(
            input_ids=torch.tensor(input["input_ids"]),
            pixel_values=torch.tensor(input["pixel_values"]),
            padding_mask=torch.ones(input["input_ids"].shape),
        )
        assert output.logits.shape == torch.Size([1, 3129])
        np.testing.assert_allclose(
            output.logits.detach().numpy()[:, :3], torch.tensor([[-10.862484, -12.388088, -7.6599636]]), rtol=1e-05
        )
    elif "beit3_base_patch16_224_nlvr2" in checkpoint_url:
        output = model(
            input_ids=torch.tensor(input["input_ids"]),
            pixel_values1=torch.tensor(input["pixel_values"]),
            pixel_values2=torch.tensor(input["pixel_values"]),
            padding_mask=torch.ones(input["input_ids"].shape),
        )
        assert output.logits.shape == torch.Size([1, 2])
        np.testing.assert_allclose(output.logits.detach().numpy(), torch.tensor([[6.5938, -6.5821]]), rtol=1e-05)
    elif "beit3_base_patch16_480_coco_captioning" in checkpoint_url:
        language_masked_pos = torch.zeros(input["input_ids"].shape)
        to_fill = list(range(0, input["input_ids"].shape[1], 3))
        language_masked_pos[:, to_fill] = 1
        output = model(
            input_ids=torch.tensor(input["input_ids"]),
            pixel_values=torch.tensor(input["pixel_values"]),
            padding_mask=torch.ones(input["input_ids"].shape),
            language_masked_pos=language_masked_pos,
        )
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
        assert round(float(output.loss.detach().numpy()), 4) == 0.7461

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    # feature_extractor.save_pretrained(pytorch_dump_folder_path)


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
        help="Beit3 model type, it has to be one of image_classification, vqa,visual_reasoning,"
        "image_captioning,image_text_retrieval",
    )
    args = parser.parse_args()

    if args.beit3_model_type not in model_type_to_class_mapping.keys():
        raise Exception(
            "beit3_model_type should be one of image_classification, vqa,visual_reasoning,"
            "image_captioning,image_text_retrieval"
        )

    args = parser.parse_args()
    convert_beit3_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.beit3_model_type)
