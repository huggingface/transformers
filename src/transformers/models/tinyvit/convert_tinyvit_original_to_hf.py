# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert TinyViT checkpoints from the original repository.

URL: https://github.com/microsoft/Cream/tree/main/TinyViT"""

import argparse
import json

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

from transformers import BitImageProcessor, TinyVitConfig, TinyVitForImageClassification
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_tinyvit_config(model_name):
    config = TinyVitConfig()

    if "5m" in model_name:
        hidden_sizes = [64, 128, 160, 320]
        depths = [2, 2, 6, 2]
        num_heads = [2, 4, 5, 10]
        window_sizes = [7, 7, 14, 7]
    elif "11m" in model_name:
        hidden_sizes = [64, 128, 256, 448]
        depths = [2, 2, 6, 2]
        num_heads = [2, 4, 8, 14]
        window_sizes = [7, 7, 14, 7]
    elif "21m" in model_name:
        hidden_sizes = [96, 192, 384, 576]
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 18]
        window_sizes = [7, 7, 14, 7]
    else:
        raise ValueError("Model name should include either 5m, 11m or 21m")

    if model_name.replace("-distill", "").endswith("22k"):
        num_labels = 21841
        filename = "imagenet-22k-id2label.json"
    elif model_name.replace("-distill", "").endswith("1k"):
        num_labels = 1000
        filename = "imagenet-1k-id2label.json"
    else:
        raise ValueError("Model name should include either 22k or 1k")

    repo_id = "huggingface/label-files"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    if num_labels == 21841:
        # this dataset contains 21843 labels but the model only has 21841
        # we delete the classes as mentioned in https://github.com/google-research/big_transfer/issues/18
        del id2label[9205]
        del id2label[15027]

    config.hidden_sizes = hidden_sizes
    config.depths = depths
    config.num_heads = num_heads
    config.window_sizes = window_sizes
    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


name_to_checkpoint_url = {
    "tinyvit-5m-22k-distill": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22k_distill.pth",
    "tinyvit-5m-22kto1k-distill": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22kto1k_distill.pth",
    "tinyvit-11m-22k-distill": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22k_distill.pth",
    "tinyvit-11m-22kto1k-distill": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22kto1k_distill.pth",
    "tinyvit-21m-22k-distill": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22k_distill.pth",
    "tinyvit-21m-22kto1k-distill": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_distill.pth",
    "tinyvit-21m-22kto1k-384-distill": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_384_distill.pth",
    "tinyvit-21m-22kto1k-512-distill": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_512_distill.pth",
    "tinyvit-5m-1k": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_1k.pth",
    "tinyvit-11m-1k": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_1k.pth",
    "tinyvit-21m-1k": "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_1k.pth",
}


def convert_tinyvit_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    config = get_tinyvit_config(model_name)
    model = TinyVitForImageClassification(config)

    checkpoint_url = name_to_checkpoint_url[model_name]
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["model"]

    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if "patch_embed" in key:
            key = key.replace("patch_embed", "tinyvit.embeddings")
        if "layers" in key:
            key = key.replace("layers", "tinyvit.encoder.stages")
        if "blocks" in key:
            key = key.replace("blocks", "layers")
        if "norm_head" in key:
            key = key.replace("norm_head", "layernorm")
        if "head" in key:
            key = key.replace("head", "classifier")
        if "bn" in key:
            key = key.replace("bn", "batch_norm")
        if "c.weight" in key:
            key = key.replace("c.weight", "convolution.weight")
        state_dict[key] = val

    model.load_state_dict(state_dict)
    model.eval()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    transforms = Compose(
        [
            Resize(size=256, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size=224),
            ToTensor(),
            Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )

    original_pixel_values = transforms(image).unsqueeze(0)

    # assert values
    image_processor = BitImageProcessor(
        do_resize=True,
        size={"shortest_edge": 256},
        do_center_crop=True,
        crop_size={"height": 224, "width": 224},
        do_normalize=True,
        do_rescale=True,
        image_mean=IMAGENET_DEFAULT_MEAN,
        image_std=IMAGENET_DEFAULT_STD,
    )
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    assert torch.allclose(pixel_values, original_pixel_values)

    with torch.no_grad():
        logits = model(pixel_values).logits

    print("First values of logits:", logits[0, :3])
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

    expected_slice = torch.tensor([-0.4772, 0.1134, -1.1261])
    assert torch.allclose(logits[0, :3], expected_slice, atol=1e-4)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model and processor to the 🤗 hub")
        model.push_to_hub(f"nielsr/{model_name}")
        image_processor.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="tinyvit-21m-22kto1k-distill",
        type=str,
        choices=name_to_checkpoint_url.keys(),
        help="Name of the model.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_tinyvit_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
