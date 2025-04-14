# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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

"""Convert EfficientFormer checkpoints from the original repository.

URL: https://github.com/snap-research/EfficientFormer
"""

import argparse
import re
from pathlib import Path

import requests
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from transformers import (
    EfficientFormerConfig,
    EfficientFormerForImageClassificationWithTeacher,
    EfficientFormerImageProcessor,
)
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling


def rename_key(old_name, num_meta4D_last_stage):
    new_name = old_name

    if "patch_embed" in old_name:
        _, layer, param = old_name.split(".")

        if layer == "0":
            new_name = old_name.replace("0", "convolution1")
        elif layer == "1":
            new_name = old_name.replace("1", "batchnorm_before")
        elif layer == "3":
            new_name = old_name.replace("3", "convolution2")
        else:
            new_name = old_name.replace("4", "batchnorm_after")

    if "network" in old_name and re.search(r"\d\.\d", old_name):
        two_digit_num = r"\b\d{2}\b"
        if bool(re.search(two_digit_num, old_name)):
            match = re.search(r"\d\.\d\d.", old_name).group()
        else:
            match = re.search(r"\d\.\d.", old_name).group()
        if int(match[0]) < 6:
            trimmed_name = old_name.replace(match, "")
            trimmed_name = trimmed_name.replace("network", match[0] + ".meta4D_layers.blocks." + match[2:-1])
            new_name = "intermediate_stages." + trimmed_name
        else:
            trimmed_name = old_name.replace(match, "")
            if int(match[2]) < num_meta4D_last_stage:
                trimmed_name = trimmed_name.replace("network", "meta4D_layers.blocks." + match[2])
            else:
                layer_index = str(int(match[2]) - num_meta4D_last_stage)
                trimmed_name = trimmed_name.replace("network", "meta3D_layers.blocks." + layer_index)
                if "norm1" in old_name:
                    trimmed_name = trimmed_name.replace("norm1", "layernorm1")
                elif "norm2" in old_name:
                    trimmed_name = trimmed_name.replace("norm2", "layernorm2")
                elif "fc1" in old_name:
                    trimmed_name = trimmed_name.replace("fc1", "linear_in")
                elif "fc2" in old_name:
                    trimmed_name = trimmed_name.replace("fc2", "linear_out")

            new_name = "last_stage." + trimmed_name

    elif "network" in old_name and re.search(r".\d.", old_name):
        new_name = old_name.replace("network", "intermediate_stages")

    if "fc" in new_name:
        new_name = new_name.replace("fc", "convolution")
    elif ("norm1" in new_name) and ("layernorm1" not in new_name):
        new_name = new_name.replace("norm1", "batchnorm_before")
    elif ("norm2" in new_name) and ("layernorm2" not in new_name):
        new_name = new_name.replace("norm2", "batchnorm_after")
    if "proj" in new_name:
        new_name = new_name.replace("proj", "projection")
    if "dist_head" in new_name:
        new_name = new_name.replace("dist_head", "distillation_classifier")
    elif "head" in new_name:
        new_name = new_name.replace("head", "classifier")
    elif "patch_embed" in new_name:
        new_name = "efficientformer." + new_name
    elif new_name == "norm.weight" or new_name == "norm.bias":
        new_name = new_name.replace("norm", "layernorm")
        new_name = "efficientformer." + new_name
    else:
        new_name = "efficientformer.encoder." + new_name

    return new_name


def convert_torch_checkpoint(checkpoint, num_meta4D_last_stage):
    for key in checkpoint.copy().keys():
        val = checkpoint.pop(key)
        checkpoint[rename_key(key, num_meta4D_last_stage)] = val

    return checkpoint


# We will verify our results on a COCO image
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    return image


def convert_efficientformer_checkpoint(
    checkpoint_path: Path, efficientformer_config_file: Path, pytorch_dump_path: Path, push_to_hub: bool
):
    orig_state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
    config = EfficientFormerConfig.from_json_file(efficientformer_config_file)
    model = EfficientFormerForImageClassificationWithTeacher(config)
    model_name = "_".join(checkpoint_path.split("/")[-1].split(".")[0].split("_")[:-1])

    num_meta4D_last_stage = config.depths[-1] - config.num_meta3d_blocks + 1
    new_state_dict = convert_torch_checkpoint(orig_state_dict, num_meta4D_last_stage)

    model.load_state_dict(new_state_dict)
    model.eval()

    pillow_resamplings = {
        "bilinear": PILImageResampling.BILINEAR,
        "bicubic": PILImageResampling.BICUBIC,
        "nearest": PILImageResampling.NEAREST,
    }

    # prepare image
    image = prepare_img()
    image_size = 256
    crop_size = 224
    processor = EfficientFormerImageProcessor(
        size={"shortest_edge": image_size},
        crop_size={"height": crop_size, "width": crop_size},
        resample=pillow_resamplings["bicubic"],
    )
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # original processing pipeline
    image_transforms = Compose(
        [
            Resize(image_size, interpolation=pillow_resamplings["bicubic"]),
            CenterCrop(crop_size),
            ToTensor(),
            Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    original_pixel_values = image_transforms(image).unsqueeze(0)

    assert torch.allclose(original_pixel_values, pixel_values)

    outputs = model(pixel_values)
    logits = outputs.logits

    expected_shape = (1, 1000)

    if "l1" in model_name:
        expected_logits = torch.Tensor(
            [-0.1312, 0.4353, -1.0499, -0.5124, 0.4183, -0.6793, -1.3777, -0.0893, -0.7358, -2.4328]
        )
        assert torch.allclose(logits[0, :10], expected_logits, atol=1e-3)
        assert logits.shape == expected_shape
    elif "l3" in model_name:
        expected_logits = torch.Tensor(
            [-1.3150, -1.5456, -1.2556, -0.8496, -0.7127, -0.7897, -0.9728, -0.3052, 0.3751, -0.3127]
        )
        assert torch.allclose(logits[0, :10], expected_logits, atol=1e-3)
        assert logits.shape == expected_shape
    elif "l7" in model_name:
        expected_logits = torch.Tensor(
            [-1.0283, -1.4131, -0.5644, -1.3115, -0.5785, -1.2049, -0.7528, 0.1992, -0.3822, -0.0878]
        )
        assert logits.shape == expected_shape
    else:
        raise ValueError(
            f"Unknown model checkpoint: {checkpoint_path}. Supported version of efficientformer are l1, l3 and l7"
        )

    # Save Checkpoints
    Path(pytorch_dump_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_path)
    print(f"Checkpoint successfuly converted. Model saved at {pytorch_dump_path}")
    processor.save_pretrained(pytorch_dump_path)
    print(f"Processor successfuly saved at {pytorch_dump_path}")

    if push_to_hub:
        print("Pushing model to the hub...")

        model.push_to_hub(
            repo_id=f"Bearnardd/{pytorch_dump_path}",
            commit_message="Add model",
            use_temp_dir=True,
        )
        processor.push_to_hub(
            repo_id=f"Bearnardd/{pytorch_dump_path}",
            commit_message="Add image processor",
            use_temp_dir=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pytorch_model_path",
        default=None,
        type=str,
        required=True,
        help="Path to EfficientFormer pytorch checkpoint.",
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The json file for EfficientFormer model config.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )

    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image processor to the hub")
    parser.add_argument(
        "--no-push_to_hub",
        dest="push_to_hub",
        action="store_false",
        help="Do not push model and image processor to the hub",
    )
    parser.set_defaults(push_to_hub=True)

    args = parser.parse_args()
    convert_efficientformer_checkpoint(
        checkpoint_path=args.pytorch_model_path,
        efficientformer_config_file=args.config_file,
        pytorch_dump_path=args.pytorch_dump_path,
        push_to_hub=args.push_to_hub,
    )
