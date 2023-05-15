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
"""Convert ICT checkpoints from the original library."""


import argparse
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision.transforms import Compose, Lambda, Resize

from transformers import IctConfig, IctImageProcessor, IctModel
from transformers.image_utils import PILImageResampling
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# rename parameter names
def rename_key(name):
    if "stem.conv" in name:
        name = name.replace("stem.conv", "bit.embedder.convolution")
    if "blocks" in name:
        name = name.replace("blocks", "layers")
    if "head.fc" in name:
        name = name.replace("head.fc", "classifier.1")
    if name.startswith("norm"):
        name = "bit." + name
    if "bit" not in name and "classifier" not in name:
        name = "bit.encoder." + name

    return name


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_ict_checkpoint(
    checkpoint_path: Path,
    ict_config_file: Path,
    ict_image_processor_config_file: Path,
    pytorch_dump_path: Path,
    push_to_hub: bool,
):
    torch.load(checkpoint_path, map_location="cpu")["model"]
    config = IctConfig.from_json_file(ict_config_file)
    model = IctModel(config)
    model_name = checkpoint_path.split("/")[-1].split(".")[0]

    # print(orig_state_dict)
    # model.load_state_dict(orig_state_dict)
    # model.eval()

    # prepare image
    image = prepare_img()
    image_size = 32
    image_processor = IctImageProcessor.from_json_file(ict_image_processor_config_file)
    clusters = image_processor.clusters
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    # original processing pipeline
    image_transforms = Compose(
        [
            Resize((image_size, image_size), interpolation=PILImageResampling.BILINEAR),
            Lambda(lambda img: torch.from_numpy(np.array(img)).view(-1, 3)),
            Lambda(lambda img: ((img[:, None, :] - clusters[None, :, :]) ** 2).sum(-1).argmin(1)),
        ]
    )
    original_pixel_values = image_transforms(image).unsqueeze(0)

    assert torch.allclose(original_pixel_values, pixel_values)
    inputs = {}
    inputs["pixel_values"] = pixel_values

    # local_path = hf_hub_download(repo_id="hf-internal-testing/bool-masked-pos", filename="bool_masked_pos.pt")
    local_path = hf_hub_download(repo_id="sheonhan/ict-imagenet-256", filename="my_bool_masked_pos.pt")
    bool_masked_pos = torch.load(local_path)
    inputs["bool_masked_pos"] = bool_masked_pos

    outputs = model(**inputs)
    logits = outputs.logits

    expected_shape = (3, 256, 256)

    if "ImageNet" in model_name:
        expected_logits = torch.Tensor(
            [-0.1312, 0.4353, -1.0499, -0.5124, 0.4183, -0.6793, -1.3777, -0.0893, -0.7358, -2.4328]
        )
        assert torch.allclose(logits[0, :10], expected_logits, atol=1e-3)
        assert logits.shape == expected_shape
    # elif "FFHQ" in model_name:
    #     expected_logits = torch.Tensor(
    #         [-1.3150, -1.5456, -1.2556, -0.8496, -0.7127, -0.7897, -0.9728, -0.3052, 0.3751, -0.3127]
    #     )
    #     assert torch.allclose(logits[0, :10], expected_logits, atol=1e-3)
    #     assert logits.shape == expected_shape
    # elif "Places2_Nature" in model_name:
    #     expected_logits = torch.Tensor(
    #         [-1.0283, -1.4131, -0.5644, -1.3115, -0.5785, -1.2049, -0.7528, 0.1992, -0.3822, -0.0878]
    #     )
    #     assert logits.shape == expected_shape
    else:
        raise ValueError(
            f"Unknown model checkpoint: {checkpoint_path}. Supported version of efficientformer are l1, l3 and l7"
        )

    # Save Checkpoints
    Path(pytorch_dump_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_path)
    print(f"Checkpoint successfuly converted. Model saved at {pytorch_dump_path}")
    image_processor.save_pretrained(pytorch_dump_path)
    print(f"Image processor successfuly saved at {pytorch_dump_path}")

    if push_to_hub:
        print("Pushing model to the hub...")

        model.push_to_hub(
            repo_id=f"sheonhan/{pytorch_dump_path}",
            commit_message="Add model",
            use_temp_dir=True,
        )
        image_processor.push_to_hub(
            repo_id=f"sheonhan/{pytorch_dump_path}",
            commit_message="Add feature extractor",
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
        help="Path to ICT pytorch checkpoint.",
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The json file for ICT model config.",
    )
    parser.add_argument(
        "--image_processor_config_file",
        default=None,
        type=str,
        required=True,
        help="The json file for IctImageProcessor config.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        default="model",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image preprocessor to the hub")

    args = parser.parse_args()
    convert_ict_checkpoint(
        checkpoint_path=args.pytorch_model_path,
        ict_config_file=args.config_file,
        ict_image_processor_config_file=args.image_processor_config_file,
        pytorch_dump_path=args.pytorch_dump_path,
        push_to_hub=args.push_to_hub,
    )
