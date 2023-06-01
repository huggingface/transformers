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


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []
    rename_keys.append(("pos_emb", "transformer.embeddings.position_embedding"))
    rename_keys.append(("tok_emb.weight", "transformer.embeddings.token_embedding.weight"))
    # NOTE: masks token does not exist in the original weights

    for i in range(config.num_hidden_layers):
        rename_keys.append((f"blocks.{i}.ln1.weight", f"transformer.encoder.layers.{i}.ln_1.weight"))
        rename_keys.append((f"blocks.{i}.ln1.bias", f"transformer.encoder.layers.{i}.ln_1.bias"))
        rename_keys.append((f"blocks.{i}.ln2.weight", f"transformer.encoder.layers.{i}.ln_2.weight"))
        rename_keys.append((f"blocks.{i}.ln2.bias", f"transformer.encoder.layers.{i}.ln_2.bias"))
        rename_keys.append((f"blocks.{i}.attn.key.weight", f"transformer.encoder.layers.{i}.attention.key.weight"))
        rename_keys.append((f"blocks.{i}.attn.key.bias", f"transformer.encoder.layers.{i}.attention.key.bias"))
        rename_keys.append((f"blocks.{i}.attn.query.weight", f"transformer.encoder.layers.{i}.attention.query.weight"))
        rename_keys.append((f"blocks.{i}.attn.query.bias", f"transformer.encoder.layers.{i}.attention.query.bias"))
        rename_keys.append((f"blocks.{i}.attn.value.weight", f"transformer.encoder.layers.{i}.attention.value.weight"))
        rename_keys.append((f"blocks.{i}.attn.value.bias", f"transformer.encoder.layers.{i}.attention.value.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"transformer.encoder.layers.{i}.attention.output.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"transformer.encoder.layers.{i}.attention.output.bias"))
        rename_keys.append((f"blocks.{i}.mlp.0.weight", f"transformer.encoder.layers.{i}.mlp.0.weight"))
        rename_keys.append((f"blocks.{i}.mlp.0.bias", f"transformer.encoder.layers.{i}.mlp.0.bias"))
        rename_keys.append((f"blocks.{i}.mlp.2.weight", f"transformer.encoder.layers.{i}.mlp.2.weight"))
        rename_keys.append((f"blocks.{i}.mlp.2.bias", f"transformer.encoder.layers.{i}.mlp.2.bias"))

    # Generator
    rename_keys.append(("module.encoder.1.weight", "guided_upsampler.generator.encoder.1.weight"))
    rename_keys.append(("module.encoder.1.bias", "guided_upsampler.generator.encoder.1.bias"))
    rename_keys.append(("module.encoder.3.weight", "guided_upsampler.generator.encoder.3.weight"))
    rename_keys.append(("module.encoder.3.bias", "guided_upsampler.generator.encoder.3.bias"))
    rename_keys.append(("module.encoder.5.weight", "guided_upsampler.generator.encoder.5.weight"))
    rename_keys.append(("module.encoder.5.bias", "guided_upsampler.generator.encoder.5.bias"))

    for i in range(config.num_residual_blocks):
        rename_keys.append(
            (f"module.middle.{i}.conv_block.1.weight", f"guided_upsampler.generator.middle.{i}.conv_block.1.weight")
        )
        rename_keys.append(
            (f"module.middle.{i}.conv_block.1.bias", f"guided_upsampler.generator.middle.{i}.conv_block.1.bias")
        )
        rename_keys.append(
            (f"module.middle.{i}.conv_block.4.weight", f"guided_upsampler.generator.middle.{i}.conv_block.4.weight")
        )
        rename_keys.append(
            (f"module.middle.{i}.conv_block.4.bias", f"guided_upsampler.generator.middle.{i}.conv_block.4.bias")
        )

    rename_keys.append(("module.decoder.0.weight", "guided_upsampler.generator.decoder.0.weight"))
    rename_keys.append(("module.decoder.0.bias", "guided_upsampler.generator.decoder.0.bias"))
    rename_keys.append(("module.decoder.2.weight", "guided_upsampler.generator.decoder.2.weight"))
    rename_keys.append(("module.decoder.2.bias", "guided_upsampler.generator.decoder.2.bias"))
    rename_keys.append(("module.decoder.5.weight", "guided_upsampler.generator.decoder.5.weight"))
    rename_keys.append(("module.decoder.5.bias", "guided_upsampler.generator.decoder.5.bias"))

    rename_keys.append(("ln_f.weight", "transformer.layernorm.weight"))
    rename_keys.append(("ln_f.bias", "transformer.layernorm.bias"))
    rename_keys.append(("head.weight", "transformer.head.weight"))

    return rename_keys


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
    config = IctConfig.from_json_file(ict_config_file)
    model = IctModel(config)
    model_name = checkpoint_path.split("/")[-1].split(".")[0]

    model_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    generator_local_path = hf_hub_download(repo_id="sheonhan/ict-imagenet-256", filename="generator.pt")
    generator_state_dict = torch.load(generator_local_path, map_location="cpu")

    model_state_dict.update(generator_state_dict)
    model_state_dict = {key: value for key, value in model_state_dict.items() if "attn.mask" not in key}

    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        val = model_state_dict.pop(src)
        model_state_dict[dest] = val

    model.load_state_dict(model_state_dict, strict=False)
    model.eval()

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

    bool_masked_pos_local_path = hf_hub_download(repo_id="sheonhan/ict-imagenet-256", filename="my_bool_masked_pos.pt")
    bool_masked_pos = torch.load(bool_masked_pos_local_path)
    bool_masked_pos = bool_masked_pos.unsqueeze(0)

    outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos, clusters=clusters)
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
