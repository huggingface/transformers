# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
r"""
Convert MIT EfficientViT-SAM checkpoints into Hugging Face `EfficientvitsamModel` format.

Example:

```bash
python -m transformers.models.efficientvitsam.convert_efficientvitsam_to_hf \
  --model_name efficientvit-sam-l0 \
  --pytorch_dump_folder_path ./efficientvitsam-l0
```
"""

import argparse
import re
from io import BytesIO

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    EfficientvitsamConfig,
    EfficientvitsamImageProcessor,
    EfficientvitsamModel,
    EfficientvitsamProcessor,
)


def get_config(model_name: str) -> EfficientvitsamConfig:
    variant = model_name.replace("efficientvit-sam-", "")
    return EfficientvitsamConfig(vision_config={"variant": variant})


KEYS_TO_MODIFY_MAPPING = {
    "image_encoder.": "vision_encoder.",
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_decoder.transformer.norm_final_attn": "mask_decoder.transformer.layer_norm_final_attn",
    "mask_decoder.transformer.layers.0.norm1": "mask_decoder.transformer.layers.0.layer_norm1",
    "mask_decoder.transformer.layers.0.norm2": "mask_decoder.transformer.layers.0.layer_norm2",
    "mask_decoder.transformer.layers.0.norm3": "mask_decoder.transformer.layers.0.layer_norm3",
    "mask_decoder.transformer.layers.0.norm4": "mask_decoder.transformer.layers.0.layer_norm4",
    "mask_decoder.transformer.layers.1.norm1": "mask_decoder.transformer.layers.1.layer_norm1",
    "mask_decoder.transformer.layers.1.norm2": "mask_decoder.transformer.layers.1.layer_norm2",
    "mask_decoder.transformer.layers.1.norm3": "mask_decoder.transformer.layers.1.layer_norm3",
    "mask_decoder.transformer.layers.1.norm4": "mask_decoder.transformer.layers.1.layer_norm4",
    "prompt_encoder.mask_downscaling.0": "prompt_encoder.mask_embed.conv1",
    "prompt_encoder.mask_downscaling.1": "prompt_encoder.mask_embed.layer_norm1",
    "prompt_encoder.mask_downscaling.3": "prompt_encoder.mask_embed.conv2",
    "prompt_encoder.mask_downscaling.4": "prompt_encoder.mask_embed.layer_norm2",
    "prompt_encoder.mask_downscaling.6": "prompt_encoder.mask_embed.conv3",
    "prompt_encoder.point_embeddings": "prompt_encoder.point_embed",
    "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix": "prompt_encoder.shared_embedding.positional_embedding",
}


def replace_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    model_state_dict = {}
    output_hypernetworks_mlps_pattern = r".*.output_hypernetworks_mlps.(\d+).layers.(\d+).*"

    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        match = re.match(output_hypernetworks_mlps_pattern, key)
        if match:
            layer_nb = int(match.group(2))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

        model_state_dict[key] = value

    model_state_dict["shared_image_embedding.positional_embedding"] = model_state_dict[
        "prompt_encoder.shared_embedding.positional_embedding"
    ]
    return model_state_dict


def validate_model(model: EfficientvitsamModel, processor: EfficientvitsamProcessor, device: str) -> None:
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(BytesIO(requests.get(url, timeout=60).content)).convert("RGB")
    inputs = processor(images=image, input_points=[[[400, 400]]], input_labels=[[1]], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    if outputs.pred_masks is None or outputs.iou_scores is None:
        raise ValueError("Converted EfficientViT-SAM model produced empty outputs.")


def convert_efficientvitsam_checkpoint(
    model_name: str,
    checkpoint_path: str,
    pytorch_dump_folder: str | None,
    push_to_hub: bool,
    skip_validation: bool,
    hub_repo_id: str | None = None,
) -> None:
    config = get_config(model_name)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = replace_keys(state_dict)

    image_processor = EfficientvitsamImageProcessor(
        size={"longest_edge": config.vision_config.image_size},
        pad_size={"height": config.vision_config.image_size, "width": config.vision_config.image_size},
        prompt_size={"longest_edge": config.vision_config.prompt_image_size},
    )
    processor = EfficientvitsamProcessor(image_processor=image_processor)
    hf_model = EfficientvitsamModel(config)
    missing_keys, unexpected_keys = hf_model.load_state_dict(state_dict, strict=False)

    ignored_missing = {"shared_image_embedding.positional_embedding"}
    actual_missing = [key for key in missing_keys if key not in ignored_missing]
    if actual_missing or unexpected_keys:
        raise ValueError(f"State dict mismatch. Missing: {actual_missing}. Unexpected: {unexpected_keys}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hf_model = hf_model.to(device).eval()

    if not skip_validation:
        validate_model(hf_model, processor, device)

    if pytorch_dump_folder is not None:
        processor.save_pretrained(pytorch_dump_folder)
        hf_model.save_pretrained(pytorch_dump_folder)

    if push_to_hub:
        if not hub_repo_id:
            raise ValueError("When using --push_to_hub, pass --hub_repo_id (for example: YOUR_ORG/model-name).")
        processor.push_to_hub(hub_repo_id)
        hf_model.push_to_hub(hub_repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = [
        "efficientvit-sam-l0",
        "efficientvit-sam-l1",
        "efficientvit-sam-l2",
        "efficientvit-sam-xl0",
        "efficientvit-sam-xl1",
    ]
    parser.add_argument("--model_name", default="efficientvit-sam-l0", choices=choices, type=str)
    parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to the original checkpoint.")
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether to push the model and processor to the hub."
    )
    parser.add_argument("--skip_validation", action="store_true", help="Skip the validation forward pass.")
    parser.add_argument("--hub_repo_id", type=str, default=None, help="Target Hub repo id when --push_to_hub is set.")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        filename = f"{args.model_name.replace('-', '_')}.pt"
        checkpoint_path = hf_hub_download("mit-han-lab/efficientvit-sam", filename)

    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--hub_repo_id is required when --push_to_hub is set.")

    convert_efficientvitsam_checkpoint(
        args.model_name,
        checkpoint_path,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
        args.skip_validation,
        hub_repo_id=args.hub_repo_id,
    )
'''
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
r"""
Convert checkpoints into Hugging Face `EfficientvitsamModel` + processor format.

## Supported checkpoints (SAM ViT layout)

The current [`EfficientvitsamModel`](https://huggingface.co/docs/transformers/main/en/model_doc/efficientvitsam)
follows the **Segment Anything** ViT image encoder + prompt encoder + mask decoder (same as [`SamModel`]).
Therefore you can convert **Meta SAM** and **SlimSAM** `.pth` checkpoints using the same key renames as
[`convert_sam_to_hf.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/convert_sam_to_hf.py).

Example (download Meta `sam_vit_b` checkpoint yourself or use `checkpoint_path` from Hub):

```bash
python -m transformers.models.efficientvitsam.convert_efficientvitsam_to_hf \
  --model_name sam_vit_b_01ec64 \
  --checkpoint_path path/to/sam_vit_b_01ec64.pth \
  --pytorch_dump_folder_path ./efficientvitsam-sam-vit-b
```

## MIT EfficientViT-SAM `.pt` checkpoints (official repo)

Official [EfficientViT](https://github.com/mit-han-lab/efficientvit) **EfficientViT-SAM** weights use an **EfficientViT backbone**
plus `segment_anything` prompt/mask heads. That layout is **not** identical to the ViT-based [`EfficientvitsamModel`]
in this library (which subclasses SAM). Converting those `.pt` files requires a **dedicated** key map and a
model that matches `EfficientViTSam` in the upstream repo; that is **not** implemented here yet.

## Upload to the Hugging Face Hub

1. Log in: `huggingface-cli login` (or set `HF_TOKEN`).
2. Convert and push in one step (set your org or username):

```bash
python -m transformers.models.efficientvitsam.convert_efficientvitsam_to_hf \
  --model_name sam_vit_b_01ec64 \
  --pytorch_dump_folder_path ./out_efficientvitsam \
  --push_to_hub \
  --hub_repo_id YOUR_ORG/efficientvitsam-sam-vit-b
```

`--hub_repo_id` is required when `--push_to_hub` is used. You can omit `--pytorch_dump_folder_path` to push from memory only (still uploads the same files).
"""

import argparse
import re
from io import BytesIO

import httpx
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    EfficientvitsamConfig,
    EfficientvitsamImageProcessor,
    EfficientvitsamModel,
    EfficientvitsamProcessor,
    EfficientvitsamVisionConfig,
)


def get_config(model_name: str) -> EfficientvitsamConfig:
    """Vision settings aligned with `convert_sam_to_hf.get_config` / Meta SAM variants."""
    if "slimsam-50" in model_name:
        vision_config = EfficientvitsamVisionConfig(
            hidden_size=384,
            mlp_dim=1536,
            num_hidden_layers=12,
            num_attention_heads=12,
            global_attn_indexes=[2, 5, 8, 11],
        )
    elif "slimsam-77" in model_name:
        vision_config = EfficientvitsamVisionConfig(
            hidden_size=168,
            mlp_dim=696,
            num_hidden_layers=12,
            num_attention_heads=12,
            global_attn_indexes=[2, 5, 8, 11],
        )
    elif "sam_vit_b" in model_name:
        vision_config = EfficientvitsamVisionConfig()
    elif "sam_vit_l" in model_name:
        vision_config = EfficientvitsamVisionConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            global_attn_indexes=[5, 11, 17, 23],
        )
    elif "sam_vit_h" in model_name:
        vision_config = EfficientvitsamVisionConfig(
            hidden_size=1280,
            num_hidden_layers=32,
            num_attention_heads=16,
            global_attn_indexes=[7, 15, 23, 31],
        )
    else:
        raise ValueError(f"Unknown model_name for config: {model_name}")

    return EfficientvitsamConfig(vision_config=vision_config)


KEYS_TO_MODIFY_MAPPING = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
    "point_embeddings": "point_embed",
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
    "image_encoder": "vision_encoder",
    "neck.0": "neck.conv1",
    "neck.1": "neck.layer_norm1",
    "neck.2": "neck.conv2",
    "neck.3": "neck.layer_norm2",
    "patch_embed.proj": "patch_embed.projection",
    ".norm": ".layer_norm",
    "blocks": "layers",
}


def replace_keys(state_dict: dict) -> dict:
    """Map original Meta SAM checkpoint keys to Hugging Face `EfficientvitsamModel` state dict keys."""
    model_state_dict = {}
    state_dict.pop("pixel_mean", None)
    state_dict.pop("pixel_std", None)

    output_hypernetworks_mlps_pattern = r".*.output_hypernetworks_mlps.(\\d+).layers.(\\d+).*"

    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        if re.match(output_hypernetworks_mlps_pattern, key):
            layer_nb = int(re.match(output_hypernetworks_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

        model_state_dict[key] = value

    model_state_dict["shared_image_embedding.positional_embedding"] = model_state_dict[
        "prompt_encoder.shared_embedding.positional_embedding"
    ]

    return model_state_dict


def convert_efficientvitsam_checkpoint(
    model_name: str,
    checkpoint_path: str,
    pytorch_dump_folder: str | None,
    push_to_hub: bool,
    skip_validation: bool,
    hub_repo_id: str | None = None,
) -> None:
    config = get_config(model_name)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = replace_keys(state_dict)

    image_processor = EfficientvitsamImageProcessor()
    processor = EfficientvitsamProcessor(image_processor=image_processor)
    hf_model = EfficientvitsamModel(config)
    hf_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_model.load_state_dict(state_dict)
    hf_model = hf_model.to(device)

    if not skip_validation:
        url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
        with httpx.stream("GET", url) as response:
            raw_image = Image.open(BytesIO(response.read())).convert("RGB")

        input_points = [[[500, 375]]]
        input_labels = [[1]]

        inputs = processor(images=np.array(raw_image), return_tensors="pt").to(device)

        with torch.no_grad():
            output = hf_model(**inputs)
        scores = output.iou_scores.squeeze()

        if model_name == "sam_vit_b_01ec64":
            inputs = processor(
                images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                output = hf_model(**inputs)
                scores = output.iou_scores.squeeze()

        elif model_name == "sam_vit_h_4b8939":
            inputs = processor(
                images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                output = hf_model(**inputs)
                scores = output.iou_scores.squeeze()

            assert scores[-1].item() == 0.9712603092193604

            input_boxes = ((75, 275, 1725, 850),)

            inputs = processor(images=np.array(raw_image), input_boxes=input_boxes, return_tensors="pt").to(device)

            with torch.no_grad():
                output = hf_model(**inputs)
            scores = output.iou_scores.squeeze()

            assert scores[-1].item() == 0.8686015605926514

            input_points = [[[400, 650], [800, 650]]]
            input_labels = [[1, 1]]

            inputs = processor(
                images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                output = hf_model(**inputs)
            scores = output.iou_scores.squeeze()

            assert scores[-1].item() == 0.9936047792434692

    if pytorch_dump_folder is not None:
        processor.save_pretrained(pytorch_dump_folder)
        hf_model.save_pretrained(pytorch_dump_folder)

    if push_to_hub:
        if not hub_repo_id:
            raise ValueError("When using --push_to_hub, pass --hub_repo_id (e.g. YOUR_ORG/your-model-name).")
        processor.push_to_hub(hub_repo_id)
        hf_model.push_to_hub(hub_repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = ["sam_vit_b_01ec64", "sam_vit_h_4b8939", "sam_vit_l_0b3195", "slimsam-50-uniform", "slimsam-77-uniform"]
    parser.add_argument(
        "--model_name",
        default="sam_vit_h_4b8939",
        choices=choices,
        type=str,
        help="Name of the original SAM / SlimSAM checkpoint to convert (same as convert_sam_to_hf).",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to the original checkpoint",
    )
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip forward-pass sanity checks (useful for custom checkpoints or offline runs).",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="Target Hub repo id (e.g. org/model-name). Required when --push_to_hub is set.",
    )

    args = parser.parse_args()

    if "slimsam" in args.model_name:
        checkpoint_path = args.checkpoint_path
        if checkpoint_path is None:
            raise ValueError("You need to provide a checkpoint path for SlimSAM models.")
    else:
        checkpoint_path = args.checkpoint_path
        if checkpoint_path is None:
            checkpoint_path = hf_hub_download("ybelkada/segment-anything", f"checkpoints/{args.model_name}.pth")

    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--hub_repo_id is required when --push_to_hub is set.")

    convert_efficientvitsam_checkpoint(
        args.model_name,
        checkpoint_path,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
        args.skip_validation,
        hub_repo_id=args.hub_repo_id,
    )
'''
