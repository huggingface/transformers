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
import os
import tempfile
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import HfApi
from PIL import Image

from transformers import VJEPA2Config, VJEPA2Model, VJEPA2VideoProcessor
from transformers.models.vjepa2.modeling_vjepa2 import apply_masks


HUB_REPO = "https://github.com/facebookresearch/vjepa2"
HUB_SOURCE = "github"

HUB_MODELS = {
    "vit_large": "facebook/vjepa2-vitl-fpc64-256",
    "vit_huge": "facebook/vjepa2-vith-fpc64-256",
    "vit_giant": "facebook/vjepa2-vitg-fpc64-256",
    "vit_giant_384": "facebook/vjepa2-vitg-fpc64-384",
}

S3_MODELS = {
    "vit_large": "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt",
    "vit_huge": "https://dl.fbaipublicfiles.com/vjepa2/vith.pt",
    "vit_giant": "https://dl.fbaipublicfiles.com/vjepa2/vitg.pt",
    "vit_giant_384": "https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt",
}

TOKEN = os.environ.get("HF_TOKEN", None)


def get_vjepa2_config(model_name):
    # size of the architecture
    if model_name == "vit_large":
        return VJEPA2Config(
            crop_size=256,
            frames_per_clip=64,
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=24,
            mlp_ratio=4,
            pred_hidden_size=384,
            pred_num_attention_heads=12,
            pred_num_hidden_layers=12,
            pred_num_mask_tokens=10,
        )
    elif model_name == "vit_huge":
        return VJEPA2Config(
            crop_size=256,
            frames_per_clip=64,
            hidden_size=1280,
            num_attention_heads=16,
            num_hidden_layers=32,
            mlp_ratio=4,
            pred_hidden_size=384,
            pred_num_attention_heads=12,
            pred_num_hidden_layers=12,
            pred_num_mask_tokens=10,
        )
    elif model_name == "vit_giant":
        return VJEPA2Config(
            crop_size=256,
            frames_per_clip=64,
            hidden_size=1408,
            num_attention_heads=22,
            num_hidden_layers=40,
            mlp_ratio=48 / 11,
            pred_hidden_size=384,
            pred_num_attention_heads=12,
            pred_num_hidden_layers=12,
            pred_num_mask_tokens=10,
        )
    elif model_name == "vit_giant_384":
        return VJEPA2Config(
            crop_size=384,
            frames_per_clip=64,
            hidden_size=1408,
            num_attention_heads=22,
            num_hidden_layers=40,
            mlp_ratio=48 / 11,
            pred_hidden_size=384,
            pred_num_attention_heads=12,
            pred_num_hidden_layers=12,
            pred_num_mask_tokens=10,
        )
    else:
        raise ValueError("Model not supported")


def convert_encoder_keys(model_state_dict, og_encoder_state_dict, config):
    emb_dim = config.hidden_size
    for key, val in og_encoder_state_dict.copy().items():
        val = og_encoder_state_dict.pop(key)
        key = key.replace("module.backbone.", "")
        if key.startswith("blocks."):
            key = key.replace("blocks.", "encoder.layer.")
        if "attn." in key:
            key = key.replace("attn.", "attention.")
        if key == "pos_embed":
            key = "encoder.embeddings.position_embeddings"
        if "patch_embed." in key:
            key = key.replace("patch_embed.", "encoder.embeddings.patch_embeddings.")
        if key.startswith("norm."):
            key = key.replace("norm.", "encoder.layernorm.")
        if "qkv." in key:
            prefix, suffix = key.split("qkv")
            if "bias" in suffix:
                q_e, k_e, v_e = (
                    val[0:emb_dim],
                    val[emb_dim : emb_dim * 2],
                    val[emb_dim * 2 :],
                )
            else:
                q_e, k_e, v_e = (
                    val[0:emb_dim, :],
                    val[emb_dim : emb_dim * 2, :],
                    val[emb_dim * 2 :, :],
                )
            og_encoder_state_dict[prefix + "query" + suffix] = q_e
            og_encoder_state_dict[prefix + "key" + suffix] = k_e
            og_encoder_state_dict[prefix + "value" + suffix] = v_e
        else:
            og_encoder_state_dict[key] = val
    return og_encoder_state_dict


def convert_predictor_keys(model_state_dict, og_predictor_state_dict, config):
    emb_dim = config.pred_hidden_size
    if "predictor_pos_embed" in og_predictor_state_dict:
        del og_predictor_state_dict["predictor_pos_embed"]
    # update predictor weights
    mask_tokens = {}
    mask_token_keys_to_delete = []
    for key, val in og_predictor_state_dict.copy().items():
        val = og_predictor_state_dict.pop(key)
        key = key.replace("module.backbone.", "")
        if key.startswith("predictor_blocks."):
            key = key.replace("predictor_blocks.", "predictor.layer.")
        if "attn." in key:
            key = key.replace("attn.", "attention.")
        if key == "predictor_pos_embed":
            key = "predictor.embeddings.position_embeddings"
        if "predictor_embed." in key:
            key = key.replace("predictor_embed.", "predictor.embeddings.predictor_embeddings.")
        if "mask_tokens." in key:
            mask_tokens[key.split("mask_tokens.")[-1]] = val
            mask_token_keys_to_delete.append(key)
            # key = key.replace("mask_tokens.", "predictor.embeddings.mask_tokens.")
        if key.startswith("predictor_norm."):
            key = key.replace("predictor_norm.", "predictor.layernorm.")
        if key.startswith("predictor_proj."):
            key = key.replace("predictor_proj.", "predictor.proj.")
        if "qkv." in key:
            prefix, suffix = key.split("qkv")
            if "bias" in suffix:
                q_e, k_e, v_e = (
                    val[0:emb_dim],
                    val[emb_dim : emb_dim * 2],
                    val[emb_dim * 2 :],
                )
            else:
                q_e, k_e, v_e = (
                    val[0:emb_dim, :],
                    val[emb_dim : emb_dim * 2, :],
                    val[emb_dim * 2 :, :],
                )
            og_predictor_state_dict[prefix + "query" + suffix] = q_e
            og_predictor_state_dict[prefix + "key" + suffix] = k_e
            og_predictor_state_dict[prefix + "value" + suffix] = v_e
        else:
            og_predictor_state_dict[key] = val
    mask_tokens = torch.stack([mask_tokens[f"{i}"] for i in range(len(mask_tokens))], dim=0)
    for k in mask_token_keys_to_delete:
        del og_predictor_state_dict[k]
    og_predictor_state_dict["predictor.embeddings.mask_tokens"] = mask_tokens
    return og_predictor_state_dict


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


def upload_original_ckpts(model_name):
    hf_repo = HUB_MODELS[model_name]
    original_ckpt = S3_MODELS[model_name]
    print(f"Uploading original checkpoint for vjepa2 {model_name} to {hf_repo}/original/")
    with tempfile.NamedTemporaryFile() as fn:
        local_path = fn.name
        torch.hub.download_url_to_file(original_ckpt, local_path)
        api = HfApi()
        api.upload_file(
            repo_id=hf_repo,
            path_or_fileobj=local_path,
            path_in_repo="original/model.pth",
            repo_type="model",
            token=TOKEN,
        )
        print("Uploading complete")


@torch.no_grad()
def convert_and_test_vjepa2_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our VJEPA2 structure.
    """
    config = get_vjepa2_config(model_name)

    # load original model from torch hub
    original_encoder, original_predictor = torch.hub.load(HUB_REPO, "vjepa2_" + model_name, source=HUB_SOURCE)
    original_encoder.eval()
    original_predictor.eval()
    original_preprocessor = torch.hub.load(
        HUB_REPO, "vjepa2_preprocessor", source=HUB_SOURCE, crop_size=config.crop_size
    )

    # load state_dict of original model, remove and rename some keys
    encoder_state_dict = original_encoder.state_dict()
    decoder_state_dict = original_predictor.state_dict()

    model = VJEPA2Model(config).eval()
    state_dict = model.state_dict()

    og_encoder_sd = convert_encoder_keys(state_dict, encoder_state_dict, config)
    og_predictor_sd = convert_predictor_keys(state_dict, decoder_state_dict, config)

    og_state_dict = og_encoder_sd
    og_state_dict.update(og_predictor_sd)
    model.load_state_dict(og_state_dict)

    # load image
    image = prepare_img()
    image = torch.Tensor(np.array(image)).unsqueeze(0).permute(0, 3, 1, 2)
    print("Input shape: ", image.shape)

    crop_size = config.crop_size
    processor = VJEPA2VideoProcessor(crop_size=crop_size)
    pr_out = processor(image, return_tensors="pt")
    pixel_values_videos = pr_out.pixel_values_videos
    # run original preprocessor
    original_pixel_values = original_preprocessor(image)
    assert original_pixel_values[0].permute(1, 0, 2, 3).shape == pixel_values_videos[0].shape
    assert torch.allclose(original_pixel_values[0].permute(1, 0, 2, 3), pixel_values_videos[0], atol=1e-3)

    with torch.no_grad():
        # reshape and move to gpu
        if pixel_values_videos.size(1) == 1:
            pixel_values_videos = pixel_values_videos.repeat(1, config.frames_per_clip, 1, 1, 1)
        # pixel_values_videos = pixel_values_videos.permute(0, 2, 1, 3, 4)  # B x C x T x H x W
        pixel_values_videos = pixel_values_videos.to(device="cuda", dtype=torch.float32)
        original_encoder = original_encoder.to(device="cuda", dtype=torch.float32)
        original_predictor = original_predictor.to(device="cuda", dtype=torch.float32)
        model = model.to(device="cuda", dtype=torch.float32)
        # forward
        original_encoder_outputs = original_encoder(pixel_values_videos.permute(0, 2, 1, 3, 4))
        B, N, _ = original_encoder_outputs.shape
        # test full mask
        context_mask = [torch.arange(N, device=pixel_values_videos.device).unsqueeze(0).repeat((B, 1))]
        predictor_mask = context_mask
        original_predictor_outputs = original_predictor(original_encoder_outputs, context_mask, predictor_mask)
        outputs = model(pixel_values_videos, context_mask=context_mask, target_mask=predictor_mask)
        assert torch.allclose(outputs.last_hidden_state, original_encoder_outputs, atol=1e-3)
        predictor_outputs = outputs.predictor_output
        assert torch.allclose(predictor_outputs.last_hidden_state, original_predictor_outputs, atol=1e-3)
        # test partial mask
        window_size = 256
        mask = torch.arange(N, device=pixel_values_videos.device).unsqueeze(0)
        context_mask = [mask[:, :window_size].repeat((B, 1))]
        predictor_mask = [mask[:, window_size : window_size * 2].repeat((B, 1))]
        original_predictor_outputs = original_predictor(
            apply_masks(original_encoder_outputs, context_mask),
            context_mask,
            predictor_mask,
        )
        outputs = model(pixel_values_videos, context_mask=context_mask, target_mask=predictor_mask)
        assert torch.allclose(outputs.last_hidden_state, original_encoder_outputs, atol=1e-3)
        predictor_outputs = outputs.predictor_output
        assert torch.allclose(predictor_outputs.last_hidden_state, original_predictor_outputs, atol=1e-3)

    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        name = HUB_MODELS[model_name]
        model.push_to_hub(name, private=True)
        processor.push_to_hub(name, private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="vit_large",
        type=str,
        choices=[
            "vit_large",
            "vit_huge",
            "vit_giant",
            "vit_giant_384",
        ],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the Hugging Face hub.",
    )
    parser.add_argument("--upload_original", action="store_true", help="upload the original checkpoint")

    args = parser.parse_args()
    convert_and_test_vjepa2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
    if args.upload_original:
        upload_original_ckpts(args.model_name)
