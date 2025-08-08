"""Convert DINOv3 checkpoints from the original repository.

URL: https://github.com/facebookresearch/dinov3/tree/main
"""

import argparse
import os
import random
import re
from typing import Optional

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import DINOv3ViTConfig, DINOv3ViTImageProcessorFast, DINOv3ViTModel


HUB_MODELS = {
    "vits": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vitsplus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vitb": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "vithplus": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
    "vit7b": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
}

HUB_CHECKPOINTS = {
    "vits": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "vitsplus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    "vitb": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "vitl": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "vithplus": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    "vit7b": "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
}

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"cls_token":               r"embeddings.cls_token",
    r"mask_token":              r"embeddings.mask_token",
    r"storage_tokens":          r"embeddings.register_tokens",
    r"patch_embed.proj":        r"embeddings.patch_embeddings.projection",
    r"periods":                 r"inv_freq", 
    r"rope_embed":              r"rope_embeddings",
    r"blocks.(\d+).attn.proj":  r"layer.\1.attention.o_proj",
    r"blocks.(\d+).attn.":      r"layer.\1.attention.",
    r"blocks.(\d+).ls(\d+)":    r"layer.\1.layer_scale\2",
    r"blocks.(\d+).mlp":        r"layer.\1.mlp",
    r"blocks.(\d+).norm":       r"layer.\1.norm",
}
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys: Optional[dict] = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def split_qkv(state_dict: dict):
    keys = [x for x in state_dict.keys() if "qkv" in x]
    for key in keys:
        qkv = state_dict.pop(key)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        state_dict[key.replace("qkv", "q_proj")] = q
        state_dict[key.replace("qkv", "k_proj")] = k
        state_dict[key.replace("qkv", "v_proj")] = v
    return state_dict


def get_dinov3_config(model_name: str) -> DINOv3ViTConfig:
    # size of the architecture
    if model_name == "vits":
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            mask_k_bias=True,
            qkv_bias=True,
            proj_bias=True,
            num_register_tokens=4,
            layerscale_value=1.0,
            mlp_ratio=4,
            use_swiglu_ffn=False,
            layer_norm_eps=1e-5,
            pos_embed_rope_base=100,
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
        )
    elif model_name == "vitsplus":
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            mask_k_bias=True,
            qkv_bias=True,
            num_register_tokens=4,
            layerscale_value=1.0,
            mlp_ratio=6,
            use_swiglu_ffn=True,
            layer_norm_eps=1e-5,
            pos_embed_rope_base=100,
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
        )
    elif model_name == "vitb":
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            mask_k_bias=True,
            qkv_bias=True,
            proj_bias=True,
            num_register_tokens=4,
            layerscale_value=1.0,
            mlp_ratio=4,
            use_swiglu_ffn=False,
            layer_norm_eps=1e-5,
            pos_embed_rope_base=100,
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
        )
    elif model_name == "vitl":
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            mask_k_bias=True,
            qkv_bias=True,
            num_register_tokens=4,
            layerscale_value=1.0,
            mlp_ratio=4,
            use_swiglu_ffn=False,
            layer_norm_eps=1e-5,
            pos_embed_rope_base=100,
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
        )
    elif model_name == "vithplus":
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=1280,
            num_hidden_layers=32,
            num_attention_heads=20,
            mask_k_bias=True,
            qkv_bias=True,
            num_register_tokens=4,
            layerscale_value=1.0,
            mlp_ratio=6,
            use_swiglu_ffn=True,
            layer_norm_eps=1e-5,
            pos_embed_rope_base=100,
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
        )
    elif model_name == "vit7b":
        return DINOv3ViTConfig(
            patch_size=16,
            hidden_size=4096,
            num_hidden_layers=40,
            num_attention_heads=32,
            mask_k_bias=True,
            qkv_bias=False,
            num_register_tokens=4,
            layerscale_value=1.0,
            mlp_ratio=3,
            use_swiglu_ffn=True,
            layer_norm_eps=1e-5,
            pos_embed_rope_base=100,
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
        )
    else:
        raise ValueError("Model not supported")


# TODO: remove this function
# def convert_dinov3_vit_to_hf_vit(original_dinov3_state_dict, config: DINOv3ViTConfig):
#     embed_dim = config.hidden_size
#     hf_dinov3_state_dict = {}
#     for key in original_dinov3_state_dict.keys():
#         val = original_dinov3_state_dict[key]
#         if key == "cls_token":
#             key = "embeddings.cls_token"
#         elif key == "mask_token":
#             key = "embeddings.mask_token"
#         elif key == "storage_tokens":
#             key = "embeddings.register_tokens"
#         elif key.startswith("patch_embed.proj"):
#             key = key.replace("patch_embed.proj", "embeddings.patch_embeddings.proj")
#         elif key.startswith("rope_embed"):
#             key = key.replace("rope_embed", "rope_embeddings")
#         elif key.startswith("blocks"):
#             key = key.replace("blocks", "layer")
#         if "ls1." in key:
#             key = key.replace("ls1", "layer_scale1")
#         if "ls2." in key:
#             key = key.replace("ls2", "layer_scale2")
#         if "attn." in key:
#             key = key.replace("attn.", "attention.")
#         if "qkv." in key:
#             prefix, suffix = key.split("qkv")
#             if "bias_mask" in suffix:
#                 continue
#             elif "bias" in suffix:
#                 q_e, k_e, v_e = (
#                     val[0:embed_dim],
#                     val[embed_dim : embed_dim * 2],
#                     val[embed_dim * 2 :],
#                 )
#             else:
#                 q_e, k_e, v_e = (
#                     val[0:embed_dim, :],
#                     val[embed_dim : embed_dim * 2, :],
#                     val[embed_dim * 2 :, :],
#                 )
#             hf_dinov3_state_dict[prefix + "query" + suffix] = q_e
#             if not ("bias" in suffix and config.mask_k_bias):
#                 hf_dinov3_state_dict[prefix + "key" + suffix] = k_e
#             hf_dinov3_state_dict[prefix + "value" + suffix] = v_e
#         else:
#             hf_dinov3_state_dict[key] = val
#     return hf_dinov3_state_dict


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


def get_transform(resize_size: int = 224):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])


def get_image_processor(resize_size: int = 224):
    return DINOv3ViTImageProcessorFast(
        do_resize=True,
        size={"height": resize_size, "width": resize_size},
        resample=2,  # BILINEAR
    )


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


seed = 42  # any number
set_deterministic(seed=seed)


@torch.no_grad()
def convert_and_test_dinov3_checkpoint(args):
    expected_outputs = {
        "vits_cls": [
            0.4635618329048157,
            -0.41560935974121094,
            0.40823689103126526,
            -0.12661336362361908,
            -0.28663691878318787,
        ],
        "vits_patch": [
            -0.03875422105193138,
            -0.2508954405784607,
            -0.01639290526509285,
            -0.4554736316204071,
            0.5715821981430054,
        ],
        "vitsplus_cls": [
            -0.47134941816329956,
            -1.365778923034668,
            -0.3179832398891449,
            0.37721940875053406,
            -0.769085705280304,
        ],
        "vitsplus_patch": [
            0.14455188810825348,
            -0.3881174623966217,
            -0.39343395829200745,
            -0.1576954871416092,
            -0.6003801226615906,
        ],
        "vitb_cls": [
            1.0346431732177734,
            -0.18060928583145142,
            -0.3410182595252991,
            -0.0663769543170929,
            -0.011383970268070698,
        ],
        "vitb_patch": [
            -0.08252374082803726,
            -0.45627278089523315,
            -0.7280299663543701,
            -0.4306802451610565,
            -0.15288019180297852,
        ],
        "vitl_cls": [
            0.4845271110534668,
            -0.5822147130966187,
            0.4806361198425293,
            0.5920403599739075,
            0.9451664686203003,
        ],
        "vitl_patch": [
            -0.2113673835992813,
            -0.490863561630249,
            -0.2571314871311188,
            0.10176393389701843,
            0.1545112431049347,
        ],
        "vithplus_cls": [
            -0.0645759105682373,
            -0.14886680245399475,
            -0.6215243935585022,
            0.6348787546157837,
            0.1526956558227539,
        ],
        "vithplus_patch": [
            -0.09381738305091858,
            0.287407249212265,
            -0.05003691464662552,
            0.4280431866645813,
            0.09456184506416321,
        ],
        "vit7b_cls": [
            0.2754395306110382,
            -0.261353999376297,
            0.0677720308303833,
            0.049936190247535706,
            -0.15874707698822021,
        ],
        "vit7b_patch": [
            0.04444204643368721,
            -0.05254213139414787,
            0.07077747583389282,
            -0.0651116818189621,
            -0.026546532288193703,
        ],
    }
    model_name = args.model_name
    config = get_dinov3_config(model_name)
    print(config)

    model = DINOv3ViTModel(config).eval()
    state_dict_path = hf_hub_download(repo_id=HUB_MODELS[model_name], filename=HUB_CHECKPOINTS[model_name])
    original_state_dict = torch.load(state_dict_path)

    original_state_dict = split_qkv(original_state_dict)
    original_keys = list(original_state_dict.keys())
    new_keys = convert_old_keys_to_new_keys(original_keys)

    converted_state_dict = {}
    for key in original_keys:
        new_key = new_keys[key]
        weight_tensor = original_state_dict[key]

        if "bias_mask" in key or "attn.k_proj.bias" in key:
            continue
        if "embeddings.mask_token" in new_key:
            weight_tensor = weight_tensor.unsqueeze(1)
        if "inv_freq" in new_key:
            continue

        converted_state_dict[new_key] = weight_tensor

    model.load_state_dict(converted_state_dict, strict=True)
    model = model.eval()

    transform = get_transform()
    image_processor = get_image_processor()
    image = prepare_img()

    # check preprocessing
    original_pixel_values = transform(image).unsqueeze(0)  # add batch dimension
    inputs = image_processor(image, return_tensors="pt")

    torch.testing.assert_close(original_pixel_values, inputs["pixel_values"], atol=1e-6, rtol=1e-6)
    print("Preprocessing looks ok!")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float):
        model_output = model(**inputs)

    last_layer_class_token = model_output.pooler_output
    last_layer_patch_tokens = model_output.last_hidden_state[:, config.num_register_tokens + 1 :]

    actual_outputs = {}
    actual_outputs[f"{model_name}_cls"] = last_layer_class_token[0, :5].tolist()
    actual_outputs[f"{model_name}_patch"] = last_layer_patch_tokens[0, 0, :5].tolist()

    print("Actual:  ", actual_outputs[f"{model_name}_cls"])
    print("Expected:", expected_outputs[f"{model_name}_cls"])

    torch.testing.assert_close(
        torch.Tensor(actual_outputs[f"{model_name}_cls"]),
        torch.Tensor(expected_outputs[f"{model_name}_cls"]),
        atol=1e-3,
        rtol=1e-3,
    )
    torch.testing.assert_close(
        torch.Tensor(actual_outputs[f"{model_name}_patch"]),
        torch.Tensor(expected_outputs[f"{model_name}_patch"]),
        atol=1e-3,
        rtol=1e-3,
    )
    print("Forward pass looks ok!")

    save_dir = os.path.join(args.save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    image_processor.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model-name",
        default="vits",
        type=str,
        choices=["vits", "vitsplus", "vitb", "vitl", "vithplus", "vit7b"],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--save-dir",
        default="converted_models",
        type=str,
        help="Directory to save the converted model.",
    )
    args = parser.parse_args()
    convert_and_test_dinov3_checkpoint(args)
