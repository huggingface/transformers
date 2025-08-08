"""Convert DINOv3 checkpoints from the original repository.

URL: https://github.com/facebookresearch/dinov3/tree/main
"""

import argparse
from typing import Optional
import torch

import random
import numpy as np
from torchvision import transforms
import requests
from PIL import Image
from transformers import DINOv3ViTConfig, DINOv3ViTModel
from huggingface_hub import hf_hub_download

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


def get_dinov3_config(model_name: str) -> Optional[DINOv3ViTConfig]:
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
            pos_embed_rope_normalize_coords="separate",
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
            pos_embed_rope_normalize_coords="separate",
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
            num_register_tokens=4,
            layerscale_value=1.0,
            mlp_ratio=4,
            use_swiglu_ffn=False,
            layer_norm_eps=1e-5,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
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
            pos_embed_rope_normalize_coords="separate",
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
            pos_embed_rope_normalize_coords="separate",
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
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
        )
    else:
        raise ValueError("Model not supported")


def convert_dinov3_vit_to_hf_vit(original_dinov3_state_dict, config: DINOv3ViTConfig):
    embed_dim = config.hidden_size
    hf_dinov3_state_dict = {}
    for key in original_dinov3_state_dict.keys():
        val = original_dinov3_state_dict[key]
        if key == "cls_token":
            key = "embeddings.cls_token"
        elif key == "mask_token":
            key = "embeddings.mask_token"
        elif key == "storage_tokens":
            key = "embeddings.register_tokens"
        elif key.startswith("patch_embed.proj"):
            key = key.replace("patch_embed.proj", "embeddings.patch_embeddings.proj")
        elif key.startswith("rope_embed"):
            key = key.replace("rope_embed", "rope_embeddings")
        elif key.startswith("blocks"):
            key = key.replace("blocks", "layer")
        if "ls1." in key:
            key = key.replace("ls1", "layer_scale1")
        if "ls2." in key:
            key = key.replace("ls2", "layer_scale2")
        if "attn." in key:
            key = key.replace("attn.", "attention.")
        if "qkv." in key:
            prefix, suffix = key.split("qkv")
            if "bias_mask" in suffix:
                continue
            elif "bias" in suffix:
                q_e, k_e, v_e = (
                    val[0:embed_dim],
                    val[embed_dim : embed_dim * 2],
                    val[embed_dim * 2 :],
                )
            else:
                q_e, k_e, v_e = (
                    val[0:embed_dim, :],
                    val[embed_dim : embed_dim * 2, :],
                    val[embed_dim * 2 :, :],
                )
            hf_dinov3_state_dict[prefix + "query" + suffix] = q_e
            if not ("bias" in suffix and config.mask_k_bias):
                hf_dinov3_state_dict[prefix + "key" + suffix] = k_e
            hf_dinov3_state_dict[prefix + "value" + suffix] = v_e
        else:
            hf_dinov3_state_dict[key] = val
    return hf_dinov3_state_dict


def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


def make_transform(resize_size: int = 224):
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return transforms.Compose([to_tensor, resize, normalize])


@torch.no_grad()
def convert_and_test_dinov3_checkpoint(model_name):

    expected_outputs = {
        "vits_cls": [
            0.47379571199417114,
            -0.41561394929885864,
            0.41169291734695435,
            -0.12478338927030563,
            -0.2959742844104767,
        ],
        "vits_patch": [
            -0.03959187492728233,
            -0.25311151146888733,
            -0.015847790986299515,
            -0.45699289441108704,
            0.5675609707832336,
        ],
        "vitsplus_cls": [
            -0.4748912751674652,
            -1.3652222156524658,
            -0.32735151052474976,
            0.3742392957210541,
            -0.7740300893783569,
        ],
        "vitsplus_patch": [
            0.14932650327682495,
            -0.3805270791053772,
            -0.4004722833633423,
            -0.15717053413391113,
            -0.5877845287322998,
        ],
        "vitb_cls": [
            1.048130750656128,
            -0.16398264467716217,
            -0.3483588695526123,
            -0.07031229883432388,
            -0.018643084913492203,
        ],
        "vitb_patch": [
            -0.0795423611998558,
            -0.45527052879333496,
            -0.7357183694839478,
            -0.4356740117073059,
            -0.14763328433036804,
        ],
        "vitl_cls": [
            0.4834900200366974,
            -0.587904155254364,
            0.476875901222229,
            0.5853531360626221,
            0.9454823136329651,
        ],
        "vitl_patch": [
            -0.21309036016464233,
            -0.49482738971710205,
            -0.2584819495677948,
            0.1072424128651619,
            0.14616338908672333,
        ],
        "vithplus_cls": [
            -0.06420943140983582,
            -0.1494205743074417,
            -0.618586540222168,
            0.6363415122032166,
            0.15246111154556274,
        ],
        "vithplus_patch": [
            -0.09335622191429138,
            0.28375640511512756,
            -0.049649134278297424,
            0.4244541823863983,
            0.0950070321559906,
        ],
        "vit7b_cls": [
            0.27555006742477417,
            -0.2604803442955017,
            0.06795521825551987,
            0.05062410980463028,
            -0.15915830433368683,
        ],
        "vit7b_patch": [
            0.04416150599718094,
            -0.05306466668844223,
            0.0719609260559082,
            -0.06456729769706726,
            -0.026268284767866135,
        ],
    }

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

    config = get_dinov3_config(model_name)
    print(config)

    model = DINOv3ViTModel(config).eval()
    state_dict_path = hf_hub_download(
        repo_id=HUB_MODELS[model_name], filename=HUB_CHECKPOINTS[model_name]
    )
    original_state_dict = torch.load(state_dict_path)

    hf_state_dict = convert_dinov3_vit_to_hf_vit(original_state_dict, config)
    model.load_state_dict(hf_state_dict, strict=True)
    model = model.eval()

    image_preprocessor = make_transform()
    # load image
    images = [image_preprocessor(prepare_img())]
    image_tensor = torch.stack(images, dim=0)
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model_output = model(image_tensor)

    last_layer_class_token = model_output.pooler_output
    last_layer_patch_tokens = model_output.last_hidden_state[
        :, config.num_register_tokens + 1 :
    ]
    actual_outputs = {}
    actual_outputs[f"{model_name}_cls"] = last_layer_class_token[0, :5].tolist()
    actual_outputs[f"{model_name}_patch"] = last_layer_patch_tokens[0, 0, :5].tolist()
    print(actual_outputs[f"{model_name}_cls"], expected_outputs[f"{model_name}_cls"])
    torch.testing.assert_close(
        torch.Tensor(actual_outputs[f"{model_name}_cls"]),
        torch.Tensor(expected_outputs[f"{model_name}_cls"]),
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        torch.Tensor(actual_outputs[f"{model_name}_patch"]),
        torch.Tensor(expected_outputs[f"{model_name}_patch"]),
        atol=1e-2,
        rtol=1e-2,
    )
    print("Looks ok!")


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
    args = parser.parse_args()
    convert_and_test_dinov3_checkpoint(args.model_name)
