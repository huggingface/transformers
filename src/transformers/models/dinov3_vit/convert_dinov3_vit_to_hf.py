"""Convert DINOv3 checkpoints from the original repository.

URL: https://github.com/facebookresearch/dinov3/tree/main
"""

import argparse
from typing import Optional
import torch
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
            0.47380000352859497,
            -0.4156099855899811,
            0.41168999671936035,
            -0.12477999925613403,
            -0.29596999287605286,
        ],
        "vits_patch": [
            -0.03959000110626221,
            -0.25310999155044556,
            -0.015850000083446503,
            -0.45699000358581543,
            0.5675600171089172,
        ],
        "vitsplus_cls": [
            -0.47488999366760254,
            -1.3652199506759644,
            -0.327349990606308,
            0.3742400109767914,
            -0.7740300297737122,
        ],
        "vitsplus_patch": [
            0.1493300050497055,
            -0.3805299997329712,
            -0.40046998858451843,
            -0.15716999769210815,
            -0.5877799987792969,
        ],
        "vitb_cls": [
            1.0481300354003906,
            -0.16398000717163086,
            -0.34836000204086304,
            -0.07030999660491943,
            -0.018640000373125076,
        ],
        "vitb_patch": [
            -0.07953999936580658,
            -0.455269992351532,
            -0.7357199788093567,
            -0.43566998839378357,
            -0.1476300060749054,
        ],
        "vitl_cls": [
            0.483489990234375,
            -0.5878999829292297,
            0.4768800139427185,
            0.585349977016449,
            0.9454799890518188,
        ],
        "vitl_patch": [
            -0.21309000253677368,
            -0.49483001232147217,
            -0.2584800124168396,
            0.10723999887704849,
            0.14616000652313232,
        ],
        "vithplus_cls": [
            -0.06420999765396118,
            -0.14941999316215515,
            -0.6185899972915649,
            0.6363400220870972,
            0.1524599939584732,
        ],
        "vithplus_patch": [
            -0.09335999935865402,
            0.2837600111961365,
            -0.04964999854564667,
            0.42445001006126404,
            0.09500999748706818,
        ],
        "vit7b_cls": [
            0.2755500078201294,
            -0.26047998666763306,
            0.06796000152826309,
            0.050620000809431076,
            -0.15916000306606293,
        ],
        "vit7b_patch": [
            0.04416000097990036,
            -0.05305999889969826,
            0.07196000218391418,
            -0.06457000225782394,
            -0.026270000264048576,
        ],
    }

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
    torch.allclose(
        torch.Tensor(actual_outputs[f"{model_name}_cls"]),
        torch.Tensor(expected_outputs[f"{model_name}_cls"]),
        atol=1e-3,
    )
    torch.allclose(
        torch.Tensor(actual_outputs[f"{model_name}_patch"]),
        torch.Tensor(expected_outputs[f"{model_name}_patch"]),
        atol=1e-3,
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
