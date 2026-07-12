"""Convert LingBot-Vision checkpoints from the original repository.

URL: https://github.com/robbyant/lingbot-vision
"""

import argparse

import torch
from huggingface_hub import hf_hub_download

from transformers import LingbotVisionConfig, LingbotVisionModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


VARIANT_TO_REPO_ID = {
    "small": "robbyant/lingbot-vision-vit-small",
    "base": "robbyant/lingbot-vision-vit-base",
    "large": "robbyant/lingbot-vision-vit-large",
    "giant": "robbyant/lingbot-vision-vit-giant",
}

VARIANT_TO_CONFIG = {
    "small": {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "ffn_layer": "mlp",
        "qkv_bias": True,
    },
    "base": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "ffn_layer": "mlp",
        "qkv_bias": True,
    },
    "large": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "ffn_layer": "mlp",
        "qkv_bias": True,
    },
    "giant": {
        "hidden_size": 1536,
        "num_hidden_layers": 40,
        "num_attention_heads": 24,
        "intermediate_size": 4096,
        "ffn_layer": "swiglu",
        "qkv_bias": False,
    },
}


def get_lingbot_vision_config(variant):
    config = LingbotVisionConfig(
        image_size=512,
        patch_size=16,
        num_channels=3,
        num_storage_tokens=4,
        rope_theta=100.0,
        rope_normalize_coords="separate",
        rope_rescale_coords=2.0,
        rope_dtype="fp32",
        norm_layer="layernormbf16",
        mask_k_bias=True,
        layer_scale_init_value=1e-5,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        proj_bias=True,
        ffn_bias=True,
        **VARIANT_TO_CONFIG[variant],
    )

    return config


def create_rename_keys(config):
    rename_keys = [
        ("cls_token", "embeddings.cls_token"),
        ("mask_token", "embeddings.mask_token"),
        ("storage_tokens", "embeddings.storage_tokens"),
        ("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight"),
        ("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias"),
        ("rope_embed.periods", "rope_embeddings.periods"),
        ("norm.weight", "layernorm.weight"),
        ("norm.bias", "layernorm.bias"),
    ]

    if config.untie_cls_and_patch_norms:
        rename_keys.extend(
            [
                ("cls_norm.weight", "layernorm.weight"),
                ("cls_norm.bias", "layernorm.bias"),
            ]
        )

    for i in range(config.num_hidden_layers):
        rename_keys.extend(
            [
                (f"blocks.{i}.norm1.weight", f"encoder.layers.{i}.norm1.weight"),
                (f"blocks.{i}.norm1.bias", f"encoder.layers.{i}.norm1.bias"),
                (f"blocks.{i}.attn.qkv.weight", f"encoder.layers.{i}.attention.qkv.weight"),
                (f"blocks.{i}.attn.proj.weight", f"encoder.layers.{i}.attention.projection.weight"),
                (f"blocks.{i}.attn.proj.bias", f"encoder.layers.{i}.attention.projection.bias"),
                (f"blocks.{i}.ls1.gamma", f"encoder.layers.{i}.layer_scale1.gamma"),
                (f"blocks.{i}.norm2.weight", f"encoder.layers.{i}.norm2.weight"),
                (f"blocks.{i}.norm2.bias", f"encoder.layers.{i}.norm2.bias"),
                (f"blocks.{i}.ls2.gamma", f"encoder.layers.{i}.layer_scale2.gamma"),
            ]
        )

        if config.qkv_bias:
            rename_keys.extend(
                [
                    (f"blocks.{i}.attn.qkv.bias", f"encoder.layers.{i}.attention.qkv.bias"),
                    (f"blocks.{i}.attn.qkv.bias_mask", f"encoder.layers.{i}.attention.qkv.bias_mask"),
                ]
            )

        if config.ffn_layer == "mlp":
            rename_keys.extend(
                [
                    (f"blocks.{i}.mlp.fc1.weight", f"encoder.layers.{i}.mlp.fc1.weight"),
                    (f"blocks.{i}.mlp.fc1.bias", f"encoder.layers.{i}.mlp.fc1.bias"),
                    (f"blocks.{i}.mlp.fc2.weight", f"encoder.layers.{i}.mlp.fc2.weight"),
                    (f"blocks.{i}.mlp.fc2.bias", f"encoder.layers.{i}.mlp.fc2.bias"),
                ]
            )
        elif config.ffn_layer == "swiglu":
            rename_keys.extend(
                [
                    (f"blocks.{i}.mlp.w1.weight", f"encoder.layers.{i}.mlp.w1.weight"),
                    (f"blocks.{i}.mlp.w1.bias", f"encoder.layers.{i}.mlp.w1.bias"),
                    (f"blocks.{i}.mlp.w2.weight", f"encoder.layers.{i}.mlp.w2.weight"),
                    (f"blocks.{i}.mlp.w2.bias", f"encoder.layers.{i}.mlp.w2.bias"),
                    (f"blocks.{i}.mlp.w3.weight", f"encoder.layers.{i}.mlp.w3.weight"),
                    (f"blocks.{i}.mlp.w3.bias", f"encoder.layers.{i}.mlp.w3.bias"),
                ]
            )

    return rename_keys


def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val


def unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in ("teacher", "model_state", "state_dict", "model", "backbone"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    return checkpoint


def remove_state_dict_prefixes(state_dict):
    state_dict = unwrap_state_dict(state_dict)

    new_state_dict = {}
    for key, value in state_dict.items():
        key = key.replace("_orig_mod.", "")
        key = key.removeprefix("backbone.")
        new_state_dict[key] = value

    return new_state_dict


def load_original_state_dict(checkpoint_path, variant, cache_dir=None, revision=None):
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(
            repo_id=VARIANT_TO_REPO_ID[variant],
            filename="model.pt",
            cache_dir=cache_dir,
            revision=revision,
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    return remove_state_dict_prefixes(checkpoint)


@torch.no_grad()
def convert_lingbot_vision_checkpoint(
    variant,
    pytorch_dump_folder_path,
    checkpoint_path=None,
    push_to_hub=False,
    verify_shapes=False,
    cache_dir=None,
    revision=None,
):
    config = get_lingbot_vision_config(variant)
    model = LingbotVisionModel(config)
    model.eval()

    state_dict = load_original_state_dict(checkpoint_path, variant, cache_dir=cache_dir, revision=revision)

    for src, dest in create_rename_keys(config):
        if src in state_dict:
            rename_key(state_dict, src, dest)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        raise ValueError(f"Missing keys when loading LingBot-Vision checkpoint: {missing_keys}")
    if unexpected_keys:
        raise ValueError(f"Unexpected keys when loading LingBot-Vision checkpoint: {unexpected_keys}")

    if verify_shapes:
        pixel_values = torch.zeros(1, config.num_channels, config.image_size, config.image_size)
        outputs = model(pixel_values)
        expected_sequence_length = 1 + config.num_storage_tokens + (config.image_size // config.patch_size) ** 2
        if outputs.last_hidden_state.shape != (1, expected_sequence_length, config.hidden_size):
            raise ValueError(
                "Unexpected output shape: "
                f"{outputs.last_hidden_state.shape}, expected {(1, expected_sequence_length, config.hidden_size)}"
            )

    logger.info("Saving model to %s", pytorch_dump_folder_path)
    model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        repo_id = VARIANT_TO_REPO_ID[variant].split("/")[-1]
        model.push_to_hub(repo_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        default="base",
        choices=sorted(VARIANT_TO_CONFIG),
        help="LingBot-Vision checkpoint variant to convert.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        help="Local path to the original checkpoint. If omitted, the checkpoint is downloaded from the Hub.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,
        type=str,
        help="Path to the output Transformers checkpoint directory.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Push the converted checkpoint to the Hub.")
    parser.add_argument(
        "--verify_shapes", action="store_true", help="Run a forward pass shape check after conversion."
    )
    parser.add_argument(
        "--cache_dir", default=None, type=str, help="Cache directory used when downloading from the Hub."
    )
    parser.add_argument(
        "--revision", default=None, type=str, help="Checkpoint revision used when downloading from the Hub."
    )
    args = parser.parse_args()

    convert_lingbot_vision_checkpoint(
        variant=args.variant,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        checkpoint_path=args.checkpoint_path,
        push_to_hub=args.push_to_hub,
        verify_shapes=args.verify_shapes,
        cache_dir=args.cache_dir,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
