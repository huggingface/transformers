"""Convert LingBot-Vision checkpoints from the original repository.

URL: https://github.com/robbyant/lingbot-vision
"""

import argparse

import torch
from huggingface_hub import hf_hub_download

from transformers import LingbotVisionConfig, LingbotVisionImageProcessor, LingbotVisionModel
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


VARIANT_TO_REPO_ID = {
    "small": "robbyant/lingbot-vision-vit-small",
    "base": "robbyant/lingbot-vision-vit-base",
    "large": "robbyant/lingbot-vision-vit-large",
    "giant": "robbyant/lingbot-vision-vit-giant",
}

# The original `ffn_layer: mlp` variants run a plain GELU MLP of width `4 * hidden_size`; `ffn_layer: swiglu`
# runs a SiLU-gated MLP whose width is `int(4 * hidden_size * 2 / 3)` rounded up to a multiple of 8. The
# original `qkv_bias` covers query, key and value alike - the key slice is then masked to zero, which the
# checkpoints materialize as an all-zero key bias.
VARIANT_TO_CONFIG = {
    "small": {
        "hidden_size": 384,
        "intermediate_size": 1536,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "use_gated_mlp": False,
        "hidden_act": "gelu",
        "query_bias": True,
        "key_bias": True,
        "value_bias": True,
    },
    "base": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "use_gated_mlp": False,
        "hidden_act": "gelu",
        "query_bias": True,
        "key_bias": True,
        "value_bias": True,
    },
    "large": {
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "use_gated_mlp": False,
        "hidden_act": "gelu",
        "query_bias": True,
        "key_bias": True,
        "value_bias": True,
    },
    "giant": {
        "hidden_size": 1536,
        "intermediate_size": 4096,
        "num_hidden_layers": 40,
        "num_attention_heads": 24,
        "use_gated_mlp": True,
        "hidden_act": "silu",
        "query_bias": False,
        "key_bias": False,
        "value_bias": False,
    },
}


def get_lingbot_vision_config(variant):
    config = LingbotVisionConfig(
        image_size=512,
        patch_size=16,
        num_channels=3,
        num_register_tokens=4,
        rope_parameters={"rope_theta": 100.0},
        pos_embed_rescale=2.0,
        layer_norm_eps=1e-5,
        layerscale_value=1e-5,
        attention_dropout=0.0,
        proj_bias=True,
        mlp_bias=True,
        **VARIANT_TO_CONFIG[variant],
    )

    return config


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
    verify_shapes=False,
    cache_dir=None,
    revision=None,
):
    config = get_lingbot_vision_config(variant)
    state_dict = load_original_state_dict(checkpoint_path, variant, cache_dir=cache_dir, revision=revision)
    model, loading_info = LingbotVisionModel.from_pretrained(
        None, config=config, state_dict=state_dict, output_loading_info=True
    )
    if loading_info["missing_keys"] or loading_info["unexpected_keys"]:
        raise ValueError(
            "LingBot-Vision checkpoint conversion did not load cleanly: "
            f"missing={loading_info['missing_keys']}, unexpected={loading_info['unexpected_keys']}"
        )

    if verify_shapes:
        pixel_values = torch.zeros(1, config.num_channels, config.image_size, config.image_size)
        outputs = model(pixel_values)
        expected_sequence_length = 1 + config.num_register_tokens + (config.image_size // config.patch_size) ** 2
        if outputs.last_hidden_state.shape != (1, expected_sequence_length, config.hidden_size):
            raise ValueError(
                "Unexpected output shape: "
                f"{outputs.last_hidden_state.shape}, expected {(1, expected_sequence_length, config.hidden_size)}"
            )

    logger.info("Saving model to %s", pytorch_dump_folder_path)
    model.save_pretrained(pytorch_dump_folder_path)
    LingbotVisionImageProcessor().save_pretrained(pytorch_dump_folder_path)


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
        verify_shapes=args.verify_shapes,
        cache_dir=args.cache_dir,
        revision=args.revision,
    )


if __name__ == "__main__":
    main()
