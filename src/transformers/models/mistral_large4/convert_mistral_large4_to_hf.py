import argparse
import json
from pathlib import Path
from typing import Any

from transformers.models.mistral_large4.modular_mistral_large4 import MistralLarge4Config


HF_TO_VLLM_MAPPING = {
    # Top level
    "spatial_merge_size": "vision_encoder.spatial_merge_size",
    "image_token_index": "vision_encoder.image_token_id",
    "tie_word_embeddings": "tied_embeddings",
    "text_config.tie_word_embeddings": "tied_embeddings",
    "multimodal_projector_bias": "vision_encoder.adapter_bias",
    # Text model: general
    "text_config.hidden_size": "dim",
    "text_config.num_hidden_layers": "n_layers",
    "text_config.head_dim": "head_dim",
    "text_config.intermediate_size": "hidden_dim",
    "text_config.num_attention_heads": "n_heads",
    "text_config.num_key_value_heads": "n_kv_heads",
    "text_config.rms_norm_eps": "norm_eps",
    "text_config.rope_parameters.rope_theta": "rope_theta",
    "text_config.max_position_embeddings": "max_position_embeddings",
    "text_config.vocab_size": "vocab_size",
    # Text model: YaRN (`apply_scale: false` means no magnitude scaling, see the `attention_factor` constant)
    "text_config.rope_parameters.factor": "yarn.factor",
    "text_config.rope_parameters.original_max_position_embeddings": "yarn.original_max_position_embeddings",
    "text_config.rope_parameters.beta_fast": "yarn.beta",
    "text_config.rope_parameters.beta_slow": "yarn.alpha",
    # Text model: MLA
    "text_config.kv_lora_rank": "kv_lora_rank",
    "text_config.qk_rope_head_dim": "qk_rope_head_dim",
    "text_config.qk_nope_head_dim": "qk_nope_head_dim",
    "text_config.v_head_dim": "v_head_dim",
    # Text model: MoE
    "text_config.n_routed_experts": "moe.num_experts",
    "text_config.num_experts_per_tok": "moe.num_experts_per_tok",
    "text_config.n_shared_experts": "moe.num_shared_experts",
    "text_config.moe_intermediate_size": "moe.expert_hidden_dim",
    "text_config.routed_scaling_factor": "moe.routed_scale",
    "text_config.first_k_dense_replace": "moe.first_k_dense_replace",
    "text_config.n_group": "moe.num_expert_groups",
    "text_config.topk_group": "moe.num_expert_groups_per_tok",
    # Text model: DSA indexer
    "text_config.index_topk": "dsa.topk",
    "text_config.index_n_heads": "dsa.n_heads",
    "text_config.index_head_dim": "dsa.head_dim",
    # Vision model
    "vision_config.hidden_size": "vision_encoder.hidden_size",
    "vision_config.intermediate_size": "vision_encoder.intermediate_size",
    "vision_config.num_hidden_layers": "vision_encoder.num_hidden_layers",
    "vision_config.num_attention_heads": "vision_encoder.num_attention_heads",
    "vision_config.num_channels": "vision_encoder.num_channels",
    "vision_config.image_size": "vision_encoder.image_size",
    "vision_config.patch_size": "vision_encoder.patch_size",
    "vision_config.rope_parameters.rope_theta": "vision_encoder.rope_theta",
}

def recursive_get(d: dict, key: tuple[str, ...]) -> Any:
    # Error case
    if not isinstance(key, tuple) or len(key) == 0:
        raise ValueError(f"Invalid key: {key}")
    # Stop case
    if len(key) == 1:
        return d[key[0]]
    # Recursive case
    else:
        return recursive_get(d[key[0]], key[1:])


def recursive_set(d: dict, key: tuple[str, ...], value: Any) -> None:
    # Error case
    if not isinstance(key, tuple) or len(key) == 0:
        raise ValueError(f"Invalid key: {key}")
    # Stop case
    if len(key) == 1:
        d[key[0]] = value
    # Recursive case
    else:
        if key[0] not in d:
            d[key[0]] = {}
        recursive_set(d[key[0]], key[1:], value)


def format_quantization_config(source: dict) -> dict:
    # Quantization method
    qformat_weight = source["qformat_weight"]
    quant_method = {"fp8_e4m3": "fp8"}[qformat_weight]
    # Activation scheme
    qscheme_act = source["qscheme_act"]
    activation_scheme = {"BLOCK": "dynamic", "NO_SCALES": "dynamic", "TENSOR": "static"}[qscheme_act]
    # Weight block size
    block_size_out = source["block_size_out"]
    block_size_in = source["block_size_in"]
    weight_block_size = [block_size_out, block_size_in]
    # Modules not to convert
    modules_to_not_convert = [
        "model.vision_tower",
        "model.multi_modal_projector",
        "lm_head",
        "kv_b_proj",
        "indexer.wk",
        "indexer.weights_proj"
    ]
    if not source["quantize_kv_a"]:
        modules_to_not_convert.append("kv_a_proj_with_mqa")
    # Format and return the config
    return {
        "quant_method": quant_method,
        "activation_scheme": activation_scheme,
        "weight_block_size": weight_block_size,
        "modules_to_not_convert": modules_to_not_convert,
    }



def convert_config(source: dict) -> dict:
    # Start with a config dict that has the parameters vLLM hardcodes
    config_dict = {
        "model_type": "mistral_large4",
        "projector_hidden_act": "gelu",
        "vision_feature_layer": -1,
        "text_config": {
            "model_type": "mistral_large4_text",
            "hidden_act": "silu",
            "norm_topk_prob": True,
            "rope_parameters": {"rope_type": "yarn", "attention_factor": 1.0},
        },
        "vision_config": {
            "model_type": "pixtral",
            "hidden_act": "silu",
            "rope_parameters": {"rope_type": "axial"},
        },
    }

    # Map the parameters from the source config to the new config
    for hf_key, vllm_key in HF_TO_VLLM_MAPPING.items():
        hf_key, vllm_key = (tuple(x.split(".")) for x in (hf_key, vllm_key))
        value = recursive_get(source, vllm_key)
        recursive_set(config_dict, hf_key, value)

    # For some parameters, the mapping is more complex:
    # ... indexer_types
    config_dict["text_config"]["indexer_types"] = source["dsa"]["sharing"].split(",")
    # ... no_rope_layers
    layers_pe = source["nope"].split(",")
    num_hidden_layers = config_dict["text_config"]["num_hidden_layers"]
    if num_hidden_layers % len(layers_pe) != 0:
        raise ValueError(
            f"{num_hidden_layers = } should be divisible by the number of no-rope layers ({len(layers_pe)})"
        )
    repeats = num_hidden_layers // len(layers_pe)
    no_rope_layers = [{"rope": 1, "nope": 0}[layer_pe] for layer_pe in layers_pe] * repeats
    config_dict["text_config"]["no_rope_layers"] = no_rope_layers
    # ... kv_source_layers
    kv_source_layers = [int(layer_idx) for layer_idx in source["kv_sharing"]["source_layers_str"].split(",")]
    if len(kv_source_layers) != num_hidden_layers:
        raise ValueError(
            f"kv_sharing.source_layers_str has {len(kv_source_layers)} entries, expected {num_hidden_layers}"
        )
    config_dict["text_config"]["kv_source_layers"] = kv_source_layers
    # ... mlp_layer_types
    first_k_dense_replace = source["moe"]["first_k_dense_replace"]
    mlp_layer_types = ["dense"] * first_k_dense_replace + ["sparse"] * (num_hidden_layers - first_k_dense_replace)
    config_dict["text_config"]["mlp_layer_types"] = mlp_layer_types

    # If there is a quantization config in the source, transfer it to the config
    if "quantization" in source:
        config_dict["quantization_config"] = format_quantization_config(source["quantization"])

    return config_dict


# Adapted from `convert_and_write_processor` in mistral3/convert_mistral3_weights_to_hf.py
def convert_and_write_processor(input_dir: Path, output_dir: Path, params: dict) -> PixtralProcessor:
    """Tokenizer (tekken.json -> fast tokenizer), chat template and image processor, saved as a PixtralProcessor."""
    tokenizer = convert_tekken_tokenizer(str(input_dir / "tekken.json"))
    # The v15 tokenizer emits a `[MODEL_SETTINGS]` block that the checkpoint's own template reproduces
    chat_template_path = input_dir / "chat_template.jinja"
    if not chat_template_path.exists():
        raise FileNotFoundError(f"{chat_template_path} is needed to reproduce mistral-common's chat encoding")
    chat_template = chat_template_path.read_text(encoding="utf-8")

    vision = params["vision_encoder"]
    image_processor = PixtralImageProcessorFast(
        patch_size=vision["patch_size"], size={"longest_edge": vision["max_image_size"]}
    )
    processor = PixtralProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token="[IMG]",
        patch_size=vision["patch_size"],
        spatial_merge_size=vision["spatial_merge_size"],
        chat_template=chat_template,
    )
    processor.save_pretrained(output_dir)
    return processor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Location of the mistral_large4 checkpoint")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Location to write the HF-formatted checkpoint. If not specified, only the config will be created.",
    )
    parser.add_argument("--config_name", type=str, default="params.json", help="The name of the original config file.")
    parser.add_argument("--config-only", action="store_true", help="Only write config.json.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir

    config_path = input_dir / args.config_name
    with open(config_path, encoding="utf-8") as f:
        params = json.load(f)
    if args.config_only:

        # Tokenizer, chat template, image processor
        processor = convert_and_write_processor(input_dir, output_dir, params)
