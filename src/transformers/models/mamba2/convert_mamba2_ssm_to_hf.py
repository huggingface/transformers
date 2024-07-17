import argparse
import json
import re
from os import makedirs

import torch
from safetensors.torch import save_model

from transformers import AutoTokenizer, Mamba2Config, Mamba2ForCausalLM
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def convert_ssm_config_to_hf_config(config_ssm: dict) -> Mamba2Config:
    """Convert a Mamba2Config from mamba_ssm to a Mamba2Config from here."""
    hf_config = Mamba2Config()

    # Set important values from config and recalculate other resulting entries
    hf_config.hidden_size = config_ssm["d_model"]
    hf_config.intermediate_size = config_ssm["d_model"] * hf_config.expand
    hf_config.mamba2_num_heads = hf_config.intermediate_size // hf_config.mamba2_head_dim
    hf_config.mlp_intermediate_size = config_ssm["d_intermediate"]
    hf_config.num_hidden_layers = config_ssm["n_layer"]
    hf_config.attention_layers_idx = config_ssm["attn_layer_idx"]

    # (Optional) Attention related settings
    if len(config_ssm["attn_cfg"]) > 0:
        hf_config.attention_conv_kernel = config_ssm["attn_cfg"].get("d_conv", 0)
        hf_config.attention_head_dim = config_ssm["attn_cfg"]["head_dim"]
        hf_config.num_attention_heads = config_ssm["attn_cfg"]["num_heads"]
        hf_config.num_key_value_heads = config_ssm["attn_cfg"].get("num_heads_kv", hf_config.num_attention_heads)
        hf_config.use_attention_out_bias = config_ssm["attn_cfg"]["out_proj_bias"]
        hf_config.use_attention_qkv_bias = config_ssm["attn_cfg"]["qkv_proj_bias"]
        hf_config.rope_emb_dim = config_ssm["attn_cfg"]["rotary_emb_dim"]

    hf_config.residual_in_fp32 = config_ssm["residual_in_fp32"]
    hf_config.tie_embedding_weights = config_ssm["tie_embeddings"]

    # Padded vocab size, mostly of 16 but 32 is also very common in different models
    vocab_size = config_ssm["vocab_size"]
    pad_vocab_size_multiple = config_ssm["pad_vocab_size_multiple"]
    if (vocab_size % pad_vocab_size_multiple) != 0:
        vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
    hf_config.vocab_size = vocab_size

    return hf_config


def convert_ssm_to_hf(ssm_dir, output_dir):
    original_model_params = re.search(r"(?:(?<=mamba2-)|(?<=mamba2attn-)|(?<=transformerpp-)).*", ssm_dir).group(0)

    logger.info(f"Loading original config file of mamba2-{original_model_params}")
    config_file = ssm_dir + "/config.json"
    with open(config_file, "r", encoding="utf-8") as json_file:
        original_ssm_config_dict = json.load(json_file)

    logger.info("Converting to hf format and initializing empty hf model")
    hf_config = convert_ssm_config_to_hf_config(original_ssm_config_dict)
    hf_model = Mamba2ForCausalLM(hf_config)

    logger.info("Load and transfer original weights to the new hf model")
    mamba_checkpoint_path = ssm_dir + "/pytorch_model.bin"
    hf_state_dict = torch.load(mamba_checkpoint_path, map_location="cpu")
    # TODO: check if RoPE needs extra treatment like done in Llama
    hf_model.load_state_dict(hf_state_dict)

    logger.info("Load corresponding tokenizer")
    # kinda ugly but whatever
    mamba2_to_mamba1_parameters = {
        "130m": "130m",
        "370m": "370m",
        "780m": "790m",
        "1.3b": "1.4b",
        "2.7b": "2.8b",
    }
    hf_tokenizer = AutoTokenizer.from_pretrained(
        f"state-spaces/mamba-{mamba2_to_mamba1_parameters[original_model_params]}-hf"
    )

    # TODO: make the name more specific
    save_dir = f"{output_dir}/mamba2-{original_model_params}"
    logger.info(f"Saving hf config, model, and tokenizer to {save_dir}")
    makedirs(save_dir, exist_ok=True)
    save_model(hf_model, save_dir + "/model.safetensors", metadata={"format": "pt"})
    hf_config.save_pretrained(save_dir)
    hf_tokenizer.save_pretrained(save_dir)

    logger.info("Successfully converted to hf!")


if __name__ == "__main__":
    # TODO: make it compatible with attn and transformer++, test it, cleanup
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_ssm_dir",
        type=str,
        required=True,
        help="Path to the directory containing the `pytorch_model.bin` mamba_ssm checkpoint file and "
        "the corresponding `config.json`.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory to save the converted output model and config to.",
    )
    args = parser.parse_args()

    convert_ssm_to_hf(args.input_ssm_dir, args.output_dir)
