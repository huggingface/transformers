import argparse
import json
import os

import torch
from safetensors.torch import load_file
from tokenizers import processors

from transformers import GlmConfig, GlmForCausalLM, PreTrainedTokenizerFast


STATE_DICT_MAPPING = {
    "transformer.output_layer.": "lm_head.",
    "transformer.": "model.",
    ".embedding.word_embeddings.": ".embed_tokens.",
    ".encoder.final_layernorm.": ".norm.",
    ".encoder.layers.": ".layers.",
    "rotary_pos_embed.": "rotary_emb.",
    "self_attention.": "self_attn.",
    "query_key_value.": "qkv_proj.",
    "dense.": "o_proj.",
    "dense_h_to_4h.": "gate_up_proj.",
    "dense_4h_to_h.": "down_proj.",
}


def merge_safetensors(input_dir: str):
    all_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if x.endswith(".safetensors")]
    all_files = sorted(all_files, key=lambda x: int(x.rsplit("-", 3)[1]))

    all_weights = {}
    for file in all_files:
        tensors = load_file(file)
        all_weights.update(tensors)

    return all_weights


def convert_state_dict(original_state_dict: dict, config: GlmConfig):
    new_dict = {}

    head_dim = config.hidden_size // config.num_attention_heads
    query_size = config.num_attention_heads * head_dim
    kv_size = config.num_key_value_heads * head_dim

    for key, value in original_state_dict.items():
        # Should not be part of the state dict
        if "rotary_pos_emb.inv_freq" in key:
            continue

        new_key = key
        for old, new in STATE_DICT_MAPPING.items():
            new_key = new_key.replace(old, new)

        if "qkv_proj." in new_key:
            q_proj, k_proj, v_proj = value[:query_size, ...], value[query_size : query_size + kv_size, ...], value[query_size + kv_size : , ...]
            new_dict[new_key.replace("qkv_proj.", "q_proj.")] = q_proj
            new_dict[new_key.replace("qkv_proj.", "k_proj.")] = k_proj
            new_dict[new_key.replace("qkv_proj.", "v_proj.")] = v_proj
        else:
            new_dict[new_key] = value
    return new_dict


def convert_config(original_config: dict):
    num_attention_heads = original_config.pop("num_attention_heads")

    new_config = GlmConfig(
        vocab_size=original_config.pop("padded_vocab_size"),
        hidden_size=original_config.pop("hidden_size"),
        intermediate_size=original_config.pop("ffn_hidden_size"),
        num_hidden_layers=original_config.pop("num_layers"),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=(
            num_attention_heads
            if not original_config.pop("multi_query_attention")
            else original_config.pop("multi_query_group_num")
        ),
        attention_dropout=original_config.pop("attention_dropout"),
        max_position_embeddings=original_config.pop("seq_length"),
        rms_norm_eps=original_config.pop("layernorm_epsilon"),
        rope_theta=10000.0 * original_config.pop("rope_ratio", 1),
        use_cache=original_config.pop("use_cache"),
        head_dim=original_config.pop("kv_channels"),
        attention_bias=original_config.pop("add_qkv_bias"),
        eos_token_id=original_config.pop("eos_token_id"),
        pad_token_id=original_config.pop("pad_token_id"),
        tie_word_embeddings=original_config.pop("tie_word_embeddings"),
    )
    print(f"Unused config keys: {original_config.keys(),}")
    return new_config


def convert_glm_tokenizer(input_dir):
    fast_tok = PreTrainedTokenizerFast.from_pretrained(input_dir)
    # Add the two tokens automatically with post processor
    fast_tok._tokenizer.post_processor = processors.Sequence(
        [
            processors.ByteLevel(trim_offsets=False),
            processors.TemplateProcessing(
                single="[gMASK]:0 <sop>:0 $A:0",
                pair="[gMASK]:0 <sop>:0 $A:0 $B:1",
                special_tokens=[("[gMASK]", 151331), ("<sop>", 151333)],
            ),
        ],
    )

    return fast_tok


def convert_glm_model(input_dir, output_dir):
    # Load and convert config
    with open(os.path.join(input_dir, "config.json")) as f:
        original_config = json.load(f)
    config = convert_config(original_config)
    config.save_pretrained(output_dir)

    # Load and convert weights
    original_state_dict = merge_safetensors(input_dir)
    new_dict = convert_state_dict(original_state_dict, config)
    with torch.device("meta"):
        model = GlmForCausalLM(config)
    model.load_state_dict(new_dict, strict=True, assign=True)
    model.save_pretrained(output_dir)

    # Load and convert tokenizer
    tokenizer = convert_glm_tokenizer(input_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="Location of the local folder copied from the Hub.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Location to write HF model and tokenizer",
    )

    args = parser.parse_args()
    convert_glm_model(args.input_dir, args.output_dir)
