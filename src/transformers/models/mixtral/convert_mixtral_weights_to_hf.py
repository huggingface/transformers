import torch
from glob import glob
import argparse
from transformers import MixtralConfig, MixtralForCausalLM, AutoTokenizer

KEYS_TO_MODIFY_MAPPING = {
    "tok_embeddings": "model.embed_tokens",
    "layers": "model.layers",
    "output": "lm_head",
    "wq": "q_proj",
    "wk": "k_proj",
    "wo": "o_proj",
    "wv": "v_proj",
    ".attention.": ".self_attn.",
    ".attention_norm.": ".input_layernorm.",
    ".ffn_norm.": ".post_attention_layernorm.",
}

KEYS_TO_MODIFY_EXACT_MATCH = {
    "norm.weight": "model.norm.weight"
}

torch.set_default_dtype(torch.bfloat16)

def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        for key_to_modify, new_key in KEYS_TO_MODIFY_EXACT_MATCH.items():
            if key_to_modify == key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value
    return new_state_dict


def load_and_save_weights(weights_path, save_model_path):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    pt_files = glob(weights_path + "/*.pt")
    config = MixtralConfig(num_hidden_layers=32, hidden_size=1024 * 4, intermediate_size=3584 * 4)

    with torch.device("meta"):
        model = MixtralForCausalLM(config)

    mismatched_weights = []

    for pt_file in pt_files:
        new_state_dict = {}

        partial_state_dict = torch.load(pt_file, map_location="cpu")
        partial_state_dict = convert_state_dict_to_hf(partial_state_dict)

        # Clone parameters to avoid a bug with safetensors
        new_state_dict = {k: v.clone() for k, v in partial_state_dict.items()}

        errors = model.load_state_dict(new_state_dict, strict=False, assign=True)
        mismatched_weights.append(errors)


    for n, p in model.named_parameters():
        assert p.device.type != "meta", f"{n} has not been loaded properly"

    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of Mistral weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    args = parser.parse_args()
    load_and_save_weights(args.input_dir, args.output_dir)