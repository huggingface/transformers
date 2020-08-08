import argparse
from pathlib import Path
from typing import Dict

import tensorflow as tf
import torch
from tqdm import tqdm

from transformers import PegasusConfig, PegasusForConditionalGeneration


PATTERNS = [
    # replace left string with right string to get the relevant state_dict key (identical state dict to bart)
    ["memory_attention", "encoder_attn"],
    ["attention", "attn"],
    ["/", "."],
    [".LayerNorm.gamma", "_layer_norm.weight"],
    [".LayerNorm.beta", "_layer_norm.bias"],
    ["r.layer_", "r.layers."],
    ["output_proj", "out_proj"],
    ["ffn.dense_1.", "fc2."],
    ["ffn.dense.", "fc1."],
    ["ffn_layer_norm", "final_layer_norm"],
    ["kernel", "weight"],
    ["encoder_layer_norm.", "encoder.layer_norm."],
    ["decoder_layer_norm.", "decoder.layer_norm."],
    ["embeddings.weights", "shared.weight"],
]


def rename_state_dict_key(k):

    for pegasus_name, bart_name in PATTERNS:
        k = k.replace(pegasus_name, bart_name)
    # if k == 'embeddings.weight': return 'shared.weight'
    return k


# See appendix C of paper for all hyperparams
expected_max_length = {
    # See appendix C of paper
    "xsum": 64,
    "cnn_dailymail": 128,
    "newsroom": 128,
    "wikihow": 256,
    "multinews": 256,
    "reddit_tifu": 128,
    "big_patent": 256,
    "arxiv": 256,
    "pubmed": 256,
    "gigaword": 32,
    "aeslc": 32,
    "billsum": 256,
}
expected_alpha = {
    "multinews": 0.9,
    "wikihow": 0.6,
    "reddit_tifu": 0.6,
    "big_patent": 0.7,
    "gigaword": 0.6,
    "aeslc": 0.6,
    "billsum": 0.6,
}  # otherwise 0.8


def convert_pegasus_to_bart(tf_weights: dict, cfg_updates: dict) -> PegasusForConditionalGeneration:

    cfg = PegasusConfig(
        # normalize_embedding=False, add_final_layer_norm=True, static_position_embeddings=True, scale_embedding=True,
        activation_function="relu",
        attention_dropout=0.1,
        dropout=0.1,
        activation_dropout=0.1,
        vocab_size=96103,
        num_beams=8,
        **cfg_updates,
    )
    bart = PegasusForConditionalGeneration(cfg)
    sd = bart.model.state_dict()
    mapping = {}
    for k, v in tf_weights.items():
        # if k in IGNORE_KEYS:
        #    continue
        new_k = rename_state_dict_key(k)
        if new_k not in sd:
            raise ValueError(f"could not find new key {new_k} in state dict. (converted from {k}")

        if "dense" in k or "proj" in new_k:
            v = v.T
        mapping[new_k] = torch.tensor(v, dtype=sd[new_k].dtype)
        assert v.shape == sd[new_k].shape, f"{new_k}, {k}, {v.shape}, {sd[new_k].shape}"
    # make sure embedding.padding_idx is respected
    mapping["shared.weight"][cfg.pad_token_id] = torch.zeros_like(mapping["shared.weight"][cfg.pad_token_id + 1])
    mapping["encoder.embed_tokens.weight"] = mapping["shared.weight"]
    mapping["decoder.embed_tokens.weight"] = mapping["shared.weight"]
    # mapping['decoder.embed_positions'] = sd_new
    empty_biases = {k: torch.zeros_like(v) for k, v in sd.items() if k.endswith("bias") and k not in mapping}
    mapping.update(**empty_biases)
    missing, extra = bart.model.load_state_dict(mapping, strict=False)
    # missing_bias_keys =[k for k in missing]
    unexpected_missing = [
        k for k in missing if k not in ["encoder.embed_positions.weight", "decoder.embed_positions.weight"]
    ]
    assert unexpected_missing == []
    assert extra == []
    return bart


def get_tf_weights_as_numpy(path="./ckpt/aeslc/model.ckpt-32000") -> Dict:
    init_vars = tf.train.list_variables(path)
    names = []
    tf_weights = {}
    ignore_name = ["Adafactor", "global_step"]
    for name, shape in tqdm(init_vars, desc="converting tf checkpoint to dict"):
        skip_key = any([pat in name for pat in ignore_name])
        if skip_key:
            continue
        array = tf.train.load_variable(path, name)
        names.append(name)
        tf_weights[name] = array
    return tf_weights


def convert_pegasus_ckpt_to_pytorch(ckpt_path, save_dir):
    tf_weights = get_tf_weights_as_numpy(ckpt_path)
    dataset = Path(ckpt_path).parent.name
    cfg_updates = dict(max_length=expected_max_length[dataset], beam_alpha=expected_alpha.get(dataset, 0.8))
    torch_model = convert_pegasus_to_bart(tf_weights, cfg_updates)
    torch_model.save_pretrained(save_dir)
    # pickle_save(tf_weights, f'{save_dir}/tf_weights_dict.pkl')#DEL


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("tf_ckpt_path", type=str, help="passed to tf.train.list_variables")
    parser.add_argument("save_dir", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = f"pegasus/{Path(args.tf_ckpt_path).parent.name}"
    convert_pegasus_ckpt_to_pytorch(args.tf_ckpt_path, args.save_dir)
