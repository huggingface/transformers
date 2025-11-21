import argparse
from argparse import Namespace

import torch
from torch import nn

from transformers import XGLMConfig, XGLMForCausalLM


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "decoder.version",
        "decoder.output_projection.weight",
        "_float_tensor",
        "decoder.embed_positions._float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def convert_fairseq_xglm_checkpoint_from_disk(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = Namespace(**checkpoint["cfg"]["model"])
    state_dict = checkpoint["model"]
    remove_ignore_keys_(state_dict)
    vocab_size = state_dict["decoder.embed_tokens.weight"].shape[0]

    state_dict = {key.replace("decoder", "model"): val for key, val in state_dict.items()}

    config = XGLMConfig(
        vocab_size=vocab_size,
        max_position_embeddings=args.max_target_positions,
        num_layers=args.decoder_layers,
        attention_heads=args.decoder_attention_heads,
        ffn_dim=args.decoder_ffn_embed_dim,
        d_model=args.decoder_embed_dim,
        layerdrop=args.decoder_layerdrop,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        activation_function="gelu",
        scale_embedding=not args.no_scale_embedding,
        tie_word_embeddings=args.share_decoder_input_output_embed,
    )

    model = XGLMForCausalLM(config)
    missing = model.load_state_dict(state_dict, strict=False)
    print(missing)
    model.lm_head = make_linear_from_emb(model.model.embed_tokens)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("fairseq_path", type=str, help="path to a model.pt on local filesystem.")
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    model = convert_fairseq_xglm_checkpoint_from_disk(args.fairseq_path)
    model.save_pretrained(args.pytorch_dump_folder_path)
