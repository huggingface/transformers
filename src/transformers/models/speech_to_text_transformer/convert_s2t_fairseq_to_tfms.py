import torch
from torch import nn
import fairseq
from transformers import SpeechToTextTransformerConfig, SpeechToTextTransformerForConditionalGeneration

def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "encoder.embed_positions._float_tensor",
        "decoder.embed_positions._float_tensor",
        "decoder.output_projection.weight",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)

def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        if "transformer_layers" in key:
            s_dict[key.replace("transformer_layers", "layers")] = s_dict.pop(key)

def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def convert_fairseq_s2t_checkpoint_from_disk(checkpoint_path):
    m2m_100 = torch.load(checkpoint_path, map_location="cpu")
    args = m2m_100["args"]
    state_dict = m2m_100["model"]
    remove_ignore_keys_(state_dict)
    rename_keys(state_dict)
    vocab_size = state_dict["decoder.embed_tokens.weight"].shape[0]

    config = SpeechToTextTransformerConfig(
        vocab_size=vocab_size,
        max_position_embeddings=1024,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_attention_heads=args.encoder_attention_heads,
        decoder_attention_heads=args.decoder_attention_heads,
        encoder_ffn_dim=args.encoder_ffn_embed_dim,
        decoder_ffn_dim=args.decoder_ffn_embed_dim,
        d_model=args.encoder_embed_dim,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        activation_function="relu",
        conv_channels=args.conv_channels,
        conv_kernel_sizes=args.conv_kernel_sizes,
        input_feat_per_channel=args.input_feat_per_channel,
        input_channels=args.input_channels,
    )

    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    model = SpeechToTextTransformerForConditionalGeneration(config)
    model.model.load_state_dict(state_dict)
    model.lm_head = make_linear_from_emb(model.model.shared)

    return model