# Run this script from the root of CogView repo 
# https://github.com/THUDM/CogView


import torch

from transformers import CogViewVQVAE, CogViewConfig, CogViewForCausalLM

from vqvae import new_model
from vqvae.vqvae_zc import VQVAE

def convert_vq_vae(vq_vae_checkpoint_path, save_path):
    pt_model = new_model().eval()
    ckpt = torch.load(vq_vae_checkpoint_path, map_location="cpu")
    if list(ckpt.keys())[0].startswith('module.'):
        ckpt = {k[7:]: v for k, v in ckpt.items()}
    
    config = CogViewConfig()
    hf_model = CogViewVQVAE(config).eval()
    
    # copy encoder conv blocks
    pt_enc_conv_blocks = [pt_model.enc_b.blocks[i]  for i in range(0, len(pt_model.enc_b.blocks) - 1, 2)]
    for hf_enc_conv_block, pt_enc_conv_block in zip(hf_model.encoder.conv_block.blocks, pt_enc_conv_blocks):
        hf_enc_conv_block.weight.data = pt_enc_conv_block.weight.data
        hf_enc_conv_block.bias.data = pt_enc_conv_block.bias.data
    
    # copy output conv
    hf_model.encoder.out_conv.weight.data = pt_model.enc_b.blocks[-1].weight.data
    hf_model.encoder.out_conv.bias.data = pt_model.enc_b.blocks[-1].bias.data
    
    # copy decoder in_conv
    hf_model.decoder.in_conv.weight.data = pt_model.dec.blocks[0].weight.data
    hf_model.decoder.in_conv.bias.data = pt_model.dec.blocks[0].bias.data
    
    # copy decoder conv_blocks
    pt_dec_conv_blocks = [pt_model.dec.blocks[i] for i in range(0, len(pt_model.dec.blocks), 2)][1:]
    for hf_dec_conv_block, pt_dec_conv_block in zip(hf_model.decoder.conv_blocks.blocks, pt_dec_conv_blocks):
        hf_dec_conv_block.weight.data = pt_dec_conv_block.weight.data
        hf_dec_conv_block.bias.data = pt_dec_conv_block.bias.data
    
    # copy quantize weights
    hf_model.quantizer.embed = pt_model.quantize_t.embed
    hf_model.quantizer.embed_avg = pt_model.quantize_t.embed_avg

    hf_model.save_pretrained(save_path)


def rename_key(key):
    key = key.replace("layers", "blocks")
    if "word_embeddings" in key:
        key = f"transformer.{key}"
    elif "query_key_value" in key:
        key = key.replace("query_key_value", "qkv_proj")
    elif "attention.dense" in key:
        key = key.replace("attention.dense", "attention.out_proj")
    elif "post_attention_layernorm" in key:
         key = key.replace("post_attention_layernorm", "post_attn_layernorm")
    elif "third_layernorm" in key:
        key = key.replace("third_layernorm", "scale_layernorm_1")
    elif "fourth_layernorm" in key:
        key = key.replace("fourth_layernorm", "scale_layernorm_2")
    elif "dense_h_to_4h" in key:
        key = key.replace("dense_h_to_4h", "c_fc")
    elif "dense_4h_to_h" in key:
        key = key.replace("dense_4h_to_h", "c_proj")
    return key

def convert_congview_gpt(checkpoint_path, save_path):
    config = CogViewConfig(
        n_ctx=1089,
        n_positions=1089,
        n_embd=2560,
        n_head=40,
        n_layer=48,
        vocab_size=58240,
    )
    hf_model = CogViewForCausalLM(config).eval()

    state_dict = torch.load(checkpoint_path, map_location="cpu")["module"]
    keys = list(state_dict.keys())

    for k in keys:
        renamed_key = rename_key(k)
        state_dict[renamed_key] = state_dict.pop(k)

    _ = hf_model.load_state_dict(state_dict, strict=False)

    hf_model.tie_weights()
    hf_model.save_pretrained(save_path)
