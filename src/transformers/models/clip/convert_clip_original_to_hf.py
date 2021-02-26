def copy_attn_layer(hf_attn_layer, pt_attn_layer):
    q_proj, k_proj, v_proj = pt_attn_layer.in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = pt_attn_layer.in_proj_bias.chunk(3, dim=0)
    
    out_proj_weights = pt_attn_layer.out_proj.weight
    out_proj_bias = pt_attn_layer.out_proj.bias
    
    hf_attn_layer.q_proj.weight.data = q_proj
    hf_attn_layer.q_proj.bias.data = q_proj_bias
    
    hf_attn_layer.k_proj.weight.data = k_proj
    hf_attn_layer.k_proj.bias.data = k_proj_bias
    
    hf_attn_layer.v_proj.weight.data = v_proj
    hf_attn_layer.v_proj.bias.data = v_proj_bias
    
    hf_attn_layer.out_proj.weight = out_proj_weights
    hf_attn_layer.out_proj.bias = out_proj_bias

def copy_linear(hf_linear, pt_linear):
    hf_linear.weight = pt_linear.weight
    hf_linear.bias = pt_linear.bias

def copy_mlp(hf_mlp, pt_mlp):
    copy_linear(hf_mlp.fc1, pt_mlp.c_fc)
    copy_linear(hf_mlp.fc2, pt_mlp.c_proj)


def copy_layer(hf_layer, pt_layer):
    # copy layer norms
    copy_linear(hf_layer.layer_norm1, pt_layer.ln_1)
    copy_linear(hf_layer.layer_norm2, pt_layer.ln_2)
    
    # copy MLP
    copy_mlp(hf_layer.mlp, pt_layer.mlp)
    
    # copy attn
    copy_attn_layer(hf_layer.self_attn, pt_layer.attn)


def copy_layers(hf_layers, pt_layers):
    for hf_layer, pt_layer in zip(hf_layers, pt_layers):
        copy_layer(hf_layer, pt_layer)

def copy_encoder(hf_encoder, pt_model):
    # copy  embeds
    hf_encoder.token_embedding.weight = pt_model.token_embedding.weight
    hf_encoder.positional_embedding.weight.data = pt_model.positional_embedding
    
    # copy layer norm
    copy_linear(hf_encoder.final_layer_norm, pt_model.ln_final)
    
    # copy hidden layers
    copy_layers(hf_encoder.encoder.layers, pt_model.transformer.resblocks)


def copy_text_encoder_and_projection(hf_model, pt_model):
    # copy projection
    hf_model.text_projection.weight.data = pt_model.text_projection.data.T
    
    # copy text encoder
    copy_encoder(hf_model.transformer, pt_model)


def copy_vison_transformer(hf_model, pt_model):
    # copy projection
    hf_model.visiual_projection.weight.data = pt_model.visual.proj.data.T
    
    # copy conv
    hf_model.vision_transformer.conv.weight.data = pt_model.visual.conv1.weight.data
    
    # copy layer norms
    copy_linear(hf_model.vision_transformer.pre_layrnorm, pt_model.visual.ln_pre)
    copy_linear(hf_model.vision_transformer.post_layernorm, pt_model.visual.ln_post)
    
    # copy embeds
    hf_model.vision_transformer.class_embedding = pt_model.visual.class_embedding
    hf_model.vision_transformer.positional_embedding.weight.data = pt_model.visual.positional_embedding.data
    
    # copy encoder
    copy_layers(hf_model.vision_transformer.encoder.layers, pt_model.visual.transformer.resblocks)


import torch

from clip import load

from transformers import ClipConfig, ClipModel

config = ClipConfig(output_dim=512, text_config={}, vision_config={})
hf_model = ClipModel(config).eval()

pt_model, transforms = load("./notebooks/model.pt", jit=False)

copy_text_encoder_and_projection(hf_model, pt_model)
copy_vison_transformer(hf_model, pt_model)

hf_model.save_pretrained("hf_clip")
