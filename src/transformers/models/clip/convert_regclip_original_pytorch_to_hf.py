# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import torch
from transformers import CLIPConfig, CLIPModel  # Your HuggingFace-style edited code

# Adapted for CLIP-fine-tune-registers-gated
# https://github.com/zer0int/CLIP-fine-tune-registers-gated
from INFERclipregXGATED import load  # Modified OpenAI-CLIP ViT with register tokens + fusion MLP Gates


def copy_linear(hf_linear, pt_linear):
    assert hf_linear.weight.shape == pt_linear.weight.shape, f"Linear weight shape mismatch: {hf_linear.weight.shape} vs {pt_linear.weight.shape}"
    hf_linear.weight.data.copy_(pt_linear.weight.data)
    if hf_linear.bias is not None and pt_linear.bias is not None:
        assert hf_linear.bias.shape == pt_linear.bias.shape, f"Linear bias shape mismatch: {hf_linear.bias.shape} vs {pt_linear.bias.shape}"
        hf_linear.bias.data.copy_(pt_linear.bias.data)

def copy_attn_layer(hf_attn_layer, pt_attn_layer):
    q_proj, k_proj, v_proj = pt_attn_layer.in_proj_weight.chunk(3, dim=0)
    q_proj_bias, k_proj_bias, v_proj_bias = pt_attn_layer.in_proj_bias.chunk(3, dim=0)
    hf_attn_layer.q_proj.weight.data.copy_(q_proj)
    hf_attn_layer.q_proj.bias.data.copy_(q_proj_bias)
    hf_attn_layer.k_proj.weight.data.copy_(k_proj)
    hf_attn_layer.k_proj.bias.data.copy_(k_proj_bias)
    hf_attn_layer.v_proj.weight.data.copy_(v_proj)
    hf_attn_layer.v_proj.bias.data.copy_(v_proj_bias)
    hf_attn_layer.out_proj.weight.data.copy_(pt_attn_layer.out_proj.weight.data)
    hf_attn_layer.out_proj.bias.data.copy_(pt_attn_layer.out_proj.bias.data)

def copy_mlp(hf_mlp, pt_mlp):
    copy_linear(hf_mlp.fc1, pt_mlp.c_fc)
    copy_linear(hf_mlp.fc2, pt_mlp.c_proj)

def copy_layer(hf_layer, pt_layer):
    copy_linear(hf_layer.layer_norm1, pt_layer.ln_1)
    copy_linear(hf_layer.layer_norm2, pt_layer.ln_2)
    copy_mlp(hf_layer.mlp, pt_layer.mlp)
    copy_attn_layer(hf_layer.self_attn, pt_layer.attn)

def copy_layers(hf_layers, pt_layers):
    for hf_layer, pt_layer in zip(hf_layers, pt_layers):
        copy_layer(hf_layer, pt_layer)

def copy_encoder(hf_encoder, pt_model):
    # For text encoder
    hf_encoder.embeddings.token_embedding.weight.data.copy_(pt_model.token_embedding.weight.data)
    hf_encoder.embeddings.position_embedding.weight.data.copy_(pt_model.positional_embedding)
    copy_linear(hf_encoder.final_layer_norm, pt_model.ln_final)
    copy_layers(hf_encoder.encoder.layers, pt_model.transformer.resblocks)

def copy_text_model_and_projection(hf_model, pt_model):
    hf_model.text_projection.weight.data.copy_(pt_model.text_projection.data.T.contiguous())
    copy_encoder(hf_model.text_model, pt_model)

def copy_fusion_mlp(hf_mlp, pt_mlp):
    """Copy a fusion MLP (nn.Sequential with Linear-ReLU-Linear)"""
    for lhf, lpt in zip(hf_mlp, pt_mlp):
        if isinstance(lhf, torch.nn.Linear) and isinstance(lpt, torch.nn.Linear):
            copy_linear(lhf, lpt)

def copy_all_fusion_mlps(hf_mlps, pt_mlps):
    assert len(hf_mlps) == len(pt_mlps), f"Intermediate fusion mlps len mismatch: {len(hf_mlps)} vs {len(pt_mlps)}"
    for lhf, lpt in zip(hf_mlps, pt_mlps):
        copy_fusion_mlp(lhf, lpt)

def copy_vision_embeddings(hf_embeddings, pt_visual, config):
    # Patch embedding
    hf_embeddings.patch_embedding.weight.data.copy_(pt_visual.conv1.weight.data)
    # CLS token
    hf_embeddings.class_embedding.data.copy_(pt_visual.class_embedding.data)
    # Positional embedding (note: shape is (N+1+num_reg, D) for REG, or (N+1, D) vanilla)
    assert hf_embeddings.position_embedding.weight.shape == pt_visual.positional_embedding.shape, f"Positional embedding shape mismatch: {hf_embeddings.position_embedding.weight.shape} vs {pt_visual.positional_embedding.shape}"
    hf_embeddings.position_embedding.weight.data.copy_(pt_visual.positional_embedding.data)
    # Register tokens
    if getattr(config, "use_register_tokens", False):
        assert hasattr(hf_embeddings, "register_embeddings"), "Missing register_embeddings in HF model"
        assert hasattr(pt_visual, "register_tokens"), "Missing register_tokens in PT model"
        assert hf_embeddings.register_embeddings.shape == pt_visual.register_tokens.shape, f"Register tokens shape mismatch: {hf_embeddings.register_embeddings.shape} vs {pt_visual.register_tokens.shape}"
        hf_embeddings.register_embeddings.data.copy_(pt_visual.register_tokens.data)

def copy_CLIPREGVisionModel_and_projection(hf_model, pt_model, config):
    # Visual projection
    hf_model.visual_projection.weight.data.copy_(pt_model.visual.proj.data.T.contiguous())
    # Pre and post layer norms
    copy_linear(hf_model.vision_model.pre_layrnorm, pt_model.visual.ln_pre)
    copy_linear(hf_model.vision_model.post_layernorm, pt_model.visual.ln_post)
    # Embeddings (including register tokens)
    copy_vision_embeddings(hf_model.vision_model.embeddings, pt_model.visual, config)
    # Encoder (ViT blocks)
    copy_layers(hf_model.vision_model.encoder.layers, pt_model.visual.transformer.resblocks)
    # --- Copy fusion MLPs ---
    if getattr(config, "use_fusion_mlp", False):
        # Intermediate fusion MLPs
        if hasattr(hf_model.vision_model, "intermediate_fusion_mlps") and hasattr(pt_model.visual.transformer, "intermediate_fusion_mlps"):
            copy_all_fusion_mlps(hf_model.vision_model.intermediate_fusion_mlps, pt_model.visual.transformer.intermediate_fusion_mlps)
        # Final fusion MLP
        if hasattr(hf_model.vision_model, "fusion_mlp") and hasattr(pt_model.visual, "fusion_mlp"):
            copy_fusion_mlp(hf_model.vision_model.fusion_mlp, pt_model.visual.fusion_mlp)

@torch.no_grad()
def convert_clipreg_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None):
    if config_path is not None:
        config = CLIPConfig.from_pretrained(config_path)
    else:
        raise RuntimeError("A config_path must be provided for CLIPREG model conversion.")

    config.vision_config.use_register_tokens = True
    config.vision_config.num_register_tokens = 4
    config.vision_config.use_fusion_mlp = True
    config.vision_config.gate_start_layer = 14
    
   
    hf_model = CLIPModel(config).eval()
    print("use_register_tokens:", getattr(config.vision_config, "use_register_tokens", None))
    print("num_register_tokens:", getattr(config.vision_config, "num_register_tokens", None))
    print("use_fusion_mlp:", getattr(config.vision_config, "use_fusion_mlp", None))
    print("gate_start_layer:", getattr(config.vision_config, "gate_start_layer", None))
    print("Embedding class:", type(hf_model.vision_model.embeddings))
    print("HF position embedding shape:", hf_model.vision_model.embeddings.position_embedding.weight.shape)

    pt_model, _ = load(checkpoint_path, device="cpu", jit=False)
    pt_model = pt_model.eval()

    print("gate_start_layer =", config.vision_config.gate_start_layer)
    print("num_fusion_mlps in checkpoint:", len(pt_model.visual.transformer.intermediate_fusion_mlps))
    print("num_hidden_layers:", config.vision_config.num_hidden_layers)
    print("Expected number of gates:", config.vision_config.num_hidden_layers - config.vision_config.gate_start_layer + 1)


    # TEXT: normal CLIP model
    copy_text_model_and_projection(hf_model, pt_model)
    # VISION: with register tokens/fusion MLP
    copy_CLIPREGVisionModel_and_projection(hf_model, pt_model, config.vision_config)

    # logit_scale
    hf_model.logit_scale.data.copy_(pt_model.logit_scale.data)

    # Test output consistency
    input_ids = torch.tensor(
        [
            [config.text_config.bos_token_id]
            + list(range(3, 77))
            + [config.text_config.eos_token_id]
            + [config.text_config.pad_token_id]
        ]
    )
    pixel_values = torch.randn(1, 3, config.vision_config.image_size, config.vision_config.image_size)
    hf_outputs = hf_model(input_ids=input_ids, pixel_values=pixel_values, return_dict=True)
    hf_logits_per_image = hf_outputs.logits_per_image
    hf_logits_per_text = hf_outputs.logits_per_text
    pt_logits_per_image, pt_logits_per_text = pt_model(pixel_values, input_ids)
    assert torch.allclose(hf_logits_per_image, pt_logits_per_image, atol=1e-3), "logits_per_image mismatch"
    assert torch.allclose(hf_logits_per_text, pt_logits_per_text, atol=1e-3), "logits_per_text mismatch"

    hf_model.save_pretrained(pytorch_dump_folder_path)
    print(f"Model saved to {pytorch_dump_folder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", required=True, type=str, help="Path to OpenAI checkpoint (.pt or .pth)")
    parser.add_argument("--config_path", required=True, type=str, help="Path to HF config.json of model to convert")
    args = parser.parse_args()

    convert_clipreg_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)