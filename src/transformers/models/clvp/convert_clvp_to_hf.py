# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
""" PyTorch CLVP model checkpoint conversion."""


import torch
from tortoise.models.clvp import CLVP

from transformers import CLVPConfig, CLVPModel


cfg = CLVPConfig.from_pretrained("susnato/clvp_dev")
model = CLVPModel(cfg)

# This is the official weights of clvp and autoregressive model and it is downloaded from
# https://huggingface.co/jbetker/tortoise-tts-v2/blob/main/.models/clvp2.pth
# https://huggingface.co/jbetker/tortoise-tts-v2/blob/main/.models/autoregressive.pth
weights_ar = torch.load("./autoregressive.pth")
weights_clvp = torch.load("./clvp2.pth")

# init the clvp model
clvp = (
    CLVP(
        dim_text=768,
        dim_speech=768,
        dim_latent=768,
        text_enc_depth=20,
        speech_enc_depth=20,
        text_heads=12,
        speech_heads=12,
        num_text_tokens=256,
        num_speech_tokens=8192,
        text_seq_len=350,
        speech_seq_len=430,
        use_xformers=True,
    )
    .cpu()
    .eval()
)
clvp.load_state_dict(weights_clvp, strict=True)
clvp.eval()

# Define weights for our hf model

weights_ar = torch.load("./autoregressive.pth")
weights_clvp = torch.load("./clvp2.pth")

model_weights = {}

# AutoRegressive Model weights
for i in range(cfg.autoregressive_config.n_layer):
    model_weights[f"speech_model.autoregressive_model.transformer.h.{i}.attn.c_attn.bias"] = weights_ar[
        f"gpt.h.{i}.attn.c_attn.bias"]
    model_weights[f"speech_model.autoregressive_model.transformer.h.{i}.attn.c_attn.weight"] = weights_ar[
        f"gpt.h.{i}.attn.c_attn.weight"]
    model_weights[f"speech_model.autoregressive_model.transformer.h.{i}.attn.c_proj.bias"] = weights_ar[
        f"gpt.h.{i}.attn.c_proj.bias"]
    model_weights[f"speech_model.autoregressive_model.transformer.h.{i}.attn.c_proj.weight"] = weights_ar[
        f"gpt.h.{i}.attn.c_proj.weight"]
    model_weights[f"speech_model.autoregressive_model.transformer.h.{i}.ln_1.bias"] = weights_ar[f"gpt.h.{i}.ln_1.bias"]
    model_weights[f"speech_model.autoregressive_model.transformer.h.{i}.ln_1.weight"] = weights_ar[
        f"gpt.h.{i}.ln_1.weight"]
    model_weights[f"speech_model.autoregressive_model.transformer.h.{i}.ln_2.bias"] = weights_ar[f"gpt.h.{i}.ln_2.bias"]
    model_weights[f"speech_model.autoregressive_model.transformer.h.{i}.ln_2.weight"] = weights_ar[
        f"gpt.h.{i}.ln_2.weight"]
    model_weights[f"speech_model.autoregressive_model.transformer.h.{i}.mlp.c_fc.bias"] = weights_ar[
        f"gpt.h.{i}.mlp.c_fc.bias"]
    model_weights[f"speech_model.autoregressive_model.transformer.h.{i}.mlp.c_fc.weight"] = weights_ar[
        f"gpt.h.{i}.mlp.c_fc.weight"]
    model_weights[f"speech_model.autoregressive_model.transformer.h.{i}.mlp.c_proj.bias"] = weights_ar[
        f"gpt.h.{i}.mlp.c_proj.bias"]
    model_weights[f"speech_model.autoregressive_model.transformer.h.{i}.mlp.c_proj.weight"] = weights_ar[
        f"gpt.h.{i}.mlp.c_proj.weight"]

model_weights['speech_model.autoregressive_model.lm_head.bias'] = weights_ar["mel_head.bias"]
model_weights['speech_model.autoregressive_model.lm_head.weight'] = weights_ar["mel_head.weight"]
model_weights['speech_model.autoregressive_model.transformer.ln_f.bias'] = weights_ar["gpt.ln_f.bias"]
model_weights['speech_model.autoregressive_model.transformer.ln_f.weight'] = weights_ar["gpt.ln_f.weight"]
model_weights['speech_model.autoregressive_model.transformer.wpe.weight'] = weights_ar["mel_pos_embedding.emb.weight"]
model_weights['speech_model.autoregressive_model.transformer.wte.weight'] = weights_ar["mel_embedding.weight"]

# Conditioning Encoder Model weights

model_weights['speech_model.conditioning_encoder.mel_conv.bias'] = weights_ar["conditioning_encoder.init.bias"]
model_weights['speech_model.conditioning_encoder.mel_conv.weight'] = weights_ar["conditioning_encoder.init.weight"]
model_weights['speech_model.conditioning_encoder.text_position_embedding.weight'] = weights_ar[
    "text_pos_embedding.emb.weight"]
model_weights['speech_model.conditioning_encoder.text_token_embedding.weight'] = weights_ar["text_embedding.weight"]

dim = 1024
n_heads = 16
for i in range(6):
    w1, w2, w3 = weights_ar[f"conditioning_encoder.attn.{i}.qkv.weight"].squeeze(-1).split(split_size=dim, dim=0)
    b1, b2, b3 = weights_ar[f"conditioning_encoder.attn.{i}.qkv.bias"].split(split_size=dim, dim=0)

    model_weights[f"speech_model.conditioning_encoder.mel_attn_blocks.{i}.q.weight"] = torch.concatenate(
        [
            w1[0 * (dim // n_heads): 1 * (dim // n_heads), :],
            w1[3 * (dim // n_heads): 4 * (dim // n_heads), :],
            w1[6 * (dim // n_heads): 7 * (dim // n_heads), :],
            w1[9 * (dim // n_heads): 10 * (dim // n_heads), :],
            w1[12 * (dim // n_heads): 13 * (dim // n_heads), :],
            w1[15 * (dim // n_heads): 16 * (dim // n_heads), :],
            w2[2 * (dim // n_heads): 3 * (dim // n_heads), :],
            w2[5 * (dim // n_heads): 6 * (dim // n_heads), :],
            w2[8 * (dim // n_heads): 9 * (dim // n_heads), :],
            w2[11 * (dim // n_heads): 12 * (dim // n_heads), :],
            w2[14 * (dim // n_heads): 15 * (dim // n_heads), :],
            w3[1 * (dim // n_heads): 2 * (dim // n_heads), :],
            w3[4 * (dim // n_heads): 5 * (dim // n_heads), :],
            w3[7 * (dim // n_heads): 8 * (dim // n_heads), :],
            w3[10 * (dim // n_heads): 11 * (dim // n_heads), :],
            w3[13 * (dim // n_heads): 14 * (dim // n_heads), :],
        ],
        axis=0,
    )
    model_weights[f"speech_model.conditioning_encoder.mel_attn_blocks.{i}.q.bias"] = torch.concatenate(
        [
            b1[0 * (dim // n_heads): 1 * (dim // n_heads)],
            b1[3 * (dim // n_heads): 4 * (dim // n_heads)],
            b1[6 * (dim // n_heads): 7 * (dim // n_heads)],
            b1[9 * (dim // n_heads): 10 * (dim // n_heads)],
            b1[12 * (dim // n_heads): 13 * (dim // n_heads)],
            b1[15 * (dim // n_heads): 16 * (dim // n_heads)],
            b2[2 * (dim // n_heads): 3 * (dim // n_heads)],
            b2[5 * (dim // n_heads): 6 * (dim // n_heads)],
            b2[8 * (dim // n_heads): 9 * (dim // n_heads)],
            b2[11 * (dim // n_heads): 12 * (dim // n_heads)],
            b2[14 * (dim // n_heads): 15 * (dim // n_heads)],
            b3[1 * (dim // n_heads): 2 * (dim // n_heads)],
            b3[4 * (dim // n_heads): 5 * (dim // n_heads)],
            b3[7 * (dim // n_heads): 8 * (dim // n_heads)],
            b3[10 * (dim // n_heads): 11 * (dim // n_heads)],
            b3[13 * (dim // n_heads): 14 * (dim // n_heads)],
        ],
        axis=0,
    )

    model_weights[f"speech_model.conditioning_encoder.mel_attn_blocks.{i}.k.weight"] = torch.concatenate(
        [
            w1[1 * (dim // n_heads): 2 * (dim // n_heads), :],
            w1[4 * (dim // n_heads): 5 * (dim // n_heads), :],
            w1[7 * (dim // n_heads): 8 * (dim // n_heads), :],
            w1[10 * (dim // n_heads): 11 * (dim // n_heads), :],
            w1[13 * (dim // n_heads): 14 * (dim // n_heads), :],
            w2[0 * (dim // n_heads): 1 * (dim // n_heads), :],
            w2[3 * (dim // n_heads): 4 * (dim // n_heads), :],
            w2[6 * (dim // n_heads): 7 * (dim // n_heads), :],
            w2[9 * (dim // n_heads): 10 * (dim // n_heads), :],
            w2[12 * (dim // n_heads): 13 * (dim // n_heads), :],
            w2[15 * (dim // n_heads): 16 * (dim // n_heads), :],
            w3[2 * (dim // n_heads): 3 * (dim // n_heads), :],
            w3[5 * (dim // n_heads): 6 * (dim // n_heads), :],
            w3[8 * (dim // n_heads): 9 * (dim // n_heads), :],
            w3[11 * (dim // n_heads): 12 * (dim // n_heads), :],
            w3[14 * (dim // n_heads): 15 * (dim // n_heads), :],
        ],
        axis=0,
    )
    model_weights[f"speech_model.conditioning_encoder.mel_attn_blocks.{i}.k.bias"] = torch.concatenate(
        [
            b1[1 * (dim // n_heads): 2 * (dim // n_heads)],
            b1[4 * (dim // n_heads): 5 * (dim // n_heads)],
            b1[7 * (dim // n_heads): 8 * (dim // n_heads)],
            b1[10 * (dim // n_heads): 11 * (dim // n_heads)],
            b1[13 * (dim // n_heads): 14 * (dim // n_heads)],
            b2[0 * (dim // n_heads): 1 * (dim // n_heads)],
            b2[3 * (dim // n_heads): 4 * (dim // n_heads)],
            b2[6 * (dim // n_heads): 7 * (dim // n_heads)],
            b2[9 * (dim // n_heads): 10 * (dim // n_heads)],
            b2[12 * (dim // n_heads): 13 * (dim // n_heads)],
            b2[15 * (dim // n_heads): 16 * (dim // n_heads)],
            b3[2 * (dim // n_heads): 3 * (dim // n_heads)],
            b3[5 * (dim // n_heads): 6 * (dim // n_heads)],
            b3[8 * (dim // n_heads): 9 * (dim // n_heads)],
            b3[11 * (dim // n_heads): 12 * (dim // n_heads)],
            b3[14 * (dim // n_heads): 15 * (dim // n_heads)],
        ],
        axis=0,
    )

    model_weights[f"speech_model.conditioning_encoder.mel_attn_blocks.{i}.v.weight"] = torch.concatenate(
        [
            w1[2 * (dim // n_heads): 3 * (dim // n_heads), :],
            w1[5 * (dim // n_heads): 6 * (dim // n_heads), :],
            w1[8 * (dim // n_heads): 9 * (dim // n_heads), :],
            w1[11 * (dim // n_heads): 12 * (dim // n_heads), :],
            w1[14 * (dim // n_heads): 15 * (dim // n_heads), :],
            w2[1 * (dim // n_heads): 2 * (dim // n_heads), :],
            w2[4 * (dim // n_heads): 5 * (dim // n_heads), :],
            w2[7 * (dim // n_heads): 8 * (dim // n_heads), :],
            w2[10 * (dim // n_heads): 11 * (dim // n_heads), :],
            w2[13 * (dim // n_heads): 14 * (dim // n_heads), :],
            w3[0 * (dim // n_heads): 1 * (dim // n_heads), :],
            w3[3 * (dim // n_heads): 4 * (dim // n_heads), :],
            w3[6 * (dim // n_heads): 7 * (dim // n_heads), :],
            w3[9 * (dim // n_heads): 10 * (dim // n_heads), :],
            w3[12 * (dim // n_heads): 13 * (dim // n_heads), :],
            w3[15 * (dim // n_heads): 16 * (dim // n_heads), :],
        ],
        axis=0,
    )
    model_weights[f"speech_model.conditioning_encoder.mel_attn_blocks.{i}.v.bias"] = torch.concatenate(
        [
            b1[2 * (dim // n_heads): 3 * (dim // n_heads)],
            b1[5 * (dim // n_heads): 6 * (dim // n_heads)],
            b1[8 * (dim // n_heads): 9 * (dim // n_heads)],
            b1[11 * (dim // n_heads): 12 * (dim // n_heads)],
            b1[14 * (dim // n_heads): 15 * (dim // n_heads)],
            b2[1 * (dim // n_heads): 2 * (dim // n_heads)],
            b2[4 * (dim // n_heads): 5 * (dim // n_heads)],
            b2[7 * (dim // n_heads): 8 * (dim // n_heads)],
            b2[10 * (dim // n_heads): 11 * (dim // n_heads)],
            b2[13 * (dim // n_heads): 14 * (dim // n_heads)],
            b3[0 * (dim // n_heads): 1 * (dim // n_heads)],
            b3[3 * (dim // n_heads): 4 * (dim // n_heads)],
            b3[6 * (dim // n_heads): 7 * (dim // n_heads)],
            b3[9 * (dim // n_heads): 10 * (dim // n_heads)],
            b3[12 * (dim // n_heads): 13 * (dim // n_heads)],
            b3[15 * (dim // n_heads): 16 * (dim // n_heads)],
        ],
        axis=0,
    )
    model_weights[f"speech_model.conditioning_encoder.mel_attn_blocks.{i}.o.weight"] = weights_ar[
        f"conditioning_encoder.attn.{i}.proj_out.weight"].squeeze(-1)
    model_weights[f"speech_model.conditioning_encoder.mel_attn_blocks.{i}.o.bias"] = weights_ar[
        f"conditioning_encoder.attn.{i}.proj_out.bias"].squeeze(-1)

# Transformer Encoder Models weights

for i in range(cfg.text_config.num_hidden_layers):
    ## text model
    model_weights.update(
        {
            f"text_model.text_transformer.encoder.layers.{i}.self_attn.k_proj.weight": weights_clvp[
                f"text_transformer.transformer.attn_layers.layers.{2 * i}.1.wrap.to_k.weight"
            ],
            f"text_model.text_transformer.encoder.layers.{i}.self_attn.v_proj.weight": weights_clvp[
                f"text_transformer.transformer.attn_layers.layers.{2 * i}.1.wrap.to_v.weight"
            ],
            f"text_model.text_transformer.encoder.layers.{i}.self_attn.q_proj.weight": weights_clvp[
                f"text_transformer.transformer.attn_layers.layers.{2 * i}.1.wrap.to_q.weight"
            ],
            f"text_model.text_transformer.encoder.layers.{i}.self_attn.out_proj.weight": weights_clvp[
                f"text_transformer.transformer.attn_layers.layers.{2 * i}.1.wrap.to_out.weight"
            ],
            f"text_model.text_transformer.encoder.layers.{i}.self_attn.out_proj.bias": weights_clvp[
                f"text_transformer.transformer.attn_layers.layers.{2 * i}.1.wrap.to_out.bias"
            ],
            f"text_model.text_transformer.encoder.layers.{i}.mlp.fc1.proj.weight": weights_clvp[
                f"text_transformer.transformer.attn_layers.layers.{(2 * i) + 1}.1.wrap.net.0.proj.weight"
            ],
            f"text_model.text_transformer.encoder.layers.{i}.mlp.fc1.proj.bias": weights_clvp[
                f"text_transformer.transformer.attn_layers.layers.{(2 * i) + 1}.1.wrap.net.0.proj.bias"
            ],
            f"text_model.text_transformer.encoder.layers.{i}.mlp.fc2.weight": weights_clvp[
                f"text_transformer.transformer.attn_layers.layers.{(2 * i) + 1}.1.wrap.net.3.weight"
            ],
            f"text_model.text_transformer.encoder.layers.{i}.mlp.fc2.bias": weights_clvp[
                f"text_transformer.transformer.attn_layers.layers.{(2 * i) + 1}.1.wrap.net.3.bias"
            ],
            f"text_model.text_transformer.encoder.layers.{i}.pre_branch_norm1.gain": weights_clvp[
                f"text_transformer.transformer.attn_layers.layers.{2 * i}.0.0.g"
            ],
            f"text_model.text_transformer.encoder.layers.{i}.pre_branch_norm2.gain": weights_clvp[
                f"text_transformer.transformer.attn_layers.layers.{(2 * i) + 1}.0.0.g"
            ],
        }
    )

for i in range(cfg.speech_config.num_hidden_layers):
    ## speech model
    model_weights.update(
        {
            f"speech_model.speech_transformer.encoder.layers.{i}.self_attn.k_proj.weight": weights_clvp[
                f"speech_transformer.transformer.attn_layers.layers.{2 * i}.1.wrap.to_k.weight"
            ],
            f"speech_model.speech_transformer.encoder.layers.{i}.self_attn.v_proj.weight": weights_clvp[
                f"speech_transformer.transformer.attn_layers.layers.{2 * i}.1.wrap.to_v.weight"
            ],
            f"speech_model.speech_transformer.encoder.layers.{i}.self_attn.q_proj.weight": weights_clvp[
                f"speech_transformer.transformer.attn_layers.layers.{2 * i}.1.wrap.to_q.weight"
            ],
            f"speech_model.speech_transformer.encoder.layers.{i}.self_attn.out_proj.weight": weights_clvp[
                f"speech_transformer.transformer.attn_layers.layers.{2 * i}.1.wrap.to_out.weight"
            ],
            f"speech_model.speech_transformer.encoder.layers.{i}.self_attn.out_proj.bias": weights_clvp[
                f"speech_transformer.transformer.attn_layers.layers.{2 * i}.1.wrap.to_out.bias"
            ],
            f"speech_model.speech_transformer.encoder.layers.{i}.mlp.fc1.proj.weight": weights_clvp[
                f"speech_transformer.transformer.attn_layers.layers.{(2 * i) + 1}.1.wrap.net.0.proj.weight"
            ],
            f"speech_model.speech_transformer.encoder.layers.{i}.mlp.fc1.proj.bias": weights_clvp[
                f"speech_transformer.transformer.attn_layers.layers.{(2 * i) + 1}.1.wrap.net.0.proj.bias"
            ],
            f"speech_model.speech_transformer.encoder.layers.{i}.mlp.fc2.weight": weights_clvp[
                f"speech_transformer.transformer.attn_layers.layers.{(2 * i) + 1}.1.wrap.net.3.weight"
            ],
            f"speech_model.speech_transformer.encoder.layers.{i}.mlp.fc2.bias": weights_clvp[
                f"speech_transformer.transformer.attn_layers.layers.{(2 * i) + 1}.1.wrap.net.3.bias"
            ],
            f"speech_model.speech_transformer.encoder.layers.{i}.pre_branch_norm1.gain": weights_clvp[
                f"speech_transformer.transformer.attn_layers.layers.{2 * i}.0.0.g"
            ],
            f"speech_model.speech_transformer.encoder.layers.{i}.pre_branch_norm2.gain": weights_clvp[
                f"speech_transformer.transformer.attn_layers.layers.{(2 * i) + 1}.0.0.g"
            ],
        }
    )

## for text
# inv_freq(rotary embedding)
model_weights["text_model.text_transformer.encoder.rotary_pos_emb.inv_freq"] = weights_clvp[
    "text_transformer.transformer.attn_layers.rotary_pos_emb.inv_freq"
]
# norm
model_weights["text_model.text_transformer.final_layer_norm.weight"] = weights_clvp[
    "text_transformer.transformer.norm.weight"]
model_weights["text_model.text_transformer.final_layer_norm.bias"] = weights_clvp[
    "text_transformer.transformer.norm.bias"]
# word embedding
model_weights["text_model.text_transformer.token_embedding.weight"] = weights_clvp["text_emb.weight"]
# projection
model_weights["text_model.text_projection.weight"] = weights_clvp["to_text_latent.weight"]

model_weights["logit_scale"] = weights_clvp["temperature"]

## for speech
# inv_freq(rotary embedding)
model_weights["speech_model.speech_transformer.encoder.rotary_pos_emb.inv_freq"] = weights_clvp[
    "speech_transformer.transformer.attn_layers.rotary_pos_emb.inv_freq"
]
# norm
model_weights["speech_model.speech_transformer.final_layer_norm.weight"] = weights_clvp[
    "speech_transformer.transformer.norm.weight"]
model_weights["speech_model.speech_transformer.final_layer_norm.bias"] = weights_clvp[
    "speech_transformer.transformer.norm.bias"]
# word embedding
model_weights["speech_model.speech_transformer.token_embedding.weight"] = weights_clvp["speech_emb.weight"]
# projection
model_weights["speech_model.speech_projection.weight"] = weights_clvp["to_speech_latent.weight"]


# Load hf model weights
print(model.load_state_dict(model_weights, strict=True))
model.eval()

print("Weights Loaded successfully!")

# define inputs
ipt1 = torch.randint(low=0, high=255, size=[101, 15])
ipt2 = torch.randint(low=0, high=255, size=[1, 23])

# # get both model outputs
# model_opt1 = model(ipt1, ipt2, return_loss=False)
# model_opt2 = clvp(ipt1, ipt2, return_loss=False)
#
# # check the logits
# print(f"Logits are same or not : {torch.allclose(model_opt1.logits_per_speech, model_opt2, atol=1e-4, rtol=1e-4)}")
