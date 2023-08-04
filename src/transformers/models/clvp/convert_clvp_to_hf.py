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

# This is the official weights of clvp and it is downloaded from
# https://huggingface.co/jbetker/tortoise-tts-v2/blob/main/.models/clvp2.pth
weights = torch.load("./clvp2.pth")
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
clvp.load_state_dict(weights, strict=True)
clvp.eval()

# Define weights for out hf model
model_weights = {}
for i in range(cfg.text_config.num_hidden_layers):
    ## text model
    model_weights.update(
        {
            f"text_model.encoder.layers.{i}.self_attn.k_proj.weight": weights[
                f"text_transformer.transformer.attn_layers.layers.{2*i}.1.wrap.to_k.weight"
            ],
            f"text_model.encoder.layers.{i}.self_attn.v_proj.weight": weights[
                f"text_transformer.transformer.attn_layers.layers.{2*i}.1.wrap.to_v.weight"
            ],
            f"text_model.encoder.layers.{i}.self_attn.q_proj.weight": weights[
                f"text_transformer.transformer.attn_layers.layers.{2*i}.1.wrap.to_q.weight"
            ],
            f"text_model.encoder.layers.{i}.self_attn.out_proj.weight": weights[
                f"text_transformer.transformer.attn_layers.layers.{2*i}.1.wrap.to_out.weight"
            ],
            f"text_model.encoder.layers.{i}.self_attn.out_proj.bias": weights[
                f"text_transformer.transformer.attn_layers.layers.{2*i}.1.wrap.to_out.bias"
            ],
            f"text_model.encoder.layers.{i}.mlp.fc1.proj.weight": weights[
                f"text_transformer.transformer.attn_layers.layers.{(2*i)+1}.1.wrap.net.0.proj.weight"
            ],
            f"text_model.encoder.layers.{i}.mlp.fc1.proj.bias": weights[
                f"text_transformer.transformer.attn_layers.layers.{(2*i)+1}.1.wrap.net.0.proj.bias"
            ],
            f"text_model.encoder.layers.{i}.mlp.fc2.weight": weights[
                f"text_transformer.transformer.attn_layers.layers.{(2*i)+1}.1.wrap.net.3.weight"
            ],
            f"text_model.encoder.layers.{i}.mlp.fc2.bias": weights[
                f"text_transformer.transformer.attn_layers.layers.{(2*i)+1}.1.wrap.net.3.bias"
            ],
            f"text_model.encoder.layers.{i}.pre_branch_norm1.gain": weights[
                f"text_transformer.transformer.attn_layers.layers.{2*i}.0.0.g"
            ],
            f"text_model.encoder.layers.{i}.pre_branch_norm2.gain": weights[
                f"text_transformer.transformer.attn_layers.layers.{(2*i)+1}.0.0.g"
            ],
        }
    )

for i in range(cfg.speech_config.num_hidden_layers):
    ## speech model
    model_weights.update(
        {
            f"speech_model.encoder.layers.{i}.self_attn.k_proj.weight": weights[
                f"speech_transformer.transformer.attn_layers.layers.{2*i}.1.wrap.to_k.weight"
            ],
            f"speech_model.encoder.layers.{i}.self_attn.v_proj.weight": weights[
                f"speech_transformer.transformer.attn_layers.layers.{2*i}.1.wrap.to_v.weight"
            ],
            f"speech_model.encoder.layers.{i}.self_attn.q_proj.weight": weights[
                f"speech_transformer.transformer.attn_layers.layers.{2*i}.1.wrap.to_q.weight"
            ],
            f"speech_model.encoder.layers.{i}.self_attn.out_proj.weight": weights[
                f"speech_transformer.transformer.attn_layers.layers.{2*i}.1.wrap.to_out.weight"
            ],
            f"speech_model.encoder.layers.{i}.self_attn.out_proj.bias": weights[
                f"speech_transformer.transformer.attn_layers.layers.{2*i}.1.wrap.to_out.bias"
            ],
            f"speech_model.encoder.layers.{i}.mlp.fc1.proj.weight": weights[
                f"speech_transformer.transformer.attn_layers.layers.{(2*i)+1}.1.wrap.net.0.proj.weight"
            ],
            f"speech_model.encoder.layers.{i}.mlp.fc1.proj.bias": weights[
                f"speech_transformer.transformer.attn_layers.layers.{(2*i)+1}.1.wrap.net.0.proj.bias"
            ],
            f"speech_model.encoder.layers.{i}.mlp.fc2.weight": weights[
                f"speech_transformer.transformer.attn_layers.layers.{(2*i)+1}.1.wrap.net.3.weight"
            ],
            f"speech_model.encoder.layers.{i}.mlp.fc2.bias": weights[
                f"speech_transformer.transformer.attn_layers.layers.{(2*i)+1}.1.wrap.net.3.bias"
            ],
            f"speech_model.encoder.layers.{i}.pre_branch_norm1.gain": weights[
                f"speech_transformer.transformer.attn_layers.layers.{2*i}.0.0.g"
            ],
            f"speech_model.encoder.layers.{i}.pre_branch_norm2.gain": weights[
                f"speech_transformer.transformer.attn_layers.layers.{(2*i)+1}.0.0.g"
            ],
        }
    )

## for text
# inv_freq(rotary embedding)
model_weights["text_model.encoder.rotary_pos_emb.inv_freq"] = weights[
    "text_transformer.transformer.attn_layers.rotary_pos_emb.inv_freq"
]
# norm
model_weights["text_model.final_layer_norm.weight"] = weights["text_transformer.transformer.norm.weight"]
model_weights["text_model.final_layer_norm.bias"] = weights["text_transformer.transformer.norm.bias"]
# word embedding
model_weights["text_model.token_embedding.weight"] = weights["text_emb.weight"]
# projection
model_weights["text_projection.weight"] = weights["to_text_latent.weight"]


model_weights["logit_scale"] = weights["temperature"]


## for speech
# inv_freq(rotary embedding)
model_weights["speech_model.encoder.rotary_pos_emb.inv_freq"] = weights[
    "speech_transformer.transformer.attn_layers.rotary_pos_emb.inv_freq"
]
# norm
model_weights["speech_model.final_layer_norm.weight"] = weights["speech_transformer.transformer.norm.weight"]
model_weights["speech_model.final_layer_norm.bias"] = weights["speech_transformer.transformer.norm.bias"]
# word embedding
model_weights["speech_model.token_embedding.weight"] = weights["speech_emb.weight"]
# projection
model_weights["speech_projection.weight"] = weights["to_speech_latent.weight"]


# Load hf model weights
model.load_state_dict(model_weights, strict=True)
model.eval()

# define inputs
ipt1 = torch.randint(low=0, high=255, size=[101, 15])
ipt2 = torch.randint(low=0, high=255, size=[1, 23])

# get both model outputs
model_opt1 = model(ipt1, ipt2, return_loss=False)
model_opt2 = clvp(ipt1, ipt2, return_loss=False)

# check the logits
print(f"Logits are same or not : {torch.allclose(model_opt1.logits_per_speech, model_opt2, atol=1e-4, rtol=1e-4)}")
