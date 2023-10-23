# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

""" File for loading the Pop2Piano model weights from the official repository and to show how tokenizer vocab was
 constructed"""

import json

import torch

from transformers import Pop2PianoConfig, Pop2PianoForConditionalGeneration


########################## MODEL WEIGHTS ##########################

# This weights were downloaded from the official pop2piano repository
# https://huggingface.co/sweetcocoa/pop2piano/blob/main/model-1999-val_0.67311615.ckpt
official_weights = torch.load("./model-1999-val_0.67311615.ckpt")
state_dict = {}


# load the config and init the model
cfg = Pop2PianoConfig.from_pretrained("sweetcocoa/pop2piano")
model = Pop2PianoForConditionalGeneration(cfg)


# load relative attention bias
state_dict["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = official_weights["state_dict"][
    "transformer.encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
]
state_dict["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = official_weights["state_dict"][
    "transformer.decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
]

# load embed tokens and final layer norm for both encoder and decoder
state_dict["encoder.embed_tokens.weight"] = official_weights["state_dict"]["transformer.encoder.embed_tokens.weight"]
state_dict["decoder.embed_tokens.weight"] = official_weights["state_dict"]["transformer.decoder.embed_tokens.weight"]

state_dict["encoder.final_layer_norm.weight"] = official_weights["state_dict"][
    "transformer.encoder.final_layer_norm.weight"
]
state_dict["decoder.final_layer_norm.weight"] = official_weights["state_dict"][
    "transformer.decoder.final_layer_norm.weight"
]

# load lm_head, mel_conditioner.emb and shared
state_dict["lm_head.weight"] = official_weights["state_dict"]["transformer.lm_head.weight"]
state_dict["mel_conditioner.embedding.weight"] = official_weights["state_dict"]["mel_conditioner.embedding.weight"]
state_dict["shared.weight"] = official_weights["state_dict"]["transformer.shared.weight"]

# load each encoder blocks
for i in range(cfg.num_layers):
    # layer 0
    state_dict[f"encoder.block.{i}.layer.0.SelfAttention.q.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.SelfAttention.q.weight"
    ]
    state_dict[f"encoder.block.{i}.layer.0.SelfAttention.k.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.SelfAttention.k.weight"
    ]
    state_dict[f"encoder.block.{i}.layer.0.SelfAttention.v.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.SelfAttention.v.weight"
    ]
    state_dict[f"encoder.block.{i}.layer.0.SelfAttention.o.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.SelfAttention.o.weight"
    ]
    state_dict[f"encoder.block.{i}.layer.0.layer_norm.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.0.layer_norm.weight"
    ]

    # layer 1
    state_dict[f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight"
    ]
    state_dict[f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight"
    ]
    state_dict[f"encoder.block.{i}.layer.1.DenseReluDense.wo.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.1.DenseReluDense.wo.weight"
    ]
    state_dict[f"encoder.block.{i}.layer.1.layer_norm.weight"] = official_weights["state_dict"][
        f"transformer.encoder.block.{i}.layer.1.layer_norm.weight"
    ]

# load each decoder blocks
for i in range(6):
    # layer 0
    state_dict[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.0.SelfAttention.q.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.0.SelfAttention.k.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.0.SelfAttention.k.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.0.SelfAttention.v.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.0.SelfAttention.v.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.0.SelfAttention.o.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.0.SelfAttention.o.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.0.layer_norm.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.0.layer_norm.weight"
    ]

    # layer 1
    state_dict[f"decoder.block.{i}.layer.1.EncDecAttention.q.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.1.EncDecAttention.q.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.1.EncDecAttention.k.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.1.EncDecAttention.k.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.1.EncDecAttention.v.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.1.EncDecAttention.v.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.1.EncDecAttention.o.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.1.EncDecAttention.o.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.1.layer_norm.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.1.layer_norm.weight"
    ]

    # layer 2
    state_dict[f"decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.2.DenseReluDense.wo.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.2.DenseReluDense.wo.weight"
    ]
    state_dict[f"decoder.block.{i}.layer.2.layer_norm.weight"] = official_weights["state_dict"][
        f"transformer.decoder.block.{i}.layer.2.layer_norm.weight"
    ]

model.load_state_dict(state_dict, strict=True)

# save the weights
torch.save(state_dict, "./pytorch_model.bin")

########################## TOKENIZER ##########################

# the tokenize and detokenize methods are taken from the official implementation


# link : https://github.com/sweetcocoa/pop2piano/blob/fac11e8dcfc73487513f4588e8d0c22a22f2fdc5/midi_tokenizer.py#L34
def tokenize(idx, token_type, n_special=4, n_note=128, n_velocity=2):
    if token_type == "TOKEN_TIME":
        return n_special + n_note + n_velocity + idx
    elif token_type == "TOKEN_VELOCITY":
        return n_special + n_note + idx
    elif token_type == "TOKEN_NOTE":
        return n_special + idx
    elif token_type == "TOKEN_SPECIAL":
        return idx
    else:
        return -1


# link : https://github.com/sweetcocoa/pop2piano/blob/fac11e8dcfc73487513f4588e8d0c22a22f2fdc5/midi_tokenizer.py#L48
def detokenize(idx, n_special=4, n_note=128, n_velocity=2, time_idx_offset=0):
    if idx >= n_special + n_note + n_velocity:
        return "TOKEN_TIME", (idx - (n_special + n_note + n_velocity)) + time_idx_offset
    elif idx >= n_special + n_note:
        return "TOKEN_VELOCITY", idx - (n_special + n_note)
    elif idx >= n_special:
        return "TOKEN_NOTE", idx - n_special
    else:
        return "TOKEN_SPECIAL", idx


# crate the decoder and then the encoder of the tokenizer
decoder = {}
for i in range(cfg.vocab_size):
    decoder.update({i: f"{detokenize(i)[1]}_{detokenize(i)[0]}"})

encoder = {v: k for k, v in decoder.items()}

# save the vocab
with open("./vocab.json", "w") as file:
    file.write(json.dumps(encoder))
