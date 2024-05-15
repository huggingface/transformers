# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

"""
Convert Mega pretrained checkpoint. Built to convert the Masked LM checkpoint located at
https://huggingface.co/mnaylor/mega-wikitext-103

Requirements:
  - clone the Mega repo and install fairseq from there
    1. git clone https://github.com/facebookresearch/mega.git
    2. cd mega && pip install -e
  - clone the pretrained weights for the original implementation from the hugging face repo
    * use this location as the path for pretrained weights
"""

import argparse

# utilities to import the model weights and config file
import os
import pickle as pkl

# PyTorch + new model classes
import torch
from torch import nn

from transformers import AutoTokenizer, MegaConfig, MegaForMaskedLM


# import the EncoderLayer class used to pretrain
# !! NOTE !! this requires the version of fairseq that is built when you install the Mega source
try:
    from fairseq.modules.mega_layer import MegaEncoderLayer
except ImportError:
    raise ImportError("You need to install the version of fairseq from the Mega repo!")


# define the wrapper classes used to train the MLM  (see colab notebook below)
# https://colab.research.google.com/drive/1qfUO6o5HRdxBblWlw058HVyvaEPhPpH8?usp=sharing
# MegaLM outputs hidden states
class MegaLM(nn.Module):
    "The base class for our Mega encoder - given input IDs, embed text and return encoder output"

    def __init__(self, mega_args, depth, vocab_size):
        super().__init__()
        self.mega_args = mega_args
        self.embedding_layer = nn.Embedding(vocab_size, self.mega_args.encoder_embed_dim)
        self.encoders = nn.ModuleList([MegaEncoderLayer(self.mega_args) for _ in range(depth)])
        self.depth = depth

    def forward(self, input_ids, attention_mask, batch_first=True, ignore_mask_value=0):
        """
        Code for a forward pass - expects input_ids and attention_mask to come from a Hugging Face tokenizer as PyTorch
        tensors, and returns a tensor of size (batch, n_classes) containing classification logits

        Other options:
          - batch_first: boolean indicating whether the batch dimension is first in input_ids (default: True, which
            aligns with the HF tokenizer behavior)
          - ignore_mask_value: the value in attention_mask that identifies tokens that should be ignored (default: 0,
            which aligns with HF tokenizer)
        """

        # Mega expects embeddings to be (time, batch, embedding size), but
        # Hugging Face returns tokens as (batch, time)
        if batch_first:
            input_ids = input_ids.T

        # to make things more confusing, Mega expects the attention mask to
        # be (batch, time), but with values of 0 (normal token) and 1 (ignore token)
        # which is the opposite of what HF returns
        if ignore_mask_value == 0:
            attention_mask = 1 - attention_mask

        # get token embeddings from IDs
        embeds = self.embedding_layer(input_ids)

        # pass through the Mega layers
        # input is (time, batch, encoder dim) and output is the same
        for encoder in self.encoders:
            embeds = encoder(embeds, attention_mask)

        # return according to the shape specified
        if batch_first:
            # (T, B, H) --> (B, T, H)
            return torch.transpose(embeds, 0, 1)
        else:
            return embeds


# renamed from MegaForMaskedLM to avoid confusion with new module
class OriginalMegaForMaskedLM(nn.Module):
    "A wrapper class for doing masked language modeling with Mega"

    def __init__(self, mega_args, depth, vocab_size):
        super().__init__()
        self.mega = MegaLM(mega_args, depth, vocab_size)
        self.mlm_head = nn.Linear(mega_args.encoder_embed_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, input_ids, attention_mask, batch_first=True, ignore_mask_value=0):
        """
        Perform a forward pass through the Mega encoder and the masked LM head. Returns logits for each vocabulary
        entry.

        If `batch_first` (default to align with Hugging Face tokenizer behavior), output will have the shape (Batch
        size, Sequence length, Vocab size); otherwise (S, B, V)
        """
        encoder_output = self.mega(input_ids, attention_mask, batch_first, ignore_mask_value)
        return self.mlm_head(self.dropout(encoder_output))


# code to convert the checkpoint located in the user-specified location
def convert_checkpoint_to_huggingface(pretrained_checkpoint_path, output_path, includes_tokenizer):
    with open(os.path.join(pretrained_checkpoint_path, "model_args.pkl"), "rb") as f:
        mega_original_args = pkl.load(f)

    # load the original encoder
    original_mlm = OriginalMegaForMaskedLM(**mega_original_args).eval()

    # load its weights
    print(
        "Original Mega encoder:",
        original_mlm.mega.load_state_dict(
            torch.load(os.path.join(pretrained_checkpoint_path, "encoder_weights.pt"), map_location="cpu")
        ),
    )
    print(
        "Original Mega MLM layer:",
        original_mlm.mlm_head.load_state_dict(
            torch.load(os.path.join(pretrained_checkpoint_path, "mlm_head_weights.pt"), map_location="cpu")
        ),
    )

    # create a new config from the old one
    hf_config = MegaConfig(
        num_hidden_layers=mega_original_args["depth"],
        vocab_size=mega_original_args["vocab_size"],
        hidden_size=mega_original_args["mega_args"].encoder_embed_dim,
        shared_representation_size=mega_original_args["mega_args"].encoder_z_dim,
        intermediate_size=mega_original_args["mega_args"].encoder_hidden_dim,
        ema_projection_size=mega_original_args["mega_args"].encoder_n_dim,
        dropout_prob=mega_original_args["mega_args"].dropout,
        attention_probs_dropout_prob=mega_original_args["mega_args"].attention_dropout,
        hidden_dropout_prob=mega_original_args["mega_args"].hidden_dropout,
        activation=mega_original_args["mega_args"].activation_fn,
        attention_activation=mega_original_args["mega_args"].attention_activation_fn,
        bidirectional=mega_original_args["mega_args"].bidirectional,
        use_chunking=mega_original_args["mega_args"].encoder_chunk_size > 0,
        chunk_size=mega_original_args["mega_args"].encoder_chunk_size,
        truncation=mega_original_args["mega_args"].truncation_length,
        normalization_type=mega_original_args["mega_args"].normalization_type,
        normalize_before_mega=True,
        norm_affine=True,
        use_feature_dropout=mega_original_args["mega_args"].feature_dropout,
        relative_positional_bias=mega_original_args["mega_args"].rel_pos_bias,
        max_positions=mega_original_args["mega_args"].max_source_positions,
        nffn_hidden_size=mega_original_args["mega_args"].encoder_ffn_embed_dim,
        normalize_before_ffn=mega_original_args["mega_args"].normalize_before,
        # new arguments added for HF implementation
        nffn_activation_dropout_prob=0.0,
        add_token_type_embeddings=False,
        add_lm_hidden_dense_layer=False,
    )

    hf_mlm = MegaForMaskedLM(hf_config).eval()

    # the originl checkpoint just uses nn.Embedding for the word embeddings
    # we use a wrapper module for embeddings to add support for positional embeddings
    hf_mlm.mega.embedding_layer.word_embeddings.weight = original_mlm.mega.embedding_layer.weight

    # modify the state dictionary of the original checkpoint to account for naming issues in the Hugging Face
    # ecosystem -- any names containing "beta" or "gamma" aren't safe to use and are renamed upon _load_pretrained,
    # also renaming previously confusing parameter names
    original_state_dict = original_mlm.mega.encoders.state_dict()
    updated_keys = {}
    for module_name in original_state_dict.keys():
        new_module_name = None
        # have to handle gamma, beta, and alpha differently due to their use
        # in multiple modules within the original repository;
        # beta is used in EMA, MovingAverageGatedAttention, and RotaryRelativePositionalBias, and must be renamed due to flax/tf weights
        # the EMA sublayer was renamed from "move" to "ema_gate" for readability, so that is also done here
        if "beta" in module_name:
            # EMA sub-layers were always called "move" in the original repo
            if "move.beta" in module_name:
                new_module_name = module_name.replace("move.beta", "ema_gate.ema_expansion_matrix")
            elif "mega_layer.beta" in module_name:
                new_module_name = module_name.replace("beta", "qk_bias")
            else:
                new_module_name = module_name.replace("beta", "b_param")
        # beta is used in EMA and MovingAverageGatedAttention, and must be renamed due to flax/tf weights
        elif "gamma" in module_name:
            if "move.gamma" in module_name:
                new_module_name = module_name.replace("move.gamma", "ema_gate.kernel_projection_matrix")
            elif "mega_layer.gamma" in module_name:
                new_module_name = module_name.replace("gamma", "qk_weight")
            else:
                new_module_name = module_name.replace("gamma", "g_param")
        # alpha is used in EMA and positional bias; renaming to improve readability
        elif "move.alpha" in module_name:
            new_module_name = module_name.replace("move.alpha", "ema_gate.decay_factor")
        # delta is only used in EMA; renaming to improve readability
        elif "move.delta" in module_name:
            new_module_name = module_name.replace("move.delta", "ema_gate.damping_factor")
        # omega is only used in EMA; renaming to improve readability
        elif "omega" in module_name:
            new_module_name = module_name.replace("move.omega", "ema_gate.residual_weight")

        if new_module_name:
            updated_keys[module_name] = new_module_name

    if len(updated_keys) != 0:
        print(f"Renaming these keys: {updated_keys.keys()}")
    else:
        print("No need to rename state dict entries")
    for old, new in updated_keys.items():
        original_state_dict[new] = original_state_dict.pop(old)

    # now attempt to load the state dictionary with updated names
    # note that we now call it `mega.layers` instead of `mega.encoders` due to hugging face style
    print("HF Mega encoder:", hf_mlm.mega.layers.load_state_dict(original_state_dict))

    # load the MLM head weights directly
    print(
        "HF Mega MLM layer:",
        hf_mlm.mlm_head.load_state_dict(
            torch.load(os.path.join(pretrained_checkpoint_path, "mlm_head_weights.pt"), map_location="cpu")
        ),
    )

    # test on a randomly generated input sequence
    input_ids = torch.randint(0, hf_config.vocab_size, size=(4, 256))
    input_mask = torch.ones_like(input_ids)
    # mask a few tokens to make sure masking is applied appropriately :)
    input_mask[:, -10:] = 0

    # run forward passes
    original_output = original_mlm(input_ids, input_mask, batch_first=True, ignore_mask_value=0)
    hf_output = hf_mlm(input_ids, input_mask)[0]

    # print shapes and diff
    print(f"original output {original_output.shape}")
    print(f"hf output {hf_output.shape}")
    print(f"max diff: {(original_output - hf_output).max()}")  # 0.0
    success = torch.allclose(original_output, hf_output, atol=1e-3)

    if success:
        print("Yay!")
        hf_mlm.save_pretrained(output_path)
    else:
        raise RuntimeError(f"Something's broken :(\nOriginal:\n{original_output}\n\nHF\n{hf_output}\n{hf_mlm}")

    if includes_tokenizer:
        print("Transferring tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_checkpoint_path)
        tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Point to the directory containing your model weights using the official Mega repo",
    )

    parser.add_argument(
        "--output_path", default=None, type=str, required=True, help="Location to save the Hugging Face version"
    )

    parser.add_argument(
        "--includes_tokenizer",
        action="store_true",
        help="Use this flag if there is a Hugging Face tokenizer in the original checkpoint repo",
    )

    args = parser.parse_args()

    convert_checkpoint_to_huggingface(args.pretrained_checkpoint_path, args.output_path, args.includes_tokenizer)
