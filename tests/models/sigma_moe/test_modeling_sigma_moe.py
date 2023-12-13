# coding=utf-8
# Copyright 2022 Google SwitchTransformers Authors and HuggingFace Inc. team.
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


from transformers.models.sigma_moe import (
    SigmaMoEConfiguration,
    SigmaMoEFeedForwardLayer,
    SigmaMoEDecoderLayer,
)

if __name__ == "__main__":
    import torch

    bs = 5
    seq_len = 128
    d_model = 256

    config = SigmaMoEConfiguration(
        vocab_size=51200,
        d_model=d_model,
        d_ff=1024,
        num_hidden_layers=2,
        num_attention_heads=8,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=0.5,
        qk_layernorm=False,
        n_experts=8,
        expert_size=128,
        top_k_experts=2,
        moe_dropout=0.0,
        selection_mode="sigmoid",
        activation_after_topk=False,
        activation="relu",
        moe_bias=False,
        v_dim=None,
        sinkhorn_n_iters=3,
        expert_dropout=0.0,
        weight_std_scale=1.0,
    )
    # ff = SigmaMoEFeedForwardLayer(config, is_sparse=True)
    # x = torch.randn((bs, seq_len, d_model), device=torch.device("cpu"))
    # ff(x)

    decoder_layer = SigmaMoEDecoderLayer(config, is_sparse=True, layer_idx=0)
    tgt_len = 128
    src_len = 128
    hidden_states = torch.randn((bs, seq_len, d_model), device=torch.device("cpu"))
    mask = torch.tril(torch.ones((bs, 1, tgt_len, src_len), device=torch.device("cpu")))
    decoder_layer(hidden_states, mask)
