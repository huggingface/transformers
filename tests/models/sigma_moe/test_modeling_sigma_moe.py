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


import copy
import tempfile
import unittest

from transformers.models.sigma_moe import SigmaMoEConfiguration, SigmaMoEFeedForwardLayer, SigmaMoETransformerLayer

if __name__ == "__main__":
    import torch
    bs = 5
    seq_len = 128
    d_model = 256

    config = SigmaMoEConfiguration(
        d_model=d_model,
        d_ff=1024,
        n_experts=8,
        expert_size=128,
        top_k_experts=2,
        dropout=0.0,
        selection_mode="sigmoid",
        activation_after_topk=False,
        activation="relu",
        bias=False,
        v_dim=None,
        sinkhorn_n_iters=3,
        expert_dropout=0.0,
        weight_std_scale=1.0,
    )
    ff = SigmaMoEFeedForwardLayer(config, is_sparse=True)
    x = torch.randn((bs, seq_len, d_model), device=torch.device("cpu"))
    ff(x)
