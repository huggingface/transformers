# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from contextlib import contextmanager
from unittest.mock import patch

from transformers import LlamaConfig
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig


def _tiny_llama_config(per_layer_config=None, **overrides):
    defaults = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        **overrides,
    }
    return LlamaConfig(per_layer_config=per_layer_config, **defaults)


def _tiny_gpt_oss_config(per_layer_config=None, **overrides):
    defaults = {
        "hidden_size": 64,
        "intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "sliding_window": 32,
        **overrides,
    }
    return GptOssConfig(per_layer_config=per_layer_config, **defaults)


def _tiny_llama4_config(per_layer_config=None, **overrides):
    defaults = {
        "hidden_size": 64,
        "intermediate_size": 32,
        "intermediate_size_mlp": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "num_local_experts": 4,
        "num_experts_per_tok": 1,
        "moe_layers": [1, 3],
        "attention_chunk_size": 32,
        "use_qk_norm": False,
        "attn_temperature_tuning": False,
        **overrides,
    }
    return Llama4TextConfig(per_layer_config=per_layer_config, **defaults)


def _tiny_nemotron_h_config(per_layer_config=None, **overrides):
    defaults = {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 16,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "layers_block_type": ["attention", "mamba", "moe", "attention"],
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "moe_shared_expert_intermediate_size": 32,
        "ssm_state_size": 16,
        "mamba_num_heads": 4,
        "mamba_head_dim": 16,
        "n_groups": 2,
        **overrides,
    }
    return NemotronHConfig(per_layer_config=per_layer_config, **defaults)


@contextmanager
def _hetero_context(model_key):
    """Temporarily set the production heterogeneous modeling spec on a model class."""
    from tests.heterogeneity.model_fixtures import MODEL_FIXTURES

    fixture = MODEL_FIXTURES[model_key]
    modeling_spec = fixture.spec_factory()
    with patch.object(fixture.pretrained_cls, "_heterogeneous_modeling_spec", modeling_spec, create=True):
        yield modeling_spec


def _build_model(config, model_cls, seed=42):
    """Build a model deterministically on CPU."""
    import torch

    torch.manual_seed(seed)
    return model_cls(config).eval()


def _forward_logits(model, input_ids):
    """Run a forward pass and return logits."""
    import torch

    with torch.no_grad():
        return model(input_ids).logits


def _dummy_input_ids(batch=1, seq_len=8):
    import torch

    return torch.randint(0, 32, (batch, seq_len))
