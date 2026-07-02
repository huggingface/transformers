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

from collections.abc import Callable

from transformers.integrations.heterogeneity.heterogeneous_modeling_spec import HeterogeneousModelingSpec
from transformers.integrations.heterogeneity.skip_utils import ReturnEntry, get_skip_replacement


MODEL_TO_SPEC_FACTORY: dict[str, Callable[[], HeterogeneousModelingSpec]] = {
    "gpt_oss": lambda: gpt_oss(),
    "llama": lambda: llama(),
    "llama4": lambda: llama4(),
    "nemotron_h": lambda: nemotron_h(),
}


def _identity(value):
    return value


def gpt_oss() -> HeterogeneousModelingSpec:
    import torch

    from transformers.models.gpt_oss.modeling_gpt_oss import (
        GptOssAttention,
        GptOssDecoderLayer,
        GptOssMLP,
        GptOssRMSNorm,
    )

    return HeterogeneousModelingSpec(
        layer_cls=GptOssDecoderLayer,
        layer_idx_variable_name="layer_idx",
        skip_descriptors={
            "attention": {
                "input_layernorm": get_skip_replacement(
                    GptOssRMSNorm, ReturnEntry(arg_name="hidden_states", transform=_identity)
                ),
                "self_attn": get_skip_replacement(
                    GptOssAttention, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                ),
            },
            "mlp": {
                "post_attention_layernorm": get_skip_replacement(
                    GptOssRMSNorm, ReturnEntry(arg_name="hidden_states", transform=_identity)
                ),
                "mlp": get_skip_replacement(
                    GptOssMLP, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                ),
            },
        },
    )


def llama() -> HeterogeneousModelingSpec:
    import torch

    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaDecoderLayer,
        LlamaMLP,
        LlamaRMSNorm,
    )

    return HeterogeneousModelingSpec(
        layer_cls=LlamaDecoderLayer,
        layer_idx_variable_name="layer_idx",
        skip_descriptors={
            "attention": {
                "input_layernorm": get_skip_replacement(
                    LlamaRMSNorm, ReturnEntry(arg_name="hidden_states", transform=_identity)
                ),
                "self_attn": get_skip_replacement(
                    LlamaAttention,
                    [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None],
                ),
            },
            "mlp": {
                "post_attention_layernorm": get_skip_replacement(
                    LlamaRMSNorm, ReturnEntry(arg_name="hidden_states", transform=_identity)
                ),
                "mlp": get_skip_replacement(LlamaMLP, ReturnEntry(arg_name="x", transform=torch.zeros_like)),
            },
        },
    )


def llama4() -> HeterogeneousModelingSpec:
    import torch

    from transformers.models.llama4.modeling_llama4 import (
        Llama4TextAttention,
        Llama4TextDecoderLayer,
        Llama4TextMLP,
        Llama4TextMoe,
        Llama4TextRMSNorm,
    )

    return HeterogeneousModelingSpec(
        layer_cls=Llama4TextDecoderLayer,
        layer_idx_variable_name="layer_idx",
        skip_descriptors={
            "attention": {
                "input_layernorm": get_skip_replacement(
                    Llama4TextRMSNorm, ReturnEntry(arg_name="x", transform=_identity)
                ),
                "self_attn": get_skip_replacement(
                    Llama4TextAttention, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                ),
            },
            "mlp": {
                "post_attention_layernorm": get_skip_replacement(
                    Llama4TextRMSNorm, ReturnEntry(arg_name="x", transform=_identity)
                ),
                "feed_forward": get_skip_replacement(
                    Llama4TextMLP, ReturnEntry(arg_name="x", transform=torch.zeros_like)
                ),
                ("feed_forward", Llama4TextMoe): get_skip_replacement(
                    Llama4TextMoe, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                ),
            },
        },
    )


def nemotron_h() -> HeterogeneousModelingSpec:
    import torch

    from transformers.models.nemotron_h.modeling_nemotron_h import (
        NemotronHAttention,
        NemotronHBlock,
        NemotronHMamba2Mixer,
        NemotronHMoE,
        NemotronHRMSNorm,
    )

    return HeterogeneousModelingSpec(
        layer_cls=NemotronHBlock,
        layer_idx_variable_name="layer_idx",
        skip_descriptors={
            "mixer": {
                "norm": get_skip_replacement(
                    NemotronHRMSNorm, ReturnEntry(arg_name="hidden_states", transform=_identity)
                ),
                ("mixer", NemotronHAttention): get_skip_replacement(
                    NemotronHAttention, [ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like), None]
                ),
                ("mixer", NemotronHMoE): get_skip_replacement(
                    NemotronHMoE, ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like)
                ),
                ("mixer", NemotronHMamba2Mixer): get_skip_replacement(
                    NemotronHMamba2Mixer, ReturnEntry(arg_name="hidden_states", transform=torch.zeros_like)
                ),
            },
        },
    )
