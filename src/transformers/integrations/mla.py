# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Various helpers to handle MLA and DSA input preparation"""

from functools import wraps

from torch import nn

from ..utils.generic import is_flash_attention_requested


def supports_mla(module: nn.Module):
    return getattr(module, "is_mla", False) and is_flash_attention_requested(module.config, version=4)


def conditional_kv_expansion(func):
    """Only certain implementations like FA4 can handle the unexpanded latents within as direct input"""
    @wraps(func)
    def wrapper(self, kv_nope, k_rot):
        if supports_mla(self):
            return k_rot, kv_nope

        return func(self, kv_nope, k_rot)

    return wrapper


def mla(func):
    """
    MLA/DSA implementation for flash attention to handle the proper latents

    FA4 describes its needed input as
        O = softmax(scale * (Q @ K.T + Qv @ V.T)) @ V
        where Q = q_pe, Qv = q_nope, K = pe_cache, V = kv_cache (the latent).

    Our preparation goes as
        Q = q_rot / q_pe
        Qv = q_pass @ k_latent
        K = k_rot / k_pe
        V = kv_nope

    NOTE:
        1. `Qv @ V.T` follows `q_pass @ k_nope.T == (q_pass @ k_latent) @ kv_nope.T`
        2. Output recovery follows `attn_weights @ value_states` == `(attn_weights @ kv_nope) @ v_latent.T`
    """
    @wraps(func)
    def wrapper(
        module,
        query,
        key,
        value,
        attention_mask,
        *args,
        **kwargs,
    ):
        # Normal forward
        if not supports_mla(module):
            return func(
                module,
                query,
                key,
                value,
                attention_mask,
                *args,
                **kwargs,
            )

        query_latent, query = query.split(
            [module.qk_nope_head_dim, module.qk_rope_head_dim],
            dim=-1,
        )

        kv_latents = module.kv_b_proj.weight.view(
            -1,
            module.qk_nope_head_dim + module.v_head_dim,
            module.kv_lora_rank,
        )
        key_latent, value_latent = kv_latents.split(
            [module.qk_nope_head_dim, module.v_head_dim],
            dim=1,
        )

        qv_latents = (query_latent.to(key_latent) @ key_latent).transpose(1, 2)

        attn_output, attn_weights = func(
            module,
            query,
            key,
            value,
            attention_mask,
            *args,
            qv_latents=qv_latents,
            **kwargs,
        )

        attn_output = (attn_output.transpose(1, 2) @ value_latent.transpose(-1, -2)).transpose(1, 2)

        return attn_output, attn_weights

    return wrapper
