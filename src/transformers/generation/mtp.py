# Copyright 2026 the HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from torch import nn

from ..cache_utils import Cache
from ..core_model_loading import convert_and_load_state_dict_in_model
from ..masking_utils import create_causal_mask
from ..modeling_utils import LoadStateDictConfig, PreTrainedModel, _get_resolved_checkpoint_files
from ..utils import logging
from ..utils.loading_report import log_state_dict_report


if TYPE_CHECKING:
    from ..configuration_utils import PreTrainedConfig

logger = logging.get_logger(__name__)


class MTPSharedHead(nn.Module):
    def __init__(self, config: PreTrainedConfig, norm_cls: type[nn.Module]):
        super().__init__()
        self.config = config
        self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(hidden_states))


class MtpLayer(nn.Module):
    def __init__(
        self, config: PreTrainedConfig, decoder_layer_cls: type[nn.Module], norm_cls: type[nn.Module], layer_idx: int
    ):
        super().__init__()
        self.config = config
        self.enorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.mtp_block = decoder_layer_cls(config, layer_idx)
        self.shared_head = MTPSharedHead(config, norm_cls)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        previous_hidden_state: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_values: Cache | None,
        logits_to_keep: int | torch.Tensor = 1,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eh_cat = torch.cat([self.enorm(inputs_embeds), self.hnorm(previous_hidden_state)], dim=-1)
        hidden_states = self.eh_proj(eh_cat)
        hidden_states = self.mtp_block(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.shared_head(hidden_states[:, slice_indices, :])

        return hidden_states, logits


class MtpLayerStack(PreTrainedModel):
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_flash_attn = True

    def __init__(self, main_model: PreTrainedModel, num_mtp_layers: int):
        super().__init__(main_model.config.get_text_config())
        self.num_mtp_layers = num_mtp_layers
        # Infer the type of the layers based on the main model
        layer_cls = type(main_model.base_model.layers[0])
        norm_cls = next(
            type(module) for name, module in main_model.base_model.layers[0].named_modules() if "norm" in name
        )

        # Instantiate new mtp layers
        self.layers = nn.ModuleList(
            [
                MtpLayer(self.config, layer_cls, norm_cls, self.config.num_hidden_layers + k)
                for k in range(num_mtp_layers)
            ]
        )
        # Get the embedding layer of the main model
        self.embed_tokens = main_model.get_input_embeddings()
        # Get the rotary of main model
        self.rotary_emb = main_model.base_model.rotary_emb

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_values: Cache | None,
        logits_to_keep: int | torch.Tensor = 1,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = input_ids.shape[0]

        # Here `input_ids`/`attention_mask`/`position_ids` are the inputs that the main model just processed, + the length of
        # the last drafted token from main model logits
        mtp_input_ids = input_ids[:, 1:]
        mtp_position_ids = position_ids[:, 1:]
        mtp_attention_mask = attention_mask[:, 1:]

        drafted_logits = []
        drafted_tokens = []
        for mtp_layer in self.layers:
            # We need to recompute those every layer since they change
            inputs_embeds = self.embed_tokens(mtp_input_ids)
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids=mtp_position_ids)
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=mtp_attention_mask,
                past_key_values=past_key_values,
                position_ids=mtp_position_ids,
            )

            last_hidden_states, logits = mtp_layer(
                inputs_embeds,
                last_hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=mtp_position_ids,
                past_key_values=past_key_values,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

            # Append the drafted logits
            drafted_logits.append(logits)
            # For now, assume greedy decoding
            next_mtp_token = logits.argmax(dim=-1)
            drafted_tokens.append(next_mtp_token)

            # Roll by 1 and append for next layer
            mtp_input_ids = torch.cat([mtp_input_ids[:, 1:], next_mtp_token], dim=-1)
            mtp_attention_mask = torch.cat(
                [mtp_attention_mask[:, 1:], mtp_attention_mask.new_ones(batch_size, 1)], dim=-1
            )
            mtp_position_ids = mtp_position_ids.roll(-1, dims=-1)
            mtp_position_ids[:, -1] = mtp_position_ids[:, -2] + 1

        candidate_ids = torch.cat([input_ids, torch.cat(drafted_tokens, dim=1)], dim=1)
        candidate_logits = torch.cat(drafted_logits, dim=1)
        return candidate_ids, candidate_logits

    @classmethod
    def from_pretrained(cls, main_model: PreTrainedModel, **kwargs) -> MtpLayerStack:
        pretrained_model_name_or_path = main_model.config.name_or_path
        # Heuristic: the main model should have the mtp layer patterns under `_keys_to_ignore_on_load_unexpected` to avoid
        # loading them by default, so use it to later load the correct keys from the checkpoints
        mtp_patterns = main_model._keys_to_ignore_on_load_unexpected.copy()
        if len(mtp_patterns) == 0:
            raise ValueError(f"{main_model.__class__.__name__} does not seem to register any known MTP layer patterns")
        mtp_regex = re.compile("|".join(rf"({pattern})" for pattern in mtp_patterns))

        # Get the number of layers in the checkpoint
        num_mtp_layers = main_model.config.get_text_config().num_nextn_predict_layers
        # Since we need to share some modules, let's not instantiate on meta device
        mtp_model = cls(main_model, num_mtp_layers)

        # Now, let's scan the index to obtain the mtp-specific files and weights
        checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            variant=None,
            gguf_file=None,
            use_safetensors=True,
            user_agent=None,
            is_remote_code=False,
        )
        # Filter out only the files containing mtp weights
        mtp_weight_map = {k: v for k, v in sharded_metadata["weight_map"].items() if mtp_regex.search(k) is not None}
        mtp_files = [file for file in checkpoint_files if os.path.basename(file) in mtp_weight_map.values()]

        # Open the files, get the slices corresponding only to mtp weights, rename them, and load them
        num_hidden_layers = main_model.config.get_text_config().num_hidden_layers
        mtp_state_dict = {}
        all_pointer = set()
        for file in mtp_files:
            file_pointer = safe_open(file, framework="pt", device="cpu")
            all_pointer.add(file_pointer)
            for k in file_pointer.keys():
                # It's one of the mtp weights
                if k in mtp_weight_map.keys():
                    # Rename and add to state_dict
                    renamed = re.sub(
                        r"model\.layers\.(\d+)\.", lambda m: f"layers.{int(m.group(1)) - num_hidden_layers}.", k
                    )
                    mtp_state_dict[renamed] = file_pointer.get_slice(k)  # don't materialize yet

        # Load the weights
        loading_info, _ = convert_and_load_state_dict_in_model(
            model=mtp_model,
            state_dict=mtp_state_dict,
            load_config=LoadStateDictConfig(weight_mapping=main_model._weight_conversions),
            tp_plan=None,
        )

        # finally close all opened file pointers
        for k in all_pointer:
            k.__exit__(None, None, None)

        log_state_dict_report(
            model=mtp_model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            ignore_mismatched_sizes=False,
            loading_info=loading_info,
            logger=logger,
        )

        return mtp_model

    @classmethod
    def _can_set_attn_implementation(cls) -> bool:
        # Assume we always can
        return True

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        # Assume we always can
        return True
