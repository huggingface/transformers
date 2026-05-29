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
from ..conversion_mapping import get_model_conversion_mapping
from ..core_model_loading import convert_and_load_state_dict_in_model
from ..masking_utils import create_causal_mask
from ..modeling_utils import LoadStateDictConfig, PreTrainedModel, _get_resolved_checkpoint_files
from ..utils import logging
from ..utils.loading_report import log_state_dict_report


if TYPE_CHECKING:
    from ..configuration_utils import PreTrainedConfig
    from .logits_process import LogitsProcessorList


logger = logging.get_logger(__name__)


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
        self.post_norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        previous_hidden_state: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_values: Cache | None,
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
        hidden_states = self.post_norm(hidden_states)

        return hidden_states


class MtpLayerStack(PreTrainedModel):
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_flash_attn = True
    # Since the embedding/head are shared with main model, silence any warning if they are provided again
    _keys_to_ignore_on_load_unexpected = [r"\.shared_head\.head\.weight", r"\.embed_tokens\.weight"]

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

        # The embeddings and head are shared between main model and each MTP layer
        self.embed_tokens = main_model.get_input_embeddings()
        self.shared_head = main_model.lm_head
        # Use the same rotary class
        self.rotary_emb = type(main_model.base_model.rotary_emb)(config=self.config)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_values: Cache | None,
        # Control how we sample the new token from each layer
        do_sample: bool = False,
        logits_processor: LogitsProcessorList | None = None,
        full_input_ids: torch.Tensor | None = None,  # needed as input for the logits_processor
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample 1 new token for each mtp layers present in this model. Note that the inputs are assumed to be
        already sliced and correct here, i.e. if the main model just processed inputs corresponding to tokens
        at positions [N-1, N] in the sequence, then from it you draft a new token for position N+1, and the
        `input_ids`/`position_ids`/`attention_mask` here are assumed to correspond to data for tokens at positions
        [N, N+1], i.e. shifted by 1 from the main model, by the newly drafted token. The `last_hidden_states` though
        will correspond to the same as the main model, i.e. positions [N-1, N] in the sequence length dimension.

        `full_input_ids` correspond to the full sequence of `input_ids`, which is used in case we have any `logits_processor`
        as some processors may require to check the length/value of the full previous sequence of ids.
        """
        batch_size = input_ids.shape[0]

        # We create this dummy cache simply to create the masks correctly, since they rely on the sizes of layer 0 of
        # the cache. Note that it does not create any copy of data, it simply keep a ref to internal tensors
        dummy_cache_for_masking = Cache(layers=past_key_values.layers[self.config.num_hidden_layers :])

        drafted_logits = []
        drafted_tokens = []
        for mtp_layer in self.layers:
            # We need to recompute those every layer since they change
            inputs_embeds = self.embed_tokens(input_ids).to(last_hidden_states.device)
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=dummy_cache_for_masking,
                position_ids=position_ids,
            )

            last_hidden_states = mtp_layer(
                inputs_embeds,
                last_hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

            # Only compute logits for next drafted token
            logits = self.shared_head(last_hidden_states[:, -1:, :])

            # Append the drafted logits
            drafted_logits.append(logits)

            # Decode one token
            next_token_logits = logits[:, -1, :].to(device=input_ids.device)
            if logits_processor is not None and full_input_ids is not None:
                next_token_scores = logits_processor(full_input_ids, next_token_logits.to(torch.float32))
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1, dtype=torch.float32)
                next_mtp_token = torch.multinomial(probs, num_samples=1)
            else:
                next_mtp_token = torch.argmax(next_token_scores, dim=-1, keepdim=True)
            drafted_tokens.append(next_mtp_token)

            # Roll by 1 and append for next layer
            input_ids = torch.cat([input_ids[:, 1:], next_mtp_token], dim=-1)
            attention_mask = torch.cat([attention_mask[:, 1:], attention_mask.new_ones(batch_size, 1)], dim=-1)
            position_ids = torch.cat([position_ids[:, 1:], position_ids[:, -1:] + 1], dim=-1)

            # Need to cat ful_ids as well for the processors
            if full_input_ids is not None:
                full_input_ids = torch.cat([full_input_ids, next_mtp_token], dim=-1)

        new_candidate_ids = torch.cat(drafted_tokens, dim=1)
        candidate_logits = torch.cat(drafted_logits, dim=1)
        return new_candidate_ids, candidate_logits

    @classmethod
    def from_pretrained(cls, main_model: PreTrainedModel, device_map=None, **kwargs) -> MtpLayerStack:
        pretrained_model_name_or_path = main_model.config.name_or_path
        num_hidden_layers = main_model.config.get_text_config().num_hidden_layers
        # Heuristic: the main model should have the mtp layer patterns under `_keys_to_ignore_on_load_unexpected` to avoid
        # loading them by default, so use it to later load the correct keys from the checkpoints
        mtp_patterns = main_model._keys_to_ignore_on_load_unexpected.copy()
        # Due to different released checkpoints, only keep the ones with layer number >= num_hidden_layers - otherwise
        # mtp layers in a smaller checkpoints could be wrongly added as a 2nd mtp layer of a bigger checkpoint
        final_mtp_patterns = []
        for pattern in mtp_patterns:
            match_object = re.search(r"\.(\d+)", pattern)
            if match_object is not None and match_object.group(1) < num_hidden_layers:
                continue
            final_mtp_patterns.append(pattern)
        if len(final_mtp_patterns) == 0:
            raise ValueError(f"{main_model.__class__.__name__} does not seem to register any known MTP layer patterns")
        mtp_regex = re.compile("|".join(rf"({pattern})" for pattern in final_mtp_patterns))

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
        mtp_state_dict = {}
        all_pointer = set()
        for file in mtp_files:
            file_pointer = safe_open(file, framework="pt", device="cpu")
            all_pointer.add(file_pointer)
            for k in file_pointer.keys():
                # It's one of the mtp weights
                if k in mtp_weight_map.keys():
                    # Rename dynamically to change the index of layers and add to state_dict
                    renamed = re.sub(
                        r"layers\.(\d+)\.", lambda m: f"layers.{int(m.group(1)) - num_hidden_layers}.mtp_block", k
                    )
                    mtp_state_dict[renamed] = file_pointer.get_slice(k)  # don't materialize yet

        # For the correct conversions, we need first the mtp-specific renamings, then the main_model conversions
        weight_conversions = get_model_conversion_mapping(mtp_model, add_legacy=False)
        weight_conversions.extend(main_model._weight_conversions)

        # Load the weights
        loading_info, _ = convert_and_load_state_dict_in_model(
            model=mtp_model,
            state_dict=mtp_state_dict,
            load_config=LoadStateDictConfig(weight_mapping=weight_conversions, device_map=device_map),
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
