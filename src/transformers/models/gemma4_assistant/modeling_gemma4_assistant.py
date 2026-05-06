# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass

import torch
import torch.nn as nn

from ... import initialization as init
from ...generation import GenerationMixin
from ...masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..auto.modeling_auto import AutoModel
from .configuration_gemma4_assistant import Gemma4AssistantConfig


@dataclass
@auto_docstring
class Gemma4AssistantOutput(BaseModelOutput):
    r"""
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    """

    logits: torch.FloatTensor | None = None


class Gemma4AssistantMaskedEmbedder(nn.Module):
    token_ordering: torch.Tensor

    def __init__(self, config: Gemma4AssistantConfig):
        super().__init__()
        text_config = config.get_text_config()
        self.config = config
        self.centroid_intermediate_top_k = self.config.centroid_intermediate_top_k
        self.hidden_size = text_config.hidden_size
        self.num_centroids = self.config.num_centroids
        self.vocab_size = text_config.vocab_size
        self.vocab_size_per_centroid = self.vocab_size // self.num_centroids

        self.centroids = nn.Linear(self.hidden_size, self.num_centroids, bias=False)
        self.register_buffer("token_ordering", torch.empty(self.vocab_size, dtype=torch.long))

    def forward(self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor) -> torch.Tensor:
        batch, seq_len = hidden_states.shape[:2]
        centroid_logits = self.centroids(hidden_states)

        _, top_k_indices = torch.topk(centroid_logits, k=self.centroid_intermediate_top_k, dim=-1)
        token_ordering = self.token_ordering.long()
        canonical_positions_per_cluster = token_ordering.view(self.num_centroids, self.vocab_size_per_centroid)

        # For selected top-K clusters, get canonical positions
        selected_canonical = canonical_positions_per_cluster[top_k_indices]  # [B, L, top_k, K]

        # Gather embeddings from lm_head at these canonical positions
        selected_flat = selected_canonical.reshape(-1)  # [B*L*top_k*K]
        selected_embeddings = lm_head_weight[selected_flat].view(
            batch, seq_len, self.centroid_intermediate_top_k * self.vocab_size_per_centroid, self.hidden_size
        )

        # Compute dot products: [B, L, 1, D] @ [B, L, D, top_k*K] -> [B, L, top_k*K]
        selected_logits = (hidden_states.unsqueeze(-2) @ selected_embeddings.transpose(-1, -2)).squeeze(-2)
        mask_value = selected_logits.min().item() - 1.0

        # Scatter logits directly to canonical positions in the output
        output = torch.full(
            (batch, seq_len, self.vocab_size),
            fill_value=mask_value,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        scatter_idx = selected_canonical.view(batch, seq_len, -1)  # [B, L, top_k*K]
        return output.scatter_(dim=-1, index=scatter_idx, src=selected_logits)


class Gemma4AssistantPreTrainedModel(PreTrainedModel):
    config: Gemma4AssistantConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = ["shared_kv_states"]
    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)

        if isinstance(module, Gemma4AssistantMaskedEmbedder):
            init.zeros_(module.token_ordering)


@auto_docstring(custom_intro="A model for mutli-token prediction-based assisted decoding with Gemma 4.")
class Gemma4AssistantForCausalLM(Gemma4AssistantPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Gemma4AssistantConfig):
        super().__init__(config)
        text_config = config.get_text_config()

        self.vocab_size = text_config.vocab_size
        self.hidden_size = text_config.hidden_size
        self.backbone_hidden_size = config.backbone_hidden_size

        self.model = AutoModel.from_config(text_config)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.pre_projection = nn.Linear(2 * self.backbone_hidden_size, self.hidden_size, bias=False)
        self.post_projection = nn.Linear(self.hidden_size, self.backbone_hidden_size, bias=False)

        self.masked_embedding = Gemma4AssistantMaskedEmbedder(config) if self.config.use_ordered_embeddings else None

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.Tensor | None = None,  # Not actually used, only kept in signature to be ignored
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        attention_mask: dict[str, torch.Tensor] | None = None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool | None = None,  # Not actually used, only kept in signature to be ignored
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        shared_kv_states (`dict[str, torch.Tensor` of shape `(batch_size, 1, q_len, kv_len)`, *optional*):
            A dictionary containing the computed KV values for the last layer of each `layer_type` in this model.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Gemma4AssistantForCausalLM, Gemma4ForCausalLM

        >>> model = Gemma4ForCausalLM.from_pretrained("google/gemma-4-e2b-it")
        >>> assistant_model = Gemma4AssistantForCausalLM.from_pretrained("google/gemma-4-e2b-it-assistant")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-e2b-it")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, assistant_model=assistant_model, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
        "What is your favorite condiment?"
        ```"""
        if inputs_embeds is None or shared_kv_states is None:
            raise ValueError("inputs_embeds and shared_kv_states cannot be None.")

        inputs_embeds = self.pre_projection(inputs_embeds)
        bidirectional_masks = self.create_attention_masks(inputs_embeds, attention_mask, shared_kv_states)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=bidirectional_masks,
            position_ids=position_ids,
            shared_kv_states=shared_kv_states,
            use_cache=False,
            **kwargs,
        )

        last_hidden_state = outputs.last_hidden_state
        projected_state = self.post_projection(last_hidden_state)

        if self.config.use_ordered_embeddings:
            logits = self.masked_embedding(last_hidden_state, self.lm_head.weight)
        else:
            logits = self.lm_head(last_hidden_state)

        return Gemma4AssistantOutput(
            last_hidden_state=projected_state,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def create_attention_masks(self, inputs_embeds, attention_mask, shared_kv_states):
        """
        Prepare the attention masks for the assisted model; the `shared_kv_states` acts as past cache in this instance.

        We use bidirectional masks to account for causality
            - There is no difference for the edge case of `q_len == 1` as it acts as full attention no matter what
            - SWA interprets the window as forward-looking (future) when `q_idx=1` and `kv>=1`
                - We switch from a future to a past perspective by flipping on the kv axis
                - To account for position invariant padding, we also flip the base attention mask before initial creation
        """
        config = self.config.get_text_config()
        # (bsz, num_heads, seq_len, head_dim) -> (bsz, seq_len, head_dim)
        encoder_states_full_attn = shared_kv_states["full_attention"][0][:, 0]
        encoder_states_swa_attn = shared_kv_states["sliding_attention"][0][:, 0]

        sliding_attention_mask = attention_mask
        if attention_mask is not None:
            # Adjust for full mask --> cut mask only for valid kv states
            attention_mask = attention_mask[:, : encoder_states_full_attn.shape[1]]

            # 1. Take the last x entries to account for any potential SWA cutoff (from the main model)
            # 2. Flip the mask here to stay position invariant (along the original kv); see the flip at the end
            sliding_attention_mask = attention_mask[:, -encoder_states_swa_attn.shape[1] :].flip(dims=(1,))

        full_attention_mask = create_bidirectional_mask(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_states_full_attn,
        )
        swa_mask = create_bidirectional_sliding_window_mask(
            config=config,
            inputs_embeds=inputs_embeds,
            attention_mask=sliding_attention_mask,
            encoder_hidden_states=encoder_states_swa_attn,
        )

        if swa_mask is not None:
            # Reverse the future token perspective to a past tokens perspective by flipping the construct (kv == -1)
            swa_mask = swa_mask.flip(dims=(-1,))

        return {"full_attention": full_attention_mask, "sliding_attention": swa_mask}


__all__ = ["Gemma4AssistantPreTrainedModel", "Gemma4AssistantForCausalLM"]
