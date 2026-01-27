from typing import ClassVar

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.paligemma.modeling_paligemma import PaliGemmaForConditionalGeneration

from ...cache_utils import Cache


class NewTaskModelForNewTask(PaliGemmaForConditionalGeneration):
    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config):
        super().__init__(config=config)

        self.embedding_dim = self.config.embedding_dim
        self.custom_text_proj = nn.Linear(self.config.text_config.hidden_size, self.embedding_dim)

        if self.language_model._tied_weights_keys is not None:
            prefix = "model.language_model."
            prefixed_mapping = {
                f"{prefix}{target}": f"{prefix}{source}"
                for target, source in self.language_model._tied_weights_keys.items()
            }
            if isinstance(self._tied_weights_keys, dict):
                self._tied_weights_keys.update(prefixed_mapping)
            else:
                self._tied_weights_keys = prefixed_mapping

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        token_type_ids: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        num_logits_to_keep: int = 0,
    ):
        r"""
        Returns:
        """
        vlm_outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            token_type_ids=token_type_ids,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            num_logits_to_keep=num_logits_to_keep,
        )
        last_hidden_states = vlm_outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        embeddings = proj / proj.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

        return (embeddings,) + vlm_outputs

    def resize_token_embeddings(
        self, new_num_tokens: int | None = None, pad_to_multiple_of=None, mean_resizing=True
    ) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)

        # Update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings

        return model_embeds
