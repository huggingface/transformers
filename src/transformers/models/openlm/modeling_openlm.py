# Follows OLMo's HF template

import logging
from dataclasses import fields
from typing import List, Optional, Tuple, Union

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModelForCausalLM

from .model import Params, Transformer
from .norms import get_norm_class
from .attention import get_attn_func

from .configuration_openlm import OpenLMConfig

log = logging.getLogger(__name__)


def create_model_config_from_pretrained_config(config: OpenLMConfig):
    """
    Utility function
    """

    kwargs = {}
    for field in fields(Params):
        if hasattr(config, field.name):
            kwargs[field.name] = getattr(config, field.name)

    model_config = Params(**kwargs)

    if hasattr(config, "norm_type"):
        model_config.norm_type = get_norm_class(config.norm_type)

    if hasattr(config, "attn_name"):
        model_config.attn_func = get_attn_func(config.attn_name)

    return model_config


class OpenLMForCausalLM(PreTrainedModel):
    """
    Extremely barebones HF model wrapper.
    """

    config_class = OpenLMConfig
    base_model_prefix = "model"

    def __init__(self, config: OpenLMConfig, model: Optional[Transformer] = None):
        super().__init__(config)

        if not model:
            self.model_config = create_model_config_from_pretrained_config(config)
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            self.model_config.init_device = "cpu"
            self.model = Transformer(self.model_config)

        else:
            self.model = model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[
            Cache
        ] = None,  # This is a hack mitigation of an issue in transformers `4.39.x` https://github.com/huggingface/transformers/issues/29426
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is not None:
            log.warning("inputs_embeds is set but OpenLM does not support it yet")
        if attention_bias is not None:
            log.warning("attention_bias is et but OpenLM does not support it yet")
        if use_cache is None:
            use_cache = True
        if output_attentions:
            raise ValueError("output_attentions is not yet supported in OpenLM")
        if output_hidden_states:
            raise ValueError("output_hidden_states is not yet supported in OpenLM")

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print("outer past_key_values: ", type(past_key_values))
        # if past_key_values is not None:
        #     print(len(past_key_values), type(past_key_values[0]))
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = outputs[0]
        past_key_values = outputs[2]
        hidden_states = None

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.model_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
        )

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values[0][1], int):
                # This assumes that the second item of past key values is the length of the past (this is the case for linear attention)
                past_length = past_key_values[0][1]
            else:
                # This assumes that the first item of past key values is a list of all the past keys, thus the
                # shape 1 is the length of the past (this is the case for attention without window)
                past_length = past_key_values[0][0].shape[1]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.pop("use_cache", True),
        }
        return model_inputs

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.tok_embeddings

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        if self.model_config.weight_tying:
            return self.model.tok_embeddings
        else:
            return self.model.output

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.model_config.weight_tying:
            self.model.tok_embeddings = value
        else:
            self.model.output = value

    def tie_weights(self):
        """
        Copied from OLMo (description below). I removed it and the results just became garbage, so this pass is needed.
        This function is intentionally left as a no-op.
        Weight tying is handled as follows:
        - When the model is initialized, the `ff_out` layer is conditionally defined based on the `weight_tying` configuration.
        See: `if not config.weight_tying: self.transformer.update(...)` in `olmo/model.py`.
        - When computing logits, the `wte` weights are used directly if `weight_tying` is enabled.
        See: `if self.config.weight_tying: logits = F.linear(x, self.transformer.wte.weight, None)` in the `forward` method.
        Therefore, there is no need to explicitly tie the weights in this function.
        """
        pass

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> torch.nn.Embedding:
        raise NotImplementedError
