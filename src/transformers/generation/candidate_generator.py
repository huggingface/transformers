# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from .logits_process import LogitsProcessorList


class CandidateGenerator:
    """Abstract base class for all candidate generators that can be applied during assisted generation."""

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and, optionally, a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `get_candidates`."
        )

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call "
            "`update_candidate_strategy`."
        )


class AssistedCandidateGenerator(CandidateGenerator):
    """
    `CandidateGenerator` class to be used for assisted generation. This class generates candidates through the use of
    a smaller model. Read the following blog post for more information: https://huggingface.co/blog/assisted-generation

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        assistant_model (`PreTrainedModel`):
            The model to be used for generating candidates. This model should be smaller than the main model.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        model_kwargs (`Dict`):
            The keyword arguments that will be passed to the main model, and are used as base inputs for the assistant
            model as well.
        inputs_tensor (`torch.Tensor`, *optional*):
            The model input tensor. In encoder-decoder models, this is the encoder input.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
    """

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        logits_processor: "LogitsProcessorList",
        model_kwargs: Dict,
        inputs_tensor: Optional[torch.Tensor] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
    ):
        self.assistant_model = assistant_model

        # Prepare the number of candidate tokens
        if hasattr(assistant_model, "num_assistant_tokens"):
            warnings.warn(
                "Setting `num_assistant_tokens` via `assistant_model.num_assistant_tokens` is deprecated and will be "
                "removed in v4.37. Make sure to set `num_assistant_tokens` via the generation_config instead.",
                FutureWarning,
            )
            self.num_assistant_tokens = assistant_model.num_assistant_tokens
        else:
            self.num_assistant_tokens = assistant_model.generation_config.num_assistant_tokens

        # Prepare the kwargs for the assistant model
        assistant_kwargs = {}
        for key, value in model_kwargs.items():  # deepcopy crashes if we attempt to copy encoder outputs with grads
            if key not in ("encoder_outputs", "assistant_encoder_outputs"):
                assistant_kwargs[key] = value.detach() if isinstance(value, torch.Tensor) else copy.deepcopy(value)

        if "assistant_encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["assistant_encoder_outputs"]
        elif assistant_model.config.is_encoder_decoder:
            inputs_tensor, model_input_name, assistant_kwargs = assistant_model._prepare_model_inputs(
                inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_kwargs
            )
            assistant_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, assistant_kwargs, model_input_name
            )
        elif "encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["encoder_outputs"]
        self.assistant_kwargs = assistant_kwargs

        # Prepare assistant model's keys of inputs
        if assistant_model.config.is_encoder_decoder:
            # both are encoder-decoder
            self.input_ids_key = "decoder_input_ids"
            self.attention_key = "decoder_attention_mask"
        elif "encoder_outputs" in assistant_kwargs:
            # special case for encoder-decoder with decoder-only assistant (like DistilWhisper)
            self.input_ids_key = "input_ids"
            self.attention_key = "attention_mask"
            self.assistant_kwargs["attention_mask"] = self.assistant_kwargs.get(
                "decoder_attention_mask",
                torch.ones((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.long),
            )
        else:
            # both are decoder-only
            self.input_ids_key = "input_ids"
            self.attention_key = "attention_mask"

        # Prepare other attributes
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id_tensor = (
            torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        )
        self.logits_processor = logits_processor

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """
        # 1. If it is not the first round of candidate generation, prepare the inputs based on the input_ids length
        # (which implicitly contains the number of accepted candidates from the previous round)
        has_past_key_values = self.assistant_kwargs.get("past_key_values", None) is not None
        if has_past_key_values:
            new_cur_len = input_ids.shape[-1]

            new_cache_size = new_cur_len - 1
            self.assistant_kwargs["past_key_values"] = _crop_past_key_values(
                self.assistant_model, self.assistant_kwargs["past_key_values"], new_cache_size - 1
            )  # the assistant does not have the token after the last match, hence the -1

            self.assistant_kwargs = _prepare_attention_mask(
                self.assistant_kwargs, new_cur_len, self.assistant_model.config.is_encoder_decoder
            )
            self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)

        # 2. Forecast next N tokens using the assistant model.
        assistant_generation_kwargs = {
            self.input_ids_key: input_ids,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": int(self.num_assistant_tokens),
            "return_dict_in_generate": True,
            "output_scores": True,
        }
        assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)

        # 3. Update variables for the next round of candidate generation
        self.assistant_kwargs["past_key_values"] = assistant_output.past_key_values

        # 4. Prepare variables for output
        candidate_logits = torch.stack(assistant_output.scores, dim=1)
        candidate_ids = assistant_output.sequences
        return candidate_ids, candidate_logits

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        # Adjust the max number of assistant tokens to use in the next iteration. This is a simple heuristic,
        # probably can be improved -- we want to balance the benefits of getting assistant tokens correct with the
        # cost of forecasting incorrect assistant tokens.
        if self.assistant_model.generation_config.num_assistant_tokens_schedule == "heuristic":
            if num_matches == int(self.num_assistant_tokens):
                self.num_assistant_tokens += 2.0
            else:
                self.num_assistant_tokens = max(1.0, self.num_assistant_tokens - 1.0)


def _crop_past_key_values(model, past_key_values, maximum_length):
    """Crops the past key values up to a certain maximum length."""
    new_past = []
    if model.config.is_encoder_decoder:
        for idx in range(len(past_key_values)):
            new_past.append(
                (
                    past_key_values[idx][0][:, :, :maximum_length, :],
                    past_key_values[idx][1][:, :, :maximum_length, :],
                    past_key_values[idx][2],
                    past_key_values[idx][3],
                )
            )
        past_key_values = tuple(new_past)
    # bloom is special
    elif "bloom" in model.__class__.__name__.lower() or (
        model.config.architectures is not None and "bloom" in model.config.architectures[0].lower()
    ):
        for idx in range(len(past_key_values)):
            new_past.append(
                (
                    past_key_values[idx][0][:, :, :maximum_length],
                    past_key_values[idx][1][:, :maximum_length, :],
                )
            )
        past_key_values = tuple(new_past)
    # gptbigcode is too
    elif "gptbigcode" in model.__class__.__name__.lower() or (
        model.config.architectures is not None and "gptbigcode" in model.config.architectures[0].lower()
    ):
        if model.config.multi_query:
            for idx in range(len(past_key_values)):
                past_key_values[idx] = past_key_values[idx][:, :maximum_length, :]
        else:
            for idx in range(len(past_key_values)):
                past_key_values[idx] = past_key_values[idx][:, :, :maximum_length, :]
    else:
        for idx in range(len(past_key_values)):
            new_past.append(
                (
                    past_key_values[idx][0][:, :, :maximum_length, :],
                    past_key_values[idx][1][:, :, :maximum_length, :],
                )
            )
        past_key_values = tuple(new_past)
    return past_key_values


def _prepare_attention_mask(model_kwargs: Dict[str, Any], new_length: int, is_encoder_decoder: bool) -> Dict[str, Any]:
    """Expands or crops the model's mask for decoding purposes, to the defined length"""

    mask_key = "decoder_attention_mask" if is_encoder_decoder else "attention_mask"
    if mask_key not in model_kwargs:
        return model_kwargs

    mask = model_kwargs[mask_key]
    mask_length_diff = new_length - mask.shape[1]

    if mask_length_diff < 0:
        model_kwargs[mask_key] = mask[:, :mask_length_diff]
    elif mask_length_diff > 0:
        model_kwargs[mask_key] = torch.cat([mask, mask.new_ones((mask.shape[0], mask_length_diff))], dim=-1)
    return model_kwargs


def _prepare_token_type_ids(model_kwargs: Dict[str, Any], new_length: int) -> Dict[str, Any]:
    """Expands or crops the model's token_type_ids for decoding purposes, to the defined length"""
    if "token_type_ids" not in model_kwargs or model_kwargs["token_type_ids"] is None:
        return model_kwargs

    token_type_ids = model_kwargs["token_type_ids"]
    final_token_type = token_type_ids[:, -1].unsqueeze(-1)
    type_length_diff = new_length - token_type_ids.shape[1]

    if type_length_diff < 0:
        token_type_ids = token_type_ids[:, :type_length_diff]
    elif type_length_diff > 0:
        token_type_copies = final_token_type.repeat(1, type_length_diff)
        model_kwargs["token_type_ids"] = torch.cat([model_kwargs["token_type_ids"], token_type_copies], dim=-1)
    return model_kwargs
