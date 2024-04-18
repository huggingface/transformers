# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from ..cache_utils import Cache, DynamicCache, StaticCache
from ..integrations.deepspeed import is_deepspeed_zero3_enabled
from ..modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from ..models.auto import (
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
)
from ..utils import ModelOutput, is_accelerate_available, is_torchdynamo_compiling, logging
from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from .candidate_generator import (
    AssistedCandidateGenerator,
    CandidateGenerator,
    PromptLookupCandidateGenerator,
    _crop_past_key_values,
    _prepare_attention_mask,
    _prepare_token_type_ids,
)
from .configuration_utils import GenerationConfig, GenerationMode
from .logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    MinNewTokensLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SequenceBiasLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from .stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from .streamers import BaseStreamer

logger = logging.get_logger(__name__)

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

NEED_SETUP_CACHE_CLASSES_MAPPING = {
    "static": StaticCache,
}


@dataclass
class GenerateDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True` is passed or when `config.output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


@dataclass
class GenerateEncoderDecoderOutput(ModelOutput):
    """
    Outputs of encoder-decoder generation models, when using non-beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Processed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True` is passed or when `config.output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """

    sequences: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


@dataclass
class GenerateBeamDecoderOnlyOutput(ModelOutput):
    """
    Outputs of decoder-only generation models, when using beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
            Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True` is passed or when `config.output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams, num_heads, generated_length, sequence_length)`.
        hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


@dataclass
class GenerateBeamEncoderDecoderOutput(ModelOutput):
    """
    Outputs of encoder-decoder generation models, when using beam methods.

    Args:
        sequences (`torch.LongTensor` of shape `(batch_size*num_return_sequences, sequence_length)`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter
            if all batches finished early due to the `eos_token_id`.
        sequences_scores (`torch.FloatTensor` of shape `(batch_size*num_return_sequences)`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Final beam scores of the generated `sequences`.
        scores (`tuple(torch.FloatTensor)` *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
            Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
            with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        logits (`tuple(torch.FloatTensor)` *optional*, returned when `output_logits=True` is passed or when `config.output_logits=True`):
            Unprocessed prediction scores of the language modeling head (scores for each vocabulary token before SoftMax)
            at each generation step. Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for
            each generated token), with each tensor of shape `(batch_size, config.vocab_size)`.
        beam_indices (`torch.LongTensor`, *optional*, returned when `output_scores=True` is passed or when `config.output_scores=True`):
            Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer of the decoder) of shape `(batch_size, num_heads,
            sequence_length, sequence_length)`.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size*num_beams*num_return_sequences, sequence_length, hidden_size)`.
        decoder_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, num_heads, generated_length,
            sequence_length)`.
        cross_attentions (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_attentions=True` is passed or `config.output_attentions=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size, num_heads, generated_length, sequence_length)`.
        decoder_hidden_states (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple (one element for each generated token) of tuples (one element for each layer of the decoder) of
            `torch.FloatTensor` of shape `(batch_size*num_beams*num_return_sequences, generated_length, hidden_size)`.
        past_key_values (`tuple(tuple(torch.FloatTensor)))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            NOTE: some models have a different `past_key_values` format, confirm with the model's documentation.
            Usually a Tuple (one element for each layer of the decoder) of tuples (two elements, key tensor and value
            tensor). The first Tuple is of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
    """

    sequences: torch.LongTensor = None
    sequences_scores: Optional[torch.FloatTensor] = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[Tuple[torch.FloatTensor]] = None
    beam_indices: Optional[torch.LongTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    cross_attentions: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_key_values: Optional[Tuple[Tuple[Tuple[torch.FloatTensor]]]] = None


# Equivalent classes (kept for retrocompatibility purposes)
GreedySearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
ContrastiveSearchDecoderOnlyOutput = GenerateDecoderOnlyOutput
SampleDecoderOnlyOutput = GenerateDecoderOnlyOutput

ContrastiveSearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
GreedySearchEncoderDecoderOutput = GenerateEncoderDecoderOutput
SampleEncoderDecoderOutput = GenerateEncoderDecoderOutput

BeamSearchDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput
BeamSampleDecoderOnlyOutput = GenerateBeamDecoderOnlyOutput

BeamSearchEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput
BeamSampleEncoderDecoderOutput = GenerateBeamEncoderDecoderOutput

GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
ContrastiveSearchOutput = Union[ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput]

# Typing shortcuts
GenerateNonBeamOutput = Union[GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput]
GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]
GenerateOutput = Union[GenerateNonBeamOutput, GenerateBeamOutput]


class GenerationMixin:
    """
    A class containing all functions for auto-regressive text generation, to be used as a mixin in [`PreTrainedModel`].

    The class exposes [`~generation.GenerationMixin.generate`], which can be used for:
        - *greedy decoding* by calling [`~generation.GenerationMixin._greedy_search`] if `num_beams=1` and
          `do_sample=False`
        - *contrastive search* by calling [`~generation.GenerationMixin._contrastive_search`] if `penalty_alpha>0` and
          `top_k>1`
        - *multinomial sampling* by calling [`~generation.GenerationMixin._sample`] if `num_beams=1` and
          `do_sample=True`
        - *beam-search decoding* by calling [`~generation.GenerationMixin._beam_search`] if `num_beams>1` and
          `do_sample=False`
        - *beam-search multinomial sampling* by calling [`~generation.GenerationMixin._beam_sample`] if `num_beams>1`
          and `do_sample=True`
        - *diverse beam-search decoding* by calling [`~generation.GenerationMixin._group_beam_search`], if `num_beams>1`
          and `num_beam_groups>1`
        - *constrained beam-search decoding* by calling [`~generation.GenerationMixin._constrained_beam_search`], if
          `constraints!=None` or `force_words_ids!=None`
        - *assisted decoding* by calling [`~generation.GenerationMixin._assisted_decoding`], if
            `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`

    You do not need to call any of the above methods directly. Pass custom parameter values to 'generate' instead. To
    learn more about decoding strategies refer to the [text generation strategies guide](../generation_strategies).
    """

    def prepare_inputs_for_generation(self, *args, **kwargs):
        raise NotImplementedError(
            "A model class needs to define a `prepare_inputs_for_generation` method in order to use `.generate()`."
        )

    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[str], Dict[str, torch.Tensor]]:
        """
        This function extracts the model-specific `inputs` for generation.
        """
        # 1. retrieve all kwargs that are non-None or non-model input related.
        # some encoder-decoder models have different names for model and encoder
        if (
            self.config.is_encoder_decoder
            and hasattr(self, "encoder")
            and self.encoder.main_input_name != self.main_input_name
        ):
            input_name = self.encoder.main_input_name
        else:
            input_name = self.main_input_name

        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        # 2. check whether model_input_name is passed as kwarg
        # if yes and `inputs` is None use kwarg inputs
        inputs_kwarg = model_kwargs.pop(input_name, None)
        if inputs_kwarg is not None and inputs is not None:
            raise ValueError(
                f"`inputs`: {inputs}` were passed alongside {input_name} which is not allowed. "
                f"Make sure to either pass {inputs} or {input_name}=..."
            )
        elif inputs_kwarg is not None:
            inputs = inputs_kwarg

        # 3. In the presence of `inputs_embeds` for text models:
        # - decoder-only models should complain if the user attempts to pass `inputs_embeds`, but the model
        # doesn't have its forwarding implemented. `inputs_embeds` is kept in `model_kwargs` and can coexist with
        # input_ids (`inputs_embeds` will be used in the 1st generation step, as opposed to `input_ids`)
        # - encoder-decoder models should complain if the user attempts to pass `inputs_embeds` and `input_ids`, and
        # pull the former to inputs. It will be used in place of `input_ids` to get the encoder hidden states.
        if input_name == "input_ids" and "inputs_embeds" in model_kwargs:
            if not self.config.is_encoder_decoder:
                has_inputs_embeds_forwarding = "inputs_embeds" in set(
                    inspect.signature(self.prepare_inputs_for_generation).parameters.keys()
                )
                if not has_inputs_embeds_forwarding:
                    raise ValueError(
                        f"You passed `inputs_embeds` to `.generate()`, but the model class {self.__class__.__name__} "
                        "doesn't have its forwarding implemented. See the GPT2 implementation for an example "
                        "(https://github.com/huggingface/transformers/pull/21405), and feel free to open a PR with it!"
                    )
                # In this case, `input_ids` is moved to the `model_kwargs`, so a few automations (like the creation of
                # the attention mask) can rely on the actual model input.
                model_kwargs["input_ids"] = self._maybe_initialize_input_ids_for_generation(
                    inputs, bos_token_id, model_kwargs=model_kwargs
                )
            else:
                if inputs is not None:
                    raise ValueError("You passed `inputs_embeds` and `input_ids` to `.generate()`. Please pick one.")
            inputs, input_name = model_kwargs["inputs_embeds"], "inputs_embeds"

        # 4. if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, input_name, model_kwargs

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.LongTensor:
        """Initializes input ids for generation, if necessary."""
        if inputs is not None:
            return inputs

        encoder_outputs = model_kwargs.get("encoder_outputs")
        if self.config.is_encoder_decoder and encoder_outputs is not None:
            # make dummy input_ids with value -100, as a sanity check ensuring that they won't be used for encoding
            shape = encoder_outputs.last_hidden_state.size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100

        # If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
        # soft-prompting or in multimodal implementations built on top of decoder-only language models.
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break

        if "inputs_embeds" in model_kwargs:
            return torch.ones((batch_size, 0), dtype=torch.long, device=self.device)

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[Union[int, List[int]]],
    ) -> torch.LongTensor:
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
        is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id not in eos_token_id)

        # Check if input is input_ids and padded -> only then is attention_mask defined
        if is_input_ids and is_pad_token_in_inputs and is_pad_token_not_equal_to_eos_token_id:
            return inputs.ne(pad_token_id).long()
        else:
            return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(self, "hf_device_map"):
            if hasattr(encoder, "_hf_hook"):
                encoder._hf_hook.io_same_device = True
            else:
                add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size: int,
        model_input_name: str,
        model_kwargs: Dict[str, torch.Tensor],
        decoder_start_token_id: Union[int, List[int]] = None,
        bos_token_id: int = None,
        device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        if isinstance(decoder_start_token_id, list):
            if len(decoder_start_token_id) != batch_size:
                raise ValueError(
                    f"`decoder_start_token_id` expcted to have length {batch_size} but got {len(decoder_start_token_id)}"
                )
            decoder_input_ids_start = torch.tensor(decoder_start_token_id, dtype=torch.long, device=device)
            decoder_input_ids_start = decoder_input_ids_start.view(-1, 1)
        else:
            decoder_input_ids_start = (
                torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id
            )

        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        # exception: Donut checkpoints have task-specific decoder starts and don't expect a BOS token
        elif self.config.model_type == "vision-encoder-decoder" and "donut" in self.name_or_path.lower():
            pass
        elif self.config.model_type in ["whisper"]:
            pass
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (
            isinstance(decoder_start_token_id, int)
            and (decoder_input_ids[:, 0] != decoder_start_token_id).all().item()
        ) or (
            isinstance(decoder_start_token_id, torch.Tensor)
            and (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all().item()
        ):
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    def _get_decoder_start_token_id(
        self, decoder_start_token_id: Union[int, List[int]] = None, bos_token_id: int = None
    ) -> int:
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.generation_config.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.generation_config.bos_token_id

        if decoder_start_token_id is not None:
            return decoder_start_token_id
        elif bos_token_id is not None:
            return bos_token_id
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    def _extract_past_from_model_output(self, outputs: ModelOutput, standardize_cache_format: bool = False):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states

        # Bloom fix: standardizes the cache format when requested
        if standardize_cache_format and hasattr(self, "_convert_to_standard_cache"):
            batch_size = outputs.logits.shape[0]
            past_key_values = self._convert_to_standard_cache(past_key_values, batch_size=batch_size)
        return past_key_values

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        if "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        raise NotImplementedError(
            f"Make sure that a `_reorder_cache` function is correctly implemented in {self.__class__.__module__} to"
            f" enable beam search for {self.__class__}"
        )

    def _get_candidate_generator(
        self,
        generation_config: GenerationConfig,
        input_ids: torch.LongTensor,
        inputs_tensor: torch.Tensor,
        assistant_model: "PreTrainedModel",
        logits_processor: LogitsProcessorList,
        model_kwargs: Dict,
    ) -> CandidateGenerator:
        """
        Returns the candidate generator to be used in `assisted_generation`
        """
        if generation_config.prompt_lookup_num_tokens is not None:
            candidate_generator = PromptLookupCandidateGenerator(
                num_output_tokens=generation_config.prompt_lookup_num_tokens,
                max_matching_ngram_size=generation_config.max_matching_ngram_size,
                max_length=generation_config.max_length,
            )
        else:
            candidate_generator = AssistedCandidateGenerator(
                input_ids=input_ids,
                assistant_model=assistant_model,
                generation_config=generation_config,
                logits_processor=logits_processor,
                model_kwargs=model_kwargs,
                inputs_tensor=inputs_tensor,
            )
        return candidate_generator

    def _get_logits_warper(
        self,
        generation_config: GenerationConfig,
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = LogitsProcessorList()

        # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
        # better score (i.e. keep len(list(generation_config.eos_token_id)) + 1)
        if generation_config.num_beams > 1:
            if isinstance(generation_config.eos_token_id, list):
                min_tokens_to_keep = len(generation_config.eos_token_id) + 1
            else:
                min_tokens_to_keep = 2
        else:
            min_tokens_to_keep = 1

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if generation_config.temperature is not None and generation_config.temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(generation_config.temperature))
        if generation_config.top_k is not None and generation_config.top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=generation_config.top_k, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.top_p is not None and generation_config.top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=generation_config.top_p, min_tokens_to_keep=min_tokens_to_keep))
        if generation_config.typical_p is not None and generation_config.typical_p < 1.0:
            warpers.append(
                TypicalLogitsWarper(mass=generation_config.typical_p, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.epsilon_cutoff is not None and 0.0 < generation_config.epsilon_cutoff < 1.0:
            warpers.append(
                EpsilonLogitsWarper(epsilon=generation_config.epsilon_cutoff, min_tokens_to_keep=min_tokens_to_keep)
            )
        if generation_config.eta_cutoff is not None and 0.0 < generation_config.eta_cutoff < 1.0:
            warpers.append(
                EtaLogitsWarper(epsilon=generation_config.eta_cutoff, min_tokens_to_keep=min_tokens_to_keep)
            )
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            warpers.append(LogitNormalization())
        return warpers

    def _get_logits_processor(
        self,
        generation_config: GenerationConfig,
        input_ids_seq_length: int,
        encoder_input_ids: torch.LongTensor,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        logits_processor: Optional[LogitsProcessorList],
        model_kwargs: Optional[Dict[str, Any]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorList:
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
        instances used to modify the scores of the language model head.
        """
        # instantiate processors list
        processors = LogitsProcessorList()

        if generation_config.guidance_scale is not None and generation_config.guidance_scale != 1:
            processors.append(
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    generation_config.guidance_scale,
                    self,
                    unconditional_ids=negative_prompt_ids,
                    unconditional_attention_mask=negative_prompt_attention_mask,
                    use_cache=model_kwargs["use_cache"],
                )
            )
        if generation_config.sequence_bias is not None:
            processors.append(SequenceBiasLogitsProcessor(sequence_bias=generation_config.sequence_bias))

        if generation_config.diversity_penalty is not None and generation_config.diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=generation_config.diversity_penalty,
                    num_beams=generation_config.num_beams,
                    num_beam_groups=generation_config.num_beam_groups,
                )
            )
        if (
            generation_config.encoder_repetition_penalty is not None
            and generation_config.encoder_repetition_penalty != 1.0
        ):
            processors.append(
                EncoderRepetitionPenaltyLogitsProcessor(
                    penalty=generation_config.encoder_repetition_penalty, encoder_input_ids=encoder_input_ids
                )
            )
        if generation_config.repetition_penalty is not None and generation_config.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=generation_config.repetition_penalty))
        if generation_config.no_repeat_ngram_size is not None and generation_config.no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(generation_config.no_repeat_ngram_size))
        if (
            generation_config.encoder_no_repeat_ngram_size is not None
            and generation_config.encoder_no_repeat_ngram_size > 0
        ):
            processors.append(
                EncoderNoRepeatNGramLogitsProcessor(generation_config.encoder_no_repeat_ngram_size, encoder_input_ids)
            )
        if generation_config.bad_words_ids is not None:
            processors.append(
                NoBadWordsLogitsProcessor(generation_config.bad_words_ids, generation_config.eos_token_id)
            )
        if (
            generation_config.min_length is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_length > 0
        ):
            processors.append(MinLengthLogitsProcessor(generation_config.min_length, generation_config.eos_token_id))
        if (
            generation_config.min_new_tokens is not None
            and generation_config.eos_token_id is not None
            and generation_config.min_new_tokens > 0
        ):
            processors.append(
                MinNewTokensLengthLogitsProcessor(
                    input_ids_seq_length, generation_config.min_new_tokens, generation_config.eos_token_id
                )
            )
        if prefix_allowed_tokens_fn is not None:
            processors.append(
                PrefixConstrainedLogitsProcessor(
                    prefix_allowed_tokens_fn, generation_config.num_beams // generation_config.num_beam_groups
                )
            )
        if generation_config.forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(generation_config.forced_bos_token_id))
        if generation_config.forced_eos_token_id is not None:
            processors.append(
                ForcedEOSTokenLogitsProcessor(generation_config.max_length, generation_config.forced_eos_token_id)
            )
        if generation_config.remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())
        if generation_config.exponential_decay_length_penalty is not None:
            processors.append(
                ExponentialDecayLengthPenalty(
                    generation_config.exponential_decay_length_penalty,
                    generation_config.eos_token_id,
                    input_ids_seq_length,
                )
            )
        if generation_config.suppress_tokens is not None:
            processors.append(SuppressTokensLogitsProcessor(generation_config.suppress_tokens))
        if generation_config.begin_suppress_tokens is not None:
            begin_index = input_ids_seq_length
            begin_index = (
                begin_index
                if (input_ids_seq_length > 1 or generation_config.forced_bos_token_id is None)
                else begin_index + 1
            )
            if generation_config.forced_decoder_ids is not None:
                # generation starts after the last token that is forced
                begin_index += generation_config.forced_decoder_ids[-1][0]
            processors.append(
                SuppressTokensAtBeginLogitsProcessor(generation_config.begin_suppress_tokens, begin_index)
            )
        if generation_config.forced_decoder_ids is not None:
            # TODO(Sanchit): deprecate in v4.40 by removing this logic
            warnings.warn(
                "You have explicitly specified `forced_decoder_ids`. This functionality has been deprecated and will throw an error in v4.40. Please remove the `forced_decoder_ids` argument in favour of `input_ids` or `decoder_input_ids` respectively.",
                FutureWarning,
            )
            processors.append(ForceTokensLogitsProcessor(generation_config.forced_decoder_ids, _has_warned=True))
        processors = self._merge_criteria_processor_list(processors, logits_processor)
        # `LogitNormalization` should always be the last logit processor, when present
        if generation_config.renormalize_logits is True:
            processors.append(LogitNormalization())
        return processors

    def _get_stopping_criteria(
        self, generation_config: GenerationConfig, stopping_criteria: Optional[StoppingCriteriaList]
    ) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if generation_config.max_length is not None:
            max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
            criteria.append(
                MaxLengthCriteria(
                    max_length=generation_config.max_length,
                    max_position_embeddings=max_position_embeddings,
                )
            )
        if generation_config.max_time is not None:
            criteria.append(MaxTimeCriteria(max_time=generation_config.max_time))
        if generation_config.eos_token_id is not None:
            criteria.append(EosTokenCriteria(eos_token_id=generation_config.eos_token_id))
        criteria = self._merge_criteria_processor_list(criteria, stopping_criteria)
        return criteria

    def _merge_criteria_processor_list(
        self,
        default_list: Union[LogitsProcessorList, StoppingCriteriaList],
        custom_list: Union[LogitsProcessorList, StoppingCriteriaList],
    ) -> Union[LogitsProcessorList, StoppingCriteriaList]:
        if len(custom_list) == 0:
            return default_list
        for default in default_list:
            for custom in custom_list:
                if type(custom) is type(default):
                    object_type = "stopping criteria" if isinstance(custom, StoppingCriteria) else "logits processor"
                    raise ValueError(
                        f"A custom {object_type} of type {type(custom)} with values {custom} has been passed to"
                        f" `.generate()`, but it has already been created with the values {default}. {default} has been"
                        " created by passing the corresponding arguments to generate or by the model's config default"
                        f" values. If you just want to change the default values of {object_type} consider passing"
                        f" them as arguments to `.generate()` instead of using a custom {object_type}."
                    )
        default_list.extend(custom_list)
        return default_list

    def compute_transition_scores(
        self,
        sequences: torch.Tensor,
        scores: Tuple[torch.Tensor],
        beam_indices: Optional[torch.Tensor] = None,
        normalize_logits: bool = False,
    ) -> torch.Tensor:
        """
        Computes the transition scores of sequences given the generation scores (and beam indices, if beam search was
        used). This is a convenient method to quicky obtain the scores of the selected tokens at generation time.

        Parameters:
            sequences (`torch.LongTensor`):
                The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or
                shorter if all batches finished early due to the `eos_token_id`.
            scores (`tuple(torch.FloatTensor)`):
                Transition scores for each vocabulary token at each generation step. Beam transition scores consisting
                of log probabilities of tokens conditioned on log softmax of previously generated tokens in this beam.
                Tuple of `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token),
                with each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
            beam_indices (`torch.LongTensor`, *optional*):
                Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
                `(batch_size*num_return_sequences, sequence_length)`. Only required if a `num_beams>1` at
                generate-time.
            normalize_logits (`bool`, *optional*, defaults to `False`):
                Whether to normalize the logits (which, for legacy reasons, may be unnormalized).

        Return:
            `torch.Tensor`: A `torch.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)` containing
                the transition scores (logits)

        Examples:

        ```python
        >>> from transformers import GPT2Tokenizer, AutoModelForCausalLM
        >>> import numpy as np

        >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> tokenizer.pad_token_id = tokenizer.eos_token_id
        >>> inputs = tokenizer(["Today is"], return_tensors="pt")

        >>> # Example 1: Print the scores for each token generated with Greedy Search
        >>> outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
        >>> transition_scores = model.compute_transition_scores(
        ...     outputs.sequences, outputs.scores, normalize_logits=True
        ... )
        >>> # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
        >>> # encoder-decoder models, like BART or T5.
        >>> input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        >>> generated_tokens = outputs.sequences[:, input_length:]
        >>> for tok, score in zip(generated_tokens[0], transition_scores[0]):
        ...     # | token | token string | log probability | probability
        ...     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
        |   262 |  the     | -1.414 | 24.33%
        |  1110 |  day     | -2.609 | 7.36%
        |   618 |  when    | -2.010 | 13.40%
        |   356 |  we      | -1.859 | 15.58%
        |   460 |  can     | -2.508 | 8.14%

        >>> # Example 2: Reconstruct the sequence scores from Beam Search
        >>> outputs = model.generate(
        ...     **inputs,
        ...     max_new_tokens=5,
        ...     num_beams=4,
        ...     num_return_sequences=4,
        ...     return_dict_in_generate=True,
        ...     output_scores=True,
        ... )
        >>> transition_scores = model.compute_transition_scores(
        ...     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
        ... )
        >>> # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
        >>> # Tip 1: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
        >>> # use case, you might want to recompute it with `normalize_logits=True`.
        >>> # Tip 2: the output length does NOT include the input length
        >>> output_length = np.sum(transition_scores.numpy() < 0, axis=1)
        >>> length_penalty = model.generation_config.length_penalty
        >>> reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
        >>> print(np.allclose(outputs.sequences_scores, reconstructed_scores))
        True
        ```"""
        # 1. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
        # to a beam search approach were the first (and only) beam is always selected
        if beam_indices is None:
            beam_indices = torch.arange(scores[0].shape[0]).view(-1, 1).to(sequences.device)
            beam_indices = beam_indices.expand(-1, len(scores))

        # 2. reshape scores as [batch_size*vocab_size, # generation steps] with # generation steps being
        # seq_len - input_length
        scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)

        # 3. Optionally normalize the logits (across the vocab dimension)
        if normalize_logits:
            scores = scores.reshape(-1, self.config.vocab_size, scores.shape[-1])
            scores = torch.nn.functional.log_softmax(scores, dim=1)
            scores = scores.reshape(-1, scores.shape[-1])

        # 4. cut beam_indices to longest beam length
        beam_indices_mask = beam_indices < 0
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
        beam_indices = beam_indices.clone()[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]

        # 5. Set indices of beams that finished early to 0; such indices will be masked correctly afterwards
        beam_indices[beam_indices_mask] = 0

        # 6. multiply beam_indices with vocab size to gather correctly from scores
        beam_sequence_indices = beam_indices * self.config.vocab_size

        # 7. Define which indices contributed to scores
        cut_idx = sequences.shape[-1] - max_beam_length
        indices = sequences[:, cut_idx:] + beam_sequence_indices

        # 8. Compute scores
        transition_scores = scores.gather(0, indices)

        # 9. Mask out transition_scores of beams that stopped early
        transition_scores[beam_indices_mask] = 0

        return transition_scores

    def _validate_model_class(self):
        """
        Confirms that the model class is compatible with generation. If not, raises an exception that points to the
        right class to use.
        """
        if not self.can_generate():
            generate_compatible_mappings = [
                MODEL_FOR_CAUSAL_LM_MAPPING,
                MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
                MODEL_FOR_VISION_2_SEQ_MAPPING,
                MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
                MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
            ]
            generate_compatible_classes = set()
            for model_mapping in generate_compatible_mappings:
                supported_models = model_mapping.get(type(self.config), default=None)
                if supported_models is not None:
                    generate_compatible_classes.add(supported_models.__name__)
            exception_message = (
                f"The current model class ({self.__class__.__name__}) is not compatible with `.generate()`, as "
                "it doesn't have a language model head."
            )
            if generate_compatible_classes:
                exception_message += f" Please use one of the following classes instead: {generate_compatible_classes}"
            raise TypeError(exception_message)

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Validates model kwargs for generation. Generate argument typos will also be caught here."""
        # If a `Cache` instance is passed, checks whether the model is compatible with it
        if isinstance(model_kwargs.get("past_key_values", None), Cache) and not self._supports_cache_class:
            raise ValueError(
                f"{self.__class__.__name__} does not support an instance of `Cache` as `past_key_values`. Please "
                "check the model documentation for supported cache formats."
            )

        # Excludes arguments that are handled before calling any model function
        if self.config.is_encoder_decoder:
            for key in ["decoder_input_ids"]:
                model_kwargs.pop(key, None)

        unused_model_args = []
        model_args = set(inspect.signature(self.prepare_inputs_for_generation).parameters)
        # `kwargs`/`model_kwargs` is often used to handle optional forward pass inputs like `attention_mask`. If
        # `prepare_inputs_for_generation` doesn't accept them, then a stricter check can be made ;)
        if "kwargs" in model_args or "model_kwargs" in model_args:
            model_args |= set(inspect.signature(self.forward).parameters)

        # Encoder-Decoder models may also need Encoder arguments from `model_kwargs`
        if self.config.is_encoder_decoder:
            base_model = getattr(self, self.base_model_prefix, None)

            # allow encoder kwargs
            encoder = getattr(self, "encoder", None)
            # `MusicgenForConditionalGeneration` has `text_encoder` and `audio_encoder`.
            # Also, it has `base_model_prefix = "encoder_decoder"` but there is no `self.encoder_decoder`
            # TODO: A better way to handle this.
            if encoder is None and base_model is not None:
                encoder = getattr(base_model, "encoder", None)

            if encoder is not None:
                encoder_model_args = set(inspect.signature(encoder.forward).parameters)
                model_args |= encoder_model_args

            # allow decoder kwargs
            decoder = getattr(self, "decoder", None)
            if decoder is None and base_model is not None:
                decoder = getattr(base_model, "decoder", None)

            if decoder is not None:
                decoder_model_args = set(inspect.signature(decoder.forward).parameters)
                model_args |= {f"decoder_{x}" for x in decoder_model_args}

            # allow assistant_encoder_outputs to be passed if we're doing assisted generating
            if "assistant_encoder_outputs" in model_kwargs:
                model_args |= {"assistant_encoder_outputs"}

        for key, value in model_kwargs.items():
            if value is not None and key not in model_args:
                unused_model_args.append(key)

        if unused_model_args:
            raise ValueError(
                f"The following `model_kwargs` are not used by the model: {unused_model_args} (note: typos in the"
                " generate arguments will also show up in this list)"
            )

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

        # 2. Min length warnings due to unfeasible parameter combinations
        min_length_error_suffix = (
            " Generation will stop at the defined maximum length. You should decrease the minimum length and/or "
            "increase the maximum length."
        )
        if has_default_max_length:
            min_length_error_suffix += (
                f" Note that `max_length` is set to {generation_config.max_length}, its default value."
            )
        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            warnings.warn(
                f"Unfeasible length constraints: `min_length` ({generation_config.min_length}) is larger than"
                f" the maximum possible length ({generation_config.max_length})." + min_length_error_suffix,
                UserWarning,
            )
        if generation_config.min_new_tokens is not None:
            min_length = generation_config.min_new_tokens + input_ids_length
            if min_length > generation_config.max_length:
                warnings.warn(
                    f"Unfeasible length constraints: `min_new_tokens` ({generation_config.min_new_tokens}), when "
                    f"added to the prompt length ({input_ids_length}), is larger than"
                    f" the maximum possible length ({generation_config.max_length})." + min_length_error_suffix,
                    UserWarning,
                )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        has_default_min_length,
        model_input_name,
        input_ids_length,
        inputs_tensor,
    ):
        """Prepared max and min length in generaion configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        # if both `inputs_embeds` and `input_ids` are passed, we do not correct the length
        # otherwise we need total length [inputs-embeds-len + new-tokens-len] to not go beyond indicated `max_length``
        elif (
            model_input_name == "inputs_embeds"
            and input_ids_length != inputs_tensor.shape[1]
            and not self.config.is_encoder_decoder
        ):
            generation_config.max_length -= inputs_tensor.shape[1]

        # same for min length
        if generation_config.min_new_tokens is not None:
            if not has_default_min_length:
                logger.warning(
                    f"Both `min_new_tokens` (={generation_config.min_new_tokens}) and `min_length`(="
                    f"{generation_config.min_length}) seem to have been set. `min_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.min_length = generation_config.min_new_tokens + input_ids_length

        elif (
            model_input_name == "inputs_embeds"
            and input_ids_length != inputs_tensor.shape[1]
            and not self.config.is_encoder_decoder
        ):
            generation_config.min_length = max(generation_config.min_length - inputs_tensor.shape[1], 0)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: GenerationConfig, **kwargs: Dict
    ) -> Tuple[GenerationConfig, Dict]:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs.
        """
        # TODO joao: when we can detect `fullgraph=True` in `torch.compile` (https://github.com/pytorch/pytorch/pull/120400)
        # replace `is_torchdynamo_compiling` by the corresponding check. As it is, we are being too restrictive with
        # the parameterization in `fullgraph=False` so as to enable `fullgraph=True`.

        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        if generation_config is None:
            # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
            # three conditions must be met
            # 1) the generation config must have been created from the model config (`_from_model_config` field);
            # 2) the generation config must have seen no modification since its creation (the hash is the same);
            # 3) the user must have set generation parameters in the model config.
            # NOTE: `torch.compile` can't compile `hash`, this legacy support is disabled with compilation.
            if (
                not is_torchdynamo_compiling()
                and self.generation_config._from_model_config
                and self.generation_config._original_object_hash == hash(self.generation_config)
                and self.config._has_non_default_generation_parameters()
            ):
                new_generation_config = GenerationConfig.from_model_config(self.config)
                if new_generation_config != self.generation_config:
                    warnings.warn(
                        "You have modified the pretrained model configuration to control generation. This is a"
                        " deprecated strategy to control generation and will be removed soon, in a future version."
                        " Please use and modify the model generation configuration (see"
                        " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
                    )
                    self.generation_config = new_generation_config
            generation_config = self.generation_config

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled.
        if is_torchdynamo_compiling():
            model_kwargs = kwargs
            generate_attributes_in_kwargs = [
                key for key, value in kwargs.items() if getattr(generation_config, key, None) != value
            ]
            if len(generate_attributes_in_kwargs) > 0:
                raise ValueError(
                    "`torch.compile` exception: all generation configuration attributes must be passed within a "
                    f"`generation_config` instance passed to `generate` (found: {generate_attributes_in_kwargs})."
                )
        else:
            generation_config = copy.deepcopy(generation_config)
            model_kwargs = generation_config.update(**kwargs)

        return generation_config, model_kwargs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        r"""

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complements the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
                sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
                intended for advanced users.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            synced_gpus (`bool`, *optional*):
                Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
                `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
                generating before other GPUs. Otherwise it'll be set to `False`.
            assistant_model (`PreTrainedModel`, *optional*):
                An assistant model that can be used to accelerate generation. The assistant model must have the exact
                same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
                is much faster than running generation with the model you're calling generate from. As such, the
                assistant model should be much smaller.
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The negative prompt needed for some processors such as CFG. The batch size must match the input batch
                size. This is an experimental feature, subject to breaking API changes in future versions.
            negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Attention_mask for `negative_prompt_ids`.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            generation_config.pad_token_id = eos_token_id

        # 3. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        # 4. Define other model kwargs
        model_kwargs["output_attentions"] = generation_config.output_attentions
        model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            model_kwargs["use_cache"] = generation_config.use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
            )

        # decoder-only models should use left-padding for generation
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config.pad_token_id is not None
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config.decoder_start_token_id,
                bos_token_id=generation_config.bos_token_id,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
            if generation_config.cache_implementation == "static":
                if model_kwargs.get("past_key_values", False) is not False:
                    raise ValueError(
                        "Using `past_key_values` argument with `generate()` when using a static KV cache is not supported. Please open an issue in Transformers GitHub repository."
                    )
                cache_cls = NEED_SETUP_CACHE_CLASSES_MAPPING["static"]
                if not callable(getattr(self, "_setup_cache", None)):
                    raise ValueError(
                        "The `generation_config` defines a `cache_implementation` that is not compatible with this model."
                        " Make sure it has a `_setup_cache` function."
                    )
                self._setup_cache(cache_cls, max_batch_size=batch_size, max_cache_len=generation_config.max_length)

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        # 9. prepare stopping criteria
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria
        )
        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")

            # 11. Get the candidate generator, given the parameterization
            candidate_generator = self._get_candidate_generator(
                generation_config=generation_config,
                input_ids=input_ids,
                inputs_tensor=inputs_tensor,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                model_kwargs=model_kwargs,
            )

            # 12. run assisted generate
            result = self._assisted_decoding(
                input_ids,
                candidate_generator=candidate_generator,
                do_sample=generation_config.do_sample,
                logits_processor=prepared_logits_processor,
                logits_warper=self._get_logits_warper(generation_config) if generation_config.do_sample else None,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        if generation_mode == GenerationMode.GREEDY_SEARCH:
            # 11. run greedy search
            result = self._greedy_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")

            result = self._contrastive_search(
                input_ids,
                top_k=generation_config.top_k,
                penalty_alpha=generation_config.penalty_alpha,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                sequential=generation_config.low_memory,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.SAMPLE:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run sample
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                sequential=generation_config.low_memory,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.BEAM_SAMPLE:
            # 11. prepare logits warper
            logits_warper = self._get_logits_warper(generation_config)

            # 12. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )

            # 13. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 14. run beam sample
            result = self._beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                pad_token_id=generation_config.pad_token_id,
                output_scores=generation_config.output_scores,
                output_logits=generation_config.output_logits,
                return_dict_in_generate=generation_config.return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
            if not callable(getattr(self, "_reset_cache", None)):
                raise ValueError(
                    "A `static_cache` was used to generate but there was a failure when trying to  release the cache. "
                    " Make sure this model implements a `_reset_cache` function."
                )
            self._reset_cache()

        return result

    def _has_unfinished_sequences(self, this_peer_finished: bool, synced_gpus: bool, device: torch.device) -> bool:
        """
        Returns whether there are still unfinished sequences in the device. The existence of unfinished sequences is
        fed through `this_peer_finished`. ZeRO stage 3-friendly.
        """
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                return False
        elif this_peer_finished:
            return False
        return True

    def contrastive_search(self, *args, **kwargs):
        logger.warning_once(
            "Calling `contrastive_search` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        return self._contrastive_search(*args, **kwargs)

    @torch.no_grad()
    def _contrastive_search(
        self,
        input_ids: torch.LongTensor,
        top_k: Optional[int] = 1,
        penalty_alpha: Optional[float] = 0,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        sequential: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **contrastive search** and can
        be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin._contrastive_search`] directly. Use
        generate() instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            top_k (`int`, *optional*, defaults to 1):
                The size of the candidate set that is used to re-rank for contrastive search
            penalty_alpha (`float`, *optional*, defaults to 0):
                The degeneration penalty for contrastive search; activate when it is larger than 0
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors
                for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            sequential (`bool`, *optional*):
                Switches topk hidden state computation from parallel to sequential to reduce memory if True.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:
        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        >>> model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> # set pad_token_id to eos_token_id because OPT does not have a PAD token
        >>> model.config.pad_token_id = model.config.eos_token_id
        >>> input_prompt = "DeepMind Company is"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt")
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=64)])
        >>> outputs = model._contrastive_search(
        ...     **input_ids, penalty_alpha=0.6, top_k=4, stopping_criteria=stopping_criteria
        ... )
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['DeepMind Company is a company that focuses on the development and commercialization of artificial intelligence (AI). DeepMind’s mission is to help people understand and solve problems that are difficult to solve in the world today.\n\nIn this post, we talk about the benefits of deep learning in business and how it']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        if eos_token_id is not None:
            logger.warning_once(
                "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
                " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
                " Otherwise make sure to set `model.generation_config.eos_token_id`",
                FutureWarning,
            )
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # TODO remove when the method is totally private
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        sequential = sequential if sequential is not None else self.generation_config.low_memory
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        this_peer_finished = False

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # if the first step in the loop, encode all the prefix and obtain: (1) past_key_values;
            # (2) last_hidden_states; (3) logit_for_next_step; (4) update model kwargs for the next step
            if model_kwargs.get("past_key_values") is None:
                # prepare inputs
                model_kwargs["use_cache"] = True
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # encode the given prefix and prepare model inputs; encoder-decoder model process the prefix and save
                # the `encoder_outputs`
                outputs = self(
                    **model_inputs, return_dict=True, output_hidden_states=True, output_attentions=output_attentions
                )

                # last decoder hidden states will be used to compute the degeneration penalty (cosine similarity with
                # previous tokens)
                if self.config.is_encoder_decoder:
                    last_hidden_states = outputs.decoder_hidden_states[-1]
                else:
                    last_hidden_states = outputs.hidden_states[-1]

                # next logit for contrastive search to select top-k candidate tokens
                logit_for_next_step = outputs.logits[:, -1, :]

                model_kwargs = self._update_model_kwargs_for_generation(
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=self.config.is_encoder_decoder,
                    standardize_cache_format=True,
                )
                if not sequential:
                    # Expands model inputs top_k times, for batched forward passes (akin to beam search).
                    _, model_kwargs = self._expand_inputs_for_generation(
                        expand_size=top_k, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
                    )

                past_key_values = model_kwargs.get("past_key_values")
                if past_key_values is None:
                    raise ValueError(
                        f"{self.__class__.__name__} does not support caching and therefore **can't** be used "
                        "for contrastive search."
                    )
                elif (
                    not isinstance(past_key_values[0], (tuple, torch.Tensor))
                    or past_key_values[0][0].shape[0] != batch_size
                ):
                    raise ValueError(
                        f"{self.__class__.__name__} does not have a standard cache format and therefore **can't** be "
                        "used for contrastive search without further modifications."
                    )

            # contrastive_search main logic start:
            # contrastive search decoding consists of two steps: (1) candidate tokens recall; (2) candidate re-rank by
            # degeneration penalty
            processed_logit_for_next_step = logits_processor(input_ids, logit_for_next_step)
            processed_logit_for_next_step = logits_warper(input_ids, processed_logit_for_next_step)
            next_probs = nn.functional.softmax(processed_logit_for_next_step, dim=-1)

            top_k_probs, top_k_ids = torch.topk(next_probs, dim=-1, k=top_k)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_logits:
                    raw_logits += (logit_for_next_step,)
                if output_scores:
                    scores += (processed_logit_for_next_step,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # Replicates the new past_key_values to match the `top_k` candidates
            new_key_values = []
            past = model_kwargs["past_key_values"]
            for layer in past:
                items = []
                # item is either the key or the value matrix
                for item in layer:
                    if sequential:
                        items.append(item.repeat_interleave(1, dim=0))
                    else:
                        items.append(item.repeat_interleave(top_k, dim=0))
                new_key_values.append(tuple(items))
            if not isinstance(past, DynamicCache):
                past = tuple(new_key_values)
            else:
                for layer_idx in range(len(new_key_values)):
                    past.key_cache[layer_idx] = new_key_values[layer_idx][0]
                    past.value_cache[layer_idx] = new_key_values[layer_idx][1]
            model_kwargs["past_key_values"] = past

            if sequential:
                all_outputs = []
                for i in range(top_k):
                    # compute the candidate tokens by the language model and collect their hidden_states
                    next_model_inputs = self.prepare_inputs_for_generation(top_k_ids[:, i].view(-1, 1), **model_kwargs)

                    outputs = self(
                        **next_model_inputs,
                        return_dict=True,
                        output_hidden_states=True,
                        output_attentions=output_attentions,
                    )
                    all_outputs.append(outputs)
                outputs = stack_model_outputs(all_outputs)

            else:
                # compute the candidate tokens by the language model and collect their hidden_states
                # assembles top_k_ids into batch of size k
                next_model_inputs = self.prepare_inputs_for_generation(top_k_ids.view(-1, 1), **model_kwargs)

                outputs = self(
                    **next_model_inputs,
                    return_dict=True,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                )
            # name is different for encoder-decoder and decoder-only models
            if self.config.is_encoder_decoder:
                next_hidden = outputs.decoder_hidden_states[-1]
                full_hidden_states = outputs.decoder_hidden_states
            else:
                next_hidden = outputs.hidden_states[-1]
                full_hidden_states = outputs.hidden_states

            logits = outputs.logits[:, -1, :]

            context_hidden = last_hidden_states.repeat_interleave(top_k, dim=0)

            # compute the degeneration penalty and re-rank the candidates based on the degeneration penalty and the
            # model confidence. Keeping `selected_idx` on CPU enables multi-device contrastive search and doesn't
            # introduce (noticeable) slowdowns on single-device runs.
            selected_idx = _ranking_fast(context_hidden, next_hidden, top_k_probs, penalty_alpha, top_k)
            selected_idx = selected_idx.to("cpu")

            # prepare for the next step: (1) next token_id; (2) past_key_values; (3) last_hidden_states for computing
            # the degeneration penalty; (4) logits for selecting next top-k candidates; (5) selected tokens scores
            # (model confidence minus degeneration penalty); (6) decoder hidden_states
            next_tokens = top_k_ids[range(len(top_k_ids)), selected_idx]
            next_hidden = torch.stack(torch.split(next_hidden.squeeze(dim=1), top_k))
            next_hidden = next_hidden[range(batch_size), selected_idx, :]
            last_hidden_states = torch.cat([last_hidden_states, next_hidden.unsqueeze(1)], dim=1)

            next_decoder_hidden_states = ()
            for layer in full_hidden_states:
                layer = torch.stack(torch.split(layer, top_k))[range(batch_size), selected_idx, :]
                next_decoder_hidden_states += (layer,)

            # generate past_key_values cache of only the selected token
            if sequential:
                next_model_input = self.prepare_inputs_for_generation(
                    top_k_ids[:, selected_idx].view(-1, 1), **model_kwargs
                )

                selected_outputs = self(
                    **next_model_input,
                    return_dict=True,
                    output_hidden_states=False,
                    output_attentions=False,
                )
                next_past_key_values = selected_outputs["past_key_values"]

            else:
                next_past_key_values = self._extract_past_from_model_output(outputs, standardize_cache_format=True)
                new_key_values = []
                for layer in next_past_key_values:
                    items = []
                    # item is either the key or the value matrix
                    for item in layer:
                        item = torch.stack(torch.split(item, top_k, dim=0))  # [B, K, num_head, seq_len, esz]
                        item = item[range(batch_size), selected_idx, ...]  # [B, num_head, seq_len, esz]
                        items += [item]
                    new_key_values += [items]

                if not isinstance(next_past_key_values, DynamicCache):
                    next_past_key_values = tuple(new_key_values)
                else:
                    for layer_idx in range(len(new_key_values)):
                        next_past_key_values.key_cache[layer_idx] = new_key_values[layer_idx][0]
                        next_past_key_values.value_cache[layer_idx] = new_key_values[layer_idx][1]

            logit_for_next_step = torch.stack(torch.split(logits, top_k))[range(batch_size), selected_idx, :]

            # Rebuilds the relevant parts of the model output for the selected token, for use in the next iteration
            if self.config.is_encoder_decoder:
                next_step_cross_attentions = ()
                next_step_decoder_attentions = ()
                if output_attentions:
                    for layer in outputs.cross_attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, ...]
                        next_step_cross_attentions += (layer,)
                    for layer in outputs.decoder_attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, ...]
                        next_step_decoder_attentions += (layer,)
                outputs = Seq2SeqLMOutput(
                    past_key_values=next_past_key_values,
                    decoder_hidden_states=next_decoder_hidden_states,
                    decoder_attentions=next_step_decoder_attentions or None,
                    cross_attentions=next_step_cross_attentions or None,
                )
            else:
                next_step_attentions = ()
                if output_attentions:
                    for layer in outputs.attentions:
                        layer = torch.stack(torch.split(layer, top_k, dim=0))[range(batch_size), selected_idx, ...]
                        next_step_attentions += (layer,)
                outputs = CausalLMOutputWithPast(
                    past_key_values=next_past_key_values,
                    hidden_states=next_decoder_hidden_states,
                    attentions=next_step_attentions or None,
                )
            # contrastive_search main logic end

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # stop when each sentence is finished
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            # Contrastive search works by forward looking at the next token, so we need to exclude it from
            # `past_key_values` to be consistent with the other decoding methods
            if model_kwargs.get("past_key_values") is not None:
                past_key_values = []
                for layer in model_kwargs["past_key_values"]:
                    layer_past_key_values = []
                    for item in layer:
                        layer_past_key_values.append(item[..., :-1, :])
                    past_key_values.append(tuple(layer_past_key_values))
                model_kwargs["past_key_values"] = tuple(past_key_values)

            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def greedy_search(self, *args, **kwargs):
        logger.warning_once(
            "Calling `greedy_search` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        return self._greedy_search(*args, **kwargs)

    def _greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin._greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors
                for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model._greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        if eos_token_id is not None:
            logger.warning_once(
                "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
                " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
                " Otherwise make sure to set `model.generation_config.eos_token_id`",
                FutureWarning,
            )
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # TODO remove when the method is totally private
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def sample(self, *args, **kwargs):
        logger.warning_once(
            "Calling `sample` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        return self._sample(*args, **kwargs)

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin._sample`] directly. Use generate() instead.
        For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id
        >>> model.generation_config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        >>> outputs = model._sample(
        ...     input_ids,
        ...     logits_processor=logits_processor,
        ...     logits_warper=logits_warper,
        ...     stopping_criteria=stopping_criteria,
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        if eos_token_id is not None:
            logger.warning_once(
                "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
                " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
                " Otherwise make sure to set `model.generation_config.eos_token_id`",
                FutureWarning,
            )
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # TODO remove when the method is totally private
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    def _temporary_reorder_cache(self, past_key_values, beam_idx):
        """
        Temporary function to handle the different types of cache reordering processes while we roll out `Cache`.

        TODO: standardize cache formats and make all models compatible with `Cache`. It would remove the need
        for this function, with `Cache.reorder_cache` being the sole remaining code path
        """
        model_class = self.__class__.__name__.lower()
        # Exception 1: code path for models using the legacy cache format
        if isinstance(past_key_values, (tuple, list)):
            past_key_values = self._reorder_cache(past_key_values, beam_idx)
        # Exception 2: models with different cache formats. These are limited to `DynamicCache` until their
        # cache format is standardized, to avoid adding complexity to the codebase.
        elif "bloom" in model_class or "gptbigcode" in model_class:
            if not isinstance(past_key_values, DynamicCache):
                raise ValueError(
                    f"Using an unsupported cache format with {model_class}. Currently, it only supports the "
                    "legacy tuple format or `DynamicCache`"
                )
            past_key_values = self._reorder_cache(past_key_values, beam_idx)
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        # Standard code path: use the `Cache.reorder_cache`
        else:
            past_key_values.reorder_cache(beam_idx)
        return past_key_values

    def beam_search(self, *args, **kwargs):
        logger.warning_once(
            "Calling `beam_search` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        return self._beam_search(*args, **kwargs)

    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        sequential: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin._beam_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            sequential (`bool`, defaults to `False`):
                By default, beam search has `batch_size * num_beams` as effective batch size (see `beam_search()` for
                more details). This flag will avoid parallelizing the beam search and will instead run beam search
                sequentially.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model._beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        sequential = sequential if sequential is not None else self.generation_config.low_memory
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        if eos_token_id is not None:
            logger.warning_once(
                "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
                " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
                " Otherwise make sure to set `model.generation_config.eos_token_id`",
                FutureWarning,
            )
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # TODO remove when the method is totally private and beam scorer refactored
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # if sequential is True, split the input to batches of batch_size and run sequentially
            if sequential:
                if any(
                    model_name in self.__class__.__name__.lower()
                    for model_name in [
                        "fsmt",
                        "reformer",
                        "bloom",
                        "ctrl",
                        "gpt_bigcode",
                        "transo_xl",
                        "xlnet",
                        "cpm",
                        "jamba",
                    ]
                ):
                    raise RuntimeError(
                        f"Currently generation for {self.__class__.__name__} is not supported "
                        f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                    )

                inputs_per_sub_batches = _split_model_inputs(
                    model_inputs, split_size=batch_size, full_batch_size=batch_beam_size
                )
                outputs_per_sub_batch = [
                    self(
                        **inputs_per_sub_batch,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                    for inputs_per_sub_batch in inputs_per_sub_batches
                ]

                outputs = stack_model_outputs(outputs_per_sub_batch)

            else:  # Unchanged original behavior
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    def beam_sample(self, *args, **kwargs):
        logger.warning_once(
            "Calling `beam_sample` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        return self._beam_sample(*args, **kwargs)

    def _beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search multinomial
        sampling** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin._beam_sample`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     max_length=model.config.max_length,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> outputs = model._beam_sample(
        ...     input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        if eos_token_id is not None:
            logger.warning_once(
                "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
                " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
                " Otherwise make sure to set `model.generation_config.eos_token_id`",
                FutureWarning,
            )
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # TODO remove when the method is totally private and beam scorer refactored
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores_processed = logits_warper(input_ids, next_token_scores_processed)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = nn.functional.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    def group_beam_search(self, *args, **kwargs):
        logger.warning_once(
            "Calling `group_beam_search` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        return self._group_beam_search(*args, **kwargs)

    def _group_beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ):
        r"""
        Generates sequences of token ids for models with a language modeling head using **diverse beam search
        decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin._group_beam_search`] directly. Use
        generate() instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)

            model_kwargs:
                Additional model specific kwargs that will be forwarded to the `forward` function of the model. If
                model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     HammingDiversityLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run diverse beam search using 6 beams
        >>> num_beams = 6
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     max_length=model.config.max_length,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ...     num_beam_groups=3,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         HammingDiversityLogitsProcessor(5.5, num_beams=6, num_beam_groups=3),
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model._group_beam_search(
        ...     input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        if eos_token_id is not None:
            logger.warning_once(
                "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
                " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
                " Otherwise make sure to set `model.generation_config.eos_token_id`",
                FutureWarning,
            )
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # TODO remove when the method is totally private and beam scorer refactored
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        num_beams = beam_scorer.num_beams
        num_beam_groups = beam_scorer.num_beam_groups
        num_sub_beams = num_beams // num_beam_groups
        batch_size = len(beam_scorer._beam_hyps) // num_beam_groups
        device = input_ids.device

        batch_beam_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        if return_dict_in_generate and output_scores:
            beam_indices = [tuple(() for _ in range(num_sub_beams * batch_size)) for _ in range(num_beam_groups)]
        else:
            beam_indices = None

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam of each group with 0 and the rest with -1e9. This ensures that the beams in
        # the same group don't produce same tokens everytime.
        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            # do one decoder step on all beams of all sentences in batch
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            if output_scores:
                processed_score = torch.zeros_like(outputs.logits[:, -1, :])
            if output_logits:
                raw_logit_score = outputs.logits[:, -1, :]

            for beam_group_idx in range(num_beam_groups):
                group_start_idx = beam_group_idx * num_sub_beams
                group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                group_size = group_end_idx - group_start_idx

                # indices of beams of current group among all sentences in batch
                batch_group_indices = []

                for batch_idx in range(batch_size):
                    batch_group_indices.extend(
                        [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                    )
                group_input_ids = input_ids[batch_group_indices]

                # select outputs of beams of current group only
                next_token_logits = outputs.logits[batch_group_indices, -1, :]

                next_token_scores = nn.functional.log_softmax(
                    next_token_logits, dim=-1
                )  # (batch_size * group_size, vocab_size)
                vocab_size = next_token_scores.shape[-1]

                next_token_scores_processed = logits_processor(
                    group_input_ids, next_token_scores, current_tokens=current_tokens, beam_group_idx=beam_group_idx
                )
                next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                next_token_scores = next_token_scores.expand_as(next_token_scores_processed)

                if output_scores:
                    processed_score[batch_group_indices] = next_token_scores_processed

                # reshape for beam search
                next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
                n_eos_tokens = len(eos_token_id) if eos_token_id else 0
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, max(2, 1 + n_eos_tokens) * group_size, dim=1, largest=True, sorted=True
                )

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                process_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
                beam_outputs = beam_scorer.process(
                    group_input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    beam_indices=process_beam_indices,
                    group_index=beam_group_idx,
                    decoder_prompt_len=decoder_prompt_len,
                )
                beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                if return_dict_in_generate and output_scores:
                    beam_indices[beam_group_idx] = tuple(
                        beam_indices[beam_group_idx][beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices[0]))
                    )

                input_ids[batch_group_indices] = group_input_ids[beam_idx]
                group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                current_tokens[batch_group_indices] = group_input_ids[:, -1]

                # (beam_idx // group_size) -> batch_idx
                # (beam_idx % group_size) -> offset of idx inside the group
                reordering_indices[batch_group_indices] = (
                    num_beams * torch.div(beam_idx, group_size, rounding_mode="floor")
                    + group_start_idx
                    + (beam_idx % group_size)
                )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (processed_score,)
                if output_logits:
                    raw_logits += (raw_logit_score,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], reordering_indices
                )

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        final_beam_indices = sum(beam_indices, ()) if beam_indices is not None else None
        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=final_beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    def constrained_beam_search(self, *args, **kwargs):
        logger.warning_once(
            "Calling `constrained_beam_search` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        return self._constrained_beam_search(*args, **kwargs)

    def _constrained_beam_search(
        self,
        input_ids: torch.LongTensor,
        constrained_beam_scorer: ConstrainedBeamSearchScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **constrained beam search
        decoding** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin._constrained_beam_search`] directly. Use
        generate() instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            constrained_beam_scorer (`ConstrainedBeamSearchScorer`):
                A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation, while satisfying a list of positive constraints. For more information, the
                documentation of [`ConstrainedBeamSearchScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     ConstrainedBeamSearchScorer,
        ...     PhrasalConstraint,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> constraint_str = "Sie"
        >>> constraint_token_ids = tokenizer.encode(constraint_str)[:-1]  # slice to remove eos token
        >>> constraints = [PhrasalConstraint(token_ids=constraint_token_ids)]


        >>> # instantiate beam scorer
        >>> beam_scorer = ConstrainedBeamSearchScorer(
        ...     batch_size=1, num_beams=num_beams, device=model.device, constraints=constraints
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model._constrained_beam_search(
        ...     input_ids, beam_scorer, constraints=constraints, logits_processor=logits_processor, **model_kwargs
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt sind Sie?']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        if eos_token_id is not None:
            logger.warning_once(
                "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
                " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
                " Otherwise make sure to set `model.generation_config.eos_token_id`",
                FutureWarning,
            )
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # TODO remove when the method is totally private and beam scorer refactored
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(constrained_beam_scorer._beam_hyps)
        num_beams = constrained_beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)

            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            scores_for_all_vocab = next_token_scores.clone()

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = (next_tokens / vocab_size).long()
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = constrained_beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                scores_for_all_vocab,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if constrained_beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = constrained_beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"]

    def assisted_decoding(self, *args, **kwargs):
        logger.warning_once(
            "Calling `_assisted_decoding` directly is deprecated and will be removed in v4.41. Use `generate` or a "
            "custom generation loop instead.",
        )
        return self._assisted_decoding(*args, **kwargs)

    def _assisted_decoding(
        self,
        input_ids: torch.LongTensor,
        candidate_generator: Optional["CandidateGenerator"] = None,
        do_sample: bool = False,
        logits_processor: Optional[LogitsProcessorList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        output_logits: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** or
        **sample** (depending on `do_sample`), assisted by candidate sequences. Assisted generation is an example of a
        candidate decoding strategy. Can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text
        models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin._assisted_decoding`] directly. Use
        generate() instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            candidate_generator (`CandidateGenerator`, *optional*):
                A derived instance of [`CandidateGenerator`] that defines how candidate sequences are generated. For
                more information, the documentation of [`CandidateGenerator`] should be read.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )
        >>> from transformers.generation import AssistedCandidateGenerator

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> assistant_model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id
        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
        >>> candidate_generator = AssistedCandidateGenerator(
        ...     input_ids=input_ids,
        ...     assistant_model=assistant_model,
        ...     generation_config=model.generation_config,
        ...     logits_processor=logits_processor,
        ...     model_kwargs={},
        ... )
        >>> outputs = model._assisted_decoding(
        ...     input_ids,
        ...     candidate_generator=candidate_generator,
        ...     logits_processor=logits_processor,
        ...     stopping_criteria=stopping_criteria,
        ... )
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        if eos_token_id is not None:
            logger.warning_once(
                "`eos_token_id` is deprecated in this function and will be removed in v4.41, use"
                " `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead."
                " Otherwise make sure to set `model.generation_config.eos_token_id`",
                FutureWarning,
            )
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
        else:
            # TODO remove when the method is totally private and beam scorer refactored
            # need to get `eos_token_id` and add stopping criteria, so that generation does not go forever
            eos_token_id = [
                criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, "eos_token_id")
            ]
            eos_token_id = eos_token_id[0] if eos_token_id else None
            if eos_token_id is None and self.generation_config.eos_token_id is not None:
                eos_token_id = self.generation_config.eos_token_id
                stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)

        this_peer_finished = False
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            cur_len = input_ids.shape[-1]

            #  1. Fetch candidate sequences from a `CandidateGenerator`
            candidate_input_ids, candidate_logits = candidate_generator.get_candidates(input_ids)
            candidate_input_ids = candidate_input_ids.to(self.device)
            if candidate_logits is not None:
                candidate_logits = candidate_logits.to(self.device)

            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
            is_done_candidate = stopping_criteria(candidate_input_ids, None)

            # 2. Use the original model to obtain the next token logits given the candidate sequence. We obtain
            # `candidate_length + 1` relevant logits from this process: in the event that all candidates are correct,
            # we use this forward pass to also pick the subsequent logits in the original model.

            # 2.1. Prepare the model inputs
            model_kwargs = _prepare_attention_mask(
                model_kwargs, candidate_input_ids.shape[1], self.config.is_encoder_decoder
            )
            model_kwargs = _prepare_token_type_ids(model_kwargs, candidate_input_ids.shape[1])
            if "cache_position" in model_kwargs:
                model_kwargs["cache_position"] = torch.cat(
                    (
                        model_kwargs["cache_position"],
                        torch.arange(cur_len, cur_len + candidate_length, device=input_ids.device, dtype=torch.long),
                    ),
                    dim=0,
                )

            model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **model_kwargs)
            if "num_logits_to_keep" in model_inputs:
                model_inputs["num_logits_to_keep"] = candidate_length + 1

            # 2.2. Run a forward pass on the candidate sequence
            outputs = self(
                **model_inputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # 2.3. Process the new logits
            new_logits = outputs.logits[:, -candidate_length - 1 :]  # excludes the input prompt if present
            next_token_logits = new_logits.clone()
            if len(logits_processor) > 0:
                for i in range(candidate_length + 1):
                    new_logits[:, i, :] = logits_processor(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])
            if len(logits_warper) > 0:
                for i in range(candidate_length + 1):
                    new_logits[:, i, :] = logits_warper(candidate_input_ids[:, : cur_len + i], new_logits[:, i, :])

            # 3. Select the accepted tokens. There are two possible cases:
            # Case 1: `do_sample=True` and we have logits for the candidates (originally from speculative decoding)
            # 👉 Apply algorithm 1 from the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf).
            if do_sample and candidate_logits is not None:
                valid_tokens, n_matches = _speculative_sampling(
                    candidate_input_ids,
                    candidate_logits,
                    candidate_length,
                    new_logits,
                    is_done_candidate,
                )

            # Case 2: all other cases (originally from assisted generation) 👉 Compare the tokens selected from the
            # original model logits with the candidate tokens. We can keep the candidate tokens until the first
            # mismatch, or until the max length is reached.
            else:
                if do_sample:
                    probs = new_logits.softmax(dim=-1)
                    selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
                else:
                    selected_tokens = new_logits.argmax(dim=-1)

                candidate_new_tokens = candidate_input_ids[:, cur_len:]
                n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()

                # Ensure we don't generate beyond max_len or an EOS token
                if is_done_candidate and n_matches == candidate_length:
                    n_matches -= 1
                valid_tokens = selected_tokens[:, : n_matches + 1]

            # 4. Update variables according to the number of matching assistant tokens. Remember: the token generated
            # by the model after the last candidate match is also valid, as it is generated from a correct sequence.
            # Because of this last token, assisted generation search reduces to a normal greedy search/sample if there
            # is no match.

            # 4.1. Get the valid continuation, after the matching tokens
            input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
            if streamer is not None:
                streamer.put(valid_tokens.cpu())
            new_cur_len = input_ids.shape[-1]

            # 4.2. Discard past key values relative to unused assistant tokens
            new_cache_size = new_cur_len - 1
            outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)

            # 5. Update the candidate generation strategy if needed
            candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Store scores, attentions and hidden_states when required
            # Assistant: modified to append one tuple element per token, as in the other generation methods.
            if return_dict_in_generate:
                if output_scores:
                    scores += tuple(new_logits[:, i, :] for i in range(n_matches + 1))
                if output_logits:
                    raw_logits += (next_token_logits,)

                if "past_key_values" not in model_kwargs:
                    added_len = new_cur_len
                else:
                    added_len = n_matches + 1

                if output_attentions:
                    if self.config.is_encoder_decoder:
                        cross_attentions = _split_model_outputs(
                            cross_attentions, outputs.cross_attentions, cur_len, added_len
                        )
                        decoder_attentions = _split_model_outputs(
                            decoder_attentions,
                            outputs.decoder_attentions,
                            cur_len,
                            added_len,
                            is_decoder_attention=True,
                        )
                    else:
                        decoder_attentions = _split_model_outputs(
                            decoder_attentions,
                            outputs.attentions,
                            cur_len,
                            added_len,
                            is_decoder_attention=True,
                        )
                if output_hidden_states:
                    if self.config.is_encoder_decoder:
                        decoder_hidden_states = _split_model_outputs(
                            decoder_hidden_states, outputs.decoder_hidden_states, cur_len, added_len
                        )
                    else:
                        decoder_hidden_states = _split_model_outputs(
                            decoder_hidden_states, outputs.hidden_states, cur_len, added_len
                        )

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if (
            hasattr(candidate_generator, "assistant_model")
            and candidate_generator.assistant_model.generation_config.num_assistant_tokens_schedule == "heuristic"
        ):
            candidate_generator.assistant_model.generation_config.num_assistant_tokens = (
                candidate_generator.num_assistant_tokens
            )
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


def _speculative_sampling(
    candidate_input_ids,
    candidate_logits,
    candidate_length,
    new_logits,
    is_done_candidate,
):
    """
    Applies sampling as in the speculative decoding paper (https://arxiv.org/pdf/2211.17192.pdf, algorithm 1). Returns
    the selected tokens, as well as the number of candidate matches.

    NOTE: Unless otherwise stated, the variable names match those in the paper.
    """
    new_candidate_input_ids = candidate_input_ids[:, -candidate_length:]
    # Gets the probabilities from the logits. q_i and p_i denote the assistant and model probabilities of the tokens
    # selected by the assistant, respectively.
    q = candidate_logits.softmax(dim=-1)
    q_i = q[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    p = new_logits.softmax(dim=-1)
    p_i = p[:, torch.arange(candidate_length), new_candidate_input_ids].squeeze(0, 1)
    probability_ratio = p_i / q_i

    # When probability_ratio > 1 (i.e. q_i(x) < p_i(x), or "assistant probability of the candidate token is smaller
    # than the model probability for the same token"), keep the token. Otherwise reject with p = 1 - probability_ratio
    # (= keep with p = probability_ratio). Keep all the tokens until the first rejection
    r_i = torch.rand_like(probability_ratio)
    is_accepted = r_i <= probability_ratio
    n_matches = ((~is_accepted).cumsum(dim=-1) < 1).sum()  # this is `n` in algorithm 1

    # Ensure we don't generate beyond max_len or an EOS token (not in algorithm 1, but needed for correct behavior)
    if is_done_candidate and n_matches == candidate_length:
        # Output length is assumed to be `n_matches + 1`. Since we won't generate another token with the target model
        # due to acceptance on EOS we fix `n_matches`
        n_matches -= 1
        valid_tokens = new_candidate_input_ids[:, : n_matches + 1]
    else:
        # Next token selection: if there is a rejection, adjust the distribution from the main model before sampling.
        gamma = candidate_logits.shape[1]
        p_n_plus_1 = p[:, n_matches, :]
        if n_matches < gamma:
            q_n_plus_1 = q[:, n_matches, :]
            p_prime = torch.clamp((p_n_plus_1 - q_n_plus_1), min=0)
            p_prime.div_(p_prime.sum())
        else:
            p_prime = p_n_plus_1
        t = torch.multinomial(p_prime, num_samples=1).squeeze(1)[None, :]

        # The selected tokens include the matches (if any) plus the next sampled tokens
        if n_matches > 0:
            valid_tokens = torch.cat((new_candidate_input_ids[:, :n_matches], t), dim=-1)
        else:
            valid_tokens = t

    return valid_tokens, n_matches


def _split_model_outputs(outputs, new_outputs, cur_len, added_len, is_decoder_attention=False):
    """
    Given the (decoder/cross attentions)/(decoder hidden states) for multiple generated tokens, splits it into a tuple
    where each member corresponds to a single generated token.
    """
    # Retrocompatibility: in our generation functions, the first iteration includes the attention/hidden states for the
    # prompt.
    if len(outputs) == 0:
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., :cur_len, :last_dim_size],)
        outputs += (new_tuple,)
        # The first iteration contains the prompt + 1 generated token, let's update the length variables accordingly
        cur_len += 1
        added_len -= cur_len

    for i in range(added_len):
        new_tuple = ()
        for layer in new_outputs:
            last_dim_size = cur_len + i if is_decoder_attention else layer.shape[-1]
            new_tuple += (layer[..., i : i + 1, :last_dim_size],)
        outputs += (new_tuple,)
    return outputs


def _ranking_fast(
    context_hidden: torch.FloatTensor,
    next_hidden: torch.FloatTensor,
    next_top_k_probs: torch.FloatTensor,
    alpha: float,
    beam_width: int,
) -> torch.FloatTensor:
    """
    Reranks the top_k candidates based on a degeneration penalty (cosine similarity with previous tokens), as described
    in the paper "A Contrastive Framework for Neural Text Generation". Returns the index of the best candidate for each
    row in the batch.
    """
    norm_context_hidden = context_hidden / context_hidden.norm(dim=2, keepdim=True)
    norm_next_hidden = next_hidden / next_hidden.norm(dim=2, keepdim=True)
    cosine_matrix = torch.matmul(norm_context_hidden, norm_next_hidden.transpose(1, 2)).squeeze(-1)  # [B*K, S]
    degeneration_penalty, _ = torch.max(cosine_matrix, dim=-1)  # [B*K]
    next_top_k_probs = next_top_k_probs.view(-1)  # [B*K]
    contrastive_score = (1.0 - alpha) * next_top_k_probs - alpha * degeneration_penalty
    contrastive_score = torch.stack(torch.split(contrastive_score, beam_width))  # [B, K]
    _, selected_idx = contrastive_score.max(dim=-1)  # [B]
    return selected_idx


def _split(data, full_batch_size: int, split_size: int = None):
    """
    Takes care of three cases:
    1. data is a tensor: e.g. last_hidden_state, pooler_output etc. split them on the batch_size dim
    2. data is a tuple: e.g. hidden_states, attentions etc. Keep the tuple as it is and split each tensor in it and
       return a list of tuples
    3. data is a tuple of tuples, e.g. past_key_values. Keep the tuple as it is and split each tuple in it and
       return a list of tuples of tuples
    (see documentation of ModelOutput)
    """
    if data is None:
        return [None] * (full_batch_size // split_size)
    if isinstance(data, torch.Tensor):
        return [data[i : i + split_size] for i in range(0, full_batch_size, split_size)]
    elif isinstance(data, tuple):
        # If the elements of the tuple are also tuples (e.g., past_key_values in our earlier example)
        if isinstance(data[0], tuple):
            return [
                tuple(tuple(tensor[i : i + split_size] for tensor in inner_tuple) for inner_tuple in data)
                for i in range(0, full_batch_size, split_size)
            ]

        else:
            return [
                tuple(sub_tensor[i : i + split_size] for sub_tensor in data)
                for i in range(0, full_batch_size, split_size)
            ]
    else:
        raise ValueError(f"Unexpected attribute type: {type(data)}")


def _split_model_inputs(
    model_input: Union[ModelOutput, Dict], split_size: int, full_batch_size: int
) -> List[Union[ModelOutput, Dict]]:
    """
    Split a ModelOutput object (or its subclasses) or Dict into a list of same-class objects based on a specified split
    size. The input object is dict when it was prepared for forward pass and ModelOutput when it was returned from
    previous forward pass.
    """
    # Edge case: if model_input is None, return a list of Nones
    # this happens with Whisper where encoder_outputs is None
    if model_input is None:
        return [model_input] * (full_batch_size // split_size)
    # Infer the class from the object
    model_output_cls = type(model_input)
    if (full_batch_size % split_size) != 0:
        raise ValueError("`full_batch_size` must be divisible by `split_size`")

    if split_size > full_batch_size:
        raise ValueError("`split_size` must be smaller or equal to `full_batch_size`")

    # Helper function to split tensors or tuples of tensors

    # Find all the dataclass fields (e.g., last_hidden_state, pooler_output etc.) and split them
    keys = (
        model_input.__dataclass_fields__.keys() if hasattr(model_input, "__dataclass_fields__") else model_input.keys()
    )
    # We only keep keys that are in the model_input
    keys = [k for k in keys if k in model_input]
    # Here we can have four types of values: tensors, tuples of tensors and booleans, and encoder_outputs which is a
    # ModelOutput object.
    # bool should not be split but replicated for each split
    bool_keys = [k for k in keys if isinstance(model_input[k], bool) or k == "cache_position"]
    keys_to_ignore = ["cache_position", "encoder_outputs", "num_logits_to_keep"]
    non_bool_keys = [k for k in keys if not isinstance(model_input[k], bool) and k not in keys_to_ignore]

    # we split the tensors and tuples of tensors
    data_split_list = [
        {k: _split(model_input[k], full_batch_size, split_size)[i] for k in non_bool_keys}
        for i in range(full_batch_size // split_size)
    ]
    # bool values are the same and replicated for each split
    bool_data = {k: model_input[k] for k in bool_keys}
    # encoder_outputs is a ModelOutput object and should be split by its own
    if "encoder_outputs" in model_input:
        encoder_outputs_split = _split_model_inputs(model_input["encoder_outputs"], split_size, full_batch_size)
        data_split_list = [
            {**data_split, "encoder_outputs": encoder_outputs_split[i]} for i, data_split in enumerate(data_split_list)
        ]
    # num_logits_to_keep should be replicated for each split, similar to bool values
    if "num_logits_to_keep" in model_input:
        data_split_list = [
            {**data_split, "num_logits_to_keep": model_input["num_logits_to_keep"]} for data_split in data_split_list
        ]

    # Convert each dictionary in the list to an object of the inferred class
    split_model_inputs: List[Union[ModelOutput, Dict]] = [
        model_output_cls(**data_split, **bool_data) for data_split in data_split_list
    ]

    return split_model_inputs


def stack_model_outputs(model_outputs: List[ModelOutput]) -> ModelOutput:
    """
    Stack a list of ModelOutput objects (or its subclasses) along the batch_size dimension. The function infers the
    specific ModelOutput subclass from the list provided.
    """
    if not model_outputs:
        raise ValueError("Input list is empty.")

    # Infer the class from the first object in the list
    model_output_cls = type(model_outputs[0])

    # Ensure all objects are of the same type
    if not all(isinstance(obj, model_output_cls) for obj in model_outputs):
        raise ValueError("All elements in the list should be of the same type.")

    # Helper function to concat tensors or tuples of tensors
    def _concat(data):
        """
        Reverse of `_split` function above.
        """
        if any(data is None for data in data):
            return None
        if isinstance(data[0], torch.Tensor):
            return torch.cat(data, dim=0)
        elif isinstance(data[0], tuple):
            # If the elements of the tuple are also tuples (e.g., past_key_values in our earlier example)
            if isinstance(data[0][0], tuple):
                return tuple(
                    tuple(torch.cat([attr[i][j] for attr in data], dim=0) for j in range(len(data[0][0])))
                    for i in range(len(data[0]))
                )
            else:
                return tuple(torch.cat([attr[i] for attr in data], dim=0) for i in range(len(data[0])))
        elif isinstance(data[0], (int, float)):
            # If the elements are integers or floats, return a tensor
            return torch.tensor(data)
        else:
            raise ValueError(f"Unexpected attribute type: {type(data[0])}")

    # Use a dictionary comprehension to gather attributes from all objects and concatenate them
    concatenated_data = {
        k: _concat([getattr(model_output, k) for model_output in model_outputs])
        for k in model_output_cls.__dataclass_fields__.keys()
    }

    # Return a new object of the inferred class with the concatenated attributes
    return model_output_cls(**concatenated_data)
