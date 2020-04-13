# coding=utf-8
# Copyright 2020 Marian Team Authors and The HuggingFace Inc. team.
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
"""PyTorch model, ported from the Marian repo."""
import copy
import logging
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .activations import ACT2FN
from .configuration_bart import BartConfig
from .configuration_marian import MarianConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_bart import _make_linear_from_emb, _prepare_bart_decoder_inputs
from .modeling_bert import *
from .modeling_utils import PreTrainedModel, create_position_ids_from_input_ids


logger = logging.getLogger(__name__)


PRETRAINED_MODEL_ARCHIVE_MAP = {}

INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
               Indices of input sequence tokens in the vocabulary. Use BartTokenizer.encode to produce them.
            Padding will be ignored by default should you provide it.
            Indices can be obtained using :class:`transformers.BartTokenizer.encode(text)`.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices in input_ids.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
            Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`)
            `last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`) is a sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention of the decoder.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
            target language input ids
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`, defaults to :obj:`None`):
            a tensor that ignores pad tokens in decoder_input_ids.
            If you want to change padding behavior, you should read :func:`~transformers.modeling_bart._prepare_decoder_inputs` and modify.
            See diagram 1 in the paper for more info on the default strategy
"""



def invert_mask(attention_mask):
    assert attention_mask.dim() == 2
    return attention_mask.eq(0)

def append_dummy_token(input_ids, attention_mask, token_id):
    effective_batch_size = input_ids.shape[0]
    dummy_token = torch.full(
        (effective_batch_size, 1), token_id, dtype=torch.long, device=input_ids.device
    )
    input_ids = torch.cat([input_ids, dummy_token], dim=1)
    attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
    return input_ids, attention_mask

class MarianModel(PreTrainedModel):
    config_class = MarianConfig
    pretrained_model_archive_map = PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "zczcx"  # HACK to avoid start_prefix = '.' in from_pretrained

    def _init_weights(self, module):
        pass
        # self.encoder.init_weights()
        # self.decoder.init_weights()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs

    def __init__(self, config: MarianConfig):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        enc_config = copy.deepcopy(config)
        enc_config.is_decoder = False
        self.encoder = BertModel(enc_config)
        self.decoder = BertForMaskedLM(config)
        self.init_weights()

    @add_start_docstrings_to_callable(INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[Tuple[Tensor]] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        decoder_cached_states: Optional[Tensor] = None,
        lm_labels: Optional[Tensor] = None,
        use_cache: Optional[Tensor] = True,
        **unused
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels
            in ``[0, ..., config.vocab_size]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(encoder_outputs, tuple):
            encoder_outputs = encoder_outputs[0]

        assert isinstance(encoder_outputs, torch.Tensor)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder.forward(
            input_ids=decoder_input_ids,
            attention_mask=decoder_padding_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            # decoder_causal_mask=causal_mask,
            # decoder_cached_states=decoder_cached_states,
            # use_cache=use_cache,
            lm_labels=lm_labels,
        )
        # Attention and hidden_states will be [] or None if they aren't needed
        if use_cache:
            cache = (encoder_outputs,)
            return (decoder_outputs[0], cache, decoder_outputs[1:])
        else:
            return decoder_outputs + encoder_outputs
        return outputs

    def prepare_inputs_for_generation(self, input_ids: Tensor, past: tuple, attention_mask: Tensor, **kwargs) -> Dict:
        assert past is not None, "past has to be defined for encoder_outputs"
        assert isinstance(past, tuple)

        # first step
        if len(past) < 2:
            encoder_outputs, decoder_past_key_value_states = past, None
        else:
            encoder_outputs, decoder_past_key_value_states = past[0], past[1]



        return {
            "decoder_input_ids": input_ids,
            "decoder_cached_states": decoder_past_key_value_states,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
        }

    def _reorder_cache(self, past: Tuple, beam_idx) -> Tuple:
        (encoder_outputs,) = past
        reordered_encoder_outputs = encoder_outputs.index_select(0, beam_idx)
        return (reordered_encoder_outputs,)

    def _do_output_past(self, *args, **kwargs) -> bool:
        return True

        # return self.decoder.prepare_inputs_for_generation(*args, **kwargs)

    def get_encoder(self):
        return self.encoder

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def _do_output_past(self, *args, **kwargs):
        """ We should always use the cache in generate."""
        return True

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)
