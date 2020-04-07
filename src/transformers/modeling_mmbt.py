# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) HuggingFace Inc. team.
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
"""PyTorch MMBT model. """


import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from .file_utils import add_start_docstrings


logger = logging.getLogger(__name__)


class ModalEmbeddings(nn.Module):
    """Generic Modal Embeddings which takes in an encoder, and a transformer embedding.
    """

    def __init__(self, config, encoder, embeddings):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.proj_embeddings = nn.Linear(config.modal_hidden_size, config.hidden_size)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, input_modal, start_token=None, end_token=None, position_ids=None, token_type_ids=None):
        token_embeddings = self.proj_embeddings(self.encoder(input_modal))
        seq_length = token_embeddings.size(1)

        if start_token is not None:
            start_token_embeds = self.word_embeddings(start_token)
            seq_length += 1
            token_embeddings = torch.cat([start_token_embeds.unsqueeze(1), token_embeddings], dim=1)

        if end_token is not None:
            end_token_embeds = self.word_embeddings(end_token)
            seq_length += 1
            token_embeddings = torch.cat([token_embeddings, end_token_embeds.unsqueeze(1)], dim=1)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_modal.device)
            position_ids = position_ids.unsqueeze(0).expand(input_modal.size(0), seq_length)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (input_modal.size(0), seq_length), dtype=torch.long, device=input_modal.device
            )

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


MMBT_START_DOCSTRING = r"""    MMBT model was proposed in
    `Supervised Multimodal Bitransformers for Classifying Images and Text`_
    by Douwe Kiela, Suvrat Bhooshan, Hamed Firooz, Davide Testuggine.
    It's a supervised multimodal bitransformer model that fuses information from text and other image encoders,
    and obtain state-of-the-art performance on various multimodal classification benchmark tasks.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`Supervised Multimodal Bitransformers for Classifying Images and Text`:
        https://github.com/facebookresearch/mmbt

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.MMBTConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
        transformer (:class: `~nn.Module`): A text transformer that is used by MMBT.
            It should have embeddings, encoder, and pooler attributes.
        encoder (:class: `~nn.Module`): Encoder for the second modality.
            It should take in a batch of modal inputs and return k, n dimension embeddings.
"""

MMBT_INPUTS_DOCSTRING = r"""    Inputs:
        **input_modal**: ``torch.FloatTensor`` of shape ``(batch_size, ***)``:
            The other modality data. It will be the shape that the encoder for that type expects.
            e.g. With an Image Encoder, the shape would be (batch_size, channels, height, width)
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            It does not expect [CLS] token to be added as it's appended to the end of other modality embeddings.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **modal_start_tokens**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Optional start token to be added to Other Modality Embedding. [CLS] Most commonly used for Classification tasks.
        **modal_end_tokens**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Optional end token to be added to Other Modality Embedding. [SEP] Most commonly used.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate different portions of the inputs.
        **modal_token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, modal_sequence_length)``:
            Segment token indices to indicate different portions of the non-text modality.
            The embeddings from these tokens will be summed with the respective token embeddings for the non-text modality.
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
        **modal_position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, modal_sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings for the non-text modality.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        **encoder_hidden_states**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``:
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model
            is configured as a decoder.
        **encoder_attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""


@add_start_docstrings(
    "The bare MMBT Model outputting raw hidden-states without any specific head on top.",
    MMBT_START_DOCSTRING,
    MMBT_INPUTS_DOCSTRING,
)
class MMBTModel(nn.Module):
    r"""
        Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
            **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
                Sequence of hidden-states at the output of the last layer of the model.
            **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
                Last layer hidden-state of the first token of the sequence (classification token)
                further processed by a Linear layer and a Tanh activation function. The Linear
                layer weights are trained from the next sentence prediction (classification)
                objective during Bert pretraining. This output is usually *not* a good summary
                of the semantic content of the input, you're often better with averaging or pooling
                the sequence of hidden-states for the whole input sequence.
            **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
                list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
                of shape ``(batch_size, sequence_length, hidden_size)``:
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            **attentions**: (`optional`, returned when ``config.output_attentions=True``)
                list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

        Examples::

            # For example purposes. Not runnable.
            transformer = BertModel.from_pretrained('bert-base-uncased')
            encoder = ImageEncoder(args)
            mmbt = MMBTModel(config, transformer, encoder)
        """

    def __init__(self, config, transformer, encoder):
        super().__init__()
        self.config = config
        self.transformer = transformer
        self.modal_encoder = ModalEmbeddings(config, encoder, transformer.embeddings)

    def forward(
        self,
        input_modal,
        input_ids=None,
        modal_start_tokens=None,
        modal_end_tokens=None,
        attention_mask=None,
        token_type_ids=None,
        modal_token_type_ids=None,
        position_ids=None,
        modal_position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_txt_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_txt_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        modal_embeddings = self.modal_encoder(
            input_modal,
            start_token=modal_start_tokens,
            end_token=modal_end_tokens,
            position_ids=modal_position_ids,
            token_type_ids=modal_token_type_ids,
        )

        input_modal_shape = modal_embeddings.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.ones(input_txt_shape, dtype=torch.long, device=device)

        txt_embeddings = self.transformer.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        embedding_output = torch.cat([modal_embeddings, txt_embeddings], 1)

        input_shape = embedding_output.size()[:-1]

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        else:
            attention_mask = torch.cat(
                [torch.ones(input_modal_shape, device=device, dtype=torch.long), attention_mask], dim=1
            )

        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        else:
            encoder_attention_mask = torch.cat(
                [torch.ones(input_modal_shape, device=device), encoder_attention_mask], dim=1
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.transformer.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.transformer.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


@add_start_docstrings(
    """MMBT Model with a sequence classification/regression head on top (a linear layer on top of
                      the pooled output)""",
    MMBT_START_DOCSTRING,
    MMBT_INPUTS_DOCSTRING,
)
class MMBTForClassification(nn.Module):
    r"""
            **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
                Labels for computing the sequence classification/regression loss.
                Indices should be in ``[0, ..., config.num_labels - 1]``.
                If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
                If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

        Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
            **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
                Classification (or regression if config.num_labels==1) loss.
            **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
                list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
                of shape ``(batch_size, sequence_length, hidden_size)``:
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            **attentions**: (`optional`, returned when ``config.output_attentions=True``)
                list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
                Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

        Examples::

            # For example purposes. Not runnable.
            transformer = BertModel.from_pretrained('bert-base-uncased')
            encoder = ImageEncoder(args)
            model = MMBTForClassification(config, transformer, encoder)
            outputs = model(input_modal, input_ids, labels=labels)
            loss, logits = outputs[:2]
        """

    def __init__(self, config, transformer, encoder):
        super().__init__()
        self.num_labels = config.num_labels

        self.mmbt = MMBTModel(config, transformer, encoder)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_modal,
        input_ids=None,
        modal_start_tokens=None,
        modal_end_tokens=None,
        attention_mask=None,
        token_type_ids=None,
        modal_token_type_ids=None,
        position_ids=None,
        modal_position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.mmbt(
            input_modal=input_modal,
            input_ids=input_ids,
            modal_start_tokens=modal_start_tokens,
            modal_end_tokens=modal_end_tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            modal_token_type_ids=modal_token_type_ids,
            position_ids=position_ids,
            modal_position_ids=modal_position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
