# coding=utf-8
# Copyright 2018 Hao Tan, Mohit Bansal
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
""" PyTorch LXMERT model. """


import logging
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss

from .activations import gelu, swish
from .configuration_lxmert import LxmertConfig
from .file_utils import ModelOutput, add_start_docstrings
from .modeling_outputs import LxmertModelOutput
from .modeling_utils import PreTrainedModel


logger = logging.getLogger(__name__)

LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST = {
    "lxmert-base-uncased": "",
}


class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


@dataclass
class LxmertForPretrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.LxmertForPretrainingModel`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cross_relationship_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the textual matching objective (classification) head (scores of True/False
            continuation before SoftMax).
        question_answering_score: (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, n_qa_answers)`):
            Prediction scores of question answering objective (classification).
        attentions_l_encoder (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        attentions_v_encoder (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        attentions_x_encoder (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: [torch.FloatTensor] = None
    prediction_logits: Optional[torch.FloatTensor] = None
    cross_relationship_score: Optional[torch.FloatTensor] = None
    question_answering_score: Optional[torch.FloatTensor] = None
    attentions_v_encoder: Optional[Tuple[torch.FloatTensor]] = None
    attentions_l_encoder: Optional[Tuple[torch.FloatTensor]] = None
    attentions_x_encoder: Optional[Tuple[torch.FloatTensor]] = None


def load_tf_weights_in_lxmert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

LxmertLayerNorm = torch.nn.LayerNorm


class LxmertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LxmertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class LxmertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class LxmertAttOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LxmertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LxmertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = LxmertAttention(config)
        self.output = LxmertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None, output_attentions=False):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask, output_attentions=output_attentions)
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], input_tensor)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs


class LxmertSelfattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LxmertAttention(config)
        self.output = LxmertAttOutput(config)

    def forward(self, input_tensor, attention_mask, output_attentions=False):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        output = self.self(input_tensor, input_tensor, attention_mask, output_attentions=output_attentions)
        if output_attentions:
            attention_probs = output[1]
        attention_output = self.output(output[0], input_tensor)
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)
        return outputs


class LxmertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class LxmertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LxmertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LxmertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = LxmertAttention(config)
        self.intermediate = LxmertIntermediate(config)
        self.output = LxmertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        outputs = self.attention(hidden_states, hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs[1:]  # add attentions if we output them
        return outputs


class LxmertXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = LxmertCrossattLayer(config)

        # Self-attention Layers
        self.lang_self_att = LxmertSelfattLayer(config)
        self.visn_self_att = LxmertSelfattLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = LxmertIntermediate(config)
        self.lang_output = LxmertOutput(config)
        self.visn_inter = LxmertIntermediate(config)
        self.visn_output = LxmertOutput(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask, output_x_attentions=False):
        # Cross Attention
        lang_att_output = self.visual_attention(
            lang_input, visn_input, ctx_att_mask=visn_attention_mask, output_attentions=output_x_attentions
        )
        visn_att_output = self.visual_attention(
            visn_input, lang_input, ctx_att_mask=lang_attention_mask, output_attentions=False
        )
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, output_attentions=False)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask, output_attentions=False)
        return lang_att_output[0], visn_att_output[0]

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)

        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask, output_attentions=False):

        lang_att_output = (lang_feats,)
        visn_att_output = (visn_feats,)

        lang_att_output, visn_att_output = self.cross_att(
            lang_att_output[0], lang_attention_mask, visn_att_output[0], visn_attention_mask,
        )
        attention_probs = lang_att_output[1:]
        lang_att_output, visn_att_output = self.self_att(
            lang_att_output[0], lang_attention_mask, visn_att_output[0], visn_attention_mask
        )

        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)
        return (lang_output, visn_output) + (attention_probs,) if output_attentions else (lang_output, visn_output)


class VisualFeatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = config.visual_feat_dim
        pos_dim = config.visual_pos_dim

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = LxmertLayerNorm(config.hidden_size, eps=1e-12)

        # Box position encoding
        self.box_fc = nn.Linear(pos_dim, config.hidden_size)
        self.box_layer_norm = LxmertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        feats, boxes = visn_input
        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)
        y = self.box_fc(boxes)
        y = self.box_layer_norm(y)
        output = (x + y) / 2

        output = self.dropout(output)
        return output


class LxmertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Obj-level image embedding layer
        self.visn_fc = VisualFeatEncoder(config)

        # Number of layers
        self.num_l_layers = config.l_layers
        self.num_x_layers = config.x_layers
        self.num_r_layers = config.r_layers

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList([LxmertLayer(config) for _ in range(self.num_l_layers)])
        self.x_layers = nn.ModuleList([LxmertXLayer(config) for _ in range(self.num_x_layers)])
        self.r_layers = nn.ModuleList([LxmertLayer(config) for _ in range(self.num_r_layers)])

    def forward(
        self,
        lang_feats,
        lang_attention_mask,
        visn_feats,
        visn_attention_mask=None,
        output_v_attentions=None,
        output_l_attentions=None,
        output_x_attentions=None,
    ):

        all_v_hidden_states = ()
        all_l_hidden_states = ()
        all_v_attentions = () if output_v_attentions else None
        all_l_attentions = () if output_l_attentions else None
        all_x_attentions = () if output_x_attentions else None

        visn_feats = self.visn_fc(visn_feats)

        # Run language layers
        for layer_module in self.layer:
            l_outputs = layer_module(lang_feats, visn_attention_mask, output_attentions=output_l_attentions)
            lang_feats = l_outputs[0]
            if all_l_attentions:
                all_l_attentions = all_l_attentions + (l_outputs[1],)

        # Run relational layers
        for layer_module in self.r_layers:
            v_outputs = layer_module(visn_feats, visn_attention_mask, output_attentions=output_v_attentions)
            visn_feats = v_outputs[0]
            if all_v_attentions:
                all_v_attentions = all_v_attentions + (v_outputs[1],)

        # Run cross-modality layers
        all_v_hidden_states = all_v_hidden_states + (visn_feats,)
        all_l_hidden_states = all_l_hidden_states + (lang_feats,)
        for layer_module in self.x_layers:
            x_outputs = layer_module(
                lang_feats, lang_attention_mask, visn_feats, visn_attention_mask, output_attentions=output_x_attentions
            )
            lang_feats, visn_feats = x_outputs[:2]
            all_v_hidden_states = all_v_hidden_states + (visn_feats,)
            all_l_hidden_states = all_l_hidden_states + (lang_feats,)
            if all_x_attentions:
                all_x_attentions = all_x_attentions + (x_outputs[2],)

        visn_encoder_outputs = (all_v_hidden_states, all_v_attentions if output_v_attentions else None)
        lang_encoder_outputs = (all_l_hidden_states, all_l_attentions if output_l_attentions else None)
        return (visn_encoder_outputs, lang_encoder_outputs, all_x_attentions if output_x_attentions else None)


class LxmertPooler(nn.Module):
    def __init__(self, config):
        super(LxmertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LxmertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(LxmertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = LxmertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LxmertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(LxmertLMPredictionHead, self).__init__()
        self.transform = LxmertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1), bert_model_embedding_weights.size(0), bias=False
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class LxmertVisualAnswerHead(nn.Module):
    def __init__(self, config, num_answers):
        super().__init__()
        hid_dim = config.hidden_size
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            LxmertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class LxmertVisualObjHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = LxmertPredictionHeadTransform(config)
        # Decide the use of visual losses
        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {"shape": (-1,), "num": config.n_object_labels}
        if config.visual_attr_loss:
            visual_losses["attr"] = {"shape": (-1,), "num": config.n_attr_labels}
        if config.visual_obj_loss:
            visual_losses["feat"] = {"shape": (-1, config.visual_feat_dim), "num": config.visual_feat_dim}
        self.visual_losses = visual_losses

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict(
            {key: nn.Linear(config.hidden_size, self.visual_losses[key]["num"]) for key in self.visual_losses}
        )

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


class LxmertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(LxmertPreTrainingHeads, self).__init__()
        self.predictions = LxmertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class LxmertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = LxmertConfig
    load_tf_weights = load_tf_weights_in_lxmert
    base_model_prefix = "lxmert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LxmertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


LXMERT_START_DOCSTRING = r"""    The LXMERT model was proposed in
    `LXMERT: Learning Cross-Modality Encoder Representations from Transformers
    by Hao Tan and Mohit Bansal. It's a vision and language transformer model,
    pre-trained on a variety of multi-modal datasets comprising of GQA, VQAv2.0, MCSCOCO captions, and Visual genome,
    using a combination of masked language modeling, region of interest feature regression,
    cross entropy loss for question answering attribute prediction, and object tag predicition.


    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`LXMERT: Learning Cross-Modality Encoder Representations from Transformers`
        https://arxiv.org/pdf/1908.07490.pdf

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.LxmertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

LXMERT_INPUTS_DOCSTRING = r"""

     Args:
         input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
             Indices of input sequence tokens in the vocabulary.

             Indices can be obtained using :class:`transformers.LxmertTokenizer`.
             See :func:`transformers.PreTrainedTokenizer.encode` and
             :func:`transformers.PreTrainedTokenizer.__call__` for details.

             `What are input IDs? <../glossary.html#input-ids>`__
         attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
             Mask to avoid performing attention on padding token indices.
             Mask values selected in ``[0, 1]``:
             ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

             `What are attention masks? <../glossary.html#attention-mask>`__
         token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
             Segment token indices to indicate first and second portions of the inputs.
             Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
             corresponds to a `sentence B` token

             `What are token type IDs? <../glossary.html#token-type-ids>`_


        visual_feats: (:obj: `Tuple[torch.FloatTensor, torch.FloatTensor]`):
            the first item in the tuple represents the actual visual features
            (ROI pooled object features from bounding boxes using a faster-RCNN model) of shape ՝(batch_size, num_visual_features, visual_feat_dim)՝
            and where the second item represents the normalized bounding boxes on a scale of 0~1  of shape ՝(batch_size, 4)՝.
            These are currently not provided by the transformers library
        visual_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
             Mask to avoid performing attention on padding token indices.
             Mask values selected in ``[0, 1]``:
             ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        output_v_attentions: (:obj:`bool`, `optional`, defaults to :obj:`False`):
             If set to ``True``, the attentions tensors of all attention layers for the visual encoder are returned. See ``attentions`` under returned tensors for more detail.
        output_l_attentions: (:obj:`bool`, `optional`, defaults to :obj:`False`):
             If set to ``True``, the attentions tensors of all attention layers for the language encoder are returned. See ``attentions`` under returned tensors for more detail.
        output_x_attentions: (:obj:`bool`, `optional`, defaults to :obj:`False`):
             If set to ``True``, the attentions tensors of all attention layers for the cross modality encoder are returned. See ``attentions`` under returned tensors for more detail.
         output_v_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`False`):
             If set to ``True``, the visual hidden states of all layers from the cross modality encoder are returned where the visual features are used as the input and the language features are used as the context. See ``hidden_states`` under returned tensors for more detail.
         output_l_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`False`):
             If set to ``True``, the language hidden states of all layers from the cross modality encoder are returned where the visual features are used as the context. See ``hidden_states`` under returned tensors for more detail.
        return_dict (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the model will return a :class:`~transformers.file_utils.LxmertModelOutput` instead of a
            plain tuple.
"""


@add_start_docstrings(
    "The bare Lxmert Model transformer outputting raw hidden-states without any specific head on top.",
    LXMERT_START_DOCSTRING,
    LXMERT_INPUTS_DOCSTRING,
)
class LxmertModel(LxmertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = LxmertEmbeddings(config)
        self.encoder = LxmertEncoder(config)
        self.pooler = LxmertPooler(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, *args, **kwargs):
        pass

    def forward(
        self,
        input_ids,
        visual_feats,
        token_type_ids=None,
        attention_mask=None,
        visual_attention_mask=None,
        output_v_attentions=False,
        output_l_attentions=False,
        output_x_attentions=False,
        return_dict=False,
    ):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Process the visual attention mask
        if visual_attention_mask is not None:
            extended_visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=next(self.parameters()).dtype)
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0
        else:
            extended_visual_attention_mask = None

        # Positional Word Embeddings
        embedding_output = self.embeddings(input_ids, token_type_ids)

        # Run Lxmert encoder
        x_encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            visn_feats=visual_feats,
            visn_attention_mask=extended_visual_attention_mask,
            output_v_attentions=output_v_attentions,
            output_l_attentions=output_l_attentions,
            output_x_attentions=output_x_attentions,
        )

        v_outputs, l_outputs = x_encoder_outputs[0], x_encoder_outputs[1]
        visn_output = v_outputs[0][-1]
        lang_output = l_outputs[0][-1]
        pooled_output = self.pooler(lang_output)

        if not return_dict:
            return (
                (lang_output, visn_output, pooled_output),
                (v_outputs[0][:-1], l_outputs[0][:-1]),
                (
                    v_outputs[1:] if output_v_attentions else None,
                    l_outputs[1:] if output_l_attentions else None,
                    x_encoder_outputs[2:] if output_x_attentions else None,
                ),
            )

        return LxmertModelOutput(
            last_hidden_state_l=lang_output,
            hidden_states_l=l_outputs[0][:-1],
            attentions_l_encoder=output_l_attentions,
            last_hidden_state_v=visn_output,
            hidden_states_v=v_outputs[0][:-1],
            attentions_v_encoder=output_v_attentions,
            pooled_output_x_encoder=pooled_output,
            attentions_x_encoder=output_x_attentions,
        )


@add_start_docstrings(
    """Lxmert Model with a specified pre-training heads on top. """, LXMERT_START_DOCSTRING, LXMERT_INPUTS_DOCSTRING
)
class LxmertForPretraining(LxmertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_answers = config.n_qa_labels
        self.visual_loss_normalizer = config.visual_loss_normalizer

        # Use of pre-training tasks
        self.task_mask_lm = config.task_mask_lm
        self.task_obj_predict = config.task_obj_predict
        self.task_matched = config.task_matched
        self.task_qa = config.task_qa

        # Lxmert backbone
        self.bert = LxmertModel(config)

        # Pre-training heads
        self.cls = LxmertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        if self.task_obj_predict:
            self.obj_predict_head = LxmertVisualObjHead(config)
        if self.task_qa:
            self.answer_head = LxmertVisualAnswerHead(config, self.num_answers)

        # Weight initialization
        self.init_weights()

        # Loss functions
        self.loss_fcts = {
            "l2": SmoothL1Loss(reduction="none"),
            "visn_ce": CrossEntropyLoss(ignore_index=-1, reduction="none"),
            "ce": CrossEntropyLoss(ignore_index=-1),
        }

        visual_losses = {}
        if config.visual_obj_loss:
            visual_losses["obj"] = {"shape": (-1,), "num": config.n_object_labels, "loss": "visn_ce"}
        if config.visual_attr_loss:
            visual_losses["attr"] = {"shape": (-1,), "num": config.n_attr_labels, "loss": "visn_ce"}
        if config.visual_obj_loss:
            visual_losses["feat"] = {
                "shape": (-1, config.visual_feat_dim),
                "num": config.visual_feat_dim,
                "loss": "l2",
            }
        self.visual_losses = visual_losses

    def forward(
        self,
        input_ids,
        visual_feats,
        token_type_ids=None,
        attention_mask=None,
        masked_lm_labels=None,
        obj_labels=None,
        matched_label=None,
        ans=None,
        output_v_attentions=False,
        output_l_attentions=False,
        output_x_attentions=False,
        return_dict=False,
        **kwargs
    ):

        (visual_feats, pos) = visual_feats

        r"""
        masked_lm_labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        matched_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`, defaults to :obj:`None`):
            Labels for computing the whether or not the text input matches the image (classification) loss. Input should be a sequence pair (see :obj:`input_ids` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates that the sentence does not match the image
            ``1`` indicates that the sentence does match the image
        input_ids (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``):
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, LXMERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:
            Indices can be obtained using :class:`transformers.LxmertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        visual_feats: (:obj: `Tuple[torch.FloatTensor, torch.FloatTensor]`):
            the first item in the tuple represents the actual visual features
            (ROI pooled object features from bounding boxes using a faster-RCNN model) of shape ՝(batch_size, num_visual_features, visual_feat_dim)՝
            and where the second item represents the normalized bounding boxes on a scale of 0~1  of shape ՝(batch_size, 4)՝.
            These are currently not provided by the transformers library
        obj_labels: (``Dict[Str: Tuple[Torch.FloatTensor, Torch.FloatTensor]]``, `optional`, defaults to :obj: `None`):
            each key is named after each one of the visual losses and each element of the tuple is of the shape
            ``(batch_size, num_features)`` and ``(batch_size, num_features, visual_feature_dim)``
            for each the label id and the label score respectively
        ans: (``Torch.Tensor`` of shape ``(batch_size)``, `optional`, defaults to :obj: `None`):
            a one hot representation hof the correct answer `optional`
        kwargs: (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:
        """

        (
            (lang_output, visn_output, pooled_output),
            x_encoder_outputs,
            (attentions_v_encoder, attentions_l_encoder, attentions_x_encoder),
        ) = self.bert(
            input_ids=input_ids,
            visual_feats=(visual_feats, pos),
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_v_attentions=output_v_attentions,
            output_l_attentions=output_l_attentions,
            output_x_attentions=output_x_attentions,
        )

        lang_prediction_scores, cross_relationship_score = self.cls(lang_output, pooled_output)
        if self.task_qa:
            answer_score = self.answer_head(pooled_output)
        else:
            answer_score = pooled_output[0][0]

        total_loss = 0.0
        if masked_lm_labels is not None and self.task_mask_lm:
            masked_lm_loss = self.loss_fcts["ce"](
                lang_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)
            )
            total_loss += masked_lm_loss
        if matched_label is not None and self.task_matched:
            matched_loss = self.loss_fcts["ce"](cross_relationship_score.view(-1, 2), matched_label.view(-1))
            total_loss += matched_loss
        if obj_labels is not None and self.task_obj_predict:
            total_visn_loss = 0.0
            visn_prediction_scores_dict = self.obj_predict_head(visn_output)
            for key, key_info in self.visual_losses.items():
                label, mask_conf = obj_labels[key]
                output_dim = key_info["num"]
                loss_fct_name = key_info["loss"]
                label_shape = key_info["shape"]
                weight = self.visual_loss_normalizer
                visn_loss_fct = self.loss_fcts[loss_fct_name]
                visn_prediction_scores = visn_prediction_scores_dict[key]
                visn_loss = visn_loss_fct(visn_prediction_scores.view(-1, output_dim), label.view(*label_shape))
                if visn_loss.dim() > 1:  # Regression Losses
                    visn_loss = visn_loss.mean(1)
                visn_loss = (visn_loss * mask_conf.view(-1)).mean() * weight
                total_visn_loss += visn_loss
            total_loss += total_visn_loss
        if ans is not None and self.task_qa:
            print(answer_score.shape)
            print(ans.shape)
            answer_loss = self.loss_fcts["ce"](answer_score.view(-1, self.num_answers), ans.view(-1))
            total_loss += answer_loss

        if not return_dict:
            output = (
                (lang_prediction_scores, cross_relationship_score, answer_score.detach()),
                (attentions_v_encoder, attentions_l_encoder, attentions_x_encoder),
            )
            return ((total_loss,) + output) if total_loss is not None else output

        return LxmertForPretrainingOutput(
            loss=total_loss,
            question_answering_score=answer_score.detach(),
            prediction_logits=lang_prediction_scores,
            cross_relationship_score=cross_relationship_score,
            attentions_v_encoder=attentions_v_encoder,
            attentions_l_encoder=attentions_l_encoder,
            attentions_x_encoder=attentions_x_encoder,
        )
