# coding=utf-8
"""PyTorch UniLM model. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import logging
import numpy as np
import torch
from torch import nn
import random
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from transformers.modeling_utils import PreTrainedModel
from .configuration_unilm import UnilmConfig
from transformers.models.bert.modeling_bert import load_tf_weights_in_bert, BertPooler, BertIntermediate, BertOutput, BertPredictionHeadTransform, BertSelfOutput, BertLMPredictionHead, BertOnlyMLMHead, BertOnlyMLMHead, BertEmbeddings, BertOnlyNSPHead

logger = logging.getLogger(__name__)

UNILM_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'unilm-base-cased': "",
    'unilm-large-cased': ""
}

BertLayerNorm = torch.nn.LayerNorm


class UnilmOutput:
    def __init__(self, loss):
        self.loss = loss


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        x = x.view(*sz)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, history_states=None):
        if history_states is None:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        else:
            x_states = torch.cat((history_states, hidden_states), dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, history_states=None):
        self_output = self.self(
            input_tensor, attention_mask, history_states=history_states)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, history_states=None):
        attention_output = self.attention(
            hidden_states, attention_mask, history_states=history_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, prev_embedding=None, prev_encoded_layers=None):
        assert (prev_embedding is None) == (prev_encoded_layers is None)

        all_encoder_layers = []
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(
                    hidden_states, attention_mask, history_states=history_states)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for layer_module in self.layer:
                hidden_states = layer_module(
                    hidden_states, attention_mask)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class UnilmPreTrainedModel(PreTrainedModel):
    config_class = UnilmConfig
    pretrained_model_archive_map = UNILM_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "unilm"

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class UnilmModel(UnilmPreTrainedModel):
    def __init__(self, config):
        super(UnilmModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):

        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)
        embedding_output = self.embeddings(
            input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

    def generate_forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None, output_all_encoded_layers=True, prev_embedding=None, prev_encoded_layers=None):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        embedding_output = self.embeddings(
            input_ids, token_type_ids, position_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return embedding_output, encoded_layers, pooled_output


class LabelSmoothingLoss(_Loss):
    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0, size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        assert self.tgt_vocab_size == output.size(2)
        batch_size, num_pos = target.size(0), target.size(1)
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob.type_as(output), reduction='none').view(batch_size, num_pos, -1).sum(2)


class UnilmForSeq2Seq(UnilmPreTrainedModel):
    def __init__(self, config,
                 search_beam_size=5, length_penalty=0.,
                 forbid_duplicate_ngrams=False, forbid_ignore_set=None, ngram_size=3, min_len=0):
        super(UnilmForSeq2Seq, self).__init__(config)
        self.bert = UnilmModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.search_beam_size = None
        self.length_penalty = length_penalty
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set = forbid_ignore_set
        self.ngram_size = ngram_size
        self.min_len = min_len
        self.init_weights()
        self.tie_weights()
        self.mask_word_id = None
        self.eos_id = None
        self.sos_id = None
        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None

    def grouped_param(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters

    def additional_init(self, mask_word_id=0, eos_id=0, sos_id=0):
        self.mask_word_id = mask_word_id
        self.eos_id = eos_id
        self.sos_id = sos_id

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, masked_pos=None, masked_weights=None):
        sequence_output, __ = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_all_encoded_layers=False)

        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

        def gather_seq_out_by_pos_average(seq, pos, mask):
            batch_size, max_token_num = pos.size(0), pos.size(-1)
            pos_vec = torch.gather(seq, 1, pos.view(batch_size, -1).unsqueeze(
                2).expand(-1, -1, seq.size(-1))).view(batch_size, -1, max_token_num, seq.size(-1))
            mask = mask.type_as(pos_vec)
            pos_vec_masked_sum = (
                pos_vec * mask.unsqueeze(3).expand_as(pos_vec)).sum(2)
            return pos_vec_masked_sum / mask.sum(2, keepdim=True).expand_as(pos_vec_masked_sum)

        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask) + 1e-5
            return (loss / denominator).sum()

        if masked_lm_labels is None:
            if masked_pos is None:
                prediction_scores = self.cls(sequence_output)
            else:
                sequence_output_masked = gather_seq_out_by_pos(
                    sequence_output, masked_pos)
                prediction_scores = self.cls(sequence_output_masked)
            return UnilmOutput(loss=prediction_scores)

        sequence_output_masked = gather_seq_out_by_pos(
            sequence_output, masked_pos)
        prediction_scores_masked = self.cls(sequence_output_masked)
        if self.crit_mask_lm_smoothed:
            masked_lm_loss = self.crit_mask_lm_smoothed(
                F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
        else:
            masked_lm_loss = self.crit_mask_lm(
                prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
        masked_lm_loss = loss_mask_and_normalize(
            masked_lm_loss.float(), masked_weights)

        return UnilmOutput(loss=masked_lm_loss)

    def generate(self, **kwargs):
        self.search_beam_size = kwargs['num_beams']
        return self.beam_search(input_ids=kwargs['input_ids'],
                                attention_mask=kwargs['attention_mask'],
                                position_ids=kwargs['position_ids'],
                                token_type_ids=kwargs['token_type_ids'])

    def beam_search(self, input_ids, token_type_ids, position_ids, attention_mask):
        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        next_pos = input_length
        K = self.search_beam_size
        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []
        forbid_word_mask = None
        buf_matrix = None

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]

            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)

            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:,
                                                 start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]
            new_embedding, new_encoded_layers, _ = \
                self.bert.generate_forward(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                          output_all_encoded_layers=True, prev_embedding=prev_embedding, prev_encoded_layers=prev_encoded_layers)

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores = self.cls(last_hidden)
            log_scores = torch.nn.functional.log_softmax(
                prediction_scores, dim=-1)
            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0)
            if self.min_len and (next_pos-input_length+1 <= self.min_len):
                log_scores[:, :, self.eos_id].fill_(-10000.0)
            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            if len(total_scores) == 0:
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K)
                back_ptrs = torch.div(k_ids, K, rounding_mode="trunc")
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids)
            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).float())
            total_scores.append(k_scores)

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x

            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])
                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y

            is_first = (prev_embedding is None)

            if prev_embedding is None:
                prev_embedding = first_expand(new_embedding[:, :-1, :])
            else:
                prev_embedding = torch.cat(
                    (prev_embedding, new_embedding[:, :-1, :]), dim=1)
                prev_embedding = select_beam_items(
                    prev_embedding, back_ptrs)
            if prev_encoded_layers is None:
                prev_encoded_layers = [first_expand(
                    x[:, :-1, :]) for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                       for x in zip(prev_encoded_layers, new_encoded_layers)]
                prev_encoded_layers = [select_beam_items(
                    x, back_ptrs) for x in prev_encoded_layers]

            curr_ids = torch.reshape(k_ids, [batch_size * K, 1])

            if is_first:
                token_type_ids = first_expand(token_type_ids)
                position_ids = first_expand(position_ids)
                attention_mask = first_expand(attention_mask)
                mask_ids = first_expand(mask_ids)

            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(
                                partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n-1):]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not(self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(
                            get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros(
                                (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).cuda()
                    else:
                        forbid_word_mask = None
            next_pos += 1
        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
        for b in range(batch_size):
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            ptrs = [x[b] for x in step_back_ptrs]
            traces['scores'].append(scores)
            traces['wids'].append(wids_list)
            traces['ptrs'].append(ptrs)
            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1
            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.eos_id or fid == last_frame_id:
                        s = scores[fid][i]
                        if self.length_penalty > 0:
                            s /= math.pow((5 + fid + 1) / 6.0,
                                          self.length_penalty)
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1:
                traces['pred_seq'].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces['pred_seq'].append(seq)

        def _pad_sequence(sequences, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims
            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                out_tensor[i, :length, ...] = tensor
            return out_tensor

        output_length -= input_length
        ts_list = traces['pred_seq'].tolist() if isinstance(traces['pred_seq'][0], torch.Tensor) else traces['pred_seq']
        out_ids_list = []
        for ts in ts_list:
            out_ids = []
            for id in ts:
                if id in (self.eos_id, 0):
                    break
                out_ids.append(id)
            out_ids_list.append(out_ids)
        ts_list = [torch.tensor(it, dtype=torch.long) for it in out_ids_list]
        ts_list = _pad_sequence(
            ts_list, output_length, padding_value=0).to(input_ids.device)

        return ts_list
