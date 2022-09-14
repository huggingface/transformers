import os
from collections import defaultdict
import argparse
import json
import string
import random
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, AutoModel, AutoConfig, AutoModelWithLMHead
from scripts.triviaqa_utils import evaluation_utils

import pytorch_lightning as pl
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

from longformer.longformer import Longformer
from longformer.sliding_chunks import pad_to_window_size


class TriviaQADataset(Dataset):
    """
    Largely based on
    https://github.com/allenai/allennlp/blob/master/allennlp/data/dataset_readers/reading_comprehension/triviaqa.py
    and
    https://github.com/huggingface/transformers/blob/master/examples/run_squad.py
    """
    def __init__(self, file_path, tokenizer, max_seq_len, max_doc_len, doc_stride,
                 max_num_answers, ignore_seq_with_no_answers, max_question_len):
        assert os.path.isfile(file_path)
        self.file_path = file_path
        with open(self.file_path, "r", encoding='utf-8') as f:
            print(f'reading file: {self.file_path}')
            self.data_json = json.load(f)['data']
            print(f'done reading file: {self.file_path}')
        self.tokenizer = tokenizer

        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
        self.doc_stride = doc_stride
        self.max_num_answers = max_num_answers
        self.ignore_seq_with_no_answers = ignore_seq_with_no_answers
        self.max_question_len = max_question_len

        # A mapping from qid to an int, which can be synched across gpus using `torch.distributed`
        if 'train' not in self.file_path:  # only for the evaluation set
            self.val_qid_string_to_int_map =  \
                {
                    self._get_qid(entry["paragraphs"][0]['qas'][0]['id']): index
                    for index, entry in enumerate(self.data_json)
                }
        else:
            self.val_qid_string_to_int_map = None

    def _normalize_text(self, text: str) -> str:  # copied from the official triviaqa repo
        return " ".join(
            [
                token
                for token in text.lower().strip(self.STRIPPED_CHARACTERS).split()
                if token not in self.IGNORED_TOKENS
            ]
        )
    IGNORED_TOKENS = {"a", "an", "the"}
    STRIPPED_CHARACTERS = string.punctuation + "".join([u"‘", u"’", u"´", u"`", "_"])

    def __len__(self):
        return len(self.data_json)

    def __getitem__(self, idx):
        entry = self.data_json[idx]
        tensors_list = self.one_example_to_tensors(entry, idx)
        assert len(tensors_list) == 1
        return tensors_list[0]

    def one_example_to_tensors(self, example, idx):
        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False
        tensors_list = []
        for paragraph in example["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                answer_spans = []
                for answer in qa["answers"]:
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    try:
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        token_ids = self.tokenizer.encode(orig_answer_text)
                    except RuntimeError:
                        print(f'Reading example {idx} failed')
                        start_position = 0
                        end_position = 0
                    answer_spans.append({'start': start_position, 'end': end_position,
                                         'text': orig_answer_text, 'token_ids': token_ids})

                # ===== Given an example, convert it into tensors  =============
                query_tokens = self.tokenizer.tokenize(question_text)
                query_tokens = query_tokens[:self.max_question_len]
                tok_to_orig_index = []
                orig_to_tok_index = []
                all_doc_tokens = []
                for (i, token) in enumerate(doc_tokens):
                    orig_to_tok_index.append(len(all_doc_tokens))
                    # hack: the line below should have been `self.tokenizer.tokenize(token')`
                    # but roberta tokenizer uses a different subword if the token is the beginning of the string
                    # or in the middle. So for all tokens other than the first, simulate that it is not the first
                    # token by prepending a period before tokenizing, then dropping the period afterwards
                    sub_tokens = self.tokenizer.tokenize(f'. {token}')[1:] if i > 0 else self.tokenizer.tokenize(token)
                    for sub_token in sub_tokens:
                        tok_to_orig_index.append(i)
                        all_doc_tokens.append(sub_token)

                all_doc_tokens = all_doc_tokens[:self.max_doc_len]

                # The -3 accounts for [CLS], [SEP] and [SEP]
                max_tokens_per_doc_slice = self.max_seq_len - len(query_tokens) - 3
                assert max_tokens_per_doc_slice > 0
                if self.doc_stride < 0:
                    # negative doc_stride indicates no sliding window, but using first slice
                    self.doc_stride = -100 * len(all_doc_tokens)  # large -ve value for the next loop to execute once
                input_ids_list = []
                input_mask_list = []
                segment_ids_list = []
                start_positions_list = []
                end_positions_list = []
                answer_token_ids_list = []
                for slice_start in range(0, len(all_doc_tokens), max_tokens_per_doc_slice - self.doc_stride):
                    slice_end = min(slice_start + max_tokens_per_doc_slice, len(all_doc_tokens))

                    doc_slice_tokens = all_doc_tokens[slice_start:slice_end]
                    tokens = [self.tokenizer.cls_token] + query_tokens + [self.tokenizer.sep_token] \
                                                        + doc_slice_tokens + [self.tokenizer.sep_token]
                    segment_ids = [0] * (len(query_tokens) + 2) + [1] * (len(doc_slice_tokens) + 1)
                    assert len(segment_ids) == len(tokens)

                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    input_mask = [1] * len(input_ids)

                    if self.doc_stride >= 0:  # no need to pad if document is not strided
                        # Zero-pad up to the sequence length.
                        padding_len = self.max_seq_len - len(input_ids)
                        input_ids.extend([self.tokenizer.pad_token_id] * padding_len)
                        input_mask.extend([0] * padding_len)
                        segment_ids.extend([0] * padding_len)

                        assert len(input_ids) == self.max_seq_len
                        assert len(input_mask) == self.max_seq_len
                        assert len(segment_ids) == self.max_seq_len

                    doc_offset = len(query_tokens) + 2 - slice_start
                    start_positions = []
                    end_positions = []
                    answer_token_ids = []
                    for answer_span in answer_spans:
                        start_position = answer_span['start']
                        end_position = answer_span['end']
                        tok_start_position_in_doc = orig_to_tok_index[start_position]
                        not_end_of_doc = int(end_position + 1 < len(orig_to_tok_index))
                        tok_end_position_in_doc = orig_to_tok_index[end_position + not_end_of_doc] - not_end_of_doc
                        if tok_start_position_in_doc < slice_start or tok_end_position_in_doc > slice_end:
                            # this answer is outside the current slice
                            continue
                        start_positions.append(tok_start_position_in_doc + doc_offset)
                        end_positions.append(tok_end_position_in_doc + doc_offset)
                        answer_token_ids.append(answer_span['token_ids'])
                    assert len(start_positions) == len(end_positions)
                    if self.ignore_seq_with_no_answers and len(start_positions) == 0:
                        continue

                    # answers from start_positions and end_positions if > self.max_num_answers
                    start_positions = start_positions[:self.max_num_answers]
                    end_positions = end_positions[:self.max_num_answers]
                    answer_token_ids = answer_token_ids[:self.max_num_answers]

                    # -1 padding up to self.max_num_answers
                    padding_len = self.max_num_answers - len(start_positions)
                    start_positions.extend([-1] * padding_len)
                    end_positions.extend([-1] * padding_len)
                    answer_token_ids.extend([[]] * padding_len)

                    # replace duplicate start/end positions with `-1` because duplicates can result into -ve loss values
                    found_start_positions = set()
                    found_end_positions = set()
                    found_answer_token_ids = set()
                    for i, (start_position, end_position, answer_tokens) in enumerate(
                            zip(start_positions, end_positions, answer_token_ids)
                            ):
                        if start_position in found_start_positions:
                            start_positions[i] = -1
                        if end_position in found_end_positions:
                            end_positions[i] = -1
                        answer_tokens_as_str = ','.join([str(x) for x in answer_tokens])
                        if answer_tokens_as_str in found_answer_token_ids:
                            answer_token_ids[i] = []
                        found_start_positions.add(start_position)
                        found_end_positions.add(end_position)
                        found_answer_token_ids.add(answer_tokens_as_str)

                    input_ids_list.append(input_ids)
                    input_mask_list.append(input_mask)
                    segment_ids_list.append(segment_ids)
                    start_positions_list.append(start_positions)
                    end_positions_list.append(end_positions)
                    answer_token_ids_list.append(answer_token_ids)

                # pad answers in answer_token_ids_list to the longest answer
                max_answer_len = max([len(item) for sublist in answer_token_ids_list for item in sublist])  # flat list
                if max_answer_len == 0:
                    max_answer_len = 2
                for answers_of_one_slice in answer_token_ids_list:
                    for answer_tokens in answers_of_one_slice:
                        if len(answer_tokens) == 0:
                            # TODO: <s></s><pad><pad><pad> or <pad><pad><pad><pad><pad> ?
                            padding_len = max_answer_len - len(answer_tokens) - 2
                            answer_tokens.extend([self.tokenizer.bos_token_id, self.tokenizer.eos_token_id] +
                                                 ([self.tokenizer.pad_token_id] * padding_len))
                        else:
                            padding_len = max_answer_len - len(answer_tokens)
                            answer_tokens.extend([self.tokenizer.pad_token_id] * padding_len)

                tensors_list.append((torch.tensor(input_ids_list), torch.tensor(input_mask_list),
                                     torch.tensor(segment_ids_list),
                                     torch.tensor(start_positions_list), torch.tensor(end_positions_list),
                                     torch.tensor(answer_token_ids_list),
                                     self._get_qid(qa['id']),  qa["aliases"]))  # for eval
        return tensors_list

    def _get_qid(self, qid):
        """all input qids are formatted uniqueID__evidenceFile, but for wikipedia, qid = uniqueID,
        and for web, qid = uniqueID__evidenceFile. This function takes care of this conversion.
        """
        if 'wikipedia' in self.file_path:
            # for evaluation on wikipedia, every question has one answer even if multiple evidence documents are given
            return qid.split('--')[0]
        elif 'web' in self.file_path:
            # for evaluation on web, every question/document pair have an answer
            return qid
        elif 'sample' in self.file_path:
            return qid
        else:
            raise RuntimeError('Unexpected filename')

    @staticmethod
    def collate_one_doc_and_lists(batch):
        num_metadata_fields = 2  # qids and aliases
        fields = [x for x in zip(*batch)]
        stacked_fields = [torch.stack(field) for field in fields[:-num_metadata_fields]]  # don't stack metadata fields
        stacked_fields.extend(fields[-num_metadata_fields:])  # add them as lists not torch tensors

        # always use batch_size=1 where each batch is one document
        # will use grad_accum to increase effective batch size
        assert len(batch) == 1
        fields_with_batch_size_one = [f[0] for f in stacked_fields]
        return fields_with_batch_size_one


class TriviaQA(pl.LightningModule):

    def __init__(self, args):
        super(TriviaQA, self).__init__()
        self.args = args
        self.hparams = args

        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.tokenizer.model_max_length = self.args.max_seq_len
        self.model = self.load_model()
        self.num_labels = 2
        if not self.args.seq2seq:
            self.qa_outputs = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.train_dataloader_object = self.val_dataloader_object = self.test_dataloader_object = None

    def load_model(self):
        if 'longformer' in self.args.model_path:
            model = Longformer.from_pretrained(self.args.model_path)
            for layer in model.encoder.layer:
                layer.attention.self.attention_mode = self.args.attention_mode
                self.args.attention_window = layer.attention.self.attention_window
        elif self.args.model_path in ['bart.large', 'bart.base']:
            model = torch.hub.load('pytorch/fairseq', self.args.model_path)
            model.config = model.args
            model.config.hidden_size = model.config.decoder_output_dim
        elif 'bart' in self.args.model_path and 'base' in self.args.model_path:
            config = AutoConfig.from_pretrained(self.args.model_path)
            config.encoder_attention_heads = 12
            config.decoder_attention_heads = 12
            config.attention_dropout = 0.1
            if self.args.seq2seq:
                model = AutoModelWithLMHead.from_pretrained(self.args.model_path, config=config)
            else:
                model = AutoModel.from_pretrained(self.args.model_path, config=config)
        elif 'bart' in self.args.model_path and 'large' in self.args.model_path:
            config = AutoConfig.from_pretrained(self.args.model_path)
            config.attention_dropout = 0.1
            config.gradient_checkpointing = True
            if self.args.seq2seq:
                model = AutoModelWithLMHead.from_pretrained(self.args.model_path, config=config)
            else:
                model = AutoModel.from_pretrained(self.args.model_path, config=config)
        else:
            model = AutoModel.from_pretrained(self.args.model_path)

        print("Loaded model with config:")
        print(model.config)

        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        return model

    def forward(self, input_ids, attention_mask, segment_ids, start_positions, end_positions, answer_token_ids):
        if 'longformer' in self.args.model_path:
            question_end_index = self._get_question_end_index(input_ids)
            # Each batch is one document, and each row of the batch is a chunck of the document.
            # Make sure all rows have the same question length.
            assert (question_end_index[0].float() == question_end_index.float().mean()).item()

            # local attention everywhere
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
            # global attention for the question tokens
            attention_mask[:, :question_end_index.item()] = 2

            # sliding_chunks implemenation of selfattention requires that seqlen is multiple of window size
            input_ids, attention_mask = pad_to_window_size(
                input_ids, attention_mask, self.args.attention_window, self.tokenizer.pad_token_id)

            sequence_output = self.model(
                    input_ids,
                    attention_mask=attention_mask)[0]

            # The pretrained TriviaQA model wasn't trained with padding, so remove padding tokens
            # before computing loss and decoding.
            padding_len = input_ids[0].eq(self.tokenizer.pad_token_id).sum()
            if padding_len > 0:
                sequence_output = sequence_output[:, :-padding_len]
        elif self.args.model_path in ['bart.large', 'bart.base']:
            sequence_output = self.model.extract_features(input_ids)
        else:
            if self.args.seq2seq:
                decoder_input_ids = answer_token_ids[:, 0, :-1].clone()
                decoder_input_ids[decoder_input_ids == self.tokenizer.eos_token_id] = self.tokenizer.pad_token_id
                decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
                labels = answer_token_ids[:, 0, 1:].contiguous()
                labels[answer_token_ids[:, 0, 1:] == self.tokenizer.pad_token_id] = -100
                outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        labels=labels)
                loss = outputs[0]
                logit_scores = outputs[1].softmax(dim=2)[:, :, 0].sum(dim=1)
                return [loss, logit_scores]
            else:
                sequence_output = self.model(input_ids, attention_mask=attention_mask)[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            if not self.args.regular_softmax_loss:
                # loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf
                # NOTE: this returns sum of losses, not mean, so loss won't be normalized across different batch sizes
                # but batch size is always 1, so this is not a problem
                start_loss = self.or_softmax_cross_entropy_loss_one_doc(start_logits, start_positions, ignore_index=-1)
                end_loss = self.or_softmax_cross_entropy_loss_one_doc(end_logits, end_positions, ignore_index=-1)
            else:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
                start_positions = start_positions[:, 0:1]
                end_positions = end_positions[:, 0:1]
                start_loss = loss_fct(start_logits, start_positions[:, 0])
                end_loss = loss_fct(end_logits, end_positions[:, 0])

            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

    def or_softmax_cross_entropy_loss_one_doc(self, logits, target, ignore_index=-1, dim=-1):
        """loss function suggested in section 2.2 here https://arxiv.org/pdf/1710.10723.pdf"""
        assert logits.ndim == 2
        assert target.ndim == 2
        assert logits.size(0) == target.size(0)

        # with regular CrossEntropyLoss, the numerator is only one of the logits specified by the target
        # here, the numerator is the sum of a few potential targets, where some of them is the correct answer

        # compute a target mask
        target_mask = target == ignore_index
        # replaces ignore_index with 0, so `gather` will select logit at index 0 for the msked targets
        masked_target = target * (1 - target_mask.long())
        # gather logits
        gathered_logits = logits.gather(dim=dim, index=masked_target)
        # Apply the mask to gathered_logits. Use a mask of -inf because exp(-inf) = 0
        gathered_logits[target_mask] = float('-inf')

        # each batch is one example
        gathered_logits = gathered_logits.view(1, -1)
        logits = logits.view(1, -1)

        # numerator = log(sum(exp(gathered logits)))
        log_score = torch.logsumexp(gathered_logits, dim=dim, keepdim=False)
        # denominator = log(sum(exp(logits)))
        log_norm = torch.logsumexp(logits, dim=dim, keepdim=False)

        # compute the loss
        loss = -(log_score - log_norm)

        # some of the examples might have a loss of `inf` when `target` is all `ignore_index`.
        # remove those from the loss before computing the sum. Use sum instead of mean because
        # it is easier to compute
        return loss[~torch.isinf(loss)].sum()

    def training_step(self, batch, batch_nb):
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, answer_token_ids, qids, aliases = batch
        output = self.forward(input_ids, input_mask, segment_ids, subword_starts, subword_ends, answer_token_ids)
        loss = output[0]
        lr = loss.new_zeros(1) + self.trainer.optimizers[0].param_groups[0]['lr']
        tensorboard_logs = {'train_loss': loss, 'lr': lr,
                            'input_size': input_ids.numel(),
                            'mem': torch.cuda.memory_allocated(input_ids.device) / 1024 ** 3}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, answer_token_ids, qids, aliases = batch
        output = self.forward(input_ids, input_mask, segment_ids, subword_starts, subword_ends, answer_token_ids)
        if self.args.seq2seq:
            logit_scores = output[1]
            answer_score_indices = logit_scores.sort().indices
            generated_ids = self.model.generate(input_ids=input_ids, attention_mask=input_mask, use_cache=True,)
            answer_text = ''
            best_answer_score = 0
            for i in answer_score_indices:
                generated_answer_ids = generated_ids[answer_score_indices[i]]
                generated_answer_ids[-1] = self.tokenizer.eos_token_id
                index_of_eos_token = (generated_answer_ids == self.tokenizer.eos_token_id).nonzero()[0, 0].item()
                generated_answer_ids = generated_answer_ids[1:index_of_eos_token]
                answer_text = self.tokenizer.decode(generated_answer_ids)
                if answer_text != '':
                    best_answer_score = logit_scores[answer_score_indices[i]]
                    break
            f1_score = evaluation_utils.metric_max_over_ground_truths(evaluation_utils.f1_score, answer_text, aliases)
            em_score = evaluation_utils.metric_max_over_ground_truths(evaluation_utils.exact_match_score, answer_text, aliases)
            return {'vloss': output[0], 'vem': generated_answer_ids.new_zeros([1]).float(),
                    'qids': [qids], 'answer_scores': [best_answer_score],
                    'f1': [f1_score], 'em': [em_score]}

        loss, start_logits, end_logits = output[:3]
        answers = self.decode(input_ids, start_logits, end_logits)

        # each batch is one document
        answers = sorted(answers, key=lambda x: x['score'], reverse=True)[0:1]
        qids = [qids]
        aliases = [aliases]

        f1_scores = [evaluation_utils.metric_max_over_ground_truths(evaluation_utils.f1_score, answer['text'],
                                                                    aliase_list)
                     for answer, aliase_list in zip(answers, aliases)]
        # TODO: if slow, skip em_scores, and use (f1_score == 1.0) instead
        em_scores = [evaluation_utils.metric_max_over_ground_truths(evaluation_utils.exact_match_score, answer['text'],
                                                                    aliase_list)
                     for answer, aliase_list in zip(answers, aliases)]
        answer_scores = [answer['score'] for answer in answers]  # start_logit + end_logit
        assert len(answer_scores) == len(f1_scores) == len(em_scores) == len(qids) == len(aliases) == 1

        # TODO: delete this metric
        pred_subword_starts = start_logits.argmax(dim=1)
        pred_subword_ends = end_logits.argmax(dim=1)
        exact_match = (subword_ends[:, 0].squeeze(dim=-1) == pred_subword_ends).float() *  \
                      (subword_starts[:, 0].squeeze(dim=-1) == pred_subword_starts).float()

        return {'vloss': loss, 'vem': exact_match.mean(),
                'qids': qids, 'answer_scores': answer_scores,
                'f1': f1_scores, 'em': em_scores}

    def _get_question_end_index(self, input_ids):
        eos_token_indices = (input_ids == self.tokenizer.eos_token_id).nonzero()
        assert eos_token_indices.ndim == 2
        assert eos_token_indices.size(0) == 2 * input_ids.size(0)
        assert eos_token_indices.size(1) == 2
        return eos_token_indices.view(input_ids.size(0), 2, 2)[:, 0, 1]

    def decode(self, input_ids, start_logits, end_logits):
        # find beginning of document
        question_end_index = self._get_question_end_index(input_ids)

        # bsz x seqlen => bsz x n_best_size
        start_logits_indices = start_logits.topk(k=self.args.n_best_size, dim=-1).indices
        end_logits_indices = end_logits.topk(k=self.args.n_best_size, dim=-1).indices

        answers = []
        # This loop can't be vectorized, so loop over each example in the batch separetly
        for i in range(start_logits_indices.size(0)):  # bsz
            potential_answers = []
            for start_logit_index in start_logits_indices[i]:  # n_best_size
                for end_logit_index in end_logits_indices[i]:  # n_best_size
                    if start_logit_index <= question_end_index[i]:
                        continue
                    if end_logit_index <= question_end_index[i]:
                        continue
                    if start_logit_index > end_logit_index:
                        continue
                    answer_len = end_logit_index - start_logit_index + 1
                    if answer_len > self.args.max_answer_length:
                        continue
                    potential_answers.append({'start': start_logit_index, 'end': end_logit_index,
                                              'start_logit': start_logits[i][start_logit_index].item(),
                                              'end_logit': end_logits[i][end_logit_index].item()})
            sorted_answers = sorted(potential_answers, key=lambda x: (x['start_logit'] + x['end_logit']), reverse=True)
            if len(sorted_answers) == 0:
                answers.append({'text': 'NoAnswerFound', 'score': -1000000})
            else:
                answer = sorted_answers[0]
                answer_token_ids = input_ids[i, answer['start']: answer['end'] + 1]
                answer_tokens = self.tokenizer.convert_ids_to_tokens(answer_token_ids.tolist())
                text = self.tokenizer.convert_tokens_to_string(answer_tokens)
                score = answer['start_logit'] + answer['end_logit']
                answers.append({'text': text, 'score': score})
        return answers

    def sync_list_across_gpus(self, list_to_sync, device, dtype):
        l_tensor = torch.tensor(list_to_sync, device=device, dtype=dtype)
        gather_l_tensor = [torch.ones_like(l_tensor) for _ in range(self.trainer.world_size)]
        torch.distributed.all_gather(gather_l_tensor, l_tensor)
        return torch.cat(gather_l_tensor).tolist()

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['vloss'] for x in outputs]).mean()
        avg_em = torch.stack([x['vem'] for x in outputs]).mean()
        string_qids = [item for sublist in outputs for item in sublist['qids']]
        int_qids = [self.val_dataloader_object.dataset.val_qid_string_to_int_map[qid] for qid in string_qids]
        answer_scores = [item for sublist in outputs for item in sublist['answer_scores']]
        f1_scores = [item for sublist in outputs for item in sublist['f1']]
        em_scores = [item for sublist in outputs for item in sublist['em']]
        print(f'before sync --> sizes: {len(int_qids)}, {len(answer_scores)}, {len(f1_scores)}, {len(em_scores)}')
        if self.trainer.use_ddp:
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss /= self.trainer.world_size
            torch.distributed.all_reduce(avg_em, op=torch.distributed.ReduceOp.SUM)
            avg_em /= self.trainer.world_size

            int_qids = self.sync_list_across_gpus(int_qids, avg_loss.device, torch.int)
            answer_scores = self.sync_list_across_gpus(answer_scores, avg_loss.device, torch.float)
            f1_scores = self.sync_list_across_gpus(f1_scores, avg_loss.device, torch.float)
            em_scores = self.sync_list_across_gpus(em_scores, avg_loss.device, torch.int)
        print(f'after sync --> sizes: {len(int_qids)}, {len(answer_scores)}, {len(f1_scores)}, {len(em_scores)}')

        # Because of having multiple documents per questions, some questions might have multiple corresponding answers
        # Here, we only keep the answer with the highest answer_score
        qa_with_duplicates = defaultdict(list)
        for qid, answer_score, f1_score, em_score in zip(int_qids, answer_scores, f1_scores, em_scores):
            qa_with_duplicates[qid].append({'answer_score': answer_score, 'f1': f1_score, 'em': em_score})
        f1_scores = []
        em_scores = []
        for qid, answer_metrics in qa_with_duplicates.items():
            top_answer = sorted(answer_metrics, key=lambda x: x['answer_score'], reverse=True)[0]
            f1_scores.append(top_answer['f1'])
            em_scores.append(top_answer['em'])
        avg_val_f1 = sum(f1_scores) / len(f1_scores)
        avg_val_em = sum(em_scores) / len(em_scores)

        logs = {'val_loss': avg_loss, 'val_em': avg_em, 'avg_val_f1': avg_val_f1, 'avg_val_em': avg_val_em}

        return {'avg_val_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def test_step(self, batch, batch_nb):
        input_ids, input_mask, segment_ids, subword_starts, subword_ends, answer_token_ids, qids, aliases = batch
        output = self.forward(input_ids, input_mask, segment_ids, subword_starts, subword_ends, answer_token_ids)
        if self.args.seq2seq:
            raise NotImplemented

        loss, start_logits, end_logits = output[:3]
        answers = self.decode(input_ids, start_logits, end_logits)

        # each batch is one document
        answers = sorted(answers, key=lambda x: x['score'], reverse=True)[0:1]
        qids = [qids]
        assert len(answers) == len(qids)
        return {'qids': qids, 'answers': answers}

    def test_end(self, outputs):
        qids = [item for sublist in outputs for item in sublist['qids']]
        answers = [item for sublist in outputs for item in sublist['answers']]

        qa_with_duplicates = defaultdict(list)
        for qid, answer in zip(qids, answers):
            qa_with_duplicates[qid].append({'answer_score': answer['score'], 'answer_text': answer['text'], })

        qid_to_answer_text = {}
        for qid, answer_metrics in qa_with_duplicates.items():
            top_answer = sorted(answer_metrics, key=lambda x: x['answer_score'], reverse=True)[0]
            qid_to_answer_text[qid] = top_answer['answer_text']

        with open('predictions.json', 'w') as f:
            json.dump(qid_to_answer_text, f)

        return {'count': len(qid_to_answer_text)}

    def configure_optimizers(self):
        def lr_lambda(current_step):
            if current_step < self.args.warmup:
                return float(current_step) / float(max(1, self.args.warmup))
            return max(0.0, float(self.args.steps - current_step) / float(max(1, self.args.steps - self.args.warmup)))
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @pl.data_loader
    def train_dataloader(self):
        if self.train_dataloader_object is not None:
            return self.train_dataloader_object
        dataset = TriviaQADataset(file_path=self.args.train_dataset, tokenizer=self.tokenizer,
                                  max_seq_len=self.args.max_seq_len, max_doc_len=self.args.max_doc_len,
                                  doc_stride=self.args.doc_stride,
                                  max_num_answers=self.args.max_num_answers,
                                  max_question_len=self.args.max_question_len,
                                  ignore_seq_with_no_answers=self.args.ignore_seq_with_no_answers)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if self.trainer.use_ddp else None
        dl = DataLoader(dataset, batch_size=1, shuffle=(sampler is None),
                        num_workers=self.args.num_workers, sampler=sampler,
                        collate_fn=TriviaQADataset.collate_one_doc_and_lists)
        self.train_dataloader_object = dl
        return self.train_dataloader_object

    @pl.data_loader
    def val_dataloader(self):
        if self.val_dataloader_object is not None:
            return self.val_dataloader_object
        dataset = TriviaQADataset(file_path=self.args.dev_dataset, tokenizer=self.tokenizer,
                                  max_seq_len=self.args.max_seq_len, max_doc_len=self.args.max_doc_len,
                                  doc_stride=self.args.doc_stride,
                                  max_num_answers=self.args.max_num_answers,
                                  max_question_len=self.args.max_question_len,
                                  ignore_seq_with_no_answers=False)  # evaluation data should keep all examples
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if self.trainer.use_ddp else None
        dl = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=self.args.num_workers, sampler=sampler,
                        collate_fn=TriviaQADataset.collate_one_doc_and_lists)
        self.val_dataloader_object = dl
        return self.val_dataloader_object

    @pl.data_loader
    def test_dataloader(self):
        if self.test_dataloader_object is not None:
            return self.test_dataloader_object
        dataset = TriviaQADataset(file_path=self.args.dev_dataset, tokenizer=self.tokenizer,
                                  max_seq_len=self.args.max_seq_len, max_doc_len=self.args.max_doc_len,
                                  doc_stride=self.args.doc_stride,
                                  max_num_answers=self.args.max_num_answers,
                                  max_question_len=self.args.max_question_len,
                                  ignore_seq_with_no_answers=False)  # evaluation data should keep all examples

        dl = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=self.args.num_workers, sampler=None,
                        collate_fn=TriviaQADataset.collate_one_doc_and_lists)
        self.test_dataloader_object = dl
        return self.test_dataloader_object

    def configure_ddp(self, model, device_ids):
        model = LightningDistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=False
        )
        return model

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        parser.add_argument("--save_dir", type=str, default='triviaqa')
        parser.add_argument("--save_prefix", type=str, required=True)
        parser.add_argument("--train_dataset", type=str, required=False, help="Path to the training squad-format")
        parser.add_argument("--dev_dataset", type=str, required=True, help="Path to the dev squad-format")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
        parser.add_argument("--gpus", type=int, default=1,
                            help="Number of gpus. 0 for CPU")
        parser.add_argument("--warmup", type=int, default=200, help="Number of warmup steps")
        parser.add_argument("--lr", type=float, default=0.0001, help="Maximum learning rate")
        parser.add_argument("--val_every", type=float, default=0.5, help="Number of training steps between validations")
        parser.add_argument("--val_percent_check", default=1.00, type=float, help='Percent of validation data used')
        parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
        parser.add_argument("--seed", type=int, default=1234, help="Seed")
        parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
        parser.add_argument("--max_seq_len", type=int, default=4096,
                            help="Maximum length of seq passed to the transformer model")
        parser.add_argument("--max_doc_len", type=int, default=4096,
                            help="Maximum number of wordpieces of the input document")
        parser.add_argument("--max_num_answers", type=int, default=64,
                            help="Maximum number of answer spans per document (64 => 94%)")
        parser.add_argument("--max_question_len", type=int, default=55,
                            help="Maximum length of the question")
        parser.add_argument("--doc_stride", type=int, default=-1,
                            help="Overlap between document chunks. Use -1 to only use the first chunk")
        parser.add_argument("--ignore_seq_with_no_answers", action='store_true',
                            help="each example should have at least one answer. Default is False")
        parser.add_argument("--disable_checkpointing", action='store_true', help="No logging or checkpointing")
        parser.add_argument("--n_best_size", type=int, default=20,
                            help="Number of answer candidates. Used at decoding time")
        parser.add_argument("--max_answer_length", type=int, default=30,
                            help="maximum num of wordpieces/answer. Used at decoding time")
        parser.add_argument("--regular_softmax_loss", action='store_true',
                            help="IF true, use regular softmax. Default is using ORed softmax loss")
        parser.add_argument("--test", action='store_true', help="Test only, no training")
        parser.add_argument("--model_path", type=str, required=True,
                            help="Path to the checkpoint directory")
        parser.add_argument("--no_progress_bar", action='store_true', help="no progress bar. Good for printing")
        parser.add_argument("--attention_mode", type=str, choices=['tvm', 'sliding_chunks'],
                            default='sliding_chunks', help='Which implementation of selfattention to use')
        parser.add_argument("--fp32", action='store_true', help="default is fp16. Use --fp32 to switch to fp32")
        parser.add_argument("--seq2seq", action='store_true', help="Use an answer generation model")
        parser.add_argument("--resume_ckpt", type=str, help="Path of a checkpoint to resume from")


        return parser


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = TriviaQA(args)

    logger = TestTubeLogger(
        save_dir=args.save_dir,
        name=args.save_prefix,
        version=0  # always use version=0
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(args.save_dir, args.save_prefix, "checkpoints"),
        save_top_k=5,
        verbose=True,
        monitor='avg_val_loss',
        # save_last=True,
        mode='min',
        period=-1,
        prefix=''
    )

    print(args)
    train_set_size = 110648  # hardcode dataset size. Needed to compute number of steps for the lr scheduler
    args.steps = args.epochs * train_set_size / (args.batch_size * max(args.gpus, 1))
    print(f'>>>>>>> #steps: {args.steps}, #epochs: {args.epochs}, batch_size: {args.batch_size * args.gpus} <<<<<<<')

    trainer = pl.Trainer(gpus=args.gpus, distributed_backend='ddp' if args.gpus and args.gpus > 1 else None,
                         track_grad_norm=-1, max_epochs=args.epochs, early_stop_callback=None,
                         replace_sampler_ddp=False,
                         accumulate_grad_batches=args.batch_size,
                         val_check_interval=args.val_every,
                         num_sanity_val_steps=2,
                         # check_val_every_n_epoch=2,
                         val_percent_check=args.val_percent_check,
                         test_percent_check=args.val_percent_check,
                         logger=logger if not args.disable_checkpointing else False,
                         checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else False,
                         show_progress_bar=not args.no_progress_bar,
                         use_amp=not args.fp32, amp_level='O2',
                         resume_from_checkpoint=args.resume_ckpt,
                         )
    if not args.test:
        trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="triviaQa")
    parser = TriviaQA.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)
