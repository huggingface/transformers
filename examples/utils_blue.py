# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" BERT classification fine-tuning: utilities to work with BLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class StsProcessor(DataProcessor):
    """Processor for the STS data set."""

    def get_train_examples(self, data_dir, *args):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, *args):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # ADDED
    def get_test_examples(self, data_dir, *args):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[-3]
            text_b = line[-2]
            label = float(line[-1])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class NerProcessor(DataProcessor):
    """Processor for the NER data set."""

    def get_train_examples(self, data_dir, *args):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, *args):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # ADDED
    def get_test_examples(self, data_dir, *args):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")


class RelProcessor(DataProcessor):
    """Processor for the REL data set."""

    def get_train_examples(self, data_dir, *args):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, *args):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # ADDED
    def get_test_examples(self, data_dir, *args):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # skip header
            if i == 0:
                continue
            guid = line[0]
            text_a = line[1]
            if set_type == "test":
                label = self.get_labels()[-1]
            else:
                try:
                    label = line[2]
                except IndexError:
                    logging.exception(line)
                    exit(1)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class BiossesProcessor(StsProcessor):
    """Processor for the BIOSSES data set (BLUE version)."""

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[-3]
            text_b = line[-2]
            label = str(round(float(line[-1])))
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class BiossesORGProcessor(DataProcessor):
    """Processor for the BIOSSES (ORG) data set (BLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class BC5CDRProcessor(NerProcessor):
    """Processor for the BC5CDR data set (BLUE version)."""

    def get_train_examples(self, data_dir, *args):
        """See base class."""
        tokenizer = args[0]
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", tokenizer)

    def get_dev_examples(self, data_dir, *args):
        """See base class."""
        tokenizer = args[0]
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", tokenizer)

    def get_labels(self):
        """See base class."""
        return ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]

    def _create_examples(self, lines, set_type, tokenizer):
        """Creates examples for the training and dev sets."""
        examples = []
        idx = 0
        tokens = []
        token_labels = []
        for line in lines:
            if len(line) < 2:
                guid = "%s-%s" % (set_type, idx)
                # text_a = tokens
                # label = token_labels
                text_a, label = self.tokenize(tokens, token_labels, tokenizer)
                text_b = None
                examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                idx += 1
                tokens = []
                token_labels = []
            else:
                tokens.append(line[0])
                token_labels.append(line[1].split()[-1])

        return examples

    def tokenize(self, tokens, token_labels, tokenizer):
        subtoken_labels = []
        subtokens = []
        for token, label in zip(tokens, token_labels):
            subtoken = tokenizer.tokenize(token)
            subtokens.extend(subtoken)
            if label.startswith('B-'):
                subtoken_labels.append(label)
                for i in range(len(subtoken) - 1):
                    subtoken_labels.append(label.replace('B-', 'I-'))
            else:
                for i in range(len(subtoken)):
                    subtoken_labels.append(label)
        assert len(subtoken_labels) == len(subtokens)
        return subtokens, subtoken_labels


class DDI2013Processor(RelProcessor):
    def get_labels(self):
        return ["DDI-advise", "DDI-effect", "DDI-int", "DDI-mechanism", 'DDI-false']


class HOCProcessor(DataProcessor):
    def get_train_examples(self, data_dir, *args):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, *args):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    # ADDED
    def get_test_examples(self, data_dir, *args):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ['sustaining', 'proliferative', 'signaling', 'evading', 'growth', 'suppressors', 'resisting', 'cell',
                'death', 'avoiding', 'immune', 'destruction', 'activating', 'invasion', 'metastasis', 'tumor',
                'promoting', 'inflammation', 'enabling', 'replicative', 'immortality', 'genomic', 'instability',
                'mutation', 'inducing', 'angiogenesis', 'cellular', 'energetics']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = None
            label = line[2].replace('and', ' ').replace(',', ' ').split()
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


# For multi-label task only
def build_mlb(label_list):
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    label_list = ['None'] + label_list
    label_map = {label: i for i, label in enumerate(label_list)}
    mlb.fit([sorted(label_map.values())[1:]])  # [1:] skip label O
    return mlb, label_map


def flatten_labels(preds, label_ids):
    flat_preds = []
    flat_label_ids = []
    for pred_t, label_id_t in zip(preds, label_ids):
        if len(pred_t) > 0 or len(label_id_t) > 0:
            for pred in pred_t:
                flat_preds.append(pred)
                if pred in label_id_t:
                    flat_label_ids.append(pred)
                else:
                    flat_label_ids.append(0)

            for label_id in label_id_t:
                if label_id not in pred_t:
                    flat_label_ids.append(label_id)
                    flat_preds.append(0)
        else:
            flat_preds.append(0)
            flat_label_ids.append(0)

    return flat_preds, flat_label_ids


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    if output_mode == 'mlb_classification':
        mlb, label_map = build_mlb(label_list)
    else:
        label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        if output_mode == 'token_classification':
            tokens_a = example.text_a
            label_id = [label_map[lbl] for lbl in example.label]
        else:
            tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if output_mode == 'token_classification':
            label_id = label_id + [-1]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            if output_mode == 'token_classification':
                label_id = label_id + [-1]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            if output_mode == 'token_classification':
                label_id = label_id + [-1]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            if output_mode == 'token_classification':
                label_id = [-1] + label_id
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            if output_mode == 'token_classification':
                label_id = ([-1] * padding_length) + label_id
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)
            if output_mode == 'token_classification':
                label_id = label_id + ([-1] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        if output_mode == 'token_classification':
            assert len(label_id) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        elif output_mode == 'mlb_classification':
            label_id = [label_map[lbl] for lbl in example.label]
            label_id = mlb.transform([label_id])[-1]
        elif output_mode != "token_classification":
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if output_mode == "token_classification" or output_mode == "mlb_classification":
                logger.info("label: %s" % " ".join([str(x) for x in label_id]))
            else:
                logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(data, preds, labels):
    assert len(preds) == len(labels)
    if data == "biosses":
        res = pearson_and_spearman(preds, labels)
        res['main'] = 'corr'
        return res
    if data in ["bc5cdr", "ddi", "hoc"]:
        res = acc_and_f1(preds, labels)
        res['main'] = 'f1'
        return res
    else:
        raise KeyError(data)


processors = {
    "biosses": BiossesProcessor,
    "bc5cdr": BC5CDRProcessor,
    "ddi": DDI2013Processor,
    "hoc": HOCProcessor,
}

output_modes = {
    "sts": "classification",
    "ner": 'token_classification',
    "rel": 'classification',
    "mul": 'mlb_classification'
}

BLUE_TASKS_NUM_LABELS = {
    "biosses": 1,
    "bc5cdr": 2,
    "ddi": 3,
    "hoc": 4,
}
