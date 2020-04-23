#!/usr/bin/env python3
# coding=utf-8

"""
This defines an input example format as well as classes to hold the data.
The logic here was largely derived from that in utils_ner.py though a lot of if statements for tokenization
was moved into the convert_examples_into_features function.
"""
import logging
import os


logger = logging.getLogger(__name__)


def pairwise(it):
    """
    A function to pairwise retrieve pieces of an iterable
    If given [1,2,3,4], this will return
    (1,2)
    (3,4)
    """
    it = iter(it)
    try:
        while True:
            yield next(it), next(it)
    # in python3.7 this raises a stopiteration that blocks example reading
    except StopIteration:
        pass


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, formatted_words=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            formatted_words: (Optional) list. The formatted version of the input words. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.formatted_words = formatted_words


"""
Example structure:
input:  march 23 2019
output: 3/23/2019
"""


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        formatted_words = []
        for words, formatted_words in pairwise(f):
            words = words.lstrip("input: ").strip()
            formatted_words = formatted_words.lstrip("output: ").strip()
            examples.append(
                InputExample(guid="{}-{}".format(mode, guid_index), words=words, formatted_words=formatted_words,)
            )
            guid_index += 1
    return examples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self, input_ids, output_ids, input_mask, output_mask, segment_ids, formatted_tokens,
    ):
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.input_mask = input_mask
        self.output_mask = output_mask
        self.segment_ids = segment_ids
        self.formatted_tokens = formatted_tokens


def convert_examples_to_features(
    examples, max_seq_length, tokenizer, model_type, pad_token_label_id=-100,
):
    """
    Loads a data file into a list of `InputBatch`s
    """
    #  `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    cls_token_segment_id = 2 if model_type in ["xlnet"] else 0
    pad_token_segment_id = 4 if model_type in ["xlnet"] else 0
    features = []

    cls_token, sep_token, pad_token = cls_token = (
        tokenizer.cls_token,
        tokenizer.sep_token,
        tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
    )
    sequence_a_segment_id = 0
    mask_padding_with_zero = True

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = tokenizer.tokenize(example.words)
        formatted_tokens = tokenizer.tokenize(example.formatted_words)

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if model_type in ["roberta"] else (2 if model_type != "gpt2" else 0)

        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            formatted_tokens = formatted_tokens[: (max_seq_length - special_tokens_count)]

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
        if model_type not in ["gpt2"]:
            tokens += [sep_token]
            formatted_tokens += [pad_token_label_id]

        if model_type in ["roberta"]:
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            tokens += [sep_token]
            formatted_tokens += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        # defines the location of the CLS token:
        # - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
        # - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        # xlnet/gpt is special
        if model_type in ["xlnet"]:
            tokens += [cls_token]
            formatted_tokens += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        # gpt2 has no cls_token
        elif model_type not in ["gpt2"]:
            pass
        # everything else does
        else:
            tokens = [cls_token] + tokens
            formatted_tokens = [pad_token_label_id] + formatted_tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        output_ids = tokenizer.convert_tokens_to_ids(formatted_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        output_mask = [1 if mask_padding_with_zero else 0] * len(output_ids)

        # Zero-pad up to the sequence length.
        input_padding_length = max_seq_length - len(input_ids)
        output_padding_length = max_seq_length - len(output_ids)

        # pad from the left this time
        if model_type in ["xlnet"]:
            input_ids = ([pad_token] * input_padding_length) + input_ids
            output_ids = ([pad_token] * output_padding_length) + output_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * input_padding_length) + input_mask
            output_mask = ([0 if mask_padding_with_zero else 1] * output_padding_length) + output_mask
            segment_ids = ([pad_token_segment_id] * input_padding_length) + segment_ids
            formatted_tokens = ([pad_token_label_id] * output_padding_length) + formatted_tokens
        # pad from the right this time
        else:
            input_ids += [pad_token] * input_padding_length
            output_ids += [pad_token] * output_padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * input_padding_length
            output_mask += [0 if mask_padding_with_zero else 1] * output_padding_length
            segment_ids += [pad_token_segment_id] * input_padding_length
            formatted_tokens += [pad_token_label_id] * output_padding_length

        # sanity checks
        assert all(token_id is not None for token_id in input_ids)
        assert all(token_id is not None for token_id in output_ids)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(output_ids) == max_seq_length
        assert len(output_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(formatted_tokens) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("formatted_tokens: %s", " ".join([str(x) for x in formatted_tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("output_ids: %s", " ".join([str(x) for x in output_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("output_mask: %s", " ".join([str(x) for x in output_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                output_ids=output_ids,
                input_mask=input_mask,
                output_mask=output_mask,
                segment_ids=segment_ids,
                formatted_tokens=formatted_tokens,
            )
        )
    return features
