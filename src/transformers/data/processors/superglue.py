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
""" SuperGLUE processors and helpers """


import json
import logging
import os
from collections import defaultdict

import numpy as np

from ...file_utils import is_tf_available
from .utils import DataProcessor, InputExample, InputFeatures, SpanClassificationExample, SpanClassificationFeatures


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def tokenize_tracking_span(tokenizer, text, spans):
    """
    Tokenize while tracking what tokens spans (char idxs) get mapped to
    Strategy: split input around span, tokenize left of span, the span,
        and then recursively apply to remaning text + spans
    We assume spans are
        - inclusive on start and end
        - non-overlapping (TODO)
        - sorted (TODO)

    Args:

    Returns:

    """
    toks = tokenizer.encode_plus(text, return_token_type_ids=True)
    full_toks = toks["input_ids"]
    prefix_len = len(tokenizer.decode(full_toks[:1])) + 1  # add a space
    len_covers = []
    for i in range(2, len(full_toks)):
        # iterate over the tokens and decode the length of the sequence
        # we start at 2 b/c 0 is empty (indexing from end); 1 is CLS/SOS
        partial_txt_len = len(tokenizer.decode(full_toks[:i], clean_up_tokenization_spaces=False))
        len_covers.append(partial_txt_len - prefix_len)

    span_locs = []
    for start, end in spans:
        start_tok, end_tok = None, None
        for tok_n, len_cover in enumerate(len_covers):
            if len_cover >= start and start_tok is None:
                start_tok = tok_n + 1  # account for [CLS] tok
            if len_cover >= end:
                assert start_tok is not None
                end_tok = tok_n + 1
                break
        assert start_tok is not None, "start_tok is None!"
        assert end_tok is not None, "end_tok is None!"
        span_locs.append((start_tok, end_tok))

    return toks, span_locs


def superglue_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: SuperGLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
            NB(AW): Writing predictions assumes the labels are in the same order as when building features.
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = superglue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = superglue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        if isinstance(example, SpanClassificationExample):
            inputs_a, span_locs_a = tokenize_tracking_span(tokenizer, example.text_a, example.spans_a)
            if example.spans_b is not None:
                inputs_b, span_locs_b = tokenize_tracking_span(tokenizer, example.text_b, example.spans_b)
                num_non_special_tokens = len(inputs_a["input_ids"]) + len(inputs_b["input_ids"]) - 4

                # TODO(AW): assumption is same number of non-special tokens + sos + eos
                #   This handles varying number of intervening tokens (e.g. different models)
                inputs = tokenizer.encode_plus(
                    example.text_a,
                    example.text_b,
                    add_special_tokens=True,
                    max_length=max_length,
                    return_token_type_ids=True,
                )
                num_joiner_specials = len(inputs["input_ids"]) - num_non_special_tokens - 2
                offset = len(inputs_a["input_ids"]) - 1 + num_joiner_specials - 1
                span_locs_b = [(s + offset, e + offset) for s, e in span_locs_b]
                span_locs = span_locs_a + span_locs_b
                input_ids = inputs["input_ids"]
                token_type_ids = inputs["token_type_ids"]

                if num_joiner_specials == 1:
                    tmp = inputs_a["input_ids"] + inputs_b["input_ids"][1:]
                elif num_joiner_specials == 2:
                    tmp = inputs_a["input_ids"] + inputs_b["input_ids"]
                else:
                    assert False, "Something is wrong"

                # check that the length of the input ids is expected (not necessarily the exact ids)
                assert len(input_ids) == len(tmp), "Span tracking tokenization produced inconsistent result!"

            else:
                input_ids, token_type_ids = inputs_a["input_ids"], inputs_a["token_type_ids"]
                span_locs = span_locs_a

        else:
            inputs = tokenizer.encode_plus(
                example.text_a,
                example.text_b,
                add_special_tokens=True,
                max_length=max_length,
                return_token_type_ids=True,
            )
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            # TODO(AW): will mess up span tracking
            assert False, "Not implemented correctly wrt span tracking!"
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids

        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )
        if output_mode in ["classification", "span_classification"]:
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input text: %s" % tokenizer.decode(input_ids, clean_up_tokenization_spaces=False))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        if isinstance(example, SpanClassificationExample):
            feats = SpanClassificationFeatures(
                guid=example.guid,
                input_ids=input_ids,
                span_locs=span_locs,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        else:
            feats = InputFeatures(
                guid=example.guid,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )

        features.append(feats)

    if is_tf_available() and is_tf_dataset:
        # TODO(AW): include span classification version

        def gen():
            for ex in features:
                yield (
                    {
                        "guid": ex.guid,
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "guid": tf.TensorShape([None]),
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features


class BoolqProcessor(DataProcessor):
    """Processor for the BoolQ data set (SuperGLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return [True, False]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = line["idx"]
            text_a = line["passage"]
            text_b = line["question"]
            label = line["label"] if "label" in line else False
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        """Write predictions in SuperGLUE format."""
        preds = preds[ex_ids]  # sort just in case we got scrambled
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "BoolQ.jsonl"), "w") as pred_fh:
            for idx, pred in enumerate(preds):
                pred_label = idx2label[int(pred)]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': pred_label})}\n")
        logger.info(f"Wrote {len(preds)} predictions to {out_dir}.")


class CbProcessor(DataProcessor):
    """Processor for the CommitmentBank data set (SuperGLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "contradiction", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # guid = "%s-%s" % (set_type, line["idx"])
            guid = line["idx"]
            text_a = line["premise"]
            text_b = line["hypothesis"]
            label = line["label"] if "label" in line else "contradiction"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        """Write predictions in SuperGLUE format."""
        preds = preds[ex_ids]  # sort just in case we got scrambled
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "CB.jsonl"), "w") as pred_fh:
            for idx, pred in enumerate(preds):
                pred_label = idx2label[int(pred)]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': pred_label})}\n")
        logger.info(f"Wrote {len(preds)} predictions to {out_dir}.")


class CopaProcessor(DataProcessor):
    """Processor for the COPA data set (SuperGLUE version)."""

    # TODO(AW)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # guid = "%s-%s" % (set_type, line["idx"])
            guid = line["idx"]
            label = line["label"] if "label" in line else 0
            premise = line["premise"][:-1]
            choice1 = line["choice1"]
            choice2 = line["choice2"]
            joiner = "because" if line["question"] == "cause" else "so"
            text_a = f"{premise} {joiner} {choice1}"
            text_b = f"{premise} {joiner} {choice2}"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        """Write predictions in SuperGLUE format."""
        preds = preds[ex_ids]  # sort just in case we got scrambled
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "COPA.jsonl"), "w") as pred_fh:
            for idx, pred in enumerate(preds):
                pred_label = idx2label[int(pred)]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': pred_label})}\n")
        logger.info(f"Wrote {len(preds)} predictions to {out_dir}.")


class MultircProcessor(DataProcessor):
    """Processor for the Multirc data set (SuperGLUE version)."""

    # TODO(AW)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        # NOTE(Alex): currently concatenating passage and question,
        # which might lead to the question getting cut off. An
        # alternative is to concatenate question and answer, but that
        # feels like there's a missing [SEP] token. Maybe the robust
        # solution is to use [SEP] tokens between everything.
        examples = []
        for (i, line) in enumerate(lines):
            passage_id = line["idx"]
            passage = line["passage"]["text"]
            for question_dict in line["passage"]["questions"]:
                question_id = question_dict["idx"]
                question = question_dict["question"]
                passage_and_question = " ".join([passage, question])
                for answer_dict in question_dict["answers"]:
                    answer_id = answer_dict["idx"]
                    # guid = "%s-%s-%s-%s" % (set_type, passage_id, question_id, answer_id)
                    guid = [passage_id, question_id, answer_id]
                    answer = answer_dict["text"]
                    label = answer_dict["label"] if "label" in answer_dict else 0
                    assert passage_and_question, "Empty passage and question!"
                    if answer == "":
                        # training data has a few blank answers
                        continue
                    examples.append(InputExample(guid=guid, text_a=passage_and_question, text_b=answer, label=label))
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        """Write predictions in SuperGLUE format."""
        # TODO(AW)

        psg2qst2ans = defaultdict(lambda: defaultdict(dict))
        for pred, ex_id in zip(preds, ex_ids):
            psg_id, qst_id, ans_id = map(int, ex_id)
            psg2qst2ans[psg_id][qst_id][ans_id] = pred

        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "MultiRC.jsonl"), "w") as pred_fh:
            for psg_id, qst2ans in psg2qst2ans.items():
                psgs = []
                for qst_id, ans2pred in qst2ans.items():
                    anss = []
                    for ans_id, pred in ans2pred.items():
                        pred_label = idx2label[pred]
                        anss.append({"idx": ans_id, "label": pred_label})
                    psgs.append({"idx": qst_id, "answers": anss})
                pred_fh.write(f"{json.dumps({'idx': psg_id, 'passage': {'questions': psgs}})}\n")

        logger.info(f"Wrote predictions to {out_dir}.")


class RecordProcessor(DataProcessor):
    """Processor for the ReCoRD data set (SuperGLUE version)."""

    def __init__(self):
        self._answers = None

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def get_answers(self, data_dir, set_type):
        """ """
        if self._answers is None or set_type not in self._answers:
            self._answers = {set_type: {}}
            data_file = f"{set_type}.jsonl"
            data = self._read_jsonl(os.path.join(data_dir, data_file))
            for (i, line) in enumerate(data):
                passage_id = line["idx"]
                passage = line["passage"]["text"]

                ents = []
                for ent_dict in line["passage"]["entities"]:
                    ents.append(passage[ent_dict["start"] : ent_dict["end"] + 1])
                for question_dict in line["qas"]:
                    question_id = question_dict["idx"]
                    # TODO(AW): no answer case
                    answers = [a["text"] for a in question_dict["answers"]] if "answers" in question_dict else []
                    self._answers[set_type][(passage_id, question_id)] = (ents, answers)

        return self._answers[set_type]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        qst2ans = {}
        for (i, line) in enumerate(lines):
            passage_id = line["idx"]
            passage = line["passage"]["text"]

            ents = []
            for ent_dict in line["passage"]["entities"]:
                ents.append(passage[ent_dict["start"] : ent_dict["end"] + 1])

            for question_dict in line["qas"]:
                question_id = question_dict["idx"]
                question_template = question_dict["query"]
                # TODO(AW): no answer case
                answers = [a["text"] for a in question_dict["answers"]] if "answers" in question_dict else []
                qst2ans[(passage_id, question_id)] = answers

                for ent_id, ent in enumerate(ents):
                    label = 1 if ent in answers else 0
                    candidate = question_template.replace("@placeholder", ent)
                    guid = [passage_id, question_id, ent_id]
                    examples.append(InputExample(guid=guid, text_a=passage, text_b=candidate, label=label))
        return examples

    # TODO(AW)
    def write_preds(self, preds, ex_ids, out_dir, answers):
        """Write predictions in SuperGLUE format."""
        # iterate over examples and aggregate predictions
        qst2ans = defaultdict(list)
        for idx, pred in zip(ex_ids, preds):
            qst_idx = (idx[0], idx[1])
            qst2ans[qst_idx].append((idx[2], pred))

        with open(os.path.join(out_dir, "ReCoRD.jsonl"), "w") as pred_fh:
            for qst, idxs_and_prds in qst2ans.items():
                cands, golds = answers[qst]
                psg_idx, qst_idx = map(int, qst)

                idxs_and_prds.sort(key=lambda x: x[0])
                logits = np.vstack([i[1] for i in idxs_and_prds])

                # take the most probable choice as the prediction
                pred_idx = logits[:, -1].argmax().item()
                pred = cands[pred_idx]

                pred_fh.write(f"{json.dumps({'idx': qst_idx, 'label': pred})}\n")
        logger.info(f"Wrote predictions to {out_dir}.")


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # guid = "%s-%s" % (set_type, line["idx"])
            guid = line["idx"]
            text_a = line["premise"]
            text_b = line["hypothesis"]
            label = line["label"] if "label" in line else "not_entailment"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        """Write predictions in SuperGLUE format."""
        preds = preds[ex_ids]  # sort just in case we got scrambled
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "RTE.jsonl"), "w") as pred_fh:
            for idx, pred in enumerate(preds):
                pred_label = idx2label[int(pred)]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': pred_label})}\n")
        logger.info(f"Wrote {len(preds)} predictions to {out_dir}.")


class WicProcessor(DataProcessor):
    """Processor for the WiC data set (SuperGLUE version)."""

    # TODO(AW)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return [True, False]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # guid = "%s-%s" % (set_type, line["idx"])
            guid = line["idx"]
            text_a = line["sentence1"]
            text_b = line["sentence2"]
            span_a = (line["start1"], line["end1"])
            span_b = (line["start2"], line["end2"])
            label = line["label"] if "label" in line else False
            examples.append(
                SpanClassificationExample(
                    guid=guid, text_a=text_a, spans_a=[span_a], text_b=text_b, spans_b=[span_b], label=label
                )
            )
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        """Write predictions in SuperGLUE format."""
        preds = preds[ex_ids]  # sort just in case we got scrambled
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "WiC.jsonl"), "w") as pred_fh:
            for idx, pred in enumerate(preds):
                pred_label = idx2label[int(pred)]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': pred_label})}\n")
        logger.info(f"Wrote {len(preds)} predictions to {out_dir}.")


class WscProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return [True, False]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # guid = "%s-%s" % (set_type, line["idx"])
            guid = line["idx"]
            text_a = line["text"]
            span_start1 = line["target"]["span1_index"]
            span_start2 = line["target"]["span2_index"]
            span_end1 = span_start1 + len(line["target"]["span1_text"])
            span_end2 = span_start2 + len(line["target"]["span2_text"])
            span1 = (span_start1, span_end1)
            span2 = (span_start2, span_end2)
            label = line["label"] if "label" in line else False
            examples.append(SpanClassificationExample(guid=guid, text_a=text_a, spans_a=[span1, span2], label=label))
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        """Write predictions in SuperGLUE format."""
        preds = preds[ex_ids]  # sort just in case we got scrambled
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "WSC.jsonl"), "w") as pred_fh:
            for idx, pred in enumerate(preds):
                pred_label = idx2label[int(pred)]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': pred_label})}\n")
        logger.info(f"Wrote {len(preds)} predictions to {out_dir}.")


class DiagnosticBroadProcessor(DataProcessor):
    """Processor for the braod coverage diagnostic data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        raise AssertionError("Diagnostic tasks only have test data! Call get_test_examples instead")

    def get_dev_examples(self, data_dir):
        """See base class."""
        raise AssertionError("Diagnostic tasks only have test data! Call get_test_examples instead")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "AX-b.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # guid = "%s-%s" % (set_type, line["idx"])
            guid = int(line["idx"])
            text_a = line["sentence1"]
            text_b = line["sentence2"]
            label = line["label"] if "label" in line else "not_entailment"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        """Write predictions in SuperGLUE format."""
        preds = preds[ex_ids]  # sort just in case we got scrambled
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "AX-b.jsonl"), "w") as pred_fh:
            for idx, pred in enumerate(preds):
                pred_label = idx2label[int(pred)]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': pred_label})}\n")
        logger.info(f"Wrote {len(preds)} predictions to {out_dir}.")


class DiagnosticGenderProcessor(DataProcessor):
    """Processor for the gender bias diagnostic dataset."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        raise AssertionError("Diagnostic tasks only have test data! Call get_test_examples instead")

    def get_dev_examples(self, data_dir):
        """See base class."""
        raise AssertionError("Diagnostic tasks only have test data! Call get_test_examples instead")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "AX-g.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # guid = "%s-%s" % (set_type, line["idx"])
            guid = int(line["idx"])
            text_a = line["premise"]
            text_b = line["hypothesis"]
            label = line["label"] if "label" in line else "not_entailment"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def write_preds(self, preds, ex_ids, out_dir):
        """Write predictions in SuperGLUE format."""
        preds = preds[ex_ids]  # sort just in case we got scrambled
        idx2label = {i: label for i, label in enumerate(self.get_labels())}
        with open(os.path.join(out_dir, "AX-g.jsonl"), "w") as pred_fh:
            for idx, pred in enumerate(preds):
                pred_label = idx2label[int(pred)]
                pred_fh.write(f"{json.dumps({'idx': idx, 'label': pred_label})}\n")
        logger.info(f"Wrote {len(preds)} predictions to {out_dir}.")


superglue_tasks_num_labels = {
    "ax-b": 2,
    "ax-g": 2,
    "boolq": 2,
    "cb": 3,
    "copa": 2,
    "rte": 2,
    "wic": 2,
    "wsc": 2,
}

superglue_tasks_num_spans = {
    "wic": 2,
    "wsc": 2,
}

superglue_processors = {
    "ax-b": DiagnosticBroadProcessor,
    "ax-g": DiagnosticGenderProcessor,
    "boolq": BoolqProcessor,
    "cb": CbProcessor,
    "copa": CopaProcessor,
    "multirc": MultircProcessor,
    "record": RecordProcessor,
    "rte": RteProcessor,
    "wic": WicProcessor,
    "wsc": WscProcessor,
}

superglue_output_modes = {
    "ax-b": "classification",
    "ax-g": "classification",
    "boolq": "classification",
    "cb": "classification",
    "copa": "classification",
    "multirc": "classification",
    "record": "classification",
    "rte": "classification",
    "wic": "span_classification",
    "wsc": "span_classification",
}

superglue_tasks_metrics = {
    "boolq": "acc",
    "cb": "acc_and_f1",
    "copa": "acc",
    "multirc": "em_and_f1",
    "record": "em_and_f1",
    "rte": "acc",
    "wic": "acc",
    "wsc": "acc_and_f1",
}
