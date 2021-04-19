# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors.
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
#
# This file has been modified by Graphcore Ltd.

"""
This script has been adapated from the original google-research/bert repo found here:
  https://github.com/google-research/bert/blob/master/create_pretraining_data.py

Main changes:
  Load tokeniser from transformers
  Update to Tensorflow 2
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import collections
import random
import argparse
import glob
import struct
from itertools import chain
from functools import reduce
import six

from transformers import BertTokenizerFast

try:
  import tensorflow as tf
except ImportError:
  raise ImportError("Tensorflow is required to generate data for this application. "
                    "Please install with: 'pip install tensorflow'")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    mask_tokens, output_files,
                                    ignore_index_value=0, max_samples=-1):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(tqdm(instances)):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < mask_tokens:
      masked_lm_positions.append(0)
      masked_lm_ids.append(ignore_index_value)
      masked_lm_weights.append(0.0)

    next_sentence_label = [1 if instance.is_random_next else 0]

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature(next_sentence_label)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if max_samples != -1 and total_written >= max_samples:
      break

  for writer in writers:
    writer.close()

  print("Wrote %d total instances" % total_written)

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature

def count_lines(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, mlm_prob,
                              mask_tokens, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    lines = count_lines(input_file)
    print(f"Line count {lines}")
    with open(input_file, "r") as reader:
      print(f"*** Tokenising input file '{input_file}' ***")
      for line_count in tqdm(range(lines+1)):
        line = convert_to_unicode(reader.readline())
        # Need to preprocess for lower case etc.
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)

  print("*** Done Tokenising ***")

  # Remove empty documents
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = list(tokenizer.get_vocab().keys())
  instances = []
  for dup in range(dupe_factor):
    print(f"*** Generating Duplicate {dup}/{dupe_factor} ***")
    for document_index in tqdm(range(len(all_documents))):
      instances.extend(
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              mlm_prob, mask_tokens, vocab_words, rng))

  rng.shuffle(instances)
  return instances


def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        mlm_prob, mask_tokens, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    segment = document[i]
    current_chunk.append(segment)
    current_length += len(segment)
    if i == len(document) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        # Random next
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          random_document = all_documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        # Actual next
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, mlm_prob, mask_tokens, vocab_words, rng, max_seq_length)
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)
      current_chunk = []
      current_length = 0
    i += 1

  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, mlm_prob,
                                 mask_tokens, vocab_words, rng,
                                 max_seq_length):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(mask_tokens,
                       max(1, int(round(len(tokens) * mlm_prob))))
  # MAJOR CHANGE: Make sure that the desired number of mask_tokens is used
  # to prevent issues with too many tokens needed. Ie
  # sequence_length 384, mask_tokens 60. if num_to_predict is 58
  # and len(tokens) is 384 then there will be 326 sequence_tokens with 60 additional tokens
  # reserved for mask tokens. This will give a total tokens of 386 which is too many.
  num_to_predict = max(num_to_predict, len(tokens) - max_seq_length + mask_tokens)

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()


def main(args):
  # We use the HuggingFace BERT tokenizer rather than that from
  # the google-research/albert repo so that we can ensure
  # the generated dataset is compatible when pretraining with
  # HuggingFace BERT.
  tokenizer = BertTokenizerFast.from_pretrained(args.model,
    do_lower_case=args.do_lower_case
  )

  input_files = []
  for input_pattern in args.input_file.split(","):
    input_files.extend(glob.glob(input_pattern))

  print("*** Reading from input files ***")
  for input_file in input_files:
    print(f"  {input_file}")

  print("*** Done reading files ***")

  rng = random.Random(args.seed)
  instances = create_training_instances(
      input_files, tokenizer, args.sequence_length, args.duplication_factor,
      args.short_seq_prob, args.mlm_prob, args.mask_tokens,
      rng)

  output_files = args.output_file.split(",")
  print("*** Writing to output files ***")
  for output_file in output_files:
    print(f"  {output_file}")

  write_instance_to_example_files(instances, tokenizer, args.sequence_length,
                                  args.mask_tokens, output_files,
                                  args.ignore_index_value, args.max_samples)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      "PreTraining Data for Mlm/Next Sentence prediction")
  parser.add_argument("--input-file", type=str, required=True)
  parser.add_argument('--model', type=str, required=True)
  parser.add_argument("--output-file", type=str, required=True)
  parser.add_argument("--do-lower-case", action="store_true")
  parser.add_argument("--sequence-length", type=int, default=128)
  parser.add_argument("--mask-tokens", type=int, default=20)
  parser.add_argument("--seed", type=int, default=1984)
  parser.add_argument("--duplication-factor", type=int, default=6)
  parser.add_argument("--mlm-prob", type=float, default=0.15)
  parser.add_argument("--short-seq-prob", type=float, default=0.1)
  parser.add_argument("--max-samples", type=int, default=-1)
  parser.add_argument("--ignore-index-value", type=int, default=0,
                      help="The value to set to be ignored in the masked_lm label. "
                           "Tensorflow BERT typically uses 0. Whereas Pytorch uses -100 by default.")
  args = parser.parse_args()

  main(args)
