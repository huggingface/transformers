# coding=utf-8
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
"""Utility function for nq evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import glob
from gzip import GzipFile
import json
import multiprocessing
from absl import flags
from absl import logging
from multiprocessing import cpu_count
threads_num = cpu_count()
# flags.DEFINE_integer(
#     'long_non_null_threshold', 2,
#     'Require this many non-null long answer annotations '
#     'to count gold as containing a long answer.')
# flags.DEFINE_integer(
#     'short_non_null_threshold', 2,
#     'Require this many non-null short answer annotations '
#     'to count gold as containing a short answer.')

# FLAGS = flags.FLAGS
short_non_null_threshold =2
long_non_null_threshold=2

# A data structure for storing prediction and annotation.
# When a example has multiple annotations, multiple NQLabel will be used.
NQLabel = collections.namedtuple(
    'NQLabel',
    [
        'example_id',  # the unique id for each NQ example.
        'long_answer_span',  # A Span object for long answer.
        'short_answer_span_list',  # A list of Spans for short answer.
        #   Note that In NQ, the short answers
        #   do not need to be in a single span.
        'yes_no_answer',  # Indicate if the short answer is an yes/no answer
        #   The possible values are "yes", "no", "none".
        #   (case insensitive)
        #   If the field is "yes", short_answer_span_list
        #   should be empty or only contain null spans.
        'long_score',  # The prediction score for the long answer prediction.
        'short_score'  # The prediction score for the short answer prediction.
    ])


class Span(object):
  """A class for handling token and byte spans.

    The logic is:

    1) if both start_byte !=  -1 and end_byte != -1 then the span is defined
       by byte offsets
    2) else, if start_token != -1 and end_token != -1 then the span is define
       by token offsets
    3) else, this is a null span.

    Null spans means that there is no (long or short) answers.
    If your systems only care about token spans rather than byte spans, set all
    byte spans to -1.

  """

  def __init__(self, start_byte, end_byte, start_token_idx, end_token_idx):

    if ((start_byte < 0 and end_byte >= 0) or
        (start_byte >= 0 and end_byte < 0)):
      raise ValueError('Inconsistent Null Spans (Byte).')

    if ((start_token_idx < 0 and end_token_idx >= 0) or
        (start_token_idx >= 0 and end_token_idx < 0)):
      raise ValueError('Inconsistent Null Spans (Token).')

    if start_byte >= 0 and end_byte >= 0 and start_byte >= end_byte:
      raise ValueError('Invalid byte spans (start_byte >= end_byte).')

    if ((start_token_idx >= 0 and end_token_idx >= 0) and
        (start_token_idx >= end_token_idx)):
      raise ValueError('Invalid token spans (start_token_idx >= end_token_idx)')

    self.start_byte = start_byte
    self.end_byte = end_byte
    self.start_token_idx = start_token_idx
    self.end_token_idx = end_token_idx

  def is_null_span(self):
    """A span is a null span if the start and end are both -1."""

    if (self.start_byte < 0 and self.end_byte < 0 and
        self.start_token_idx < 0 and self.end_token_idx < 0):
      return True
    return False

  def __str__(self):
    byte_str = 'byte: [' + str(self.start_byte) + ',' + str(self.end_byte) + ')'
    tok_str = ('tok: [' + str(self.start_token_idx) + ',' +
               str(self.end_token_idx) + ')')

    return byte_str + ' ' + tok_str

  def __repr__(self):
    return self.__str__()


def is_null_span_list(span_list):
  """Returns true iff all spans in span_list are null or span_list is empty."""
  if not span_list or all([span.is_null_span() for span in span_list]):
    return True
  return False


def nonnull_span_equal(span_a, span_b):
  """Given two spans, return if they are equal.

  Args:
    span_a: a Span object.
    span_b: a Span object.  Only compare non-null spans. First, if the bytes are
      not negative, compare byte offsets, Otherwise, compare token offsets.

  Returns:
    True or False
  """
  assert isinstance(span_a, Span)
  assert isinstance(span_b, Span)
  assert not span_a.is_null_span()
  assert not span_b.is_null_span()

  # if byte offsets are not negative, compare byte offsets
  if ((span_a.start_byte >= 0 and span_a.end_byte >= 0) and
      (span_b.start_byte >= 0 and span_b.end_byte >= 0)):

    if ((span_a.start_byte == span_b.start_byte) and
        (span_a.end_byte == span_b.end_byte)):
      return True

  # if token offsets are not negative, compare token offsets
  if ((span_a.start_token_idx >= 0 and span_a.end_token_idx >= 0) and
      (span_b.start_token_idx >= 0 and span_b.end_token_idx >= 0)):

    if ((span_a.start_token_idx == span_b.start_token_idx) and
        (span_a.end_token_idx == span_b.end_token_idx)):
      return True

  return False


def span_set_equal(gold_span_list, pred_span_list):
  """Make the spans are completely equal besides null spans."""

  gold_span_list = [span for span in gold_span_list if not span.is_null_span()]
  pred_span_list = [span for span in pred_span_list if not span.is_null_span()]

  for pspan in pred_span_list:
    # not finding pspan equal to any spans in gold_span_list
    if not any([nonnull_span_equal(pspan, gspan) for gspan in gold_span_list]):
      return False

  for gspan in gold_span_list:
    # not finding gspan equal to any spans in pred_span_list
    if not any([nonnull_span_equal(pspan, gspan) for pspan in pred_span_list]):
      return False

  return True


def gold_has_short_answer(gold_label_list):
  """Gets vote from multi-annotators for judging if there is a short answer."""

  #  We consider if there is a short answer if there is an short answer span or
  #  the yes/no answer is not none.
  gold_has_answer = gold_label_list and sum([
      ((not is_null_span_list(label.short_answer_span_list)) or
       (label.yes_no_answer != 'none')) for label in gold_label_list
  ]) >= short_non_null_threshold

  return gold_has_answer


def gold_has_long_answer(gold_label_list):
  """Gets vote from multi-annotators for judging if there is a long answer."""

  gold_has_answer = gold_label_list and (sum([
      not label.long_answer_span.is_null_span()  # long answer not null
      for label in gold_label_list  # for each annotator
  ]) >=long_non_null_threshold)

  return gold_has_answer


def read_prediction_json(predictions_path):
  """Read the prediction json with scores.

  Args:
    predictions_path: the path for the prediction json.

  Returns:
    A dictionary with key = example_id, value = NQInstancePrediction.

  """
  logging.info('Reading predictions from file: %s', format(predictions_path))
  with open(predictions_path, 'r') as f:
    predictions = json.loads(f.read())

  nq_pred_dict = {}
  for single_prediction in predictions['predictions']:

    if 'long_answer' in single_prediction:
      long_span = Span(single_prediction['long_answer']['start_byte'],
                       single_prediction['long_answer']['end_byte'],
                       single_prediction['long_answer']['start_token'],
                       single_prediction['long_answer']['end_token'])
    else:
      long_span = Span(-1, -1, -1, -1)  # Span is null if not presented.

    short_span_list = []
    if 'short_answers' in single_prediction:
      for short_item in single_prediction['short_answers']:
        short_span_list.append(
            Span(short_item['start_byte'], short_item['end_byte'],
                 short_item['start_token'], short_item['end_token']))

    yes_no_answer = 'none'
    if 'yes_no_answer' in single_prediction:
      yes_no_answer = single_prediction['yes_no_answer'].lower()
      if yes_no_answer not in ['yes', 'no', 'none']:
        raise ValueError('Invalid yes_no_answer value in prediction')

      if yes_no_answer != 'none' and not is_null_span_list(short_span_list):
        raise ValueError('yes/no prediction and short answers cannot coexist.')

    pred_item = NQLabel(
        example_id=single_prediction['example_id'],
        long_answer_span=long_span,
        short_answer_span_list=short_span_list,
        yes_no_answer=yes_no_answer,
        long_score=single_prediction['long_answer_score'],
        short_score=single_prediction['short_answers_score'])

    nq_pred_dict[single_prediction['example_id']] = pred_item

  return nq_pred_dict


def read_annotation_from_one_split(gzipped_input_file):
  """Read annotation from one split of file."""
  if isinstance(gzipped_input_file, str):
    gzipped_input_file = open(gzipped_input_file, 'rb')
  logging.info('parsing %s ..... ', gzipped_input_file.name)
  annotation_dict = {}
  with GzipFile(fileobj=gzipped_input_file) as input_file:
    for line in input_file:
      json_example = json.loads(line)
      example_id = json_example['example_id']

      # There are multiple annotations for one nq example.
      annotation_list = []

      for annotation in json_example['annotations']:
        long_span_rec = annotation['long_answer']
        long_span = Span(long_span_rec['start_byte'], long_span_rec['end_byte'],
                         long_span_rec['start_token'],
                         long_span_rec['end_token'])

        short_span_list = []
        for short_span_rec in annotation['short_answers']:
          short_span = Span(short_span_rec['start_byte'],
                            short_span_rec['end_byte'],
                            short_span_rec['start_token'],
                            short_span_rec['end_token'])
          short_span_list.append(short_span)

        gold_label = NQLabel(
            example_id=example_id,
            long_answer_span=long_span,
            short_answer_span_list=short_span_list,
            long_score=0,
            short_score=0,
            yes_no_answer=annotation['yes_no_answer'].lower())

        annotation_list.append(gold_label)
      annotation_dict[example_id] = annotation_list

  return annotation_dict


def read_annotation(path_name, n_threads=10):
  """Read annotations with real multiple processes."""
  # input_paths = glob.glob(path_name)
  input_paths = []
  for path in glob.glob("{}/*.gz".format(path_name)):
      input_paths.append(path)

  pool = multiprocessing.Pool(n_threads)
  try:
    dict_list = pool.map(read_annotation_from_one_split, input_paths)
  finally:
    pool.close()
    pool.join()

  final_dict = {}
  for single_dict in dict_list:
    final_dict.update(single_dict)

  return final_dict
