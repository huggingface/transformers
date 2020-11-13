import collections
import csv
import enum
import os
from typing import MutableMapping, Text, Tuple, Iterable, List

import pandas as pd
from absl import logging
from tapas_file_utils import (list_directory, make_directories)

from tapas_text_utils import (wtq_normalize)

_TABLE_DIR_NAME = 'table_csv'  # Name that the table folder has in SQA.

class Version(enum.Enum):
  V_02 = 1
  V_10 = 2

def _export_table(table, output_dir,
                  sqa_table_id):
  output_file = os.path.join(output_dir, sqa_table_id)
  with open(output_file, 'w') as table_out:
    table.to_csv(
        table_out,
        sep=',',
        escapechar='\\',
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding='utf-8')

def _get_reader(file_path):
  return open(file_path, 'r')

def _get_sqa_file_path(input_dir, file_name):
  return os.path.join(input_dir, 'data', file_name)

def _get_random_split_name(
    split_number,
    version,
):
  """Gets train and dev files for a split index."""
  if version == Version.V_02:
    name = 'random-split-seed-{}-{}.tsv'
    return name.format(split_number, 'train'), name.format(split_number, 'test')
  if version == Version.V_10:
    name = 'random-split-{}-{}.tsv'
    return name.format(split_number, 'train'), name.format(split_number, 'dev')
  raise ValueError(f'Unknown version {version}')

def _get_sqa_table_id(wtq_table_id):
  """Goes from 'csv/123-csv/123.csv' to 'table_csv/123-123.csv'."""
  return u'table_csv/' + wtq_table_id[4:].replace('/', '-').replace('-csv', '')

def _read_wtq_table(input_dir, wtq_table_id):
  """Reads table file as pandas frame."""
  table_path = os.path.join(input_dir, wtq_table_id)
  with open(table_path, 'r') as table_in:
    return pd.read_csv(
        table_in,
        delimiter=',',
        escapechar='\\',
        dtype='str',
    )

def _iterate_examples(
    file_in,
    version,
):
  """Reads examples from TSV file."""
  if version == Version.V_02:
    for line in file_in:
      fields = line.rstrip().split('\t')
      qid = fields[0]
      question = fields[1]
      wtq_table_id = fields[2]
      answers = fields[3:]
      yield qid, question, wtq_table_id, answers
  if version == Version.V_10:
    for line in csv.DictReader(file_in, delimiter='\t'):
      # Parse question and answers.
      qid = line['id']
      question = line['utterance']
      wtq_table_id = line['context']
      answers = line['targetValue'].split('|')
      yield qid, question, wtq_table_id, answers

def _convert_data(
    table_cache,
    input_dir,
    output_dir,
    file_name,
    version,
):
  """Converts WTQ data to SQA TSV format."""
  logging.info('Converting data from: %s...', file_name)

  counter = collections.Counter()  # Counter for stats.
  sqa_data = []  # List of rows with data in SQA format.

  with _get_reader(_get_sqa_file_path(input_dir, file_name)) as file_in:
    for example in _iterate_examples(file_in, version):
      # Parse question and answers.
      qid, question, wtq_table_id, answers = example

      sqa_table_id = _get_sqa_table_id(wtq_table_id)

      # Get table from disk or from cache.
      if sqa_table_id in table_cache:
        table = table_cache[sqa_table_id]
      else:
        table = _read_wtq_table(input_dir, wtq_table_id)
        table = table.applymap(wtq_normalize)
        table_cache[sqa_table_id] = table

      sqa_row = []
      sqa_row.append(qid)
      sqa_row.append('0')
      sqa_row.append('0')
      sqa_row.append(question)
      sqa_row.append(sqa_table_id)
      sqa_row.append(str(list(map(str, [(-1, -1) for _ in answers]))))
      sqa_row.append(str(answers))
      sqa_row.append('NONE')
      sqa_row.append('')
      sqa_data.append(sqa_row)

      counter['questions'] += 1
      if counter['questions'] % 100 == 0:
        logging.info('Processed %s questions...', counter['questions'])

    df_columns = [
        'id', 'annotator', 'position', 'question', 'table_file',
        'answer_coordinates', 'answer_text', 'aggregation', 'float_answer'
    ]
    df = pd.DataFrame(data=sqa_data, columns=df_columns, dtype=str)

    # Manipulate names to match the expected names of SQA files.
    if file_name == 'training.tsv':
      file_name = 'train.tsv'
    elif file_name == 'pristine-unseen-tables.tsv':
      file_name = 'test.tsv'
    elif 'random-split-seed' in file_name:
      file_name = file_name.replace('-test', '-dev').replace('-seed', '')

    # Write to disk.
    output_file = os.path.join(output_dir, file_name)
    df.to_csv(
        open(output_file, 'w'),
        sep='\t',
        index=False,
        encoding='utf-8')

def _get_train_test(
    split_number,
    version,
):
  """Returns the name of the train/test data splits.
  Args:
    split_number: Index of random split.
    version: WTQ version.
  Returns:
    Name of train and test file.
  Provide 1 to 5 for dev splits, None or any other value for test splits.
  """
  if 1 <= split_number <= 5:
    return _get_random_split_name(split_number, version)
  return 'training.tsv', 'pristine-unseen-tables.tsv'

def _create_dirs(output_dir):
  make_directories(os.path.join(output_dir, _TABLE_DIR_NAME))

def convert(
    input_dir,
    output_dir,
    version = Version.V_10,
):
  """Converts from WTQ to SQA format.
  Args:
    input_dir: The original WTQ data.
    output_dir: Where files converted to SQA format will be written to.
    version: WTQ version.
  This will create the following file structure in 'output_dir':
    random-split-i-dev.tsv (i: 1 ... 5)
    random-split-i-train.tsv (i: 1 ... 5)
    test.tsv
    train.tsv
    table_csv/???-???.csv (e.g. '202-184.csv' or '200-0.csv', 2100 files)
  """
  _create_dirs(output_dir)
  table_cache = {}
  # 0 is test split, 1 to 5 are dev splits.
  for idx in range(0, 6):
    train_file, test_file = _get_train_test(idx, version)
    _convert_data(table_cache, input_dir, output_dir, train_file, version)
    _convert_data(table_cache, input_dir, output_dir, test_file, version)

  for sqa_table_id, table in table_cache.items():
    _export_table(table, output_dir, sqa_table_id)