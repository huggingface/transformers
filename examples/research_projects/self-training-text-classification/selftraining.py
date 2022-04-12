# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

#!/usr/bin/env python
# coding=utf-8
"""Self-training for sequence classification."""

import argparse
import dataclasses
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional
from accelerate import Accelerator
import datasets
from datasets import load_dataset
from finetuning import finetune
from tqdm.auto import tqdm
import transformers
from transformers import AutoConfig
from transformers import set_seed
from transformers.trainer_utils import IntervalStrategy

logger = logging.getLogger(__name__)

MODEL_BIN_FILE = 'pytorch_model.bin'


@dataclasses.dataclass
class STModelArguments:
  """Arguments pertaining to which config/tokenizer/model we are going to fine-tune from."""

  model_name_or_path: str = dataclasses.field(
      metadata={
          'help':
              'Path to pretrained model or model identifier from huggingface.co/models.'
      })
  cache_dir: Optional[str] = dataclasses.field(
      default=None,
      metadata={
          'help':
              'Where do you want to store the pretrained models downloaded from huggingface.co.'
      },
  )


@dataclasses.dataclass
class STDataArguments:
  """Arguments pertaining to what data we are going to input our model for training and evaluation."""

  train_file: str = dataclasses.field(
      metadata={'help': 'A csv or a json file containing the training data.'})
  infer_file: str = dataclasses.field(metadata={
      'help': 'A csv or a json file containing the data to predict on.'
  })
  eval_file: Optional[str] = dataclasses.field(
      default=None,
      metadata={'help': 'A csv or a json file containing the validation data.'})
  task_name: Optional[str] = dataclasses.field(
      default=None,
      metadata={'help': 'The name of the task to train on.'},
  )
  label_list: Optional[List[str]] = dataclasses.field(
      default=None, metadata={'help': 'The list of labels for the task.'})


@dataclasses.dataclass
class STTrainingArguments():
  """Training arguments pertaining to the training loop itself."""

  output_dir: str = dataclasses.field(
      metadata={
          'help':
              'The output directory where the model predictions and checkpoints will be written.'
      })
  eval_metric: Optional[str] = dataclasses.field(
      default='accuracy',
      metadata={'help': 'The evaluation metric used for the task.'})
  evaluation_strategy: Optional[str] = dataclasses.field(
      default='no',
      metadata={
          'help':
              'The evaluation strategy to adopt during training. Possible values are: ["no", "step", "epoch]'
      })
  early_stopping_patience: Optional[int] = dataclasses.field(
      default=10,
      metadata={
          'help':
              'Number of evaluation calls with no improvement after which training will be stopped.'
      },
  )
  early_stopping_threshold: Optional[float] = dataclasses.field(
      default=0.0,
      metadata={
          'help':
              'How much the specified evaluation metric must improve to satisfy early stopping conditions.'
      },
  )
  do_filter_by_confidence: Optional[bool] = dataclasses.field(
      default=False,
      metadata={
          'help':
              'Whether to filter the pseudo-labeled data based on the confidence score.'
      },
  )
  do_filter_by_val_performance: Optional[bool] = dataclasses.field(
      default=False,
      metadata={
          'help':
              'Whether to filter the pseudo-labeled data based on the validation performance.'
      },
  )
  finetune_on_labeled_data: Optional[bool] = dataclasses.field(
      default=False,
      metadata={
          'help': 'Whether to fine-tune on labeled data after pseudo training.'
      },
  )
  confidence_threshold: Optional[float] = dataclasses.field(
      default=0.0,
      metadata={
          'help': 'Confidence threshold for pseudo-labeled data filtering.'
      },
  )
  max_selftrain_iterations: Optional[int] = dataclasses.field(
      default=100,
      metadata={
          'help':
              'Number of evaluation calls with no improvement after which training will be stopped.'
      },
  )
  seed: Optional[int] = dataclasses.field(
      default=None,
      metadata={'help': 'Random seed for initialization.'},
  )


def create_pseudo_labeled_data(args,
                               infer_input,
                               infer_output,
                               eval_result, id2label,
                               next_data_dir):
  """Create pseudeo labeled data for the next self-training iteration."""

  dataset = datasets.concatenate_datasets([infer_input, infer_output], axis=1)

  if args.do_filter_by_confidence:
    dataset = dataset.filter(
        lambda example: example['probability'] > args.confidence_threshold)

  if args.do_filter_by_val_performance:
    assert eval_result >= 0.0 and eval_result <= 1.0
    num_selected_rows = int(eval_result * len(dataset))
    print(num_selected_rows)
    dataset = dataset.sort('probability', reverse=True)
    dataset = dataset.select(range(num_selected_rows))

  dataset = dataset.remove_columns(['label', 'probability'])
  dataset = dataset.rename_column('prediction', 'label')
  dataset = dataset.map(lambda example: {'label': id2label[example['label']]})
  dataset = dataset.shuffle(seed=args.seed)

  pseudo_labeled_data_file = os.path.join(
      next_data_dir, f'train_pseudo.{args.data_file_extension}')
  if args.data_file_extension == 'csv':
    dataset.to_csv(pseudo_labeled_data_file, index=False)
  else:
    dataset.to_json(pseudo_labeled_data_file)


def selftrain(model_name_or_path, train_file, infer_file,
              output_dir, **kwargs):
  """Self-training a pre-trained model on a downstream task.

  Args:
    model_name_or_path: Path to pretrained model or model identifier from
      huggingface.co/models.
    train_file: A csv or a json file containing the training data.
    infer_file: A csv or a json file containing the data to predict on.
    output_dir: The output directory where the model predictions and checkpoints
      will be written.
    **kwargs: Dictionary of key/value pairs with which to update the
      configuration object after loading. The values in kwargs of any keys which
      are configuration attributes will be used to override the loaded values.
  """
  # Initialize the accelerator. We will let the accelerator handle device
  # placement for us.
  accelerator = Accelerator()
  # Make one log on every process with the configuration for debugging.
  logging.basicConfig(
      format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
      datefmt='%m/%d/%Y %H:%M:%S',
      level=logging.INFO,
  )
  logger.info(accelerator.state)

  # Setup logging, we only want one process per machine to log things on the
  # screen. accelerator.is_local_main_process is only True for one process per
  # machine.
  logger.setLevel(
      logging.INFO if accelerator.is_local_main_process else logging.ERROR)

  if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
  else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

  model_args = STModelArguments(model_name_or_path=model_name_or_path)
  data_args = STDataArguments(train_file=train_file, infer_file=infer_file)
  training_args = STTrainingArguments(output_dir=output_dir)
  args = argparse.Namespace()

  for arg_class in (model_args, data_args, training_args):
    for key, value in vars(arg_class).items():
      setattr(args, key, value)

  for key, value in kwargs.items():
    if hasattr(args, key):
      setattr(args, key, value)

  # Sanity checks
  data_files = {}
  args.data_file_extension = None

  # You need to provide the training data and the data to predict on
  assert args.train_file is not None
  assert args.infer_file is not None
  data_files['train'] = args.train_file
  data_files['infer'] = args.infer_file

  if args.evaluation_strategy != IntervalStrategy.NO.value:
    assert args.eval_file is not None
    data_files['eval'] = args.eval_file

  for key in data_files:
    extension = data_files[key].split('.')[-1]
    assert extension in ['csv', 'json'
                        ], f'`{key}_file` should be a csv or a json file.'
    if args.data_file_extension is None:
      args.data_file_extension = extension
    else:
      assert (extension == args.data_file_extension
             ), f'`{key}_file` should be a {args.data_file_extension} file`.'

  assert (
      args.eval_metric in datasets.list_metrics()
  ), f'{args.eval_metric} not in the list of supported metrics {datasets.list_metrics()}.'

  # If passed along, set the training seed now.
  if args.seed is not None:
    set_seed(args.seed)

  logger.info('Creating the initial data directory for self-training...')
  data_dir_format = f'{args.output_dir}/self-train_iter-{{}}'.format
  initial_data_dir = data_dir_format(0)

  if accelerator.is_main_process:
    if args.output_dir is not None:
      os.makedirs(args.output_dir, exist_ok=True)
      os.makedirs(initial_data_dir, exist_ok=True)
  accelerator.wait_for_everyone()

  best_iteration = None
  best_eval_result = None
  early_stopping_patience_counter = 0
  should_training_stop = False
  # Show the progress bar
  progress_bar = tqdm(
      range(args.max_selftrain_iterations),
      disable=not accelerator.is_local_main_process)

  # Self-train
  for iteration in range(0, int(args.max_selftrain_iterations)):
    current_data_dir = data_dir_format(iteration)
    assert os.path.exists(current_data_dir)

    # Stage 1: initial fine-tuning for iteration = 0 or pseudo-training for
    # iteration > 0
    current_output_dir = os.path.join(current_data_dir, 'stage-1')
    arguments_dict = {
        'accelerator':
            accelerator,
        'model_name_or_path':
            args.model_name_or_path,
        'cache_dir':
            args.cache_dir,
        'do_train':
            True,
        'train_file':
            data_files['train']
            if iteration == 0 else data_files['train_pseudo'],
        'do_eval':
            True if args.eval_file is not None else False,
        'eval_file':
            data_files['eval'],
        'do_predict':
            True,
        'infer_file':
            data_files['infer'],
        'task_name':
            args.task_name,
        'label_list':
            args.label_list,
        'output_dir':
            current_output_dir,
        'eval_metric':
            args.eval_metric,
        'evaluation_strategy':
            args.evaluation_strategy,
        'early_stopping_patience':
            args.early_stopping_patience,
        'early_stopping_threshold':
            args.early_stopping_threshold,
        'seed':
            args.seed,
    }
    # Add additional training arguments
    for key, value in kwargs.items():
      if key not in arguments_dict and not hasattr(training_args, key):
        arguments_dict.update({key: value})

    model_bin_file_path = os.path.join(current_output_dir, 'best-checkpoint',
                                       MODEL_BIN_FILE)
    if os.path.exists(model_bin_file_path):
      logger.info(
          'Found existing model checkpoint at %s. Skipping self-training: iteration: %d, stage: 1.',
          model_bin_file_path, iteration)
    else:
      logger.info('***** Running self-training: iteration: %d, stage: 1 *****',
                  iteration)
      finetune(**arguments_dict)
      accelerator.wait_for_everyone()
      assert os.path.exists(model_bin_file_path)
      logger.info('Self-training job completed: iteration: %d, stage: 1.',
                  iteration)

    if iteration > 0 and args.finetune_on_labeled_data:
      # Stage 2 (optional): fine-tuning on the original labeled data
      model_path = os.path.join(current_output_dir, 'best-checkpoint')
      current_output_dir = os.path.join(current_data_dir, 'stage-2')
      # Update arguments_dict
      arguments_dict['model_name_or_path'] = model_path
      arguments_dict['train_file'] = data_files['train']
      arguments_dict['output_dir'] = current_output_dir

      model_bin_file_path = os.path.join(current_output_dir, 'best-checkpoint',
                                         MODEL_BIN_FILE)
      if os.path.exists(model_bin_file_path):
        logger.info(
            'Found existing model checkpoint at %s. Skipping self-training: iteration: %d, stage: 2.',
            model_bin_file_path, iteration)
      else:
        logger.info(
            '***** Running self-training: iteration: %d, stage: 2 *****',
            iteration)
        finetune(**arguments_dict)
        accelerator.wait_for_everyone()
        assert os.path.exists(model_bin_file_path)
        logger.info('Self-training job completed: iteration: %d, stage: 2.',
                    iteration)

    new_iteration = iteration
    next_data_dir = data_dir_format(iteration + 1)

    config = AutoConfig.from_pretrained(
        os.path.join(current_output_dir, 'best-checkpoint'))
    id2label = config.id2label
    eval_results_file = os.path.join(current_output_dir,
                                     'eval_results_best-checkpoint.json')
    test_results_file = os.path.join(current_output_dir,
                                     'test_results_best-checkpoint.json')
    assert os.path.exists(eval_results_file)

    with open(eval_results_file, 'r') as f:
      eval_result = float(json.load(f)[args.eval_metric])
    infer_output_file = os.path.join(current_output_dir,
                                     'infer_output_best-checkpoint.csv')
    assert os.path.exists(infer_output_file)
    # Loading the dataset from local csv or json files.
    infer_input = load_dataset(
        args.data_file_extension,
        data_files={'data': data_files['infer']})['data']
    infer_output = load_dataset(
        'csv', data_files={'data': infer_output_file})['data']

    if accelerator.is_main_process:
      os.makedirs(next_data_dir, exist_ok=True)
      shutil.copy(
          eval_results_file,
          os.path.join(output_dir, f'eval_results_iter-{iteration}.json'))
      if os.path.exists(test_results_file):
        shutil.copy(
            eval_results_file,
            os.path.join(output_dir, f'test_results_iter-{iteration}.json'))
      create_pseudo_labeled_data(args, infer_input, infer_output, eval_result,
                                 id2label, next_data_dir)
    accelerator.wait_for_everyone()

    data_files['train_pseudo'] = os.path.join(
        next_data_dir, f'train_pseudo.{args.data_file_extension}')

    if args.evaluation_strategy != IntervalStrategy.NO.value:
      new_eval_result = eval_result

      if best_iteration is None:
        best_iteration = new_iteration
        best_eval_result = new_eval_result
      else:
        if new_eval_result - best_eval_result > args.early_stopping_threshold:
          best_iteration = new_iteration
          best_eval_result = new_eval_result
          early_stopping_patience_counter = 0
        else:
          if new_eval_result == best_eval_result:
            best_iteration = new_iteration
            best_eval_result = new_eval_result
          early_stopping_patience_counter += 1

        if early_stopping_patience_counter >= args.early_stopping_patience:
          should_training_stop = True

    progress_bar.update(1)

    if should_training_stop:
      break

  if best_iteration is not None:
    # Save the best iteration
    logger.info('Best iteration: %d', best_iteration)
    logger.info('Best evaluation result: %s = %f', args.eval_metric,
                best_eval_result)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
      shutil.copy(
          os.path.join(output_dir, f'eval_results_iter-{iteration}.json'),
          os.path.join(output_dir, 'eval_results_best-iteration.json'))
  else:
    # Assume that the last iteration is the best
    logger.info('Best iteration: %d', args.max_selftrain_iterations - 1)
    logger.info('Best evaluation result: %s = %f', args.eval_metric,
                eval_result)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
      shutil.copy(
          os.path.join(
              output_dir,
              f'eval_results_iter-{args.max_selftrain_iterations - 1}.json'),
          os.path.join(output_dir, 'eval_results_best-iteration.json'))
