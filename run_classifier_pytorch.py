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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import csv
# import os
# import modeling_pytorch
# import optimization
# import tokenization

import argparse

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--data_dir",
                    default = None,
                    type = str,
                    required = True,
                    help = "The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_config_file",
                    default = None,
                    type = str,
                    required = True,
                    help = "The config json file corresponding to the pre-trained BERT model. \n"
                        "This specifies the model architecture.")
parser.add_argument("--task_name",
                    default = None,
                    type = str,
                    required = True,
                    help = "The name of the task to train.")
parser.add_argument("--vocab_file",
                    default = None,
                    type = str,
                    required = True,
                    help = "The vocabulary file that the BERT model was trained on.")                    
parser.add_argument("--output_dir",
                    default = None,
                    type = str,
                    required = True,
                    help = "The output directory where the model checkpoints will be written.")                   

## Other parameters
parser.add_argument("--init_checkpoint",
                    default = None,
                    type = str,
                    help = "Initial checkpoint (usually from a pre-trained BERT model).")
parser.add_argument("--do_lower_case",
                    default = True,
                    type = bool,
                    help = "Whether to lower case the input text. Should be True for uncased models and False for cased models.")
parser.add_argument("--max_seq_length",
                    default = 128,
                    type = int,
                    help = "The maximum total input sequence length after WordPiece tokenization. \n"
                        "Sequences longer than this will be truncated, and sequences shorter \n"
                        "than this will be padded.")
parser.add_argument("--do_train",
                    default = False,
                    type = bool,
                    help = "Whether to run training.")
parser.add_argument("--do_eval",
                    default = False,
                    type = bool,
                    help = "Whether to run eval on the dev set.")                                             
parser.add_argument("--train_batch_size",
                    default = 32,
                    type = int,
                    help = "Total batch size for training.")
parser.add_argument("--eval_batch_size",
                    default = 8,
                    type = int,
                    help = "Total batch size for eval.")
parser.add_argument("--learning_rate",
                    default = 5e-5,
                    type = float,
                    help = "The initial learning rate for Adam.")                                       
parser.add_argument("--num_train_epochs",
                    default = 3.0,
                    type = float,
                    help = "Total number of training epochs to perform.")                    
parser.add_argument("--warmup_proportion",
                    default = 0.1,
                    type = float,
                    help = "Proportion of training to perform linear learning rate warmup for. "
                        "E.g., 0.1 = 10%% of training.")
parser.add_argument("--save_checkpoints_steps",
                    default = 1000,
                    type = int,
                    help = "How often to save the model checkpoint.")                    
parser.add_argument("--iterations_per_loop",
                    default = 1000,
                    type = int,
                    help = "How many steps to make in each estimator call.")
                    
### BEGIN - TO DELETE EVENTUALLY --> NO SENSE IN PYTORCH ###                   
parser.add_argument("--use_tpu",
                    default = False,
                    type = bool,
                    help = "Whether to use TPU or GPU/CPU.") 
parser.add_argument("--tpu_name",
                    default = None,
                    type = str,
                    help = "The Cloud TPU to use for training. This should be either the name "
                        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
                        "url.")
parser.add_argument("--tpu_zone",
                    default = None,
                    type = str,
                    help = "[Optional] GCE zone where the Cloud TPU is located in. If not "
                        "specified, we will attempt to automatically detect the GCE project from "
                        "metadata.")                    
parser.add_argument("--gcp_project",
                    default = None,
                    type = str,
                    help = "[Optional] Project name for the Cloud TPU-enabled project. If not "
                        "specified, we will attempt to automatically detect the GCE project from "
                        "metadata.")                     
parser.add_argument("--master",
                    default = None,
                    type = str,
                    help = "[Optional] TensorFlow master URL.")                                                    
parser.add_argument("--num_tpu_cores",
                    default = 8,
                    type = int,
                    help = "Only used if `use_tpu` is True. Total number of TPU cores to use.")
### END - TO DELETE EVENTUALLY --> NO SENSE IN PYTORCH ### 
  
args = parser.parse_args()