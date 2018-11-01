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

import csv
import os
# import modeling_pytorch
# import optimization
import tokenization_pytorch
import torch

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

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
 
parser.add_argument("--use_gpu",
                    default = True,
                    type = bool,
                    help = "Whether to use GPU")                    
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

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
            

class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization_pytorch.convert_to_unicode(line[0]))
            text_a = tokenization_pytorch.convert_to_unicode(line[8])
            text_b = tokenization_pytorch.convert_to_unicode(line[9])
            label = tokenization_pytorch.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
        

class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization_pytorch.convert_to_unicode(line[3])
            label = tokenization_pytorch.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
        
        
def convert_examples_to_features(examples, label_list, max_seq_length,
                                                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [tokenization_pytorch.printable_text(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(
                        input_ids=input_ids,
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
            

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                                 labels, num_labels, use_one_hot_embeddings):
    raise NotImplementedError()


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                                         num_train_steps, num_warmup_steps,
                                         use_one_hot_embeddings):
    raise NotImplementedError()
    ### ATTENTION - I removed the `use_tpu` argument
    

def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator.""" ### ATTENTION - To rewrite ###

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)
        
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        d = {"input_ids":
                        torch.IntTensor(all_input_ids, device = device), #Requires_grad=False by default
            "input_mask":
                        torch.IntTensor(all_input_mask, device = device),
            "segment_ids":
                        torch.IntTensor(all_segment_ids, device = device),
            "label_ids":
                        torch.IntTensor(all_label_ids, device = device)
            }
                        
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def main(_):
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
    }
    
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
        
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)
    
    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
                    raise ConfigurationError(f"Output directory ({args.output_dir}) already exists and is "
                                     f"not empty.")
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()
    
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
        
    # tpu_cluster_resolver = None
    # if FLAGS.use_tpu and FLAGS.tpu_name:
    #     tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    #         FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    # is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    # run_config = tf.contrib.tpu.RunConfig(
    #     cluster=tpu_cluster_resolver,
    #     master=FLAGS.master,
    #     model_dir=FLAGS.output_dir,
    #     save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    #     tpu_config=tf.contrib.tpu.TPUConfig(
    #         iterations_per_loop=FLAGS.iterations_per_loop,
    #         num_shards=FLAGS.num_tpu_cores,
    #         per_host_input_for_training=is_per_host))
    
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size * args.num_train_epochs)
        num_warmup_steps = int(num_train_steps * args.warmup_proportion)
    
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=args.init_checkpoint,
        learning_rate=args.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_gpu=args.use_gpu,
        use_one_hot_embeddings=args.use_gpu) ### TO DO - to check when model_fn is written)
    
    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU. - TO DO
    for batch in 
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=args.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size)
    
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        train_input_fn = input_fn_builder(
            features=train_features,
            seq_length=args.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    
    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", args.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if args.use_tpu:
            # Eval will be slightly WRONG on the TPU because it will truncate
            # the last batch.
            eval_steps = int(len(eval_examples) / args.eval_batch_size)

        eval_drop_remainder = True if args.use_tpu else False
        eval_input_fn = input_fn_builder(
            features=eval_features,
            seq_length=args.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    
if __name__ == "__main__":
    main()
    return None