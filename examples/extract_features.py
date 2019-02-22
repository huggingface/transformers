# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
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
"""Extract pre-computed feature vectors from a PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import re

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
import numpy

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
import h5py

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids,orig_to_tok_maps,orig_tokens):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.orig_to_tok_maps=orig_to_tok_maps
        self.orig_tokens=orig_tokens

def tokenize_map(orig_tokens,tokenizer):
    ### Input
    labels = ["NNP", "NNP", "POS", "NN"]

    ### Output
    bert_tokens = []

    # Token map will be an int -> int mapping between the `orig_tokens` index and
    # the `bert_tokens` index.
    orig_to_tok_map = []

    for orig_token in orig_tokens:
        orig_to_tok_map.append(len(bert_tokens)+1)
        bert_tokens.extend(tokenizer.tokenize(orig_token))
    return bert_tokens,orig_to_tok_map

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        orig_tokens=example.text_a.split()
        tokens_a,orig_to_tok_map_a = tokenize_map(orig_tokens,tokenizer)
        tokens_b = None
        if example.text_b:
            tokens_b,orig_to_tok_map_b = tokenize_map(example.text_b.split(),tokenizer)
            orig_tokens+=example.text_b.split()
            orig_to_tok_map_a+=orig_to_tok_map_b
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                print ('exceed length:',tokens_a)
                continue #skip when the length exceeds
                # tokens_a = tokens_a[0:(seq_length - 2)]

        # orig_to_tok_maps.append(orig_to_tok_map_a)
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
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
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        while len(orig_to_tok_map_a) < seq_length:
            orig_to_tok_map_a.append(0)

        assert len(orig_to_tok_map_a) == seq_length
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                orig_to_tok_maps=orig_to_tok_map_a,
                orig_tokens=orig_tokens))
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


def read_examples(input_file,example_batch):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip().split('\t')[0]
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
            if len(examples)>=example_batch:
                yield examples
                examples=[]
    if examples!=[]:
        yield examples

def get_orig_seq(input_mask_batch):
    seq=[i for i in input_mask_batch if i!=0]
    return seq
def feature_orig_to_tok_map(average_layer_batch, orig_to_token_map_batch,input_mask_batch):
    average_layer_batch_out=[]
    for sent_i, sent_embed in enumerate(average_layer_batch):
        sent_embed_out=[]
        orig_to_token_map_batch_sent=get_orig_seq(orig_to_token_map_batch[sent_i])
        seq_len = len(get_orig_seq(input_mask_batch[sent_i]))

        for i in range(len(orig_to_token_map_batch_sent)):
            start = orig_to_token_map_batch_sent[i]
            if i==(len(orig_to_token_map_batch_sent)-1):
                sent_embed_out.append(sum(sent_embed[start:seq_len-1]) / (seq_len-1 - start))
                continue
            end = orig_to_token_map_batch_sent[i + 1]
            sent_embed_out.append(sum(sent_embed[start:end])/(end-start))
        average_layer_batch_out.append(numpy.array(sent_embed_out))
    return numpy.array(average_layer_batch_out)

def examples2embeds(examples,tokenizer,model,device,writer,args):
    features = convert_examples_to_features(
        examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    


    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    all_input_orig_to_token_maps = torch.tensor([f.orig_to_tok_maps for f in features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index, all_input_orig_to_token_maps)
    if args.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data)
    else:
        eval_sampler = DistributedSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    model.eval()
    batch_counter = 0
    # sent_set = set()
    # with h5py.File(args.output_file, 'w') as writer:
    for input_ids, input_mask, example_indices, input_orig_to_token_maps in eval_dataloader:
        print('batch no. {0}'.format(batch_counter))
        batch_counter += 1
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)

        all_encoder_layers, _ = model(input_ids, token_type_ids=None, attention_mask=input_mask)
        all_encoder_layers = all_encoder_layers

        average_layer_batch = sum(all_encoder_layers[-4:]) / 4
        # if orig_to_token_map_batch!=None:
        average_layer_batch = feature_orig_to_tok_map(average_layer_batch.cpu().detach().numpy(),
                                                      input_orig_to_token_maps.cpu().detach().numpy(), input_mask)

        for b, example_index in enumerate(example_indices):

            feature = features[example_index.item()]
            sent = '\t'.join(feature.orig_tokens)
            sent = sent.replace('.', '$period$')
            sent = sent.replace('/', '$backslash$')
            # if sent in sent_set:
            #     continue
            # sent_set.add(sent)
            if sent not in writer:
                payload = average_layer_batch[b]
                writer.create_dataset(sent, payload.shape, dtype='float32', compression="gzip", compression_opts=9,
                                      data=payload)

        #     # feature = unique_id_to_feature[unique_id]
        #     output_json = collections.OrderedDict()
        #     output_json["linex_index"] = unique_id
        #     all_out_features = []
        #     for (i, token) in enumerate(feature.tokens):
        #         all_layers = []
        #         for (j, layer_index) in enumerate(layer_indexes):
        #             layer_output = all_encoder_layers[int(layer_index)].detach().cpu().numpy()
        #             layer_output = layer_output[b]
        #             layers = collections.OrderedDict()
        #             layers["index"] = layer_index
        #             layers["values"] = [
        #                 round(x.item(), 6) for x in layer_output[i]
        #             ]
        #             all_layers.append(layers)
        #         out_features = collections.OrderedDict()
        #         out_features["token"] = token
        #         out_features["layers"] = all_layers
        #         all_out_features.append(out_features)
        #     output_json["features"] = all_out_features
        #     writer.write(json.dumps(output_json) + "\n")
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    # parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument('--example_batch', default=100000,type=int, help='batch size for input examples')
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help = "local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    args = parser.parse_args()

    writer= h5py.File(args.output_file, 'w')

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    # layer_indexes = [int(x) for x in args.layers.split(",")]

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = BertModel.from_pretrained(args.bert_model)
    model.to(device)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
       model = torch.nn.DataParallel(model)
        
    example_counter=0
    for examples in read_examples(args.input_file,args.example_batch):
        example_counter+=1
        print ('processed {0} examples'.format (str(args.example_batch*example_counter)))
        examples2embeds(examples,tokenizer,model,device,writer,args)
    writer.close()

if __name__ == "__main__":
    main()
