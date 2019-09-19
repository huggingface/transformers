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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import logging
import os
import re
import torch
from torch.utils.data import (TensorDataset)
from sklearn.metrics import  f1_score

logger = logging.getLogger(__name__)


class RBertUtils:
    """
    Utility class to prepare text for RBERT


    Args:
    same as BertTokenizer, apart from
    replace_conflicting_entity_offset_char: since the forward method of RBert classification models detects
                                            the special characters that bound each entity, any pre-existing characters
                                            of this type will cause a conflict. To prevent this, we replace any affected
                                            characters with this, prior to encoding. Defaults to '|'
    ent1_sep_token: the char to delimit entity 1. Default is '$'
    ent2_sep_token: the char to delimit entity 2. Default is '#'


    """

    def __init__(self, replace_conflicting_entity_offset_char='|',
                 ent1_sep_token="$", ent2_sep_token="#", ):
        self.ent1_sep_token = ent1_sep_token
        self.ent2_sep_token = ent2_sep_token
        self.replace_conflicting_entity_offset_char = replace_conflicting_entity_offset_char

    def _clean_special_chars(self, text: str):
        return text.replace(self.ent1_sep_token,
                            self.replace_conflicting_entity_offset_char).replace(self.ent2_sep_token,
                                                                                 self.replace_conflicting_entity_offset_char)

    def _insert_relationship_special_chars(self, text, e1_offset_tup, e2_offset_tup):
        reverse_ordered_tups = sorted([e1_offset_tup, e2_offset_tup], reverse=True, key=lambda x: x[1])
        last_token_to_process = True
        for start, end in reverse_ordered_tups:
            before = text[:start]
            during = text[start:end]
            after = text[end:]
            if last_token_to_process:
                text = before + " " + self.ent2_sep_token + " " + during + " " + self.ent2_sep_token + " " + after
            else:
                text = before + " " + self.ent1_sep_token + " " + during + " " + self.ent1_sep_token + " " + after
            last_token_to_process = False
        return text

    def insert_special_chars_into_text(self, text, e1_offset_tup, e2_offset_tup):
        text = self._clean_special_chars(text)
        text = self._insert_relationship_special_chars(text, e1_offset_tup, e2_offset_tup)
        return text

class InputExample(object):
    """A single SemEval 2010 Task 8 example"""

    def __init__(self, id: int, text: str, label: str, comment: str):
        self.comment = comment
        self.label = label
        self.text = text
        self.id = id


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, instance_id):
        self.instance_id = instance_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id



class SemEval2010Task8DataProcessor():
    """Processor for the SemEval 2010 Task 8 Dataset. Note, it's not clear from the RBert paper how many classes the
    model is trained on. Clearly this has a big impact on the result.


    """
    def __init__(self,include_directionality=True):
        self.include_directionality = include_directionality
        self.e1_e2_labels = [
                         'Product-Producer(e1,e2)',
                         'Entity-Origin(e1,e2)',
                         'Entity-Destination(e1,e2)',
                         'Message-Topic(e1,e2)',
                         'Component-Whole(e1,e2)',
                         'Content-Container(e1,e2)',
                         'Instrument-Agency(e1,e2)',
                         'Cause-Effect(e1,e2)',
                         'Member-Collection(e1,e2)'
                       ]
        self.e2_e1_labels = [
                         'Product-Producer(e2,e1)',
                         'Entity-Origin(e2,e1)',
                         'Entity-Destination(e2,e1)',
                         'Message-Topic(e2,e1)',
                         'Component-Whole(e2,e1)',
                         'Content-Container(e2,e1)',
                         'Instrument-Agency(e2,e1)',
                         'Cause-Effect(e2,e1)',
                         'Member-Collection(e2,e1)'
                       ]
        self.undirected_labels_mapping = dict(zip(self.e2_e1_labels,self.e1_e2_labels))
        self.other_label = ['Other']
        self.undirected_labels_mapping[self.other_label[0]] = self.other_label[0]

        if self.include_directionality:
            self.all_labels = self.e1_e2_labels + self.e2_e1_labels + self.other_label
        else:
            self.all_labels = self.e1_e2_labels  + self.other_label

    def _instance_generator(self, data,include_other):
        for i in range(0, len(data), 4):
            id, text = data[i].split('\t')
            text = text.strip()
            label = str(data[i + 1]).strip()
            # map all labels to single direction if not using directionality
            if not self.include_directionality and label in self.undirected_labels_mapping:
                label = self.undirected_labels_mapping[label]
            # Don't include Other label in training set
            if label == 'Other' and not include_other:
                logger.info(f'Skipping Other labeled instance at {id}')
            else:
                comment = data[i + 2].strip()
                yield InputExample(id=id, text=text, label=label, comment=comment)


    def _strip_direction(self,label):
        return re.sub(r'\(.*', "", label)

    def _create_examples(self, path,include_other):
        with open(path, 'r') as f:
            data = f.readlines()
        return self._instance_generator(data,include_other)

    def get_train_examples(self, data_dir,include_other):
        return self._create_examples(os.path.join(data_dir, "SemEval2010_task8_training","TRAIN_FILE.TXT"),
                                     include_other)

    def get_dev_examples(self, data_dir,include_other):
        return self._create_examples(os.path.join(data_dir, "SemEval2010_task8_testing_keys","TEST_FILE_FULL.TXT"),
                                     include_other)

    def write_dev_examples_to_official_format(self,data_dir,examples):
        with open(os.path.join(data_dir,"TEST_FILE_SEMEVAL_SCRIPT_FORMAT.tsv"),'w') as f:
            for example in examples:
                f.write(str(example.id)+'\t'+example.label+'\n')


    def get_labels(self):
        return self.all_labels



# def find_entity_indices(id_list, tokenizer):
#     ent1_bounding_id_list = [i for i, e in enumerate(id_list) if e == tokenizer.entity_1_token_id]
#     ent1_bounding_id_list = [ent1_bounding_id_list[0], ent1_bounding_id_list[1] + 1]
#     ent2_bounding_id_list = [i for i, e in enumerate(id_list) if e == tokenizer.entity_2_token_id]
#     ent2_bounding_id_list = [ent2_bounding_id_list[0], ent2_bounding_id_list[1] + 1]
#     return ent1_bounding_id_list, ent2_bounding_id_list

def mod_text(text, hit):
    before = text[:hit.start()]
    during = hit.group(2)
    after = text[hit.end():]

    start = len(before)
    end = len(before) + len(during)
    return (before + during + after, [start, end],)


def find_ents_and_modify_string(text):
    e1_hits = re.search("(<e1>)(.*)(</e1>)", text)
    new_text, e1_offsets = mod_text(text, e1_hits)
    e2_hits = re.search("(<e2>)(.*)(</e2>)", new_text)
    new_text, e2_offsets = mod_text(new_text, e2_hits)

    return new_text, e1_offsets, e2_offsets


def get_entity_seperator_token_ids(rbert_utils, tokenizer):
    return tokenizer.encode(rbert_utils.ent1_sep_token)[0], tokenizer.encode(rbert_utils.ent2_sep_token)[0]

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, rbert_utils,pad_token=0,
                                 mask_padding_with_zero=True):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d" % (ex_index))


        text_with_tags_removed, e1_offsets, e2_offsets = find_ents_and_modify_string(example.text)
        new_text = rbert_utils.insert_special_chars_into_text(text_with_tags_removed,e1_offsets,e2_offsets)
        input_ids = tokenizer.encode(new_text,text_pair=None, add_special_tokens=True)
        entity_1_token_id,entity_2_token_id = get_entity_seperator_token_ids(rbert_utils, tokenizer)


        #check that the special tokens have been encoded in the right order
        special_tokens_ordered = list(filter(lambda n: n in [entity_1_token_id, entity_2_token_id], input_ids))
        assert special_tokens_ordered[0] == entity_1_token_id
        assert special_tokens_ordered[1] == entity_1_token_id
        assert special_tokens_ordered[2] == entity_2_token_id
        assert special_tokens_ordered[3] == entity_2_token_id

        # Account for [CLS] and [SEP] with "- 2"
        if len(input_ids) > max_seq_length - 2:
            input_ids = input_ids[:(max_seq_length - 2)]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        label_id = label_map[example.label]
        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=[0] * len(input_ids),
                          label_id=label_id,
                          instance_id=int(example.id)
                          ))
    return features


def convert_features_to_dataset(features, output_mode='classification'):
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    else:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    instance_ids = torch.tensor([f.instance_id for f in features], dtype=torch.int)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,instance_ids)
    return dataset

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds,average='macro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }
def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "semeval2010_task8":
        return acc_and_f1(labels, preds)
    else:
        raise KeyError(task_name)


processors = {
    "semeval2010_task8": SemEval2010Task8DataProcessor
}

output_modes = {
    "semeval2010_task8": "classification"
}
