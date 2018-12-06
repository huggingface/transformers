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
"""BERT finetuning runner."""

import pandas as pd


class SwagExample(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 swag_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label = None):
        self.swag_id = swag_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.ending_0 = ending_0
        self.ending_1 = ending_1
        self.ending_2 = ending_2
        self.ending_3 = ending_3
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f'swag_id: {self.swag_id}',
            f'context_sentence: {self.context_sentence}',
            f'start_ending: {self.start_ending}',
            f'ending_0: {self.ending_0}',
            f'ending_1: {self.ending_1}',
            f'ending_2: {self.ending_2}',
            f'ending_3: {self.ending_3}',
        ]

        if self.label is not None:
            l.append(f'label: {self.label}')

        return ', '.join(l)

def read_swag_examples(input_file, is_training):
    input_df = pd.read_csv(input_file)

    if is_training and 'label' not in input_df.columns:
        raise ValueError(
            "For training, the input file must contain a label column.")

    examples = [
        SwagExample(
            swag_id = row['fold-ind'],
            context_sentence = row['sent1'],
            start_ending = row['sent2'],
            ending_0 = row['ending0'],
            ending_1 = row['ending1'],
            ending_2 = row['ending2'],
            ending_3 = row['ending3'],
            label = row['label'] if is_training else None
        ) for _, row in input_df.iterrows()
    ]

    return examples


if __name__ == "__main__":
    examples = read_swag_examples('data/train.csv', True)
    print(len(examples))
    for example in examples[:5]:
        print('###########################')
        print(example)
