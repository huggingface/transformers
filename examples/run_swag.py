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

if __name__ == "__main__":
    e = SwagExample(
        3416,
        'Members of the procession walk down the street holding small horn brass instruments.',
        'A drum line',
        'passes by walking down the street playing their instruments.',
        'has heard approaching them.',
        "arrives and they're outside dancing and asleep.",
        'turns the lead singer watches the performance.',
    )
    print(e)

    e = SwagExample(
        3416,
        'Members of the procession walk down the street holding small horn brass instruments.',
        'A drum line',
        'passes by walking down the street playing their instruments.',
        'has heard approaching them.',
        "arrives and they're outside dancing and asleep.",
        'turns the lead singer watches the performance.',
        0
    )
    print(e)
