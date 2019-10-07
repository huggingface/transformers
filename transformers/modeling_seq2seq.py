# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Conditional generation class. """


class Seq2SeqModel(object):
    def __init__(self):
        raise EnvironmentError(
            """Seq2Seq is designed to be instantiated using the
        `Seq2Seq.from_pretrained(encoder_name_or_path, decoder_name_or_path)` method."""
        )

    @classmethod
    def from_pretrained(cls, encoder_name, decoder_name):
        # Here we should call AutoModel to initialize the models depending
        # on the pretrained models taken as an input.
        # For a first iteration we only work with Bert.
        raise NotImplementedError

    def __call__(self):
        # allows to call an instance of the class
        # model = Seq2Seq(encode='bert', decoder='bert')
        raise NotImplementedError

    def process(self):
        # alternative API to __call__ it is more explicit.
        raise NotImplementedError
