# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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


import unittest

from transformers import DonutProcessor


DONUT_PRETRAINED_MODEL_NAME = "naver-clova-ix/donut-base"


class DonutProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = DonutProcessor.from_pretrained(DONUT_PRETRAINED_MODEL_NAME)

    def test_token2json(self):
        expected_json = {
            "name": "John Doe",
            "age": "99",
            "city": "Atlanta",
            "state": "GA",
            "zip": "30301",
            "phone": "123-4567",
            "nicknames": [{"nickname": "Johnny"}, {"nickname": "JD"}],
            "cars": [{"make": "BMW", "model": "X5"}],  # list of length 1 containing a dict
            "favorite_car": {"make": "Porsche", "model": "911"},  # similar to the above but dict instead of list
            "hobbies": ["running"],  # list of length 1 containing only string
            # some deeper nesting with keys also appearing at other levels of nesting
            "friends": [{"name": "William", "nicknames": [{"nickname": "Will"}]}],
        }

        sequence = (
            "<s_name-str>John Doe</s_name-str><s_age-str>99</s_age-str><s_city-str>Atlanta</s_city-str>"
            "<s_state-str>GA</s_state-str><s_zip-str>30301</s_zip-str><s_phone-str>123-4567</s_phone-str>"
            "<s_nicknames-list><s_nickname-str>Johnny</s_nickname-str>"
            "<sep/><s_nickname-str>JD</s_nickname-str></s_nicknames-list>"
            "<s_cars-list><s_make-str>BMW</s_make-str><s_model-str>X5</s_model-str></s_cars-list>"
            "<s_favorite_car-dict><s_make-str>Porsche</s_make-str><s_model-str>911</s_model-str></s_favorite_car-dict>"
            "<s_hobbies-list>running</s_hobbies-list>"
            "<s_friends-list><s_name-str>William</s_name-str>"
            "<s_nicknames-list><s_nickname-str>Will</s_nickname-str></s_nicknames-list></s_friends-list>"
        )
        actual_json = self.processor.token2json(sequence)

        self.assertDictEqual(actual_json, expected_json)
