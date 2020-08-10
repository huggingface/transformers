# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
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


import os
import shutil
import tempfile
import unittest
from typing import Any, Text

from transformers.file_io import json_pickle_dump, json_pickle_load


class Baz:
    """"A type for testing File IO utils."""

    def __init__(self, a: Text, b: int, c: bool, d: Any) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __eq__(self, obj: "Baz") -> bool:
        return self.a == obj.a and self.b == obj.b and self.c == obj.c and self.d == obj.d


class FileIOTest(unittest.TestCase):
    objects_to_test = [
        123,
        42.0,
        "test_string",
        ["1", 2, True, {"obj": None}],
        None,
        Baz(a="z", b=9, c=False, d=None),
        {
            "string": "plain string",
            "integer": 123,
            "float": 1e-10,
            "boolean": True,
            "array": [1, 2, 3],
            "array_extended": [1, "a", True, 1e-10],
            "object": Baz(a="z", b=9, c=False, d=None),
        },
    ]

    def test_dump_and_load_json_pickle(self) -> None:
        for object_to_test in self.objects_to_test:
            tmp_dirname = tempfile.mkdtemp()
            tmp_filename = os.path.join(tmp_dirname, "saved_object.json")

            json_pickle_dump(tmp_filename, object_to_test)
            loaded_object = json_pickle_load(tmp_filename)

            # test the dictionary object save/load
            if isinstance(object_to_test, dict):
                self.assertIsInstance(loaded_object, dict)
                self.assertEqual(len(object_to_test), len(loaded_object))

                values = zip(object_to_test.values(), loaded_object.values())
                for left, right in values:
                    self.assertEqual(left, right, f"Values {left} and {right} are not equal.")

            # test the list object save/load
            elif isinstance(object_to_test, list):
                self.assertIsInstance(loaded_object, list)
                self.assertEqual(len(loaded_object), len(object_to_test))

                for left, right in zip(object_to_test, loaded_object):
                    self.assertEqual(left, right, f"Values {left} and {right} are not equal.")

            # test other objects save/load
            else:
                failure_msg = f"Values {loaded_object} and {object_to_test} are not equal."
                self.assertEqual(loaded_object, object_to_test, failure_msg)

            shutil.rmtree(tmp_dirname)
