# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

from transformers.testing_utils import CaptureStdout
from transformers.agents.python_interpreter import evaluate_python_code


# Fake function we will use as tool
def add_two(x):
    return x + 2


class PythonInterpreterTester(unittest.TestCase):
    def test_evaluate_assign(self):
        code = "x = 3"
        state = {}
        result = evaluate_python_code(code, {}, state=state)
        assert result == 3
        self.assertDictEqual(state, {"x": 3, 'print_outputs': ''})

        code = "x = y"
        state = {"y": 5}
        result = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == 5
        self.assertDictEqual(state, {"x": 5, "y": 5, 'print_outputs': ''})

    def test_evaluate_call(self):
        code = "y = add_two(x)"
        state = {"x": 3}
        result = evaluate_python_code(code, {"add_two": add_two}, state=state)
        assert result == 5
        self.assertDictEqual(state, {"x": 3, "y": 5, 'print_outputs': ''})

        # Won't work without the tool
        with CaptureStdout() as out:
            result = evaluate_python_code(code, {}, state=state)
        assert result is None
        assert "tried to execute add_two" in out.out

    def test_evaluate_constant(self):
        code = "x = 3"
        state = {}
        result = evaluate_python_code(code, {}, state=state)
        assert result == 3
        self.assertDictEqual(state, {"x": 3, 'print_outputs': ''})

    def test_evaluate_dict(self):
        code = "test_dict = {'x': x, 'y': add_two(x)}"
        state = {"x": 3}
        result = evaluate_python_code(code, {"add_two": add_two}, state=state)
        self.assertDictEqual(result, {"x": 3, "y": 5})
        self.assertDictEqual(state, {"x": 3, "test_dict": {"x": 3, "y": 5}, 'print_outputs': ''})

    def test_evaluate_expression(self):
        code = "x = 3\ny = 5"
        state = {}
        result = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == 5
        self.assertDictEqual(state, {"x": 3, "y": 5, 'print_outputs': ''})

    def test_evaluate_f_string(self):
        code = "text = f'This is x: {x}.'"
        state = {"x": 3}
        result = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == "This is x: 3."
        self.assertDictEqual(state, {"x": 3, "text": "This is x: 3.", 'print_outputs': ''})

    def test_evaluate_if(self):
        code = "if x <= 3:\n    y = 2\nelse:\n    y = 5"
        state = {"x": 3}
        result = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == 2
        self.assertDictEqual(state, {"x": 3, "y": 2, 'print_outputs': ''})

        state = {"x": 8}
        result = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == 5
        self.assertDictEqual(state, {"x": 8, "y": 5, 'print_outputs': ''})

    def test_evaluate_list(self):
        code = "test_list = [x, add_two(x)]"
        state = {"x": 3}
        result = evaluate_python_code(code, {"add_two": add_two}, state=state)
        self.assertListEqual(result, [3, 5])
        self.assertDictEqual(state, {"x": 3, "test_list": [3, 5], 'print_outputs': ''})

    def test_evaluate_name(self):
        code = "y = x"
        state = {"x": 3}
        result = evaluate_python_code(code, {}, state=state)
        assert result == 3
        self.assertDictEqual(state, {"x": 3, "y": 3, 'print_outputs': ''})

    def test_evaluate_subscript(self):
        code = "test_list = [x, add_two(x)]\ntest_list[1]"
        state = {"x": 3}
        result = evaluate_python_code(code, {"add_two": add_two}, state=state)
        assert result == 5
        self.assertDictEqual(state, {"x": 3, "test_list": [3, 5], 'print_outputs': ''})

        code = "test_dict = {'x': x, 'y': add_two(x)}\ntest_dict['y']"
        state = {"x": 3}
        result = evaluate_python_code(code, {"add_two": add_two}, state=state)
        assert result == 5
        self.assertDictEqual(state, {"x": 3, "test_dict": {"x": 3, "y": 5}, 'print_outputs': ''})

    def test_evaluate_for(self):
        code = "x = 0\nfor i in range(3):\n    x = i"
        state = {}
        result = evaluate_python_code(code, {"range": range}, state=state)
        assert result == 2
        self.assertDictEqual(state, {"x": 2, "i": 2, 'print_outputs': ''})

    def test_evaluate_binop(self):
        code = "y + x"
        state = {"x": 3, "y": 6}
        result = evaluate_python_code(code, {}, state=state)
        assert result == 9
        self.assertDictEqual(state, {"x": 3, "y": 6, 'print_outputs': ''})

    def test_evaluate_recursive_function(self):
        code = """
def recur_fibo(n):
    if n <= 1:
        return n
    else:
        return(recur_fibo(n-1) + recur_fibo(n-2))
recur_fibo(6)"""
        result = evaluate_python_code(code, {}, state={})
        assert result == 8


    def test_evaluate_string_methods(self):
        code = "string = 'hello'\nstring = string.replace('h', 'o')\nstring.split('e')"
        result = evaluate_python_code(code, {}, state={})
        assert result == ["o", "llo"]

    def test_evaluate_slicing(self):
        code = "'hello'[1:3][::-1]"
        result = evaluate_python_code(code, {}, state={})
        assert result == "le"
