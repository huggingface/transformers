# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import pytest

from transformers import load_tool
from transformers.agents.agent_types import AGENT_TYPE_MAPPING
from transformers.agents.default_tools import BASE_PYTHON_TOOLS
from transformers.agents.python_interpreter import InterpretorError, evaluate_python_code

from .test_tools_common import ToolTesterMixin


# Fake function we will use as tool
def add_two(x):
    return x + 2


class PythonInterpreterToolTester(unittest.TestCase, ToolTesterMixin):
    def setUp(self):
        self.tool = load_tool("python_interpreter")
        self.tool.setup()

    def test_exact_match_arg(self):
        result = self.tool("(2 / 2) * 4")
        self.assertEqual(result, "4.0")

    def test_exact_match_kwarg(self):
        result = self.tool(code="(2 / 2) * 4")
        self.assertEqual(result, "4.0")

    def test_agent_type_output(self):
        inputs = ["2 * 2"]
        output = self.tool(*inputs)
        output_type = AGENT_TYPE_MAPPING[self.tool.output_type]
        self.assertTrue(isinstance(output, output_type))

    def test_agent_types_inputs(self):
        inputs = ["2 * 2"]
        _inputs = []

        for _input, expected_input in zip(inputs, self.tool.inputs.values()):
            input_type = expected_input["type"]
            if isinstance(input_type, list):
                _inputs.append([AGENT_TYPE_MAPPING[_input_type](_input) for _input_type in input_type])
            else:
                _inputs.append(AGENT_TYPE_MAPPING[input_type](_input))

        # Should not raise an error
        output = self.tool(*inputs)
        output_type = AGENT_TYPE_MAPPING[self.tool.output_type]
        self.assertTrue(isinstance(output, output_type))


class PythonInterpreterTester(unittest.TestCase):
    def test_evaluate_assign(self):
        code = "x = 3"
        state = {}
        result = evaluate_python_code(code, {}, state=state)
        assert result == 3
        self.assertDictEqual(state, {"x": 3, "print_outputs": ""})

        code = "x = y"
        state = {"y": 5}
        result = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == 5
        self.assertDictEqual(state, {"x": 5, "y": 5, "print_outputs": ""})

    def test_evaluate_call(self):
        code = "y = add_two(x)"
        state = {"x": 3}
        result = evaluate_python_code(code, {"add_two": add_two}, state=state)
        assert result == 5
        self.assertDictEqual(state, {"x": 3, "y": 5, "print_outputs": ""})

        # Should not work without the tool
        with pytest.raises(InterpretorError) as e:
            evaluate_python_code(code, {}, state=state)
        assert "tried to execute add_two" in str(e.value)

    def test_evaluate_constant(self):
        code = "x = 3"
        state = {}
        result = evaluate_python_code(code, {}, state=state)
        assert result == 3
        self.assertDictEqual(state, {"x": 3, "print_outputs": ""})

    def test_evaluate_dict(self):
        code = "test_dict = {'x': x, 'y': add_two(x)}"
        state = {"x": 3}
        result = evaluate_python_code(code, {"add_two": add_two}, state=state)
        self.assertDictEqual(result, {"x": 3, "y": 5})
        self.assertDictEqual(state, {"x": 3, "test_dict": {"x": 3, "y": 5}, "print_outputs": ""})

    def test_evaluate_expression(self):
        code = "x = 3\ny = 5"
        state = {}
        result = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == 5
        self.assertDictEqual(state, {"x": 3, "y": 5, "print_outputs": ""})

    def test_evaluate_f_string(self):
        code = "text = f'This is x: {x}.'"
        state = {"x": 3}
        result = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == "This is x: 3."
        self.assertDictEqual(state, {"x": 3, "text": "This is x: 3.", "print_outputs": ""})

    def test_evaluate_if(self):
        code = "if x <= 3:\n    y = 2\nelse:\n    y = 5"
        state = {"x": 3}
        result = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == 2
        self.assertDictEqual(state, {"x": 3, "y": 2, "print_outputs": ""})

        state = {"x": 8}
        result = evaluate_python_code(code, {}, state=state)
        # evaluate returns the value of the last assignment.
        assert result == 5
        self.assertDictEqual(state, {"x": 8, "y": 5, "print_outputs": ""})

    def test_evaluate_list(self):
        code = "test_list = [x, add_two(x)]"
        state = {"x": 3}
        result = evaluate_python_code(code, {"add_two": add_two}, state=state)
        self.assertListEqual(result, [3, 5])
        self.assertDictEqual(state, {"x": 3, "test_list": [3, 5], "print_outputs": ""})

    def test_evaluate_name(self):
        code = "y = x"
        state = {"x": 3}
        result = evaluate_python_code(code, {}, state=state)
        assert result == 3
        self.assertDictEqual(state, {"x": 3, "y": 3, "print_outputs": ""})

    def test_evaluate_subscript(self):
        code = "test_list = [x, add_two(x)]\ntest_list[1]"
        state = {"x": 3}
        result = evaluate_python_code(code, {"add_two": add_two}, state=state)
        assert result == 5
        self.assertDictEqual(state, {"x": 3, "test_list": [3, 5], "print_outputs": ""})

        code = "test_dict = {'x': x, 'y': add_two(x)}\ntest_dict['y']"
        state = {"x": 3}
        result = evaluate_python_code(code, {"add_two": add_two}, state=state)
        assert result == 5
        self.assertDictEqual(state, {"x": 3, "test_dict": {"x": 3, "y": 5}, "print_outputs": ""})

    def test_evaluate_for(self):
        code = "x = 0\nfor i in range(3):\n    x = i"
        state = {}
        result = evaluate_python_code(code, {"range": range}, state=state)
        assert result == 2
        self.assertDictEqual(state, {"x": 2, "i": 2, "print_outputs": ""})

    def test_evaluate_binop(self):
        code = "y + x"
        state = {"x": 3, "y": 6}
        result = evaluate_python_code(code, {}, state=state)
        assert result == 9
        self.assertDictEqual(state, {"x": 3, "y": 6, "print_outputs": ""})

    def test_recursive_function(self):
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
        code = "'hello'.replace('h', 'o').split('e')"
        result = evaluate_python_code(code, {}, state={})
        assert result == ["o", "llo"]

    def test_evaluate_slicing(self):
        code = "'hello'[1:3][::-1]"
        result = evaluate_python_code(code, {}, state={})
        assert result == "le"

    def test_access_attributes(self):
        code = "integer = 1\nobj_class = integer.__class__\nobj_class"
        result = evaluate_python_code(code, {}, state={})
        assert result == int

    def test_list_comprehension(self):
        code = "sentence = 'THESEAGULL43'\nmeaningful_sentence = '-'.join([char.lower() for char in sentence if char.isalpha()])"
        result = evaluate_python_code(code, {}, state={})
        assert result == "t-h-e-s-e-a-g-u-l-l"

    def test_string_indexing(self):
        code = """text_block = [
    "THESE",
    "AGULL"
]
sentence = ""
for block in text_block:
    for col in range(len(text_block[0])):
        sentence += block[col]
        """
        result = evaluate_python_code(code, {"len": len, "range": range}, state={})
        assert result == "THESEAGULL"

    def test_tuples(self):
        code = "x = (1, 2, 3)\nx[1]"
        result = evaluate_python_code(code, {}, state={})
        assert result == 2

    def test_listcomp(self):
        code = "x = [i for i in range(3)]"
        result = evaluate_python_code(code, {"range": range}, state={})
        assert result == [0, 1, 2]

    def test_break_continue(self):
        code = "for i in range(10):\n    if i == 5:\n        break\ni"
        result = evaluate_python_code(code, {"range": range}, state={})
        assert result == 5

        code = "for i in range(10):\n    if i == 5:\n        continue\ni"
        result = evaluate_python_code(code, {"range": range}, state={})
        assert result == 9

    def test_call_int(self):
        code = "import math\nstr(math.ceil(149))"
        result = evaluate_python_code(code, {"str": lambda x: str(x)}, state={})
        assert result == "149"

    def test_lambda(self):
        code = "f = lambda x: x + 2\nf(3)"
        result = evaluate_python_code(code, {}, state={})
        assert result == 5

    def test_dictcomp(self):
        code = "x = {i: i**2 for i in range(3)}"
        result = evaluate_python_code(code, {"range": range}, state={})
        assert result == {0: 0, 1: 1, 2: 4}

    def test_tuple_assignment(self):
        code = "a, b = 0, 1\nb"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == 1

    def test_while(self):
        code = "i = 0\nwhile i < 3:\n    i += 1\ni"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == 3

        # test infinite loop
        code = "i = 0\nwhile i < 3:\n    i -= 1\ni"
        with pytest.raises(InterpretorError) as e:
            evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert "iterations in While loop exceeded" in str(e)

    def test_generator(self):
        code = "a = [1, 2, 3, 4, 5]; b = (i**2 for i in a); list(b)"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == [1, 4, 9, 16, 25]

    def test_boolops(self):
        code = """if (not (a > b and a > c)) or d > e:
    best_city = "Brooklyn"
else:
    best_city = "Manhattan"
    best_city
    """
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
        assert result == "Brooklyn"

        code = """if d > e and a < b:
    best_city = "Brooklyn"
elif d < e and a < b:
    best_city = "Sacramento"
else:
    best_city = "Manhattan"
    best_city
    """
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={"a": 1, "b": 2, "c": 3, "d": 4, "e": 5})
        assert result == "Sacramento"

    def test_if_conditions(self):
        code = """char='a'
if char.isalpha():
    print('2')"""
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == "2"

    def test_imports(self):
        code = "import math\nmath.sqrt(4)"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == 2.0

        code = "from random import choice, seed\nseed(12)\nchoice(['win', 'lose', 'draw'])"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == "lose"

        code = "import time\ntime.sleep(0.1)"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result is None

        code = "from queue import Queue\nq = Queue()\nq.put(1)\nq.get()"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == 1

        code = "import itertools\nlist(itertools.islice(range(10), 3))"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == [0, 1, 2]

        code = "import re\nre.search('a', 'abc').group()"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == "a"

        code = "import stat\nstat.S_ISREG(0o100644)"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result

        code = "import statistics\nstatistics.mean([1, 2, 3, 4, 4])"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == 2.8

        code = "import unicodedata\nunicodedata.name('A')"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == "LATIN CAPITAL LETTER A"

    def test_multiple_comparators(self):
        code = "0x30A0 <= ord('a') <= 0x30FF"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result

    def test_print_output(self):
        code = "print('Hello world!')\nprint('Ok no one cares')"
        state = {}
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state)
        assert result == "Ok no one cares"
        assert state["print_outputs"] == "Hello world!\nOk no one cares\n"
