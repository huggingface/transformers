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

import numpy as np
import pytest

from transformers import load_tool
from transformers.agents.agent_types import AGENT_TYPE_MAPPING
from transformers.agents.default_tools import BASE_PYTHON_TOOLS
from transformers.agents.python_interpreter import InterpreterError, evaluate_python_code

from .test_tools_common import ToolTesterMixin


# Fake function we will use as tool
def add_two(x):
    return x + 2


class PythonInterpreterToolTester(unittest.TestCase, ToolTesterMixin):
    def setUp(self):
        self.tool = load_tool("python_interpreter", authorized_imports=["sqlite3"])
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

        code = "a=1;b=None"
        result = evaluate_python_code(code, {}, state={})
        # evaluate returns the value of the last assignment.
        assert result is None

    def test_assignment_cannot_overwrite_tool(self):
        code = "print = '3'"
        with pytest.raises(InterpreterError) as e:
            evaluate_python_code(code, {"print": print}, state={})
        assert "Cannot assign to name 'print': doing this would erase the existing tool!" in str(e)

    def test_evaluate_call(self):
        code = "y = add_two(x)"
        state = {"x": 3}
        result = evaluate_python_code(code, {"add_two": add_two}, state=state)
        assert result == 5
        self.assertDictEqual(state, {"x": 3, "y": 5, "print_outputs": ""})

        # Should not work without the tool
        with pytest.raises(InterpreterError) as e:
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

        code = "vendor = {'revenue': 31000, 'rent': 50312}; vendor['ratio'] = round(vendor['revenue'] / vendor['rent'], 2)"
        state = {}
        evaluate_python_code(code, {"min": min, "print": print, "round": round}, state=state)
        assert state["vendor"] == {"revenue": 31000, "rent": 50312, "ratio": 0.62}

    def test_subscript_string_with_string_index_raises_appropriate_error(self):
        code = """
search_results = "[{'title': 'Paris, Ville de Paris, France Weather Forecast | AccuWeather', 'href': 'https://www.accuweather.com/en/fr/paris/623/weather-forecast/623', 'body': 'Get the latest weather forecast for Paris, Ville de Paris, France , including hourly, daily, and 10-day outlooks. AccuWeather provides you with reliable and accurate information on temperature ...'}]"
for result in search_results:
    if 'current' in result['title'].lower() or 'temperature' in result['title'].lower():
        current_weather_url = result['href']
        print(current_weather_url)
        break"""
        with pytest.raises(InterpreterError) as e:
            evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
            assert "You're trying to subscript a string with a string index" in e

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
        assert result is int

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

        code = """
digits, i = [1, 2, 3], 1
digits[i], digits[i + 1] = digits[i + 1], digits[i]"""
        evaluate_python_code(code, {"range": range, "print": print, "int": int}, {})

        code = """
def calculate_isbn_10_check_digit(number):
    total = sum((10 - i) * int(digit) for i, digit in enumerate(number))
    remainder = total % 11
    check_digit = 11 - remainder
    if check_digit == 10:
        return 'X'
    elif check_digit == 11:
        return '0'
    else:
        return str(check_digit)

# Given 9-digit numbers
numbers = [
    "478225952",
    "643485613",
    "739394228",
    "291726859",
    "875262394",
    "542617795",
    "031810713",
    "957007669",
    "871467426"
]

# Calculate check digits for each number
check_digits = [calculate_isbn_10_check_digit(number) for number in numbers]
print(check_digits)
"""
        state = {}
        evaluate_python_code(
            code, {"range": range, "print": print, "sum": sum, "enumerate": enumerate, "int": int, "str": str}, state
        )

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

        code = "{num: name for num, name in {101: 'a', 102: 'b'}.items() if name not in ['a']}"
        result = evaluate_python_code(code, {"print": print}, state={}, authorized_imports=["pandas"])
        assert result == {102: "b"}

        code = """
shifts = {'A': ('6:45', '8:00'), 'B': ('10:00', '11:45')}
shift_minutes = {worker: ('a', 'b') for worker, (start, end) in shifts.items()}
"""
        result = evaluate_python_code(code, {}, state={})
        assert result == {"A": ("a", "b"), "B": ("a", "b")}

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
        with pytest.raises(InterpreterError) as e:
            evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert "iterations in While loop exceeded" in str(e)

        # test lazy evaluation
        code = """
house_positions = [0, 7, 10, 15, 18, 22, 22]
i, n, loc = 0, 7, 30
while i < n and house_positions[i] <= loc:
    i += 1
"""
        state = {}
        evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state)

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
        state = {}
        evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state)
        assert state["print_outputs"] == "2\n"

    def test_imports(self):
        code = "import math\nmath.sqrt(4)"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == 2.0

        code = "from random import choice, seed\nseed(12)\nchoice(['win', 'lose', 'draw'])"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == "lose"

        code = "import time, re\ntime.sleep(0.1)"
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

        # Test submodules are handled properly, thus not raising error
        code = "import numpy.random as rd\nrng = rd.default_rng(12345)\nrng.random()"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={}, authorized_imports=["numpy"])

        code = "from numpy.random import default_rng as d_rng\nrng = d_rng(12345)\nrng.random()"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={}, authorized_imports=["numpy"])

    def test_additional_imports(self):
        code = "import numpy as np"
        evaluate_python_code(code, authorized_imports=["numpy"], state={})

        code = "import numpy.random as rd"
        evaluate_python_code(code, authorized_imports=["numpy.random"], state={})
        evaluate_python_code(code, authorized_imports=["numpy"], state={})
        with pytest.raises(InterpreterError):
            evaluate_python_code(code, authorized_imports=["random"], state={})

    def test_multiple_comparators(self):
        code = "0 <= -1 < 4 and 0 <= -5 < 4"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert not result

        code = "0 <= 1 < 4 and 0 <= -5 < 4"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert not result

        code = "0 <= 4 < 4 and 0 <= 3 < 4"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert not result

        code = "0 <= 3 < 4 and 0 <= 3 < 4"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result

    def test_print_output(self):
        code = "print('Hello world!')\nprint('Ok no one cares')"
        state = {}
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state=state)
        assert result is None
        assert state["print_outputs"] == "Hello world!\nOk no one cares\n"

        # test print in function
        code = """
print("1")
def function():
    print("2")
function()"""
        state = {}
        evaluate_python_code(code, {"print": print}, state=state)
        assert state["print_outputs"] == "1\n2\n"

    def test_tuple_target_in_iterator(self):
        code = "for a, b in [('Ralf Weikert', 'Austria'), ('Samuel Seungwon Lee', 'South Korea')]:res = a.split()[0]"
        result = evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert result == "Samuel"

    def test_classes(self):
        code = """
class Animal:
    species = "Generic Animal"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def sound(self):
        return "The animal makes a sound."

    def __str__(self):
        return f"{self.name}, {self.age} years old"

class Dog(Animal):
    species = "Canine"

    def __init__(self, name, age, breed):
        super().__init__(name, age)
        self.breed = breed

    def sound(self):
        return "The dog barks."

    def __str__(self):
        return f"{self.name}, {self.age} years old, {self.breed}"

class Cat(Animal):
    def sound(self):
        return "The cat meows."

    def __str__(self):
        return f"{self.name}, {self.age} years old, {self.species}"


# Testing multiple instances
dog1 = Dog("Fido", 3, "Labrador")
dog2 = Dog("Buddy", 5, "Golden Retriever")

# Testing method with built-in function
animals = [dog1, dog2, Cat("Whiskers", 2)]
num_animals = len(animals)

# Testing exceptions in methods
class ExceptionTest:
    def method_that_raises(self):
        raise ValueError("An error occurred")

try:
    exc_test = ExceptionTest()
    exc_test.method_that_raises()
except ValueError as e:
    exception_message = str(e)


# Collecting results
dog1_sound = dog1.sound()
dog1_str = str(dog1)
dog2_sound = dog2.sound()
dog2_str = str(dog2)
cat = Cat("Whiskers", 2)
cat_sound = cat.sound()
cat_str = str(cat)
    """
        state = {}
        evaluate_python_code(code, {"print": print, "len": len, "super": super, "str": str, "sum": sum}, state=state)

        # Assert results
        assert state["dog1_sound"] == "The dog barks."
        assert state["dog1_str"] == "Fido, 3 years old, Labrador"
        assert state["dog2_sound"] == "The dog barks."
        assert state["dog2_str"] == "Buddy, 5 years old, Golden Retriever"
        assert state["cat_sound"] == "The cat meows."
        assert state["cat_str"] == "Whiskers, 2 years old, Generic Animal"
        assert state["num_animals"] == 3
        assert state["exception_message"] == "An error occurred"

    def test_variable_args(self):
        code = """
def var_args_method(self, *args, **kwargs):
    return sum(args) + sum(kwargs.values())

var_args_method(1, 2, 3, x=4, y=5)
"""
        state = {}
        result = evaluate_python_code(code, {"sum": sum}, state=state)
        assert result == 15

    def test_exceptions(self):
        code = """
def method_that_raises(self):
    raise ValueError("An error occurred")

try:
    method_that_raises()
except ValueError as e:
    exception_message = str(e)
    """
        state = {}
        evaluate_python_code(code, {"print": print, "len": len, "super": super, "str": str, "sum": sum}, state=state)
        assert state["exception_message"] == "An error occurred"

    def test_print(self):
        code = "print(min([1, 2, 3]))"
        state = {}
        evaluate_python_code(code, {"min": min, "print": print}, state=state)
        assert state["print_outputs"] == "1\n"

    def test_types_as_objects(self):
        code = "type_a = float(2); type_b = str; type_c = int"
        state = {}
        result = evaluate_python_code(code, {"float": float, "str": str, "int": int}, state=state)
        assert result is int

    def test_tuple_id(self):
        code = """
food_items = {"apple": 2, "banana": 3, "orange": 1, "pear": 1}
unique_food_items = [item for item, count in food_item_counts.items() if count == 1]
"""
        state = {}
        result = evaluate_python_code(code, {}, state=state)
        assert result == ["orange", "pear"]

    def test_nonsimple_augassign(self):
        code = """
counts_dict = {'a': 0}
counts_dict['a'] += 1
counts_list = [1, 2, 3]
counts_list += [4, 5, 6]

class Counter:
    self.count = 0

a = Counter()
a.count += 1
"""
        state = {}
        evaluate_python_code(code, {}, state=state)
        assert state["counts_dict"] == {"a": 1}
        assert state["counts_list"] == [1, 2, 3, 4, 5, 6]
        assert state["a"].count == 1

    def test_adding_int_to_list_raises_error(self):
        code = """
counts = [1, 2, 3]
counts += 1"""
        with pytest.raises(InterpreterError) as e:
            evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert "Cannot add non-list value 1 to a list." in str(e)

    def test_error_highlights_correct_line_of_code(self):
        code = """# Ok this is a very long code
# It has many commented lines
a = 1
b = 2

# Here is another piece
counts = [1, 2, 3]
counts += 1
b += 1"""
        with pytest.raises(InterpreterError) as e:
            evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert "Evaluation stopped at line 'counts += 1" in str(e)

    def test_assert(self):
        code = """
assert 1 == 1
assert 1 == 2
"""
        with pytest.raises(AssertionError) as e:
            evaluate_python_code(code, BASE_PYTHON_TOOLS, state={})
        assert "1 == 2" in str(e) and "1 == 1" not in str(e)

    def test_with_context_manager(self):
        code = """
class SimpleLock:
    def __init__(self):
        self.locked = False

    def __enter__(self):
        self.locked = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.locked = False

lock = SimpleLock()

with lock as l:
    assert l.locked == True

assert lock.locked == False
    """
        state = {}
        tools = {}
        evaluate_python_code(code, tools, state=state)

    def test_default_arg_in_function(self):
        code = """
def f(a, b=333, n=1000):
    return b + n
n = f(1, n=667)
"""
        res = evaluate_python_code(code, {}, {})
        assert res == 1000

    def test_set(self):
        code = """
S1 = {'a', 'b', 'c'}
S2 = {'b', 'c', 'd'}
S3 = S1.difference(S2)
S4 = S1.intersection(S2)
"""
        state = {}
        evaluate_python_code(code, {}, state=state)
        assert state["S3"] == {"a"}
        assert state["S4"] == {"b", "c"}

    def test_break(self):
        code = """
i = 0

while True:
    i+= 1
    if i==3:
        break

i"""
        result = evaluate_python_code(code, {"print": print, "round": round}, state={})
        assert result == 3

    def test_return(self):
        # test early returns
        code = """
def add_one(n, shift):
    if True:
        return n + shift
    return n

add_one(1, 1)
"""
        state = {}
        result = evaluate_python_code(code, {"print": print, "range": range, "ord": ord, "chr": chr}, state=state)
        assert result == 2

        # test returning None
        code = """
def returns_none(a):
    return

returns_none(1)
"""
        state = {}
        result = evaluate_python_code(code, {"print": print, "range": range, "ord": ord, "chr": chr}, state=state)
        assert result is None

    def test_nested_for_loop(self):
        code = """
all_res = []
for i in range(10):
    subres = []
    for j in range(i):
        subres.append(j)
    all_res.append(subres)

out = [i for sublist in all_res for i in sublist]
out[:10]
"""
        state = {}
        result = evaluate_python_code(code, {"print": print, "range": range}, state=state)
        assert result == [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]

    def test_pandas(self):
        code = """
import pandas as pd

df = pd.DataFrame.from_dict({'SetCount': ['5', '4', '5'], 'Quantity': [1, 0, -1]})

df['SetCount'] = pd.to_numeric(df['SetCount'], errors='coerce')

parts_with_5_set_count = df[df['SetCount'] == 5.0]
parts_with_5_set_count[['Quantity', 'SetCount']].values[1]
"""
        state = {}
        result = evaluate_python_code(code, {}, state=state, authorized_imports=["pandas"])
        assert np.array_equal(result, [-1, 5])

        code = """
import pandas as pd

df = pd.DataFrame.from_dict({"AtomicNumber": [111, 104, 105], "ok": [0, 1, 2]})
print("HH0")

# Filter the DataFrame to get only the rows with outdated atomic numbers
filtered_df = df.loc[df['AtomicNumber'].isin([104])]
"""
        result = evaluate_python_code(code, {"print": print}, state={}, authorized_imports=["pandas"])
        assert np.array_equal(result.values[0], [104, 1])

        code = """import pandas as pd
data = pd.DataFrame.from_dict([
    {"Pclass": 1, "Survived": 1},
    {"Pclass": 2, "Survived": 0},
    {"Pclass": 2, "Survived": 1}
])
survival_rate_by_class = data.groupby('Pclass')['Survived'].mean()
"""
        result = evaluate_python_code(code, {}, state={}, authorized_imports=["pandas"])
        assert result.values[1] == 0.5

    def test_starred(self):
        code = """
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of the Earth in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance

coords_geneva = (46.1978, 6.1342)
coords_barcelona = (41.3869, 2.1660)

distance_geneva_barcelona = haversine(*coords_geneva, *coords_barcelona)
"""
        result = evaluate_python_code(code, {"print": print, "map": map}, state={}, authorized_imports=["math"])
        assert round(result, 1) == 622395.4

    def test_for(self):
        code = """
shifts = {
    "Worker A": ("6:45 pm", "8:00 pm"),
    "Worker B": ("10:00 am", "11:45 am")
}

shift_intervals = {}
for worker, (start, end) in shifts.items():
    shift_intervals[worker] = end
shift_intervals
"""
        result = evaluate_python_code(code, {"print": print, "map": map}, state={})
        assert result == {"Worker A": "8:00 pm", "Worker B": "11:45 am"}
