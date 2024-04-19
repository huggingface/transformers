#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import math

from .base import Tool
from .python_interpreter import evaluate_python_code
from .agents import BASE_PYTHON_TOOLS


class CalculatorTool(Tool):
    name = "calculator"
    description = "This is a tool that calculates. It can be used to perform simple arithmetic operations."

    inputs = {
        "expression": {
            "type": str,
            "description": "The expression to be evaluated.The variables used CANNOT be placeholders like 'x' or 'mike's age', they must be numbers",
        }
    }
    output_type = str

    def __init__(self):
        import numexpr

        self.numexpr = numexpr

    def __call__(self, expression):
        if type(expression) != str:
            expression = expression['expression']
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            self.numexpr.evaluate(
                expression.strip().replace("^", "**"),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return output


class PythonEvaluatorTool(Tool):
    name = "python_evaluator"
    description = "This is a tool that evaluates python code. It can be used to perform calculations. It does not have access to any imports or function definitions."

    inputs = {
        "code": {
            "type": str,
            "description": "The code snippet to evaluate. All variables used in this snippet must be defined in this same snippet, else you will get an error.",
        }
    }
    output_type = str
    available_tools = BASE_PYTHON_TOOLS.copy()

    def __call__(self, code):
        output = str(
            evaluate_python_code(code, tools=self.available_tools)
        )
        return output
