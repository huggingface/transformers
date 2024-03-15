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


class CalculatorTool(Tool):
    name = "calculator"
    description = "This is a tool that calculates. It can be used to perform simple arithmetic operations."

    inputs = {"expression": {"type": str, "description": "The expression to be evaluated.The variables used CANNOT be placeholders like 'x' or 'mike's age', they must be numbers"}}
    output_type = str

    def __init__(self):
        import numexpr
        self.numexpr = numexpr

    def __call__(self, expression):
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            self.numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return output


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem"
    inputs = {"answer": {"type": str, "description": "The final answer to the problem"}}
    output_type = str

    def __call__(self):
        pass
