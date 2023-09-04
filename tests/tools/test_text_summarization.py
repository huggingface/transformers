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

from transformers import load_tool

from .test_tools_common import ToolTesterMixin


TEXT = """
Hugging Face was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf originally as a company that developed a chatbot app targeted at teenagers.[2] After open-sourcing the model behind the chatbot, the company pivoted to focus on being a platform for machine learning.

In March 2021, Hugging Face raised $40 million in a Series B funding round.[3]

On April 28, 2021, the company launched the BigScience Research Workshop in collaboration with several other research groups to release an open large language model.[4] In 2022, the workshop concluded with the announcement of BLOOM, a multilingual large language model with 176 billion parameters.[5]
"""


class TextSummarizationToolTester(unittest.TestCase, ToolTesterMixin):
    def setUp(self):
        self.tool = load_tool("summarization")
        self.tool.setup()
        self.remote_tool = load_tool("summarization", remote=True)

    def test_exact_match_arg(self):
        result = self.tool(TEXT)
        self.assertEqual(
            result,
            "Hugging Face was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf. In March 2021, Hugging Face raised $40 million in a Series B funding round. On April 28, 2021, the company launched the BigScience Research Workshop in collaboration with several other research groups to release an open large language model. In 2022, the workshop concluded with the announcement of BLOOM.",
        )

    def test_exact_match_arg_remote(self):
        result = self.remote_tool(TEXT)
        self.assertEqual(
            result,
            "Hugging Face was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf. In March 2021, Hugging Face raised $40 million in a Series B funding round. On April 28, 2021, the company launched the BigScience Research Workshop in collaboration with several other research groups to release an open large language model. In 2022, the workshop concluded with the announcement of BLOOM.",
        )

    def test_exact_match_kwarg(self):
        result = self.tool(text=TEXT)
        self.assertEqual(
            result,
            "Hugging Face was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf. In March 2021, Hugging Face raised $40 million in a Series B funding round. On April 28, 2021, the company launched the BigScience Research Workshop in collaboration with several other research groups to release an open large language model. In 2022, the workshop concluded with the announcement of BLOOM.",
        )

    def test_exact_match_kwarg_remote(self):
        result = self.remote_tool(text=TEXT)
        self.assertEqual(
            result,
            "Hugging Face was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf. In March 2021, Hugging Face raised $40 million in a Series B funding round. On April 28, 2021, the company launched the BigScience Research Workshop in collaboration with several other research groups to release an open large language model. In 2022, the workshop concluded with the announcement of BLOOM.",
        )
