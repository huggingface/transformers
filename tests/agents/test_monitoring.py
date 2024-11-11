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

from transformers.agents.agents import AgentError, ReactCodeAgent, ReactJsonAgent


class MonitoringTester(unittest.TestCase):
    def test_code_agent_metrics(self):
        class FakeLLMEngine:
            def __init__(self):
                self.last_input_token_count = 10
                self.last_output_token_count = 20

            def __call__(self, prompt, **kwargs):
                return """
Code:
```py
final_answer('This is the final answer.')
```"""

        agent = ReactCodeAgent(
            tools=[],
            llm_engine=FakeLLMEngine(),
            max_iterations=1,
        )

        agent.run("Fake task")

        self.assertEqual(agent.monitor.total_input_token_count, 10)
        self.assertEqual(agent.monitor.total_output_token_count, 20)

    def test_json_agent_metrics(self):
        class FakeLLMEngine:
            def __init__(self):
                self.last_input_token_count = 10
                self.last_output_token_count = 20

            def __call__(self, prompt, **kwargs):
                return 'Action:{"action": "final_answer", "action_input": {"answer": "image"}}'

        agent = ReactJsonAgent(
            tools=[],
            llm_engine=FakeLLMEngine(),
            max_iterations=1,
        )

        agent.run("Fake task")

        self.assertEqual(agent.monitor.total_input_token_count, 10)
        self.assertEqual(agent.monitor.total_output_token_count, 20)

    def test_code_agent_metrics_max_iterations(self):
        class FakeLLMEngine:
            def __init__(self):
                self.last_input_token_count = 10
                self.last_output_token_count = 20

            def __call__(self, prompt, **kwargs):
                return "Malformed answer"

        agent = ReactCodeAgent(
            tools=[],
            llm_engine=FakeLLMEngine(),
            max_iterations=1,
        )

        agent.run("Fake task")

        self.assertEqual(agent.monitor.total_input_token_count, 20)
        self.assertEqual(agent.monitor.total_output_token_count, 40)

    def test_code_agent_metrics_generation_error(self):
        class FakeLLMEngine:
            def __init__(self):
                self.last_input_token_count = 10
                self.last_output_token_count = 20

            def __call__(self, prompt, **kwargs):
                raise AgentError

        agent = ReactCodeAgent(
            tools=[],
            llm_engine=FakeLLMEngine(),
            max_iterations=1,
        )

        agent.run("Fake task")

        self.assertEqual(agent.monitor.total_input_token_count, 20)
        self.assertEqual(agent.monitor.total_output_token_count, 40)
