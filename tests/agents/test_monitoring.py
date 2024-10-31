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
from transformers.agents.monitoring import stream_to_gradio
from transformers.agents.agents import ReactCodeAgent, ReactJsonAgent, AgentError
from transformers.agents.agent_types import AgentImage

class TestMonitoring(unittest.TestCase):
    def test_streaming_agent_text_output(self):
        # Create a dummy LLM engine that returns a final answer
        def dummy_llm_engine(prompt, **kwargs):
            return "final_answer('This is the final answer.')"

        agent = ReactCodeAgent(
            tools=[],
            llm_engine=dummy_llm_engine,
            max_iterations=1,
        )

        # Use stream_to_gradio to capture the output
        outputs = list(stream_to_gradio(agent, task="Test task"))

        # Check that the final output is a ChatMessage with the expected content
        self.assertEqual(len(outputs), 3)
        final_message = outputs[-1]
        self.assertEqual(final_message.role, "assistant")
        self.assertIn("This is the final answer.", final_message.content)

    def test_streaming_agent_image_output(self):
        # Create a dummy LLM engine that returns an image as final answer
        def dummy_llm_engine(prompt, **kwargs):
            return 'Action:{"action": "final_answer", "action_input": {"answer": "image"}}'

        agent = ReactJsonAgent(
            tools=[],
            llm_engine=dummy_llm_engine,
            max_iterations=1,
        )

        # Use stream_to_gradio to capture the output
        outputs = list(stream_to_gradio(agent, task="Test task", image=AgentImage(value="path.png")))

        # Check that the final output is a ChatMessage with the expected content
        self.assertEqual(len(outputs), 3)
        final_message = outputs[-1]
        self.assertEqual(final_message.role, "assistant")
        self.assertIsInstance(final_message.content, dict)
        self.assertEqual(final_message.content["path"], "path.png")
        self.assertEqual(final_message.content["mime_type"], "image/png")

    def test_streaming_with_agent_error(self):
        # Create a dummy LLM engine that raises an error
        def dummy_llm_engine(prompt, **kwargs):
            raise AgentError("Simulated agent error.")

        agent = ReactCodeAgent(
            tools=[],
            llm_engine=dummy_llm_engine,
            max_iterations=1,
        )

        # Use stream_to_gradio to capture the output
        outputs = list(stream_to_gradio(agent, task="Test task"))

        # Check that the error message is yielded
        print("OUTPUTTTTS", outputs)
        self.assertEqual(len(outputs), 3)
        final_message = outputs[-1]
        self.assertEqual(final_message.role, "assistant")
        self.assertIn("Simulated agent error", final_message.content)

