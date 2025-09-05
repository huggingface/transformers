# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
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

import shutil
import tempfile
import unittest

from PIL import Image
import librosa
import requests
import io

from transformers import (
    AutoProcessor,
    MiniCPM_o_2_6Processor,
)
from transformers.testing_utils import require_librosa, require_torch, require_torchaudio, require_vision
from transformers.utils import is_torch_available 


if is_torch_available():
    import torch

@require_vision
@require_torch
@require_torchaudio
class MiniCPM_o_2_6ProcessorTest(unittest.TestCase):
    processor_class = MiniCPM_o_2_6Processor

    # 添加这个属性，告知测试框架处理器有自己的chat template实现
    has_chat_template = True

    # 输入名称定义
    images_input_name = "pixel_values"
    audio_input_name = "audio_features"

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        processor_kwargs = cls.prepare_processor_dict()
        processor = AutoProcessor.from_pretrained("openbmb/MiniCPM-o-2_6", **processor_kwargs)
        processor.save_pretrained(cls.tmpdirname)

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
        }

    def get_processor(self, **kwargs):
        return MiniCPM_o_2_6Processor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def tearDown(self):
        # 防止复写父类的tearDown方法
        pass

    def prepare_image_inputs(self, batch_size=1):
        """Preparing the image input for testing"""
        image_url = "https://bkimg.cdn.bcebos.com/pic/d043ad4bd11373f082022267f9585cfbfbedaa64aeb6"
        image = Image.open(requests.get(image_url, stream=True).raw)
        if batch_size == 1:
            return image
        return [[image]] * batch_size

    def prepare_audio_inputs(self, batch_size=1):
        """Preparing the audio input for testing"""
        audio_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"
        response = requests.get(audio_url)
        audio, _ = librosa.load(io.BytesIO(response.content), sr=16000, mono=True)
        if batch_size == 1:
            return audio
        return [[audio]] * batch_size

    def prepare_text_inputs(self, batch_size=1, modality="text"):
        """准备文本输入"""
        if modality == "text":
            text = "Hello, how are you?"
        elif modality == "image":
            text = "What is in this image: (<image>./</image>)?"
        elif modality == "audio":
            text = "What was said in this audio: (<audio>./</audio>)?"
        elif modality == "mixed":
            text = "Describe this image (<image>./</image>) and this audio (<audio>./</audio>)."

        if batch_size == 1:
            return text
        return [text] * batch_size

    def test_processor_with_multi_modal_inputs(self):
        """Test the processor with multiple modality inputs"""
        processor = self.get_processor()

        input_text = self.prepare_text_inputs(modality="mixed")
        image = self.prepare_image_inputs()
        audio = self.prepare_audio_inputs()

        # Test with all modalities
        inputs = processor(text=[input_text], images=[image], audios=[audio], return_tensors="pt")

        # 检查输入中包含所有模态相关的键
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("pixel_values", inputs)
        self.assertIn("image_bound", inputs)
        self.assertIn("audio_features", inputs)
        self.assertIn("audio_bounds", inputs)

        # 检查处理器能处理批量输入
        batch_size = 2
        input_texts = self.prepare_text_inputs(batch_size=batch_size, modality="mixed")
        images = self.prepare_image_inputs(batch_size=batch_size)
        audios = self.prepare_audio_inputs(batch_size=batch_size)

        batch_inputs = processor(text=input_texts, images=images, audios=audios, return_tensors="pt")
        self.assertEqual(batch_inputs["input_ids"].shape[0], batch_size)
        self.assertEqual(batch_inputs["attention_mask"].shape[0], batch_size)

    @require_torch
    def test_apply_chat_template_text(self):
        """Test applying chat template with text only"""
        processor = self.get_processor()

        # 准备聊天消息
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
        ]

        # 测试不带标记化的应用
        formatted_prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False)
        self.assertIsInstance(formatted_prompt, str)
        self.assertTrue("<|im_start|>" in formatted_prompt)
        self.assertTrue("<|im_end|>" in formatted_prompt)

        # 测试带标记化的应用
        tokenized_prompt = processor.tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
        self.assertIsInstance(tokenized_prompt, torch.Tensor)

    @require_torch
    @require_vision
    def test_apply_chat_template_with_image(self):
        """Test applying chat template with image"""
        processor = self.get_processor()
        image = self.prepare_image_inputs()

        # 准备包含图像的消息
        messages = [{"role": "user", "content": [image, "What is in this image?"]}]

        # 测试聊天模板应用
        inputs = processor.apply_chat_template(msgs=messages, omni_input=True, return_tensors="pt")

        # 检查输出中有图像相关的数据
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("pixel_values", inputs)
        self.assertIn("image_bound", inputs)

    @require_torch
    @require_librosa
    def test_apply_chat_template_with_audio(self):
        """Test applying chat template with audio"""
        processor = self.get_processor()
        audio = self.prepare_audio_inputs()

        # 准备包含音频的消息
        messages = [{"role": "user", "content": [audio, "What was said in this audio?"]}]

        # 测试聊天模板应用
        inputs = processor.apply_chat_template(
            msgs=messages, omni_input=True, use_tts_template=True, return_tensors="pt"
        )

        # 检查输出中有音频相关的数据
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("audio_features", inputs)
        self.assertIn("audio_bounds", inputs)

    @require_torch
    @require_vision
    @require_librosa
    def test_apply_chat_template_with_mixed_media(self):
        """Test applying chat template with both image and audio"""
        processor = self.get_processor()
        image = self.prepare_image_inputs()
        audio = self.prepare_audio_inputs()

        # 准备包含多种媒体的消息
        messages = [{"role": "user", "content": [image, audio, "Describe this image and audio."]}]

        # 测试聊天模板应用
        inputs = processor.apply_chat_template(
            msgs=messages, omni_input=True, use_tts_template=True, return_tensors="pt"
        )

        # 检查输出中有所有模态相关的数据
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertIn("pixel_values", inputs)
        self.assertIn("image_bound", inputs)
        self.assertIn("audio_features", inputs)
        self.assertIn("audio_bounds", inputs)

    def test_get_sys_prompt(self):
        """Test the system prompt generation function"""
        processor = self.get_processor()
        audio = self.prepare_audio_inputs()

        # 测试默认系统提示
        sys_prompt = processor.get_sys_prompt()
        self.assertIn("role", sys_prompt)
        self.assertIn("content", sys_prompt)

        # 测试 omni 模式
        sys_prompt = processor.get_sys_prompt(mode="omni", language="zh")
        self.assertIn("你是一个AI助手", sys_prompt["content"][0])

        # 测试英文提示
        sys_prompt = processor.get_sys_prompt(mode="omni", language="en")
        self.assertIn("You are a helpful assistant", sys_prompt["content"][0])

        # 测试语音克隆提示
        sys_prompt = processor.get_sys_prompt(ref_audio=audio, mode="voice_cloning", language="en")
        self.assertIn("Clone the voice", sys_prompt["content"][0])

        # 测试语音助手模式
        sys_prompt = processor.get_sys_prompt(ref_audio=audio, mode="audio_assistant", language="zh")
        self.assertIn("模仿输入音频中的声音特征", sys_prompt["content"][0])

    def test_decode(self):
        """Test the decoding functionality"""
        processor = self.get_processor()

        # 创建模拟输出
        class MockOutput:
            def __init__(self, sequences):
                self.sequences = sequences

        # 测试单个输出
        mock_output = MockOutput(torch.tensor([[1, 2, 3, 4, 0, 0]]))
        result = processor.decode(mock_output.sequences)
        self.assertIsInstance(result[0], str)

        # 测试批量输出
        mock_output = MockOutput(torch.tensor([[1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 0, 0]]))
        result = processor.decode(mock_output.sequences)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
