# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import inspect
import unittest

import numpy as np

from transformers.testing_utils import require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image

    from transformers import Phi4MultimodalProcessor


@require_vision
class Phi4MultimodalProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Phi4MultimodalProcessor
    checkpoint_path = "microsoft/Phi-4-multimodal-instruct"
    revision = "refs/pr/70"
    text_input_name = "input_ids"
    images_input_name = "image_pixel_values"
    audio_input_name = "audio_input_features"

    # Max-length values used in image-text kwargs tests. Override as phi4 needs lots of tokens for images.
    image_text_kwargs_max_length = 400
    image_text_kwargs_override_max_length = 396
    image_unstructured_max_length = 407

    # Max-length values used in audio-text kwargs tests. Override as phi4 needs lots of tokens for audio.
    audio_text_kwargs_max_length = 300
    audio_processor_tester_max_length = 117
    audio_unstructured_max_length = 76

    # Max-length values used in video-text kwargs tests. Override in subclasses if needed.
    video_text_kwargs_max_length = 167
    video_text_kwargs_override_max_length = 162
    video_unstructured_max_length = 176

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        return tokenizer_class.from_pretrained(cls.checkpoint_path, revision=cls.revision)

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        cls.image_token_id = processor.image_token_id
        cls.audio_token = processor.audio_token
        cls.audio_token_id = processor.audio_token_id

    # override: audio_attention_mask is returned conditionally, and not expected in the input names in this case
    def test_model_input_names(self):
        processor = self.get_processor()

        text = self.prepare_text_inputs(modalities=["image", "video", "audio"])
        image_input = self.prepare_image_inputs()
        video_inputs = self.prepare_video_inputs()
        audio_inputs = self.prepare_audio_inputs()
        inputs_dict = {"text": text, "images": image_input, "videos": video_inputs, "audio": audio_inputs}

        call_signature = inspect.signature(processor.__call__)
        input_args = [param.name for param in call_signature.parameters.values()]
        inputs_dict = {k: v for k, v in inputs_dict.items() if k in input_args}

        inputs = processor(**inputs_dict, return_tensors="pt")

        # audio_attention_mask is returned conditionally, and not expected in the input names in this case
        input_names_expected = set(processor.model_input_names) - {"audio_attention_mask"}
        self.assertSetEqual(set(inputs.keys()), input_names_expected)

    def test_dynamic_hd_kwarg_passed_to_image_processor(self):
        processor = self.get_processor()
        # 1000x1000 image: with size=448, w_crop_num=3, h_crop_num=3 -> 9 HD crops (1 global + 9 = 10 total)
        # With dynamic_hd=4: limits to 2x2 grid -> 4 HD crops (1 global + 4 = 5 total)
        arr = np.random.randint(255, size=(3, 1000, 1000), dtype=np.uint8)
        image_input = Image.fromarray(np.moveaxis(arr, 0, -1))
        input_str = self.prepare_text_inputs(modalities="image")

        inputs_default = processor(text=input_str, images=image_input, return_tensors="pt")
        inputs_limited = processor(
            text=input_str,
            images=image_input,
            dynamic_hd=4,
            return_tensors="pt",
        )

        self.assertEqual(inputs_limited[self.images_input_name].shape[1], 5)
        self.assertEqual(inputs_default[self.images_input_name].shape[1], 10)
