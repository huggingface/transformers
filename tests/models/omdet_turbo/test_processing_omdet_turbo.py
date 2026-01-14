# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from transformers import OmDetTurboProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


IMAGE_MEAN = [123.675, 116.28, 103.53]
IMAGE_STD = [58.395, 57.12, 57.375]

if is_torch_available():
    import torch

    from transformers.models.omdet_turbo.modeling_omdet_turbo import OmDetTurboObjectDetectionOutput


@require_torch
@require_vision
class OmDetTurboProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = OmDetTurboProcessor
    text_input_name = "classes_input_ids"
    input_keys = [
        "tasks_input_ids",
        "tasks_attention_mask",
        "classes_input_ids",
        "classes_attention_mask",
        "classes_structure",
        "pixel_values",
        "pixel_mask",
    ]

    batch_size = 5
    num_queries = 5
    embed_dim = 3

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        return tokenizer_class.from_pretrained("openai/clip-vit-base-patch32")

    def get_fake_omdet_turbo_output(self):
        classes = self.get_fake_omdet_turbo_classes()
        classes_structure = torch.tensor([len(sublist) for sublist in classes])
        torch.manual_seed(42)
        return OmDetTurboObjectDetectionOutput(
            decoder_coord_logits=torch.rand(self.batch_size, self.num_queries, 4),
            decoder_class_logits=torch.rand(self.batch_size, self.num_queries, self.embed_dim),
            classes_structure=classes_structure,
        )

    def get_fake_omdet_turbo_classes(self):
        return [[f"class{i}_{j}" for i in range(self.num_queries)] for j in range(self.batch_size)]

    def test_post_process_grounded_object_detection(self):
        processor = self.get_processor()
        omdet_turbo_output = self.get_fake_omdet_turbo_output()
        omdet_turbo_classes = self.get_fake_omdet_turbo_classes()

        post_processed = processor.post_process_grounded_object_detection(
            omdet_turbo_output, omdet_turbo_classes, target_sizes=[(400, 30) for _ in range(self.batch_size)]
        )

        self.assertEqual(len(post_processed), self.batch_size)
        self.assertEqual(list(post_processed[0].keys()), ["boxes", "scores", "labels", "text_labels"])
        self.assertEqual(post_processed[0]["boxes"].shape, (self.num_queries, 4))
        self.assertEqual(post_processed[0]["scores"].shape, (self.num_queries,))
        expected_scores = torch.tensor([0.7310, 0.6579, 0.6513, 0.6444, 0.6252])
        torch.testing.assert_close(post_processed[0]["scores"], expected_scores, rtol=1e-4, atol=1e-4)

        expected_box_slice = torch.tensor([14.9657, 141.2052, 30.0000, 312.9670])
        torch.testing.assert_close(post_processed[0]["boxes"][0], expected_box_slice, rtol=1e-4, atol=1e-4)
