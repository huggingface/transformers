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

import os
import unittest

from transformers import GroundingDinoProcessor
from transformers.models.bert.tokenization_bert import VOCAB_FILES_NAMES
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

    from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoObjectDetectionOutput


@require_torch
@require_vision
class GroundingDinoProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    model_id = "IDEA-Research/grounding-dino-base"
    processor_class = GroundingDinoProcessor
    batch_size = 7
    num_queries = 5
    embed_dim = 5
    seq_length = 5

    @classmethod
    def _setup_image_processor(cls):
        image_processor_class = cls._get_component_class_from_processor("image_processor")
        return image_processor_class(
            do_resize=True,
            size=None,
            do_normalize=True,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            do_rescale=True,
            rescale_factor=1 / 255,
            do_pad=True,
        )

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer_class = cls._get_component_class_from_processor("tokenizer")
        vocab_tokens = ["[UNK]","[CLS]","[SEP]","[PAD]","[MASK]","want","##want","##ed","wa","un","runn","##ing",",","low","lowest"]  # fmt: skip
        vocab_file = os.path.join(cls.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        with open(vocab_file, "w", encoding="utf-8") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))
        return tokenizer_class.from_pretrained(cls.tmpdirname)

    @unittest.skip("GroundingDinoProcessor merges candidate labels text")
    def test_tokenizer_defaults(self):
        pass

    def prepare_text_inputs(self, batch_size: int | None = None, **kwargs):
        labels = ["a cat", "remote control"]
        labels_longer = ["a person", "a car", "a dog", "a cat"]

        if batch_size is None:
            return labels

        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")

        if batch_size == 1:
            return [labels]
        return [labels, labels_longer] + [labels] * (batch_size - 2)

    def get_fake_grounding_dino_output(self):
        torch.manual_seed(42)
        return GroundingDinoObjectDetectionOutput(
            pred_boxes=torch.rand(self.batch_size, self.num_queries, 4),
            logits=torch.rand(self.batch_size, self.num_queries, self.embed_dim),
            input_ids=self.get_fake_grounding_dino_input_ids(),
        )

    def get_fake_grounding_dino_input_ids(self):
        input_ids = torch.tensor([101, 1037, 4937, 1012, 102])
        return torch.stack([input_ids] * self.batch_size, dim=0)

    def test_post_process_grounded_object_detection(self):
        processor = self.get_processor()

        grounding_dino_output = self.get_fake_grounding_dino_output()

        post_processed = processor.post_process_grounded_object_detection(grounding_dino_output)

        self.assertEqual(len(post_processed), self.batch_size)
        self.assertEqual(list(post_processed[0].keys()), ["scores", "boxes", "text_labels", "labels"])
        self.assertEqual(post_processed[0]["boxes"].shape, (self.num_queries, 4))
        self.assertEqual(post_processed[0]["scores"].shape, (self.num_queries,))

        expected_scores = torch.tensor([0.7050, 0.7222, 0.7222, 0.6829, 0.7220])
        torch.testing.assert_close(post_processed[0]["scores"], expected_scores, rtol=1e-4, atol=1e-4)

        expected_box_slice = torch.tensor([0.6908, 0.4354, 1.0737, 1.3947])
        torch.testing.assert_close(post_processed[0]["boxes"][0], expected_box_slice, rtol=1e-4, atol=1e-4)

    def test_text_preprocessing_equivalence(self):
        processor = self.get_processor()

        # check for single input
        formatted_labels = "a cat. a remote control."
        labels = ["a cat", "a remote control"]
        inputs1 = processor(text=formatted_labels, return_tensors="pt")
        inputs2 = processor(text=labels, return_tensors="pt")
        self.assertTrue(
            torch.allclose(inputs1["input_ids"], inputs2["input_ids"]),
            f"Input ids are not equal for single input: {inputs1['input_ids']} != {inputs2['input_ids']}",
        )

        # check for batched input
        formatted_labels = ["a cat. a remote control.", "a car. a person."]
        labels = [["a cat", "a remote control"], ["a car", "a person"]]
        inputs1 = processor(text=formatted_labels, return_tensors="pt", padding=True)
        inputs2 = processor(text=labels, return_tensors="pt", padding=True)
        self.assertTrue(
            torch.allclose(inputs1["input_ids"], inputs2["input_ids"]),
            f"Input ids are not equal for batched input: {inputs1['input_ids']} != {inputs2['input_ids']}",
        )

    def test_processor_text_has_no_visual(self):
        # Overwritten: text inputs have to be nested as well
        processor = self.get_processor()

        text = self.prepare_text_inputs(batch_size=3, modalities="image")
        image_inputs = self.prepare_image_inputs(batch_size=3)
        processing_kwargs = {"return_tensors": "pt", "padding": True}

        # Call with nested list of vision inputs
        image_inputs_nested = [[image] if not isinstance(image, list) else image for image in image_inputs]
        inputs_dict_nested = {"text": text, "images": image_inputs_nested}
        inputs = processor(**inputs_dict_nested, **processing_kwargs)
        self.assertTrue(self.text_input_name in inputs)

        # Call with one of the samples with no associated vision input
        plain_text = ["lower newer"]
        image_inputs_nested[0] = []
        text[0] = plain_text
        inputs_dict_no_vision = {"text": text, "images": image_inputs_nested}
        inputs_nested = processor(**inputs_dict_no_vision, **processing_kwargs)

        # Check that text samples are same and are expanded with placeholder tokens correctly. First sample
        # has no vision input associated, so we skip it and check it has no vision
        self.assertListEqual(
            inputs[self.text_input_name][1:].tolist(), inputs_nested[self.text_input_name][1:].tolist()
        )
