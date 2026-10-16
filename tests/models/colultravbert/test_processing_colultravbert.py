# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import torch

from transformers import AutoTokenizer, UltraVBertImageProcessor
from transformers.models.colultravbert.processing_colultravbert import ColUltraVBertProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image


# The order sets ids 49152-49191, as in the UltraVBert checkpoints.
IMAGE_TOKENS = (
    ["<global-img>"]
    + [f"<row_{i}_col_{j}>" for i in range(1, 7) for j in range(1, 7)]
    + ["<end_of_utterance>", "<fake_token_around_image>", "<image>"]
)


@require_vision
class ColUltraVBertProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = ColUltraVBertProcessor

    @classmethod
    def _setup_tokenizer(cls):
        tokenizer = AutoTokenizer.from_pretrained("dmis-lab/UltraBERT-150M")
        tokenizer.add_special_tokens({"extra_special_tokens": IMAGE_TOKENS}, replace_extra_special_tokens=False)
        return tokenizer

    @classmethod
    def _setup_image_processor(cls):
        return UltraVBertImageProcessor(size={"longest_edge": 2048}, max_image_size={"longest_edge": 512})

    @require_torch
    def test_process_images(self):
        processor = self.get_processor()
        batch_feature = processor.process_images(images=self.prepare_images_inputs())

        self.assertIsInstance(batch_feature["pixel_values"], torch.Tensor)
        self.assertEqual(batch_feature["pixel_values"].shape[1], 5)
        num_image_tokens = (batch_feature["input_ids"] == processor.image_token_id).sum().item()
        self.assertEqual(num_image_tokens, 5 * processor.image_seq_len)
        self.assertEqual(
            processor.visual_prompt_prefix, "User:<image>Describe the image.<end_of_utterance>\nAssistant:"
        )

    @require_torch
    def test_process_queries(self):
        queries = [
            "Is attention really all you need?",
            "Are Benjamin, Antoine, Merve, and Jo best friends?",
        ]
        processor = self.get_processor()
        batch_feature = processor.process_queries(text=queries)

        self.assertIsInstance(batch_feature["input_ids"], torch.Tensor)
        self.assertEqual(batch_feature["input_ids"].shape[0], len(queries))
        end_of_utterance_id = processor.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
        self.assertEqual((batch_feature["input_ids"] == end_of_utterance_id).sum(dim=1).tolist(), [10, 10])

    @require_torch
    def test_process_images_with_max_image_tiles(self):
        image_processor = UltraVBertImageProcessor(
            size={"longest_edge": 3072}, max_image_size={"longest_edge": 512}, max_image_tiles=27
        )
        processor = self.processor_class(image_processor=image_processor, tokenizer=self.get_component("tokenizer"))
        pages = [Image.new("RGB", size, "white") for size in ((2480, 3508), (1920, 1080), (3000, 3000))]
        batch_feature = processor.process_images(pages)

        num_tiles = [21, 25, 26]
        self.assertEqual(batch_feature["pixel_values"].shape[1], max(num_tiles))
        for row, count in enumerate(num_tiles):
            num_image_tokens = (batch_feature["input_ids"][row] == processor.image_token_id).sum().item()
            self.assertEqual(num_image_tokens, count * processor.image_seq_len)
            # The model drops all-zero tiles as padding.
            self.assertEqual(batch_feature["pixel_values"][row, count:].abs().sum().item(), 0)

    @unittest.skip("Model doesn't take images+text as input")
    def test_replacement_offsets(self):
        pass

    def _test_modality_processor_defaults_preserved_by_modality_kwargs(self, modality):
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=-1.0
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(**processor_components)

        image_input = self.prepare_images_inputs()

        inputs = processor(images=image_input, return_tensors="pt")
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def _test_kwargs_overrides_default_modality_processor_kwargs(self, modality):
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=1
        )
        processor_components["tokenizer"] = self.get_component("tokenizer", padding=None)

        processor = self.processor_class(**processor_components)

        image_input = self.prepare_images_inputs()

        inputs = processor(
            images=image_input,
            do_rescale=True,
            rescale_factor=-1.0,
            max_length=117,
            padding="max_length",
            return_tensors="pt",
        )
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def _test_unstructured_kwargs(self, modality):
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)

        input_str = self.prepare_text_inputs()
        inputs = processor(
            text=input_str,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1.0,
            padding="max_length",
            max_length=76,
        )

        self.assertEqual(inputs[self.text_input_name].shape[-1], 76)

    def _test_unstructured_kwargs_batched(self, modality):
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)

        image_input = self.prepare_images_inputs(batch_size=2)
        inputs = processor(
            images=image_input,
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1.0,
            padding="longest",
            max_length=76,
        )

        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def _test_doubly_passed_kwargs(self, modality):
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)

        image_input = self.prepare_images_inputs()
        with self.assertRaises(ValueError):
            _ = processor(
                images=image_input,
                images_kwargs={"do_rescale": True, "rescale_factor": -1.0},
                do_rescale=True,
                return_tensors="pt",
            )

    def _test_structured_kwargs_nested_from_dict(self, modality):
        processor_components = self.prepare_components()
        processor = self.processor_class(**processor_components)

        image_input = self.prepare_images_inputs()

        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"do_rescale": True, "rescale_factor": -1.0},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(images=image_input, **all_kwargs)
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_model_input_names(self):
        processor = self.get_processor()
        inputs = processor(images=self.prepare_images_inputs())
        self.assertIn("pixel_values", inputs)

    @unittest.skip("ColUltraVBert can't process text+image inputs at the same time")
    def test_processor_text_has_no_visual(self):
        pass

    @unittest.skip("ColUltraVBert can't process text+image inputs at the same time")
    def test_processor_with_multiple_inputs(self):
        pass

    @unittest.skip("ColUltraVBert can't process text+image inputs at the same time")
    def test_get_num_multimodal_tokens_matches_processor_call(self):
        pass

    @unittest.skip("ColUltraVBert can't process text+image inputs at the same time")
    def test_flat_kwarg_applied_when_modality_dict_lacks_it(self):
        pass

    @unittest.skip("ColUltraVBert has no chat template, force-set to None at runtime")
    def test_chat_template_save_loading(self):
        pass
