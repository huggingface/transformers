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
import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from parameterized import parameterized

from tests.integrations.mistral.tekken_fixtures import write_fake_tekken_json
from transformers import AutoProcessor
from transformers.testing_utils import require_mistral_common, require_torchvision, require_vision
from transformers.utils import is_mistral_common_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_vision_available():
    from transformers import PixtralProcessor

if is_mistral_common_available():
    from transformers.tokenization_mistral_common import MistralCommonBackend


def _build_mistral_common_tokenizer():
    """Build a real MistralCommonBackend with all special tokens required by mistral-common."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tekken_path = write_fake_tekken_json(Path(tmpdir))
        tokenizer = MistralCommonBackend(tokenizer_path=tekken_path)
    return tokenizer


@require_vision
class PixtralProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = PixtralProcessor
    tiny_model_id = "hf-internal-testing/tiny-processor-pixtral"
    model_id = "mistral-community/pixtral-12b"

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.url_0 = url_to_local_path(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg"
        )
        cls.image_0 = np.random.randint(255, size=(3, 876, 1300), dtype=np.uint8)
        cls.url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
        cls.image_1 = np.random.randint(255, size=(3, 480, 640), dtype=np.uint8)
        cls.image_2 = np.random.randint(255, size=(3, 1024, 1024), dtype=np.uint8)
        cls.image_token = processor.image_token

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        processor = super()._setup_from_pretrained(model_id, **kwargs)
        processor.tokenizer.pad_token_id = 0  # loaded tokenizer has no PAD defined
        return processor

    @parameterized.expand([(1, "pt"), (2, "pt")])
    @unittest.skip("Not tested before, to investigate")
    def test_apply_chat_template_image(self, batch_size, return_tensors):
        pass

    def test_image_token_filling(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        # Important to check with non square image
        image = torch.randint(0, 2, (3, 500, 316))
        expected_image_tokens = 640
        image_token_index = 10

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        inputs = processor(
            text=[processor.apply_chat_template(messages)],
            images=[image],
            return_tensors="pt",
        )
        image_tokens = (inputs["input_ids"] == image_token_index).sum().item()
        self.assertEqual(expected_image_tokens, image_tokens)

    def test_from_pretrained_subfolder_tokenizer(self):
        processor = PixtralProcessor.from_pretrained("hf-internal-testing/tiny-flux2", subfolder="tokenizer")
        self.assertIsInstance(processor, PixtralProcessor)
        self.assertIsNotNone(processor.tokenizer)

    def test_processor_with_single_image(self):
        processor = self.processor_class.from_pretrained(self.full_tmpdirname)
        prompt_string = "USER: [IMG]\nWhat's the content of the image? ASSISTANT:"

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=self.image_0, return_tensors="pt")
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 1)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to "USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the content of the image? ASSISTANT:"
            [21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=self.url_0, return_tensors="pt")
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 1)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to "USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the content of the image? ASSISTANT:"
            [21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

        # Test passing inputs as a single list
        inputs_image = processor(text=prompt_string, images=[self.image_0], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 32, 32]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

        # Test as nested single list
        inputs_image = processor(text=prompt_string, images=[[self.image_0]], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 32, 32]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

    def test_processor_with_multiple_images_single_list(self):
        processor = self.processor_class.from_pretrained(self.full_tmpdirname)
        prompt_string = "USER: [IMG][IMG]\nWhat's the difference between these two images? ASSISTANT:"

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=[self.image_0, self.image_1], return_tensors="pt")
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 1)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
                    )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=[self.url_0, self.url_1], return_tensors="pt")
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 1)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing in as a nested list
        inputs_url = processor(text=prompt_string, images=[[self.image_0, self.image_1]], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 32, 32]))

        # fmt: off
        self.assertEqual(
            inputs_url["input_ids"][0].tolist(),
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

    def test_processor_with_multiple_images_multiple_lists(self):
        processor = self.processor_class.from_pretrained(self.full_tmpdirname)
        prompt_string = [
            "USER: [IMG][IMG]\nWhat's the difference between these two images? ASSISTANT:",
            "USER: [IMG]\nWhat's the content of the image? ASSISTANT:",
        ]
        processor.tokenizer.pad_token = "</s>"
        image_inputs = [[self.image_0, self.image_1], [self.image_2]]

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 2)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 2)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing as a single flat list
        inputs_image = processor(
            text=prompt_string, images=[self.image_0, self.image_1, self.image_2], return_tensors="pt", padding=True
        )
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 32, 32]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

    @require_mistral_common
    def test_apply_chat_template_with_mistral_common_backend(self):
        """PixtralProcessor.apply_chat_template delegates to MistralCommonBackend and produces real tokens."""

        processor = self.processor_class.from_pretrained(self.tmpdirname)

        mc_tokenizer = _build_mistral_common_tokenizer()

        processor.tokenizer = mc_tokenizer

        conversation = [{"role": "user", "content": "Hello"}]

        result_str = processor.apply_chat_template(conversation, tokenize=False)
        self.assertIsInstance(result_str, str)
        self.assertIn("Hello", result_str)

        result_dict = processor.apply_chat_template(conversation, tokenize=True, return_dict=True)
        self.assertIsInstance(result_dict, Mapping)
        self.assertIn("input_ids", result_dict)
        self.assertGreater(len(result_dict["input_ids"]), 0)

        result_pt = processor.apply_chat_template(conversation, tokenize=True, return_dict=True, return_tensors="pt")
        self.assertIsInstance(result_pt, Mapping)
        self.assertIn("input_ids", result_pt)
        self.assertIsInstance(result_pt["input_ids"], torch.Tensor)
        self.assertGreater(result_pt["input_ids"].numel(), 0)

        result_ids = processor.apply_chat_template(conversation, tokenize=True, return_dict=False)
        self.assertIsInstance(result_ids, list)
        self.assertGreater(len(result_ids), 0)

    def test_processor_returns_full_length_batches(self):
        # to avoid https://github.com/huggingface/transformers/issues/34204
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        prompt_string = [
            "USER: [IMG]\nWhat's the content of the image? ASSISTANT:",
        ] * 5
        processor.tokenizer.pad_token = "</s>"
        image_inputs = [[self.image_0]] * 5

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 5)
        self.assertTrue(len(inputs_image["pixel_values"]) == 5)


def _write_fake_params_json(directory: Path) -> None:
    """Write a minimal `params.json` with a `vision_encoder` block."""
    params = {
        "vision_encoder": {
            "patch_size": 16,
            "image_size": 512,
            "spatial_merge_size": 1,
        }
    }
    with open(directory / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f)


def _write_fake_preprocessor_config(directory: Path) -> None:
    """Write a minimal `preprocessor_config.json` for PixtralImageProcessor."""
    config = {
        "image_processor_type": "PixtralImageProcessor",
        "patch_size": {"height": 16, "width": 16},
        "size": {"longest_edge": 512},
    }
    with open(directory / "preprocessor_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)


class PixtralNativeCheckpointTest(unittest.TestCase):
    """Fast (no-network) unit tests for native Mistral checkpoint support in PixtralProcessor."""

    @require_mistral_common
    @require_torchvision
    def test_from_pretrained_native_returns_mistral_backend(self):
        """from_pretrained on a native checkpoint (tekken.json + params.json) returns MistralCommonBackend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            write_fake_tekken_json(tmpdir_path)
            _write_fake_params_json(tmpdir_path)

            processor = PixtralProcessor.from_pretrained(tmpdir)

            self.assertIsInstance(processor.tokenizer, MistralCommonBackend)

    @require_torchvision
    def test_from_pretrained_mistral_format_false_yields_non_mistral(self):
        """mistral_format=False on a native checkpoint never yields MistralCommonBackend."""
        from transformers.integrations.mistral import convert_tekken_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            write_fake_tekken_json(tmpdir_path)
            _write_fake_params_json(tmpdir_path)
            _write_fake_preprocessor_config(tmpdir_path)

            # Produce a standard HF tokenizer.json from the fake tekken vocab so that
            # the TokenizersBackend fallback path (mistral_format=False) can load it.
            hf_tok = convert_tekken_tokenizer(str(tmpdir_path / "tekken.json"))
            hf_tok.save_pretrained(tmpdir)

            processor = PixtralProcessor.from_pretrained(tmpdir, mistral_format=False)

            # String comparison avoids importing MistralCommonBackend when mistral_common is absent;
            # the test is only gated on torchvision, not on mistral_common.
            self.assertNotEqual(type(processor.tokenizer).__name__, "MistralCommonBackend")

    @require_mistral_common
    @require_torchvision
    def test_from_pretrained_native_missing_params_raises(self):
        """from_pretrained on a native checkpoint without params.json raises OSError mentioning params.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            write_fake_tekken_json(tmpdir_path)
            # Deliberately do NOT write params.json

            with self.assertRaisesRegex(OSError, "params.json"):
                PixtralProcessor.from_pretrained(tmpdir)

    @require_mistral_common
    @require_torchvision
    def test_auto_processor_native_full_path(self):
        """AutoProcessor.from_pretrained on a native dir returns PixtralProcessor with MistralCommonBackend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            write_fake_tekken_json(tmpdir_path)
            _write_fake_params_json(tmpdir_path)
            # No HF-format markers (config.json / tokenizer_config.json / tokenizer.json)

            processor = AutoProcessor.from_pretrained(tmpdir)

            self.assertIsInstance(processor, PixtralProcessor)
            self.assertIsInstance(processor.tokenizer, MistralCommonBackend)

    @require_mistral_common
    @require_torchvision
    def test_both_formats_auto_prefers_native(self):
        """Auto mode prefers the native (tekken) format even when HF-format markers coexist with tekken.json.

        A both-formats checkpoint (tekken.json + params.json + saved HF tokenizer) loaded via
        ``PixtralProcessor.from_pretrained`` (auto) must yield a ``MistralCommonBackend`` tokenizer
        because auto-detection is tekken-first: if ``mistral-common`` is available and ``tekken.json``
        is present, native format wins regardless of HF-format markers.
        """
        from transformers.integrations.mistral import convert_tekken_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            write_fake_tekken_json(tmpdir_path)
            _write_fake_params_json(tmpdir_path)
            _write_fake_preprocessor_config(tmpdir_path)

            # Produce HF tokenizer files (tokenizer.json + tokenizer_config.json) — HF-format markers
            hf_tok = convert_tekken_tokenizer(str(tmpdir_path / "tekken.json"))
            hf_tok.save_pretrained(tmpdir)

            processor = PixtralProcessor.from_pretrained(tmpdir)  # auto: mistral_format=None

            self.assertIsInstance(processor.tokenizer, MistralCommonBackend)

    @require_mistral_common
    @require_torchvision
    def test_explicit_native_wins_over_hf_markers(self):
        """mistral_format=True on a both-formats checkpoint still yields MistralCommonBackend.

        Explicit opt-in to native format must take precedence over HF-format markers even when
        ``tokenizer.json`` / ``tokenizer_config.json`` are present in the directory.
        """
        from transformers.integrations.mistral import convert_tekken_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            write_fake_tekken_json(tmpdir_path)
            _write_fake_params_json(tmpdir_path)
            _write_fake_preprocessor_config(tmpdir_path)

            # Write HF tokenizer markers alongside the native files
            hf_tok = convert_tekken_tokenizer(str(tmpdir_path / "tekken.json"))
            hf_tok.save_pretrained(tmpdir)

            processor = PixtralProcessor.from_pretrained(tmpdir, mistral_format=True)

            self.assertIsInstance(processor.tokenizer, MistralCommonBackend)

    # ------------------------------------------------------------------
    # AutoProcessor explicit-flag tests (Fix: AutoProcessor honors mistral_format)
    # ------------------------------------------------------------------

    @require_mistral_common
    @require_torchvision
    def test_auto_processor_explicit_true_native_dir(self):
        """AutoProcessor.from_pretrained with mistral_format=True on a native dir returns MistralCommonBackend.

        Native dir has only tekken.json + params.json (no config.json, no HF markers), so
        AutoProcessor falls through to the OSError handler and probes resolve_mistral_format.
        The explicit True must be forwarded to PixtralProcessor.from_pretrained.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            write_fake_tekken_json(tmpdir_path)
            _write_fake_params_json(tmpdir_path)

            processor = AutoProcessor.from_pretrained(tmpdir, mistral_format=True)

            self.assertIsInstance(processor, PixtralProcessor)
            self.assertIsInstance(processor.tokenizer, MistralCommonBackend)

    @require_mistral_common
    @require_torchvision
    def test_auto_processor_explicit_false_yields_non_mistral(self):
        """AutoProcessor.from_pretrained with mistral_format=False on a both-formats dir returns a
        non-MistralCommonBackend tokenizer, confirming the explicit False is honored end-to-end.

        A `preprocessor_config.json` with `processor_class` is used so AutoProcessor can route
        to PixtralProcessor before hitting the config-probe OSError handler, allowing the
        mistral_format=False kwarg to be forwarded cleanly via the standard kwargs path.
        """
        from transformers.integrations.mistral import convert_tekken_tokenizer

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            write_fake_tekken_json(tmpdir_path)
            _write_fake_params_json(tmpdir_path)

            # Write HF tokenizer markers
            hf_tok = convert_tekken_tokenizer(str(tmpdir_path / "tekken.json"))
            hf_tok.save_pretrained(tmpdir)

            # preprocessor_config.json with explicit processor_class lets AutoProcessor route
            # to PixtralProcessor via the early-detection path (no OSError needed).
            import json

            preprocessor_cfg = {
                "image_processor_type": "PixtralImageProcessor",
                "processor_class": "PixtralProcessor",
                "patch_size": {"height": 16, "width": 16},
                "size": {"longest_edge": 512},
            }
            with open(tmpdir_path / "preprocessor_config.json", "w", encoding="utf-8") as f:
                json.dump(preprocessor_cfg, f)

            processor = AutoProcessor.from_pretrained(tmpdir, mistral_format=False)

            self.assertIsInstance(processor, PixtralProcessor)
            self.assertNotIsInstance(processor.tokenizer, MistralCommonBackend)
