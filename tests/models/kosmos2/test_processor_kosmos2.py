# Copyright 2023 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
import shutil
import tempfile
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import requests

from transformers.models.auto.processing_auto import processor_class_from_name
from transformers.testing_utils import (
    get_tests_dir,
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    require_vision,
)
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image

    from transformers import (
        AutoProcessor,
        CLIPImageProcessor,
        Kosmos2Processor,
        PreTrainedTokenizerFast,
        XLMRobertaTokenizer,
        XLMRobertaTokenizerFast,
    )


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
@require_vision
class Kosmos2ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Kosmos2Processor

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()

        image_processor = CLIPImageProcessor(do_center_crop=False)

        # We have a SentencePiece fixture for testing
        slow_tokenizer = XLMRobertaTokenizer(SAMPLE_VOCAB)
        fast_tokenizer = XLMRobertaTokenizerFast(__slow_tokenizer=slow_tokenizer)

        processor = Kosmos2Processor(image_processor, fast_tokenizer)
        processor.save_pretrained(cls.tmpdirname)

    # We override this method to take the fast tokenizer by default
    def get_component(self, attribute, **kwargs):
        assert attribute in self.processor_class.attributes
        component_class_name = getattr(self.processor_class, f"{attribute}_class")
        if isinstance(component_class_name, tuple):
            if attribute == "image_processor":
                component_class_name = component_class_name[0]
            else:
                component_class_name = component_class_name[-1]

        component_class = processor_class_from_name(component_class_name)
        component = component_class.from_pretrained(self.tmpdirname, **kwargs)  # noqa
        if attribute == "tokenizer" and not component.pad_token:
            component.pad_token = "[TEST_PAD]"

        return component

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname, ignore_errors=True)

    def test_image_procesor_load_save_reload(self):
        # make sure load from Hub repo. -> save -> reload locally work
        image_processor = CLIPImageProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        with TemporaryDirectory() as tmp_dir:
            image_processor.save_pretrained(tmp_dir)
            reloaded_image_processor = CLIPImageProcessor.from_pretrained(tmp_dir)
            assert image_processor.to_dict() == reloaded_image_processor.to_dict()
            assert image_processor.to_json_string() == reloaded_image_processor.to_json_string()

    def test_save_load_pretrained_additional_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            processor = Kosmos2Processor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
            processor.save_pretrained(tmpdir)

            tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
            image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

            processor = Kosmos2Processor.from_pretrained(
                tmpdir, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
            )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, PreTrainedTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, CLIPImageProcessor)

    def test_image_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)

        image_input = self.prepare_image_inputs()

        input_image_processor = image_processor(image_input, return_tensors="np")
        input_processor = processor(images=image_input, return_tensors="np")

        for key in input_image_processor:
            self.assertAlmostEqual(input_image_processor[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "This is a test"

        encoded_processor = processor(text=input_str, add_eos_token=True)

        encoded_tok = tokenizer(input_str, return_token_type_ids=False)

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "This is a test"
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)

        self.assertListEqual(
            list(inputs.keys()), ["pixel_values", "input_ids", "attention_mask", "image_embeds_position_mask"]
        )

        # test if it raises when no input is passed
        with pytest.raises(ValueError):
            processor()

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = Kosmos2Processor(tokenizer=tokenizer, image_processor=image_processor)

        input_str = "This is a test"
        image_input = self.prepare_image_inputs()

        # both image and text
        inputs = processor(text=input_str, images=image_input)
        self.assertListEqual(
            list(inputs.keys()), ["pixel_values", "input_ids", "attention_mask", "image_embeds_position_mask"]
        )

        # only text
        inputs = processor(text=input_str)
        self.assertListEqual(list(inputs.keys()), ["input_ids", "attention_mask"])

        # only image
        inputs = processor(images=image_input)
        self.assertListEqual(list(inputs.keys()), ["pixel_values"])

    @require_torch
    def test_full_processor(self):
        url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/two_dogs.jpg"

        processor = Kosmos2Processor.from_pretrained("microsoft/kosmos-2-patch14-224")

        # test with different input formats.
        # fmt: off
        texts = [
            # no phrase
            "<grounding> Two puppies sit in a field of grass.",
            # 1 phrase
            "<grounding> <phrase> Two puppies </phrase> sit in a field of grass.",
            # 2 phrases
            "<grounding> <phrase> Two puppies </phrase> sit in a field of <phrase> grass </phrase>.",
            # 2 phrases:  bboxes already specified for the 1st phrase
            "<grounding> <phrase> Two puppies </phrase> <object> <patch_index_0079> <patch_index_1016> </delimiter_of_multi_objects/> <patch_index_0135> <patch_index_1008> </object> sit in a field of <phrase> grass </phrase>.",
        ]
        # fmt: on

        image = Image.open(requests.get(url, stream=True).raw)
        # To match the official (microsoft) Kosmos-2 demo from which the expected values here are grabbed
        image_path = os.path.join(self.tmpdirname, "image.jpg")
        image.save(image_path)
        image = Image.open(image_path)

        # fmt: off
        bboxes = [
            [None, []],
            [[None], [[]], [(79, 1016)], [[(79, 1016)]], [[(79, 1016), (135, 1008)]]],
            [[[(79, 1016), (135, 1008)], None], [[(79, 1016), (135, 1008)], []], [[(79, 1016), (135, 1008)], (480, 1023)], [[(79, 1016), (135, 1008)], [(480, 1023)]]],
            [[None, [(480, 1023)]]],
        ]
        # fmt: on

        batch_image = [image] * 4
        batch_text = [texts[0], texts[1], texts[1], texts[2]]
        batch_bboxes = [
            None,  # no phrase
            [[]],  # 1 phrase: no bbox
            [(79, 1016)],  # 1 phrase: 1 bbox
            [[(79, 1016), (135, 1008)], (480, 1023)],  # 2 phrase: 2 bboxes + 1 bbox
        ]

        # fmt: off
        expected_input_ids = [
            [0, 64012, 1264, 17772, 1357, 12, 10, 770, 9, 4464, 4, 2],
            [0, 64012, 64007, 1264, 17772, 64008, 1357, 12, 10, 770, 9, 4464, 4, 2],
            [0, 64012, 64007, 1264, 17772, 64008, 64009, 64092, 65029, 64010, 1357, 12, 10, 770, 9, 4464, 4, 2],
            [0, 64012, 64007, 1264, 17772, 64008, 64009, 64092, 65029, 64011, 64148, 65021, 64010, 1357, 12, 10, 770, 9, 4464, 4, 2],
            [0, 64012, 64007, 1264, 17772, 64008, 64009, 64092, 65029, 64011, 64148, 65021, 64010, 1357, 12, 10, 770, 9, 64007, 4464, 64008, 106, 4, 2],
            [0, 64012, 64007, 1264, 17772, 64008, 64009, 64092, 65029, 64011, 64148, 65021, 64010, 1357, 12, 10, 770, 9, 64007, 4464, 64008, 64009, 64493, 65036, 64010, 106, 4, 2],
        ]
        # fmt: on

        EXPECTED_PIXEL_VALUES_1 = np.array(
            [
                [
                    [-0.6535852551460266, -0.6389868259429932, -0.6243883967399597],
                    [-0.6535852551460266, -0.6389868259429932, -0.6243883967399597],
                    [-0.6243883967399597, -0.6243883967399597, -0.5951915383338928],
                ],
                [
                    [-0.20629698038101196, -0.19128920137882233, -0.19128920137882233],
                    [-0.20629698038101196, -0.19128920137882233, -0.17628143727779388],
                    [-0.2213047444820404, -0.20629698038101196, -0.16127367317676544],
                ],
                [
                    [-0.5843556523323059, -0.5701355338096619, -0.5701355338096619],
                    [-0.5843556523323059, -0.5701355338096619, -0.5559154152870178],
                    [-0.5843556523323059, -0.5559154152870178, -0.5416953563690186],
                ],
            ]
        )
        EXPECTED_PIXEL_VALUES_2 = np.array(
            [
                [
                    [-0.4346088469028473, -0.47840413451194763, -0.7849710583686829],
                    [-0.5221993923187256, -0.5076009631156921, -0.755774199962616],
                    [-0.5221993923187256, -0.5076009631156921, -0.7411757707595825],
                ],
                [
                    [-0.2813358008861542, -0.2963435649871826, -0.431413471698761],
                    [-0.26632803678512573, -0.2963435649871826, -0.4764367938041687],
                    [-0.2213047444820404, -0.2813358008861542, -0.49144455790519714],
                ],
                [
                    [-0.5701355338096619, -0.641235888004303, -0.7549964189529419],
                    [-0.5843556523323059, -0.641235888004303, -0.7834365367889404],
                    [-0.5559154152870178, -0.641235888004303, -0.7834365367889404],
                ],
            ]
        )

        def check(texts, bboxes, expected_input_ids):
            outputs = processor(images=None, text=texts, bboxes=bboxes, add_eos_token=True)
            self.assertListEqual(outputs.input_ids, expected_input_ids)

        # no phrase
        check(texts[0], bboxes[0][0], expected_input_ids[0])

        # no phrase
        check(texts[0], bboxes[0][1], expected_input_ids[0])

        # 1 phrase: no bbox
        check(texts[1], bboxes[1][0], expected_input_ids[1])

        # 1 phrase: no bbox
        check(texts[1], bboxes[1][1], expected_input_ids[1])

        # 1 phrase: 1 bbox
        check(texts[1], bboxes[1][2], expected_input_ids[2])

        # 1 phrase: 1 bbox
        check(texts[1], bboxes[1][3], expected_input_ids[2])

        # 1 phrase: 2 bboxes
        check(texts[1], bboxes[1][4], expected_input_ids[3])

        # could not contain `[None]`
        with pytest.raises(ValueError):
            _ = processor.preprocess_examples(images=None, texts=texts[1], bboxes=[[None]])

        # 2 phrase: 2 bboxes + no bbox
        check(texts[2], bboxes[2][0], expected_input_ids[4])

        # 2 phrase: 2 bboxes + no bbox
        check(texts[2], bboxes[2][1], expected_input_ids[4])

        # 2 phrase: 2 bboxes + 1 bbox
        check(texts[2], bboxes[2][2], expected_input_ids[5])

        # 2 phrase: 2 bboxes + 1 bbox
        check(texts[2], bboxes[2][3], expected_input_ids[5])

        # 2 phrase: no box (as already specified in the text) + 1 bbox
        check(texts[3], bboxes[3][0], expected_input_ids[5])

        # could not contain `[None]`
        with pytest.raises(ValueError):
            _ = processor.preprocess_examples(images=None, texts=texts[2], bboxes=[[(79, 1016), (135, 1008)], [None]])

        # test batch
        outputs = processor(
            images=None,
            text=batch_text,
            bboxes=batch_bboxes,
            add_eos_token=True,
        )
        self.assertListEqual(
            outputs.input_ids,
            [expected_input_ids[0], expected_input_ids[1], expected_input_ids[2], expected_input_ids[5]],
        )

        # test batch with padding (without `return_tensors`)
        outputs = processor(
            images=None,
            text=batch_text,
            bboxes=batch_bboxes,
            padding=True,
            add_eos_token=True,
        )
        # padding on the right
        self.assertListEqual(
            outputs.input_ids[0],
            expected_input_ids[0] + [1] * (len(expected_input_ids[5]) - len(expected_input_ids[0])),
        )
        self.assertListEqual(
            outputs.attention_mask[0],
            [1] * len(expected_input_ids[0]) + [0] * (len(expected_input_ids[5]) - len(expected_input_ids[0])),
        )
        # no padding for the longest sequence
        self.assertListEqual(outputs.input_ids[-1], expected_input_ids[5])
        self.assertListEqual(outputs.attention_mask[-1], [1] * len(expected_input_ids[5]))

        # test batch with padding (with `return_tensors`)
        outputs = processor(
            images=None,
            text=batch_text,
            bboxes=batch_bboxes,
            return_tensors="pt",
            padding=True,
            add_eos_token=True,
        )
        # padding on the right
        self.assertListEqual(
            outputs.input_ids.numpy().tolist()[0],
            expected_input_ids[0] + [1] * (len(expected_input_ids[5]) - len(expected_input_ids[0])),
        )
        self.assertListEqual(
            outputs.attention_mask.numpy().tolist()[0],
            [1] * len(expected_input_ids[0]) + [0] * (len(expected_input_ids[5]) - len(expected_input_ids[0])),
        )
        # no padding for the longest sequence
        self.assertListEqual(outputs.input_ids.numpy().tolist()[-1], expected_input_ids[5])
        self.assertListEqual(outputs.attention_mask.numpy().tolist()[-1], [1] * len(expected_input_ids[5]))

        # test with image
        num_image_tokens = 64

        outputs = processor(images=image, text=texts[0], bboxes=None, add_eos_token=True)
        self.assertTupleEqual(outputs.pixel_values[0].shape, (3, 224, 224))
        self.assertListEqual(
            outputs.input_ids,
            [0, 64003] + list(range(4, 4 + num_image_tokens)) + [64004] + expected_input_ids[0][1:],
        )
        self.assertListEqual(
            outputs.image_embeds_position_mask,
            [0] * 2 + [1] * num_image_tokens + [0] + [0] * (len(expected_input_ids[0]) - 1),
        )
        np.testing.assert_allclose(outputs.pixel_values[0][:3, :3, :3], EXPECTED_PIXEL_VALUES_1, atol=1e-9)
        np.testing.assert_allclose(outputs.pixel_values[0][:3, -3:, -3:], EXPECTED_PIXEL_VALUES_2, atol=1e-9)

        # test with image in batch (right padding)
        outputs = processor(
            images=batch_image,
            text=batch_text,
            bboxes=batch_bboxes,
            return_tensors="pt",
            padding=True,
            add_eos_token=True,
        )
        self.assertTupleEqual(outputs.pixel_values.shape, (4, 3, 224, 224))
        np.testing.assert_allclose(
            outputs.pixel_values[:, :3, :3, :3].numpy(), [EXPECTED_PIXEL_VALUES_1] * len(batch_image), atol=1e-9
        )
        np.testing.assert_allclose(
            outputs.pixel_values[:, :3, -3:, -3:].numpy(), [EXPECTED_PIXEL_VALUES_2] * len(batch_image), atol=1e-9
        )
        # padding on the right: the `[1:]` below is because the part for `BOS` is already added in the beginning of each (dynamically computed) expected value  # noqa
        # fmt: off
        EXPECTED_IDS_BATCH_RIGHT_PADDING = [
            [0, 64003] + list(range(4, 4 + num_image_tokens)) + [64004] + expected_input_ids[0][1:] + [1] * (len(expected_input_ids[5]) - len(expected_input_ids[0])),
            [0, 64003] + list(range(4, 4 + num_image_tokens)) + [64004] + expected_input_ids[5][1:],
        ]
        EXPECTED_MASK_BATCH_RIGHT_PADDING = [
            [1, 1] + [1] * num_image_tokens + [1] + [1] * len(expected_input_ids[0][1:]) + [0] * (len(expected_input_ids[5]) - len(expected_input_ids[0])),
            [1] * (2 + num_image_tokens + len(expected_input_ids[5])),
        ]
        # fmt: on
        self.assertListEqual(outputs.input_ids.numpy().tolist()[0], EXPECTED_IDS_BATCH_RIGHT_PADDING[0])
        self.assertListEqual(outputs.attention_mask.numpy().tolist()[0], EXPECTED_MASK_BATCH_RIGHT_PADDING[0])
        self.assertListEqual(outputs.input_ids.numpy().tolist()[-1], EXPECTED_IDS_BATCH_RIGHT_PADDING[-1])
        self.assertListEqual(outputs.attention_mask.numpy().tolist()[-1], EXPECTED_MASK_BATCH_RIGHT_PADDING[-1])
        self.assertListEqual(
            outputs.image_embeds_position_mask.numpy().tolist(),
            [[0, 0] + [1] * num_image_tokens + [0] + [0] * (len(expected_input_ids[5]) - 1)] * len(batch_image),
        )

        processor = Kosmos2Processor.from_pretrained("microsoft/kosmos-2-patch14-224", padding_side="left")

        # test with image in batch (left padding)
        outputs = processor(
            images=batch_image,
            text=batch_text,
            bboxes=batch_bboxes,
            return_tensors="pt",
            padding=True,
            add_eos_token=True,
        )
        # padding on the left: the `[1:]` below is because the part for `BOS` is already added in the beginning of each (dynamically computed) expected value  # noqa
        # fmt: off
        EXPECTED_IDS_BATCH = [
            [1] * (len(expected_input_ids[5]) - len(expected_input_ids[0])) + [0, 64003] + list(range(4, 4 + num_image_tokens)) + [64004] + expected_input_ids[0][1:],
            [0, 64003] + list(range(4, 4 + num_image_tokens)) + [64004] + expected_input_ids[5][1:],
        ]
        EXPECTED_MASK_BATCH =[
            [0] * (len(expected_input_ids[5]) - len(expected_input_ids[0])) + [1, 1] + [1] * num_image_tokens + [1] + [1] * len(expected_input_ids[0][1:]),
            [1] * (2 + num_image_tokens + len(expected_input_ids[5])),
        ]
        EXPECTED_IMG_POS_MASK_BATCH = [
            [0] * (len(expected_input_ids[5]) - len(expected_input_ids[0])) + [0, 0] + [1] * num_image_tokens + [0] + [0] * len(expected_input_ids[0][1:]),
            [0, 0] + [1] * num_image_tokens + [0] + [0] * (len(expected_input_ids[5]) - 1),
        ]
        # fmt: on

        self.assertListEqual(outputs.input_ids.numpy().tolist()[0], EXPECTED_IDS_BATCH[0])
        self.assertListEqual(outputs.attention_mask.numpy().tolist()[0], EXPECTED_MASK_BATCH[0])
        self.assertListEqual(outputs.image_embeds_position_mask.numpy().tolist()[0], EXPECTED_IMG_POS_MASK_BATCH[0])

        # no padding for the longest sequence
        self.assertListEqual(outputs.input_ids.numpy().tolist()[-1], EXPECTED_IDS_BATCH[-1])
        self.assertListEqual(outputs.attention_mask.numpy().tolist()[-1], EXPECTED_MASK_BATCH[-1])
        self.assertListEqual(outputs.image_embeds_position_mask.numpy().tolist()[-1], EXPECTED_IMG_POS_MASK_BATCH[-1])

    # Rewrite as Kosmos-2 supports custom padding only when image is None.
    @require_vision
    @require_torch
    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=117)

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        # set image input to None
        image_input = None

        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            max_length=112,
            padding="max_length",
        )

        self.assertEqual(len(inputs["input_ids"][0]), 112)

    # Rewrite to test only image_processor kwargs
    @require_torch
    @require_vision
    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"size": {"height": 214, "width": 214}},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)

        self.assertEqual(inputs["pixel_values"].shape[2], 214)

    # Rewrite to test only image_processor kwargs
    @require_torch
    @require_vision
    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")

        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"size": {"height": 214, "width": 214}},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertEqual(inputs["pixel_values"].shape[2], 214)

    # Rewrite as Kosmos-2 supports custom padding only when image is None.
    @require_vision
    @require_torch
    def test_tokenizer_defaults_preserved_by_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer", max_length=117, padding="max_length")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        # set image input to None
        image_input = None

        inputs = processor(text=input_str, images=image_input, return_tensors="pt")
        self.assertEqual(len(inputs["input_ids"][0]), 117)

    # Rewrite as Kosmos-2 supports custom padding only when image is None.
    @require_torch
    @require_vision
    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        # set image input to None
        image_input = None
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            padding="max_length",
            max_length=76,
        )

        self.assertEqual(len(inputs["input_ids"][0]), 76)

    # Rewrite as Kosmos-2 supports custom padding only when image is None.
    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=2)
        # set image input to None
        image_input = None
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            size={"height": 214, "width": 214},
            padding="longest",
            max_length=76,
        )

        self.assertEqual(len(inputs["input_ids"][0]), 10)
