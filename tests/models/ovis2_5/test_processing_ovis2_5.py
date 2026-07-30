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

import numpy as np

from transformers import AutoTokenizer, BaseVideoProcessor, Ovis2_5ImageProcessorPil, Ovis2_5Processor
from transformers.testing_utils import require_tokenizers, require_torch, require_torchvision, require_vision, slow
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.utils import is_tokenizers_available, is_torch_available, is_torchvision_available

from ...test_processing_common import ProcessorTesterMixin


if is_tokenizers_available():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from transformers import PreTrainedTokenizerFast

if is_torch_available():
    import torch

if is_torchvision_available():
    from transformers import Ovis2_5ImageProcessor, Ovis2_5VideoProcessor


VISUAL_TOKENS_AND_IDS = {
    "<ovis_visual_atom>": 151669,
    "<ovis_image_start>": 151670,
    "<ovis_image_end>": 151671,
    "<ovis_video_start>": 151672,
    "<ovis_video_end>": 151673,
}


class Ovis2_5TestTokenizer(PreTrainedTokenizerBase):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        super().__init__(pad_token="<pad>", eos_token="<eos>")
        self.init_kwargs = {}
        self._token_to_id = {**VISUAL_TOKENS_AND_IDS, "<pad>": 0, "<eos>": 2}

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.convert_tokens_to_ids(token) for token in tokens]
        return self._token_to_id.get(tokens, 1)

    def _encode(self, text):
        input_ids = []
        position = 0
        while position < len(text):
            token = next(
                (token for token in VISUAL_TOKENS_AND_IDS if text.startswith(token, position)),
                None,
            )
            if token is None:
                input_ids.append(1)
                position += 1
            else:
                input_ids.append(VISUAL_TOKENS_AND_IDS[token])
                position += len(token)
        return input_ids

    def __call__(
        self,
        text,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        return_tensors=None,
        **kwargs,
    ):
        is_batched = not isinstance(text, str)
        texts = list(text) if is_batched else [text]
        input_ids = [self._encode(sample) for sample in texts]
        if truncation and max_length is not None:
            input_ids = [sample[:max_length] for sample in input_ids]
        attention_mask = [[1] * len(sample) for sample in input_ids]
        if not is_batched:
            input_ids = input_ids[0]
            attention_mask = attention_mask[0]
        return BatchEncoding(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            tensor_type=return_tensors,
        )

    def batch_decode(self, sequences, **kwargs):
        return [""] * len(sequences)


class Ovis2_5TestVideoProcessor(BaseVideoProcessor):
    """A non-runtime component used to exercise image-only processor behavior without torchvision."""

    model_input_names = []
    merge_size = 2

    def __init__(self):
        # The dummy dependency object raises on construction when torchvision is unavailable.
        # ProcessorMixin only needs the correctly typed component for image-only calls.
        pass

    def __call__(self, videos, **kwargs):
        return {}

    def get_number_of_video_patches(self, num_frames, height, width, videos_kwargs=None):
        return 0


@require_vision
class Ovis2_5ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.image_processor = Ovis2_5ImageProcessorPil()
        self.tokenizer = Ovis2_5TestTokenizer()
        self.video_processor = Ovis2_5TestVideoProcessor()
        self.processor = Ovis2_5Processor(
            self.image_processor,
            self.tokenizer,
            self.video_processor,
        )
        self.image = np.zeros((448, 448, 3), dtype=np.uint8)
        self.video = np.zeros((1, 32, 32, 3), dtype=np.uint8)

    def test_video_processor_is_a_required_typed_component(self):
        self.assertEqual(
            Ovis2_5Processor.get_attributes(),
            ["image_processor", "tokenizer", "video_processor"],
        )
        with self.assertRaisesRegex(TypeError, "BaseVideoProcessor"):
            Ovis2_5Processor(self.image_processor, self.tokenizer, None)
        with self.assertRaises(TypeError):
            Ovis2_5Processor(self.image_processor, self.tokenizer)

    def test_official_positive_visual_token_ids(self):
        self.assertEqual(
            [
                self.processor.visual_atom_token_id,
                self.processor.image_start_token_id,
                self.processor.image_end_token_id,
                self.processor.video_start_token_id,
                self.processor.video_end_token_id,
            ],
            list(VISUAL_TOKENS_AND_IDS.values()),
        )
        self.assertTrue(all(token_id > 0 for token_id in VISUAL_TOKENS_AND_IDS.values()))

    def test_image_prompt_expansion_matches_patch_grid(self):
        output = self.processor(
            images=self.image,
            text="<image>Describe this image.",
            return_tensors=None,
        )

        self.assertEqual(output.pixel_values.shape, (784, 768))
        self.assertEqual(output.image_grid_thw.tolist(), [[1, 28, 28]])
        input_ids = output.input_ids[0]
        image_start = input_ids.index(VISUAL_TOKENS_AND_IDS["<ovis_image_start>"])
        expected_image_ids = (
            [VISUAL_TOKENS_AND_IDS["<ovis_image_start>"]]
            + [VISUAL_TOKENS_AND_IDS["<ovis_visual_atom>"]] * 196
            + [VISUAL_TOKENS_AND_IDS["<ovis_image_end>"]]
        )
        self.assertEqual(input_ids[image_start : image_start + 198], expected_image_ids)

    def test_image_placeholder_count_must_match_images(self):
        with self.assertRaisesRegex(ValueError, "image placeholders must match"):
            self.processor(images=self.image, text="Describe this image.")
        with self.assertRaisesRegex(ValueError, "image placeholders must match"):
            self.processor(text="<image>Describe this image.")
        with self.assertRaisesRegex(ValueError, "image placeholders must match"):
            self.processor(images=self.image, text="<image><image>Describe these images.")

    def test_mixed_media_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "one visual modality at a time"):
            self.processor(
                images=self.image,
                videos=self.video,
                text="<image><video>Describe this.",
            )
        with self.assertRaisesRegex(ValueError, "cannot contain both"):
            self.processor(text="<image><video>Describe this.")

    def test_processor_generated_tokens_are_rejected_in_raw_text(self):
        with self.assertRaisesRegex(ValueError, "processor-generated"):
            self.processor(text="<ovis_visual_atom>")

    @slow
    def test_official_tokenizers_receive_expected_visual_token_ids(self):
        for model_id in ("AIDC-AI/Ovis2.5-2B", "AIDC-AI/Ovis2.5-9B"):
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
            tokenizer.add_special_tokens(
                {"additional_special_tokens": list(VISUAL_TOKENS_AND_IDS)},
            )
            self.assertEqual(
                tokenizer.convert_tokens_to_ids(list(VISUAL_TOKENS_AND_IDS)),
                list(VISUAL_TOKENS_AND_IDS.values()),
            )


@require_vision
@require_torch
@require_torchvision
@require_tokenizers
class Ovis2_5ProcessorMixinTest(ProcessorTesterMixin, unittest.TestCase):
    """Run the common processor contract against fully local, serializable components."""

    processor_class = Ovis2_5Processor

    @classmethod
    def _setup_tokenizer(cls):
        vocab = {
            "<unk>": 0,
            "<pad>": 1,
            "<bos>": 2,
            "<eos>": 3,
            "<ovis_visual_atom>": 4,
            "<ovis_image_start>": 5,
            "<ovis_image_end>": 6,
            "<ovis_video_start>": 7,
            "<ovis_video_end>": 8,
            "lower": 9,
            "newer": 10,
            "upper": 11,
            "older": 12,
            "longer": 13,
            "string": 14,
        }
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        return PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<bos>",
            eos_token="<eos>",
            additional_special_tokens=list(VISUAL_TOKENS_AND_IDS),
        )

    @classmethod
    def _setup_image_processor(cls):
        return Ovis2_5ImageProcessor(
            size={"shortest_edge": 64 * 64, "longest_edge": 64 * 1024},
        )

    @classmethod
    def _setup_video_processor(cls):
        return Ovis2_5VideoProcessor(
            size={"shortest_edge": 64 * 64, "longest_edge": 64 * 1024},
        )

    def prepare_text_inputs(self, batch_size: int | None = None, modalities: str | list | None = None):
        if isinstance(modalities, str):
            modalities = [modalities]
        modalities = modalities or []
        placeholder = "<image>" if "image" in modalities else "<video>" if "video" in modalities else ""
        text = f"{placeholder} lower newer"
        if batch_size is None:
            return text
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        return [text] * batch_size

    def prepare_image_inputs(self, batch_size: int | None = None, nested: bool = False):
        image = np.zeros((3, 64, 64), dtype=np.uint8)
        if batch_size is None:
            return image
        images = [image.copy() for _ in range(batch_size)]
        return [[item] for item in images] if nested else images

    def prepare_video_inputs(self, batch_size: int | None = None):
        video = np.zeros((4, 3, 64, 64), dtype=np.uint8)
        if batch_size is None:
            return video
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        return [video.copy() for _ in range(batch_size)]

    def test_model_input_names(self):
        processor = self.get_processor()

        image_inputs = processor(
            text=self.prepare_text_inputs(modalities="image"),
            images=self.prepare_image_inputs(),
            return_tensors="pt",
        )
        self.assertSetEqual(
            set(image_inputs),
            {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"},
        )

        video_inputs = processor(
            text=self.prepare_text_inputs(modalities="video"),
            videos=self.prepare_video_inputs(),
            do_sample_frames=False,
            return_tensors="pt",
        )
        self.assertSetEqual(
            set(video_inputs),
            {"input_ids", "attention_mask", "pixel_values_videos", "video_grid_thw"},
        )
        self.assertSetEqual(set(processor.model_input_names), set(image_inputs) | set(video_inputs))

    def test_processor_with_multiple_inputs(self):
        processor = self.get_processor()

        with self.assertRaisesRegex(ValueError, "one visual modality at a time"):
            processor(
                text="<image><video> lower newer",
                images=self.prepare_image_inputs(),
                videos=self.prepare_video_inputs(),
                do_sample_frames=False,
                return_tensors="pt",
            )

    def test_flat_kwarg_applied_when_modality_dict_lacks_it(self):
        processor = self.get_processor()
        inputs = processor(
            text=self.prepare_text_inputs(modalities="image"),
            images=self.prepare_image_inputs(),
            text_kwargs={},
            return_tensors="pt",
        )
        for value in inputs.values():
            self.assertIsInstance(value, torch.Tensor)

    def test_unstructured_kwargs_batched_video(self):
        processor = self.get_processor()
        with self.assertRaisesRegex(ValueError, "exactly one video"):
            processor(
                text=self.prepare_text_inputs(batch_size=2, modalities="video"),
                videos=self.prepare_video_inputs(batch_size=2),
                do_sample_frames=False,
                return_tensors="pt",
            )

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        processor = self.get_processor()
        image_sizes = [(100, 100), (300, 100), (500, 30), (213, 167)]
        images = [np.zeros((height, width, 3), dtype=np.uint8) for height, width in image_sizes]
        inputs = processor(
            text=[f"{processor.image_token} lower newer"] * len(images),
            images=images,
            padding=True,
            return_tensors="pt",
        )

        visual_atom_counts = (inputs.input_ids == processor.visual_atom_token_id).sum(-1).tolist()
        helper_counts = processor._get_num_multimodal_tokens(image_sizes=image_sizes)["num_image_tokens"]
        self.assertListEqual(visual_atom_counts, helper_counts)

    def test_processor_text_has_no_visual(self):
        self.skipTest("Ovis2.5 deliberately accepts only one visual modality per request.")


if __name__ == "__main__":
    unittest.main()
