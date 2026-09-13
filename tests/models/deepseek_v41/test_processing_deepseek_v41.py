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
import json
import tempfile
import unittest

from transformers.testing_utils import require_torch, require_torchvision, require_vision, slow
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image

    from transformers import DeepseekV41ImageProcessor, DeepseekV41Processor


IMAGE_TOKEN = "<｜deepseek_image｜>"


def tiny_vl_tokenizer():
    """Lossless tiny byte-level tokenizer with the release's non-special added token."""
    from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

    from transformers import PreTrainedTokenizerFast

    vocab = {token: i for i, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    vocab["<pad>"] = len(vocab)
    vocab["<unk>"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_tokens([AddedToken(IMAGE_TOKEN, special=False)])
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        model_input_names=["input_ids", "attention_mask"],
    )


@require_vision
@require_torch
@require_torchvision
class DeepseekV41ProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = DeepseekV41Processor

    @classmethod
    def _setup_tokenizer(cls):
        return tiny_vl_tokenizer()

    @classmethod
    def _setup_image_processor(cls):
        return DeepseekV41ImageProcessor(min_pixels=56 * 56, max_image_tokens=64)

    @staticmethod
    def prepare_processor_dict():
        return {"image_token": IMAGE_TOKEN}

    def test_image_token_expansion(self):
        processor = self.get_processor()
        image = Image.new("RGB", (20, 40), (10, 30, 60))
        inputs = processor(text=f"describe {IMAGE_TOKEN}", images=image, return_tensors="pt")
        # Reference plan: 20x40 -> 42x84 -> 3x6 ViT grid -> 1x2 LLM grid.
        self.assertEqual(inputs.image_grid_thw.tolist(), [[1, 6, 3]])
        self.assertEqual(tuple(inputs.pixel_values.shape), (18, 3 * 14 * 14))
        prefix = processor.tokenizer.encode("describe ", add_special_tokens=False)
        self.assertEqual(inputs.input_ids[0].tolist(), prefix + [processor.image_token_id] * 6)
        # The release's image token survives skip_special_tokens=True.
        self.assertEqual(
            processor.decode(inputs.input_ids[0], skip_special_tokens=True), "describe " + IMAGE_TOKEN * 6
        )

    def test_repeated_placeholders_and_offsets(self):
        processor = self.get_processor()
        texts = [f"a{IMAGE_TOKEN}b{IMAGE_TOKEN}c", f"d{IMAGE_TOKEN}e", "text only"]
        original_texts = list(texts)
        images = [Image.new("RGB", size) for size in [(20, 40), (40, 100), (100, 40)]]
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
            return_text_replacement_offsets=True,
            return_mm_token_type_ids=True,
        )
        self.assertEqual(texts, original_texts)
        self.assertEqual(inputs.mm_token_type_ids.sum(-1).tolist(), [6 + 8, 6, 0])
        decoded = processor.batch_decode(inputs.input_ids, skip_special_tokens=True)
        self.assertEqual(
            decoded, ["a" + IMAGE_TOKEN * 6 + "b" + IMAGE_TOKEN * 8 + "c", "d" + IMAGE_TOKEN * 6 + "e", "text only"]
        )
        self.assertEqual([len(offsets) for offsets in inputs.text_replacement_offsets], [2, 1, 0])
        for before, after, offsets in zip(original_texts, decoded, inputs.text_replacement_offsets):
            for offset in offsets:
                start, end = offset["span"]
                new_start, new_end = offset["new_span"]
                self.assertEqual(before[start:end], IMAGE_TOKEN)
                self.assertEqual(after[new_start:new_end], offset["replacement"])

    def test_multimodal_counts_with_call_time_overrides(self):
        processor = self.get_processor()
        sizes = [(28, 70), (70, 28)]
        images = [Image.new("RGB", (width, height)) for height, width in sizes]
        kwargs = {"do_resize": False, "downsample_ratio": 2}
        inputs = processor(
            images=images,
            text=[IMAGE_TOKEN, IMAGE_TOKEN],
            images_kwargs=kwargs,
            return_mm_token_type_ids=True,
            padding=True,
            return_tensors="pt",
        )
        counts = processor._get_num_multimodal_tokens(image_sizes=sizes, **kwargs)
        # The row delimiters make transposed grids cost different token counts.
        self.assertEqual(counts.num_image_tokens, [6, 8])
        self.assertEqual(counts.num_image_patches, [10, 10])
        self.assertEqual(inputs.mm_token_type_ids.sum(-1).tolist(), counts.num_image_tokens)
        self.assertEqual(inputs.pixel_values.shape[0], sum(counts.num_image_patches))
        nested_counts = processor._get_num_multimodal_tokens(image_sizes=sizes, images_kwargs=kwargs)
        self.assertEqual(nested_counts.num_image_tokens, counts.num_image_tokens)

    def test_token_options_roundtrip_without_mutating_tokenizer(self):
        from tokenizers import AddedToken

        tokenizer = tiny_vl_tokenizer()
        override = "<custom_image>"
        tokenizer.add_tokens([AddedToken(override, special=False)])
        image_token_id = tokenizer.convert_tokens_to_ids(override)
        original_vocab = tokenizer.get_vocab()
        processor = DeepseekV41Processor(
            self._setup_image_processor(),
            tokenizer,
            image_token=override,
            image_token_id=image_token_id,
        )
        image = Image.new("RGB", (20, 40))
        expected = processor(text=override, images=image, return_tensors="pt")
        with tempfile.TemporaryDirectory() as directory:
            processor.save_pretrained(directory)
            reloaded = DeepseekV41Processor.from_pretrained(directory)
        actual = reloaded(text=override, images=image, return_tensors="pt")
        self.assertEqual(actual.input_ids.tolist(), expected.input_ids.tolist())
        self.assertEqual(reloaded.decode(actual.input_ids[0], skip_special_tokens=True), override * 6)
        self.assertFalse(reloaded.tokenizer.added_tokens_decoder[image_token_id].special)
        # Token options are processor state, not newly registered tokenizer aliases.
        self.assertEqual(reloaded.image_token, override)
        self.assertEqual(reloaded.image_token_id, image_token_id)
        self.assertEqual(tokenizer.get_vocab(), original_vocab)

    def test_tokenizer_alias_preserves_special_status(self):
        tokenizer = tiny_vl_tokenizer()
        tokenizer.add_special_tokens({"additional_special_tokens": ["<custom_image>"]})
        tokenizer.image_token = "<custom_image>"
        processor = DeepseekV41Processor(self._setup_image_processor(), tokenizer)
        inputs = processor(text="<custom_image>", images=Image.new("RGB", (20, 40)), return_tensors="pt")
        self.assertEqual(inputs.input_ids[0].tolist(), [tokenizer.convert_tokens_to_ids("<custom_image>")] * 6)
        self.assertEqual(processor.decode(inputs.input_ids[0], skip_special_tokens=True), "")

    def test_invalid_image_token_options(self):
        tokenizer = tiny_vl_tokenizer()
        for kwargs in ({"image_token": "<|image_pad|>"}, {"image_token_id": 129264}, {"image_token": ""}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DeepseekV41Processor(self._setup_image_processor(), tokenizer, **kwargs)

    @slow
    def test_released_tokenizer_contract(self):
        from huggingface_hub import hf_hub_download

        from transformers import PreTrainedTokenizerFast

        repo_id = "deepseek-ai/DeepSeek-V4.1-Flash"
        revision = "dba1be0a40aa45a94ad051997016db3960a90277"
        tokenizer_file = hf_hub_download(repo_id, "tokenizer.json", revision=revision)
        tokenizer_config_file = hf_hub_download(repo_id, "tokenizer_config.json", revision=revision)
        with open(tokenizer_config_file) as file:
            tokenizer_config = json.load(file)
        self.assertNotIn("image_token", tokenizer_config)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        processor = DeepseekV41Processor(self._setup_image_processor(), tokenizer)
        self.assertEqual(processor.image_token_id, 129264)
        self.assertEqual(processor.image_token, IMAGE_TOKEN)
        self.assertFalse(tokenizer.added_tokens_decoder[129264].special)
        inputs = processor(text=IMAGE_TOKEN, images=Image.new("RGB", (20, 40)), return_tensors="pt")
        self.assertEqual(inputs.input_ids[0].tolist(), [129264] * 6)
        self.assertEqual(processor.decode(inputs.input_ids[0], skip_special_tokens=True), IMAGE_TOKEN * 6)
