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

"""Testing suite for the Isaac processor."""

import os
import re
import unittest
from pathlib import Path

import pytest
import torch
from huggingface_hub import is_offline_mode

from transformers import IsaacConfig, PythonBackend
from transformers.models.isaac.image_processing_isaac_fast import IsaacImageProcessorFast
from transformers.models.isaac.processing_isaac import IsaacProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_vision_available():
    from PIL import Image
else:
    Image = None


ISAAC_OUTPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "vision_patches",
    "vision_patch_attention_mask",
    "vision_token_grids",
    "vision_token_offsets",
    "vision_token_lengths",
    "vision_image_attention_mask",
}


class SimpleIsaacTokenizer(PythonBackend):
    vocab_files_names = {}
    model_input_names = ["input_ids"]

    def __init__(self):
        self._vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "<image>": 4,
            "<|image_pad|>": 5,
        }
        self._ids_to_tokens = {idx: tok for tok, idx in self._vocab.items()}
        super().__init__(
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
            additional_special_tokens=["<image>"],
            extra_special_tokens={"image_pad_token": "<|image_pad|>"},
            model_max_length=512,
        )

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text):
        clean = text.replace("\n", " ").strip()
        if not clean:
            return []

        special_tokens = sorted(
            (token for token in self._vocab if token.startswith("<") and token.endswith(">")),
            key=len,
            reverse=True,
        )
        if not special_tokens:
            return [token for token in clean.split(" ") if token]

        split_pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
        tokens = []
        for chunk in re.split(split_pattern, clean):
            if not chunk or chunk.isspace():
                continue
            if chunk in self._vocab:
                tokens.append(chunk)
            else:
                tokens.extend(token for token in chunk.split(" ") if token)
        return tokens

    def _convert_token_to_id(self, token):
        if token not in self._vocab:
            next_id = len(self._vocab)
            self._vocab[token] = next_id
            self._ids_to_tokens[next_id] = token
        return self._vocab[token]

    def _convert_id_to_token(self, index):
        return self._ids_to_tokens.get(index, self.unk_token)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1
        return [self.bos_token_id] + list(token_ids_0) + [self.eos_token_id]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()


class SimpleIsaacTokenizerWithNamedImagePad(PythonBackend):
    vocab_files_names = {}
    model_input_names = ["input_ids"]

    def __init__(self):
        self._vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "<image>": 4,
            "<custom_image_pad>": 5,
            "<|image_pad|>": 6,
        }
        self._ids_to_tokens = {idx: tok for tok, idx in self._vocab.items()}
        super().__init__(
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
            extra_special_tokens={"image_pad_token": "<custom_image_pad>"},
            model_max_length=512,
        )

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text):
        clean = text.replace("\n", " ").strip()
        if not clean:
            return []

        special_tokens = sorted(
            (token for token in self._vocab if token.startswith("<") and token.endswith(">")),
            key=len,
            reverse=True,
        )
        split_pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"
        tokens = []
        for chunk in re.split(split_pattern, clean):
            if not chunk or chunk.isspace():
                continue
            if chunk in self._vocab:
                tokens.append(chunk)
            else:
                tokens.extend(token for token in chunk.split(" ") if token)
        return tokens

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, self._vocab["<unk>"])

    def _convert_id_to_token(self, index):
        return self._ids_to_tokens.get(index, self.unk_token)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is not None:
            token_ids_0 = token_ids_0 + token_ids_1
        return [self.bos_token_id] + list(token_ids_0) + [self.eos_token_id]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()


def _make_dummy_image(size=(32, 32), color=(255, 0, 0)):
    if Image is None:
        raise RuntimeError("PIL.Image is not available in this environment.")
    return Image.new("RGB", size, color=color)


def _make_processor_with_max_len(tokenizer, base_config, max_len):
    config = IsaacConfig(**base_config.to_dict())
    config.max_sequence_length = max_len
    vision_config = config.vision_config
    image_processor = IsaacImageProcessorFast(
        patch_size=vision_config.patch_size,
        max_num_patches=vision_config.num_patches,
        pixel_shuffle_scale=vision_config.pixel_shuffle_scale_factor,
        rescale_factor=config.vision_rescale_factor,
    )
    return IsaacProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        vision_token=config.vision_token,
        max_sequence_length=config.max_sequence_length,
    )


def _run_processor(processor, text, images=None):
    return processor(text=text, images=images, return_tensors="pt")


def _make_post_process_processor():
    return IsaacProcessor(image_processor=IsaacImageProcessorFast(), tokenizer=SimpleIsaacTokenizer())


def test_processor_prefers_named_image_pad_token():
    processor = IsaacProcessor(
        image_processor=IsaacImageProcessorFast(), tokenizer=SimpleIsaacTokenizerWithNamedImagePad()
    )

    assert processor.image_token == "<custom_image_pad>"
    assert processor.image_pad_token_id == processor.tokenizer.image_pad_token_id
    assert processor.image_pad_token_id != processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")


def _assert_common(outputs, batch_size=1):
    assert set(outputs.keys()) == ISAAC_OUTPUT_KEYS

    input_ids = outputs["input_ids"]
    attention_mask = outputs["attention_mask"]
    mm_token_type_ids = outputs["mm_token_type_ids"]
    vision_patches = outputs["vision_patches"]
    vision_patch_attention_mask = outputs["vision_patch_attention_mask"]
    vision_token_grids = outputs["vision_token_grids"]
    vision_token_offsets = outputs["vision_token_offsets"]
    vision_token_lengths = outputs["vision_token_lengths"]
    vision_image_attention_mask = outputs["vision_image_attention_mask"]

    assert input_ids.shape[0] == batch_size
    assert attention_mask.shape == input_ids.shape
    assert mm_token_type_ids.shape == input_ids.shape
    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.long
    assert mm_token_type_ids.dtype == torch.long

    assert vision_patches.shape[:2] == vision_patch_attention_mask.shape[:2]
    assert vision_patches.shape[0] == batch_size
    assert vision_token_grids.shape == (batch_size, vision_patches.shape[1], 2)
    assert vision_token_offsets.shape == (batch_size, vision_patches.shape[1])
    assert vision_token_lengths.shape == (batch_size, vision_patches.shape[1])
    assert vision_image_attention_mask.shape == (batch_size, vision_patches.shape[1])

    return outputs


def _assert_no_vision(outputs, batch_index=0):
    assert outputs["vision_patch_attention_mask"][batch_index].sum().item() == 0
    assert outputs["vision_token_grids"][batch_index].sum().item() == 0
    assert outputs["vision_token_offsets"][batch_index].sum().item() == 0
    assert outputs["vision_token_lengths"][batch_index].sum().item() == 0
    assert outputs["vision_image_attention_mask"][batch_index].sum().item() == 0
    assert not outputs["mm_token_type_ids"][batch_index].eq(1).any()


def _assert_vision_segments(outputs, expected_segments, batch_index=0):
    active_segments = int(outputs["vision_image_attention_mask"][batch_index].sum().item())
    assert active_segments == expected_segments
    assert torch.all(outputs["vision_token_lengths"][batch_index, :expected_segments] > 0)
    assert torch.all(outputs["vision_patch_attention_mask"][batch_index, :expected_segments].sum(dim=-1) > 0)


def _count_modality(outputs, modality_value, batch_index=0):
    return int(
        (outputs["attention_mask"][batch_index].bool() & outputs["mm_token_type_ids"][batch_index].eq(modality_value))
        .sum()
        .item()
    )


def _get_active_vision_grids(outputs, batch_index=0):
    mask = outputs["vision_image_attention_mask"][batch_index].bool()
    return outputs["vision_token_grids"][batch_index][mask]


def _get_active_vision_lengths(outputs, batch_index=0):
    mask = outputs["vision_image_attention_mask"][batch_index].bool()
    return outputs["vision_token_lengths"][batch_index][mask]


@pytest.fixture
def isaac_tiny_config():
    text_config = {
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "hidden_act": "silu",
        "head_dim": 32 // 4,
        "hidden_size": 32,
        "vocab_size": 99,
        "intermediate_size": 32 * 3,
        "max_position_embeddings": 128,
        "model_type": "qwen3",
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        "tie_word_embeddings": True,
    }

    vision_config = {
        "hidden_size": 32,
        "intermediate_size": 32 * 2,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_channels": 3,
        "num_patches": 64,
        "patch_size": 4,
        "pixel_shuffle_scale_factor": 1,
        "attention_dropout": 0.0,
        "layer_norm_eps": 1e-6,
    }

    config = IsaacConfig(text_config=text_config, vision_config=vision_config)
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    config.vision_attn_implementation = "sdpa"
    return config


@pytest.fixture
def isaac_tokenizer():
    return SimpleIsaacTokenizer()


@pytest.fixture
def isaac_processor(isaac_tokenizer, isaac_tiny_config):
    vision_config = isaac_tiny_config.vision_config
    image_processor = IsaacImageProcessorFast(
        patch_size=vision_config.patch_size,
        max_num_patches=vision_config.num_patches,
        pixel_shuffle_scale=vision_config.pixel_shuffle_scale_factor,
        rescale_factor=isaac_tiny_config.vision_rescale_factor,
    )
    return IsaacProcessor(
        image_processor=image_processor,
        tokenizer=isaac_tokenizer,
        vision_token=isaac_tiny_config.vision_token,
        max_sequence_length=isaac_tiny_config.max_sequence_length,
    )


BASE_MODEL_ID = os.environ.get("ISAAC_TEST_MODEL_ID", "PerceptronAI/Isaac-0.1-Base")
BASE_MODEL_REVISION = os.environ.get("ISAAC_TEST_MODEL_REVISION", "refs/pr/3") or None
LOCAL_CHECKPOINT = os.environ.get("ISAAC_TEST_MODEL_PATH")


def _checkpoint_or_skip(model_id=BASE_MODEL_ID):
    if LOCAL_CHECKPOINT:
        resolved = Path(LOCAL_CHECKPOINT).expanduser()
        if not resolved.exists():
            pytest.skip(f"Local checkpoint path {resolved} does not exist.")
        return str(resolved)
    if is_offline_mode():
        pytest.skip("Offline mode: set ISAAC_TEST_MODEL_PATH to a local checkpoint to run these tests.")
    return model_id


@require_torch
@require_vision
class IsaacProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = IsaacProcessor
    model_id = BASE_MODEL_ID
    images_input_name = "vision_patches"

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        checkpoint = _checkpoint_or_skip(model_id)
        return super()._setup_from_pretrained(
            checkpoint,
            revision=BASE_MODEL_REVISION,
            patch_size=4,
            max_num_patches=4,
            **kwargs,
        )

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.vision_token
        cls.pad_token_id = processor.tokenizer.pad_token_id
        cls.image_pad_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def prepare_image_inputs(self, batch_size: int | None = None, nested: bool = False):
        if batch_size is None:
            return _make_dummy_image(size=(16, 16))
        images = [_make_dummy_image(size=(16, 16), color=(50 * (i + 1), 0, 0)) for i in range(batch_size)]
        if nested:
            return [[image] for image in images]
        return images

    def test_model_input_names(self):
        processor = self.get_processor()
        inputs = processor(
            text=self.prepare_text_inputs(modalities="image"),
            images=self.prepare_image_inputs(),
            return_tensors="pt",
        )

        expected_input_names = set(processor.model_input_names) | {
            "mm_token_type_ids",
            "vision_token_offsets",
            "vision_token_lengths",
            "vision_image_attention_mask",
        }
        self.assertSetEqual(set(inputs.keys()), expected_input_names)

    @unittest.skip("IsaacProcessor expands image placeholders into image pad tokens before tokenization")
    def test_tokenizer_defaults(self):
        pass

    @unittest.skip("IsaacProcessor does not return offset mappings needed for assistant masks")
    def test_apply_chat_template_assistant_mask(self):
        pass

    def test_single_vs_batched_consistency(self):
        processor = self.get_processor()
        prompt = f"hello {processor.vision_token} world"
        image = self.prepare_image_inputs()

        single = _assert_common(processor(text=prompt, images=[image], return_tensors="pt"))
        batch = _assert_common(
            processor(text=[prompt, "short"], images=[[image], []], return_tensors="pt"), batch_size=2
        )

        single_ids = single["input_ids"].squeeze(0)
        batch_ids = batch["input_ids"][0]
        self.assertTrue(torch.equal(batch_ids[-single_ids.size(0) :], single_ids))

        image_positions = batch["mm_token_type_ids"][0].eq(1)
        if image_positions.any():
            self.assertTrue(torch.all(batch_ids[image_positions] == self.image_pad_token_id))
            self.assertTrue(torch.all(batch["attention_mask"][0][image_positions] == 1))

        _assert_vision_segments(batch, expected_segments=1, batch_index=0)
        _assert_no_vision(batch, batch_index=1)


@require_torch
@require_vision
def test_text_only_has_no_vision_fields(isaac_processor):
    outputs = _assert_common(_run_processor(isaac_processor, text="Hello, how are you?", images=None))
    _assert_no_vision(outputs)


@require_torch
def test_post_process_generation_extracts_boxes_and_cleans_text():
    processor = _make_post_process_processor()

    generated_text = (
        "No, it is not safe to cross the street. "
        '<point_box mention="traffic light" t="0.5">(808, 247), (863, 386)</point_box>'
    )

    clean_text, annotations = processor.post_process_generation(generated_text)

    assert clean_text == "No, it is not safe to cross the street."
    assert len(annotations) == 1
    box = annotations[0]
    assert box.mention == "traffic light"
    assert box.t == pytest.approx(0.5)
    assert box.top_left.x == 808
    assert box.top_left.y == 247
    assert box.bottom_right.x == 863
    assert box.bottom_right.y == 386


@require_torch
@require_vision
def test_single_image_returns_offsets_and_lengths(isaac_processor):
    vision_token = isaac_processor.vision_token
    outputs = _assert_common(
        _run_processor(
            isaac_processor, text=f"Look at this {vision_token} and describe it.", images=[_make_dummy_image()]
        )
    )
    _assert_vision_segments(outputs, expected_segments=1)

    grid_tokens = torch.prod(_get_active_vision_grids(outputs), dim=-1)
    torch.testing.assert_close(_get_active_vision_lengths(outputs), grid_tokens)
    torch.testing.assert_close(
        outputs["vision_token_offsets"][0, :1], torch.zeros_like(outputs["vision_token_offsets"][0, :1])
    )


@require_torch
@require_vision
def test_multiple_images_have_matching_offsets_lengths_and_grids(isaac_processor):
    vision_token = isaac_processor.vision_token
    images = [_make_dummy_image(color=(255, 0, 0)), _make_dummy_image(color=(0, 255, 0))]

    outputs = _assert_common(
        _run_processor(isaac_processor, text=f"First {vision_token} then {vision_token}", images=images)
    )
    _assert_vision_segments(outputs, expected_segments=2)

    grid_tokens = torch.prod(_get_active_vision_grids(outputs), dim=-1)
    torch.testing.assert_close(_get_active_vision_lengths(outputs), grid_tokens)
    torch.testing.assert_close(
        outputs["vision_token_offsets"][0, :2], torch.zeros_like(outputs["vision_token_offsets"][0, :2])
    )


@require_torch
@require_vision
def test_error_on_image_mismatch(isaac_processor):
    vision_token = isaac_processor.vision_token
    with pytest.raises(ValueError, match="one image per"):
        _run_processor(isaac_processor, text=f"{vision_token} {vision_token}", images=[_make_dummy_image()])


@require_torch
@require_vision
def test_consecutive_vision_tokens_allow_empty_text_segments(isaac_processor):
    vision_token = isaac_processor.vision_token
    images = [_make_dummy_image(), _make_dummy_image(color=(0, 0, 255))]

    outputs = _assert_common(
        _run_processor(isaac_processor, text=f"prefix {vision_token}{vision_token} suffix", images=images)
    )
    _assert_vision_segments(outputs, expected_segments=2)

    torch.testing.assert_close(
        outputs["vision_token_offsets"][0, :2], torch.zeros_like(outputs["vision_token_offsets"][0, :2])
    )
    grid_tokens = torch.prod(_get_active_vision_grids(outputs), dim=-1)
    torch.testing.assert_close(_get_active_vision_lengths(outputs), grid_tokens)


@require_torch
@require_vision
def test_device_and_dtype_consistency(isaac_processor):
    vision_token = isaac_processor.vision_token
    outputs = _assert_common(
        _run_processor(isaac_processor, text=f"Describe this {vision_token}", images=[_make_dummy_image()])
    )
    _assert_vision_segments(outputs, expected_segments=1)

    tensors = [
        outputs["input_ids"],
        outputs["attention_mask"],
        outputs["mm_token_type_ids"],
        outputs["vision_token_offsets"],
        outputs["vision_token_lengths"],
        outputs["vision_token_grids"],
    ]
    devices = {tensor.device for tensor in tensors}
    assert len(devices) == 1
    for tensor in tensors:
        assert tensor.dtype == torch.long


@require_torch
@require_vision
def test_no_crop_when_total_below_max(isaac_processor):
    vision_token = isaac_processor.vision_token
    outputs = _assert_common(
        _run_processor(isaac_processor, text=f"hello {vision_token} world", images=[_make_dummy_image()])
    )
    _assert_vision_segments(outputs, expected_segments=1)

    grid_tokens = torch.prod(_get_active_vision_grids(outputs), dim=-1)
    text_tokens = _count_modality(outputs, 0)
    assert outputs["input_ids"].shape[1] == grid_tokens.item() + text_tokens


@require_torch
@require_vision
def test_exact_fit_keeps_all_tokens(isaac_processor, isaac_tokenizer, isaac_tiny_config):
    vision_token = isaac_processor.vision_token
    text = f"hey {vision_token} there"
    image = _make_dummy_image()

    base_outputs = _assert_common(_run_processor(isaac_processor, text=text, images=[image]))
    base_length = base_outputs["input_ids"].shape[1]
    base_vision_length = _get_active_vision_lengths(base_outputs).item()

    processor = _make_processor_with_max_len(isaac_tokenizer, isaac_tiny_config, base_length)
    outputs = _assert_common(_run_processor(processor, text=text, images=[image]))

    _assert_vision_segments(outputs, expected_segments=1)
    assert outputs["input_ids"].shape[1] == base_length
    assert _get_active_vision_lengths(outputs).item() == base_vision_length


@require_torch
@require_vision
def test_crop_truncates_text_segment_only(isaac_processor, isaac_tokenizer, isaac_tiny_config):
    vision_token = isaac_processor.vision_token
    text_prefix_tokens = " ".join([f"t{i}" for i in range(8)])
    text = f"{text_prefix_tokens} {vision_token} tail end"
    image = _make_dummy_image()

    base_outputs = _assert_common(_run_processor(isaac_processor, text=text, images=[image]))
    full_text_tokens = _count_modality(base_outputs, 0)
    vision_length = _get_active_vision_lengths(base_outputs).item()

    max_len = base_outputs["input_ids"].shape[1] - 4
    processor = _make_processor_with_max_len(isaac_tokenizer, isaac_tiny_config, max_len)
    outputs = _assert_common(_run_processor(processor, text=text, images=[image]))

    _assert_vision_segments(outputs, expected_segments=1)
    assert outputs["input_ids"].shape[1] == max_len
    assert _count_modality(outputs, 0) == full_text_tokens - 4
    torch.testing.assert_close(
        outputs["vision_token_offsets"][0, :1], torch.zeros_like(outputs["vision_token_offsets"][0, :1])
    )
    assert _get_active_vision_lengths(outputs).item() == vision_length


@require_torch
@require_vision
def test_crop_cuts_through_image_segment(isaac_processor, isaac_tokenizer, isaac_tiny_config):
    vision_token = isaac_processor.vision_token
    text_before = "hi"
    text_after = "bye"
    text = f"{text_before} {vision_token} {text_after}"
    image = _make_dummy_image()

    base_outputs = _assert_common(_run_processor(isaac_processor, text=text, images=[image]))
    vision_full = _get_active_vision_lengths(base_outputs).item()
    text_before_len = len(isaac_tokenizer.encode(text_before, add_special_tokens=False))
    text_after_len = len(isaac_tokenizer.encode(text_after, add_special_tokens=False))
    total_length = vision_full + text_before_len + text_after_len

    max_len = 40
    start = total_length - max_len
    expected_offset = max(0, start - text_before_len)
    expected_length = vision_full - expected_offset

    processor = _make_processor_with_max_len(isaac_tokenizer, isaac_tiny_config, max_len)
    outputs = _assert_common(_run_processor(processor, text=text, images=[image]))

    _assert_vision_segments(outputs, expected_segments=1)
    assert outputs["input_ids"].shape[1] == max_len
    assert outputs["vision_token_offsets"][0, 0].item() == expected_offset
    assert _get_active_vision_lengths(outputs).item() == expected_length
    assert _count_modality(outputs, 0) == text_after_len


@require_torch
@require_vision
def test_crop_removes_all_vision_when_window_excludes_images(isaac_processor, isaac_tokenizer, isaac_tiny_config):
    vision_token = isaac_processor.vision_token
    text_tail = "closing"
    image = _make_dummy_image()

    tail_tokens = len(isaac_processor.tokenizer.encode(text_tail, add_special_tokens=False))
    processor = _make_processor_with_max_len(isaac_tokenizer, isaac_tiny_config, tail_tokens)
    outputs = _assert_common(_run_processor(processor, text=f"{vision_token} {text_tail}", images=[image]))

    _assert_no_vision(outputs)
    assert outputs["input_ids"].shape[1] == tail_tokens
    assert _count_modality(outputs, 0) == tail_tokens


@require_torch
@require_vision
def test_batch_outputs_match_individual_calls(isaac_processor):
    texts = ["hi", "this one is longer"]

    per_sample = [_assert_common(_run_processor(isaac_processor, text=text, images=None)) for text in texts]
    batch_outputs = _assert_common(_run_processor(isaac_processor, text=texts, images=None), batch_size=len(texts))

    pad_id = isaac_processor.pad_token_id
    for index, single_output in enumerate(per_sample):
        single_ids = single_output["input_ids"].squeeze(0)
        single_mask = single_output["attention_mask"].squeeze(0)
        single_mm = single_output["mm_token_type_ids"].squeeze(0)

        batch_ids = batch_outputs["input_ids"][index]
        batch_mask = batch_outputs["attention_mask"][index]
        batch_mm = batch_outputs["mm_token_type_ids"][index]

        single_len = single_ids.shape[0]
        assert torch.equal(batch_ids[-single_len:], single_ids)
        assert torch.equal(batch_mask[-single_len:], single_mask)
        assert torch.equal(batch_mm[-single_len:], single_mm)

        if single_len < batch_ids.shape[0]:
            pad_span = batch_ids[: batch_ids.shape[0] - single_len]
            assert torch.all(pad_span == pad_id)
            assert not torch.any(batch_mask[: batch_ids.shape[0] - single_len])

        _assert_no_vision(batch_outputs, batch_index=index)
