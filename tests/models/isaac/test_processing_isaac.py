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

import numpy as np
import pytest
import torch
from huggingface_hub import is_offline_mode

from transformers import IsaacConfig, PythonBackend
from transformers.models.isaac.image_processing_isaac import IsaacImageProcessor
from transformers.models.isaac.processing_isaac import IsaacProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.tokenization_utils_base import BatchEncoding
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
    "pixel_values",
    "image_grid_thw",
    "image_metadata",
}


def _simple_tokenizer_call(
    tokenizer,
    text,
    padding=False,
    truncation=None,
    max_length=None,
    pad_to_multiple_of=None,
    return_attention_mask=True,
    return_overflowing_tokens=False,
    return_tensors=None,
    add_special_tokens=True,
    **kwargs,
):
    texts = [text] if isinstance(text, str) else list(text)
    rows = []
    row_kinds = []
    overflow_to_sample_mapping = []

    for sample_idx, sample in enumerate(texts):
        token_ids = [tokenizer._convert_token_to_id(token) for token in tokenizer._tokenize(sample)]
        if add_special_tokens:
            token_ids = tokenizer.build_inputs_with_special_tokens(token_ids)

        kept_ids = list(token_ids)
        dropped_ids = []
        if truncation and max_length is not None and len(token_ids) > max_length:
            if tokenizer.truncation_side == "left":
                dropped_ids = token_ids[:-max_length]
                kept_ids = token_ids[-max_length:]
            else:
                kept_ids = token_ids[:max_length]
                dropped_ids = token_ids[max_length:]

        rows.append(kept_ids)
        row_kinds.append("kept")
        overflow_to_sample_mapping.append(sample_idx)

        if return_overflowing_tokens and dropped_ids:
            rows.append(dropped_ids)
            row_kinds.append("overflow")
            overflow_to_sample_mapping.append(sample_idx)

    kept_rows = [row for row, row_kind in zip(rows, row_kinds, strict=True) if row_kind == "kept"]
    target_length = None
    if padding in (True, "longest"):
        target_length = max((len(row) for row in kept_rows), default=0)
    elif padding == "max_length":
        target_length = max_length

    if target_length is not None and pad_to_multiple_of is not None and target_length % pad_to_multiple_of != 0:
        target_length = ((target_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

    padded_rows = []
    attention_masks = []
    for row, row_kind in zip(rows, row_kinds, strict=True):
        if row_kind == "kept" and target_length is not None:
            pad_len = target_length - len(row)
            if tokenizer.padding_side == "left":
                padded_row = [tokenizer.pad_token_id] * pad_len + row
                attention_mask = [0] * pad_len + [1] * len(row)
            else:
                padded_row = row + [tokenizer.pad_token_id] * pad_len
                attention_mask = [1] * len(row) + [0] * pad_len
        else:
            padded_row = row
            attention_mask = [1] * len(row)

        padded_rows.append(padded_row)
        attention_masks.append(attention_mask)

    data = {"input_ids": padded_rows}
    if return_attention_mask:
        data["attention_mask"] = attention_masks
    if return_overflowing_tokens:
        data["overflow_to_sample_mapping"] = overflow_to_sample_mapping

    return BatchEncoding(data=data, tensor_type=return_tensors)


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

    def __call__(self, text, **kwargs):
        return _simple_tokenizer_call(self, text, **kwargs)


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

    def __call__(self, text, **kwargs):
        return _simple_tokenizer_call(self, text, **kwargs)


class IsaacProcessorTestDouble(IsaacProcessor):
    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)


def _make_dummy_image(size=(32, 32), color=(255, 0, 0)):
    if Image is None:
        raise RuntimeError("PIL.Image is not available in this environment.")
    return Image.new("RGB", size, color=color)


def _make_processor_with_max_len(tokenizer, base_config, max_len):
    config = IsaacConfig(**base_config.to_dict())
    config.max_sequence_length = max_len
    vision_config = config.vision_config
    image_processor = IsaacImageProcessor(
        patch_size=vision_config.patch_size,
        max_num_patches=vision_config.num_patches,
        pixel_shuffle_scale=vision_config.pixel_shuffle_scale_factor,
        rescale_factor=config.vision_rescale_factor,
    )
    return IsaacProcessorTestDouble(
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_sequence_length=config.max_sequence_length,
    )


def _run_processor(processor, text, images=None):
    return processor(text=text, images=images, return_tensors="pt")


def _make_post_process_processor():
    return IsaacProcessorTestDouble(image_processor=IsaacImageProcessor(), tokenizer=SimpleIsaacTokenizer())


def test_processor_prefers_named_image_pad_token():
    processor = IsaacProcessorTestDouble(
        image_processor=IsaacImageProcessor(), tokenizer=SimpleIsaacTokenizerWithNamedImagePad()
    )

    assert processor.image_token == "<custom_image_pad>"
    assert processor.image_token_id == processor.tokenizer.image_pad_token_id
    assert processor.image_token_id != processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")


def _assert_common(outputs, batch_size=1):
    assert set(outputs.keys()) == ISAAC_OUTPUT_KEYS

    input_ids = outputs["input_ids"]
    attention_mask = outputs["attention_mask"]
    mm_token_type_ids = outputs["mm_token_type_ids"]
    pixel_values = outputs["pixel_values"]
    image_grid_thw = outputs["image_grid_thw"]
    image_metadata = outputs["image_metadata"]

    assert input_ids.shape[0] == batch_size
    assert attention_mask.shape == input_ids.shape
    assert mm_token_type_ids.shape == input_ids.shape
    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.long
    assert mm_token_type_ids.dtype == torch.long

    if pixel_values is None:
        assert image_grid_thw is None
        assert image_metadata is None
    else:
        assert pixel_values.ndim == 4
        assert image_grid_thw.shape == (batch_size, pixel_values.shape[1], 3)
        assert image_metadata.shape == (batch_size, pixel_values.shape[1], 2)
        assert image_grid_thw.dtype == torch.long
        assert image_metadata.dtype == torch.long

        active_slots = image_grid_thw[..., 0].eq(1)
        assert torch.all(image_grid_thw[~active_slots].eq(0))
        if active_slots.any():
            assert torch.all(image_grid_thw[active_slots, 1:] > 0)
            assert torch.all(image_metadata[active_slots] >= 0)

    return outputs


def _get_sample_image_mask(outputs, batch_index=0):
    image_grid_thw = outputs["image_grid_thw"]
    if image_grid_thw is None:
        return torch.zeros((0,), dtype=torch.bool)
    return image_grid_thw[batch_index, :, 0].eq(1)


def _assert_no_vision(outputs, batch_index=0):
    assert not _get_sample_image_mask(outputs, batch_index=batch_index).any()
    assert not outputs["mm_token_type_ids"][batch_index].eq(1).any()


def _assert_vision_segments(outputs, expected_segments, batch_index=0):
    sample_image_mask = _get_sample_image_mask(outputs, batch_index=batch_index)
    active_segments = int(sample_image_mask.sum().item())
    assert active_segments == expected_segments
    assert torch.all(outputs["image_metadata"][batch_index, sample_image_mask, 1] > 0)
    assert torch.all(outputs["image_grid_thw"][batch_index, sample_image_mask, 1:].prod(dim=-1) > 0)


def _count_modality(outputs, modality_value, batch_index=0):
    return int(
        (outputs["attention_mask"][batch_index].bool() & outputs["mm_token_type_ids"][batch_index].eq(modality_value))
        .sum()
        .item()
    )


def _get_active_vision_grids(outputs, batch_index=0):
    image_grid_thw = outputs["image_grid_thw"]
    if image_grid_thw is None:
        return torch.zeros((0, 2), dtype=torch.long)
    return image_grid_thw[batch_index, _get_sample_image_mask(outputs, batch_index=batch_index), 1:]


def _get_active_vision_offsets(outputs, batch_index=0):
    image_metadata = outputs["image_metadata"]
    if image_metadata is None:
        return torch.zeros((0,), dtype=torch.long)
    return image_metadata[batch_index, _get_sample_image_mask(outputs, batch_index=batch_index), 0]


def _get_active_vision_lengths(outputs, batch_index=0):
    image_metadata = outputs["image_metadata"]
    if image_metadata is None:
        return torch.zeros((0,), dtype=torch.long)
    return image_metadata[batch_index, _get_sample_image_mask(outputs, batch_index=batch_index), 1]


def _get_expected_vision_lengths(outputs, pixel_shuffle_scale=1, batch_index=0):
    grids = _get_active_vision_grids(outputs, batch_index=batch_index)
    if grids.numel() == 0:
        return grids.new_zeros((0,))
    return torch.prod(grids, dim=-1) // (pixel_shuffle_scale**2)


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
    image_processor = IsaacImageProcessor(
        patch_size=vision_config.patch_size,
        max_num_patches=vision_config.num_patches,
        pixel_shuffle_scale=vision_config.pixel_shuffle_scale_factor,
        rescale_factor=isaac_tiny_config.vision_rescale_factor,
    )
    return IsaacProcessorTestDouble(
        image_processor=image_processor,
        tokenizer=isaac_tokenizer,
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
    processor_class = IsaacProcessorTestDouble
    model_id = BASE_MODEL_ID
    images_input_name = "pixel_values"

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
        cls.image_token = processor.image_token
        cls.pad_token_id = processor.tokenizer.pad_token_id
        cls.image_pad_token_id = processor.image_token_id

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

        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    @unittest.skip("IsaacProcessor expands image placeholders into image pad tokens before tokenization")
    def test_tokenizer_defaults(self):
        pass

    @unittest.skip("IsaacProcessor does not return offset mappings needed for assistant masks")
    def test_apply_chat_template_assistant_mask(self):
        pass

    @unittest.skip("Isaac chat templates emit <image> placeholders but the processor consumes image pad tokens")
    def test_apply_chat_template_image_0(self):
        pass

    @unittest.skip("Isaac chat templates emit <image> placeholders but the processor consumes image pad tokens")
    def test_apply_chat_template_image_1(self):
        pass

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        processor = self.get_processor()

        image_sizes = [(100, 100), (300, 100), (500, 30), (213, 167)]
        image_inputs = [np.random.randint(255, size=(h, w, 3), dtype=np.uint8) for h, w in image_sizes]

        text = [f"This is an image {self.image_token}"] * len(image_inputs)
        inputs = processor(
            text=text,
            images=[[image] for image in image_inputs],
            padding=True,
            return_mm_token_type_ids=True,
            return_tensors="pt",
        )

        num_image_tokens_from_call = inputs.mm_token_type_ids.sum(-1).tolist()
        num_image_tokens_from_helper = processor._get_num_multimodal_tokens(image_sizes=image_sizes)
        self.assertListEqual(num_image_tokens_from_call, num_image_tokens_from_helper["num_image_tokens"])

    def test_single_vs_batched_consistency(self):
        processor = self.get_processor()
        prompt = f"hello {processor.image_token} world"
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

        single_image_mask = _get_sample_image_mask(single, batch_index=0)
        batch_image_mask = _get_sample_image_mask(batch, batch_index=0)
        torch.testing.assert_close(
            batch["pixel_values"][0, batch_image_mask],
            single["pixel_values"][0, single_image_mask],
        )
        torch.testing.assert_close(
            batch["image_grid_thw"][0, batch_image_mask],
            single["image_grid_thw"][0, single_image_mask],
        )
        torch.testing.assert_close(
            batch["image_metadata"][0, batch_image_mask],
            single["image_metadata"][0, single_image_mask],
        )

        _assert_vision_segments(batch, expected_segments=1, batch_index=0)
        _assert_no_vision(batch, batch_index=1)


@require_torch
@require_vision
def test_text_only_has_no_vision_fields(isaac_processor):
    outputs = _assert_common(_run_processor(isaac_processor, text="Hello, how are you?", images=None))
    assert outputs["pixel_values"] is None
    assert outputs["image_grid_thw"] is None
    assert outputs["image_metadata"] is None
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
def test_post_process_generation_extracts_polygons_and_filters_by_expected_type():
    processor = _make_post_process_processor()

    generated_text = (
        'Point <point mention="cone">(1, 2)</point> '
        'Box <point_box mention="sign">(3, 4), (5, 6)</point_box> '
        'Polygon <polygon mention="lane" t="0.25">(10, 20), (30, 40), (50, 60)</polygon>'
    )

    clean_text, annotations = processor.post_process_generation(generated_text, expected="polygon")

    assert clean_text == "Point Box Polygon"
    assert len(annotations) == 1
    polygon = annotations[0]
    assert polygon.mention == "lane"
    assert polygon.t == pytest.approx(0.25)
    assert len(polygon.points) == 3
    assert polygon.points[0].x == 10
    assert polygon.points[0].y == 20
    assert polygon.points[1].x == 30
    assert polygon.points[1].y == 40
    assert polygon.points[2].x == 50
    assert polygon.points[2].y == 60

    _, boxes = processor.post_process_generation(generated_text, expected="box")
    assert len(boxes) == 1
    assert boxes[0].mention == "sign"


@require_torch
def test_post_process_generation_rejects_polygons_with_fewer_than_three_points():
    processor = _make_post_process_processor()

    with pytest.raises(ValueError, match=r"Malformed <polygon> tag"):
        processor.post_process_generation('<polygon mention="lane">(10, 20), (30, 40)</polygon>', expected="polygon")


@require_torch
@require_vision
def test_single_image_returns_offsets_and_lengths(isaac_processor):
    image_token = isaac_processor.image_token
    outputs = _assert_common(
        _run_processor(
            isaac_processor, text=f"Look at this {image_token} and describe it.", images=[_make_dummy_image()]
        )
    )
    _assert_vision_segments(outputs, expected_segments=1)

    grid_tokens = _get_expected_vision_lengths(outputs, isaac_processor.image_processor.pixel_shuffle_scale)
    torch.testing.assert_close(_get_active_vision_lengths(outputs), grid_tokens)
    torch.testing.assert_close(
        _get_active_vision_offsets(outputs), torch.zeros_like(_get_active_vision_offsets(outputs))
    )


@require_torch
@require_vision
def test_multiple_images_have_matching_offsets_lengths_and_grids(isaac_processor):
    image_token = isaac_processor.image_token
    images = [_make_dummy_image(color=(255, 0, 0)), _make_dummy_image(color=(0, 255, 0))]

    outputs = _assert_common(
        _run_processor(isaac_processor, text=f"First {image_token} then {image_token}", images=images)
    )
    _assert_vision_segments(outputs, expected_segments=2)

    grid_tokens = _get_expected_vision_lengths(outputs, isaac_processor.image_processor.pixel_shuffle_scale)
    torch.testing.assert_close(_get_active_vision_lengths(outputs), grid_tokens)
    torch.testing.assert_close(
        _get_active_vision_offsets(outputs), torch.zeros_like(_get_active_vision_offsets(outputs))
    )


@require_torch
@require_vision
def test_error_on_image_mismatch(isaac_processor):
    image_token = isaac_processor.image_token
    with pytest.raises(ValueError, match="one image per"):
        _run_processor(isaac_processor, text=f"{image_token} {image_token}", images=[_make_dummy_image()])


@require_torch
@require_vision
def test_consecutive_vision_tokens_allow_empty_text_segments(isaac_processor):
    image_token = isaac_processor.image_token
    images = [_make_dummy_image(), _make_dummy_image(color=(0, 0, 255))]

    outputs = _assert_common(
        _run_processor(isaac_processor, text=f"prefix {image_token}{image_token} suffix", images=images)
    )
    _assert_vision_segments(outputs, expected_segments=2)

    torch.testing.assert_close(
        _get_active_vision_offsets(outputs), torch.zeros_like(_get_active_vision_offsets(outputs))
    )
    grid_tokens = _get_expected_vision_lengths(outputs, isaac_processor.image_processor.pixel_shuffle_scale)
    torch.testing.assert_close(_get_active_vision_lengths(outputs), grid_tokens)


@require_torch
@require_vision
def test_device_and_dtype_consistency(isaac_processor):
    image_token = isaac_processor.image_token
    outputs = _assert_common(
        _run_processor(isaac_processor, text=f"Describe this {image_token}", images=[_make_dummy_image()])
    )
    _assert_vision_segments(outputs, expected_segments=1)

    tensors = [
        outputs["input_ids"],
        outputs["attention_mask"],
        outputs["mm_token_type_ids"],
        outputs["image_grid_thw"],
        outputs["image_metadata"],
    ]
    devices = {tensor.device for tensor in tensors}
    assert len(devices) == 1
    for tensor in tensors:
        assert tensor.dtype == torch.long


@require_torch
@require_vision
def test_no_crop_when_total_below_max(isaac_processor):
    image_token = isaac_processor.image_token
    outputs = _assert_common(
        _run_processor(isaac_processor, text=f"hello {image_token} world", images=[_make_dummy_image()])
    )
    _assert_vision_segments(outputs, expected_segments=1)

    grid_tokens = _get_expected_vision_lengths(outputs, isaac_processor.image_processor.pixel_shuffle_scale)
    text_tokens = _count_modality(outputs, 0)
    assert outputs["input_ids"].shape[1] == grid_tokens.item() + text_tokens


@require_torch
@require_vision
def test_exact_fit_keeps_all_tokens(isaac_processor, isaac_tokenizer, isaac_tiny_config):
    image_token = isaac_processor.image_token
    text = f"hey {image_token} there"
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
    image_token = isaac_processor.image_token
    text_prefix_tokens = " ".join([f"t{i}" for i in range(8)])
    text = f"{text_prefix_tokens} {image_token} tail end"
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
        _get_active_vision_offsets(outputs), torch.zeros_like(_get_active_vision_offsets(outputs))
    )
    assert _get_active_vision_lengths(outputs).item() == vision_length


@require_torch
@require_vision
def test_crop_cuts_through_image_segment(isaac_processor, isaac_tokenizer, isaac_tiny_config):
    image_token = isaac_processor.image_token
    text_before = "hi"
    text_after = "bye"
    text = f"{text_before} {image_token} {text_after}"
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
    assert _get_active_vision_offsets(outputs).item() == expected_offset
    assert _get_active_vision_lengths(outputs).item() == expected_length
    assert _count_modality(outputs, 0) == text_after_len


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
