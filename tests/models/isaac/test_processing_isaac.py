# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import unittest
from pathlib import Path

import pytest
import torch
from huggingface_hub import is_offline_mode

from transformers import IsaacConfig, PythonBackend
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.models.isaac.image_processing_isaac_fast import IsaacImageProcessorFast
from transformers.models.isaac.modeling_isaac import ModalityType
from transformers.models.isaac.processing_isaac import IsaacProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available
from transformers.utils.generic import TensorType


if is_vision_available():
    from PIL import Image
else:
    Image = None


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
        }
        self._ids_to_tokens = {idx: tok for tok, idx in self._vocab.items()}
        super().__init__(
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
            extra_special_tokens=["<image>"],
            model_max_length=512,
        )
        self.chat_template = (
            "{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] | trim }}\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant:{% endif %}"
        )

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text):
        clean = text.replace("\n", " ").strip()
        if not clean:
            return []
        return [token for token in clean.split(" ") if token]

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
    return IsaacProcessor(image_processor=image_processor, tokenizer=tokenizer, config=config)


def _run_processor(processor, text, images=None):
    return processor(text=text, images=images, return_tensors="pt")


def _assert_common(outputs):
    assert set(outputs.keys()) == {"input_ids", "packed_inputs"}
    input_ids = outputs["input_ids"]
    packed_inputs = outputs["packed_inputs"]

    expected_packed_keys = {
        "vision_patches",
        "vision_token_grids",
        "vision_token_offsets",
        "vision_token_lengths",
        "vision_token_batch_indices",
        "modality_tensor",
        "position_ids",
    }
    assert set(packed_inputs.keys()) == expected_packed_keys

    assert input_ids.shape[0] == 1
    assert input_ids.dtype == torch.long

    modality = packed_inputs["modality_tensor"]
    position_ids = packed_inputs["position_ids"]
    assert modality.shape == (1, input_ids.shape[1])
    assert position_ids.shape == (1, input_ids.shape[1], 3)
    assert modality.dtype == torch.long
    assert position_ids.dtype == torch.long
    assert modality.device == input_ids.device == position_ids.device

    return input_ids, packed_inputs


def _assert_no_vision(packed_inputs):
    assert packed_inputs["vision_patches"] is None
    assert packed_inputs["vision_token_grids"] is None
    assert packed_inputs["vision_token_offsets"] is None
    assert packed_inputs["vision_token_lengths"] is None
    assert packed_inputs["vision_token_batch_indices"] is None


def _assert_vision_segments(packed_inputs, expected_segments):
    assert packed_inputs["vision_patches"] is not None
    assert packed_inputs["vision_token_grids"] is not None
    assert packed_inputs["vision_token_offsets"] is not None
    assert packed_inputs["vision_token_lengths"] is not None
    assert packed_inputs["vision_token_batch_indices"] is not None

    assert packed_inputs["vision_token_grids"].shape[0] == expected_segments
    assert packed_inputs["vision_token_offsets"].shape == (expected_segments,)
    assert packed_inputs["vision_token_lengths"].shape == (expected_segments,)
    assert packed_inputs["vision_token_batch_indices"].shape == (expected_segments,)


def _count_modality(packed_inputs, modality_value):
    modality = packed_inputs["modality_tensor"]
    return int((modality == modality_value).sum().item())


def _pad_to_max(tensors: list[torch.Tensor], pad_value: int) -> torch.Tensor:
    """Pad a list of (L, ...) tensors to (B, L_max, ...)."""
    max_len = max(t.shape[0] for t in tensors)
    batch = len(tensors)
    if tensors[0].ndim == 1:
        out = torch.full((batch, max_len), pad_value, device=tensors[0].device, dtype=tensors[0].dtype)
        for i, t in enumerate(tensors):
            out[i, : t.shape[0]] = t
        return out
    # assume (L, K)
    k = tensors[0].shape[1]
    out = torch.full((batch, max_len, k), pad_value, device=tensors[0].device, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.shape[0]] = t
    return out


def _get_image_token_length(processor, image, vision_token):
    outputs = _run_processor(processor, text=vision_token, images=[image])
    _, packed = _assert_common(outputs)
    _assert_vision_segments(packed, expected_segments=1)
    return packed["vision_token_lengths"][0].item()


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
        config=isaac_tiny_config,
    )


@require_torch
@require_vision
def test_isaac_processor_matches_config_defaults(isaac_processor, isaac_tiny_config):
    assert isaac_processor.vision_token == isaac_tiny_config.vision_token
    assert isaac_processor.max_sequence_length == isaac_tiny_config.max_sequence_length
    assert isaac_processor.config is isaac_tiny_config
    assert isinstance(isaac_processor.image_processor, IsaacImageProcessorFast)
    assert isaac_processor.image_processor.rescale_factor == pytest.approx(isaac_tiny_config.vision_rescale_factor)


@require_torch
@require_vision
def test_text_only_has_no_vision_fields(isaac_processor):
    outputs = _run_processor(isaac_processor, text="Hello, how are you?", images=None)
    _, packed = _assert_common(outputs)
    _assert_no_vision(packed)


@require_torch
@require_vision
def test_accepts_batchencoding_chat_template(isaac_processor):
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    batch_encoding = isaac_processor.apply_chat_template(messages, add_generation_prompt=True)

    outputs = _run_processor(isaac_processor, text=batch_encoding, images=None)
    _, packed = _assert_common(outputs)
    _assert_no_vision(packed)


@require_torch
@require_vision
def test_single_image_returns_offsets_and_lengths(isaac_processor):
    vision_token = isaac_processor.vision_token
    text = f"Look at this {vision_token} and describe it."
    image = _make_dummy_image()

    outputs = _run_processor(isaac_processor, text=text, images=[image])
    _, packed = _assert_common(outputs)
    _assert_vision_segments(packed, expected_segments=1)

    grid_tokens = torch.prod(packed["vision_token_grids"], dim=-1)
    torch.testing.assert_close(packed["vision_token_lengths"], grid_tokens)
    torch.testing.assert_close(packed["vision_token_offsets"], torch.zeros_like(packed["vision_token_offsets"]))


@require_torch
@require_vision
def test_multiple_images_have_matching_offsets_lengths_and_grids(isaac_processor):
    vision_token = isaac_processor.vision_token
    text = f"First {vision_token} then {vision_token}"
    images = [_make_dummy_image(color=(255, 0, 0)), _make_dummy_image(color=(0, 255, 0))]

    outputs = _run_processor(isaac_processor, text=text, images=images)
    _, packed = _assert_common(outputs)
    _assert_vision_segments(packed, expected_segments=2)

    grid_tokens = torch.prod(packed["vision_token_grids"], dim=-1)
    torch.testing.assert_close(packed["vision_token_lengths"], grid_tokens)
    torch.testing.assert_close(packed["vision_token_offsets"], torch.zeros_like(packed["vision_token_offsets"]))


@require_torch
@require_vision
def test_error_on_image_mismatch(isaac_processor):
    vision_token = isaac_processor.vision_token
    text = f"{vision_token} {vision_token}"
    image = _make_dummy_image()

    with pytest.raises(ValueError, match="one image per"):
        _run_processor(isaac_processor, text=text, images=[image])


@require_torch
@require_vision
def test_consecutive_vision_tokens_allow_empty_text_segments(isaac_processor):
    vision_token = isaac_processor.vision_token
    text = f"prefix {vision_token}{vision_token} suffix"
    images = [_make_dummy_image(), _make_dummy_image(color=(0, 0, 255))]

    outputs = _run_processor(isaac_processor, text=text, images=images)
    _, packed = _assert_common(outputs)
    _assert_vision_segments(packed, expected_segments=2)

    torch.testing.assert_close(packed["vision_token_offsets"], torch.zeros_like(packed["vision_token_offsets"]))
    grid_tokens = torch.prod(packed["vision_token_grids"], dim=-1)
    torch.testing.assert_close(packed["vision_token_lengths"], grid_tokens)


@require_torch
@require_vision
def test_device_and_dtype_consistency(isaac_processor):
    vision_token = isaac_processor.vision_token
    text = f"Describe this {vision_token}"
    image = _make_dummy_image()

    outputs = _run_processor(isaac_processor, text=text, images=[image])
    input_ids, packed = _assert_common(outputs)
    _assert_vision_segments(packed, expected_segments=1)

    tensors = [
        input_ids,
        packed["position_ids"],
        packed["modality_tensor"],
        packed["vision_token_offsets"],
        packed["vision_token_lengths"],
        packed["vision_token_grids"],
    ]
    devices = {t.device for t in tensors}
    assert len(devices) == 1
    for t in tensors:
        assert t.dtype == torch.long


@require_torch
@require_vision
def test_no_crop_when_total_below_max(isaac_processor):
    vision_token = isaac_processor.vision_token
    text = f"hello {vision_token} world"
    image = _make_dummy_image()

    outputs = _run_processor(isaac_processor, text=text, images=[image])
    input_ids, packed = _assert_common(outputs)
    _assert_vision_segments(packed, expected_segments=1)

    grid_tokens = torch.prod(packed["vision_token_grids"], dim=-1)
    text_tokens = _count_modality(packed, ModalityType.text.value)
    assert input_ids.shape[1] == grid_tokens.item() + text_tokens


@require_torch
@require_vision
def test_exact_fit_keeps_all_tokens(isaac_processor, isaac_tokenizer, isaac_tiny_config):
    vision_token = isaac_processor.vision_token
    text = f"hey {vision_token} there"
    image = _make_dummy_image()

    base_outputs = _run_processor(isaac_processor, text=text, images=[image])
    base_length = base_outputs["input_ids"].shape[1]
    base_packed = base_outputs["packed_inputs"]
    base_vision_length = base_packed["vision_token_lengths"][0].item()

    processor = _make_processor_with_max_len(isaac_tokenizer, isaac_tiny_config, base_length)
    outputs = _run_processor(processor, text=text, images=[image])

    input_ids, packed = _assert_common(outputs)
    _assert_vision_segments(packed, expected_segments=1)
    assert input_ids.shape[1] == base_length
    assert packed["vision_token_lengths"].item() == base_vision_length


@require_torch
@require_vision
def test_crop_truncates_text_segment_only(isaac_processor, isaac_tokenizer, isaac_tiny_config):
    vision_token = isaac_processor.vision_token
    text_prefix_tokens = " ".join([f"t{i}" for i in range(8)])
    text_suffix = "tail end"
    text = f"{text_prefix_tokens} {vision_token} {text_suffix}"
    image = _make_dummy_image()

    base_outputs = _run_processor(isaac_processor, text=text, images=[image])
    base_packed = base_outputs["packed_inputs"]
    full_text_tokens = _count_modality(base_packed, ModalityType.text.value)
    vision_length = base_packed["vision_token_lengths"][0].item()

    max_len = base_outputs["input_ids"].shape[1] - 4
    processor = _make_processor_with_max_len(isaac_tokenizer, isaac_tiny_config, max_len)
    outputs = _run_processor(processor, text=text, images=[image])

    input_ids, packed = _assert_common(outputs)
    _assert_vision_segments(packed, expected_segments=1)
    assert input_ids.shape[1] == max_len

    kept_text_tokens = _count_modality(packed, ModalityType.text.value)
    assert kept_text_tokens == full_text_tokens - 4
    torch.testing.assert_close(packed["vision_token_offsets"], torch.zeros_like(packed["vision_token_offsets"]))
    assert packed["vision_token_lengths"].item() == vision_length


@require_torch
@require_vision
def test_crop_cuts_through_image_segment(isaac_processor, isaac_tokenizer, isaac_tiny_config):
    vision_token = isaac_processor.vision_token
    text_before = "hi"
    text_after = "bye"
    text = f"{text_before} {vision_token} {text_after}"
    image = _make_dummy_image()

    base_outputs = _run_processor(isaac_processor, text=text, images=[image])
    base_packed = base_outputs["packed_inputs"]
    vision_full = base_packed["vision_token_lengths"][0].item()
    text_before_len = len(isaac_tokenizer.encode(text_before, add_special_tokens=False))
    text_after_len = len(isaac_tokenizer.encode(text_after, add_special_tokens=False))
    total_length = vision_full + text_before_len + text_after_len

    max_len = 40
    start = total_length - max_len
    expected_offset = max(0, start - text_before_len)
    expected_length = vision_full - expected_offset

    processor = _make_processor_with_max_len(isaac_tokenizer, isaac_tiny_config, max_len)
    outputs = _run_processor(processor, text=text, images=[image])

    input_ids, packed = _assert_common(outputs)
    _assert_vision_segments(packed, expected_segments=1)

    assert input_ids.shape[1] == max_len
    assert packed["vision_token_offsets"].item() == expected_offset
    assert packed["vision_token_lengths"].item() == expected_length
    assert _count_modality(packed, ModalityType.text.value) == text_after_len


@require_torch
@require_vision
def test_crop_removes_all_vision_when_window_excludes_images(isaac_processor, isaac_tokenizer, isaac_tiny_config):
    vision_token = isaac_processor.vision_token
    text_tail = "closing"
    text = f"{vision_token} {text_tail}"
    image = _make_dummy_image()

    tail_tokens = len(isaac_processor.tokenizer.encode(text_tail, add_special_tokens=False))
    processor = _make_processor_with_max_len(isaac_tokenizer, isaac_tiny_config, tail_tokens)
    outputs = _run_processor(processor, text=text, images=[image])

    input_ids, packed = _assert_common(outputs)
    _assert_no_vision(packed)
    assert input_ids.shape[1] == tail_tokens
    assert _count_modality(packed, ModalityType.text.value) == tail_tokens


@require_torch
@require_vision
def test_batch_outputs_match_individual_calls(isaac_processor):
    texts = ["hi", "this one is longer"]

    per_sample = [_run_processor(isaac_processor, text=t, images=None) for t in texts]
    batch_outputs = _run_processor(isaac_processor, text=texts, images=None)

    assert set(batch_outputs.keys()) == {"input_ids", "packed_inputs"}
    batch_input_ids = batch_outputs["input_ids"]
    batch_packed = batch_outputs["packed_inputs"]

    assert set(batch_packed.keys()) == {
        "vision_patches",
        "vision_token_grids",
        "vision_token_offsets",
        "vision_token_lengths",
        "vision_token_batch_indices",
        "modality_tensor",
        "position_ids",
    }

    assert batch_input_ids.shape[0] == len(texts)
    assert batch_packed["modality_tensor"].shape[0] == len(texts)
    assert batch_packed["position_ids"].shape[0] == len(texts)

    sample_lengths = [output["input_ids"].squeeze(0).shape[0] for output in per_sample]
    max_length = max(sample_lengths)
    pad_id = isaac_processor.pad_token_id

    for i, (single_output, batch_ids, single_len) in enumerate(zip(per_sample, batch_input_ids, sample_lengths)):
        single_ids = single_output["input_ids"].squeeze(0)
        single_packed = single_output["packed_inputs"]

        torch.testing.assert_close(batch_ids[-single_len:], single_ids)

        batch_modality_row = batch_packed["modality_tensor"][i]
        expected_modality = torch.full(
            (max_length,),
            batch_modality_row[-1].item(),
            dtype=batch_modality_row.dtype,
            device=batch_modality_row.device,
        )
        expected_modality[-single_len:] = single_packed["modality_tensor"].squeeze(0)
        torch.testing.assert_close(batch_modality_row, expected_modality)

        batch_positions_row = batch_packed["position_ids"][i]
        expected_positions = torch.zeros(
            (max_length, 3), dtype=batch_positions_row.dtype, device=batch_positions_row.device
        )
        expected_positions[-single_len:] = single_packed["position_ids"].squeeze(0)
        torch.testing.assert_close(batch_positions_row, expected_positions)

        if single_len == max_length:
            continue

        pad_span = batch_ids[: max_length - single_len]
        assert torch.all(pad_span == pad_id)

        attention_mask = batch_ids.ne(pad_id).long()
        assert not torch.any(attention_mask[: max_length - single_len])
        assert torch.all(attention_mask[-single_len:])

    _assert_no_vision(batch_packed)


class StubTokenizer(SimpleIsaacTokenizer):
    def __init__(self):
        super().__init__()
        self._base = 2000

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        token_ids = torch.tensor([self._base + b for b in text.encode("utf-8")], dtype=torch.long)
        if return_tensors in {"pt", TensorType.PYTORCH}:
            return token_ids.unsqueeze(0)
        return token_ids

    def convert_tokens_to_ids(self, token):
        if token == "<|image_pad|>":
            return 151655
        if token == self.pad_token:
            return super().convert_tokens_to_ids(token)
        return None


class StubImageProcessor(ImageProcessingMixin):
    def __call__(self, images=None, return_tensors=None):
        patches = torch.ones((1, 2, 2, 3), dtype=torch.float32)
        sizes = torch.tensor([[1, 2, 2]], dtype=torch.long)
        return {
            "patches": patches,
            "virtual_pixel_size": sizes,
            "real_pixel_size": sizes,
        }


BASE_MODEL_ID = os.environ.get("ISAAC_TEST_MODEL_ID", "PerceptronAI/Isaac-0.1-Base")
BASE_MODEL_REVISION = os.environ.get("ISAAC_TEST_MODEL_REVISION", "refs/pr/3") or None
LOCAL_CHECKPOINT = os.environ.get("ISAAC_TEST_MODEL_PATH")


def _checkpoint_or_skip():
    if LOCAL_CHECKPOINT:
        resolved = Path(LOCAL_CHECKPOINT).expanduser()
        if not resolved.exists():
            pytest.skip(f"Local checkpoint path {resolved} does not exist.")
        return str(resolved)
    if is_offline_mode():
        pytest.skip("Offline mode: set ISAAC_TEST_MODEL_PATH to a local checkpoint to run these tests.")
    return BASE_MODEL_ID


def _create_real_processor():
    checkpoint = _checkpoint_or_skip()
    config = IsaacConfig.from_pretrained(checkpoint, revision=BASE_MODEL_REVISION)
    processor = IsaacProcessor.from_pretrained(checkpoint, revision=BASE_MODEL_REVISION)
    tokenizer = processor.tokenizer
    return processor, tokenizer, config


@require_torch
@require_vision
class TestIsaacProcessorRealPadding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor, cls.tokenizer, cls.config = _create_real_processor()
        cls.dummy_image = _make_dummy_image()
        cls.vision_token = cls.config.vision_token
        cls.pad_id = cls.tokenizer.pad_token_id
        cls.image_pad_id = cls.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if cls.pad_id is None or cls.image_pad_id is None:
            pytest.skip("pad/image pad ids unavailable for processor")

    def _check_padding_and_masks(self, input_ids: torch.Tensor, pad_id: int):
        for row in range(input_ids.size(0)):
            row_ids = input_ids[row]
            nonpad_positions = (row_ids != pad_id).nonzero(as_tuple=False)
            last_nonpad = int(nonpad_positions.max()) if nonpad_positions.numel() else -1
            if last_nonpad + 1 < row_ids.numel():
                tail = row_ids[last_nonpad + 1 :]
                assert torch.all(tail == pad_id)
            attn = (row_ids != pad_id).long()
            if last_nonpad >= 0:
                assert torch.all(attn[: last_nonpad + 1] == 1)
            assert int(attn[last_nonpad + 1 :].sum()) == 0

    def test_single_vs_batched_consistency(self):
        prompt = f"hello {self.vision_token} world"
        images_single = [self.dummy_image]

        single = self.processor(text=prompt, images=images_single, return_tensors="pt")
        single_ids = single["input_ids"].squeeze(0)

        batch_prompts = [prompt, "short"]
        batch_images = [images_single, None]
        batch = self.processor(text=batch_prompts, images=batch_images, return_tensors="pt")
        batch_ids = batch["input_ids"][0]
        modality = batch["packed_inputs"]["modality_tensor"][0]

        assert torch.equal(batch_ids[: single_ids.size(0)], single_ids)

        image_positions = modality == ModalityType.image.value
        if image_positions.any():
            assert torch.all(batch_ids[image_positions] == self.image_pad_id)
            assert torch.all(batch_ids[image_positions] != self.pad_id)

        nonpad = (batch_ids != self.pad_id).nonzero(as_tuple=False)
        last_nonpad = int(nonpad.max()) if nonpad.numel() else -1
        if last_nonpad + 1 < batch_ids.numel():
            tail = batch_ids[last_nonpad + 1 :]
            assert torch.all(tail == self.pad_id)

        attn = (batch_ids != self.pad_id).long()
        if last_nonpad >= 0:
            assert torch.all(attn[: last_nonpad + 1] == 1)
        assert int(attn[last_nonpad + 1 :].sum()) == 0
