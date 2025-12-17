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

import pytest

from transformers import IsaacConfig, PythonBackend
from transformers.models.isaac.image_processing_isaac_fast import IsaacImageProcessorFast
from transformers.models.isaac.processing_isaac import IsaacProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available
from transformers.utils.import_utils import is_perceptron_available


if is_vision_available():
    from PIL import Image
else:
    Image = None


if is_perceptron_available():
    from perceptron.tensorstream.tensorstream import TensorStream
else:
    TensorStream = None


require_tensorstream = pytest.mark.skipif(TensorStream is None, reason="TensorStream backend is not available")


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
@require_tensorstream
def test_isaac_processor_matches_config_defaults(isaac_processor, isaac_tiny_config):
    assert isaac_processor.vision_token == isaac_tiny_config.vision_token
    assert isaac_processor.max_sequence_length == isaac_tiny_config.max_sequence_length
    assert isaac_processor.config is isaac_tiny_config
    assert isinstance(isaac_processor.image_processor, IsaacImageProcessorFast)
    assert isaac_processor.image_processor.rescale_factor == pytest.approx(isaac_tiny_config.vision_rescale_factor)


@require_torch
@require_vision
@require_tensorstream
def test_isaac_processor_text_only_round_trip(isaac_processor):
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    prompt = isaac_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = isaac_processor(text=prompt, images=None, return_tensors="pt")

    assert "input_ids" in outputs
    assert "tensor_stream" in outputs
    assert TensorStream is not None
    assert isinstance(outputs["tensor_stream"], TensorStream)
    assert outputs["input_ids"].shape[0] == 1


@require_torch
@require_tensorstream
def test_isaac_processor_accepts_batchencoding_chat_template(isaac_processor):
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    batch_encoding = isaac_processor.apply_chat_template(messages, add_generation_prompt=True)

    outputs = isaac_processor(text=batch_encoding, images=None, return_tensors="pt")

    assert "input_ids" in outputs
    assert "tensor_stream" in outputs
    assert TensorStream is not None
    assert isinstance(outputs["tensor_stream"], TensorStream)
    assert outputs["input_ids"].shape[0] == 1


@require_torch
@require_vision
@require_tensorstream
def test_isaac_processor_with_single_image(isaac_processor):
    vision_token = isaac_processor.vision_token
    text = f"Look at this {vision_token} and describe it."
    image = _make_dummy_image()

    outputs = isaac_processor(text=text, images=[image], return_tensors="pt")
    assert TensorStream is not None
    assert isinstance(outputs["tensor_stream"], TensorStream)
    assert outputs["input_ids"].ndim == 2


@require_torch
@require_vision
@require_tensorstream
def test_isaac_processor_with_multiple_images(isaac_processor):
    vision_token = isaac_processor.vision_token
    text = f"First {vision_token} then {vision_token}"
    images = [_make_dummy_image(color=(255, 0, 0)), _make_dummy_image(color=(0, 255, 0))]

    outputs = isaac_processor(text=text, images=images, return_tensors="pt")
    assert TensorStream is not None
    assert isinstance(outputs["tensor_stream"], TensorStream)
    assert outputs["input_ids"].shape[0] == 1


@require_torch
@require_vision
@require_tensorstream
def test_isaac_processor_error_on_image_mismatch(isaac_processor):
    vision_token = isaac_processor.vision_token
    text = f"{vision_token} {vision_token}"
    image = _make_dummy_image()

    with pytest.raises(ValueError, match="must match number of images"):
        isaac_processor(text=text, images=[image], return_tensors="pt")


@require_torch
@require_vision
@require_tensorstream
def test_isaac_processor_consistent_tensor_stream_types(isaac_processor):
    text_only = "Simple question?"
    text_with_image = f"Describe this {isaac_processor.vision_token}"
    image = _make_dummy_image()

    outputs_text = isaac_processor(text=text_only, images=None, return_tensors="pt")
    outputs_image = isaac_processor(text=text_with_image, images=[image], return_tensors="pt")

    assert TensorStream is not None
    assert isinstance(outputs_text["tensor_stream"], TensorStream)
    assert isinstance(outputs_image["tensor_stream"], TensorStream)
    assert outputs_text["input_ids"].shape[0] == outputs_image["input_ids"].shape[0] == 1
