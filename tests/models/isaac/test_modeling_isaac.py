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

"""Testing suite for the Isaac model."""

import base64
import hashlib
import io
import random
import numpy as np
import json
import os
import unittest
from functools import lru_cache
from pathlib import Path

import pytest
from huggingface_hub import is_offline_mode

from tests.generation.test_utils import GenerationTesterMixin
from tests.test_configuration_common import ConfigTester
from tests.test_pipeline_mixin import PipelineTesterMixin
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    IsaacConfig,
    IsaacForConditionalGeneration,
    IsaacModel,
    PythonBackend,
    is_torch_available,
)
from transformers.image_utils import load_image
from transformers.masking_utils import eager_mask, sdpa_mask
from transformers.models.isaac.configuration_isaac import IsaacVisionConfig
from transformers.models.isaac.image_processing_isaac_fast import IsaacImageProcessorFast
from transformers.models.isaac.modeling_isaac import (
    IsaacVisionAttention,
    document_mask_function_from_cu_seqlens,
    ensure_document_attention_mask,
)
from transformers.models.isaac.processing_isaac import IsaacProcessor
from transformers.testing_utils import (
    get_tests_dir,
    require_flash_attn,
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_vision_available
from transformers.utils.import_utils import is_perceptron_available


if is_vision_available():
    from PIL import Image
else:
    Image = None

from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

if is_perceptron_available():
    from perceptron.tensorstream.ops import modality_mask, role_mask, tensor_stream_token_view
    from perceptron.tensorstream.tensorstream import TensorStream
else:
    TensorStream = None


require_tensorstream = pytest.mark.skipif(TensorStream is None, reason="TensorStream backend is not available")

MODEL_ID = os.environ.get("ISAAC_TEST_MODEL_ID", "PerceptronAI/Isaac-0.1-Base")
MODEL_REVISION = os.environ.get("ISAAC_TEST_MODEL_REVISION", "refs/pr/3") or None
LOCAL_CHECKPOINT = os.environ.get("ISAAC_TEST_MODEL_PATH")
FIXTURES_DIR = Path(get_tests_dir("fixtures/isaac"))
HASH_FILE = FIXTURES_DIR / "isaac_checkpoint_hashes.json"
GENERATION_GOLDEN_FILE = FIXTURES_DIR / "isaac_generation_golden.json"
HASH_FILTERS = {
    "full_model": {"include": None, "exclude": None},
    "core_model": {"include": None, "exclude": {"vision_embedding", "audio_embedding", "inv_freq"}},
    "vision_modules": {"include": {"vision_embedding"}, "exclude": None},
}
RED_DOT_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="


def tensor_stream_snapshot(ts: TensorStream) -> dict[str, object]:
    """Summarize TensorStream tokens/modalities using public utilities."""

    token_view = tensor_stream_token_view(ts).cpu().tolist()
    modality = modality_mask(ts).cpu().tolist()
    roles = role_mask(ts).cpu().tolist()

    return {
        "shape": list(ts.shape),
        "token_view": token_view,
        "modality_mask": modality,
        "role_mask": roles,
    }


def document_to_messages(
    document: list[dict], vision_token: str = "<image>"
) -> tuple[list[dict[str, str]], list[Image]]:
    """
    Convert a Document to messages format compatible with chat templates.
    Each content turn creates its own message entry.

    Args:
        document: list of dicts containing Text and/or Image content
        vision_token: Token to use for image placeholder

    Returns:
        Tuple of (messages, images) where messages is a list of dicts with 'role' and 'content'
    """
    messages = []
    images = []

    for item in document:
        itype = item.get("type")
        if itype == "text":
            content = item.get("content")
            if content:
                messages.append(
                    {
                        "role": item.get("role", "user"),
                        "content": content,
                    }
                )
        elif itype == "image":
            content = item.get("content")
            if content:
                img = load_image(content)
                images.append(img)
                messages.append(
                    {
                        "role": item.get("role", "user"),
                        "content": vision_token,
                    }
                )

    return messages, images


def _tensor_to_bytes(tensor):
    cpu_tensor = tensor.detach().cpu().contiguous()
    if cpu_tensor.is_floating_point():
        cpu_tensor = cpu_tensor.to(dtype=torch.float32)
    return cpu_tensor.numpy().tobytes()


def _iter_filtered_items(state_dict, include=None, exclude=None):
    for name, tensor in state_dict.items():
        if include and not any(token in name for token in include):
            continue
        if exclude and any(token in name for token in exclude):
            continue
        yield name, tensor


def _hash_state_dict(state_dict, *, include=None, exclude=None):
    hasher = hashlib.sha256()
    items = sorted(_iter_filtered_items(state_dict, include=include, exclude=exclude), key=lambda kv: kv[0])
    for name, tensor in items:
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(_tensor_to_bytes(tensor))
    return hasher.hexdigest()


def compute_logits_statistics(tensor: torch.Tensor) -> dict[str, object]:
    """
    Summarize logits with simple statistics that are stable across minor
    implementation changes yet still sensitive to behavioral regressions.
    """

    float_tensor = tensor.detach().to(torch.float32).cpu()
    flat = float_tensor.reshape(-1).to(torch.float64)

    def _rounded(value: torch.Tensor | float) -> float:
        return round(float(value), 10)

    return {
        "shape": list(float_tensor.shape),
        "numel": flat.numel(),
        "mean": _rounded(flat.mean()),
        "std": _rounded(flat.std(unbiased=False)),
        "min": _rounded(flat.min()),
        "max": _rounded(flat.max()),
        "sum": _rounded(flat.sum()),
        "l2_norm": _rounded(torch.linalg.vector_norm(flat, ord=2)),
    }


def _assert_logits_statistics_close(
    actual: dict[str, object],
    expected: dict[str, object],
    *,
    rel: float = 1e-5,
    abs_tol: float = 1e-6,
) -> None:
    assert actual["shape"] == expected["shape"], "Logits shape changed"
    assert actual["numel"] == expected["numel"], "Logits numel changed"
    for key in ("mean", "std", "min", "max", "sum", "l2_norm"):
        assert actual[key] == pytest.approx(
            expected[key],
            rel=rel,
            abs=abs_tol,
        ), f"Logits statistic '{key}' drifted"


def _hf_from_pretrained(cls, pretrained_id, **kwargs):
    """
    Wrapper around `cls.from_pretrained` that automatically injects
    the test revision (if any) from MODEL_REVISION.
    """
    if MODEL_REVISION is not None:
        kwargs.setdefault("revision", MODEL_REVISION)
    return cls.from_pretrained(pretrained_id, **kwargs)


@pytest.fixture(scope="session")
def tokenizer(isaac_reference_checkpoint):
    """Load the tokenizer from the converted Perceptron HF checkpoint."""
    return _hf_from_pretrained(
        AutoTokenizer,
        isaac_reference_checkpoint,
        trust_remote_code=True,
    )


@require_torch
def test_document_mask_function_from_cu_seqlens():
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    mask_fn = document_mask_function_from_cu_seqlens(cu_seqlens)

    assert mask_fn is not None
    # Same document (indices 1 and 2)
    assert mask_fn(0, 0, 1, 2)
    # Cross-document (index 1 in first doc, 3 in second doc)
    assert not mask_fn(0, 0, 1, 3)
    # Same second document (indices 3 and 4)
    assert mask_fn(0, 0, 4, 3)


@require_torch
def test_ensure_document_attention_mask_prefers_callable_when_requested():
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    total_tokens = 5
    dtype = torch.float32

    mask_callable = ensure_document_attention_mask(
        attention_mask=None,
        cu_seqlens=cu_seqlens,
        total_tokens=total_tokens,
        dtype=dtype,
        device=cu_seqlens.device,
        return_mask_function=True,
    )
    assert callable(mask_callable)

    additive = ensure_document_attention_mask(
        attention_mask=None,
        cu_seqlens=cu_seqlens,
        total_tokens=total_tokens,
        dtype=dtype,
        device=cu_seqlens.device,
        return_mask_function=False,
    )
    assert additive is None


def create_isaac_processor(
    tokenizer,
    isaac_config,
    *,
    image_processor=None,
    **overrides,
):
    """Helper to construct IsaacProcessor without requiring an IsaacConfig instance."""
    params = {
        "vision_token": isaac_config.vision_token,
        "max_sequence_length": isaac_config.max_sequence_length,
        "vision_patch_size": isaac_config.vision_patch_size,
        "vision_max_num_patches": isaac_config.vision_max_num_patches,
        "vision_min_num_patches": isaac_config.vision_min_num_patches,
        "pixel_shuffle_scale": isaac_config.pixel_shuffle_scale,
        "rescale_factor": isaac_config.vision_rescale_factor,
        "image_mean": tuple(isaac_config.vision_mean),
        "image_std": tuple(isaac_config.vision_std),
    }
    params.update(overrides)

    processor_image = image_processor
    if processor_image is None:
        processor_image = IsaacImageProcessorFast(
            patch_size=params["vision_patch_size"],
            max_num_patches=params["vision_max_num_patches"],
            min_num_patches=params["vision_min_num_patches"],
            pixel_shuffle_scale=params["pixel_shuffle_scale"],
            rescale_factor=params["rescale_factor"],
            image_mean=params["image_mean"],
            image_std=params["image_std"],
        )
    processor_params = {
        "vision_token": isaac_config.vision_token,
        "max_sequence_length": isaac_config.max_sequence_length,
        "rescale_factor": isaac_config.vision_rescale_factor,
    }

    return IsaacProcessor(
        image_processor=processor_image,
        tokenizer=tokenizer,
        **processor_params,
    )


@require_torch
def test_document_mask_function_materializes_with_masking_utils():
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    total_tokens = 4
    mask_fn = document_mask_function_from_cu_seqlens(cu_seqlens)

    cache_position = torch.arange(total_tokens, device=cu_seqlens.device, dtype=torch.long)
    expected_bool = torch.tensor(
        [
            [
                [
                    [True, True, False, False],
                    [True, True, False, False],
                    [False, False, True, True],
                    [False, False, True, True],
                ]
            ]
        ],
        device=cu_seqlens.device,
    )

    sdpa = sdpa_mask(
        batch_size=1,
        cache_position=cache_position,
        kv_length=total_tokens,
        kv_offset=0,
        mask_function=mask_fn,
        attention_mask=None,
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=False,
        allow_torch_fix=False,
        use_vmap=False,
    )
    # sdpa_mask returns True for allowed positions; SDPA expects True to mean "mask out"
    assert torch.equal(sdpa, expected_bool)

    eager = eager_mask(
        batch_size=1,
        cache_position=cache_position,
        kv_length=total_tokens,
        kv_offset=0,
        mask_function=mask_fn,
        attention_mask=None,
        allow_is_bidirectional_skip=False,
        use_vmap=False,
        dtype=torch.float32,
    )
    expected_additive = torch.where(
        expected_bool,
        torch.tensor(0.0, device=cu_seqlens.device, dtype=torch.float32),
        torch.tensor(torch.finfo(torch.float32).min, device=cu_seqlens.device, dtype=torch.float32),
    )
    assert torch.equal(eager, expected_additive)




@lru_cache(maxsize=1)
def _load_expected_hashes():
    if not HASH_FILE.exists():
        return None
    with HASH_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _load_generation_golden():
    if not GENERATION_GOLDEN_FILE.exists():
        return None
    with GENERATION_GOLDEN_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def safe_decode(tokenizer, token_ids):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception:
        tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
        tokens = [tok for tok in tokens if tok is not None]
        text = tokenizer.convert_tokens_to_string(tokens)
    return text.strip() if isinstance(text, str) else text


@lru_cache(maxsize=1)
def _load_red_dot_image():
    if Image is None:
        return None
    data = base64.b64decode(RED_DOT_B64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _reference_checkpoint_or_skip():
    if TensorStream is None:
        pytest.skip("TensorStream dependency is required for Isaac integration tests.")
    if LOCAL_CHECKPOINT:
        resolved = Path(LOCAL_CHECKPOINT).expanduser()
        if not resolved.exists():
            pytest.skip(f"Local checkpoint path {resolved} does not exist.")
        return str(resolved)
    if is_offline_mode():
        pytest.skip("Offline mode: set ISAAC_TEST_MODEL_PATH to a local checkpoint to run these tests.")
    return MODEL_ID


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
    tester = IsaacModelTester(parent=None)
    return tester.get_config()


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


@pytest.fixture(scope="session")
def isaac_reference_checkpoint():
    return _reference_checkpoint_or_skip()


@pytest.fixture(scope="session")
def isaac_config(isaac_reference_checkpoint):
    """Load IsaacConfig from the converted checkpoint."""
    # Load the config directly from the converted checkpoint
    config = _hf_from_pretrained(IsaacConfig, isaac_reference_checkpoint)
    # Most tests assume flash attention in vision unless they explicitly override it.
    config.vision_attn_implementation = "flash_attention_2"
    return config


@pytest.fixture(scope="session")
def isaac_reference_model(isaac_reference_checkpoint, isaac_config):
    model_config = IsaacConfig.from_dict(isaac_config.to_dict())
    model_config.vision_config._attn_implementation = "flash_attention_2"
    model = _hf_from_pretrained(
        IsaacForConditionalGeneration,
        isaac_reference_checkpoint,
        config=model_config,
        attn_implementation="sdpa",
    )
    return model


@pytest.fixture(scope="session")
def isaac_reference_processor(isaac_reference_checkpoint):
    try:
        processor = _hf_from_pretrained(AutoProcessor, isaac_reference_checkpoint)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Unable to load reference Isaac processor from {isaac_reference_checkpoint}") from error
    print(f"[Isaac tests] Loaded processor type: {type(processor)} from {isaac_reference_checkpoint}")
    if not isinstance(processor, IsaacProcessor):
        pytest.skip("Loaded processor is not an IsaacProcessor instance.")
    return processor


class IsaacModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=5,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.is_training = True
        self.expected_num_hidden_layers = 1

        self.text_config = {
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "hidden_act": "silu",
            "head_dim": hidden_size // num_attention_heads,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "intermediate_size": hidden_size * 3,
            "max_position_embeddings": 128,
            "model_type": "qwen3",
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "num_key_value_heads": num_attention_heads,
            # Keep the same multi-RoPE setup as the reference checkpoints but shrink the
            # sections so they sum to the rotary half-dimension (4) of this tiny test model.
            "rope_parameters": {"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True},
            "tie_word_embeddings": True,
        }

        self.vision_config = {
            "hidden_size": hidden_size,
            "intermediate_size": hidden_size * 2,
            "num_hidden_layers": 1,
            "num_attention_heads": num_attention_heads,
            "num_channels": 3,
            "num_patches": 64,
            "patch_size": 4,
            "pixel_shuffle_scale_factor": 1,
            "attention_dropout": 0.0,
            "layer_norm_eps": 1e-6,
        }

    def get_config(self):
        config = IsaacConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
        )
        # Rely on vanilla SDPA so the tests do not need flash attention.
        config._attn_implementation = "sdpa"
        config.text_config._attn_implementation = "sdpa"
        config.vision_attn_implementation = "sdpa"
        return config

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(
            (self.batch_size, self.seq_length),
            dtype=torch.long,
            device=torch_device,
        )
        labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        return config, input_ids, attention_mask, labels

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask, labels = self.prepare_config_and_inputs()
        position_ids = torch.arange(self.seq_length, device=torch_device).view(1, -1)
        position_ids = position_ids.expand(self.batch_size, -1).unsqueeze(2).expand(-1, -1, 3)
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        if labels is not None:
            inputs_dict["labels"] = labels
        return config, inputs_dict


@require_torch
class IsaacModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (IsaacModel, IsaacForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "image-to-text": IsaacForConditionalGeneration,
            "image-text-to-text": IsaacForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    _is_composite = True
    test_attention_outputs = False
    test_all_params_have_gradient = False

    def setUp(self):
        self.model_tester = IsaacModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=IsaacConfig,
            has_text_modality=False,
        )

    def test_config(self):
        self.maxDiff = None
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Assisted decoding not supported; Qwen3 backbone does not implement returning attentions")
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip(reason="Assisted decoding not supported; Qwen3 backbone does not implement returning attentions")
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    @unittest.skip(reason="Assisted decoding not supported; Qwen3 backbone does not implement returning attentions")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(reason="Prompt lookup decoding not supported; Qwen3 backbone does not return attentions")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip(reason="Output attentions not supported")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @require_tensorstream
    def test_model_forward(self):
        config, input_ids, attention_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = IsaacModel(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask)

        self.assertEqual(
            result.last_hidden_state.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, config.hidden_size),
        )

    @require_tensorstream
    def test_for_conditional_generation(self):
        config, input_ids, attention_mask, labels = self.model_tester.prepare_config_and_inputs()
        model = IsaacForConditionalGeneration(config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size),
        )
        self.assertIsNotNone(result.loss)

    def test_prepare_inputs_for_generation(self):
        config, input_ids, attention_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = IsaacForConditionalGeneration(config)
        model.to(torch_device)

        prepared_inputs = model.prepare_inputs_for_generation(input_ids=input_ids, attention_mask=attention_mask)
        self.assertIn("input_ids", prepared_inputs)
        self.assertIn("position_ids", prepared_inputs)


@require_torch
@require_tensorstream
def test_isaac_for_conditional_generation_initialization(isaac_tiny_config):
    model = IsaacForConditionalGeneration(isaac_tiny_config)
    model.to(torch_device)
    assert hasattr(model, "model")
    assert hasattr(model, "lm_head")
    assert hasattr(model.model, "vision_embedding")
    assert hasattr(model.model, "embed_fns")

    input_ids = torch.randint(0, isaac_tiny_config.vocab_size, (1, 10), device=torch_device, dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True)
    assert outputs.logits.shape == (1, 10, isaac_tiny_config.vocab_size)


@require_torch
@require_tensorstream
def test_isaac_for_conditional_generation_loss_and_generate_flag(isaac_tiny_config):
    model = IsaacForConditionalGeneration(isaac_tiny_config).to(torch_device)
    assert model.can_generate()

    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, isaac_tiny_config.vocab_size, (batch_size, seq_len), device=torch_device)
    labels = torch.randint(0, isaac_tiny_config.vocab_size, (batch_size, seq_len), device=torch_device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
    assert outputs.loss is not None
    assert outputs.loss.ndim == 0
    assert outputs.logits.shape == (batch_size, seq_len, isaac_tiny_config.vocab_size)


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

    assert isinstance(outputs_text["tensor_stream"], TensorStream)
    assert isinstance(outputs_image["tensor_stream"], TensorStream)
    assert outputs_text["input_ids"].shape[0] == outputs_image["input_ids"].shape[0] == 1


@require_torch
@require_vision
@require_tensorstream
def test_isaac_generation_with_tensor_stream(isaac_processor, isaac_tiny_config):
    model = IsaacForConditionalGeneration(isaac_tiny_config).to(torch_device)
    model.eval()

    messages = [{"role": "user", "content": "Hello there!"}]
    prompt = isaac_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processed = isaac_processor(text=prompt, images=None, return_tensors="pt")

    input_ids = processed["input_ids"].to(torch_device)
    tensor_stream = processed["tensor_stream"]
    tensor_stream = tensor_stream.to(torch_device)
    generated = model.generate(
        input_ids=input_ids,
        tensor_stream=tensor_stream,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=isaac_processor.tokenizer.pad_token_id,
        eos_token_id=isaac_processor.tokenizer.eos_token_id,
    )

    assert generated.shape[0] == 1
    assert generated.shape[1] >= input_ids.shape[1]
    decoded_prompt = isaac_processor.tokenizer.decode(generated[0], skip_special_tokens=True)
    assert isinstance(decoded_prompt, str)
    assert decoded_prompt.strip() != ""


def test_hf_generate_something():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    # Configuration
    MAX_NEW_TOKENS = 10
    DTYPE = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    genesis_hf_checkpoint = "/home/phil/backup/genesis/genesis_isaac_base_hf_converted_checkpoint/"

    hf_config = IsaacConfig.from_pretrained(genesis_hf_checkpoint)
    # messages, images = document_to_messages(document, vision_token=hf_config.vision_token)
    messages = [{"role": "user", "content": "Describe this image:"}, {"role": "user", "content": "<image>"}]
    images = []
    image_bytes = base64.b64decode(RED_DOT_B64)
    pil_image = Image.open(io.BytesIO(image_bytes))
    images.append(pil_image)
    print("----------")
    print(messages)
    print("----------")

    tokenizer = AutoTokenizer.from_pretrained(genesis_hf_checkpoint, trust_remote_code=True, use_fast=False)
    genesis_processor = create_isaac_processor(tokenizer, hf_config)
    # Apply chat template with roles (add_generation_prompt=True to match DocumentProcessor)
    # Added strip because our generation events don't add new line
    text = genesis_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).strip()
    processor_output = genesis_processor(text=text, images=images, return_tensors="pt")
    tensor_stream = processor_output["tensor_stream"].to(device)

    # Process document to TensorStream
    hf_config.vision_config._attn_implementation = "flash_attention_2"
    hf_config.vision_config.attn_implementation = "flash_attention_2"
    hf_model = IsaacForConditionalGeneration.from_pretrained(genesis_hf_checkpoint, config=hf_config)
    hf_model = hf_model.to(device=device, dtype=DTYPE)
    hf_model.eval()

    # Load HF tokenizer

    # Validate that weights are identical between models

    with torch.inference_mode():
        print("\n1️⃣ Running HuggingFace model.generate()...")
        # Generate with HF model using the training tensor stream converted to Open variant
        hf_output = hf_model.generate(
            tensor_stream=tensor_stream,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # Deterministic generation
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_logits=True,
        )

        hf_generated_ids = hf_output.sequences
        hf_generated_text = tokenizer.decode(hf_generated_ids[0], skip_special_tokens=True)
        print(f"   HF Generated: '{hf_generated_text}'")
        assert "is" in hf_generated_text
        hf_logits = torch.cat(hf_output.logits, dim=0)
        logit_stats = compute_logits_statistics(hf_logits)
        print("-------------")
        print(logit_stats)
        print("-------------")


@require_torch
@require_vision
@slow
@require_tensorstream
def test_hf_generate_vs_training_generate_logits(isaac_reference_model, isaac_reference_processor):
    device = "cuda"
    dtype = torch.bfloat16
    isaac_reference_model = isaac_reference_model.to(device=device, dtype=dtype)
    isaac_reference_model.eval()
    golden = _load_generation_golden()
    if not golden:
        pytest.skip(f"Missing generation golden file at {GENERATION_GOLDEN_FILE}.")

    image = _load_red_dot_image()
    if image is None:
        pytest.skip("PIL.Image is required for Isaac generation tests.")

    messages = [
        {
            "role": "user",
            "content": "Describe this image:",
        },
        {
            "role": "user",
            "content": "<image>",
        },
    ]
    prompt = isaac_reference_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ).strip()
    batch = isaac_reference_processor(text=prompt, images=[image], return_tensors="pt")

    input_ids = batch["input_ids"]
    tensor_stream = batch["tensor_stream"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    isaac_reference_model.to(device)
    input_ids = input_ids.to(device)
    if tensor_stream is not None and hasattr(tensor_stream, "to"):
        tensor_stream = tensor_stream.to(device)

    torch.manual_seed(0)
    with torch.no_grad():
        outputs = isaac_reference_model.generate(
            input_ids=input_ids,
            tensor_stream=tensor_stream,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=isaac_reference_processor.tokenizer.eos_token_id,
            eos_token_id=isaac_reference_processor.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_logits=True,
        )

    logits = torch.cat(outputs.logits, dim=0).to(torch.float32).cpu()
    logits_stats = compute_logits_statistics(logits)
    generated_ids = outputs.sequences[0].tolist()

    assert generated_ids == golden["token_ids"], "Generated token ids changed"
    if "logits_statistics" in golden:
        _assert_logits_statistics_close(logits_stats, golden["logits_statistics"])
    else:
        pytest.fail(
            "Golden file missing both logits_statistics and logits_hash. "
            f"Regenerate {GENERATION_GOLDEN_FILE} via scripts/update_isaac_hashes.py."
        )

    isaac_reference_model.to("cpu")


@require_torch
@require_vision
@slow
@require_tensorstream
@require_flash_attn
class IsaacGenerationIntegrationTest(unittest.TestCase):
    max_new_tokens = 25
    dtype = torch.bfloat16

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = _reference_checkpoint_or_skip()
        self.hf_config = IsaacConfig.from_pretrained(self.checkpoint, revision=MODEL_REVISION)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, trust_remote_code=True, use_fast=False, revision=MODEL_REVISION
        )
        self.processor = create_isaac_processor(self.tokenizer, self.hf_config)
        self.hf_config.vision_config._attn_implementation = "flash_attention_2"
        self.hf_config.vision_config.attn_implementation = "flash_attention_2"
        self.model = IsaacForConditionalGeneration.from_pretrained(
            self.checkpoint, config=self.hf_config, revision=MODEL_REVISION
        )
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    def _generate_from_messages(self, messages, images, num_tokens=None):
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).strip()
        processor_output = self.processor(text=prompt, images=images, return_tensors="pt")
        tensor_stream = processor_output["tensor_stream"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                tensor_stream=tensor_stream,
                max_new_tokens=num_tokens or self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_logits=True,
            )

        generated_ids = outputs.sequences
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

    def test_generate_from_image_text(self):
        image = _load_red_dot_image()
        if image is None:
            pytest.skip("PIL.Image is required for Isaac generation tests.")

        messages = [
            {"role": "user", "content": "Describe this image:"},
            {"role": "user", "content": "<image>"},
        ]
        generated_text = self._generate_from_messages(messages, [image])
        expected_fragment = "The image is a close-up photograph of a red cross symbol."
        assert expected_fragment in generated_text

    def test_generate_from_text_only(self):
        document = [
            {
                "type": "text",
                "content": "What is the pythogorean theorem?",
                "role": "user",
            }
        ]
        messages, _ = document_to_messages(document)
        generated_text = self._generate_from_messages(messages, [], num_tokens=100)
        expected_fragmenet = "The Pythagorean theorem is a fundamental principle in geometry that relates the lengths of the sides of a right-angled triangle. Let's break down the theorem step by step:"
        assert expected_fragmenet in generated_text

    def test_vqa_from_image(self):
        document = [
            {
                "type": "image",
                "content": "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/refs/heads/main/huggingface/assets/example.webp",
                "role": "user",
            },
            {
                "type": "text",
                "content": "Is it safe to cross the street at this moment?",
                "role": "user",
            },
        ]
        messages, images = document_to_messages(document)
        generated_text = self._generate_from_messages(messages, images, num_tokens=256)
        expected_response = "\nNo, it is not safe to cross the street at this moment. The traffic light for pedestrians is red, indicating that it is not safe to cross."
        assert generated_text == expected_response
