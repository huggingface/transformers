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
import io
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
    AutoTokenizer,
    IsaacConfig,
    IsaacForConditionalGeneration,
    IsaacModel,
    PythonBackend,
    is_torch_available,
)
from transformers.image_utils import load_image
from transformers.masking_utils import eager_mask, sdpa_mask
from transformers.models.isaac.image_processing_isaac_fast import IsaacImageProcessorFast
from transformers.models.isaac.modeling_isaac import (
    document_mask_function_from_cu_seqlens,
    IsaacVisionAttention,
    IsaacVisionConfig,
)
from transformers.models.isaac.processing_isaac import IsaacProcessor
from transformers.testing_utils import (
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
    from perceptron.pointing.parser import extract_points
    from perceptron.tensorstream.tensorstream import TensorStream
else:
    TensorStream = None
    extract_points = None


require_tensorstream = pytest.mark.skipif(TensorStream is None, reason="TensorStream backend is not available")

BASE_MODEL_ID = os.environ.get("ISAAC_TEST_MODEL_ID", "PerceptronAI/Isaac-0.1-Base")
MODEL_ID = os.environ.get("ISAAC_TEST_MODEL_ID", "PerceptronAI/Isaac-0.1")

BASE_MODEL_REVISION = os.environ.get("ISAAC_TEST_MODEL_REVISION", "refs/pr/3") or None
MODEL_REVISION = os.environ.get("ISAAC_TEST_MODEL_REVISION", "refs/pr/5") or None

LOCAL_CHECKPOINT = os.environ.get("ISAAC_TEST_MODEL_PATH")
RED_DOT_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="


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


@require_torch
class IsaacDocumentMaskingTest(unittest.TestCase):
    def test_document_mask_function_from_cu_seqlens(self):
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
        mask_fn = document_mask_function_from_cu_seqlens(cu_seqlens)

        self.assertIsNotNone(mask_fn)
        # Same document (indices 1 and 2)
        self.assertTrue(mask_fn(0, 0, 1, 2))
        # Cross-document (index 1 in first doc, 3 in second doc)
        self.assertFalse(mask_fn(0, 0, 1, 3))
        # Same second document (indices 3 and 4)
        self.assertTrue(mask_fn(0, 0, 4, 3))

    def test_document_mask_function_materializes_with_masking_utils(self):
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
        self.assertTrue(torch.equal(sdpa, expected_bool))

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
        self.assertTrue(torch.equal(eager, expected_additive))


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


@lru_cache(maxsize=1)
def _load_red_dot_image():
    if Image is None:
        return None
    data = base64.b64decode(RED_DOT_B64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _base_reference_checkpoint_or_skip():
    if LOCAL_CHECKPOINT:
        resolved = Path(LOCAL_CHECKPOINT).expanduser()
        if not resolved.exists():
            pytest.skip(f"Local checkpoint path {resolved} does not exist.")
        return str(resolved)
    if is_offline_mode():
        pytest.skip("Offline mode: set ISAAC_TEST_MODEL_PATH to a local checkpoint to run these tests.")
    return BASE_MODEL_ID


def _reference_checkpoint_or_skip():
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
    def test_modality_tensor_requires_matching_shape(self):
        config, input_ids, attention_mask, _ = self.model_tester.prepare_config_and_inputs()
        model = IsaacModel(config).to(torch_device)
        model.eval()

        modality_tensor = torch.zeros(
            (self.model_tester.batch_size, self.model_tester.seq_length),
            device=torch_device,
            dtype=torch.long,
        )
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask, modality_tensor=modality_tensor)

        self.assertEqual(
            result.last_hidden_state.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, config.hidden_size),
        )

        bad_modality_tensor = torch.zeros(
            (self.model_tester.batch_size, self.model_tester.seq_length + 1),
            device=torch_device,
            dtype=torch.long,
        )
        with self.assertRaisesRegex(ValueError, "modality_tensor must have shape"):
            model(input_ids=input_ids, attention_mask=attention_mask, modality_tensor=bad_modality_tensor)

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

    @require_tensorstream
    def test_isaac_for_conditional_generation_initialization(self):
        config = self.model_tester.get_config()
        model = IsaacForConditionalGeneration(config)
        model.to(torch_device)

        self.assertTrue(hasattr(model, "model"))
        self.assertTrue(hasattr(model, "lm_head"))
        self.assertTrue(hasattr(model.model, "vision_embedding"))
        self.assertTrue(hasattr(model.model, "embed_fns"))

        input_ids = torch.randint(0, config.vocab_size, (1, 10), device=torch_device, dtype=torch.long)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, return_dict=True)
        self.assertEqual(outputs.logits.shape, (1, 10, config.vocab_size))

    @require_tensorstream
    def test_isaac_for_conditional_generation_loss_and_generate_flag(self):
        config = self.model_tester.get_config()
        model = IsaacForConditionalGeneration(config).to(torch_device)
        self.assertTrue(model.can_generate())

        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=torch_device)
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=torch_device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels, return_dict=True)
        self.assertIsNotNone(outputs.loss)
        self.assertEqual(outputs.loss.ndim, 0)
        self.assertEqual(outputs.logits.shape, (batch_size, seq_len, config.vocab_size))

    @require_vision
    @require_tensorstream
    def test_isaac_generation_with_tensor_stream(self):
        config = self.model_tester.get_config()
        tokenizer = SimpleIsaacTokenizer()
        image_processor = IsaacImageProcessorFast(
            patch_size=config.vision_config.patch_size,
            max_num_patches=config.vision_config.num_patches,
            pixel_shuffle_scale=config.vision_config.pixel_shuffle_scale_factor,
            rescale_factor=config.vision_rescale_factor,
        )
        processor = IsaacProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            config=config,
        )

        model = IsaacForConditionalGeneration(config).to(torch_device)
        model.eval()

        messages = [{"role": "user", "content": "Hello there!"}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        processed = processor(text=prompt, images=None, return_tensors="pt")

        input_ids = processed["input_ids"].to(torch_device)
        tensor_stream = processed["tensor_stream"].to(torch_device)
        generated = model.generate(
            input_ids=input_ids,
            tensor_stream=tensor_stream,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        self.assertEqual(generated.shape[0], 1)
        self.assertGreaterEqual(generated.shape[1], input_ids.shape[1])
        decoded_prompt = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
        self.assertIsInstance(decoded_prompt, str)
        self.assertNotEqual(decoded_prompt.strip(), "")


@require_torch
@require_flash_attn
class IsaacAttentionDtypeTest(unittest.TestCase):
    def _make_config(self):
        return IsaacVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_channels=3,
            num_patches=64,
            patch_size=4,
            attention_dropout=0.0,
            pixel_shuffle_scale_factor=1,
        )

    def _skip_if_no_cuda_bf16(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for flash attention dtype/parity tests.")
        if not torch.cuda.is_bf16_supported():
            pytest.skip("CUDA bfloat16 support required.")

    def test_flash_attention_matches_weight_dtype_bf16(self):
        self._skip_if_no_cuda_bf16()
        torch.manual_seed(0)

        device = torch.device("cuda")
        config = self._make_config()
        config._attn_implementation = "flash_attention_2"

        attn = IsaacVisionAttention(config).to(device=device, dtype=torch.bfloat16).eval()

        hidden_states = torch.randn(2, 4, config.hidden_size, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            attn_output, _ = attn(hidden_states)

        assert attn_output.dtype == attn.out_proj.weight.dtype
        assert attn_output.dtype == hidden_states.dtype

    def test_flash_attention_matches_weight_dtype_bf16_with_padding(self):
        self._skip_if_no_cuda_bf16()
        torch.manual_seed(0)

        device = torch.device("cuda")
        config = self._make_config()
        config._attn_implementation = "flash_attention_2"

        attn = IsaacVisionAttention(config).to(device=device, dtype=torch.bfloat16).eval()

        hidden_states = torch.randn(2, 4, config.hidden_size, device=device, dtype=torch.bfloat16)
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], device=device, dtype=torch.bool)

        with torch.no_grad():
            attn_output, _ = attn(hidden_states, attention_mask=attention_mask)

        assert attn_output.dtype == attn.out_proj.weight.dtype
        assert attn_output.dtype == hidden_states.dtype

    def test_flash_attention_matches_weight_dtype_bf16_with_cu_seqlens(self):
        self._skip_if_no_cuda_bf16()
        torch.manual_seed(0)

        device = torch.device("cuda")
        config = self._make_config()
        config._attn_implementation = "flash_attention_2"

        attn = IsaacVisionAttention(config).to(device=device, dtype=torch.bfloat16).eval()

        hidden_states = torch.randn(1, 5, config.hidden_size, device=device, dtype=torch.bfloat16)
        cu_seqlens = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)

        with torch.no_grad():
            attn_output, _ = attn(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=3)

        assert attn_output.dtype == attn.out_proj.weight.dtype
        assert attn_output.dtype == hidden_states.dtype

    def test_flash_attention_parity_with_sdpa_bf16(self):
        self._skip_if_no_cuda_bf16()
        torch.manual_seed(0)

        device = torch.device("cuda")
        config_sdpa = self._make_config()
        config_sdpa._attn_implementation = "sdpa"

        config_fa2 = self._make_config()
        config_fa2._attn_implementation = "flash_attention_2"

        attn_sdpa = IsaacVisionAttention(config_sdpa).to(device=device, dtype=torch.bfloat16).eval()
        attn_fa2 = IsaacVisionAttention(config_fa2).to(device=device, dtype=torch.bfloat16).eval()

        # Align weights so the only difference is the backend
        attn_fa2.load_state_dict(attn_sdpa.state_dict())

        hidden_states = torch.randn(2, 4, config_sdpa.hidden_size, device=device, dtype=torch.bfloat16)

        with torch.no_grad():
            out_sdpa, _ = attn_sdpa(hidden_states)
            out_fa2, _ = attn_fa2(hidden_states)

        torch.testing.assert_close(
            out_fa2.float(),
            out_sdpa.float(),
            rtol=1e-3,
            atol=1e-3,
            msg="FlashAttention2 output deviates from SDPA baseline beyond tolerance",
        )


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
        self.checkpoint = _base_reference_checkpoint_or_skip()
        self.hf_config = IsaacConfig.from_pretrained(self.checkpoint, revision=BASE_MODEL_REVISION)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, trust_remote_code=True, use_fast=False, revision=BASE_MODEL_REVISION
        )
        self.processor = create_isaac_processor(self.tokenizer, self.hf_config)
        self.hf_config.vision_config._attn_implementation = "flash_attention_2"
        self.hf_config.vision_config.attn_implementation = "flash_attention_2"
        self.model = IsaacForConditionalGeneration.from_pretrained(
            self.checkpoint, config=self.hf_config, revision=BASE_MODEL_REVISION
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

    def test_logit_equivalence(self):
        image = _load_red_dot_image()
        if image is None:
            pytest.skip("PIL.Image is required for Isaac generation tests.")
        image_bytes = base64.b64decode(RED_DOT_B64)
        pil_image = Image.open(io.BytesIO(image_bytes))
        images = []
        images.append(pil_image)
        num_tokens = 10

        messages = [
            {"role": "user", "content": "Describe this image:"},
            {"role": "user", "content": "<image>"},
        ]
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
        hf_logits = torch.cat(outputs.logits, dim=0)
        logit_stats = compute_logits_statistics(hf_logits)
        expected_logit_stats = {
            "shape": [10, 151936],
            "numel": 1519360,
            "mean": 0.0879677375,
            "std": 2.8382794404,
            "min": -12.125,
            "max": 31.0,
            "sum": 133654.661714755,
            "l2_norm": 3500.2090570868,
        }
        assert logit_stats == expected_logit_stats


@require_torch
@require_vision
@slow
@require_tensorstream
@require_flash_attn
class IsaacBoxPointingIntegrationTest(unittest.TestCase):
    max_new_tokens = 256
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

    def test_hf_generate_box_points(self):
        document = [
            {
                "type": "text",
                "content": "<hint>BOX</hint>",
                "role": "user",
            },
            {
                "type": "image",
                "content": "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/refs/heads/main/huggingface/assets/example.webp",
                "role": "user",
            },
            {
                "type": "text",
                "content": "Determine whether it is safe to cross the street. Look for signage and moving traffic.",
                "role": "user",
            },
        ]
        messages, images = document_to_messages(document, vision_token=self.hf_config.vision_token)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).strip()
        processor_output = self.processor(text=prompt, images=images, return_tensors="pt")
        tensor_stream = processor_output["tensor_stream"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                tensor_stream=tensor_stream,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences
        hf_generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        points = extract_points(hf_generated_text)
        assert len(points) == 1
        first_point = points[0]
        assert first_point.top_left.x < first_point.bottom_right.x
        assert first_point.top_left.y < first_point.bottom_right.y
        assert first_point.mention == "traffic light"
        assert first_point.top_left.x == 808
        assert first_point.top_left.y == 247
        assert first_point.bottom_right.x == 863
        assert first_point.bottom_right.y == 386
