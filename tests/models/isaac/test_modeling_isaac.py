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

"""Testing suite for the Isaac model."""

import base64
import io
import os
import unittest
from functools import lru_cache
from pathlib import Path

import pytest
from huggingface_hub import is_offline_mode

from tests.generation.test_utils import (
    GenerationTesterMixin,
)
from tests.test_configuration_common import ConfigTester
from tests.test_pipeline_mixin import PipelineTesterMixin
from transformers import (
    IsaacConfig,
    IsaacForConditionalGeneration,
    IsaacModel,
    is_torch_available,
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


if is_vision_available():
    from PIL import Image
else:
    Image = None

from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch


BASE_MODEL_ID = os.environ.get("ISAAC_TEST_MODEL_ID", "PerceptronAI/Isaac-0.1-Base")
MODEL_ID = os.environ.get("ISAAC_TEST_MODEL_ID", "PerceptronAI/Isaac-0.1")

BASE_MODEL_REVISION = os.environ.get("ISAAC_TEST_MODEL_REVISION", "refs/pr/3") or None
MODEL_REVISION = os.environ.get("ISAAC_TEST_MODEL_REVISION", "refs/pr/5") or None

LOCAL_CHECKPOINT = os.environ.get("ISAAC_TEST_MODEL_PATH")
RED_DOT_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
ISAAC_IMAGE_TOKEN = "<|image_pad|>"


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


def pack_image_inputs(pixel_values, image_token_grids, image_token_offsets=None, image_token_lengths=None):
    batch_size, max_images, _, _ = pixel_values.shape
    device = pixel_values.device

    if image_token_offsets is None:
        image_token_offsets = torch.zeros((batch_size, max_images), device=device, dtype=torch.long)
    if image_token_lengths is None:
        image_token_lengths = image_token_grids[..., 0] * image_token_grids[..., 1]

    image_grid_thw = torch.zeros((batch_size, max_images, 3), device=device, dtype=torch.long)
    active_slots = image_token_grids.prod(dim=-1).gt(0)
    image_grid_thw[..., 0] = active_slots.to(dtype=torch.long)
    image_grid_thw[..., 1:] = image_token_grids

    image_metadata = torch.stack(
        (
            image_token_offsets.to(device=device, dtype=torch.long),
            image_token_lengths.to(device=device, dtype=torch.long),
        ),
        dim=-1,
    )

    return pixel_values, image_grid_thw, image_metadata


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
        self.expected_num_hidden_layers = num_hidden_layers + 1

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
        # Rely on eager attention so output_attentions tests remain compatible without flash attention.
        config._attn_implementation = "eager"
        config.text_config._attn_implementation = "eager"
        config.vision_attn_implementation = "eager"
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
        patch_size = self.vision_config["patch_size"]
        patch_dim = self.vision_config["num_channels"] * patch_size * patch_size
        num_image_patches = 4
        vision_patches = torch.randn(
            (self.batch_size, 1, num_image_patches, patch_dim), device=torch_device, dtype=torch.float32
        )
        image_token_grids = torch.tensor([[[2, 2]]] * self.batch_size, device=torch_device, dtype=torch.long)
        pixel_values, image_grid_thw, image_metadata = pack_image_inputs(
            pixel_values=vision_patches,
            image_token_grids=image_token_grids,
        )
        mm_token_type_ids = torch.zeros((self.batch_size, self.seq_length), device=torch_device, dtype=torch.long)
        mm_token_type_ids[:, :num_image_patches] = 1
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "image_metadata": image_metadata,
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

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        input_keys_to_ignore = [
            "decoder_input_ids",
            "decoder_attention_mask",
            "use_cache",
            "labels",
        ]

        filtered_inputs_dict = {
            k: v[:batch_size, ...]
            if isinstance(v, torch.Tensor) and k not in ["pixel_values", "image_grid_thw", "image_metadata"]
            else v
            for k, v in inputs_dict.items()
            if k not in input_keys_to_ignore
        }

        filtered_inputs_dict["pixel_values"] = inputs_dict["pixel_values"][:batch_size]
        filtered_inputs_dict["image_grid_thw"] = inputs_dict["image_grid_thw"][:batch_size]
        filtered_inputs_dict["image_metadata"] = inputs_dict["image_metadata"][:batch_size]

        text_gen_config = config.get_text_config(decoder=True)
        if text_gen_config.eos_token_id is not None and text_gen_config.pad_token_id is None:
            text_gen_config.pad_token_id = (
                text_gen_config.eos_token_id
                if isinstance(text_gen_config.eos_token_id, int)
                else text_gen_config.eos_token_id[0]
            )
        text_gen_config.eos_token_id = None
        text_gen_config.forced_eos_token_id = None

        return config, filtered_inputs_dict

    @pytest.mark.generate
    def test_left_padding_compatibility(self):
        _, inputs_dict = self.prepare_config_and_inputs_for_generate()
        mm_token_type_ids = inputs_dict["mm_token_type_ids"]
        pad_size = (mm_token_type_ids.shape[0], 32)
        padded_mm_token_type_ids = torch.cat(
            (torch.zeros(pad_size, dtype=mm_token_type_ids.dtype, device=torch_device), mm_token_type_ids), dim=1
        )

        super().test_left_padding_compatibility(
            unpadded_custom_inputs={"mm_token_type_ids": mm_token_type_ids},
            padded_custom_inputs={"mm_token_type_ids": padded_mm_token_type_ids},
        )

    @unittest.skip(reason="Assisted decoding not supported; Qwen3 backbone does not implement returning attentions")
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip(reason="Assisted decoding not supported; Qwen3 backbone does not implement returning attentions")
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    @unittest.skip(reason="Unsupported")
    def test_flash_attn_kernels_inference_equivalence(self):
        pass

    @unittest.skip(reason="Isaac is image-only.")
    def test_get_video_features_output_0(self):
        pass

    @unittest.skip(reason="Isaac is image-only.")
    def test_get_video_features_output_1(self):
        pass

    @unittest.skip(reason="Isaac is image-only.")
    def test_get_video_features_output_2(self):
        pass

    @unittest.skip(reason="Isaac is image-only.")
    def test_get_video_features_hidden_states(self):
        pass

    @unittest.skip(reason="Isaac is image-only.")
    def test_get_video_features_attentions(self):
        pass


@require_torch
@require_vision
@slow
@require_flash_attn
class IsaacGenerationIntegrationTest(unittest.TestCase):
    max_new_tokens = 25
    dtype = torch.bfloat16

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = _base_reference_checkpoint_or_skip()
        self.hf_config = IsaacConfig.from_pretrained(self.checkpoint, revision=BASE_MODEL_REVISION)
        self.processor = IsaacProcessor.from_pretrained(self.checkpoint, revision=BASE_MODEL_REVISION, do_pad=True)
        self.tokenizer = self.processor.tokenizer
        self.hf_config.vision_config._attn_implementation = "flash_attention_2"
        self.hf_config.vision_config.attn_implementation = "flash_attention_2"
        self.model = IsaacForConditionalGeneration.from_pretrained(
            self.checkpoint, config=self.hf_config, revision=BASE_MODEL_REVISION
        )
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    def test_generate_from_image_text(self):
        image = _load_red_dot_image()
        if image is None:
            pytest.skip("PIL.Image is required for Isaac generation tests.")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image:"},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1] :]
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        expected_fragment = "The image is a close-up photograph of a red cross symbol."
        assert expected_fragment in generated_text

    def test_generate_from_text_only(self):
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the pythogorean theorem?"}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1] :]
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        expected_fragmenet = "The Pythagorean theorem is a fundamental principle in geometry that relates the lengths of the sides of a right-angled triangle. Let's break it down step by step:"
        assert expected_fragmenet in generated_text

    def test_vqa_from_image(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/refs/heads/main/huggingface/assets/example.webp",
                    },
                    {"type": "text", "text": "Is it safe to cross the street at this moment?"},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1] :]
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        expected_response = "\nNo, it is not safe to cross the street at this moment. The traffic light for pedestrians is red, indicating that it is not safe to cross."
        assert generated_text == expected_response

    def test_logit_equivalence(self):
        image = _load_red_dot_image()
        if image is None:
            pytest.skip("PIL.Image is required for Isaac generation tests.")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image:"},
                    {"type": "image", "image": image},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
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
            "mean": 0.0608877803,
            "std": 2.8308793244,
            "min": -12.0625,
            "max": 31.0,
            "sum": 92510.4578057677,
            "l2_norm": 3490.2146142251,
        }
        assert logit_stats == expected_logit_stats

    def test_batched_generation_matches_individual(self):
        image = _load_red_dot_image()
        if image is None:
            pytest.skip("PIL.Image is required for Isaac generation tests.")

        conversations = [
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is the pythogorean theorem?"}],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image:"},
                        {"type": "image", "image": image},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/refs/heads/main/huggingface/assets/example.webp",
                        },
                        {"type": "text", "text": "Is it safe to cross the street at this moment?"},
                    ],
                }
            ],
        ]

        single_inputs = [
            self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            for conversation in conversations
        ]
        batch_inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
            processor_kwargs={"padding_side": "left"},
        )
        batch_input_ids = batch_inputs["input_ids"]
        max_length = batch_input_ids.shape[1]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = getattr(self.processor, "pad_token_id", 0)

        sample_lengths = [single_input["input_ids"].squeeze(0).shape[0] for single_input in single_inputs]
        for i, (single_input, batch_ids, single_len) in enumerate(zip(single_inputs, batch_input_ids, sample_lengths)):
            single_ids = single_input["input_ids"].squeeze(0)
            torch.testing.assert_close(batch_ids[-single_len:], single_ids)

            batch_modality_row = batch_inputs["mm_token_type_ids"][i]
            expected_modality = torch.full(
                (max_length,),
                batch_modality_row[-1].item(),
                dtype=batch_modality_row.dtype,
                device=batch_modality_row.device,
            )
            expected_modality[-single_len:] = single_input["mm_token_type_ids"].squeeze(0)
            torch.testing.assert_close(batch_modality_row, expected_modality)

            if batch_inputs["image_grid_thw"] is not None:
                batch_image_mask = batch_inputs["image_grid_thw"][i, :, 0].eq(1)
                expected_image_count = int(batch_image_mask.sum().item())
                if single_input["image_grid_thw"] is None:
                    assert expected_image_count == 0
                else:
                    single_image_mask = single_input["image_grid_thw"][0, :, 0].eq(1)
                    assert expected_image_count == int(single_image_mask.sum().item())
                    if expected_image_count > 0:
                        batch_image_grid_thw = batch_inputs["image_grid_thw"][i, batch_image_mask]
                        single_image_grid_thw = single_input["image_grid_thw"][0, single_image_mask]
                        batch_image_metadata = batch_inputs["image_metadata"][i, batch_image_mask]
                        single_image_metadata = single_input["image_metadata"][0, single_image_mask]

                        torch.testing.assert_close(batch_image_grid_thw, single_image_grid_thw)
                        torch.testing.assert_close(batch_image_metadata, single_image_metadata)

                        for batch_pixel_values, single_pixel_values, grid_thw in zip(
                            batch_inputs["pixel_values"][i, batch_image_mask],
                            single_input["pixel_values"][0, single_image_mask],
                            batch_image_grid_thw,
                            strict=True,
                        ):
                            valid_patch_count = int((grid_thw[1] * grid_thw[2]).item())
                            torch.testing.assert_close(
                                batch_pixel_values[:valid_patch_count],
                                single_pixel_values[:valid_patch_count],
                            )

            if single_len == max_length:
                continue

            pad_span = batch_ids[: max_length - single_len]
            assert torch.all(pad_span == pad_id), f"sample {i} left pad span not padded with pad id"
            torch.testing.assert_close(
                batch_inputs["attention_mask"][i],
                batch_ids.ne(pad_id).long(),
            )

        single_texts = []
        for single_input in single_inputs:
            single_input = single_input.to(self.device, dtype=self.dtype)
            with torch.no_grad():
                outputs = self.model.generate(
                    **single_input,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                )
            generated_ids = outputs.sequences[:, single_input["input_ids"].shape[1] :]
            single_texts.append(self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0])

        batch_inputs = batch_inputs.to(self.device, dtype=self.dtype)
        with torch.no_grad():
            batch_outputs = self.model.generate(
                **batch_inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
        batch_generated_ids = batch_outputs.sequences[:, batch_inputs["input_ids"].shape[1] :]
        batch_texts = self.processor.batch_decode(batch_generated_ids, skip_special_tokens=True)
        assert len(batch_texts) == len(single_texts) == 3

        for i, (batch_text, single_text) in enumerate(zip(batch_texts, single_texts)):
            assert single_text in batch_text, f"batch[{i}] mismatch: {batch_text!r} vs single[{i}] {single_text!r}"

    def test_batched_beam_generation_matches_individual(self):
        image = _load_red_dot_image()
        if image is None:
            pytest.skip("PIL.Image is required for Isaac generation tests.")

        conversations = [
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What is the pythogorean theorem?"}],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image:"},
                        {"type": "image", "image": image},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "url": "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/refs/heads/main/huggingface/assets/example.webp",
                        },
                        {"type": "text", "text": "Is it safe to cross the street at this moment?"},
                    ],
                }
            ],
        ]
        beam_kwargs = {"num_beams": 2}

        single_texts = []
        for conversation in conversations:
            single_input = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device, dtype=self.dtype)
            with torch.no_grad():
                outputs = self.model.generate(
                    **single_input,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    **beam_kwargs,
                )
            generated_ids = outputs.sequences[:, single_input["input_ids"].shape[1] :]
            single_texts.append(self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0])

        batch_inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
            processor_kwargs={"padding_side": "left"},
        ).to(self.device, dtype=self.dtype)
        with torch.no_grad():
            batch_outputs = self.model.generate(
                **batch_inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                **beam_kwargs,
            )
        batch_generated_ids = batch_outputs.sequences[:, batch_inputs["input_ids"].shape[1] :]
        batch_texts = self.processor.batch_decode(batch_generated_ids, skip_special_tokens=True)
        assert len(batch_texts) == len(single_texts) == 3

        for i, (batch_text, single_text) in enumerate(zip(batch_texts, single_texts)):
            assert single_text in batch_text, (
                f"beam batch[{i}] mismatch: {batch_text!r} vs single[{i}] {single_text!r}"
            )


@require_torch
@require_vision
@slow
@require_flash_attn
class IsaacBoxPointingIntegrationTest(unittest.TestCase):
    max_new_tokens = 256
    dtype = torch.bfloat16

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = _reference_checkpoint_or_skip()
        self.hf_config = IsaacConfig.from_pretrained(self.checkpoint, revision=MODEL_REVISION)
        # The current local slow fallback only supports padded packing for this checkpoint.
        self.processor = IsaacProcessor.from_pretrained(self.checkpoint, revision=MODEL_REVISION, do_pad=True)
        self.tokenizer = self.processor.tokenizer
        self.hf_config.vision_config._attn_implementation = "flash_attention_2"
        self.hf_config.vision_config.attn_implementation = "flash_attention_2"
        self.model = IsaacForConditionalGeneration.from_pretrained(
            self.checkpoint, config=self.hf_config, revision=MODEL_REVISION
        )
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    def test_hf_generate_box_points(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<hint>BOX</hint>"},
                    {
                        "type": "image",
                        "url": "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/refs/heads/main/huggingface/assets/example.webp",
                    },
                    {
                        "type": "text",
                        "text": "Determine whether it is safe to cross the street. Look for signage and moving traffic.",
                    },
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1] :]
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        _, points = self.processor.post_process_generation(generated_text, expected="box")
        assert len(points) == 1
        first_point = points[0]
        assert first_point.top_left.x < first_point.bottom_right.x
        assert first_point.top_left.y < first_point.bottom_right.y
        assert first_point.mention == "traffic light"
        assert first_point.top_left.x == 808
        assert first_point.top_left.y == 247
        assert first_point.bottom_right.x == 863
        assert first_point.bottom_right.y == 386

    def test_hf_generate_polygon_points(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<hint>POLYGON</hint>"},
                    {
                        "type": "image",
                        "url": "https://raw.githubusercontent.com/perceptron-ai-inc/perceptron/refs/heads/main/huggingface/assets/example.webp",
                    },
                    {
                        "type": "text",
                        "text": "Determine whether it is safe to cross the street. Look for signage and moving traffic.",
                    },
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[:, inputs["input_ids"].shape[1] :]
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        _, polygons = self.processor.post_process_generation(generated_text, expected="polygon")
        assert len(polygons) == 1
        first_polygon = polygons[0]
        xs = [point.x for point in first_polygon.points]
        ys = [point.y for point in first_polygon.points]
        expected_left, expected_top, expected_right, expected_bottom = 808, 247, 863, 386

        assert len(first_polygon.points) >= 3
        assert first_polygon.mention == "traffic light"
        assert min(xs) >= expected_left - 4
        assert max(xs) <= expected_right + 4
        assert min(ys) >= expected_top - 4
        assert max(ys) <= expected_bottom + 4
        assert max(xs) - min(xs) >= 35
        assert max(ys) - min(ys) >= 100
        assert any(abs(x - expected_left) <= 12 for x in xs)
        assert any(abs(x - expected_right) <= 12 for x in xs)
        assert any(abs(y - expected_top) <= 12 for y in ys)
        assert any(abs(y - expected_bottom) <= 12 for y in ys)
