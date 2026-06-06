# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch Gemma4Unified model."""

import copy
import string
import tempfile
import unittest
from unittest.mock import patch

import pytest
from parameterized import parameterized

from transformers import (
    AutoTokenizer,
    Gemma4UnifiedConfig,
    Gemma4UnifiedTextConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    require_torch_multi_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForCausalLM,
        Gemma4UnifiedForCausalLM,
        Gemma4UnifiedForConditionalGeneration,
        Gemma4UnifiedModel,
        Gemma4UnifiedProcessor,
        Gemma4UnifiedTextModel,
        PreTrainedModel,
        set_seed,
    )
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedRMSNorm


def _normalize_text(text):
    text = text.lower()
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    return text


class Gemma4UnifiedTextModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Gemma4UnifiedTextConfig
        base_model_class = Gemma4UnifiedTextModel
        causal_lm_class = Gemma4UnifiedForCausalLM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hidden_layers = 4  # override to correctly test sharing cache pattern
        self.num_kv_shared_layers = 2  # important to override
        self.layer_types = [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]  # similarly we want to test sharing on both types
        self.global_head_dim = self.head_dim  # gemma4 use a different head_dim for full and sliding layers

        # Test if bidirectional image mask path works
        self.use_bidirectional_attention = "vision"


@require_torch
class Gemma4UnifiedTextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Gemma4UnifiedTextModelTester
    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = Gemma4UnifiedForCausalLM if is_torch_available() else None

    @unittest.skip("We need 4 layers to correctly test cache sharing.")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip("Gemma4Unified uses different rope per layer type, which is not compatible with this test")
    def test_model_rope_scaling_frequencies(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("Gemma4Unified uses different rope per layer type, which is not compatible with this test")
    def test_model_rope_scaling_from_config(self):
        pass

    @unittest.skip(
        "Flaky on CI, but not locally on Mac. If model is set to fp32 instead of bf16, not flaky anymore."
        "TODO Cyril/Anton: investigate where the loss of precision between bf16 and fp32 comes from."
    )
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        "Same as gemma4: Fails after fully removing the unused weights, even if `forward` is exactly the same. Investigate why."
    )
    def test_tp_generation_quantized(self):
        pass

    def test_flash_attn_2_equivalence(self):
        """Override: Exchange RMS norm with identity as it creates too big shifts otherwise"""

        def identity_forward(self, hidden_states):
            return hidden_states

        with patch.object(Gemma4UnifiedRMSNorm, "forward", identity_forward):
            super().test_flash_attn_2_equivalence()

    def flash_attn_inference_equivalence(
        self, attn_implementation: str, padding_side: str, atol: float = 4e-2, rtol: float = 4e-2
    ) -> None:
        """Override: Exchange RMS norm with identity as it creates too big shifts otherwise"""

        def identity_forward(self, hidden_states):
            return hidden_states

        with patch.object(Gemma4UnifiedRMSNorm, "forward", identity_forward):
            super().flash_attn_inference_equivalence(attn_implementation, padding_side, atol, rtol)


class Gemma4UnifiedAudio2TextModelTester:
    def __init__(
        self,
        parent,
        image_token_id=4,
        boi_token_id=5,
        eoi_token_id=6,
        audio_token_id=7,
        boa_token_id=8,
        eoa_token_index=9,
        video_token_id=10,
        seq_length=50,
        audio_seq_length=50,
        audio_num_channels=32,
        is_training=True,
        audio_config={"audio_embed_dim": 32},
    ):
        self.parent = parent
        self.image_token_id = image_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.audio_token_id = audio_token_id
        self.boa_token_id = boa_token_id
        self.eoa_token_index = eoa_token_index
        self.video_token_id = video_token_id
        self.llm_tester = Gemma4UnifiedTextModelTester(self.parent)
        self.llm_tester.use_bidirectional_attention = None
        self.text_config = self.llm_tester.get_config()
        self.audio_config = audio_config
        self.seq_length = seq_length
        self.audio_seq_length = audio_seq_length
        self.audio_num_channels = audio_num_channels
        self.pad_token_id = self.text_config.pad_token_id

        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.num_attention_heads = self.text_config.num_attention_heads
        self.is_training = is_training

        self.batch_size = 3
        self.encoder_seq_length = seq_length

    def get_config(self):
        return Gemma4UnifiedConfig(
            text_config=self.text_config,
            vision_config=None,
            audio_config=self.audio_config,
            image_token_id=self.image_token_id,
            boi_token_id=self.boi_token_id,
            eoi_token_id=self.eoi_token_id,
            audio_token_id=self.audio_token_id,
            boa_token_id=self.boa_token_id,
            eoa_token_index=self.eoa_token_index,
            video_token_id=self.video_token_id,
        )

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, self.audio_seq_length, self.audio_num_channels])
        input_features_mask = torch.ones(self.batch_size, self.audio_seq_length, dtype=torch.bool, device=torch_device)
        config = self.get_config()
        return config, input_features, input_features_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_features, input_features_mask = self.prepare_config_and_inputs()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # Ensure no tokens accidentally match special token IDs
        for token_id in [config.image_token_id, config.video_token_id, config.audio_token_id]:
            input_ids[input_ids == token_id] = self.pad_token_id

        # For the unified model, there is no subsampling.
        # We need as many placeholder tokens as audio features.
        num_audio_tokens = self.audio_seq_length
        input_ids[:, :num_audio_tokens] = config.audio_token_id

        inputs_dict = {
            "input_features": input_features,
            "input_features_mask": input_features_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": torch.zeros_like(input_ids),
        }
        return config, inputs_dict


@require_torch
class Gemma4UnifiedAudio2TextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Gemma4UnifiedModel, Gemma4UnifiedForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (Gemma4UnifiedForConditionalGeneration,) if is_torch_available() else ()
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = Gemma4UnifiedAudio2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gemma4UnifiedConfig, hidden_size=37)
        self.skip_mm_output_format()

    def skip_mm_output_format(self):
        skippable_tests = [
            "test_get_image_features_hidden_states",
            "test_get_image_features_attentions",
            "test_get_video_features_hidden_states",
            "test_get_video_features_attentions",
            "test_get_audio_features_hidden_states",
            "test_get_audio_features_attentions",
            "test_get_image_features_output",
            "test_get_video_features_output",
            "test_get_audio_features_output",
        ]

        for test in skippable_tests:
            if self._testMethodName.startswith(test):
                self.skipTest(reason="Gemma4 unified does not collect any hidden states or attentions (no mm tower)")

    @unittest.skip("We need 4 layers to correctly test cache sharing.")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip("Conversions only happen when a vision embedder is included!")
    def test_reverse_loading_mapping(self, check_keys_were_modified=False, skip_base_model=False):
        pass

    def flash_attn_inference_equivalence(
        self, attn_implementation: str, padding_side: str, atol: float = 4e-2, rtol: float = 4e-2
    ) -> None:
        """Override: Exchange RMS norm with identity as it creates too big shifts otherwise"""

        def identity_forward(self, hidden_states):
            return hidden_states

        with patch.object(Gemma4UnifiedRMSNorm, "forward", identity_forward):
            super().flash_attn_inference_equivalence(attn_implementation, padding_side, atol, rtol)


class Gemma4UnifiedVision2TextModelTester:
    def __init__(
        self,
        parent,
        mm_tokens_per_image=2,
        image_token_id=4,
        video_token_id=7,
        audio_token_id=8,
        boi_token_id=5,
        eoi_token_id=6,
        seq_length=25,
        is_training=True,
        vision_config={
            "use_labels": True,
            "mm_embed_dim": 64,
            "output_proj_dims": 64,
            "image_size": 20,
            "patch_size": 5,
            "num_channels": 3,
            "is_training": True,
            "initializer_range": 0.02,
            "pooling_kernel_size": 2,
        },
    ):
        self.parent = parent
        # `image_token_id` is set to 0 to pass "resize_embeddings" test, do not modify
        self.mm_tokens_per_image = mm_tokens_per_image
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.llm_tester = Gemma4UnifiedTextModelTester(self.parent)
        self.text_config = self.llm_tester.get_config()
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.pad_token_id = self.text_config.pad_token_id

        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.num_attention_heads = self.text_config.num_attention_heads
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]
        self.encoder_seq_length = seq_length

    def get_config(self):
        return Gemma4UnifiedConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
            boi_token_id=self.boi_token_id,
            eoi_token_id=self.eoi_token_id,
            mm_tokens_per_image=self.mm_tokens_per_image,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        # (num_images, max_num_patches, model_patch_size * model_patch_size * num_channels)
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["image_size"],
                config.vision_config.model_patch_size
                * config.vision_config.model_patch_size
                * self.vision_config["num_channels"],
            ]
        )
        # (num_images, max_num_patches, 2) for height/width positions. Let it be all ones for testign
        pixel_position_ids = torch.ones(self.vision_config["image_size"], device=torch_device, dtype=torch.long)
        pixel_position_ids = pixel_position_ids[None, :, None].repeat(self.batch_size, 1, 2)

        return config, pixel_values, pixel_position_ids

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, pixel_position_ids = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # Ensure no tokens accidentally match special token IDs
        for token_id in [config.image_token_id, config.video_token_id, config.audio_token_id]:
            input_ids[input_ids == token_id] = self.pad_token_id

        num_image_tokens = 1
        pixel_values = pixel_values[:, :num_image_tokens, :]
        pixel_position_ids = pixel_position_ids[:, :num_image_tokens, :]
        input_ids[:, :num_image_tokens] = config.image_token_id

        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == config.image_token_id] = 1

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_position_ids": pixel_position_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
        }
        return config, inputs_dict


@require_torch
class Gemma4UnifiedVision2TextModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Gemma4UnifiedModel, Gemma4UnifiedForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (Gemma4UnifiedForConditionalGeneration,) if is_torch_available() else ()
    additional_model_inputs = ["mm_token_type_ids"]
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = Gemma4UnifiedVision2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gemma4UnifiedConfig, hidden_size=37)
        self.skip_mm_output_format()

    def skip_mm_output_format(self):
        skippable_tests = [
            "test_get_image_features_hidden_states",
            "test_get_image_features_attentions",
            "test_get_video_features_hidden_states",
            "test_get_video_features_attentions",
            "test_get_audio_features_hidden_states",
            "test_get_audio_features_attentions",
            "test_get_image_features_output",
            "test_get_video_features_output",
            "test_get_audio_features_output",
        ]

        for test in skippable_tests:
            if self._testMethodName.startswith(test):
                self.skipTest(reason="Gemma4 unified does not collect any hidden states or attentions (no mm tower)")

    @unittest.skip("We need 4 layers to correctly test cache sharing.")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip("We use vision based masks that use `block_sequence_ids` which force mask materialization")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def flash_attn_inference_equivalence(
        self, attn_implementation: str, padding_side: str, atol: float = 4e-2, rtol: float = 4e-2
    ) -> None:
        """
        Overriden to allow passing image position ids as it's mandatory in the vision portion
        #
        Exchange RMS norm with identity as it creates too big shifts otherwise
        """

        def identity_forward(self, hidden_states):
            return hidden_states

        with patch.object(Gemma4UnifiedRMSNorm, "forward", identity_forward):
            if not self.has_attentions:
                self.skipTest(reason="Model architecture does not support attentions")

            # This flag is used to know if the test was skipped for all `self.all_model_classes` or not
            _has_run_at_least_one_model = False

            for model_class in self.all_model_classes:
                # Custom kernel which needs the mask interface to be properly usable on these models
                if not model_class._supports_attention_backend and not attn_implementation.startswith(
                    "flash_attention"
                ):
                    continue

                # Set seed for deterministic test - ensures reproducible model initialization and inputs
                set_seed(42)
                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

                # flash attention variants does not always support arbitrary headim
                config = self._prepare_config_headdim(config, 16)

                # forcing the prefill size to go over sliding window size to check for SWA correctness
                if getattr(config, "sliding_window", None):
                    config.sliding_window = 2

                model = model_class(config)
                if not all(
                    submodel._supports_flash_attn
                    for submodel in model.modules()
                    if isinstance(submodel, PreTrainedModel)
                ):
                    continue

                # Some models only support a sub set of all FA implementations
                valid_fa_implementations = model._compatible_flash_implementations
                if valid_fa_implementations is not None and attn_implementation not in valid_fa_implementations:
                    continue

                # If we end up here, at least one model class was not skipped
                _has_run_at_least_one_model = True
                with tempfile.TemporaryDirectory() as tmpdirname:
                    # Save the model so we can reload with correct attention
                    model.save_pretrained(tmpdirname)

                    # Create first inputs without attention mask
                    main_input = inputs_dict[model.main_input_name]
                    # Only keep first batch sequence
                    if isinstance(main_input, torch.Tensor):
                        main_input = main_input[:1]
                        # Fix the dtype
                        if torch.is_floating_point(main_input):
                            main_input = main_input.to(torch.bfloat16)
                    first_inputs = {model.main_input_name: main_input, "output_hidden_states": True}
                    # Some models have main input name which is different from input_ids, but require input_ids... e.g. BarkFine
                    if model.main_input_name != "input_ids" and "input_ids" in inputs_dict:
                        first_inputs["input_ids"] = inputs_dict["input_ids"][:1]
                    # If we have some pixel values, use them as well
                    if model.main_input_name != "pixel_values" and "pixel_values" in inputs_dict:
                        # NOTE: this fixes qwen2_5_vl/omni because test break w/ pixel values
                        if "image_grid_thw" in inputs_dict:
                            continue
                        first_inputs["pixel_values"] = inputs_dict["pixel_values"][:1].to(torch.bfloat16)
                    # Some VLMs require image_sizes alongside pixel_values, e.g. lighton_ocr, llava_onevision
                    if "image_sizes" in inputs_dict:
                        first_inputs["image_sizes"] = inputs_dict["image_sizes"][:1]
                    # Key change: Allow image position ids to be passed as well
                    if "image_position_ids" in inputs_dict:
                        first_inputs["image_position_ids"] = inputs_dict["image_position_ids"][:1]
                    if model.config.is_encoder_decoder:
                        decoder_input_ids = inputs_dict.get("decoder_input_ids", first_inputs.get("input_ids"))
                        if decoder_input_ids is not None:
                            first_inputs["decoder_input_ids"] = decoder_input_ids[:1]

                    # Create attention mask with padding
                    dummy_attention_mask = inputs_dict.get("attention_mask", None)
                    if dummy_attention_mask is not None:
                        dummy_attention_mask = dummy_attention_mask[:1]
                        if padding_side == "left":
                            dummy_attention_mask[:, 1:] = 1
                            dummy_attention_mask[:, 0] = 0
                        else:
                            dummy_attention_mask[:, :-1] = 1
                            dummy_attention_mask[:, -1] = 0

                    # Create second inputs with attention mask and padding
                    second_inputs = copy.deepcopy(first_inputs)
                    if dummy_attention_mask is not None:
                        second_inputs["attention_mask"] = dummy_attention_mask
                        if model.config.is_encoder_decoder:
                            second_inputs["decoder_attention_mask"] = dummy_attention_mask

                    # Use prepare for class to account for special attributes (e.g. in QnA models)
                    first_inputs = self._prepare_for_class(first_inputs, model_class)
                    first_inputs = {
                        k: v.to(torch_device) if isinstance(v, torch.Tensor) else v for k, v in first_inputs.items()
                    }
                    second_inputs = self._prepare_for_class(second_inputs, model_class)
                    second_inputs = {
                        k: v.to(torch_device) if isinstance(v, torch.Tensor) else v for k, v in second_inputs.items()
                    }

                    model = model_class.from_pretrained(
                        tmpdirname, dtype=torch.bfloat16, attn_implementation="eager", device_map=torch_device
                    )

                    def _get_output_logits(outputs):
                        if "hidden_states" in outputs:
                            return outputs.hidden_states[-1]
                        elif model.config.is_encoder_decoder:
                            return outputs.decoder_hidden_states[-1]
                        elif "logits_per_image" in outputs:
                            return outputs.logits_per_image
                        elif "logits_per_video" in outputs:
                            return outputs.logits_per_video
                        else:
                            return outputs.logits

                    # First run without attention mask
                    outputs = model(**first_inputs)
                    logits_1_eager = _get_output_logits(outputs)
                    # Second run with attention mask and padding
                    outputs = model(**second_inputs)
                    logits_2_eager = _get_output_logits(outputs)

                    # Switch to FA
                    del model
                    model = model_class.from_pretrained(
                        tmpdirname,
                        dtype=torch.bfloat16,
                        attn_implementation=attn_implementation,
                        device_map=torch_device,
                    )
                    outputs = model(**first_inputs)
                    logits_1_fa = _get_output_logits(outputs)
                    # Second run with attention mask and padding
                    outputs = model(**second_inputs)
                    logits_2_fa = _get_output_logits(outputs)

                    # Check the results
                    torch.testing.assert_close(logits_1_eager, logits_1_fa, atol=atol, rtol=rtol)
                    if padding_side == "left":
                        torch.testing.assert_close(logits_2_eager[1:], logits_2_fa[1:], atol=atol, rtol=rtol)
                    else:
                        torch.testing.assert_close(logits_2_eager[:-1], logits_2_fa[:-1], atol=atol, rtol=rtol)

            # In this case, the test should appear as skipped, not successful
            if not _has_run_at_least_one_model:
                self.skipTest(
                    f"Model architecture does not support {attn_implementation}, or setting its attention dynamically"
                )


@slow
@require_torch_accelerator
@unittest.skip(reason="Update after release")  # TODO(vasqu)
class Gemma4UnifiedIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_name = "gg-hf-gu/gemma-4-12B-it"
        self.processor = Gemma4UnifiedProcessor.from_pretrained(self.model_name)

        self.url1 = url_to_local_path(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        self.url2 = url_to_local_path(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg"
        )
        self.messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.url1},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_with_image(self):
        model = Gemma4UnifiedForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 8): ['This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean and a blue sky** in the background'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_model_with_image_batch(self):
        model = Gemma4UnifiedForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)

        messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": self.url1,
                    },
                    {"type": "image", "url": self.url2},
                    {"type": "text", "text": "Are these images identical?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            [self.messages, messages_2],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            processor_kwargs={"padding": True},
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", (8, 0)): [
                    "This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean and a blue sky** in the background",
                    "No, these images are not identical.\n\nThe first image is a photograph of a **cow** standing on a beach under a blue sky.\n\n",
                ],
                ("cuda", (8, 6)): [
                    "This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean and a blue sky** in the background",
                    "No, these images are not identical.\n\nThe first image is a photograph of a **brown and white cow standing on a beach** under a blue",
                ],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_model_multiimage(self):
        model = Gemma4UnifiedForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.url2},
                    {"type": "text", "text": "What do you see here?"},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            processor_kwargs={"padding": True},
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)
        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 8): ['Based on the image, here is a description of what I see:\n\n**Foreground & Street Scene:**\n* **Traffic Sign:** The most prominent'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    @require_torch_multi_gpu
    def test_model_text_only_multigpu(self):
        """Accelerate destroys the input dict `shared_kv_states` if it's not passed as kwarg and part of
        `_skip_keys_device_placement`, so test this to avoid regresions.
        """
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a poem about Machine Learning."}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", (8, 0)): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
                ("cuda", (8, 6)): ['## The Algorithmic Mind\n\nA tapestry of data, vast and deep,\nWhere silent numbers in their slumber sleep.\nA sea of text'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_model_text_only(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a poem about Machine Learning."}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", (8, 0)): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
                ("cuda", (8, 6)): ['## The Algorithmic Mind\n\nA tapestry of data, vast and deep,\nWhere silent numbers in their slumber sleep.\nA sea of text'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_states_sharing_with_and_without_cache(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Who are you? What can you do?"}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)
        input_size = inputs.input_ids.shape[-1]

        # With and without cache generatiom should share kv states the same way
        output_with_cache = model.generate(**inputs, max_new_tokens=30, do_sample=False, use_cache=True)
        output_without_cache = model.generate(**inputs, max_new_tokens=30, do_sample=False, use_cache=False)

        output_text_with_cache = tokenizer.batch_decode(output_with_cache[:, input_size:], skip_special_tokens=True)
        output_text_without_cache = tokenizer.batch_decode(
            output_without_cache[:, input_size:], skip_special_tokens=True
        )

        self.assertEqual(output_text_with_cache, output_text_without_cache)

    # Note: we do not test FA2 as the head dim is 512 on some layers, which is not compatible with the kernels
    @parameterized.expand([("sdpa",), ("eager",)])
    def test_generation_beyond_sliding_window(self, attn_implementation: str):
        """Test that we can correctly generate beyond the sliding window. Outputs for every attention functions
        should be coherent and identical.
        """

        input_text = [
            "This is a nice place. " * 800 + "I really enjoy the scenery,",  # This is larger than 4096 tokens
            "A list of colors: red, blue",  # This will almost all be padding tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding="left")
        input_text = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": item}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in input_text
        ]
        inputs = tokenizer(input_text, padding=True, return_tensors="pt").to(torch_device)

        model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map=torch_device,
            attn_implementation=attn_implementation,
        )

        # Make sure prefill is larger than sliding window
        input_size = inputs.input_ids.shape[-1]
        self.assertTrue(input_size > model.config.get_text_config().sliding_window)

        out = model.generate(**inputs, max_new_tokens=16, do_sample=False, cache_implementation="static")
        output_text = tokenizer.batch_decode(out[:, input_size:])

        EXPECTED_COMPLETIONS = Expectations(
            {
                ("cuda", 8): [
                    "That sounds lovely! It seems like you're really enjoying the place you'",
                    "Here are a few ways you could use or expand upon that list, depending on",
                ]
            }
        )
        self.assertEqual(output_text, EXPECTED_COMPLETIONS.get_expectation())

    def test_model_with_audio(self):
        model = Gemma4UnifiedForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)

        audio_url = url_to_local_path(
            "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav"
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this:"},
                    {"type": "audio", "url": audio_url},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 8): ["come on dude you got a tattoo"],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(_normalize_text(output_text[0]), EXPECTED_TEXT[0])

    def test_model_with_audio_batch(self):
        model = Gemma4UnifiedForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)

        audio_url1 = url_to_local_path(
            "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav"
        )
        audio_url2 = url_to_local_path(
            "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"
        )
        messages_1 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this:"},
                    {"type": "audio", "url": audio_url1},
                ],
            },
        ]
        messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Are these audio clips saying the same thing?"},
                    {"type": "audio", "url": audio_url2},
                    {"type": "audio", "url": audio_url1},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            [messages_1, messages_2],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            processor_kwargs={"padding": True},
        ).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 8): ["come on dude you got a tattoo", "the first audio clip is a speech and the"],
            }
        )
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(_normalize_text(output_text[0]), EXPECTED_TEXT[0])
        self.assertEqual(_normalize_text(output_text[1]), EXPECTED_TEXT[1])

    @pytest.mark.torch_export_test
    def test_export_text_only(self):
        from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM

        model = Gemma4UnifiedForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        exportable_module = TorchExportableModuleForDecoderOnlyLM(
            model, batch_size=1, max_cache_len=1024, device=torch_device
        )
        exported_program = exportable_module.export(
            input_ids=torch.tensor([[1]], device=torch_device, dtype=torch.long),
        )

        # Test generation with the exported model
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France?"}],
            tokenize=False,
            add_generation_prompt=True,
        )

        max_new_tokens_to_generate = 20
        # Generate text with the exported model
        export_generated_text = TorchExportableModuleForDecoderOnlyLM.generate(
            exported_program, tokenizer, prompt, max_new_tokens=max_new_tokens_to_generate, device=torch_device
        )

        input_text = tokenizer(prompt, return_tensors="pt").to(torch_device)
        eager_outputs = model.generate(
            **input_text,
            max_new_tokens=max_new_tokens_to_generate,
            do_sample=False,  # Use greedy decoding to match the exported model
        )

        eager_generated_text = tokenizer.decode(eager_outputs[0], skip_special_tokens=True)
        self.assertEqual(export_generated_text, eager_generated_text)
