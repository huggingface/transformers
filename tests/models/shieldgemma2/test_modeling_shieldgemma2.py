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
"""Testing suite for the PyTorch ShieldGemma2 model."""

import tempfile
import unittest

from transformers import (
    BitsAndBytesConfig,
    Gemma3TextConfig,
    ShieldGemma2Config,
    SiglipVisionConfig,
    is_torch_available,
)
from transformers.image_utils import load_image
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Gemma3ForConditionalGeneration,
        Gemma3Model,
        ShieldGemma2ForImageClassification,
        ShieldGemma2Processor,
    )


class ShieldGemma2ModelTester(VLMModelTester):
    config_class = ShieldGemma2Config
    text_config_class = Gemma3TextConfig
    vision_config_class = SiglipVisionConfig

    if is_torch_available():
        base_model_class = Gemma3Model
        conditional_generation_class = Gemma3ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("batch_size", 7)
        kwargs.setdefault("seq_length", 8)
        kwargs.setdefault("vocab_size", 99)
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("intermediate_size", 64)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("max_position_embeddings", 64)
        kwargs.setdefault("sliding_window", 8)
        kwargs.setdefault("layer_types", ["sliding_attention", "full_attention"])
        kwargs.setdefault("image_size", 8)
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("mm_tokens_per_image", 4)
        kwargs.setdefault("num_image_tokens", kwargs["mm_tokens_per_image"])
        kwargs.setdefault("image_token_index", 0)
        kwargs.setdefault("image_token_id", kwargs["image_token_index"])
        kwargs.setdefault("tie_word_embeddings", True)
        kwargs.setdefault("pad_token_id", 1)
        kwargs.setdefault("eos_token_id", 2)
        kwargs.setdefault("bos_token_id", 3)
        kwargs.setdefault("yes_token_index", 4)
        kwargs.setdefault("no_token_index", 5)
        super().__init__(parent, **kwargs)

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {
            self.image_token_index,
            self.yes_token_index,
            self.no_token_index,
        }

    def get_config(self):
        config = super().get_config()
        config.yes_token_index = self.yes_token_index
        config.no_token_index = self.no_token_index
        return config

    def create_attention_mask(self, input_ids):
        return input_ids.ne(self.pad_token_id).to(torch_device)

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[input_ids == config.image_token_id] = 1
        return {"token_type_ids": token_type_ids}

    def create_and_check_model(self, config, inputs_dict):
        model = ShieldGemma2ForImageClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**inputs_dict)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, 2))
        self.parent.assertEqual(result.probabilities.shape, (self.batch_size, 2))


@require_torch
class ShieldGemma2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ShieldGemma2ForImageClassification,) if is_torch_available() else ()
    _is_composite = True
    additional_model_inputs = ["pixel_values", "attention_mask", "token_type_ids"]

    test_attention_outputs = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # ShieldGemma2 does not compute its own loss, so never inject labels
        return super()._prepare_for_class(inputs_dict, model_class, return_labels=False)

    def setUp(self):
        self.model_tester = ShieldGemma2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ShieldGemma2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(config, inputs_dict)

    def test_attention_support_flags_match_underlying_model(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = ShieldGemma2ForImageClassification(config)

        for support_flag in (
            "_supports_flash_attn",
            "_supports_sdpa",
            "_supports_flex_attn",
            "_supports_attention_backend",
        ):
            self.assertEqual(
                getattr(ShieldGemma2ForImageClassification, support_flag), getattr(model.model, support_flag)
            )

    def test_sdpa_can_dispatch_composite_models(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = ShieldGemma2ForImageClassification(config)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)

            model_sdpa = ShieldGemma2ForImageClassification.from_pretrained(
                tmpdirname,
                attn_implementation="sdpa",
            )
            model_eager = ShieldGemma2ForImageClassification.from_pretrained(
                tmpdirname,
                attn_implementation="eager",
            )

        for loaded_model, expected_attn_implementation in ((model_sdpa, "sdpa"), (model_eager, "eager")):
            self.assertEqual(loaded_model.config._attn_implementation, expected_attn_implementation)
            self.assertEqual(loaded_model.model.config._attn_implementation, expected_attn_implementation)
            self.assertEqual(
                loaded_model.model.model.language_model.config._attn_implementation,
                expected_attn_implementation,
            )
            self.assertEqual(
                loaded_model.model.model.vision_tower.config._attn_implementation,
                expected_attn_implementation,
            )

    @unittest.skip(reason="ShieldGemma2 image token masks are not supported by forced flash SDPA kernels")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="ShieldGemma2ForImageClassification returns logits and probabilities only")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="ShieldGemma2ForImageClassification returns logits and probabilities only")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="ShieldGemma2ForImageClassification does not compute a training loss")
    def test_training(self):
        pass

    @unittest.skip(reason="ShieldGemma2ForImageClassification does not compute a classification loss")
    def test_problem_types(self):
        pass

    @unittest.skip(reason="ShieldGemma2ForImageClassification does not have a num_labels-based classifier head")
    def test_can_load_ignoring_mismatched_shapes(self):
        pass

    @unittest.skip(reason="DeepSpeed ZeRO-3 does not support this nested AutoModel.from_config test setup")
    def test_resize_tokens_embeddings_with_deepspeed(self):
        pass

    @unittest.skip(reason="DeepSpeed ZeRO-3 does not support this nested AutoModel.from_config test setup")
    def test_resize_tokens_embeddings_with_deepspeed_multi_gpu(self):
        pass

    @unittest.skip(reason="DeepSpeed ZeRO-3 does not support this nested AutoModel.from_config test setup")
    def test_resize_embeddings_untied_with_deepspeed(self):
        pass

    @unittest.skip(reason="DeepSpeed ZeRO-3 does not support this nested AutoModel.from_config test setup")
    def test_resize_embeddings_untied_with_deepspeed_multi_gpu(self):
        pass


@slow
@require_torch_accelerator
class ShieldGemma2IntegrationTest(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model(self):
        model_id = "google/shieldgemma-2-4b-it"

        processor = ShieldGemma2Processor.from_pretrained(model_id, padding_side="left")
        image = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
            )
        )

        model = ShieldGemma2ForImageClassification.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )

        inputs = processor(images=[image], return_tensors="pt").to(torch_device)
        output = model(**inputs)
        self.assertEqual(len(output.probabilities), 3)
        for element in output.probabilities:
            self.assertEqual(len(element), 2)
