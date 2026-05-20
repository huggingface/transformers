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
from io import BytesIO

import requests
from PIL import Image

from transformers import (
    BitsAndBytesConfig,
    Gemma3TextConfig,
    ShieldGemma2Config,
    SiglipVisionConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import ShieldGemma2ForImageClassification, ShieldGemma2Processor


class ShieldGemma2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        seq_length=8,
        vocab_size=99,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=8,
        image_size=8,
        patch_size=4,
        num_channels=3,
        mm_tokens_per_image=4,
        image_token_index=0,
        pad_token_id=1,
        eos_token_id=2,
        bos_token_id=3,
        yes_token_index=4,
        no_token_index=5,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.sliding_window = sliding_window
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.mm_tokens_per_image = mm_tokens_per_image
        self.image_token_index = image_token_index
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.yes_token_index = yes_token_index
        self.no_token_index = no_token_index
        self.is_training = is_training

    def get_config(self):
        text_config = Gemma3TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            sliding_window=self.sliding_window,
            layer_types=["sliding_attention", "full_attention"],
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
        )
        vision_config = SiglipVisionConfig(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=1,
            num_attention_heads=self.num_attention_heads,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
        )
        config = ShieldGemma2Config(
            text_config=text_config,
            vision_config=vision_config,
            mm_tokens_per_image=self.mm_tokens_per_image,
            image_token_index=self.image_token_index,
        )
        config.yes_token_index = self.yes_token_index
        config.no_token_index = self.no_token_index
        return config

    def prepare_config_and_inputs(self):
        config = self.get_config()
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 6) + 6
        input_ids[:, : self.mm_tokens_per_image] = self.image_token_index
        attention_mask = torch.ones_like(input_ids).to(torch_device)
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[:, : self.mm_tokens_per_image] = 1
        return config, input_ids, pixel_values, attention_mask, token_type_ids

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, pixel_values, attention_mask, token_type_ids = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        return config, inputs_dict

    def create_and_check_model(self, config, input_ids, pixel_values, attention_mask, token_type_ids):
        model = ShieldGemma2ForImageClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, 2))
        self.parent.assertEqual(result.probabilities.shape, (self.batch_size, 2))


@require_torch
class ShieldGemma2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ShieldGemma2ForImageClassification,) if is_torch_available() else ()
    _is_composite = True
    additional_model_inputs = ["pixel_values", "attention_mask", "token_type_ids"]

    test_attention_outputs = False

    def setUp(self):
        self.model_tester = ShieldGemma2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ShieldGemma2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

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

    @unittest.skip(reason="ShieldGemma2ForImageClassification does not support return_dict=False")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="ShieldGemma2ForImageClassification does not compute a training loss")
    def test_training(self):
        pass

    @unittest.skip(reason="ShieldGemma2ForImageClassification does not compute a training loss")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="ShieldGemma2ForImageClassification does not compute a training loss")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="ShieldGemma2ForImageClassification does not compute a training loss")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
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

    @unittest.skip(reason="ShieldGemma2ForImageClassification does not use feed-forward chunking")
    def test_feed_forward_chunking(self):
        pass


@slow
@require_torch_accelerator
class ShieldGemma2IntegrationTest(unittest.TestCase):
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model(self):
        model_id = "google/shieldgemma-2-4b-it"

        processor = ShieldGemma2Processor.from_pretrained(model_id, padding_side="left")
        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))

        model = ShieldGemma2ForImageClassification.from_pretrained(
            model_id, quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )

        inputs = processor(images=[image], return_tensors="pt").to(torch_device)
        output = model(**inputs)
        self.assertEqual(len(output.probabilities), 3)
        for element in output.probabilities:
            self.assertEqual(len(element), 2)

    def test_model_sdpa(self):
        model_id = "google/shieldgemma-2-4b-it"

        processor = ShieldGemma2Processor.from_pretrained(model_id, padding_side="left")
        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))

        model = ShieldGemma2ForImageClassification.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            attn_implementation="sdpa",
        )

        inputs = processor(images=[image], return_tensors="pt").to(torch_device)
        output = model(**inputs)
        self.assertEqual(len(output.probabilities), 3)
        for element in output.probabilities:
            self.assertEqual(len(element), 2)
