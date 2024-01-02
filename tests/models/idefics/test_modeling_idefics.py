# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Idefics model. """

import unittest

from parameterized import parameterized

from transformers import BitsAndBytesConfig, IdeficsConfig, is_torch_available, is_vision_available
from transformers.testing_utils import (
    TestCasePlus,
    require_bitsandbytes,
    require_torch,
    require_torch_sdpa,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import IdeficsForVisionText2Text, IdeficsModel, IdeficsProcessor
    from transformers.models.idefics.configuration_idefics import IdeficsPerceiverConfig, IdeficsVisionConfig
    from transformers.models.idefics.modeling_idefics import IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_0
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class IdeficsModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        seq_length=7,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        alpha_initializer="ones",
        num_labels=3,
        scope=None,
        modality_type_vocab_size=2,
        vision_embed_dim=32,
        vision_patch_size=2,
        vision_image_size=30,
        vision_num_attention_heads=4,
        vision_num_hidden_layers=5,
        vision_intermediate_size=37,
        perceiver_qk_layer_norms_perceiver=False,
        perceiver_resampler_depth=2,
        perceiver_resampler_head_dim=8,
        perceiver_resampler_n_heads=2,
        perceiver_resampler_n_latents=16,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.alpha_initializer = alpha_initializer
        self.num_labels = num_labels
        self.scope = scope
        self.modality_type_vocab_size = modality_type_vocab_size

        self.vision_embed_dim = vision_embed_dim
        self.vision_patch_size = vision_patch_size
        self.vision_image_size = vision_image_size
        self.vision_num_attention_heads = vision_num_attention_heads
        self.vision_num_hidden_layers = vision_num_hidden_layers
        self.vision_intermediate_size = vision_intermediate_size

        self.vision_config = IdeficsVisionConfig(
            embed_dim=self.vision_embed_dim,
            patch_size=self.vision_patch_size,
            image_size=self.vision_image_size,
            num_attention_heads=self.vision_num_attention_heads,
            num_hidden_layers=self.vision_num_hidden_layers,
            intermediate_size=self.vision_intermediate_size,
        )

        self.perceiver_qk_layer_norms_perceiver = perceiver_qk_layer_norms_perceiver
        self.perceiver_resampler_depth = perceiver_resampler_depth
        self.perceiver_resampler_head_dim = perceiver_resampler_head_dim
        self.perceiver_resampler_n_heads = perceiver_resampler_n_heads
        self.perceiver_resampler_n_latents = perceiver_resampler_n_latents

        self.perceiver_config = IdeficsPerceiverConfig(
            qk_layer_norms_perceiver=self.perceiver_qk_layer_norms_perceiver,
            resampler_depth=self.perceiver_resampler_depth,
            resampler_head_dim=self.perceiver_resampler_head_dim,
            resampler_n_heads=self.perceiver_resampler_n_heads,
            resampler_n_latents=self.perceiver_resampler_n_latents,
        )

        # we set the expected sequence length (which is used in several tests)
        # this is equal to the seq length of the text tokens + number of image patches + 1 for the CLS token
        self.expected_seq_len = self.seq_length + (self.image_size // self.patch_size) ** 2 + 1

    def prepare_config_and_inputs(self, num_images=1, interpolate_pos_encoding=False, image_expansion=0):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        pixel_values = floats_tensor(
            [
                self.batch_size,
                num_images,
                self.num_channels,
                self.image_size + image_expansion,
                self.image_size + image_expansion,
            ]
        )
        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        image_attention_mask = random_attention_mask([self.batch_size, self.seq_length, num_images])

        config = self.get_config()
        return (config, input_ids, input_mask, pixel_values, image_attention_mask, interpolate_pos_encoding)

    def prepare_config_and_inputs_gate_tests(self):
        # Create a list of configs and inputs, to test 2 things:
        # 1. For the same image, the output should be different when image_attention_mask is filled with 0s vs filled with 1s.
        # 2. For 2 different images, the output should be the same when image_attention_mask is filled with 0s.

        interpolate_pos_encoding = False
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = floats_tensor(
            [
                self.batch_size,
                1,
                self.num_channels,
                self.image_size,
                self.image_size,
            ]
        )
        pixel_values_list = [
            pixel_values.clone(),
            pixel_values.clone(),
            pixel_values.clone().fill_(0.6),
            pixel_values.clone().fill_(0.3),
        ]
        attention_mask = None
        if self.use_input_mask:
            attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        image_attention_mask = random_attention_mask([self.batch_size, self.seq_length, 1])
        image_attention_mask_list = [
            image_attention_mask.clone().fill_(0),
            image_attention_mask.clone().fill_(1),
            image_attention_mask.clone().fill_(0),
            image_attention_mask.clone().fill_(0),
        ]

        config = self.get_config()
        inputs_list = []
        for pixel_values, image_attention_mask in zip(pixel_values_list, image_attention_mask_list):
            inputs_list.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "image_attention_mask": image_attention_mask,
                    "interpolate_pos_encoding": interpolate_pos_encoding,
                }
            )

        inputs_w_same_img = inputs_list[:2]
        inputs_w_0_img_attn = inputs_list[2:]
        return config, inputs_w_same_img, inputs_w_0_img_attn

    def get_config(self):
        return IdeficsConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            alpha_initializer=self.alpha_initializer,
            num_labels=self.num_labels,
            modality_type_vocab_size=self.modality_type_vocab_size,
            vision_config=self.vision_config,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        input_mask,
        pixel_values,
        image_attention_mask,
        interpolate_pos_encoding,
    ):
        model = IdeficsModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            pixel_values=pixel_values,
            image_attention_mask=image_attention_mask,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, input_ids.shape[1], self.hidden_size)
        )

    def create_and_check_model_gen(
        self,
        config,
        input_ids,
        input_mask,
        pixel_values,
        image_attention_mask,
        interpolate_pos_encoding,
    ):
        model = IdeficsForVisionText2Text(config)
        model.to(torch_device)
        model.eval()
        model.generate(
            input_ids,
            attention_mask=input_mask,
            pixel_values=pixel_values,
            image_attention_mask=image_attention_mask,
            interpolate_pos_encoding=interpolate_pos_encoding,
            max_length=self.seq_length + 2,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            pixel_values,
            image_attention_mask,
            interpolate_pos_encoding,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "pixel_values": pixel_values,
            "image_attention_mask": image_attention_mask,
            "interpolate_pos_encoding": interpolate_pos_encoding,
        }
        return config, inputs_dict

    def prepare_pixel_values(self):
        return floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

    @require_torch_sdpa
    @slow
    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        self.skipTest("Idefics has a hard requirement on SDPA, skipping this test")


@unittest.skipIf(not is_torch_greater_or_equal_than_2_0, reason="pytorch 2.0 or higher is required")
@require_torch
class IdeficsModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (IdeficsModel, IdeficsForVisionText2Text) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": IdeficsModel} if is_torch_available() else {}
    test_pruning = False
    test_headmasking = False
    test_torchscript = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        # XXX: IdeficsForVisionText2TextTest has no MODEL_FOR group yet, but it should be the same
        # as MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, so for now manually changing to do the right thing
        # as super won't do it
        if return_labels:
            inputs_dict["labels"] = torch.zeros(
                (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
            )

        return inputs_dict

    def test_model_outputs_equivalence(self):
        try:
            orig = self.all_model_classes
            # IdeficsModel.forward doesn't have labels input arg - only IdeficsForVisionText2Text does
            self.all_model_classes = (IdeficsForVisionText2Text,) if is_torch_available() else ()
            super().test_model_outputs_equivalence()
        finally:
            self.all_model_classes = orig

    def setUp(self):
        self.model_tester = IdeficsModelTester(self)
        self.config_tester = ConfigTester(self, config_class=IdeficsConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_single_image(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            num_images=1, interpolate_pos_encoding=False, image_expansion=0
        )
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_multiple_images(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            num_images=2, interpolate_pos_encoding=False, image_expansion=0
        )
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_image_pos_embeddings_interpolation_single_image(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            num_images=1, interpolate_pos_encoding=True, image_expansion=2
        )
        self.model_tester.create_and_check_model(*config_and_inputs)
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            num_images=1, interpolate_pos_encoding=True, image_expansion=0
        )
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_image_pos_embeddings_interpolation_multiple_images(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            num_images=2, interpolate_pos_encoding=True, image_expansion=2
        )
        self.model_tester.create_and_check_model(*config_and_inputs)
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            num_images=2, interpolate_pos_encoding=True, image_expansion=0
        )
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_generate_with_image_pos_embeddings_interpolation_single_image(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            num_images=1, interpolate_pos_encoding=True, image_expansion=2
        )
        self.model_tester.create_and_check_model_gen(*config_and_inputs)

    def test_generate_with_image_pos_embeddings_interpolation_multiple_images(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs(
            num_images=2, interpolate_pos_encoding=True, image_expansion=2
        )
        self.model_tester.create_and_check_model_gen(*config_and_inputs)

    def test_cross_attention_gates(self):
        config, inputs_w_same_img, inputs_w_0_img_attn = self.model_tester.prepare_config_and_inputs_gate_tests()

        model = IdeficsModel(config=config).to(torch_device)
        model.eval()
        test_1_results = []
        for inputs in inputs_w_same_img:
            with torch.no_grad():
                last_hidden_states = model(**inputs).last_hidden_state
            last_hidden_states = model(**inputs).last_hidden_state
            test_1_results.append(last_hidden_states)
        self.assertNotEqual(test_1_results[0].sum().item(), test_1_results[1].sum().item())

        test_2_results = []
        for inputs in inputs_w_0_img_attn:
            with torch.no_grad():
                last_hidden_states = model(**inputs).last_hidden_state
            test_2_results.append(last_hidden_states)
        self.assertEqual(test_2_results[0].sum().item(), test_2_results[1].sum().item())

    def test_training(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes:
            # IdeficsModel does not support training, users should use
            # IdeficsForVisionText2Text for this purpose
            if model_class == IdeficsModel:
                return

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_training_gradient_checkpointing(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes:
            # IdeficsModel does not support training, users should use
            # IdeficsForVisionText2Text for this purpose
            if model_class == IdeficsModel:
                return

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.use_cache = False
            config.return_dict = True

            model = model_class(config)
            model.to(torch_device)
            model.gradient_checkpointing_enable()
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="""IDEFICS does not support retaining the gradients of the hidden states and attention""")
    def test_retain_grad_hidden_states_attentions(self):
        return

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions

            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            # IDEFICS does not support outputting attention score becuase it uses SDPA under the hood
            self.assertTrue(attentions[0] is None)
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + 1, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            # IDEFICS does not support outputting attention score becuase it uses SDPA under the hood
            self.assertTrue(self_attentions[0] is None)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @slow
    def test_model_from_pretrained(self):
        for model_name in IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = IdeficsModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @require_torch_sdpa
    @slow
    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        self.skipTest("Idefics has a hard requirement on SDPA, skipping this test")


@unittest.skipIf(not is_torch_greater_or_equal_than_2_0, reason="pytorch 2.0 or higher is required")
@require_torch
class IdeficsForVisionText2TextTest(IdeficsModelTest, unittest.TestCase):
    all_model_classes = (IdeficsForVisionText2Text,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = IdeficsModelTester(
            self,
            modality_type_vocab_size=3,
        )
        self.config_tester = ConfigTester(self, config_class=IdeficsConfig, hidden_size=37)

    @unittest.skip("We only test the model that takes in multiple images")
    def test_model(self):
        pass

    @unittest.skip("We only test the model that takes in multiple images")
    def test_for_token_classification(self):
        pass

    @unittest.skip(reason="""IDEFICS does not support retaining the gradients of the hidden states and attention""")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass


@unittest.skipIf(not is_torch_greater_or_equal_than_2_0, reason="pytorch 2.0 or higher is required")
@require_torch
@require_vision
class IdeficsModelIntegrationTest(TestCasePlus):
    @cached_property
    def default_processor(self):
        return (
            IdeficsProcessor.from_pretrained("HuggingFaceM4/idefics-9b", revision="refs/pr/11")
            if is_vision_available()
            else None
        )

    @require_bitsandbytes
    @slow
    def test_inference_natural_language_visual_reasoning(self):
        cat_image_path = self.tests_dir / "fixtures/tests_samples/COCO/000000039769.png"
        cats_image_obj = Image.open(cat_image_path)  # 2 cats
        dogs_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_nlvr2/raw/main/image1.jpeg"

        prompts = [
            [
                "User:",
                dogs_image_url,
                "Describe this image.\nAssistant: An image of two dogs.\n",
                "User:",
                cats_image_obj,
                "Describe this image.\nAssistant:",
            ],
            [
                "User:",
                cats_image_obj,
                "Describe this image.\nAssistant: An image of two kittens.\n",
                "User:",
                dogs_image_url,
                "Describe this image.\nAssistant:",
            ],
        ]

        # the CI gpu is small so using quantization to fit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
        )
        model = IdeficsForVisionText2Text.from_pretrained(
            "HuggingFaceM4/idefics-9b", quantization_config=quantization_config, device_map="auto"
        )
        processor = self.default_processor
        inputs = processor(prompts, return_tensors="pt").to(torch_device)
        generated_ids = model.generate(**inputs, max_length=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # keep for debugging
        for i, t in enumerate(generated_text):
            t = bytes(t, "utf-8").decode("unicode_escape")
            print(f"{i}:\n{t}\n")

        self.assertIn("image of two cats", generated_text[0])
        self.assertIn("image of two dogs", generated_text[1])
