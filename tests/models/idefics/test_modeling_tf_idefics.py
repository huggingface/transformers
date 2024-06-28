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
"""Testing suite for the TF Idefics model."""

import os
import tempfile
import unittest
from importlib import import_module

from transformers import IdeficsConfig, is_tf_available, is_vision_available
from transformers.testing_utils import TestCasePlus, is_pt_tf_cross_test, require_tf, require_vision, slow
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_tf_available():
    import tensorflow as tf

    from transformers import IdeficsProcessor, TFIdeficsForVisionText2Text, TFIdeficsModel
    from transformers.modeling_tf_utils import keras
    from transformers.models.idefics.configuration_idefics import IdeficsPerceiverConfig, IdeficsVisionConfig

if is_vision_available():
    from PIL import Image


IDEFICS_TINY_RANDOM_MODEL = "HuggingFaceM4/tiny-random-idefics"


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
        model = TFIdeficsModel(config=config)
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
        model = TFIdeficsForVisionText2Text(config)
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


@require_tf
class TFIdeficsModelTest(TFModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (TFIdeficsModel, TFIdeficsForVisionText2Text) if is_tf_available() else ()
    pipeline_model_mapping = {"feature-extraction": TFIdeficsModel} if is_tf_available() else {}
    test_pruning = False
    test_headmasking = False
    test_onnx = False
    test_resize_embeddings = False

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        # XXX: IdeficsForVisionText2TextTest has no MODEL_FOR group yet, but it should be the same
        # as MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, so for now manually changing to do the right thing
        # as super won't do it
        if return_labels:
            inputs_dict["labels"] = tf.zeros(
                (self.model_tester.batch_size, self.model_tester.seq_length), dtype=tf.int64
            )
        return inputs_dict

    def test_model_outputs_equivalence(self):
        try:
            orig = self.all_model_classes
            # IdeficsModel.forward doesn't have labels input arg - only IdeficsForVisionText2Text does
            self.all_model_classes = (TFIdeficsForVisionText2Text,) if is_tf_available() else ()
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

    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="""IDEFICS does not support retaining the gradients of the hidden states and attention""")
    def test_retain_grad_hidden_states_attentions(self):
        return

    @unittest.skip(reason="IDEFICS uses out-of-bounds embeddings deliberately.")
    def test_embeddings_out_of_bounds_raise_exception(self):
        pass

    @unittest.skip(reason="IDEFICS attention weights are not extracted in scaled_dot_product_attention")
    def test_prepare_serving_output(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (tf.keras.layers.Layer))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, tf.keras.layers.Layer))

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions

            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            # IDEFICS does not support outputting attention score becuase it uses SDPA under the hood
            self.assertTrue(attentions[0] is None)
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(out_len + 1, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            # IDEFICS does not support outputting attention score becuase it uses SDPA under the hood
            self.assertTrue(self_attentions[0] is None)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
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

    @is_pt_tf_cross_test
    def test_pt_tf_model_equivalence(self, allow_missing_keys=False):
        self.has_attentions = False
        super().test_pt_tf_model_equivalence(allow_missing_keys=allow_missing_keys)

    def test_keras_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        tf_main_layer_classes = {
            module_member
            for model_class in self.all_model_classes
            for module in (import_module(model_class.__module__),)
            for module_member_name in dir(module)
            if module_member_name.endswith("MainLayer")
            for module_member in (getattr(module, module_member_name),)
            if isinstance(module_member, type)
            and keras.layers.Layer in module_member.__bases__
            and getattr(module_member, "_keras_serializable", False)
        }

        for main_layer_class in tf_main_layer_classes:
            main_layer = main_layer_class(config)

            symbolic_inputs = {
                name: keras.Input(tensor.shape[1:], dtype=tensor.dtype, batch_size=2)
                for name, tensor in inputs_dict.items()
                if tf.is_tensor(tensor)
            }
            model = keras.Model(symbolic_inputs, outputs=main_layer(symbolic_inputs))
            outputs = model(inputs_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                filepath = os.path.join(tmpdirname, "keras_model.h5")
                model.save(filepath)
                model = keras.models.load_model(filepath, custom_objects={main_layer_class.__name__: main_layer_class})
                assert isinstance(model, keras.Model)
                after_outputs = model(inputs_dict)
                self.assert_outputs_same(after_outputs, outputs)

    @unittest.skip(reason="IDEFICS test_keras_fit testing done in TFIdeficsForVisionText2TextTest")
    def test_keras_fit(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = TFIdeficsModel.from_pretrained(IDEFICS_TINY_RANDOM_MODEL, from_pt=True)
        self.assertIsNotNone(model)

    @unittest.skip(reason="Currently `saved_model` doesn't work with nested outputs.")
    def test_saved_model_creation(self):
        pass

    @unittest.skip(reason="""IDEFICS loss computation not implemented yet""")
    def test_loss_computation(self):
        pass


@require_tf
class TFIdeficsForVisionText2TextTest(TFIdeficsModelTest, unittest.TestCase):
    all_model_classes = (TFIdeficsForVisionText2Text,) if is_tf_available() else ()
    test_resize_embeddings = False

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

    @unittest.skip(reason="""IDEFICS loss computation not implemented yet""")
    def test_loss_computation(self):
        pass

    @slow
    def test_keras_fit(self):
        super().test_keras_fit()


# Below is the expected output for the integration test TFIdeficsModelIntegrationTest.
# Since we are using tiny-random to be able to fit it on the CI GPU,it is better to assert on the
# ids because the generated text is gibberish

# fmt: off
EXPECTED_GENERATED_IDS = [[0, 0, 1, 4911, 29901, 32000, 32001, 32000, 20355, 915, 445, 1967, 29889, 13, 7900, 22137, 29901, 530, 1967, 310, 1023, 26361, 29889, 13, 2659, 29901, 32000, 32001, 32000, 20355, 915, 445, 1967, 29889, 13, 7900, 22137, 29901, 25519, 22326, 8071, 26357, 28004, 4428, 5916, 14383, 1033, 12358, 10536, 21834, 10447, 21201, 18102, 16886, 8875, 25388, 25914, 28304, 8558, 31048, 1322, 25952, 189, 31600, 3600, 12824, 7045, 28090, 20228, 32001, 5385, 29186, 2165, 11822, 13825, 23077, 7883, 22504, 2078, 18893, 2179, 10556, 9515, 7672, 3491, 12403, 5398, 27299, 6463, 16349, 23037, 28956, 16960, 22664, 7724, 17587, 17424, 10175, 17417, 5930, 30855, 17695, 16170, 14474, 29996, 313, 14502, 3241, 13618, 32001, 5385, 29186, 2165, 11822, 13825, 19934, 4875, 27142, 3230, 2709, 28054, 3270, 19148, 10917, 1060, 26443, 12259, 1347, 28482, 3830, 25519, 199, 12782, 9144, 12289, 1142, 18400, 21390, 19129, 7292, 28430, 24711, 5551, 30349, 30533, 13271, 17697, 4982, 8713, 5380, 17869, 12490, 5398, 27299, 11593, 19918, 15924, 29430, 10175, 17417, 5930, 30855, 17695, 16170, 14474, 19234],
                          [1, 4911, 29901, 32000, 32001, 32000, 20355, 915, 445, 1967, 29889, 13, 7900, 22137, 29901, 530, 1967, 310, 1023, 413, 986, 575, 29889, 13, 2659, 29901, 32000, 32001, 32000, 20355, 915, 445, 1967, 29889, 13, 7900, 22137, 29901, 25519, 22326, 8071, 26357, 28004, 4428, 17554, 20500, 21714, 27834, 4798, 12195, 30379, 5427, 20228, 10473, 14351, 8049, 15605, 14491, 212, 2711, 32000, 21714, 31259, 24368, 19036, 22970, 26083, 19394, 20372, 7672, 9939, 25388, 30533, 8200, 30271, 2114, 24749, 13224, 10603, 21118, 2179, 3759, 16515, 6587, 1287, 23998, 17793, 32001, 5385, 29186, 2165, 11822, 13825, 29732, 17503, 2729, 6722, 2943, 1221, 16043, 18244, 24965, 14383, 19840, 5980, 13488, 28531, 735, 26146, 22504, 2078, 18893, 20372, 7672, 32001, 5385, 29186, 2165, 11822, 13825, 29732, 17503, 2729, 6722, 19551, 220, 10528, 28940, 4453, 28266, 15416, 18693, 8199, 1153, 27706, 29231, 29186, 2165, 11822, 13825, 29732, 17503, 2729, 6722, 19551, 8231, 10739, 31992, 25906, 22254, 23127, 7689, 19614, 1149, 18844, 23037, 28956, 16960, 22664, 6975, 28938, 24002, 11026, 15020, 21964, 16307], ]

@require_tf
@require_vision
class TFIdeficsModelIntegrationTest(TestCasePlus):
    @cached_property
    def default_processor(self):
        return IdeficsProcessor.from_pretrained(IDEFICS_TINY_RANDOM_MODEL) if is_vision_available() else None

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

        model = TFIdeficsForVisionText2Text.from_pretrained(IDEFICS_TINY_RANDOM_MODEL, from_pt=True)
        processor = self.default_processor
        inputs = processor(prompts, return_tensors="tf")
        generated_ids = model.generate(**inputs, max_length=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # keep for debugging
        for i, t in enumerate(generated_text):
            t = bytes(t, "utf-8").decode("unicode_escape")
            print(f"{i}:\n{t}\n")

        self.assertListEqual(EXPECTED_GENERATED_IDS[0], generated_ids[0].numpy().tolist())
        self.assertListEqual(EXPECTED_GENERATED_IDS[1], generated_ids[1].numpy().tolist())
