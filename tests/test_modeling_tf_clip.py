# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the TensorFlow CLIP model. """


import inspect
import os
import tempfile
import unittest
from importlib import import_module

import requests
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers.file_utils import is_tf_available, is_vision_available
from transformers.testing_utils import is_pt_tf_cross_test, require_tf, require_vision, slow

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_tf_available():
    import numpy as np
    import tensorflow as tf

    from transformers import TFCLIPModel, TFCLIPTextModel, TFCLIPVisionModel, TFSharedEmbeddings
    from transformers.models.clip.modeling_tf_clip import TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import CLIPProcessor


class TFCLIPVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return CLIPVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values):
        model = TFCLIPVisionModel(config=config)
        result = model(pixel_values, training=False)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_tf
class TFCLIPVisionModelTest(TFModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as CLIP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (TFCLIPVisionModel,) if is_tf_available() else ()

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFCLIPVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLIPVisionConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_inputs_embeds(self):
        # CLIP does not use inputs_embeds
        pass

    def test_graph_mode_with_inputs_embeds(self):
        # CLIP does not use inputs_embeds
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (tf.keras.layers.Layer))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, tf.keras.layers.Layer))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.call)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        # in CLIP, the seq_len equals the number of patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.model_tester.image_size, self.model_tester.image_size)
        patch_size = (self.model_tester.patch_size, self.model_tester.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        seq_len = num_patches + 1

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)

            added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)

            self.assertListEqual(
                list(self_attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, seq_len, seq_len],
            )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)

            outputs = model(**self._prepare_for_class(inputs_dict, model_class), training=False)

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            # CLIP has a different seq_length
            image_size = (self.model_tester.image_size, self.model_tester.image_size)
            patch_size = (self.model_tester.patch_size, self.model_tester.patch_size)
            num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
            seq_length = num_patches + 1

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
        for model_name in TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFCLIPVisionModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class TFCLIPTextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return CLIPTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = TFCLIPTextModel(config=config)
        result = model(input_ids, attention_mask=input_mask, training=False)
        result = model(input_ids, training=False)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_tf
class TFCLIPTextModelTest(TFModelTesterMixin, unittest.TestCase):

    all_model_classes = (TFCLIPTextModel,) if is_tf_available() else ()
    test_pruning = False
    test_head_masking = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFCLIPTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLIPTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_inputs_embeds(self):
        # CLIP does not use inputs_embeds
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFCLIPTextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class TFCLIPModelTester:
    def __init__(self, parent, is_training=True):
        self.parent = parent
        self.text_model_tester = TFCLIPTextModelTester(parent)
        self.vision_model_tester = TFCLIPVisionModelTester(parent)
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return CLIPConfig.from_text_vision_configs(
            self.text_model_tester.get_config(), self.vision_model_tester.get_config(), projection_dim=64
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = TFCLIPModel(config)
        result = model(input_ids, pixel_values, attention_mask, training=False)
        self.parent.assertEqual(
            result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "return_loss": True,
        }
        return config, inputs_dict


@require_tf
class TFCLIPModelTest(TFModelTesterMixin, unittest.TestCase):
    all_model_classes = (TFCLIPModel,) if is_tf_available() else ()
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    test_onnx = False

    def setUp(self):
        self.model_tester = TFCLIPModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # hidden_states are tested in individual model tests
    def test_hidden_states_output(self):
        pass

    # input_embeds are tested in individual model tests
    def test_inputs_embeds(self):
        pass

    # CLIPModel does not have input/output embeddings
    def test_model_common_attributes(self):
        pass

    # overwrite from common since `TFCLIPModelTester` set `return_loss` to `True` and causes the preparation of
    # `symbolic_inputs` failed.
    def test_keras_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # remove `return_loss` to make code work
        if self.__class__.__name__ == "TFCLIPModelTest":
            inputs_dict.pop("return_loss", None)

        tf_main_layer_classes = set(
            module_member
            for model_class in self.all_model_classes
            for module in (import_module(model_class.__module__),)
            for module_member_name in dir(module)
            if module_member_name.endswith("MainLayer")
            # This condition is required, since `modeling_tf_clip.py` has 3 classes whose names end with `MainLayer`.
            and module_member_name[: -len("MainLayer")] == model_class.__name__[: -len("Model")]
            for module_member in (getattr(module, module_member_name),)
            if isinstance(module_member, type)
            and tf.keras.layers.Layer in module_member.__bases__
            and getattr(module_member, "_keras_serializable", False)
        )
        for main_layer_class in tf_main_layer_classes:
            # T5MainLayer needs an embed_tokens parameter when called without the inputs_embeds parameter
            if "T5" in main_layer_class.__name__:
                # Take the same values than in TFT5ModelTester for this shared layer
                shared = TFSharedEmbeddings(99, 32, name="shared")
                config.use_cache = inputs_dict.pop("use_cache", None)
                main_layer = main_layer_class(config, embed_tokens=shared)
            else:
                main_layer = main_layer_class(config)

            symbolic_inputs = {
                name: tf.keras.Input(tensor.shape[1:], dtype=tensor.dtype) for name, tensor in inputs_dict.items()
            }

            model = tf.keras.Model(symbolic_inputs, outputs=main_layer(symbolic_inputs))
            outputs = model(inputs_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                filepath = os.path.join(tmpdirname, "keras_model.h5")
                model.save(filepath)
                if "T5" in main_layer_class.__name__:
                    model = tf.keras.models.load_model(
                        filepath,
                        custom_objects={
                            main_layer_class.__name__: main_layer_class,
                            "TFSharedEmbeddings": TFSharedEmbeddings,
                        },
                    )
                else:
                    model = tf.keras.models.load_model(
                        filepath, custom_objects={main_layer_class.__name__: main_layer_class}
                    )
                assert isinstance(model, tf.keras.Model)
                after_outputs = model(inputs_dict)
                self.assert_outputs_same(after_outputs, outputs)

    # overwrite from common since CLIPModel/TFCLIPModel return CLIPOutput/TFCLIPOutput
    @is_pt_tf_cross_test
    def test_pt_tf_model_equivalence(self):
        import torch

        import transformers

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            pt_model_class_name = model_class.__name__[2:]  # Skip the "TF" at the beginning
            pt_model_class = getattr(transformers, pt_model_class_name)

            config.output_hidden_states = True

            tf_model = model_class(config)
            pt_model = pt_model_class(config)

            # Check we can load pt model in tf and vice-versa with model => model functions

            tf_model = transformers.load_pytorch_model_in_tf2_model(
                tf_model, pt_model, tf_inputs=self._prepare_for_class(inputs_dict, model_class)
            )
            pt_model = transformers.load_tf2_model_in_pytorch_model(pt_model, tf_model)

            # Check predictions on first output (logits/hidden-states) are close enought given low-level computational differences
            pt_model.eval()
            pt_inputs_dict = {}
            for name, key in self._prepare_for_class(inputs_dict, model_class).items():
                if type(key) == bool:
                    pt_inputs_dict[name] = key
                elif name == "input_values":
                    pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                elif name == "pixel_values":
                    pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                else:
                    pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.long)

            # need to rename encoder-decoder "inputs" for PyTorch
            if "inputs" in pt_inputs_dict and self.is_encoder_decoder:
                pt_inputs_dict["input_ids"] = pt_inputs_dict.pop("inputs")

            with torch.no_grad():
                pto = pt_model(**pt_inputs_dict)
            tfo = tf_model(self._prepare_for_class(inputs_dict, model_class), training=False)

            self.assertEqual(len(tfo), len(pto), "Output lengths differ between TF and PyTorch")
            for tf_output, pt_output in zip(tfo.to_tuple(), pto.to_tuple()):

                if not (isinstance(tf_output, tf.Tensor) and isinstance(pt_output, torch.Tensor)):
                    continue

                tf_out = tf_output.numpy()
                pt_out = pt_output.numpy()

                self.assertEqual(tf_out.shape, pt_out.shape, "Output component shapes differ between TF and PyTorch")

                if len(tf_out.shape) > 0:

                    tf_nans = np.copy(np.isnan(tf_out))
                    pt_nans = np.copy(np.isnan(pt_out))

                    pt_out[tf_nans] = 0
                    tf_out[tf_nans] = 0
                    pt_out[pt_nans] = 0
                    tf_out[pt_nans] = 0

                max_diff = np.amax(np.abs(tf_out - pt_out))
                self.assertLessEqual(max_diff, 4e-2)

            # Check we can load pt model in tf and vice-versa with checkpoint => model functions
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_checkpoint_path = os.path.join(tmpdirname, "pt_model.bin")
                torch.save(pt_model.state_dict(), pt_checkpoint_path)
                tf_model = transformers.load_pytorch_checkpoint_in_tf2_model(tf_model, pt_checkpoint_path)

                tf_checkpoint_path = os.path.join(tmpdirname, "tf_model.h5")
                tf_model.save_weights(tf_checkpoint_path)
                pt_model = transformers.load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path)

            # Check predictions on first output (logits/hidden-states) are close enought given low-level computational differences
            pt_model.eval()
            pt_inputs_dict = {}
            for name, key in self._prepare_for_class(inputs_dict, model_class).items():
                if type(key) == bool:
                    key = np.array(key, dtype=bool)
                    pt_inputs_dict[name] = torch.from_numpy(key).to(torch.long)
                elif name == "input_values":
                    pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                elif name == "pixel_values":
                    pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.float32)
                else:
                    pt_inputs_dict[name] = torch.from_numpy(key.numpy()).to(torch.long)
            # need to rename encoder-decoder "inputs" for PyTorch
            if "inputs" in pt_inputs_dict and self.is_encoder_decoder:
                pt_inputs_dict["input_ids"] = pt_inputs_dict.pop("inputs")

            with torch.no_grad():
                pto = pt_model(**pt_inputs_dict)
            tfo = tf_model(self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(len(tfo), len(pto), "Output lengths differ between TF and PyTorch")
            for tf_output, pt_output in zip(tfo.to_tuple(), pto.to_tuple()):

                if not (isinstance(tf_output, tf.Tensor) and isinstance(pt_output, torch.Tensor)):
                    continue

                tf_out = tf_output.numpy()
                pt_out = pt_output.numpy()

                self.assertEqual(tf_out.shape, pt_out.shape, "Output component shapes differ between TF and PyTorch")

                if len(tf_out.shape) > 0:
                    tf_nans = np.copy(np.isnan(tf_out))
                    pt_nans = np.copy(np.isnan(pt_out))

                    pt_out[tf_nans] = 0
                    tf_out[tf_nans] = 0
                    pt_out[pt_nans] = 0
                    tf_out[pt_nans] = 0

                max_diff = np.amax(np.abs(tf_out - pt_out))
                self.assertLessEqual(max_diff, 4e-2)

    @slow
    def test_model_from_pretrained(self):
        for model_name in TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TFCLIPModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@require_vision
@require_tf
class TFCLIPModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "openai/clip-vit-base-patch32"
        model = TFCLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)

        image = prepare_img()
        inputs = processor(
            text=["a photo of a cat", "a photo of a dog"], images=image, padding=True, return_tensors="tf"
        )

        outputs = model(**inputs, training=False)

        # verify the logits
        self.assertEqual(
            outputs.logits_per_image.shape,
            tf.TensorShape((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            tf.TensorShape((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        expected_logits = tf.constant([[24.5701, 19.3049]])

        tf.debugging.assert_near(outputs.logits_per_image, expected_logits, atol=1e-3)
