# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch ViTMAE model. """


import inspect
import math
import os
import tempfile
import unittest

import numpy as np

from transformers import ViTMAEConfig
from transformers.testing_utils import is_pt_tf_cross_test, require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ..test_configuration_common import ConfigTester
from ..test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch
    from torch import nn

    from transformers import ViTMAEForPreTraining, ViTMAEModel
    from transformers.models.vit.modeling_vit import VIT_PRETRAINED_MODEL_ARCHIVE_LIST, to_2tuple


if is_vision_available():
    from PIL import Image

    from transformers import ViTFeatureExtractor


class ViTMAEModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return ViTMAEConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = ViTMAEModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        # expected sequence length = (num_patches + 1) * (1 - config.mask_ratio), rounded above
        # (we add 1 for the [CLS] token)
        image_size = to_2tuple(self.image_size)
        patch_size = to_2tuple(self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        expected_seq_len = int(math.ceil((1 - config.mask_ratio) * (num_patches + 1)))
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, expected_seq_len, self.hidden_size))

    def create_and_check_for_pretraining(self, config, pixel_values, labels):
        model = ViTMAEForPreTraining(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        # expected sequence length = num_patches
        image_size = to_2tuple(self.image_size)
        patch_size = to_2tuple(self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        expected_seq_len = num_patches
        expected_num_channels = self.patch_size**2 * self.num_channels
        self.parent.assertEqual(result.logits.shape, (self.batch_size, expected_seq_len, expected_num_channels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class ViTMAEModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as ViTMAE does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (ViTMAEModel, ViTMAEForPreTraining) if is_torch_available() else ()

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = ViTMAEModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ViTMAEConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_inputs_embeds(self):
        # ViTMAE does not use inputs_embeds
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        # in ViTMAE, the seq_len equals (number of patches + 1) * (1 - mask_ratio), rounded above
        image_size = to_2tuple(self.model_tester.image_size)
        patch_size = to_2tuple(self.model_tester.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        seq_len = int(math.ceil((1 - config.mask_ratio) * (num_patches + 1)))
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            if chunk_length is not None:
                self.assertListEqual(
                    list(attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            if chunk_length is not None:
                self.assertListEqual(
                    list(self_attentions[0].shape[-4:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, chunk_length, encoder_key_length],
                )
            else:
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
                )

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

            # ViTMAE has a different seq_length
            image_size = to_2tuple(self.model_tester.image_size)
            patch_size = to_2tuple(self.model_tester.patch_size)
            num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
            seq_length = int(math.ceil((1 - config.mask_ratio) * (num_patches + 1)))

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

    # overwrite from common since ViTMAEForPretraining has random masking, we need to fix the noise
    # to generate masks during test
    @is_pt_tf_cross_test
    def test_pt_tf_model_equivalence(self):
        import numpy as np
        import tensorflow as tf

        import transformers

        # make masks reproducible
        np.random.seed(2)

        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        num_patches = int((config.image_size // config.patch_size) ** 2)
        noise = np.random.uniform(size=(self.model_tester.batch_size, num_patches))
        pt_noise = torch.from_numpy(noise).to(device=torch_device)
        tf_noise = tf.constant(noise)

        def prepare_tf_inputs_from_pt_inputs(pt_inputs_dict):

            tf_inputs_dict = {}
            for key, tensor in pt_inputs_dict.items():
                tf_inputs_dict[key] = tf.convert_to_tensor(tensor.cpu().numpy(), dtype=tf.float32)

            return tf_inputs_dict

        def check_outputs(tf_outputs, pt_outputs, model_class, names):
            """
            Args:
                model_class: The class of the model that is currently testing. For example, `TFBertModel`,
                    TFBertForMaskedLM`, `TFBertForSequenceClassification`, etc. Currently unused, but it could make
                    debugging easier and faster.

                names: A string, or a tuple of strings. These specify what tf_outputs/pt_outputs represent in the model outputs.
                    Currently unused, but in the future, we could use this information to make the error message clearer
                    by giving the name(s) of the output tensor(s) with large difference(s) between PT and TF.
            """

            # Allow `list` because `(TF)TransfoXLModelOutput.mems` is a list of tensors.
            if type(tf_outputs) in [tuple, list]:
                self.assertEqual(type(tf_outputs), type(pt_outputs))
                self.assertEqual(len(tf_outputs), len(pt_outputs))
                if type(names) == tuple:
                    for tf_output, pt_output, name in zip(tf_outputs, pt_outputs, names):
                        check_outputs(tf_output, pt_output, model_class, names=name)
                elif type(names) == str:
                    for idx, (tf_output, pt_output) in enumerate(zip(tf_outputs, pt_outputs)):
                        check_outputs(tf_output, pt_output, model_class, names=f"{names}_{idx}")
                else:
                    raise ValueError(f"`names` should be a `tuple` or a string. Got {type(names)} instead.")
            elif isinstance(tf_outputs, tf.Tensor):
                self.assertTrue(isinstance(pt_outputs, torch.Tensor))

                tf_outputs = tf_outputs.numpy()
                if isinstance(tf_outputs, np.float32):
                    tf_outputs = np.array(tf_outputs, dtype=np.float32)
                pt_outputs = pt_outputs.detach().to("cpu").numpy()

                tf_nans = np.isnan(tf_outputs)
                pt_nans = np.isnan(pt_outputs)

                pt_outputs[tf_nans] = 0
                tf_outputs[tf_nans] = 0
                pt_outputs[pt_nans] = 0
                tf_outputs[pt_nans] = 0

                max_diff = np.amax(np.abs(tf_outputs - pt_outputs))
                self.assertLessEqual(max_diff, 1e-5)
            else:
                raise ValueError(
                    f"`tf_outputs` should be a `tuple` or an instance of `tf.Tensor`. Got {type(tf_outputs)} instead."
                )

        def check_pt_tf_models(tf_model, pt_model, pt_inputs_dict):
            # we are not preparing a model with labels because of the formation
            # of the ViT MAE model

            # send pytorch model to the correct device
            pt_model.to(torch_device)

            # Check predictions on first output (logits/hidden-states) are close enough given low-level computational differences
            pt_model.eval()

            tf_inputs_dict = prepare_tf_inputs_from_pt_inputs(pt_inputs_dict)

            # send pytorch inputs to the correct device
            pt_inputs_dict = {
                k: v.to(device=torch_device) if isinstance(v, torch.Tensor) else v for k, v in pt_inputs_dict.items()
            }

            # Original test: check without `labels`
            with torch.no_grad():
                pt_outputs = pt_model(**pt_inputs_dict, noise=pt_noise)
            tf_outputs = tf_model(tf_inputs_dict, noise=tf_noise)

            tf_keys = tuple([k for k, v in tf_outputs.items() if v is not None])
            pt_keys = tuple([k for k, v in pt_outputs.items() if v is not None])

            self.assertEqual(tf_keys, pt_keys)
            check_outputs(tf_outputs.to_tuple(), pt_outputs.to_tuple(), model_class, names=tf_keys)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            tf_model_class_name = "TF" + model_class.__name__  # Add the "TF" at the beginning

            # Output all for aggressive testing
            config.output_hidden_states = True
            config.output_attentions = self.has_attentions

            tf_model_class = getattr(transformers, tf_model_class_name)

            tf_model = tf_model_class(config)
            pt_model = model_class(config)

            # make sure only tf inputs are forward that actually exist in function args
            tf_input_keys = set(inspect.signature(tf_model.call).parameters.keys())

            # remove all head masks
            tf_input_keys.discard("head_mask")
            tf_input_keys.discard("cross_attn_head_mask")
            tf_input_keys.discard("decoder_head_mask")

            pt_inputs_dict = self._prepare_for_class(inputs_dict, model_class)

            pt_inputs_dict = {k: v for k, v in pt_inputs_dict.items() if k in tf_input_keys}

            # Check we can load pt model in tf and vice-versa with model => model functions
            tf_inputs_dict = prepare_tf_inputs_from_pt_inputs(pt_inputs_dict)
            tf_model = transformers.load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=tf_inputs_dict)
            pt_model = transformers.load_tf2_model_in_pytorch_model(pt_model, tf_model)

            check_pt_tf_models(tf_model, pt_model, pt_inputs_dict)

            # Check we can load pt model in tf and vice-versa with checkpoint => model functions
            with tempfile.TemporaryDirectory() as tmpdirname:
                pt_checkpoint_path = os.path.join(tmpdirname, "pt_model.bin")
                torch.save(pt_model.state_dict(), pt_checkpoint_path)
                tf_model = transformers.load_pytorch_checkpoint_in_tf2_model(tf_model, pt_checkpoint_path)

                tf_checkpoint_path = os.path.join(tmpdirname, "tf_model.h5")
                tf_model.save_weights(tf_checkpoint_path)
                pt_model = transformers.load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path)
                pt_model = pt_model.to(torch_device)

            check_pt_tf_models(tf_model, pt_model, pt_inputs_dict)

    def test_save_load(self):

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            # make random mask reproducible
            torch.manual_seed(2)
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            out_2 = outputs[0].cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)
                # make random mask reproducible
                torch.manual_seed(2)
                with torch.no_grad():
                    after_outputs = model(**self._prepare_for_class(inputs_dict, model_class))

                # Make sure we don't have nans
                out_1 = after_outputs[0].cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    @unittest.skip(
        reason="""ViTMAE returns a random mask + ids_restore in each forward pass. See test_save_load
    to get deterministic results."""
    )
    def test_determinism(self):
        pass

    @unittest.skip(
        reason="""ViTMAE returns a random mask + ids_restore in each forward pass. See test_save_load
    to get deterministic results."""
    )
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(
        reason="""ViTMAE returns a random mask + ids_restore in each forward pass. See test_save_load
    to get deterministic results."""
    )
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="""ViTMAE returns a random mask + ids_restore in each forward pass. See test_save_load""")
    def test_model_outputs_equivalence(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in VIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = ViTMAEModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class ViTMAEModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return ViTFeatureExtractor.from_pretrained("facebook/vit-mae-base") if is_vision_available() else None

    @slow
    def test_inference_for_pretraining(self):
        # make random mask reproducible across the PT and TF model
        np.random.seed(2)

        model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(torch_device)

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        # prepare a noise vector that will be also used for testing the TF model
        # (this way we can ensure that the PT and TF models operate on the same inputs)
        vit_mae_config = ViTMAEConfig()
        num_patches = int((vit_mae_config.image_size // vit_mae_config.patch_size) ** 2)
        noise = np.random.uniform(size=(1, num_patches))

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs, noise=torch.from_numpy(noise).to(device=torch_device))

        # verify the logits
        expected_shape = torch.Size((1, 196, 768))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [[-0.0548, -1.7023, -0.9325], [0.3721, -0.5670, -0.2233], [0.8235, -1.3878, -0.3524]]
        )

        self.assertTrue(torch.allclose(outputs.logits[0, :3, :3], expected_slice.to(torch_device), atol=1e-4))
