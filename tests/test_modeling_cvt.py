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
""" Testing suite for the PyTorch CvT model. """


import inspect
import unittest

from transformers import CvTConfig
from transformers.file_utils import cached_property, is_torch_available, is_vision_available
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, require_vision, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor


if is_torch_available():
    import torch
    from torch import nn

    from transformers import MODEL_MAPPING, CvTForImageClassification, CvTModel
    from transformers.models.cvt.modeling_cvt import CVT_PRETRAINED_MODEL_ARCHIVE_LIST, to_2tuple


if is_vision_available():
    from PIL import Image

    from transformers import CvTFeatureExtractor


class CvTModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        image_size=256,
        in_chans = 3,
        num_classes = 1000,
        num_stages = 3,
        patch_size = [7, 3, 3],
        patch_stride = [4, 2, 2],
        patch_padding = [2, 1, 1],
        dim_embed = [64, 192, 384],
        num_heads = [1, 3, 6],
        depth = [1, 4, 16],
        type_sequence_label_size = 10,
        use_labels = False,
        is_training=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = in_chans
        self.num_stages = num_stages
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        self.depth = depth
        self.type_sequence_label_size = type_sequence_label_size
        self.use_labels = use_labels
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return CvTConfig(
            in_chans=self.num_channels,
            num_stages=self.num_stages,
            patch_padding=self.patch_padding,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            num_classes=self.num_classes,
            dim_embed=self.dim_embed,
            num_heads=self.num_heads,
            depth=self.depth,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = CvTModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = to_2tuple(self.image_size)
        patch_size = to_2tuple(self.patch_size)
        #num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, image_size/(self.patch_stride[-1]*self.patch_stride[-2]) + 1, self.dim_embed[-1]))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.type_sequence_label_size
        model = CvTForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class CvTModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as CvT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (CvTModel, CvTForImageClassification) if is_torch_available() else ()
    )

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = CvTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CvTConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()

    def test_inputs_embeds(self):
        # CvT does not use inputs_embeds
        pass

    def test_model_common_attributes(self):
        pass

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

    def test_training(self):
        if not self.model_tester.is_training:
            return

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            if model_class in get_values(MODEL_MAPPING):
                continue
            # we don't test CvTForMaskedImageModeling
            if model_class.__name__ == "CvTForMaskedImageModeling":
                continue
            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                # we skip lambda parameters as these require special initial values
                # determined by config.layer_scale_init_value
                if "lambda" in name:
                    continue
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [-1.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

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
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), sum(self.model_tester.depth))

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), sum(self.model_tester.depth))

            depth_index = 0
            img_size = self.model_tester.image_size
            for i in range(len(self.model_tester.depth)):
                depth_index = depth_index + self.model_tester.depth[i]
                img_size = (img_size + 2*self.model_tester.patch_padding[i] - self.model_tester.patch_size[i])/(self.model_tester.patch_stride[i]) 
                if i != self.model_tester.num_stages - 1: 
                    self.assertListEqual(
                        list(attentions[depth_index-1].shape[-3:]),
                        [self.model_tester.num_heads[i], img_size*img_size, img_size//4],
                    )
                else:
                    self.assertListEqual(
                        list(attentions[depth_index-1].shape[-3:]),
                        [self.model_tester.num_heads[i], img_size*img_size + 1, img_size//4 + 1],
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
            """
            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), sum(self.model_tester.depth))
            """
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), sum(self.model_tester.depth))

            depth_index = 0
            img_size = self.model_tester.image_size
            for i in range(len(self.model_tester.depth)):
                depth_index = depth_index + self.model_tester.depth[i]
                img_size = (img_size + 2*self.model_tester.patch_padding[i] - self.model_tester.patch_size[i])/(self.model_tester.patch_stride[i])
                if i != self.model_tester.num_stages - 1: 
                    self.assertListEqual(
                        list(attentions[depth_index-1].shape[-3:]),
                        [self.model_tester.num_heads[i], img_size*img_size, img_size//4],
                    )
                else:
                    self.assertListEqual(
                        list(attentions[depth_index-1].shape[-3:]),
                        [self.model_tester.num_heads[i], img_size*img_size + 1, img_size//4 + 1],
                    )
                

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states

            expected_num_layers = sum(self.model_tester.depth) + 1
            self.assertEqual(len(hidden_states), expected_num_layers)

            # CvT has a different seq_length
            depth_index = 0
            img_size = self.model_tester.image_size
            for i in range(len(self.model_tester.depth)):
                depth_index = depth_index + self.model_tester.depth[i]
                img_size = (img_size + 2*self.model_tester.patch_padding[i] - self.model_tester.patch_size[i])/(self.model_tester.patch_stride[i]) 
                if i != self.model_tester.num_stages - 1:
                    self.assertListEqual(
                    list(hidden_states[depth_index-1].shape[-2:]),
                    [img_size*img_size, self.model_tester.dim_embed[i]],
                    )
                else:
                    self.assertListEqual(
                    list(hidden_states[depth_index-1].shape[-2:]),
                    [img_size*img_size + 1, self.model_tester.dim_embed[i]],
                    )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    # def test_for_masked_lm(self):
    #     config_and_inputs = self.model_tester.prepare_config_and_inputs()
    #     self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in CVT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CvTModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_vision
class CvTModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        return (
            CvTFeatureExtractor.from_pretrained("microsoft/cvt-base-patch16-224") if is_vision_available() else None
        )

    @slow
    def test_inference_image_classification_head_imagenet_1k(self):
        model = CvTForImageClassification.from_pretrained("microsoft/cvt-base-patch16-224").to(torch_device)

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor([-1.2385, -1.0987, -1.0108]).to(torch_device)

        self.assertTrue(torch.allclose(logits[0, :3], expected_slice, atol=1e-4))

        expected_class_idx = 281
        self.assertEqual(logits.argmax(-1).item(), expected_class_idx)

    @slow
    def test_inference_image_classification_head_imagenet_22k(self):
        model = CvTForImageClassification.from_pretrained("microsoft/cvt-large-patch16-224-pt22k-ft22k").to(
            torch_device
        )

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # verify the logits
        expected_shape = torch.Size((1, 21841))
        self.assertEqual(logits.shape, expected_shape)

        expected_slice = torch.tensor([1.6881, -0.2787, 0.5901]).to(torch_device)

        self.assertTrue(torch.allclose(logits[0, :3], expected_slice, atol=1e-4))

        expected_class_idx = 2396
        self.assertEqual(logits.argmax(-1).item(), expected_class_idx)
