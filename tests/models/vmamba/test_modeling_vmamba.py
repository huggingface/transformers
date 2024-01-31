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
""" Testing suite for the PyTorch VMamba model. """


import copy
import inspect
import unittest

from transformers import VMambaConfig
from transformers.models.perceiver.image_processing_perceiver import PerceiverImageProcessor
from transformers.models.vmamba.modeling_vmamba import VMAMBA_PRETRAINED_MODEL_ARCHIVE_LIST
from transformers.testing_utils import (
    require_accelerate,
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import VMambaForImageClassification, VMambaModel


if is_vision_available():
    from PIL import Image


class VMambaModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=32,
        is_training=False,
        use_labels=True,
        patch_size=4,
        in_channels=3,
        num_labels=1000,
        num_hidden_layers=3,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        d_state=16,
        drop_rate=0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        patch_norm=True,
        use_checkpoint=False,
        type_sequence_label_size=10,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.is_training = is_training
        self.use_labels = use_labels
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_labels = num_labels
        self.num_hidden_layers = num_hidden_layers
        self.depths = depths
        self.dims = dims
        self.d_state = d_state
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.patch_norm = patch_norm
        self.use_checkpoint = use_checkpoint
        self.type_sequence_label_size = type_sequence_label_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.in_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        config = VMambaConfig(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            num_labels=self.num_labels,
            depths=self.depths,
            dims=self.dims,
            d_state=self.d_state,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            patch_norm=self.patch_norm,
            use_checkpoint=False,
        )
        return config

    def create_and_check_model(self, config, pixel_values, labels):
        model = VMambaModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.dims[-1]))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        model = VMambaForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

        # test greyscale images
        config.in_channels = 1
        model = VMambaForImageClassification(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs

        config.depths = [2, 2]
        config.dims = [8, 16]
        config.d_state = 8
        config.num_labels = 2

        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class VMambaModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as VMamba does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (
            VMambaModel,
            VMambaForImageClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"feature-extraction": VMambaModel, "image-classification": VMambaForImageClassification}
        if is_torch_available()
        else {}
    )

    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    has_attentions = False

    def setUp(self):
        self.model_tester = VMambaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VMambaConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        # we don't test common_properties and arguments_init as these don't apply for Perceiver
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()

    @unittest.skip(reason="VMamba does not use attention")
    def test_attention_outputs(self):
        pass

    # @unittest.skip(reason="VMamba does not use inputs_embeds")
    # def test_inputs_embeds(self):
    #     pass
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            pixel_values = inputs["pixel_values"]
            del inputs["pixel_values"]

            wte = model.get_input_embeddings()
            embeddings = wte(pixel_values)

            expected_spatial_shape = pixel_values.shape[-1] // config.patch_size
            self.assertEqual(
                embeddings.shape,
                (pixel_values.shape[0], config.dims[0], expected_spatial_shape, expected_spatial_shape),
            )

    unittest.skip(reason="VMamba does not use output embeddings")

    def test_model_common_attributes(self):
        pass

    unittest.skip(reason="VMamba does not use attention")

    def test_retain_grad_hidden_states_attentions(self):
        pass

    # remove
    def test_initialization(self):
        super().test_initialization()

    def test_torch_fx(self):
        super().test_torch_fx()

    def test_save_load_fast_init_from_base(self):
        super().test_save_load_fast_init_from_base()

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    def test_model_main_input_name(self):
        for model_class in self.all_model_classes:
            model_signature = inspect.signature(getattr(model_class, "forward"))
            # The main input is the name of the argument after `self`
            observed_main_input_name = list(model_signature.parameters.keys())[1]
            expected_main_input_name = "pixel_values"
            self.assertEqual(expected_main_input_name, observed_main_input_name)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in VMAMBA_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = VMambaModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class VMambaModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return PerceiverImageProcessor() if is_vision_available() else None

    @slow
    def test_inference_image_classification_head(self):
        model = VMambaForImageClassification.from_pretrained("/home/derk/hf_checkpoints/").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([-1.5905, 1.6459, -0.3650]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4))

    @slow
    @require_accelerate
    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_fp16(self):
        r"""
        A small test to make sure that inference work in half precision without any problem.
        """
        model = VMambaModel.from_pretrained("/home/derk/hf_checkpoints/", torch_dtype=torch.float16, device_map="auto")
        image_processor = self.default_image_processor

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(torch_device)

        # forward pass to make sure inference works in fp16
        with torch.no_grad():
            _ = model(pixel_values)
