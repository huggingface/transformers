# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch Hiera model. """

import math
import os
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np

from transformers import HieraConfig
from transformers.testing_utils import (
    require_accelerate,
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import (
    CONFIG_NAME,
    GENERATION_CONFIG_NAME,
    cached_property,
    is_torch_available,
    is_vision_available,
)

from ...test_backbone_common import BackboneTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn
    from torch.nn import functional as F

    from transformers import HieraBackbone, HieraForImageClassification, HieraForPreTraining, HieraModel
    from transformers.models.hiera.modeling_hiera import HIERA_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import AutoImageProcessor


class HieraModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        input_size=[224, 224],
        mlp_ratio=1.0,
        num_channels=3,
        depths=[1, 1, 1, 1],
        patch_stride=[4, 4],
        masked_unit_size=[8, 8],
        initial_num_heads=1,
        num_head_multiplier=2.0,
        embed_dim_multiplier=2.0,
        is_training=True,
        use_labels=True,
        embed_dim=8,
        hidden_act="gelu",
        decoder_embed_dim=2,
        decoder_depth=1,
        decoder_num_heads=1,
        initializer_range=0.02,
        scope=None,
        type_sequence_label_size=10,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.input_size = input_size
        self.mlp_ratio = mlp_ratio
        self.num_channels = num_channels
        self.depths = depths
        self.patch_stride = patch_stride
        self.masked_unit_size = masked_unit_size
        self.initial_num_heads = initial_num_heads
        self.num_head_multiplier = num_head_multiplier
        self.embed_dim_multiplier = embed_dim_multiplier
        self.is_training = is_training
        self.use_labels = use_labels
        self.embed_dim = embed_dim
        self.hidden_act = hidden_act
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.initializer_range = initializer_range
        self.scope = scope
        self.type_sequence_label_size = type_sequence_label_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.input_size[0], self.input_size[1]])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_noise_for_mae(self):
        mask_spatial_shape = [
            i // s // ms for i, s, ms in zip(self.input_size, self.patch_stride, self.masked_unit_size)
        ]
        num_windows = math.prod(mask_spatial_shape)
        return floats_tensor([self.batch_size, num_windows])

    def get_config(self):
        return HieraConfig(
            embed_dim=self.embed_dim,
            input_size=self.input_size,
            patch_stride=self.patch_stride,
            masked_unit_size=self.masked_unit_size,
            mlp_ratio=self.mlp_ratio,
            num_channels=self.num_channels,
            depths=self.depths,
            initial_num_heads=self.initial_num_heads,
            num_head_multiplier=self.num_head_multiplier,
            embed_dim_multiplier=self.embed_dim_multiplier,
            hidden_act=self.hidden_act,
            decoder_embed_dim=self.decoder_embed_dim,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.decoder_num_heads,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values, labels):
        model = HieraModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        tokens_spatial_shape = [i // s for i, s in zip(self.input_size, config.patch_stride)]
        expected_seq_len = math.prod(tokens_spatial_shape) // math.prod(config.query_stride) ** (config.num_query_pool)
        expected_dim = int(config.embed_dim * config.embed_dim_multiplier ** (len(config.depths) - 1))

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, expected_seq_len, expected_dim))

    def create_and_check_backbone(self, config, pixel_values, labels):
        model = HieraBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify hidden states
        self.parent.assertEqual(len(result.feature_maps), len(config.out_features))
        num_patches = config.input_size[0] // config.patch_stride[0] // config.masked_unit_size[0]
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape), [self.batch_size, model.channels[0], num_patches, num_patches]
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), len(config.out_features))

        # verify backbone works with out_features=None
        config.out_features = None
        model = HieraBackbone(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)

        # verify feature maps
        self.parent.assertEqual(len(result.feature_maps), 1)
        self.parent.assertListEqual(
            list(result.feature_maps[0].shape), [self.batch_size, model.channels[-1], num_patches, num_patches]
        )

        # verify channels
        self.parent.assertEqual(len(model.channels), 1)

    def create_and_check_for_pretraining(self, config, pixel_values, labels):
        model = HieraForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        noise = self.get_noise_for_mae()
        result = model(pixel_values, noise=noise)
        pred_stride = config.patch_stride[-1] * (config.query_stride[-1] ** config.num_query_pool)
        num_patches = self.input_size[0] // pred_stride
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, num_patches**2, self.num_channels * pred_stride**2)
        )

        # test greyscale images
        config.num_channels = 1
        model = HieraForPreTraining(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.input_size[0], self.input_size[0]])
        result = model(pixel_values, noise=noise)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, num_patches**2, pred_stride**2))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.type_sequence_label_size
        model = HieraForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

        # test greyscale images
        config.num_channels = 1
        model = HieraForImageClassification(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.input_size[0], self.input_size[0]])
        result = model(pixel_values)
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
class HieraModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as Hiera does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (
            HieraModel,
            HieraBackbone,
            HieraForImageClassification,
            HieraForPreTraining,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"image-feature-extraction": HieraModel, "image-classification": HieraForImageClassification}
        if is_torch_available()
        else {}
    )
    fx_compatible = False

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = HieraModelTester(self)
        self.config_tester = ConfigTester(self, config_class=HieraConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.create_and_test_config_to_json_string()
        self.config_tester.create_and_test_config_to_json_file()
        self.config_tester.create_and_test_config_from_and_save_pretrained()
        self.config_tester.create_and_test_config_with_num_labels()
        self.config_tester.check_config_can_be_init_without_params()
        self.config_tester.check_config_arguments_init()

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
            expected_num_attentions = len(self.model_tester.depths)
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            seq_len = math.prod([i // s for i, s in zip(config.input_size, config.patch_stride)])
            mask_unit_area = math.prod(config.masked_unit_size)
            num_windows = seq_len // mask_unit_area
            if model_class.__name__ == "HieraForPreTraining":
                num_windows = int(num_windows * (1 - config.mask_ratio))
                seq_len = int(num_windows * mask_unit_area)
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), expected_num_attentions)

            self.assertListEqual(
                list(attentions[0].shape[-4:]),
                [self.model_tester.initial_num_heads, num_windows, mask_unit_area, seq_len // num_windows],
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

            # also another +1 for reshaped_hidden_states
            added_hidden_states = 1 if model_class.__name__ == "HieraBackbone" else 2
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), expected_num_attentions)

            self.assertListEqual(
                list(self_attentions[0].shape[-4:]),
                [self.model_tester.initial_num_heads, num_windows, mask_unit_area, seq_len // num_windows],
            )

    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class, image_size):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", len(self.model_tester.depths) + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            # Swin has a different seq_length
            patch_size = config.patch_stride

            num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
            if model_class.__name__ == "HieraForPreTraining":
                mask_unit_area = math.prod(config.masked_unit_size)
                num_windows = num_patches // mask_unit_area
                num_windows = int(num_windows * (1 - config.mask_ratio))
                num_patches = int(num_windows * mask_unit_area)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [num_patches, self.model_tester.embed_dim],
            )

            if not model_class.__name__ == "HieraBackbone":
                reshaped_hidden_states = outputs.reshaped_hidden_states
                self.assertEqual(len(reshaped_hidden_states), expected_num_layers)

                batch_size = reshaped_hidden_states[0].shape[0]
                num_channels = reshaped_hidden_states[0].shape[-1]

                reshaped_hidden_states = reshaped_hidden_states[0].view(batch_size, -1, num_channels)
                self.assertListEqual(
                    list(reshaped_hidden_states.shape[-2:]),
                    [num_patches, self.model_tester.embed_dim],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        image_size = self.model_tester.input_size

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class, image_size)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class, image_size)

    def test_batching_equivalence(self):
        """
        Tests that the model supports batching and that the output is the nearly the same for the same input in
        different batch sizes.
        (Why "nearly the same" not "exactly the same"? Batching uses different matmul shapes, which often leads to
        different results: https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535)
        """

        def get_tensor_equivalence_function(batched_input):
            # models operating on continuous spaces have higher abs difference than LMs
            # instead, we can rely on cos distance for image/speech models, similar to `diffusers`
            if "input_ids" not in batched_input:
                return lambda tensor1, tensor2: (
                    1.0 - F.cosine_similarity(tensor1.float().flatten(), tensor2.float().flatten(), dim=0, eps=1e-38)
                )
            return lambda tensor1, tensor2: torch.max(torch.abs(tensor1 - tensor2))

        def recursive_check(batched_object, single_row_object, model_name, key):
            if isinstance(batched_object, (list, tuple)):
                for batched_object_value, single_row_object_value in zip(batched_object, single_row_object):
                    recursive_check(batched_object_value, single_row_object_value, model_name, key)
            elif isinstance(batched_object, dict):
                for batched_object_value, single_row_object_value in zip(
                    batched_object.values(), single_row_object.values()
                ):
                    recursive_check(batched_object_value, single_row_object_value, model_name, key)
            # do not compare returned loss (0-dim tensor) / codebook ids (int) / caching objects
            elif batched_object is None or not isinstance(batched_object, torch.Tensor):
                return
            elif batched_object.dim() == 0:
                return
            else:
                # indexing the first element does not always work
                # e.g. models that output similarity scores of size (N, M) would need to index [0, 0]
                slice_ids = [slice(0, index) for index in single_row_object.shape]
                batched_row = batched_object[slice_ids]
                self.assertFalse(
                    torch.isnan(batched_row).any(), f"Batched output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(batched_row).any(), f"Batched output has `inf` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isnan(single_row_object).any(), f"Single row output has `nan` in {model_name} for key={key}"
                )
                self.assertFalse(
                    torch.isinf(single_row_object).any(), f"Single row output has `inf` in {model_name} for key={key}"
                )
                self.assertTrue(
                    (equivalence(batched_row, single_row_object)) <= 1e-03,
                    msg=(
                        f"Batched and Single row outputs are not equal in {model_name} for key={key}. "
                        f"Difference={equivalence(batched_row, single_row_object)}."
                    ),
                )

        config, batched_input = self.model_tester.prepare_config_and_inputs_for_common()
        equivalence = get_tensor_equivalence_function(batched_input)

        for model_class in self.all_model_classes:
            config.output_hidden_states = True

            model_name = model_class.__name__
            if hasattr(self.model_tester, "prepare_config_and_inputs_for_model_class"):
                config, batched_input = self.model_tester.prepare_config_and_inputs_for_model_class(model_class)
            if model_name == "HieraForPreTraining":
                batched_input["noise"] = self.model_tester.get_noise_for_mae()
            batched_input_prepared = self._prepare_for_class(batched_input, model_class)
            model = model_class(config).to(torch_device).eval()

            batch_size = self.model_tester.batch_size
            single_row_input = {}
            for key, value in batched_input_prepared.items():
                if isinstance(value, torch.Tensor) and value.shape[0] % batch_size == 0:
                    # e.g. musicgen has inputs of size (bs*codebooks). in most cases value.shape[0] == batch_size
                    single_batch_shape = value.shape[0] // batch_size
                    single_row_input[key] = value[:single_batch_shape]
                else:
                    single_row_input[key] = value

            with torch.no_grad():
                model_batched_output = model(**batched_input_prepared)
                model_row_output = model(**single_row_input)

            if isinstance(model_batched_output, torch.Tensor):
                model_batched_output = {"model_output": model_batched_output}
                model_row_output = {"model_output": model_row_output}

            for key in model_batched_output:
                # DETR starts from zero-init queries to decoder, leading to cos_similarity = `nan`
                if hasattr(self, "zero_init_hidden_state") and "decoder_hidden_states" in key:
                    model_batched_output[key] = model_batched_output[key][1:]
                    model_row_output[key] = model_row_output[key][1:]
                recursive_check(model_batched_output[key], model_row_output[key], model_name, key)

    def test_model_outputs_equivalence(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def set_nan_tensor_to_zero(t):
            t[t != t] = 0
            return t

        def check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs={}):
            with torch.no_grad():
                tuple_output = model(**tuple_inputs, return_dict=False, **additional_kwargs)
                dict_output = model(**dict_inputs, return_dict=True, **additional_kwargs).to_tuple()

                def recursive_check(tuple_object, dict_object):
                    if isinstance(tuple_object, (List, Tuple)):
                        for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif isinstance(tuple_object, Dict):
                        for tuple_iterable_value, dict_iterable_value in zip(
                            tuple_object.values(), dict_object.values()
                        ):
                            recursive_check(tuple_iterable_value, dict_iterable_value)
                    elif tuple_object is None:
                        return
                    else:
                        self.assertTrue(
                            torch.allclose(
                                set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5
                            ),
                            msg=(
                                "Tuple and dict output are not equal. Difference:"
                                f" {torch.max(torch.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                                f" {torch.isnan(tuple_object).any()} and `inf`: {torch.isinf(tuple_object)}. Dict has"
                                f" `nan`: {torch.isnan(dict_object).any()} and `inf`: {torch.isinf(dict_object)}."
                            ),
                        )

                recursive_check(tuple_output, dict_output)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            additional_kwargs = {}
            if model_class.__name__ == "HieraForPreTraining":
                additional_kwargs["noise"] = self.model_tester.get_noise_for_mae()

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class)
            additional_kwargs["output_hidden_states"] = True
            check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs)

            tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs)

            if self.has_attentions:
                # Removing "output_hidden_states"
                del additional_kwargs["output_hidden_states"]

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class)
                additional_kwargs["output_attentions"] = True
                check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs)

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs)

                tuple_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                dict_inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                additional_kwargs["output_hidden_states"] = True
                check_equivalence(model, tuple_inputs, dict_inputs, additional_kwargs)

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            out_1 = first.cpu().numpy()
            out_2 = second.cpu().numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                if model_class.__name__ == "HieraForPreTraining":
                    inputs_dict["noise"] = self.model_tester.get_noise_for_mae()
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]
                second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_save_load(out1, out2):
            # make sure we don't have nans
            out_2 = out2.cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            out_1 = out1.cpu().numpy()
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                if model_class.__name__ == "HieraForPreTraining":
                    inputs_dict["noise"] = self.model_tester.get_noise_for_mae()
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # the config file (and the generation config file, if it can generate) should be saved
                self.assertTrue(os.path.exists(os.path.join(tmpdirname, CONFIG_NAME)))
                self.assertEqual(
                    model.can_generate(), os.path.exists(os.path.join(tmpdirname, GENERATION_CONFIG_NAME))
                )

                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)
                with torch.no_grad():
                    second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_save_load(tensor1, tensor2)
            else:
                check_save_load(first, second)

    @unittest.skip(reason="Hiera Transformer does not use feedforward chunking")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="Hiera does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_backbone(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_backbone(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in HIERA_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = HieraModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class HieraModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            AutoImageProcessor.from_pretrained("EduardoPacheco/hiera-tiny-224-in1k") if is_vision_available() else None
        )

    @slow
    def test_inference_image_classification_head(self):
        model = HieraForImageClassification.from_pretrained("EduardoPacheco/hiera-tiny-224-in1k").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        expected_pixel_values = torch.tensor(
            [
                [[0.2967, 0.4679, 0.4508], [0.3309, 0.4337, 0.3309], [0.3309, 0.3823, 0.3309]],
                [[-1.5455, -1.4930, -1.5455], [-1.5280, -1.4755, -1.5980], [-1.5630, -1.5280, -1.4755]],
                [[-0.6367, -0.4973, -0.5321], [-0.7936, -0.6715, -0.6715], [-0.8284, -0.7413, -0.5670]],
            ]
        ).to(torch_device)

        self.assertTrue(torch.allclose(inputs.pixel_values[0, :3, :3, :3], expected_pixel_values, atol=1e-4))

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1, 1000))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor([[0.8028, 0.2409, -0.2254, -0.3712, -0.2848]]).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0, :5], expected_slice, atol=1e-4))

    @slow
    def test_inference_for_pretraining(self):
        # make random mask reproducible
        np.random.seed(2)

        model = HieraForPreTraining.from_pretrained("EduardoPacheco/hiera-tiny-224-mae").to(torch_device)
        image_processor = self.default_image_processor

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt").to(torch_device)

        config = model.config
        mask_spatial_shape = [
            i // s // ms for i, s, ms in zip(config.input_size, config.patch_stride, config.masked_unit_size)
        ]
        num_windows = math.prod(mask_spatial_shape)
        noise = np.random.uniform(size=(1, num_windows))

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs, noise=torch.from_numpy(noise).to(device=torch_device))

        # verify the logits
        expected_shape = torch.Size((1, 196, 768))
        self.assertEqual(outputs.logits.shape, expected_shape)

        expected_slice = torch.tensor(
            [
                [1.5719, 1.5743, 1.5732, 1.5791, 1.5958],
                [1.9311, 1.9409, 1.9440, 1.9545, 1.9605],
                [1.6149, 1.8555, 1.2720, 1.5385, 1.5067],
                [1.2804, 1.8411, 0.8342, 1.5867, 1.5384],
                [2.1131, 2.0876, 2.0349, 1.9921, 1.9496],
            ]
        )

        self.assertTrue(torch.allclose(outputs.logits[0, :5, :5], expected_slice.to(torch_device), atol=1e-4))

    @slow
    @require_accelerate
    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_fp16(self):
        r"""
        A small test to make sure that inference work in half precision without any problem.
        """
        model = HieraModel.from_pretrained(
            "EduardoPacheco/hiera-tiny-224", torch_dtype=torch.float16, device_map="auto"
        )
        image_processor = self.default_image_processor

        image = prepare_img()
        inputs = image_processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(torch_device)

        # forward pass to make sure inference works in fp16
        with torch.no_grad():
            _ = model(pixel_values)


@require_torch
class HieraBackboneTest(unittest.TestCase, BackboneTesterMixin):
    all_model_classes = (HieraBackbone,) if is_torch_available() else ()
    config_class = HieraConfig

    def setUp(self):
        self.model_tester = HieraModelTester(self)
