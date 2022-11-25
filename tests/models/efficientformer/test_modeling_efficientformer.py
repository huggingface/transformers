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
""" Testing suite for the PyTorch EfficientFormer model. """


import inspect
import unittest
from typing import Optional, Tuple, Dict, Union

from transformers import EfficientFormerConfig
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        EfficientFormerForImageClassification,
        EfficientFormerForMaskedImageModeling,
        EfficientFormerModel,
    )
    from transformers.models.efficientformer.modeling_efficientformer import (
        EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
    )


if is_vision_available():
    from PIL import Image

    from transformers import EfficientFormerFeatureExtractor


class EfficientFormerModelTester:
    def __init__(
        self,
        # TODO: type
        parent,
        batch_size: int = 13,
        image_size: int = 224,
        patch_size: int = 2,
        num_channels: int = 3,
        is_training: bool = True,
        use_labels: bool = True,
        hidden_size: int = 448,  # output of the encoder
        # TODO: not sure
        num_hidden_layers: int = 7,  # For the l1
        num_attention_heads: int = 8,
        intermediate_size: int = 37,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        type_sequence_label_size: int = 10,
        initializer_range: float = 0.02,
        # TODO: type?
        scope=None,
        encoder_stride=2,
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
        self.encoder_stride = encoder_stride

        # in EfficientFormer, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        # num_patches = (image_size // patch_size) ** 2

        # TODO: Wrronly calculated.
        # config = self.get_config()
        # m = EfficientFormerConvStem(config, config.embed_dims)
        # num_patches = ((image_size**2) / 16**2) // patch_size
        num_patches = 48
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self) -> Tuple[EfficientFormerConfig, torch.Tensor, torch.Tensor]:
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()
        return config, pixel_values, labels

    def get_config(self) -> EfficientFormerConfig:
        return EfficientFormerConfig(
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
            encoder_stride=self.encoder_stride,
        )

    def create_and_check_model(
        self, config: EfficientFormerConfig, pixel_values: torch.Tensor, labels: torch.Tensor
    ) -> None:
        model = EfficientFormerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_image_modeling(
        self, config: EfficientFormerConfig, pixel_values: torch.Tensor, labels: torch.Tensor
    ) -> None:
        model = EfficientFormerForMaskedImageModeling(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_channels, self.image_size, self.image_size)
        )

        # test greyscale images
        config.num_channels = 1
        model = EfficientFormerForMaskedImageModeling(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, 1, self.image_size, self.image_size))

    def create_and_check_for_image_classification(
        self, config: EfficientFormerConfig, pixel_values: torch.Tensor, labels: torch.Tensor
    ) -> None:
        config.num_labels = self.type_sequence_label_size
        model = EfficientFormerForImageClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

        # test greyscale images
        config.num_channels = 1
        model = EfficientFormerForImageClassification(config)
        model.to(torch_device)
        model.eval()

        pixel_values = floats_tensor([self.batch_size, 1, self.image_size, self.image_size])
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def prepare_config_and_inputs_for_common(self) -> Tuple[EfficientFormerConfig, Dict[str, torch.Tensor]]:
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class EfficientFormerModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as EfficientFormer does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (
            EfficientFormerModel,
            EfficientFormerForImageClassification,
            # Not implemented yet
            # EfficientFormerForMaskedImageModeling,
        )
        if is_torch_available()
        else ()
    )
    fx_compatible = False

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self) -> None:
        self.model_tester = EfficientFormerModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=EfficientFormerConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self) -> None:
        self.config_tester.run_common_tests()

    @unittest.skip(reason="EfficientFormer does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self) -> None:
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self) -> None:
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self) -> None:
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="EfficientFormer does not implement masked image modeling yet")
    def test_for_masked_image_modeling(self) -> None:
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_image_modeling(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self) -> None:
        for model_name in EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = EfficientFormerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img() -> Image.Image:
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
class EfficientFormerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self) -> Union[EfficientFormerFeatureExtractor, None]:
        return (
            EfficientFormerFeatureExtractor.from_pretrained("snap-research/efficientformer-l1")
            if is_vision_available()
            else None
        )

    @slow
    def test_inference_image_classification_head(self) -> None:
        model = EfficientFormerForImageClassification.from_pretrained("snap-research/efficientformer-l1").to(
            torch_device
        )

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        inputs = feature_extractor(images=image, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size((1000))
        self.assertEqual(outputs.logits[0].shape, expected_shape)
        self.assertEqual(outputs.logits[1].shape, expected_shape)

        expected_slice_1 = torch.tensor([-0.8568, 0.3356, -0.1594]).to(torch_device)
        expected_slice_2 = torch.tensor([-1.4141, 1.7621, 0.5935]).to(torch_device)

        self.assertTrue(torch.allclose(outputs.logits[0][:3], expected_slice_1, atol=1e-4))
        self.assertTrue(torch.allclose(outputs.logits[1][:3], expected_slice_2, atol=1e-4))
