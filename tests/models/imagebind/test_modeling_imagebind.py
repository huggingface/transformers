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
"""Testing suite for the PyTorch ImageBind model."""

import inspect
import os
import tempfile
import unittest

import numpy as np
from datasets import load_dataset

from transformers import (
    ImageBindAudioConfig,
    ImageBindConfig,
    ImageBindProcessor,
    ImageBindTextConfig,
    ImageBindVisionConfig,
)
from transformers.testing_utils import (
    require_torch,
    require_torchaudio,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_speech_available, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        ImageBindAudioModel,
        ImageBindAudioModelWithProjection,
        ImageBindModel,
        ImageBindTextModel,
        ImageBindTextModelWithProjection,
        ImageBindVisionModel,
        ImageBindVisionModelWithProjection,
    )


if is_vision_available():
    pass

if is_speech_available():
    import torchaudio


class ImageBindTextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.0,
        attention_dropout=0.0,
        max_position_embeddings=512,
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        logit_scale_init_value=14.2857,
        learnable_logit_scale=True,
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
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.logit_scale_init_value = logit_scale_init_value
        self.learnable_logit_scale = learnable_logit_scale
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return ImageBindTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            layer_norm_eps=self.layer_norm_eps,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            logit_scale_init_value=self.logit_scale_init_value,
            learnable_logit_scale=self.learnable_logit_scale,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = ImageBindTextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_model_with_projection(self, config, input_ids, input_mask):
        model = ImageBindTextModelWithProjection(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.text_embeds.shape, (self.batch_size, self.projection_dim))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class ImageBindTextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ImageBindTextModel, ImageBindTextModelWithProjection) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = ImageBindTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ImageBindTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_projection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_projection(*config_and_inputs)

    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="ImageBind does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="ImageBindTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="ImageBindTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "EduardoPacheco/imagebind-huge"
        model = ImageBindTextModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @slow
    def test_model_with_projection_from_pretrained(self):
        model_name = "EduardoPacheco/imagebind-huge"
        model = ImageBindTextModelWithProjection.from_pretrained(model_name)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "text_projection"))


class ImageBindVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=32,
        patch_size=8,
        num_channels=3,
        hidden_size=32,
        mlp_ratio=1.0,
        projection_dim=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        is_training=False,
        logit_scale_init_value=None,
        learnable_logit_scale=False,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.is_training = is_training
        self.logit_scale_init_value = logit_scale_init_value
        self.learnable_logit_scale = learnable_logit_scale
        self.scope = scope

        # Though in Vision we have a 3D conv the time dimension is always 1, thus we can use only spatial dimensions
        num_patches = (image_size // patch_size) ** 2
        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return ImageBindVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            mlp_ratio=self.mlp_ratio,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            logit_scale_init_value=self.logit_scale_init_value,
            learnable_logit_scale=self.learnable_logit_scale,
        )

    # TODO: fix image size and patch_size
    def create_and_check_model(self, config, pixel_values):
        model = ImageBindVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    # TODO: fix image size and patch_size
    def create_and_check_model_with_projection(self, config, pixel_values):
        model = ImageBindVisionModelWithProjection(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.image_embeds.shape, (self.batch_size, self.projection_dim))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class ImageBindVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as IMAGEBIND does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (ImageBindVisionModel, ImageBindVisionModelWithProjection) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = ImageBindVisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=ImageBindVisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="IMAGEBIND does not use inputs_embeds")
    def test_inputs_embeds(self):
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

    def test_model_with_projection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_projection(*config_and_inputs)

    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="ImageBindVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="ImageBindVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    @slow
    def test_model_from_pretrained(self):
        model_name = "EduardoPacheco/imagebind-huge"
        model = ImageBindVisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @slow
    def test_model_with_projection_from_pretrained(self):
        model_name = "EduardoPacheco/imagebind-huge"
        model = ImageBindVisionModelWithProjection.from_pretrained(model_name)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "vision_projection"))


class ImageBindAudioModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        patch_size=8,
        stride=8,
        num_channels=1,
        is_training=False,
        num_mel_bins=32,
        target_len=48,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        mlp_ratio=1.0,
        add_kv_bias=True,
        logit_scale_init_value=20.0,
        learnable_logit_scale=False,
        scope=None,
    ):
        self.parent = parent
        # Input audio can be batched with multiple clips
        self.num_clips = 3
        # If clips are batched then the batch size is multiplied by the number of clips
        self.actual_batch_size = batch_size
        self.batch_size = batch_size * self.num_clips  # this will be used internally
        self.patch_size = patch_size
        self.stride = stride
        self.num_channels = num_channels
        self.is_training = is_training
        self.num_mel_bins = num_mel_bins
        self.target_len = target_len
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.add_kv_bias = add_kv_bias
        self.logit_scale_init_value = logit_scale_init_value
        self.learnable_logit_scale = learnable_logit_scale
        self.scope = scope

        # In audio model the mel-spectogram image size is based on the number of mel bins and the target length
        patches_along_height_dim = int((num_mel_bins - patch_size) / stride + 1)
        patches_along_width_dim = int((target_len - patch_size) / stride + 1)
        num_patches = patches_along_height_dim * patches_along_width_dim

        self.encoder_seq_length = num_patches + 1
        self.key_length = num_patches + 1 if not add_kv_bias else num_patches + 2

    def prepare_config_and_inputs(self):
        input_features = floats_tensor(
            [self.actual_batch_size, self.num_clips, self.num_channels, self.num_mel_bins, self.target_len]
        )
        config = self.get_config()

        return config, input_features

    def get_config(self):
        return ImageBindAudioConfig(
            patch_size=self.patch_size,
            stride=self.stride,
            num_channels=self.num_channels,
            num_mel_bins=self.num_mel_bins,
            target_len=self.target_len,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            mlp_ratio=self.mlp_ratio,
            add_kv_bias=self.add_kv_bias,
            logit_scale_init_value=self.logit_scale_init_value,
            learnable_logit_scale=self.learnable_logit_scale,
        )

    def create_and_check_model(self, config, input_features):
        model = ImageBindAudioModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.encoder_seq_length, self.hidden_size)
        )
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_model_with_projection(self, config, input_features):
        model = ImageBindAudioModelWithProjection(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.encoder_seq_length, self.hidden_size)
        )
        self.parent.assertEqual(result.audio_embeds.shape, (self.actual_batch_size, self.projection_dim))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_features = config_and_inputs
        inputs_dict = {"input_features": input_features}
        return config, inputs_dict


@require_torch
class ImageBindAudioModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as IMAGEBIND does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (ImageBindAudioModel, ImageBindAudioModelWithProjection) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = ImageBindAudioModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=ImageBindAudioConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="ImageBind does not use inputs_embeds")
    def test_inputs_embeds(self):
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

            expected_arg_names = ["input_features"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_with_projection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_projection(*config_and_inputs)

    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="ImageBindAudioModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="ImageBindAudioModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    def test_model_get_set_embeddings(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    @slow
    def test_model_from_pretrained(self):
        model_name = "EduardoPacheco/imagebind-huge"
        model = ImageBindAudioModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @slow
    def test_model_with_projection_from_pretrained(self):
        model_name = "EduardoPacheco/imagebind-huge"
        model = ImageBindAudioModelWithProjection.from_pretrained(model_name)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "audio_projection"))


class ImageBindModelTester:
    def __init__(
        self,
        parent,
        text_kwargs=None,
        vision_kwargs=None,
        audio_kwargs=None,
        projection_dim=32,
        modality="text",
        is_training=True,
    ):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}
        if audio_kwargs is None:
            audio_kwargs = {}

        self.parent = parent
        self.text_model_tester = ImageBindTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = ImageBindVisionModelTester(parent, **vision_kwargs)
        self.audio_model_tester = ImageBindAudioModelTester(parent, **audio_kwargs)
        self.projection_dim = projection_dim
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        # This is to make things easier and reuse ImageBindModelTester for all modalities
        self.modality = modality
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        _, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        _, input_features = self.audio_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, pixel_values, input_ids, attention_mask, input_features

    def get_config(self):
        return ImageBindConfig(
            self.text_model_tester.get_config().to_dict(),
            self.vision_model_tester.get_config().to_dict(),
            self.audio_model_tester.get_config().to_dict(),
            projection_dim=self.projection_dim,
        )

    def create_and_check_text_vision_pair(self, config, pixel_values, input_ids, attention_mask):
        model = ImageBindModel(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.text_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.text_model_tester.batch_size, self.vision_model_tester.batch_size)
        )

    def create_and_check_audio_vision_pair(self, config, pixel_values, input_features):
        model = ImageBindModel(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(pixel_values=pixel_values, input_features=input_features)
        self.parent.assertEqual(
            result.logits_per_image.shape, (self.vision_model_tester.batch_size, self.audio_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_audio.shape, (self.audio_model_tester.batch_size, self.vision_model_tester.batch_size)
        )

    def create_and_check_model(self, config, pixel_values, input_ids=None, attention_mask=None, input_features=None):
        if self.modality == "text":
            self.create_and_check_text_vision_pair(
                config,
                pixel_values,
                input_ids,
                attention_mask,
            )
        elif self.modality == "audio":
            self.create_and_check_audio_vision_pair(config, pixel_values, input_features)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, input_ids, attention_mask, input_features = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "return_loss": True,
        }

        if self.modality == "text":
            inputs_dict.pop("input_features")
        elif self.modality == "audio":
            inputs_dict.pop("input_ids")
            inputs_dict.pop("attention_mask")

        return config, inputs_dict


@require_torch
class ImageBindModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (ImageBindModel,) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": ImageBindModel} if is_torch_available() else {}
    fx_compatible = False
    test_torchscript = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = ImageBindModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="ImageBindModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    def _create_and_check_torchscript(self, config, inputs_dict):
        if not self.test_torchscript:
            return

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()

            try:
                traced_model = torch.jit.trace(model, example_kwarg_inputs=inputs_dict)
            except RuntimeError:
                self.fail("Couldn't trace module.")

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pt_file_name = os.path.join(tmp_dir_name, "traced_model.pt")

                try:
                    torch.jit.save(traced_model, pt_file_name)
                except Exception:
                    self.fail("Couldn't save module.")

                try:
                    loaded_model = torch.jit.load(pt_file_name)
                except Exception:
                    self.fail("Couldn't load module.")

            model.to(torch_device)
            model.eval()

            loaded_model.to(torch_device)
            loaded_model.eval()

            model_state_dict = model.state_dict()
            loaded_model_state_dict = loaded_model.state_dict()

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save ImageBindConfig and check if we can load ImageBindVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = ImageBindVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save ImageBindConfig and check if we can load ImageBindTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = ImageBindTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @unittest.skip(reason="ImageBindModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "EduardoPacheco/imagebind-huge"
        model = ImageBindModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


def prepare_inputs():
    ds = load_dataset("EduardoPacheco/imagebind-example-data", split="train")
    images = ds["image"]
    texts = ds["text"]
    audios = [
        torchaudio.functional.resample(
            torch.from_numpy(audio["array"]), orig_freq=audio["sampling_rate"], new_freq=16000
        ).numpy()
        for audio in ds["audio"]
    ]

    return images, texts, audios


@require_vision
@require_torchaudio
@require_torch
class ImageBindModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "EduardoPacheco/imagebind-huge"
        model = ImageBindModel.from_pretrained(model_name).to(torch_device)
        processor = ImageBindProcessor.from_pretrained(model_name)

        images, texts, audios = prepare_inputs()
        inputs = processor(text=texts, images=images, audio=audios, padding=True, return_tensors="pt").to(torch_device)

        expected_input_features = torch.tensor(
            [
                [-1.2776, -0.9167, -1.2776],
                [-1.2439, -0.8372, -0.8748],
                [-1.1235, -0.7492, -1.0867],
            ]
        )

        expected_pixel_values = torch.tensor(
            [[-0.1134, 0.7392, 1.3069], [-0.6244, 0.1089, 0.2688], [-0.8434, 0.1089, 0.9088]]
        )

        expected_input_ids = torch.tensor(
            [[49406, 320, 3329, 49407, 49407], [49406, 320, 1615, 49407, 49407], [49406, 320, 1929, 269, 49407]]
        )

        expected_attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])

        self.assertTrue(torch.allclose(inputs.input_features[:, :, 0, 0, 0], expected_input_features, atol=1e-4))
        self.assertTrue(torch.allclose(inputs.pixel_values[:, :, 0, 0], expected_pixel_values, atol=1e-4))
        self.assertTrue(torch.allclose(inputs.input_ids, expected_input_ids, atol=1e-4))
        self.assertTrue(torch.allclose(inputs.attention_mask, expected_attention_mask, atol=1e-4))

        with torch.no_grad():
            outputs_vision_text = model(
                pixel_values=inputs.pixel_values, input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
            )
            outputs_vision_audio = model(pixel_values=inputs.pixel_values, input_features=inputs.input_features)

        expected_image_embeds = torch.tensor(
            [
                [0.0188, 0.0075, 0.0532, 0.0326, -0.0159],
                [0.0190, 0.0106, 0.0275, 0.0189, -0.0268],
                [-0.0104, -0.0203, 0.0048, -0.0158, 0.0076],
            ]
        )
        expected_text_embeds = torch.tensor(
            [
                [-1.3476, -1.5732, -0.7386, 9.7949, 0.5856],
                [-0.4342, -0.9050, -4.2879, 7.4123, -0.4906],
                [-1.0745, -4.0049, -1.0697, 5.8861, -0.7583],
            ]
        )
        expected_audio_embeds = torch.tensor(
            [
                [0.3245, -0.3749, 0.3955, 0.5600, -0.1932],
                [0.7091, 0.2072, -1.0133, 0.4689, -0.2142],
                [-0.0282, -0.4923, 1.0058, 0.0459, -0.2271],
            ]
        )

        self.assertTrue(torch.allclose(outputs_vision_text.image_embeds[:, :5], expected_image_embeds, atol=1e-4))
        self.assertTrue(torch.allclose(outputs_vision_text.text_embeds[:, :5], expected_text_embeds, atol=1e-4))
        self.assertTrue(torch.allclose(outputs_vision_audio.audio_embeds[:, :5], expected_audio_embeds, atol=1e-4))
        self.assertTrue(torch.allclose(outputs_vision_text.image_embeds, outputs_vision_audio.image_embeds, atol=1e-4))

        expected_logits_per_audio = torch.tensor(
            [[7.3541, 1.1908, 2.2897], [1.1930, 3.0097, 2.0238], [0.9584, 1.2224, 4.2325]]
        )

        expected_logits_per_image_with_text = torch.tensor(
            [[23.6142, 19.1165, 13.2448], [12.1343, 23.4165, 11.8823], [15.8471, 20.1186, 24.8246]]
        )

        self.assertTrue(torch.allclose(outputs_vision_audio.logits_per_audio, expected_logits_per_audio, atol=1e-4))
        self.assertTrue(
            torch.allclose(outputs_vision_text.logits_per_image, expected_logits_per_image_with_text, atol=1e-4)
        )
