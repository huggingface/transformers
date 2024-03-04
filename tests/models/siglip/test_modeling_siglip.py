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
""" Testing suite for the PyTorch SigLIP model. """


import inspect
import os
import tempfile
import unittest

import numpy as np
import requests

from transformers import SiglipConfig, SiglipTextConfig, SiglipVisionConfig
from transformers.testing_utils import (
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

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

    from transformers import SiglipForImageClassification, SiglipModel, SiglipTextModel, SiglipVisionModel
    from transformers.models.siglip.modeling_siglip import SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import SiglipProcessor


class SiglipVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        num_hidden_layers=2,
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

        # in ViT, the seq length equals the number of patches
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches

    # Copied from tests.models.clip.test_modeling_clip.CLIPVisionModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return SiglipVisionConfig(
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
        model = SiglipVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    # Copied from tests.models.clip.test_modeling_clip.CLIPVisionModelTester.prepare_config_and_inputs_for_common
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class SiglipVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as SIGLIP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (SiglipVisionModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = SiglipVisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=SiglipVisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="SIGLIP does not use inputs_embeds")
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

    @unittest.skip(reason="SiglipVisionModel does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="SiglipVisionModel does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SiglipVisionModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="SiglipVisionModel does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="SiglipVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="SiglipVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="Siglip uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = SiglipVisionModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class SiglipTextModelTester:
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
        num_hidden_layers=2,
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

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTester.prepare_config_and_inputs
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
        return SiglipTextConfig(
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
        model = SiglipTextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTester.prepare_config_and_inputs_for_common
    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class SiglipTextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (SiglipTextModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False
    model_split_percents = [0.5, 0.8, 0.9]

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.setUp with CLIP->Siglip
    def setUp(self):
        self.model_tester = SiglipTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SiglipTextConfig, hidden_size=37)

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_config
    def test_config(self):
        self.config_tester.run_common_tests()

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_model
    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_training
    def test_training(self):
        pass

    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_training_gradient_checkpointing
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_training_gradient_checkpointing_use_reentrant
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_training_gradient_checkpointing_use_reentrant_false
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Siglip does not use inputs_embeds")
    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_inputs_embeds
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="SiglipTextModel has no base class and is not available in MODEL_MAPPING")
    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_save_load_fast_init_from_base
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="SiglipTextModel has no base class and is not available in MODEL_MAPPING")
    # Copied from tests.models.clip.test_modeling_clip.CLIPTextModelTest.test_save_load_fast_init_to_base
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="Siglip uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = SiglipTextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class SiglipModelTester:
    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = SiglipTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = SiglipVisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return SiglipConfig.from_text_vision_configs(
            self.text_model_tester.get_config(),
            self.vision_model_tester.get_config(),
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = SiglipModel(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids, pixel_values, attention_mask)
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
            "return_loss": False,
        }
        return config, inputs_dict


@require_torch
class SiglipModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (SiglipModel,) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": SiglipModel} if is_torch_available() else {}
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.setUp with CLIP->Siglip
    def setUp(self):
        self.model_tester = SiglipModelTester(self)

    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_model
    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_hidden_states_output
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_inputs_embeds
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_retain_grad_hidden_states_attentions
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="SiglipModel does not have input/output embeddings")
    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_model_common_attributes
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="SiglipModel does not support training")
    def test_training(self):
        pass

    @unittest.skip(reason="SiglipModel does not support training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SiglipModel does not support training")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="SiglipModel does not support training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Siglip uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass

    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest._create_and_check_torchscript with CLIP->Siglip
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
                input_ids = inputs_dict["input_ids"]
                pixel_values = inputs_dict["pixel_values"]  # Siglip needs pixel_values
                traced_model = torch.jit.trace(model, (input_ids, pixel_values))
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

            non_persistent_buffers = {}
            for key in loaded_model_state_dict.keys():
                if key not in model_state_dict.keys():
                    non_persistent_buffers[key] = loaded_model_state_dict[key]

            loaded_model_state_dict = {
                key: value for key, value in loaded_model_state_dict.items() if key not in non_persistent_buffers
            }

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            model_buffers = list(model.buffers())
            for non_persistent_buffer in non_persistent_buffers.values():
                found_buffer = False
                for i, model_buffer in enumerate(model_buffers):
                    if torch.equal(non_persistent_buffer, model_buffer):
                        found_buffer = True
                        break

                self.assertTrue(found_buffer)
                model_buffers.pop(i)

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_load_vision_text_config with CLIP->Siglip
    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save SiglipConfig and check if we can load SiglipVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = SiglipVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save SiglipConfig and check if we can load SiglipTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = SiglipTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    # Copied from tests.models.clip.test_modeling_clip.CLIPModelTest.test_model_from_pretrained with CLIPModel->SiglipModel, CLIP->SIGLIP
    def test_model_from_pretrained(self):
        for model_name in SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = SiglipModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class SiglipForImageClassificationModelTester(SiglipModelTester):
    def __init__(self, parent):
        super().__init__(parent)
        self.batch_size = self.vision_model_tester.batch_size
        self.num_hidden_layers = self.vision_model_tester.num_hidden_layers
        self.hidden_size = self.vision_model_tester.hidden_size
        self.seq_length = self.vision_model_tester.seq_length

    def prepare_config_and_inputs(self):
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class SiglipForImageClassificationModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (SiglipForImageClassification,) if is_torch_available() else ()
    pipeline_model_mapping = {"image-classification": SiglipForImageClassification} if is_torch_available() else {}
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = SiglipForImageClassificationModelTester(self)

    @unittest.skip(reason="SiglipForImageClassification does not support inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="SiglipForImageClassification does not support inputs_embeds")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="SiglipForImageClassification does not support gradient checkpointing yet")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="SiglipForImageClassification does not support gradient checkpointing yet")
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(reason="SiglipForImageClassification does not support gradient checkpointing yet")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Siglip uses the same initialization scheme as the Flax original implementation")
    def test_initialization(self):
        pass


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


@require_vision
@require_torch
class SiglipModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "google/siglip-base-patch16-224"
        model = SiglipModel.from_pretrained(model_name).to(torch_device)
        processor = SiglipProcessor.from_pretrained(model_name)

        image = prepare_img()
        inputs = processor(
            text=["a photo of 2 cats", "a photo of 2 dogs"], images=image, padding="max_length", return_tensors="pt"
        ).to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

        # verify the logits
        self.assertEqual(
            logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )

        expected_logits = torch.tensor([[-0.7567, -10.3354]], device=torch_device)

        self.assertTrue(torch.allclose(outputs.logits_per_image, expected_logits, atol=1e-3))

        # verify the probs
        probs = torch.sigmoid(logits_per_image)  # these are the probabilities
        expected_probs = torch.tensor([[3.1937e-01, 3.2463e-05]], device=torch_device)
        self.assertTrue(torch.allclose(probs, expected_probs, atol=1e-3))
