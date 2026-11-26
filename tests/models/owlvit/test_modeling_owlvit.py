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
"""Testing suite for the PyTorch OwlViT model."""

import inspect
import tempfile
import unittest

import numpy as np
from parameterized import parameterized

from transformers import OwlViTConfig, OwlViTTextConfig, OwlViTVisionConfig
from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    require_torch_fp16,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...model_tester_config import ModelTesterConfigMixin
from ...test_configuration_common import ConfigTester
from ...test_image_processing_common import load_test_image
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    _test_eager_matches_sdpa_inference,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers import OwlViTForObjectDetection, OwlViTModel, OwlViTTextModel, OwlViTVisionModel


if is_vision_available():
    from transformers import OwlViTProcessor


class OwlViTVisionModelTester(ModelTesterConfigMixin):
    defaults = {}

    small_config_overrides = {
        "batch_size": 12,
        "image_size": 32,
        "patch_size": 2,
        "num_channels": 3,
        "is_training": True,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 37,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "initializer_range": 0.02,
        "scope": None,
    }

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        values = super().get_small_config_defaults(**kwargs)
        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (values["image_size"] // values["patch_size"]) ** 2
        values["seq_length"] = num_patches + 1
        return values

    def __init__(self, parent, **kwargs):
        self.parent = parent
        for key, value in self.get_small_config_defaults(**kwargs).items():
            setattr(self, key, value)

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return OwlViTVisionConfig(
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
        model = OwlViTVisionModel(config=config).to(torch_device)
        model.eval()

        pixel_values = pixel_values.to(torch.float32)

        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        num_patches = (self.image_size // self.patch_size) ** 2
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class OwlViTVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as OWLVIT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    model_tester_class = OwlViTVisionModelTester

    all_model_classes = (OwlViTVisionModel,) if is_torch_available() else ()

    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = OwlViTVisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=OwlViTVisionConfig, has_text_modality=False, hidden_size=32
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="OWLVIT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
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

    @unittest.skip(reason="This module does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="This module does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="This module does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="This module does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/owlvit-base-patch32"
        model = OwlViTVisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class OwlViTTextModelTester(ModelTesterConfigMixin):
    defaults = {}

    small_config_overrides = {
        "batch_size": 12,
        "num_queries": 4,
        "seq_length": 16,
        "is_training": True,
        "use_input_mask": True,
        "use_labels": True,
        "vocab_size": 99,
        "hidden_size": 64,
        "num_hidden_layers": 12,
        "num_attention_heads": 4,
        "intermediate_size": 37,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "max_position_embeddings": 16,
        "initializer_range": 0.02,
        "scope": None,
    }

    def __init__(self, parent, **kwargs):
        self.parent = parent
        for key, value in self.get_small_config_defaults(**kwargs).items():
            setattr(self, key, value)

    def prepare_config_and_inputs(self):
        # The text model pools at `input_ids.argmax(-1)` (the EOS token has the highest id). Put a unique highest id at
        # the last position, so the pooled token never lands on the left-padded positions used in
        # `test_eager_matches_sdpa_inference`: such a token attends to nothing (causal mask) and eager/SDPA diverge.
        input_ids = ids_tensor([self.batch_size * self.num_queries, self.seq_length], self.vocab_size - 1)
        input_ids[:, -1] = self.vocab_size - 1
        input_mask = None

        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size * self.num_queries, self.seq_length])

        if input_mask is not None:
            num_text, seq_length = input_mask.shape

            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(num_text,))
            for idx, start_index in enumerate(rnd_start_indices):
                input_mask[idx, :start_index] = 1
                input_mask[idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return OwlViTTextConfig(
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
        model = OwlViTTextModel(config=config).to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=input_mask)

        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size * self.num_queries, self.seq_length, self.hidden_size)
        )
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size * self.num_queries, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class OwlViTTextModelTest(ModelTesterMixin, unittest.TestCase):
    model_tester_class = OwlViTTextModelTester

    all_model_classes = (OwlViTTextModel,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = OwlViTTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OwlViTTextConfig, hidden_size=32)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="This module does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="This module does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="This module does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="This module does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @unittest.skip(reason="OWLVIT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/owlvit-base-patch32"
        model = OwlViTTextModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class OwlViTModelTester(ModelTesterConfigMixin):
    defaults = {}

    small_config_overrides = {
        "is_training": True,
    }

    @classmethod
    def get_small_config_defaults(cls, text_kwargs=None, vision_kwargs=None, **kwargs):
        values = super().get_small_config_defaults(**kwargs)
        text_model_tester = OwlViTTextModelTester(None, **(text_kwargs or {}))
        vision_model_tester = OwlViTVisionModelTester(None, **(vision_kwargs or {}))
        values["text_config"] = text_model_tester.get_config().to_dict()
        values["vision_config"] = vision_model_tester.get_config().to_dict()
        values["batch_size"] = text_model_tester.batch_size  # need bs for batching_equivalence test
        return values

    def __init__(self, parent, text_kwargs=None, vision_kwargs=None, **kwargs):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = OwlViTTextModelTester(parent, **text_kwargs)
        self.vision_model_tester = OwlViTVisionModelTester(parent, **vision_kwargs)
        values = self.get_small_config_defaults(text_kwargs=text_kwargs, vision_kwargs=vision_kwargs, **kwargs)
        for key, value in values.items():
            setattr(self, key, value)

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return OwlViTConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            projection_dim=64,
        )

    def create_and_check_model(self, config, input_ids, attention_mask, pixel_values):
        model = OwlViTModel(config).to(torch_device).eval()

        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
            )

        image_logits_size = (
            self.vision_model_tester.batch_size,
            self.text_model_tester.batch_size * self.text_model_tester.num_queries,
        )
        text_logits_size = (
            self.text_model_tester.batch_size * self.text_model_tester.num_queries,
            self.vision_model_tester.batch_size,
        )
        self.parent.assertEqual(result.logits_per_image.shape, image_logits_size)
        self.parent.assertEqual(result.logits_per_text.shape, text_logits_size)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_loss": False,
        }
        return config, inputs_dict


@require_torch
class OwlViTModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    model_tester_class = OwlViTModelTester

    all_model_classes = (OwlViTModel,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": OwlViTModel,
            "zero-shot-object-detection": OwlViTForObjectDetection,
        }
        if is_torch_available()
        else {}
    )

    test_resize_embeddings = False
    test_attention_outputs = False
    additional_model_inputs = ["pixel_values"]
    _is_composite = True

    def setUp(self):
        self.model_tester = OwlViTModelTester(self)
        common_properties = ["projection_dim", "logit_scale_init_value"]
        self.config_tester = ConfigTester(
            self, config_class=OwlViTConfig, has_text_modality=False, common_properties=common_properties
        )

    def test_config(self):
        self.config_tester.run_common_tests()

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

    @unittest.skip(reason="OwlViTModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    def test_load_vision_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save OwlViTConfig and check if we can load OwlViTVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = OwlViTVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save OwlViTConfig and check if we can load OwlViTTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = OwlViTTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(
        self,
        name,
        dtype,
        padding_side,
        use_attention_mask,
        output_attentions,
        enable_kernels,
    ):
        # `logits_per_text` is scaled by `exp(logit_scale)` (~14), which amplifies the bf16 eager/sdpa difference
        atols = {
            ("cpu", False, torch.float32): 1e-6,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 5e-2,  # this was relaxed
            ("cpu", True, torch.float32): 1e-6,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 5e-2,  # this was relaxed
            ("cuda", False, torch.float32): 1e-6,
            ("cuda", False, torch.bfloat16): 5e-2,  # this was relaxed
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-6,
            ("cuda", True, torch.bfloat16): 5e-2,  # this was relaxed
            ("cuda", True, torch.float16): 5e-3,
        }
        _test_eager_matches_sdpa_inference(
            self, name, dtype, padding_side, use_attention_mask, output_attentions, enable_kernels, atols=atols
        )

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/owlvit-base-patch32"
        model = OwlViTModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class OwlViTForObjectDetectionTester(ModelTesterConfigMixin):
    defaults = {}

    small_config_overrides = {
        "is_training": True,
    }

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        values = super().get_small_config_defaults(**kwargs)
        text_model_tester = OwlViTTextModelTester(None)
        values["text_config"] = text_model_tester.get_config().to_dict()
        values["vision_config"] = OwlViTVisionModelTester(None).get_config().to_dict()
        values["batch_size"] = text_model_tester.batch_size  # need bs for batching_equivalence test
        return values

    def __init__(self, parent, **kwargs):
        self.parent = parent
        self.text_model_tester = OwlViTTextModelTester(parent)
        self.vision_model_tester = OwlViTVisionModelTester(parent)
        for key, value in self.get_small_config_defaults(**kwargs).items():
            setattr(self, key, value)

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, pixel_values, input_ids, attention_mask

    def get_config(self):
        return OwlViTConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            projection_dim=64,
        )

    def create_and_check_model(self, config, pixel_values, input_ids, attention_mask):
        model = OwlViTForObjectDetection(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        pred_boxes_size = (
            self.vision_model_tester.batch_size,
            (self.vision_model_tester.image_size // self.vision_model_tester.patch_size) ** 2,
            4,
        )
        pred_logits_size = (
            self.vision_model_tester.batch_size,
            (self.vision_model_tester.image_size // self.vision_model_tester.patch_size) ** 2,
            4,
        )
        pred_class_embeds_size = (
            self.vision_model_tester.batch_size,
            (self.vision_model_tester.image_size // self.vision_model_tester.patch_size) ** 2,
            self.text_model_tester.hidden_size,
        )
        self.parent.assertEqual(result.pred_boxes.shape, pred_boxes_size)
        self.parent.assertEqual(result.logits.shape, pred_logits_size)
        self.parent.assertEqual(result.class_embeds.shape, pred_class_embeds_size)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, input_ids, attention_mask = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class OwlViTForObjectDetectionTest(ModelTesterMixin, unittest.TestCase):
    model_tester_class = OwlViTForObjectDetectionTester

    all_model_classes = (OwlViTForObjectDetection,) if is_torch_available() else ()

    test_resize_embeddings = False
    test_attention_outputs = False

    additional_model_inputs = ["pixel_values", "attention_mask"]

    def setUp(self):
        self.model_tester = OwlViTForObjectDetectionTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_sdpa_inference(self, *args):
        self.skipTest("OwlViTObjectDetectionOutput has no top-level hidden_states; SDPA tested in sub-models")

    @unittest.skip(reason="Inputs_embeds is tested in individual model tests")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="OwlViTModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="Test_forward_signature is tested in individual model tests")
    def test_forward_signature(self):
        pass

    @unittest.skip(reason="This module does not support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="This module does not support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="This module does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="This module does not support standalone training")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "google/owlvit-base-patch32"
        model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
    im = load_test_image(url)
    return im


@require_vision
@require_torch
class OwlViTModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "google/owlvit-base-patch32"
        model = OwlViTModel.from_pretrained(model_name).to(torch_device)
        processor = OwlViTProcessor.from_pretrained(model_name)

        image = prepare_img()
        inputs = processor(
            text=[["a photo of a cat", "a photo of a dog"]],
            images=image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        self.assertEqual(
            outputs.logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )
        expected_logits = torch.tensor([[3.4612, 0.9404]], device=torch_device)
        torch.testing.assert_close(outputs.logits_per_image, expected_logits, rtol=1e-3, atol=1e-3)

    @slow
    def test_inference_interpolate_pos_encoding(self):
        model_name = "google/owlvit-base-patch32"
        model = OwlViTModel.from_pretrained(model_name).to(torch_device)
        processor = OwlViTProcessor.from_pretrained(model_name)
        processor.image_processor.size = {"height": 800, "width": 800}

        image = prepare_img()
        inputs = processor(
            text=[["a photo of a cat", "a photo of a dog"]],
            images=image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=True)

        # verify the logits
        self.assertEqual(
            outputs.logits_per_image.shape,
            torch.Size((inputs.pixel_values.shape[0], inputs.input_ids.shape[0])),
        )
        self.assertEqual(
            outputs.logits_per_text.shape,
            torch.Size((inputs.input_ids.shape[0], inputs.pixel_values.shape[0])),
        )
        expected_logits = torch.tensor([[3.6292, 0.8855]], device=torch_device)
        torch.testing.assert_close(outputs.logits_per_image, expected_logits, rtol=1e-3, atol=1e-3)

        expected_shape = torch.Size((1, 626, 768))
        self.assertEqual(outputs.vision_model_output.last_hidden_state.shape, expected_shape)

        # OwlViTForObjectDetection part.
        model = OwlViTForObjectDetection.from_pretrained(model_name).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=True)

        num_queries = int((inputs.pixel_values.shape[-1] // model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.pred_boxes.shape, torch.Size((1, num_queries, 4)))

        expected_slice_boxes = torch.tensor(
            [[0.0679, 0.0422, 0.1345], [0.2064, 0.0450, 0.4134], [0.1979, 0.0416, 0.3416]]
        ).to(torch_device)
        torch.testing.assert_close(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, rtol=1e-4, atol=1e-4)

        model = OwlViTForObjectDetection.from_pretrained(model_name).to(torch_device)
        query_image = prepare_img()
        inputs = processor(
            images=image,
            query_images=query_image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model.image_guided_detection(**inputs, interpolate_pos_encoding=True)

        # No need to check the logits, we just check inference runs fine.
        num_queries = int((inputs.pixel_values.shape[-1] / model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.target_pred_boxes.shape, torch.Size((1, num_queries, 4)))

        # Deactivate interpolate_pos_encoding on same model, and use default image size.
        # Verify the dynamic change caused by the activation/deactivation of interpolate_pos_encoding of variables: (self.sqrt_num_patch_h, self.sqrt_num_patch_w), self.box_bias from (OwlViTForObjectDetection).
        processor = OwlViTProcessor.from_pretrained(model_name)

        image = prepare_img()
        inputs = processor(
            text=[["a photo of a cat", "a photo of a dog"]],
            images=image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=False)

        num_queries = int((inputs.pixel_values.shape[-1] // model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.pred_boxes.shape, torch.Size((1, num_queries, 4)))

        expected_default_box_bias = torch.tensor(
            [
                [-3.1332, -3.1332, -3.1332, -3.1332],
                [-2.3968, -3.1332, -3.1332, -3.1332],
                [-1.9452, -3.1332, -3.1332, -3.1332],
            ]
        ).to(torch_device)
        torch.testing.assert_close(model.box_bias[:3, :4], expected_default_box_bias, rtol=1e-4, atol=1e-4)

        # Interpolate with any resolution size.
        processor.image_processor.size = {"height": 1264, "width": 1024}

        image = prepare_img()
        inputs = processor(
            text=[["a photo of a cat", "a photo of a dog"]],
            images=image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs, interpolate_pos_encoding=True)

        num_queries = int(
            (inputs.pixel_values.shape[-2] // model.config.vision_config.patch_size)
            * (inputs.pixel_values.shape[-1] // model.config.vision_config.patch_size)
        )
        self.assertEqual(outputs.pred_boxes.shape, torch.Size((1, num_queries, 4)))
        expected_slice_boxes = torch.tensor(
            [[0.0498, 0.0301, 0.0982], [0.2241, 0.0364, 0.4654], [0.1389, 0.0314, 0.1860]]
        ).to(torch_device)
        torch.testing.assert_close(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, rtol=1e-2, atol=1e-2)

        query_image = prepare_img()
        inputs = processor(
            images=image,
            query_images=query_image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model.image_guided_detection(**inputs, interpolate_pos_encoding=True)

        # No need to check the logits, we just check inference runs fine.
        num_queries = int(
            (inputs.pixel_values.shape[-2] // model.config.vision_config.patch_size)
            * (inputs.pixel_values.shape[-1] // model.config.vision_config.patch_size)
        )
        self.assertEqual(outputs.target_pred_boxes.shape, torch.Size((1, num_queries, 4)))

    @slow
    def test_inference_object_detection(self):
        model_name = "google/owlvit-base-patch32"
        model = OwlViTForObjectDetection.from_pretrained(model_name).to(torch_device)

        processor = OwlViTProcessor.from_pretrained(model_name)

        image = prepare_img()
        text_labels = [["a photo of a cat", "a photo of a dog"]]
        inputs = processor(
            text=text_labels,
            images=image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        num_queries = int((model.config.vision_config.image_size / model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.pred_boxes.shape, torch.Size((1, num_queries, 4)))

        expected_slice_boxes = torch.tensor(
            [[0.0691, 0.0445, 0.1374], [0.1592, 0.0456, 0.3191], [0.1632, 0.0423, 0.2477]]
        ).to(torch_device)
        torch.testing.assert_close(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, rtol=1e-4, atol=1e-4)

        # test post-processing
        post_processed_output = processor.post_process_grounded_object_detection(outputs)
        self.assertIsNone(post_processed_output[0]["text_labels"])

        post_processed_output_with_text_labels = processor.post_process_grounded_object_detection(
            outputs, text_labels=text_labels
        )

        objects_labels = post_processed_output_with_text_labels[0]["labels"].tolist()
        self.assertListEqual(objects_labels, [0, 0])

        objects_text_labels = post_processed_output_with_text_labels[0]["text_labels"]
        self.assertIsNotNone(objects_text_labels)
        self.assertListEqual(objects_text_labels, ["a photo of a cat", "a photo of a cat"])

    @slow
    def test_inference_one_shot_object_detection(self):
        model_name = "google/owlvit-base-patch32"
        model = OwlViTForObjectDetection.from_pretrained(model_name).to(torch_device)

        processor = OwlViTProcessor.from_pretrained(model_name)

        image = prepare_img()
        query_image = prepare_img()
        inputs = processor(
            images=image,
            query_images=query_image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model.image_guided_detection(**inputs)

        num_queries = int((model.config.vision_config.image_size / model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.target_pred_boxes.shape, torch.Size((1, num_queries, 4)))

        expected_slice_boxes = torch.tensor(
            [[0.0691, 0.0445, 0.1374], [0.1592, 0.0456, 0.3191], [0.1632, 0.0423, 0.2477]]
        ).to(torch_device)
        torch.testing.assert_close(outputs.target_pred_boxes[0, :3, :3], expected_slice_boxes, rtol=1e-4, atol=1e-4)

    @slow
    @require_torch_accelerator
    @require_torch_fp16
    def test_inference_one_shot_object_detection_fp16(self):
        model_name = "google/owlvit-base-patch32"
        model = OwlViTForObjectDetection.from_pretrained(model_name, dtype=torch.float16).to(torch_device)

        processor = OwlViTProcessor.from_pretrained(model_name)

        image = prepare_img()
        query_image = prepare_img()
        inputs = processor(
            images=image,
            query_images=query_image,
            max_length=16,
            padding="max_length",
            return_tensors="pt",
        ).to(torch_device)

        with torch.no_grad():
            outputs = model.image_guided_detection(**inputs)

        # No need to check the logits, we just check inference runs fine.
        num_queries = int((model.config.vision_config.image_size / model.config.vision_config.patch_size) ** 2)
        self.assertEqual(outputs.target_pred_boxes.shape, torch.Size((1, num_queries, 4)))
