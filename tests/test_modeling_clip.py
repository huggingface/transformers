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
""" Testing suite for the PyTorch CLIP model. """


import inspect
import os
import tempfile
import unittest

import numpy as np

import requests
import transformers
from transformers import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers.file_utils import is_torch_available, is_vision_available
from transformers.testing_utils import (
    is_flax_available,
    is_pt_flax_cross_test,
    require_torch,
    require_vision,
    slow,
    torch_device,
)

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch
    from torch import nn

    from transformers import CLIPModel, CLIPTextModel, CLIPVisionModel
    from transformers.models.clip.modeling_clip import CLIP_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import CLIPProcessor


if is_flax_available():
    import jax.numpy as jnp
    from transformers.modeling_flax_pytorch_utils import (
        convert_pytorch_state_dict_to_flax,
        load_flax_weights_in_pytorch_model,
    )


class CLIPVisionModelTester:
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
        model = CLIPVisionModel(config=config)
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

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class CLIPVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as CLIP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (CLIPVisionModel,) if is_torch_available() else ()

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = CLIPVisionModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLIPVisionConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_inputs_embeds(self):
        # CLIP does not use inputs_embeds
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
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

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
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

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

    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    # skip this test as CLIPVisionModel has no base class and is
    # not available in MODEL_MAPPING
    def test_save_load_fast_init_from_base(self):
        pass

    # skip this test as CLIPVisionModel has no base class and is
    # not available in MODEL_MAPPING
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLIPVisionModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class CLIPTextModelTester:
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

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

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
        model = CLIPTextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class CLIPTextModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (CLIPTextModel,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = CLIPTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLIPTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_training(self):
        pass

    def test_training_gradient_checkpointing(self):
        pass

    def test_inputs_embeds(self):
        # CLIP does not use inputs_embeds
        pass

    # skip this test as CLIPTextModel has no base class and is
    # not available in MODEL_MAPPING
    def test_save_load_fast_init_from_base(self):
        pass

    # skip this test as CLIPTextModel has no base class and is
    # not available in MODEL_MAPPING
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLIPTextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class CLIPModelTester:
    def __init__(self, parent, is_training=True):
        self.parent = parent
        self.text_model_tester = CLIPTextModelTester(parent)
        self.vision_model_tester = CLIPVisionModelTester(parent)
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
        model = CLIPModel(config).to(torch_device).eval()
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
            "return_loss": True,
        }
        return config, inputs_dict


@require_torch
class CLIPModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (CLIPModel,) if is_torch_available() else ()
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = CLIPModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    # hidden_states are tested in individual model tests
    def test_hidden_states_output(self):
        pass

    # input_embeds are tested in individual model tests
    def test_inputs_embeds(self):
        pass

    # tested in individual model tests
    def test_retain_grad_hidden_states_attentions(self):
        pass

    # CLIPModel does not have input/output embeddings
    def test_model_common_attributes(self):
        pass

    # override as the `logit_scale` parameter initilization is different for CLIP
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # check if `logit_scale` is initilized as per the original implementation
                    if name == "logit_scale":
                        self.assertAlmostEqual(
                            param.data.item(),
                            np.log(1 / 0.07),
                            delta=1e-3,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

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
                pixel_values = inputs_dict["pixel_values"]  # CLIP needs pixel_values
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

            self.assertEqual(set(model_state_dict.keys()), set(loaded_model_state_dict.keys()))

            models_equal = True
            for layer_name, p1 in model_state_dict.items():
                p2 = loaded_model_state_dict[layer_name]
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    # overwrite from common since FlaxCLIPModel returns nested output
    # which is not supported in the common test
    @is_pt_flax_cross_test
    def test_equivalence_pt_to_flax(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):

                # load PyTorch class
                pt_model = model_class(config).eval()
                # Flax models don't use the `use_cache` option and cache is not returned as a default.
                # So we disable `use_cache` here for PyTorch model.
                pt_model.config.use_cache = False

                fx_model_class_name = "Flax" + model_class.__name__

                if not hasattr(transformers, fx_model_class_name):
                    return

                fx_model_class = getattr(transformers, fx_model_class_name)

                # load Flax class
                fx_model = fx_model_class(config, dtype=jnp.float32)
                # make sure only flax inputs are forward that actually exist in function args
                fx_input_keys = inspect.signature(fx_model.__call__).parameters.keys()

                # prepare inputs
                pt_inputs = self._prepare_for_class(inputs_dict, model_class)

                # remove function args that don't exist in Flax
                pt_inputs = {k: v for k, v in pt_inputs.items() if k in fx_input_keys}

                fx_state = convert_pytorch_state_dict_to_flax(pt_model.state_dict(), fx_model)
                fx_model.params = fx_state

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs).to_tuple()

                # convert inputs to Flax
                fx_inputs = {k: np.array(v) for k, v in pt_inputs.items() if torch.is_tensor(v)}
                fx_outputs = fx_model(**fx_inputs).to_tuple()
                self.assertEqual(len(fx_outputs), len(pt_outputs), "Output lengths differ between Flax and PyTorch")
                for fx_output, pt_output in zip(fx_outputs[:4], pt_outputs[:4]):
                    self.assert_almost_equals(fx_output, pt_output.numpy(), 4e-2)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    pt_model.save_pretrained(tmpdirname)
                    fx_model_loaded = fx_model_class.from_pretrained(tmpdirname, from_pt=True)

                fx_outputs_loaded = fx_model_loaded(**fx_inputs).to_tuple()
                self.assertEqual(
                    len(fx_outputs_loaded), len(pt_outputs), "Output lengths differ between Flax and PyTorch"
                )
                for fx_output_loaded, pt_output in zip(fx_outputs_loaded[:4], pt_outputs[:4]):
                    self.assert_almost_equals(fx_output_loaded, pt_output.numpy(), 4e-2)

    # overwrite from common since FlaxCLIPModel returns nested output
    # which is not supported in the common test
    @is_pt_flax_cross_test
    def test_equivalence_flax_to_pt(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                # load corresponding PyTorch class
                pt_model = model_class(config).eval()

                # So we disable `use_cache` here for PyTorch model.
                pt_model.config.use_cache = False

                fx_model_class_name = "Flax" + model_class.__name__

                if not hasattr(transformers, fx_model_class_name):
                    # no flax model exists for this class
                    return

                fx_model_class = getattr(transformers, fx_model_class_name)

                # load Flax class
                fx_model = fx_model_class(config, dtype=jnp.float32)
                # make sure only flax inputs are forward that actually exist in function args
                fx_input_keys = inspect.signature(fx_model.__call__).parameters.keys()

                pt_model = load_flax_weights_in_pytorch_model(pt_model, fx_model.params)

                # make sure weights are tied in PyTorch
                pt_model.tie_weights()

                # prepare inputs
                pt_inputs = self._prepare_for_class(inputs_dict, model_class)

                # remove function args that don't exist in Flax
                pt_inputs = {k: v for k, v in pt_inputs.items() if k in fx_input_keys}

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs).to_tuple()

                fx_inputs = {k: np.array(v) for k, v in pt_inputs.items() if torch.is_tensor(v)}

                fx_outputs = fx_model(**fx_inputs).to_tuple()
                self.assertEqual(len(fx_outputs), len(pt_outputs), "Output lengths differ between Flax and PyTorch")

                for fx_output, pt_output in zip(fx_outputs[:4], pt_outputs[:4]):
                    self.assert_almost_equals(fx_output, pt_output.numpy(), 4e-2)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    fx_model.save_pretrained(tmpdirname)
                    pt_model_loaded = model_class.from_pretrained(tmpdirname, from_flax=True)

                with torch.no_grad():
                    pt_outputs_loaded = pt_model_loaded(**pt_inputs).to_tuple()

                self.assertEqual(
                    len(fx_outputs), len(pt_outputs_loaded), "Output lengths differ between Flax and PyTorch"
                )
                for fx_output, pt_output in zip(fx_outputs[:4], pt_outputs_loaded[:4]):
                    self.assert_almost_equals(fx_output, pt_output.numpy(), 4e-2)

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLIPModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@require_vision
class CLIPModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference(self):
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name).to(torch_device)
        processor = CLIPProcessor.from_pretrained(model_name)

        image = prepare_img()
        inputs = processor(
            text=["a photo of a cat", "a photo of a dog"], images=image, padding=True, return_tensors="pt"
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

        expected_logits = torch.tensor([[24.5701, 19.3049]], device=torch_device)

        self.assertTrue(torch.allclose(outputs.logits_per_image, expected_logits, atol=1e-3))
