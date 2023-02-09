# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch CLAP model. """


import inspect
import os
import tempfile
import unittest

import numpy as np
import requests

import transformers
from transformers import CLAPAudioConfig, CLAPConfig, CLAPTextConfig
from transformers.testing_utils import (
    is_flax_available,
    is_pt_flax_cross_test,
    require_torch,
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


if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        CLAPAudioModel,
        CLAPAudioModelWithProjection,
        CLAPModel,
        CLAPProcessor,
        CLAPTextModel,
        CLAPTextModelWithProjection,
    )
    from transformers.models.clap.modeling_clap import CLAP_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image


if is_flax_available():
    import jax.numpy as jnp

    from transformers.modeling_flax_pytorch_utils import (
        convert_pytorch_state_dict_to_flax,
        load_flax_weights_in_pytorch_model,
    )


class CLAPAudioModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=60,
        mel_bins=16,
        window_size=4,
        spec_size=64,
        patch_size=2,
        patch_stride=2,
        seq_length=16,
        freq_ratio=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        patch_embeds_hidden_size=32,
        projection_hidden_size=256,
        projection_dim=32,
        num_hidden_layers=4,
        num_heads=[2, 2, 2, 2],
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.mel_bins = mel_bins
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_attention_heads = num_heads[0]
        self.projection_hidden_size = projection_hidden_size
        self.seq_length = seq_length
        self.spec_size = spec_size
        self.freq_ratio = freq_ratio
        self.patch_stride = patch_stride
        self.patch_embeds_hidden_size = patch_embeds_hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_features = floats_tensor([self.batch_size, 1, self.hidden_size, self.mel_bins])
        config = self.get_config()

        return config, input_features

    def get_config(self):
        return CLAPAudioConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            mel_bins=self.mel_bins,
            window_size=self.window_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            patch_stride=self.patch_stride,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
            spec_size=self.spec_size,
            freq_ratio=self.freq_ratio,
            patch_embeds_hidden_size=self.patch_embeds_hidden_size,
            projection_hidden_size=self.projection_hidden_size,
        )

    def create_and_check_model(self, config, input_features):
        model = CLAPAudioModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        embedding_shape = self.hidden_size * self.window_size * self.freq_ratio
        self.parent.assertEqual(
            result.fine_grained_embedding.shape, (self.batch_size, embedding_shape, embedding_shape)
        )

    def create_and_check_model_with_projection(self, config, input_features):
        model = CLAPAudioModelWithProjection(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features)
        self.parent.assertEqual(result.audio_embeds.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_features = config_and_inputs
        inputs_dict = {"input_features": input_features}
        return config, inputs_dict


@require_torch
class CLAPAudioModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as CLAP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (CLAPAudioModel, CLAPAudioModelWithProjection) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = CLAPAudioModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLAPAudioConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="CLAP does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

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

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [self.model_tester.hidden_size, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason="CLAPAudio does not output any loss term in the forward pass")
    def test_retain_grad_hidden_states_attentions(self):
        pass

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

    @unittest.skip(reason="CLAPAudioModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="CLAPAudioModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLAP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLAPAudioModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    def test_model_with_projection_from_pretrained(self):
        for model_name in CLAP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLAPAudioModelWithProjection.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, "visual_projection"))


class CLAPTextModelTester:
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
        projection_dim=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        scope=None,
        projection_hidden_act="relu",
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
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope
        self.projection_hidden_act = projection_hidden_act

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
        return CLAPTextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            projection_hidden_act=self.projection_hidden_act,
        )

    def create_and_check_model(self, config, input_ids, input_mask):
        model = CLAPTextModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids, attention_mask=input_mask)
            result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_model_with_projection(self, config, input_ids, input_mask):
        model = CLAPTextModelWithProjection(config=config)
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
class CLAPTextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (CLAPTextModel, CLAPTextModelWithProjection) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = CLAPTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CLAPTextConfig, hidden_size=37)

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

    @unittest.skip(reason="CLAP does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="CLAPTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="CLAPTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in CLAP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLAPTextModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    def test_model_with_projection_from_pretrained(self):
        for model_name in CLAP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLAPTextModelWithProjection.from_pretrained(model_name)
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(model, "text_projection"))


class CLAPModelTester:
    def __init__(self, parent, text_kwargs=None, audio_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if audio_kwargs is None:
            audio_kwargs = {}

        self.parent = parent
        self.text_model_tester = CLAPTextModelTester(parent, **text_kwargs)
        self.audio_model_tester = CLAPAudioModelTester(parent, **audio_kwargs)
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        audio_config, input_features = self.audio_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, input_features

    def get_config(self):
        return CLAPConfig.from_text_audio_configs(
            self.text_model_tester.get_config(), self.audio_model_tester.get_config(), projection_dim=64
        )

    def create_and_check_model(self, config, input_ids, attention_mask, input_features):
        model = CLAPModel(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids, input_features, attention_mask)
        self.parent.assertEqual(
            result.logits_per_audio.shape, (self.audio_model_tester.batch_size, self.text_model_tester.batch_size)
        )
        self.parent.assertEqual(
            result.logits_per_text.shape, (self.text_model_tester.batch_size, self.audio_model_tester.batch_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, input_features = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "return_loss": True,
        }
        return config, inputs_dict


@require_torch
class CLAPModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (CLAPModel,) if is_torch_available() else ()
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = CLAPModelTester(self)

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

    @unittest.skip(reason="CLAPModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    # override as the `logit_scale` parameter initilization is different for CLAP
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
                input_features = inputs_dict["input_features"]  # CLAP needs input_features
                traced_model = torch.jit.trace(model, (input_ids, input_features))
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

    def test_load_audio_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save CLAPConfig and check if we can load CLAPAudioConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            audio_config = CLAPAudioConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.audio_config.to_dict(), audio_config.to_dict())

        # Save CLAPConfig and check if we can load CLAPTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            text_config = CLAPTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    # overwrite from common since FlaxCLAPModel returns nested output
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

    # overwrite from common since FlaxCLAPModel returns nested output
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
        for model_name in CLAP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CLAPModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@slow
@require_torch
class CLAPModelIntegrationTest(unittest.TestCase):
    paddings = ["repeatpad", "repeat", "pad"]

    def test_integration_unfused(self):
        EXPECTED_MEANS_UNFUSED = {
            "repeatpad": 0.0024,
            "pad": 0.0020,
            "repeat": 0.0023,
        }

        from datasets import load_dataset

        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        audio_sample = librispeech_dummy[-1]

        model_id = "ybelkada/clap-htsat-unfused"

        model = CLAPModel.from_pretrained(model_id).to(torch_device)
        processor = CLAPProcessor.from_pretrained(model_id)

        for padding in self.paddings:
            inputs = processor(audios=audio_sample["audio"]["array"], return_tensors="pt", padding=padding).to(
                torch_device
            )

            audio_embed = model.get_audio_features(**inputs)
            expected_mean = EXPECTED_MEANS_UNFUSED[padding]

            self.assertTrue(
                torch.allclose(audio_embed.cpu().mean(), torch.tensor([expected_mean]), atol=1e-3, rtol=1e-3)
            )

    def test_integration_fused(self):
        EXPECTED_MEANS_FUSED = {
            "repeatpad": 0.00069,
            "repeat": 0.00196,
            "pad": -0.000379,
        }

        from datasets import load_dataset

        librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        audio_sample = librispeech_dummy[-1]

        model_id = "ybelkada/clap-htsat-fused"

        model = CLAPModel.from_pretrained(model_id).to(torch_device)
        processor = CLAPProcessor.from_pretrained(model_id)

        for padding in self.paddings:
            inputs = processor(
                audios=audio_sample["audio"]["array"], return_tensors="pt", padding=padding, truncation="fusion"
            ).to(torch_device)
            inputs["is_longer"] = torch.tensor([False])

            audio_embed = model.get_audio_features(**inputs)
            expected_mean = EXPECTED_MEANS_FUSED[padding]

            self.assertTrue(
                torch.allclose(audio_embed.cpu().mean(), torch.tensor([expected_mean]), atol=1e-3, rtol=1e-3)
            )
