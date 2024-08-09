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
"""Testing suite for the PyTorch MSCLAP model."""

import inspect
import os
import tempfile
import unittest

from datasets import load_dataset

from transformers import MSClapAudioConfig, MSClapConfig, MSClapTextConfig
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.utils import is_torch_available

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
    import numpy as np
    import torch
    from torch import nn

    from transformers import (
        MSClapAudioModel,
        MSClapAudioModelWithProjection,
        MSClapModel,
        MSClapProcessor,
        MSClapTextModelWithProjection,
    )


# Copied from tests.models.clap.test_modeling_clap.ClapAudioModelTester with Clap->MSClap
class MSClapAudioModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=60,
        num_mel_bins=16,
        window_size=4,
        spec_size=64,
        patch_size=2,
        patch_stride=2,
        seq_length=16,
        freq_ratio=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        patch_embeds_hidden_size=16,
        projection_dim=32,
        depths=[2, 2],
        num_hidden_layers=2,
        num_heads=[2, 2],
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_mel_bins = num_mel_bins
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.depths = depths
        self.num_heads = num_heads
        self.num_attention_heads = num_heads[0]
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
        input_features = floats_tensor([self.batch_size, 1, self.hidden_size, self.num_mel_bins])
        config = self.get_config()

        return config, input_features

    def get_config(self):
        return MSClapAudioConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_mel_bins=self.num_mel_bins,
            window_size=self.window_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            patch_stride=self.patch_stride,
            projection_dim=self.projection_dim,
            depths=self.depths,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
            spec_size=self.spec_size,
            freq_ratio=self.freq_ratio,
            patch_embeds_hidden_size=self.patch_embeds_hidden_size,
        )

    def create_and_check_model(self, config, input_features):
        model = MSClapAudioModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features)
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_model_with_projection(self, config, input_features):
        model = MSClapAudioModelWithProjection(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_features)
        self.parent.assertEqual(result.audio_embeds.shape, (self.batch_size, self.projection_dim))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_features = config_and_inputs
        inputs_dict = {"input_features": input_features}
        return config, inputs_dict


@require_torch
# Copied from tests.models.clap.test_modeling_clap.ClapAudioModelTest with Clap->MSClap, laion/clap-htsat-fused->kamilakesbi/ms_clap, CLAP->MSCLAP
class MSClapAudioModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as MSCLAP does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (MSClapAudioModel, MSClapAudioModelWithProjection) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = MSClapAudioModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=MSClapAudioConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="MSClapAudioModel does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_get_set_embeddings(self):
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

            hidden_states = outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [2 * self.model_tester.patch_embeds_hidden_size, 2 * self.model_tester.patch_embeds_hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @unittest.skip(reason="MSClapAudioModel does not output any loss term in the forward pass")
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

    @unittest.skip(reason="MSClapAudioModel does not output any loss term in the forward pass")
    def test_training(self):
        pass

    @unittest.skip(reason="MSClapAudioModel does not output any loss term in the forward pass")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="MSClapAudioModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="MSClapAudioModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "kamilakesbi/ms_clap"
        model = MSClapAudioModel.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @slow
    def test_model_with_projection_from_pretrained(self):
        model_name = "kamilakesbi/ms_clap"
        model = MSClapAudioModelWithProjection.from_pretrained(model_name)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "audio_projection"))


class MSClapTextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        hidden_size=24,
        projection_dropout_prob=0,
        initializer_factor=1.0,
        projection_dim=24,
        text_encoder={
            "model_type": "gpt2",
            "n_embd": 24,
            "n_layer": 2,
            "n_head": 2,
            "vocab_size": 99,
            "initializer_range": 0.02,
        },
        use_input_mask=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.projection_dropout_prob = projection_dropout_prob
        self.initializer_factor = initializer_factor
        self.text_encoder = text_encoder

        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = text_encoder["n_layer"]

        self.use_input_mask = use_input_mask
        self.vocab_size = text_encoder["vocab_size"]
        self.num_attention_heads = text_encoder["n_head"]

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
        return MSClapTextConfig(
            projection_dim=self.projection_dim,
            hidden_size=self.hidden_size,
            projection_dropout_prob=self.projection_dropout_prob,
            initializer_factor=self.initializer_factor,
            text_encoder=self.text_encoder,
            num_hidden_layers=self.text_encoder["n_layer"],
        )

    def create_and_check_model_with_projection(self, config, input_ids, input_mask):
        model = MSClapTextModelWithProjection(config=config)
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
class MSClapTextModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (MSClapTextModelWithProjection,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = MSClapTextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MSClapTextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_with_projection(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_with_projection(*config_and_inputs)

    @unittest.skip(reason="MSClapTextModel does not output any loss term in the forward pass")
    def test_training(self):
        pass

    @unittest.skip(reason="MSClapTextModel does not output any loss term in the forward pass")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="MSClapTextModel does not use embeddings")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="MSClapTextModel does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="MSClapTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="MSClapTextModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @unittest.skip(reason="MSClapTextModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    @slow
    def test_model_with_projection_from_pretrained(self):
        model_name = "kamilakesbi/ms_clap"
        model = MSClapTextModelWithProjection.from_pretrained(model_name)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "text_projection"))


# Copied from tests.models.clap.test_modeling_clap.ClapModelTester with Clap->MSClap
class MSClapModelTester:
    def __init__(self, parent, text_kwargs=None, audio_kwargs=None, is_training=True):
        if text_kwargs is None:
            text_kwargs = {}
        if audio_kwargs is None:
            audio_kwargs = {}

        self.parent = parent
        self.text_model_tester = MSClapTextModelTester(parent, **text_kwargs)
        self.audio_model_tester = MSClapAudioModelTester(parent, **audio_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        _, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        _, input_features = self.audio_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, input_features

    def get_config(self):
        return MSClapConfig.from_text_audio_configs(
            self.text_model_tester.get_config(), self.audio_model_tester.get_config(), projection_dim=64
        )

    def create_and_check_model(self, config, input_ids, attention_mask, input_features):
        model = MSClapModel(config).to(torch_device).eval()
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
# Copied from tests.models.clap.test_modeling_clap.ClapModelTest with Clap->MSClap, laion/clap-htsat-fused->kamilakesbi/ms_clap, CLAP->MSCLAP
class MSClapModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MSClapModel,) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": MSClapModel} if is_torch_available() else {}
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False

    def setUp(self):
        self.model_tester = MSClapModelTester(self)

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

    @unittest.skip(reason="MSClapModel does not have input/output embeddings")
    def test_model_get_set_embeddings(self):
        pass

    # override as the `logit_scale` parameter initilization is different for MSCLAP
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
            self.skipTest(reason="test_torchscript is set to False")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.torchscript = True
        configs_no_init.return_dict = False
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()

            try:
                input_ids = inputs_dict["input_ids"]
                input_features = inputs_dict["input_features"]  # MSCLAP needs input_features
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

    # Ignore copy
    def test_load_audio_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save MSClapConfig and check if we can load MSClapAudioConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.audio_config.save_pretrained(tmp_dir_name)
            audio_config = MSClapAudioConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.audio_config.to_dict(), audio_config.to_dict())

        # Save MSClapConfig and check if we can load MSClapTextConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.text_config.save_pretrained(tmp_dir_name)
            text_config = MSClapTextConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.text_config.to_dict(), text_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        model_name = "kamilakesbi/ms_clap"
        model = MSClapModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


@slow
@require_torch
class MSClapModelIntegrationTest(unittest.TestCase):
    def test_integration(self):
        EXPECTED_MEANS = 6.15e-05
        EXPECTED_SLICE = [-0.0223, -0.0219, -0.0563, -0.0188, 0.0186, 0.0180, 0.0725, 0.0535, 0.0004, -0.0528]

        librispeech_dummy = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True
        )
        audio_sample = librispeech_dummy[-1]

        model_id = "kamilakesbi/ms_clap"

        model = MSClapModel.from_pretrained(model_id).to(torch_device)
        processor = MSClapProcessor.from_pretrained(model_id)

        inputs = processor(audios=audio_sample["audio"]["array"], return_tensors="pt").to(torch_device)

        audio_embed = model.get_audio_features(**inputs)

        self.assertTrue(torch.allclose(audio_embed.cpu().mean(), torch.tensor([EXPECTED_MEANS]), atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(audio_embed[0, :10].cpu(), torch.tensor(EXPECTED_SLICE), atol=1e-4, rtol=1e-7))

    def test_batched(self):
        EXPECTED_MEANS = 6.62e-05
        EXPECTED_SLICE = [[-0.0107, -0.0122], [-0.0068, -0.0298], [-0.0076, -0.0184], [-0.0108, -0.0248]]

        librispeech_dummy = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True
        )
        audio_samples = [sample["array"] for sample in librispeech_dummy[0:4]["audio"]]

        model_id = "kamilakesbi/ms_clap"

        model = MSClapModel.from_pretrained(model_id).to(torch_device)
        processor = MSClapProcessor.from_pretrained(model_id)

        inputs = processor(audios=audio_samples, return_tensors="pt").to(torch_device)

        audio_embed = model.get_audio_features(**inputs)

        self.assertTrue(torch.allclose(audio_embed.cpu().mean(), torch.tensor([EXPECTED_MEANS]), atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(audio_embed[:, 0:2].cpu(), torch.tensor(EXPECTED_SLICE), atol=1e-4, rtol=1e-7))
