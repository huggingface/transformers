# coding=utf-8
# Copyright 2023 The Intel Team Authors, The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch TVP model. """


import inspect
import unittest

from transformers import ResNetConfig, TvpConfig
from transformers.testing_utils import require_torch, require_vision, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

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

    from transformers import TvpForVideoGrounding, TvpModel

if is_vision_available():
    from PIL import Image

    from transformers import TvpImageProcessor


# Copied from test.models.videomae.test_modeling_videomae.VideoMAEModelTester with VideoMAE->TVP
class TVPModelTester:
    def __init__(
        self,
        parent,
        batch_size=1,
        seq_length=2,
        alpha=1.0,
        beta=0.1,
        visual_prompter_type="framepad",
        visual_prompter_apply="replace",
        num_frames=2,
        max_img_size=448,
        visual_prompt_size=96,
        vocab_size=100,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=30,
        max_grid_col_position_embeddings=30,
        max_grid_row_position_embeddings=30,
        hidden_dropout_prob=0.1,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        initializer_range=0.02,
        pad_token_id=0,
        type_vocab_size=2,
        attention_probs_dropout_prob=0.1,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.alpha = alpha
        self.beta = beta
        self.visual_prompter_type = visual_prompter_type
        self.visual_prompter_apply = visual_prompter_apply
        self.num_frames = num_frames
        self.max_img_size = max_img_size
        self.visual_prompt_size = visual_prompt_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_grid_col_position_embeddings = max_grid_col_position_embeddings
        self.max_grid_row_position_embeddings = max_grid_row_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.type_vocab_size = type_vocab_size
        self.is_training = False
        self.num_channels = 3

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        pixel_values = floats_tensor(
            [self.batch_size, self.num_frames, self.num_channels, self.max_img_size, self.max_img_size]
        )

        config = self.get_config()

        return (config, input_ids, pixel_values, attention_mask)

    def get_config(self):
        resnet_config = ResNetConfig(
            num_channels=3,
            embeddings_size=64,
            hidden_sizes=[64, 128],
            depths=[2, 2],
            hidden_act="relu",
            out_features=["stage2"],
            out_indices=[2],
        )
        return TvpConfig(
            backbone_config=resnet_config,
            alpha=self.alpha,
            beta=self.beta,
            visual_prompter_type=self.visual_prompter_type,
            visual_prompter_apply=self.visual_prompter_apply,
            num_frames=self.num_frames,
            max_img_size=self.max_img_size,
            visual_prompt_size=self.visual_prompt_size,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            max_grid_col_position_embeddings=self.max_grid_col_position_embeddings,
            max_grid_row_position_embeddings=self.max_grid_row_position_embeddings,
            layer_norm_eps=self.layer_norm_eps,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            type_vocab_size=self.type_vocab_size,
        )

    def create_and_check_model(self, config, input_ids, pixel_values, attention_mask):
        model = TvpModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, pixel_values, attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size, self.seq_length + 10 + 784, self.hidden_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, pixel_values, attention_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "pixel_values": pixel_values, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class TVPModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as TVP does not use, inputs_embeds.
    The seq_length in TVP contain textual and visual inputs, and prompt.
    """

    all_model_classes = (TvpModel, TvpForVideoGrounding) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": TvpModel, "temporal-video-grounding": TvpForVideoGrounding}
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = TVPModelTester(self)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(reason="TVP does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="TVPModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_ids"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    # override as the `logit_scale` parameter initilization is different for TVP
    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # cnn params and transformer params and head params are randomly initialized.
                    if "vision_model" in name or "visual_prompter" in name or "transformer" in name or "head" in name:
                        self.assertAlmostEqual(
                            param.data.mean().item(),
                            0.0,
                            delta=1.0,
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )
                    else:
                        self.assertIn(
                            ((param.data.mean() * 1e9).round() / 1e9).item(),
                            [0.0, 1.0],
                            msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                        )

    def test_attention_outputs(self):
        if not self.has_attentions:
            self.skipTest(reason="Model does not output attentions")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        seq_len = getattr(self.model_tester, "seq_length", None)
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)
        chunk_length = getattr(self.model_tester, "chunk_length", None)
        if chunk_length is not None and hasattr(self.model_tester, "num_hashes"):
            encoder_seq_length = encoder_seq_length * self.model_tester.num_hashes

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
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)

            if chunk_length is not None:
                self.assertListEqual(
                    list(attentions[0].shape[-4:]),
                    [
                        self.model_tester.num_attention_heads,
                        encoder_seq_length + 10 + 784,
                        chunk_length,
                        encoder_key_length + 10 + 784,
                    ],
                )
            else:
                self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        encoder_seq_length + 10 + 784,
                        encoder_key_length + 10 + 784,
                    ],
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

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            elif self.is_encoder_decoder:
                added_hidden_states = 2
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions

            self.assertEqual(len(self_attentions), self.model_tester.num_hidden_layers)
            if chunk_length is not None:
                self.assertListEqual(
                    list(self_attentions[0].shape[-4:]),
                    [
                        self.model_tester.num_attention_heads,
                        encoder_seq_length + 10 + 784,
                        chunk_length,
                        encoder_key_length + 10 + 784,
                    ],
                )
            else:
                self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [
                        self.model_tester.num_attention_heads,
                        encoder_seq_length + 10 + 784,
                        encoder_key_length + 10 + 784,
                    ],
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

            if hasattr(self.model_tester, "encoder_seq_length"):
                seq_length = self.model_tester.encoder_seq_length
                if hasattr(self.model_tester, "chunk_length") and self.model_tester.chunk_length > 1:
                    seq_length = seq_length * self.model_tester.chunk_length
            else:
                seq_length = self.model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length + 10 + 784, self.model_tester.hidden_size],
            )

            if config.is_encoder_decoder:
                hidden_states = outputs.decoder_hidden_states

                self.assertIsInstance(hidden_states, (list, tuple))
                self.assertEqual(len(hidden_states), expected_num_layers)
                seq_len = getattr(self.model_tester, "seq_length", None)
                decoder_seq_length = getattr(self.model_tester, "decoder_seq_length", seq_len)

                self.assertListEqual(
                    list(hidden_states[0].shape[-2:]),
                    [decoder_seq_length + 10 + 784, self.model_tester.hidden_size],
                )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_vision
@require_torch
class TvpModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_image_processor(self):
        return (
            TvpImageProcessor.from_pretrained(
                "Jiqing/tiny-random-tvp",
            )
            if is_vision_available()
            else None
        )

    def test_inference_no_head(self):
        model = TvpModel.from_pretrained("Jiqing/tiny-random-tvp").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        encoding = image_processor(images=image, return_tensors="pt").to(torch_device)
        input_ids = torch.tensor([[1, 2]])
        attention_mask = torch.tensor([[1, 1]])
        encoding.update({"input_ids": input_ids, "attention_mask": attention_mask})

        with torch.no_grad():
            outputs = model(**encoding)

        expected_shape = torch.Size((1, 796, 128))
        assert outputs.last_hidden_state.shape == expected_shape
        expected_slice = torch.tensor(
            [[-0.4715, -0.4075, -1.7910], [-0.1982, 2.1252, -0.9496], [0.1366, 0.5033, -0.1846]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4))

    def test_inference_with_head(self):
        model = TvpForVideoGrounding.from_pretrained("Jiqing/tiny-random-tvp").to(torch_device)

        image_processor = self.default_image_processor
        image = prepare_img()
        encoding = image_processor(images=image, return_tensors="pt").to(torch_device)
        input_ids = torch.tensor([[1, 2]])
        attention_mask = torch.tensor([[1, 1]])
        encoding.update({"input_ids": input_ids, "attention_mask": attention_mask})

        with torch.no_grad():
            outputs = model(**encoding)

        expected_shape = torch.Size((1, 2))
        assert outputs.logits.shape == expected_shape
        expected_slice = torch.tensor([[0.5060, 0.4988]]).to(torch_device)
        self.assertTrue(torch.allclose(outputs.logits, expected_slice, atol=1e-4))
