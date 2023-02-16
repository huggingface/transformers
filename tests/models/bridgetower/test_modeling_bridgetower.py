# coding=utf-8
# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch BridgeTower model. """

import tempfile
import unittest

import numpy as np

from transformers import BridgeTowerConfig, is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import BridgeTowerForImageAndTextRetrieval, BridgeTowerForMaskedLM, BridgeTowerModel
    from transformers.models.bridgetower.modeling_bridgetower import BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST
    from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_10
else:
    is_torch_greater_or_equal_than_1_10 = False

if is_vision_available():
    from PIL import Image

    from transformers import BridgeTowerProcessor


class BridgeTowerModelTester:
    def __init__(
        self,
        parent,
        share_cross_modal_transformer_layers=True,
        drop_rate=0.1,
        head_hidden_scale=2,
        hidden_act="gelu",
        hidden_size=768,
        initializer_factor=1,
        is_encoder_decoder=False,
        layer_norm_eps=1e-05,
        share_link_tower_layers=False,
        link_tower_type="add",
        num_attention_heads=12,
        num_hidden_layers=6,
        tie_word_embeddings=False,
        init_layernorm_from_vision_encoder=False,
        output_hidden_states=False,
        text_config=None,
        vision_config=None,
        image_size=288,
    ):
        self.parent = parent
        self.share_cross_modal_transformer_layers = share_cross_modal_transformer_layers
        self.drop_rate = drop_rate
        self.head_hidden_scale = head_hidden_scale
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_factor = initializer_factor
        self.is_encoder_decoder = is_encoder_decoder
        self.layer_norm_eps = layer_norm_eps
        self.share_link_tower_layers = share_link_tower_layers
        self.link_tower_type = link_tower_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.tie_word_embeddings = tie_word_embeddings
        self.init_layernorm_from_vision_encoder = init_layernorm_from_vision_encoder
        self.vocab_size = 50265
        self.num_channels = 3
        self.seq_length = 4
        self.num_image_features = 325
        self.batch_size = 1
        self.image_size = image_size
        self.is_training = False
        self.expected_num_hidden_layers = 32
        self.output_hidden_states = output_hidden_states

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        pixel_mask = random_attention_mask([self.batch_size, self.image_size, self.image_size])
        config = self.get_config()
        return (config, input_ids, attention_mask, pixel_values, pixel_mask)

    def get_config(self):
        return BridgeTowerConfig(
            share_cross_modal_transformer_layers=self.share_cross_modal_transformer_layers,
            drop_rate=self.drop_rate,
            head_hidden_scale=self.head_hidden_scale,
            hidden_act=self.hidden_act,
            hidden_size=self.hidden_size,
            initializer_factor=self.initializer_factor,
            image_size=self.image_size,
            is_encoder_decoder=self.is_encoder_decoder,
            layer_norm_eps=self.layer_norm_eps,
            share_link_tower_layers=self.share_link_tower_layers,
            link_tower_type=self.link_tower_type,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            tie_word_embeddings=self.tie_word_embeddings,
            init_layernorm_from_vision_encoder=self.init_layernorm_from_vision_encoder,
            num_channels=self.num_channels,
            output_hidden_states=self.output_hidden_states,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_mask,
    ):
        model = BridgeTowerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        self.parent.assertEqual(result["text_features"].shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(
            result["image_features"].shape, (self.batch_size, self.num_image_features, self.hidden_size)
        )
        self.parent.assertEqual(result["pooler_output"].shape, (self.batch_size, 2 * self.hidden_size))

    def create_and_check_for_image_and_text_retrieval(
        self,
        config,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_mask,
    ):
        bridgetower_itm_output_last_dimension = 2

        model = BridgeTowerForImageAndTextRetrieval(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, bridgetower_itm_output_last_dimension))

    def create_and_check_for_masked_language_modeling(
        self,
        config,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_mask,
    ):
        model = BridgeTowerForMaskedLM(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, pixel_mask=pixel_mask)
        result = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, attention_mask, pixel_values, pixel_mask) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
        }
        return config, inputs_dict


@require_torch
@unittest.skipIf(not is_torch_greater_or_equal_than_1_10, "BridgeTower is only available in torch v1.10+")
class BridgeTowerModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (BridgeTowerModel, BridgeTowerForImageAndTextRetrieval, BridgeTowerForMaskedLM) if is_torch_available() else ()
    )

    is_training = False
    test_headmasking = False
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False
    has_attentions = False

    # function to extract meaningful tensor from output per different model_class
    def extract_output(self, outputs, model_class):
        return outputs["pooler_output"] if model_class == "BridgeTowerModel" else outputs["logits"]

    def setUp(self):
        self.model_tester = BridgeTowerModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BridgeTowerConfig, hidden_size=37, vocab_size=50265)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_and_text_retrieval(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_and_text_retrieval(*config_and_inputs)

    def test_for_masked_language_modeling(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_language_modeling(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = BridgeTowerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    def test_save_load_fast_init_from_base(self):
        # Override as it is a slow test on this model
        super().test_save_load_fast_init_from_base()

    # Override as extracting meaningful tensor from output is different for BridgeTower
    def test_save_load(self):
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**input_dict)

            out_2 = self.extract_output(outputs, model_class.__name__)
            out_2 = out_2.cpu().numpy()
            out_2[np.isnan(out_2)] = 0

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname)
                model.to(torch_device)
                with torch.no_grad():
                    after_outputs = model(**input_dict)

                # Make sure we don't have nans
                out_1 = self.extract_output(after_outputs, model_class.__name__)
                out_1 = out_1.cpu().numpy()
                out_1[np.isnan(out_1)] = 0
                max_diff = np.amax(np.abs(out_1 - out_2))
                self.assertLessEqual(max_diff, 1e-5)

    # Override this as `hidden states output` is different for BridgeTower
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states_text, hidden_states_vision, hidden_states_cross = (
                outputs.encoder_hidden_states if config.is_encoder_decoder else outputs.hidden_states
            )

            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )
            self.assertEqual(
                sum((len(hidden_states_text), len(hidden_states_vision), len(hidden_states_cross))),
                expected_num_layers,
            )

            seq_length = self.model_tester.seq_length
            num_image_features = self.model_tester.num_image_features

            self.assertListEqual(
                list(hidden_states_text[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )
            self.assertListEqual(
                list(hidden_states_vision[0].shape),
                [num_image_features, 1, self.model_tester.hidden_size],
            )
            self.assertListEqual(
                list(hidden_states_cross[0][0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )
            self.assertListEqual(
                list(hidden_states_cross[0][1].shape[-2:]),
                [num_image_features, self.model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(inputs_dict, config, model_class)

    # Override as `hidden states output` is different for BridgeTower
    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions

        # no need to test all models as different heads yield the same functionality
        model_class = self.all_model_classes[0]
        model = model_class(config)
        model.to(torch_device)

        inputs = self._prepare_for_class(inputs_dict, model_class)

        outputs = model(**inputs)

        output = outputs[0]

        # Encoder-/Decoder-only models
        hidden_states = outputs.hidden_states[0][0]
        hidden_states.retain_grad()

        if self.has_attentions:
            attentions = outputs.attentions[0][0]
            attentions.retain_grad()

        output.flatten()[0].backward(retain_graph=True)

        self.assertIsNotNone(hidden_states.grad)

        if self.has_attentions:
            self.assertIsNotNone(attentions.grad)

    @unittest.skip(reason="""Bridge Tower does not have input/output embeddings. So this test is not applicable.""")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="""Bridge Tower does not have input/output embeddings. Thus this test is not applicable.""")
    def test_inputs_embeds(self):
        pass


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
    return image


@require_torch
@require_vision
@unittest.skipIf(not is_torch_greater_or_equal_than_1_10, "BridgeTower is only available in torch v1.10+")
class BridgeTowerModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return (
            BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
            if is_vision_available()
            else None
        )

    @slow
    def test_image_and_text_retrieval(self):
        model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm").to(
            torch_device
        )
        model.eval()
        processor = self.default_processor
        image = prepare_img()
        text = "a bunch of cats laying on a tower."
        inputs = processor(image, text, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size([1, 2])
        self.assertEqual(outputs.logits.shape, expected_shape)
        self.assertTrue(outputs.logits[0, 1].item() > outputs.logits[0, 0].item())

    @slow
    def test_masked_language_modeling(self):
        model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm").to(torch_device)
        model.eval()
        processor = self.default_processor
        image = prepare_img()
        text = "a bunch of <mask> laying on a tower."
        inputs = processor(image, text, return_tensors="pt").to(torch_device)

        # forward pass
        with torch.no_grad():
            outputs = model(**inputs)

        # verify the logits
        expected_shape = torch.Size([1, 11, 50265])
        self.assertEqual(outputs.logits.shape, expected_shape)

        # verify predicted word
        predicted_id = outputs.logits.argmax(dim=-1).squeeze(0).tolist()[4]
        self.assertTrue(processor.decode([predicted_id]) == " cats")
