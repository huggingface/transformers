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
""" Testing suite for the PyTorch Beit3 model. """

import inspect
import unittest

import numpy as np

from transformers import Beit3Config, Beit3Processor
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.testing_utils import require_torch, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Beit3ForCaptioning,
        Beit3ForImageClassification,
        Beit3ForImageTextRetrieval,
        Beit3ForQuestionAnswering,
        Beit3ForVisualReasoning,
        Beit3Model,
    )

if is_vision_available():
    from PIL import Image


class Beit3ModelTester:
    def __init__(
        self,
        hidden_size=37,
        attention_heads=1,
        intermediate_size=2,
        num_hidden_layers=1,
        normalize_before=True,
        activation_fn="gelu",
        dropout=0.0,
        drop_path_rate=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        deepnorm=False,
        subln=True,
        bert_init=False,
        multiway=True,
        max_source_positions=16,
        layernorm_eps=1e-5,
        vocab_size=50,
        image_size=16,
        patch_size=2,
        num_channels=3,
        num_labels=2,
        batch_size=1,
        seq_length=7,
        use_labels=True,
        is_training=True,
    ):
        self.hidden_size = hidden_size
        self.attention_heads = attention_heads
        self.attention_heads = attention_heads
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.normalize_before = normalize_before
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.deepnorm = deepnorm
        self.subln = subln
        self.bert_init = bert_init
        self.multiway = multiway
        self.max_source_positions = max_source_positions
        self.layernorm_eps = layernorm_eps
        # Text
        self.vocab_size = vocab_size
        # Vision
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.num_labels = num_labels
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_labels = use_labels
        self.is_training = is_training

    def get_config(self):
        return Beit3Config(
            hidden_size=self.hidden_size,
            num_attention_heads=self.attention_heads,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            normalize_before=self.normalize_before,
            activation_fn=self.activation_fn,
            dropout=self.dropout,
            drop_path_rate=self.drop_path_rate,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            deepnorm=self.deepnorm,
            sub_layernorm=self.subln,
            bert_init=self.bert_init,
            multiway=self.multiway,
            max_source_positions=self.max_source_positions,
            layer_norm_eps=self.layernorm_eps,
            vocab_size=self.vocab_size,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            num_labels=self.num_labels,
        )

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        attention_mask = torch.zeros_like(input_ids)
        return self.get_config(), {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
        }

    def prepare_config_and_inputs_for_visual_reasoning(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = floats_tensor([self.batch_size, 2, self.num_channels, self.image_size, self.image_size])
        text_padding_mask = torch.zeros((self.batch_size, self.seq_length))
        config = self.get_config()
        model_input = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "text_padding_mask": text_padding_mask,
        }
        labels = ids_tensor([self.batch_size], self.num_labels)
        if self.use_labels:
            model_input["labels"] = labels

        return config, model_input

    def prepare_config_and_inputs_for_image_classification(self):
        pixel_value = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.seq_length
        labels = torch.zeros(self.batch_size, dtype=torch.long, device=torch_device)
        model_input = {"pixel_values": pixel_value}
        if self.use_labels:
            model_input["labels"] = labels
        return config, model_input

    def prepare_config_and_inputs_for_captioning(self):
        language_masked_pos = torch.zeros((self.batch_size, self.seq_length))
        to_fill = list(range(0, self.seq_length, 3))
        language_masked_pos[:, to_fill] = 1
        config = self.get_config()
        label = torch.tensor([20, 5, 2])
        return config, {"language_masked_pos": language_masked_pos, "labels": label}

    def prepare_config_and_inputs_for_basemodel(self):
        language_masked_pos = torch.zeros((self.batch_size, self.seq_length))
        to_fill = list(range(0, self.seq_length, 3))
        language_masked_pos[:, to_fill] = 1
        config = self.get_config()
        torch.tensor([20, 5, 2])
        return config, {}

    def prepare_config_and_inputs_for_visual_question_answering(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        text_padding_mask = torch.zeros((self.batch_size, self.seq_length))
        return self.get_config(), {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": text_padding_mask,
        }

    def prepare_config_and_inputs_for_text_retrieval(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        text_padding_mask = torch.zeros((self.batch_size, self.seq_length))
        return self.get_config(), {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": text_padding_mask,
        }

    def create_and_check_model(self, config, input_dict):
        model = Beit3Model(config=config)
        model.to(torch_device)
        model.eval()
        model(**input_dict)

    def create_and_check_for_visual_reasoning(self, config, input_dict):
        model = Beit3ForVisualReasoning(config=config)
        model.to(torch_device)
        model.eval()
        result = model(**input_dict)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))


@require_torch
class Beit3ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Beit3Model,
            # Beit3ForVisualReasoning,
            Beit3ForImageTextRetrieval,
            Beit3ForQuestionAnswering,
            Beit3ForImageClassification,
            Beit3ForCaptioning,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": Beit3Model,
            "image-classification": Beit3ForImageClassification,
            "visual-question-answering": Beit3ForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )
    test_torchscript = False
    test_pruning = False
    test_inputs_embeds = False
    test_head_masking = False

    # special cases for Beit3ForImageClassification, Beit3ForQuestionAnswering, Beit3ForVisualReasoning, Beit3ForCaptioning
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if model_class.__name__ == "Beit3ForImageClassification":
            del inputs_dict["input_ids"]
            del inputs_dict["attention_mask"]

        if return_labels:
            if model_class.__name__ == "Beit3ForQuestionAnswering":
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, self.model_tester.num_labels, device=torch_device
                )
            elif model_class.__name__ == "Beit3ForVisualReasoning":
                inputs_dict["labels"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
            elif model_class.__name__ == "Beit3ForCaptioning":
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )

        return inputs_dict

    def setUp(self):
        self.model_tester = Beit3ModelTester()
        self.config_tester = ConfigTester(self, config_class=Beit3Config, hidden_size=37)

    def test_training(self):
        if not self.model_tester.is_training:
            return

        for model_class in self.all_model_classes:
            if model_class.__name__ in ["Beit3ForImageTextRetrieval", "Beit3ForCaptioning"]:
                continue

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            config.return_dict = True

            if model_class.__name__ in [
                *get_values(MODEL_MAPPING_NAMES),
                *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES),
            ]:
                continue

            model = model_class(config)
            model.to(torch_device)
            model.train()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
            loss = model(**inputs).loss
            loss.backward()

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            arg_names = [*signature.parameters.keys()]

            if model_class.__name__ == "Beit3ForImageClassification":
                # signature.parameters is an OrderedDict => so arg_names order is deterministic

                expected_arg_names = ["pixel_values"]
                self.assertListEqual(arg_names[:1], expected_arg_names)
            else:
                expected_arg_names = ["input_ids"]
                self.assertListEqual(arg_names[:1], expected_arg_names)

    # override as the `logit_scale` parameter initilization is different for Blip
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


@require_torch
@require_vision
class BeitModelIntegrationTest(unittest.TestCase):
    @cached_property
    def default_processor(self):
        return Beit3Processor.from_pretrained("Raghavan/beit3_base_patch16_224_in1k")

    @cached_property
    def default_image(self):
        image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
        return image

    @slow
    def test_inference_beit3_image_classification(self):
        model = Beit3ForImageClassification.from_pretrained("Raghavan/beit3_base_patch16_224_in1k").to(torch_device)

        processor = self.default_processor
        image = self.default_image
        text = "This is a photo of a cat"
        inputs = processor(text=text, images=image, return_tensors="pt")

        # forward pass
        output = model(**inputs)
        self.assertEqual(output.logits.shape, torch.Size([1, 1000]))
        expected_slice = torch.tensor([[-0.260473, -0.420061, -0.492118]])
        assert torch.allclose(output.logits.detach()[:, :3], expected_slice)

    @slow
    def test_inference_beit3_vqa(self):
        model = Beit3ForQuestionAnswering.from_pretrained("Raghavan/beit3_base_patch16_480_vqa").to(torch_device)

        processor = self.default_processor
        image = self.default_image
        text = "This is a photo of a cat"
        inputs = processor(text=text, images=image, return_tensors="pt")

        # forward pass
        output = model(**inputs)
        assert output.logits.shape == torch.Size([1, 3129])
        torch.testing.assert_allclose(
            output.logits.detach()[:, :3], torch.tensor([[-10.862484, -12.388088, -7.6599636]])
        )

    @slow
    def test_inference_beit3_visual_reasoning(self):
        model = Beit3ForVisualReasoning.from_pretrained("Raghavan/beit3_base_patch16_224_nlvr2").to(torch_device)

        processor = self.default_processor
        image = self.default_image
        text = "This is a photo of a cat"
        inputs = processor(text=text, images=image, return_tensors="pt")

        torch.cat(
            (torch.tensor(input["pixel_values"]).unsqueeze(1), torch.tensor(input["pixel_values"]).unsqueeze(1)), dim=1
        )
        # forward pass
        output = model(**inputs)
        assert output.logits.shape == torch.Size([1, 2])
        torch.testing.assert_allclose(output.logits.detach(), torch.tensor([[6.593818, -6.582055]]))

    @slow
    def test_inference_beit3_for_image_captioning(self):
        model = Beit3ForCaptioning.from_pretrained("Raghavan/beit3_base_patch16_480_coco_captioning").to(torch_device)

        processor = self.default_processor
        image = self.default_image
        text = "This is a photo of a cat"
        processor(text=text, images=image, return_tensors="pt")

        language_masked_pos = torch.zeros((input["input_ids"].shape[0], input["input_ids"].shape[1]))
        language_masked_pos[0, 5] = 1
        input_tokens = list(input["input_ids"][0])
        input_tokens[5] = 64001
        output = model(
            input_ids=torch.tensor([input_tokens]),
            pixel_values=torch.tensor(input["pixel_values"]),
            attention_mask=torch.zeros(language_masked_pos.shape),
            language_masked_pos=language_masked_pos,
        )
        assert output.logits.shape == torch.Size([1, 64010])
        assert torch.allclose(output.logits.detach()[0, :3], torch.tensor([-2.697914, -2.697912, -2.645459]))

    @slow
    def test_inference_beit3_for_image_text_retrieval(self):
        model = Beit3ForImageTextRetrieval.from_pretrained("Raghavan/beit3_base_patch16_384_coco_retrieval").to(
            torch_device
        )

        processor = self.default_processor
        image = self.default_image
        text = "This is a photo of a cat"
        processor(text=text, images=image, return_tensors="pt")

        inputs = processor(text=["This is photo of a dog"], images=image, return_tensors="pt")

        outputs = model(**inputs)

        self.assertEqual(round(float(outputs.loss.detach().numpy()), 4), 1.8435)
