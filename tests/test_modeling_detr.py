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
""" Testing suite for the PyTorch DETR model. """


import copy
import tempfile
import unittest

from PIL import Image

import requests
from transformers import is_torch_available, is_vision_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_torch, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_generation_utils import GenerationTesterMixin
from .test_modeling_common import ModelTesterMixin, floats_tensor


if is_torch_available():
    import torch
    import torchvision.transforms as T

    from transformers import DetrConfig, DetrForObjectDetection, DetrModel
    from transformers.models.detr.modeling_detr import DetrDecoder, DetrEncoder


if is_vision_available():
    from PIL import Image

    from transformers import DetrFeatureExtractor


@require_torch
class DetrModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        is_training=True,
        use_labels=False,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_queries=10,
        image_size=800,
        n_targets=15,
        n_classes=91,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_queries = num_queries
        self.image_size = image_size
        self.n_classes = n_classes

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        pixel_mask = torch.ones_like(pixel_values)

        labels = None
        if self.use_labels:
            # labels is a list of Dict (each Dict being the labels for a given example in the batch)
            labels = []
            for i in range(self.batch_size):
                target = {}
                target['class_labels'] = torch.randint(high=self.n_classes, size=(self.n_targets,))
                target['boxes'] = torch.rand(self.n_targets, 4)
                labels.append(target)

        config = DetrConfig(
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
        )
        return config, pixel_values, pixel_mask, labels

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_mask, labels = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "pixel_mask": pixel_mask}
        return config, inputs_dict

#     def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
#         model = DetrModel(config=config).get_decoder().to(torch_device).eval()
#         input_ids = inputs_dict["input_ids"]
#         attention_mask = inputs_dict["attention_mask"]

#         # first forward pass
#         outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

#         output, past_key_values = outputs.to_tuple()

#         # create hypothetical multiple next token and extent to next_input_ids
#         next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
#         next_attn_mask = ids_tensor((self.batch_size, 3), 2)

#         # append to next input_ids and
#         next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
#         next_attention_mask = torch.cat([attention_mask, next_attn_mask], dim=-1)

#         output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
#         output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)["last_hidden_state"]

#         # select random slice
#         random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
#         output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
#         output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

#         self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

#         # test that outputs are equal for slice
#         self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-2))

#     def check_encoder_decoder_model_standalone(self, config, inputs_dict):
#         model = DetrModel(config=config).to(torch_device).eval()
#         outputs = model(**inputs_dict)

#         encoder_last_hidden_state = outputs.encoder_last_hidden_state
#         last_hidden_state = outputs.last_hidden_state

#         with tempfile.TemporaryDirectory() as tmpdirname:
#             encoder = model.get_encoder()
#             encoder.save_pretrained(tmpdirname)
#             encoder = DetrEncoder.from_pretrained(tmpdirname).to(torch_device)

#         encoder_last_hidden_state_2 = encoder(inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"])[
#             0
#         ]

#         self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

#         with tempfile.TemporaryDirectory() as tmpdirname:
#             decoder = model.get_decoder()
#             decoder.save_pretrained(tmpdirname)
#             decoder = DetrDecoder.from_pretrained(tmpdirname).to(torch_device)

#         last_hidden_state_2 = decoder(
#             input_ids=inputs_dict["decoder_input_ids"],
#             attention_mask=inputs_dict["decoder_attention_mask"],
#             encoder_hidden_states=encoder_last_hidden_state,
#             encoder_attention_mask=inputs_dict["attention_mask"],
#         )[0]

#         self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)


# @require_torch
# class DetrModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
#     all_model_classes = (
#         (DetrModel, DetrForObjectDetection,)
#         if is_torch_available()
#         else ()
#     )
#     #all_generative_model_classes = (DetrForConditionalGeneration,) if is_torch_available() else ()
#     is_encoder_decoder = True
#     test_pruning = False
#     test_head_masking = False
#     test_missing_keys = False

#     def setUp(self):
#         self.model_tester = DetrModelTester(self)
#         self.config_tester = ConfigTester(self, config_class=DetrConfig)

#     def test_config(self):
#         self.config_tester.run_common_tests()

#     def test_save_load_strict(self):
#         config, inputs_dict = self.model_tester.prepare_config_and_inputs()
#         for model_class in self.all_model_classes:
#             model = model_class(config)

#             with tempfile.TemporaryDirectory() as tmpdirname:
#                 model.save_pretrained(tmpdirname)
#                 model2, info = model_class.from_pretrained(tmpdirname, output_loading_info=True)
#             self.assertEqual(info["missing_keys"], [])

#     def test_decoder_model_past_with_large_inputs(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs()
#         self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

#     def test_encoder_decoder_model_standalone(self):
#         config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
#         self.model_tester.check_encoder_decoder_model_standalone(*config_and_inputs)

#     # DetrForSequenceClassification does not support inputs_embeds
#     def test_inputs_embeds(self):
#         config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

#         for model_class in (DetrModel, DetrForConditionalGeneration, DetrForQuestionAnswering):
#             model = model_class(config)
#             model.to(torch_device)
#             model.eval()

#             inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

#             if not self.is_encoder_decoder:
#                 input_ids = inputs["input_ids"]
#                 del inputs["input_ids"]
#             else:
#                 encoder_input_ids = inputs["input_ids"]
#                 decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
#                 del inputs["input_ids"]
#                 inputs.pop("decoder_input_ids", None)

#             wte = model.get_input_embeddings()
#             if not self.is_encoder_decoder:
#                 inputs["inputs_embeds"] = wte(input_ids)
#             else:
#                 inputs["inputs_embeds"] = wte(encoder_input_ids)
#                 inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

#             with torch.no_grad():
#                 model(**inputs)[0]


def assert_tensors_close(a, b, atol=1e-12, prefix=""):
    """If tensors have different shapes, different values or a and b are not both tensors, raise a nice Assertion error."""
    if a is None and b is None:
        return True
    try:
        if torch.allclose(a, b, atol=atol):
            return True
        raise
    except Exception:
        pct_different = (torch.gt((a - b).abs(), atol)).float().mean().item()
        if a.numel() > 100:
            msg = f"tensor values are {pct_different:.1%} percent different."
        else:
            msg = f"{a} != {b}"
        if prefix:
            msg = prefix + ": " + msg
        raise AssertionError(msg)


def _long_tensor(tok_lst):
    return torch.tensor(tok_lst, dtype=torch.long, device=torch_device)


TOLERANCE = 1e-4


# We will verify our outputs against the original implementation on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    # standard PyTorch mean-std input image normalization
    transform = T.Compose([T.Resize(800), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    return img


# We will verify our results on an image of cute cats
def prepare_img():
    image = Image.open("./tests/fixtures/tests_samples/COCO/cats.png")
    return image


@require_torch
@slow
class DetrModelIntegrationTests(unittest.TestCase):
    @cached_property
    def default_feature_extractor(self):
        # TODO replace by facebook/detr-resnet-50
        return DetrFeatureExtractor() if is_vision_available() else None

    def test_inference_no_head(self):
        # TODO replace by facebook/detr-resnet-50
        model = DetrModel.from_pretrained("nielsr/detr-resnet-50-new").to(torch_device)
        model.eval()

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        encoding = feature_extractor(images=image, return_tensors="pt").to(torch_device)
        
        # standard PyTorch mean-std input image normalization
        transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # mean-std normalize the input image (batch-size: 1)
        img = transform(image).unsqueeze(0)

        assert img.shape == encoding['pixel_values'].shape
        assert torch.allclose(img[0,:3,:3,:3], encoding['pixel_values'][0,:3,:3,:3], atol=1e-4)
        
        with torch.no_grad():
            outputs = model(**encoding)

        expected_shape = torch.Size((1, 100, 256))
        assert outputs.last_hidden_state.shape == expected_shape
        expected_slice = torch.tensor(
            [[0.0616, -0.5146, -0.4032], [-0.7629, -0.4934, -1.7153], [-0.4768, -0.6403, -0.7826]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4))

    def test_inference_object_detection_head(self):
        model = DetrForObjectDetection.from_pretrained("nielsr/detr-resnet-50-new").to(torch_device)
        model.eval()

        feature_extractor = self.default_feature_extractor
        image = prepare_img()
        encoding = feature_extractor(images=image, return_tensors="pt").to(torch_device)
        pixel_values = encoding["pixel_values"].to(torch_device)
        pixel_mask = encoding["pixel_mask"].to(torch_device)

        with torch.no_grad():
            outputs = model(pixel_values, pixel_mask)

        expected_shape_logits = torch.Size((1, model.config.num_queries, model.config.num_labels + 1))
        self.assertEqual(outputs.pred_logits.shape, expected_shape_logits)
        expected_slice_logits = torch.tensor(
            [[-19.1194, -0.0893, -11.0154], [-17.3640, -1.8035, -14.0219], [-20.0461, -0.5837, -11.1060]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.pred_logits[0, :3, :3], expected_slice_logits, atol=1e-4))

        expected_shape_boxes = torch.Size((1, model.config.num_queries, 4))
        self.assertEqual(outputs.pred_boxes.shape, expected_shape_boxes)
        expected_slice_boxes = torch.tensor(
            [[0.4433, 0.5302, 0.8853], [0.5494, 0.2517, 0.0529], [0.4998, 0.5360, 0.9956]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes, atol=1e-4))
