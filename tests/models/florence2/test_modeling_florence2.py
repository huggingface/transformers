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
"""Testing suite for the PyTorch Florence2 model."""

import unittest

import requests

from transformers import (
    AutoProcessor,
    Florence2Config,
    Florence2ForConditionalGeneration,
    Florence2Model,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class Florence2VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        num_channels=3,
        image_size=8,
        seq_length=13,
        encoder_seq_length=18,
        is_training=True,
        vocab_size=99,
        max_position_embeddings=64,
        encoder_layers=1,
        encoder_ffn_dim=8,
        decoder_layers=1,
        decoder_ffn_dim=8,
        num_attention_heads=1,
        d_model=8,
        activation_function="gelu",
        dropout=0.1,
        eos_token_id=2,
        bos_token_id=0,
        pad_token_id=1,
        image_token_id=4,
        depths=[1],
        patch_size=[7],
        patch_stride=[4],
        patch_padding=[3],
        patch_prenorm=[False],
        embed_dim=[8],
        num_heads=[1],
        num_groups=[1],
        window_size=12,
        drop_path_rate=0.1,
        projection_dim=8,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.num_hidden_layers = decoder_layers
        self.hidden_size = d_model

        # Language model configs
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_attention_heads = num_attention_heads
        self.d_model = d_model
        self.activation_function = activation_function
        self.dropout = dropout
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id

        # Vision model configs
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size
        self.depths = depths
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.patch_prenorm = patch_prenorm
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.window_size = window_size
        self.projection_dim = projection_dim

        self.num_channels = 3
        self.num_image_tokens = 5
        self.seq_length = seq_length + self.num_image_tokens
        self.encoder_seq_length = encoder_seq_length

    def get_config(self):
        text_config = {
            "model_type": "bart",
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "encoder_layers": self.encoder_layers,
            "encoder_ffn_dim": self.encoder_ffn_dim,
            "encoder_attention_heads": self.num_attention_heads,
            "decoder_layers": self.decoder_layers,
            "decoder_ffn_dim": self.decoder_ffn_dim,
            "decoder_attention_heads": self.num_attention_heads,
            "d_model": self.d_model,
            "activation_function": self.activation_function,
            "dropout": self.dropout,
            "attention_dropout": self.dropout,
            "activation_dropout": self.dropout,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
        }

        vision_config = {
            "drop_path_rate": self.drop_path_rate,
            "patch_size": self.patch_size,
            "depths": self.depths,
            "patch_stride": self.patch_stride,
            "patch_padding": self.patch_padding,
            "patch_prenorm": self.patch_prenorm,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_groups": self.num_groups,
            "window_size": self.window_size,
            "activation_function": self.activation_function,
            "projection_dim": self.projection_dim,
        }

        return Florence2Config(
            text_config=text_config, vision_config=vision_config, image_token_id=self.image_token_id
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.num_channels,
                self.image_size,
                self.image_size,
            ]
        )
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size - 1) + 1
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = self.image_token_id
        input_ids[:, -1] = self.eos_token_id
        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id)

        inputs_dict = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        config = self.get_config()
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_florence2_model_fp16_forward(self, config, input_ids, pixel_values, attention_mask):
        model = Florence2ForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values.to(torch.float16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())

    @unittest.skip(
        reason="This architecture (bart) has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    )
    def test_load_save_without_tied_weights(self):
        pass


@require_torch
class Florence2ForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Florence2ForConditionalGeneration`.
    """

    all_model_classes = (Florence2Model, Florence2ForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "image-to-text": Florence2ForConditionalGeneration,
            "image-text-to-text": Florence2ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_head_masking = False
    test_attention_outputs = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Florence2VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Florence2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
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

    @unittest.skip(
        reason="This architecture has tied weights by default and there is no way to remove it, check: https://github.com/huggingface/transformers/pull/31771#issuecomment-2210915245"
    )
    def test_load_save_without_tied_weights(self):
        pass


def prepare_img():
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


@slow
@require_torch
class Florence2ForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_name = "ducviet00/Florence-2-base-hf"
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.image1 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg?download=true",
                stream=True,
            ).raw
        )
        self.image2 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true",
                stream=True,
            ).raw
        )

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_base_model_inference_eager(self):
        model = Florence2ForConditionalGeneration.from_pretrained(self.model_name, attn_implementation="eager").to(
            torch_device
        )

        prompt = "<DETAILED_CAPTION>"
        inputs = self.processor(images=self.image1, text=prompt, return_tensors="pt")
        inputs.to(device=torch_device)

        EXPECTED_INPUT_IDS = [[self.processor.image_token_id] * self.processor.num_image_tokens + [0, 47066, 21700, 11, 4617, 99, 16, 2343, 11, 5, 2274, 4, 2]]  # fmt: skip
        self.assertEqual(inputs["input_ids"].tolist(), EXPECTED_INPUT_IDS)

        predictions = model.generate(**inputs, max_new_tokens=100)

        EXPECTED_PREDICTION_IDS = [[2, 0, 133, 2274, 924, 10, 912, 1203, 2828, 15, 5, 526, 9, 10, 2014, 11, 35910, 6, 188, 469, 412, 4, 20, 2014, 16, 9321, 19, 3413, 6, 3980, 6, 8, 19638, 6, 8, 89, 32, 82, 3051, 15, 5, 2767, 22609, 4, 20, 6360, 16, 7097, 11, 5, 3618, 4, 2]]  # fmt: skip
        self.assertEqual(predictions.tolist(), EXPECTED_PREDICTION_IDS)

        generated_text = self.processor.batch_decode(predictions, skip_special_tokens=True)[0]

        EXPECTED_GENERATED_TEXT = "The image shows a stop sign sitting on the side of a street in Chinatown, New York City. The street is lined with buildings, trees, and statues, and there are people walking on the footpath. The sky is visible in the background."  # fmt: skip
        self.assertEqual(generated_text, EXPECTED_GENERATED_TEXT)

    def test_base_model_batching_inference_eager(self):
        model = Florence2ForConditionalGeneration.from_pretrained(self.model_name, attn_implementation="eager").to(
            torch_device
        )

        images = [self.image1, self.image2]
        prompts = ["<REGION_PROPOSAL>", "<OPEN_VOCABULARY_DETECTION>wheels"]
        inputs = self.processor(images=images, text=prompts, padding="longest", return_tensors="pt")

        EXPECTED_INPUT_IDS = [
            [self.processor.image_token_id] * self.processor.num_image_tokens
            + [0, 574, 22486, 5, 976, 5327, 11, 5, 2274, 4, 2],
            [self.processor.image_token_id] * self.processor.num_image_tokens
            + [0, 574, 22486, 10562, 11, 5, 2274, 4, 2, 1, 1],
        ]
        self.assertEqual(inputs["input_ids"].tolist(), EXPECTED_INPUT_IDS)

        inputs.to(device=torch_device)
        predictions = model.generate(**inputs, max_new_tokens=100)

        EXPECTED_PREDICTION_IDS = [
            [2, 0, 50269, 50269, 51267, 50980, 50269, 50269, 50688, 50942, 50269, 50333, 50633, 50941, 51033, 50269, 51267, 50934, 50794, 50814, 51190, 51032, 50432, 50402, 50634, 50692, 50269, 50334, 50340, 50927, 51224, 50417, 51267, 50930, 51076, 50944, 51159, 51028, 50836, 50947, 50915, 51030, 2],
            [2, 0, 28884,  2507, 50413, 50839, 51139, 51047, 28884,  2507, 50980, 50842, 51135, 51043, 28884, 2507, 50417, 50848, 50573, 51043,      2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]  # fmt: skip
        self.assertEqual(predictions.tolist(), EXPECTED_PREDICTION_IDS)

        generated_texts = self.processor.batch_decode(predictions, skip_special_tokens=False)

        EXPECTED_GENERATED_TEXTS = [
            "</s><s><loc_0><loc_0><loc_998><loc_711><loc_0><loc_0><loc_419><loc_673><loc_0><loc_64><loc_364><loc_672><loc_764><loc_0><loc_998><loc_665><loc_525><loc_545><loc_921><loc_763><loc_163><loc_133><loc_365><loc_423><loc_0><loc_65><loc_71><loc_658><loc_955><loc_148><loc_998><loc_661><loc_807><loc_675><loc_890><loc_759><loc_567><loc_678><loc_646><loc_761></s>",
            "</s><s>wheels<loc_144><loc_570><loc_870><loc_778>wheels<loc_711><loc_573><loc_866><loc_774>wheels<loc_148><loc_579><loc_304><loc_774></s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>",
        ]
        self.assertEqual(generated_texts, EXPECTED_GENERATED_TEXTS)

        parsed_answer_0 = self.processor.post_process_generation(
            generated_texts[0], task="<REGION_PROPOSAL>", image_size=(images[0].width, images[0].height)
        )
        EXPECTED_PARSED_ANSWER_0 = {"<REGION_PROPOSAL>": {"bboxes": [[0, 0, 1298, 623], [0, 0, 545, 589], [0, 56, 473, 589], [993, 0, 1298, 582], [683, 477, 1197, 668], [212, 116, 475, 370], [0, 57, 92, 576], [1242, 130, 1298, 579], [1049, 591, 1157, 665], [737, 594, 840, 667]], "labels": ["", "", "", "", "", "", "", "", "", ""]}}  # fmt: skip

        self.assertEqual(parsed_answer_0, EXPECTED_PARSED_ANSWER_0)

        parsed_answer_1 = self.processor.post_process_generation(
            generated_texts[1], task="<OPEN_VOCABULARY_DETECTION>", image_size=(images[1].width, images[1].height)
        )
        EXPECTED_PARSED_ANSWER_1 = {"<OPEN_VOCABULARY_DETECTION>": {"bboxes": [[92, 273, 557, 373], [455, 275, 554, 371], [95, 278, 194, 371]], "bboxes_labels": ["wheels", "wheels", "wheels"], "polygons": [], "polygons_labels": []}}  # fmt: skip

        self.assertEqual(parsed_answer_1, EXPECTED_PARSED_ANSWER_1)

    def test_base_model_inference_sdpa(self):
        model = Florence2ForConditionalGeneration.from_pretrained(self.model_name, attn_implementation="sdpa").to(
            torch_device
        )

        prompt = "<REFERRING_EXPRESSION_SEGMENTATION>a car"
        inputs = self.processor(images=self.image2, text=prompt, return_tensors="pt")
        inputs.to(device=torch_device)

        EXPECTED_INPUT_IDS = [[self.processor.image_token_id] * self.processor.num_image_tokens + [0, 574, 22486, 10, 512, 11, 5, 2274, 19, 11445, 2]]  # fmt: skip
        self.assertEqual(inputs["input_ids"].tolist(), EXPECTED_INPUT_IDS)

        predictions = model.generate(**inputs, max_new_tokens=100)

        EXPECTED_PREDICTION_IDS = [[2, 0, 50548, 50648, 50551, 50648, 50559, 50641, 50562, 50641, 50567, 50637, 50570, 50637, 50575, 50633, 50579, 50631, 50584, 50629, 50589, 50627, 50593, 50624, 50600, 50622, 50606, 50620, 50612, 50618, 50618, 50616, 50625, 50614, 50634, 50612, 50645, 50610, 50659, 50608, 50678, 50606, 50758, 50606, 50783, 50608, 50797, 50610, 50808, 50612, 50816, 50614, 50822, 50616, 50828, 50618, 50835, 50620, 50841, 50622, 50847, 50624, 50853, 50629, 50858, 50635, 50861, 50641, 50864, 50648, 50867, 50654, 50870, 50660, 50872, 50666, 50875, 50670, 50877, 50677, 50880, 50683, 50883, 50689, 50886, 50695, 50889, 50702, 50895, 50710, 50900, 50714, 50905, 50716, 50908, 50720, 50908, 50725, 50911, 50729, 2]]  # fmt: skip
        self.assertEqual(predictions.tolist(), EXPECTED_PREDICTION_IDS)

        generated_text = self.processor.batch_decode(predictions, skip_special_tokens=False)[0]

        EXPECTED_GENERATED_TEXT = "</s><s><loc_279><loc_379><loc_282><loc_379><loc_290><loc_372><loc_293><loc_372><loc_298><loc_368><loc_301><loc_368><loc_306><loc_364><loc_310><loc_362><loc_315><loc_360><loc_320><loc_358><loc_324><loc_355><loc_331><loc_353><loc_337><loc_351><loc_343><loc_349><loc_349><loc_347><loc_356><loc_345><loc_365><loc_343><loc_376><loc_341><loc_390><loc_339><loc_409><loc_337><loc_489><loc_337><loc_514><loc_339><loc_528><loc_341><loc_539><loc_343><loc_547><loc_345><loc_553><loc_347><loc_559><loc_349><loc_566><loc_351><loc_572><loc_353><loc_578><loc_355><loc_584><loc_360><loc_589><loc_366><loc_592><loc_372><loc_595><loc_379><loc_598><loc_385><loc_601><loc_391><loc_603><loc_397><loc_606><loc_401><loc_608><loc_408><loc_611><loc_414><loc_614><loc_420><loc_617><loc_426><loc_620><loc_433><loc_626><loc_441><loc_631><loc_445><loc_636><loc_447><loc_639><loc_451><loc_639><loc_456><loc_642><loc_460></s>"  # fmt: skip
        self.assertEqual(generated_text, EXPECTED_GENERATED_TEXT)

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task="<REFERRING_EXPRESSION_SEGMENTATION>",
            image_size=(self.image2.width, self.image2.height),
        )
        EXPECTED_PARSED_ANSWER = {'<REFERRING_EXPRESSION_SEGMENTATION>': {'polygons': [[[178, 182, 180, 182, 185, 178, 187, 178, 191, 176, 192, 176, 196, 174, 198, 174, 201, 173, 205, 172, 207, 170, 212, 169, 216, 168, 219, 167, 223, 166, 228, 165, 233, 164, 240, 163, 249, 162, 262, 162, 313, 162, 329, 162, 338, 163, 345, 164, 350, 165, 354, 166, 358, 167, 362, 168, 366, 169, 370, 170, 374, 173, 377, 175, 379, 178, 381, 182, 383, 185, 384, 187, 386, 190, 388, 192, 389, 196, 391, 198, 393, 201, 395, 204, 397, 208, 400, 211, 404, 213, 407, 214, 409, 216, 409, 219, 411, 221]]], 'labels': ['']}}  # fmt: skip
        self.assertEqual(parsed_answer, EXPECTED_PARSED_ANSWER)

    def test_base_model_batching_inference_sdpa(self):
        model = Florence2ForConditionalGeneration.from_pretrained(self.model_name, attn_implementation="sdpa").to(
            torch_device
        )

        images = [self.image1, self.image2]
        prompts = ["<OCR>", "<OD>"]
        inputs = self.processor(images=images, text=prompts, padding="longest", return_tensors="pt")

        EXPECTED_INPUT_IDS = [
            [self.processor.image_token_id] * self.processor.num_image_tokens + [0, 2264, 16, 5, 2788, 11, 5, 2274, 116, 2, 1, 1, 1],
            [self.processor.image_token_id] * self.processor.num_image_tokens + [0, 574, 22486, 5, 8720, 19, 4120, 766, 11, 5, 2274, 4, 2],
        ]  # fmt: skip
        self.assertEqual(inputs["input_ids"].tolist(), EXPECTED_INPUT_IDS)

        inputs.to(device=torch_device)
        predictions = model.generate(**inputs, max_new_tokens=100)

        EXPECTED_PREDICTION_IDS = [
            [2, 0, 47643, 47240, 6382, 47643, 7405, 495, 211, 2571, 4014, 5733, 36714, 11582, 11582, 36714, 18164, 9357, 36714, 6248, 3602, 37127, 27969, 7471, 44636, 23171, 41907, 27, 16948, 45895, 11582, 45262, 18537, 530, 791, 384, 229, 791, 5733, 565, 3048, 673, 10932, 5733, 565, 11120, 673, 2],
            [2, 0, 5901, 50322, 50602, 51202, 51043, 11219, 3679, 50694, 50772, 50743, 50784, 13630, 50978, 50845, 51134, 51041, 50419, 50853, 50578, 51042, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ]  # fmt: skip
        self.assertEqual(predictions.tolist(), EXPECTED_PREDICTION_IDS)

        generated_texts = self.processor.batch_decode(predictions, skip_special_tokens=False)

        EXPECTED_GENERATED_TEXTS = [
            "</s><s>中文中BBD DATSTOP第福科技有限公司KU O KUOPTUSOyesOPTUSTO</s>",
            "</s><s>car<loc_53><loc_333><loc_933><loc_774>door handle<loc_425><loc_503><loc_474><loc_515>wheel<loc_709><loc_576><loc_865><loc_772><loc_150><loc_584><loc_309><loc_773></s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>",
        ]  # fmt: skip
        self.assertEqual(generated_texts, EXPECTED_GENERATED_TEXTS)

        parsed_answer = self.processor.post_process_generation(
            generated_texts[1], task="<OD>", image_size=(images[1].width, images[1].height)
        )
        EXPECTED_PARSED_ANSWER = {'<OD>': {'bboxes': [[34, 160, 597, 371], [272, 241, 303, 247], [454, 276, 553, 370], [96, 280, 198, 371]], 'labels': ['car', 'door handle', 'wheel', 'wheel']}}  # fmt: skip
        self.assertEqual(parsed_answer, EXPECTED_PARSED_ANSWER)
