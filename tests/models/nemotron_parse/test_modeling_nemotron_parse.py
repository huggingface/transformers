# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch NemotronParse model."""

import unittest

from transformers import (
    AutoProcessor,
    NemotronParseConfig,
    NemotronParseForConditionalGeneration,
    NemotronParseModel,
    is_torch_available,
)
from transformers.image_utils import load_image
from transformers.testing_utils import Expectations, require_torch, require_vision, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...model_tester_config import ModelTesterConfigMixin
from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch


class NemotronParseModelTester(ModelTesterConfigMixin):
    defaults = {}
    small_config_overrides = {
        "batch_size": 2,
        "seq_length": 7,
        "is_training": True,
        "image_height": 16,
        "image_width": 32,
        "num_channels": 3,
        "vocab_size": 99,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "pad_token_id": 1,
        "decoder_start_token_id": 2,
        "vision_config": {
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "patch_size": 4,
            "max_img_size": 32,
            "num_cls_tokens": 4,
            "num_registers": 4,
            "summary_idxs": [0, 1, 2],
        },
        "text_config": {
            "vocab_size": 99,
            "d_model": 32,
            "decoder_layers": 2,
            "decoder_attention_heads": 2,
            "decoder_ffn_dim": 37,
            "pad_token_id": 1,
            "decoder_start_token_id": 2,
        },
    }

    @classmethod
    def get_small_config_defaults(cls, **kwargs):
        values = super().get_small_config_defaults(**kwargs)
        patch_size = values["vision_config"]["patch_size"]
        values["decoder_seq_length"] = values["seq_length"]
        values["encoder_seq_length"] = (values["image_height"] // patch_size) * (
            values["image_width"] // patch_size
        ) + 8
        return values

    def __init__(self, parent, **kwargs):
        self.parent = parent
        for key, value in self.get_small_config_defaults(**kwargs).items():
            setattr(self, key, value)

    def get_config(self):
        return NemotronParseConfig(vision_config=self.vision_config, text_config=self.text_config)

    def prepare_config_and_inputs_for_common(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_height, self.image_width])
        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        decoder_input_ids[:, 0] = self.decoder_start_token_id
        inputs_dict = {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": torch.ones_like(decoder_input_ids),
        }
        return self.get_config(), inputs_dict


@require_torch
class NemotronParseModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    model_tester_class = NemotronParseModelTester
    all_model_classes = (NemotronParseModel, NemotronParseForConditionalGeneration) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-text-to-text": NemotronParseForConditionalGeneration} if is_torch_available() else {}
    )
    is_encoder_decoder = True
    _is_composite = True

    def setUp(self):
        self.model_tester = NemotronParseModelTester(self)
        self.config_tester = ConfigTester(self, config_class=NemotronParseConfig, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(
        reason="The neck shortens the encoder output, so cross-attention keys differ from the encoder length"
    )
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="The strict RadioConfig rejects the scalar `norm_std` set by `_config_zero_init`")
    def test_can_load_ignoring_mismatched_shapes(self):
        pass

    @unittest.skip(reason="NemotronParse needs `pixel_values` to generate")
    def test_generate_without_input_ids(self):
        pass

    # The tests below read the prompt from `input_ids`; NemotronParse has no text encoder and takes `decoder_input_ids`
    @unittest.skip(reason="No `input_ids`")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="No `input_ids`")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(reason="No `input_ids`")
    def test_forward_with_logits_to_keep(self):
        pass

    @unittest.skip(reason="No `input_ids`")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="No `input_ids`")
    def test_resize_embeddings_untied(self):
        pass


@require_torch
@require_vision
@slow
class NemotronParseIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "nvidia/NVIDIA-Nemotron-Parse-2.0"

    def test_generation(self):
        model = NemotronParseForConditionalGeneration.from_pretrained(self.model_id, device_map=torch_device)
        processor = AutoProcessor.from_pretrained(self.model_id)
        image = load_image(
            url_to_local_path(
                "https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/images/paddle_doc_test.jpg"
            )
        )
        prompt = "</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device)

        generated_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        generated_text = processor.batch_decode(generated_ids[:, inputs["decoder_input_ids"].shape[1] :])[0]

        # the Hub generation config forces `</s>` at `max_new_tokens`
        expected_texts = Expectations(
            {
                (None, None): (
                    "<x_0.3018><y_0.2719># SCANNED DOCUMENT<x_0.6426><y_0.2953><class_Title>\n\n"
                    "<x_0.3008><y_0.318>Bill To: Example Customer<x_0.5146><y_0.3328><class_Text>\n\n"
                    "<x_0.3018><y_0.3406>In</s>"
                )
            }
        )
        self.assertEqual(generated_text, expected_texts.get_expectation())
