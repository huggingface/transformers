# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import unittest

from transformers import AutoProcessor, SiglipVisionConfig, UltrabertConfig, UltraVBertConfig, is_torch_available
from transformers.testing_utils import Expectations, require_torch, require_vision, slow, torch_device

from ...test_image_processing_common import load_test_image
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import floats_tensor, ids_tensor, random_attention_mask
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        AutoModel,
        AutoModelForMaskedLM,
        UltraVBertForMaskedLM,
        UltraVBertForSequenceClassification,
        UltraVBertForTokenClassification,
        UltraVBertModel,
    )


class UltraVBertModelTester(VLMModelTester):
    base_model_class = UltraVBertModel
    config_class = UltraVBertConfig
    text_config_class = UltrabertConfig
    vision_config_class = SiglipVisionConfig
    # NOTE: we don't want a generation class here so it's a workaround
    _required_attributes = ("config_class", "text_config_class", "base_model_class", "vision_config_class")

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_size", 8)
        kwargs.setdefault("patch_size", 2)
        kwargs.setdefault("pixel_shuffle_factor", 2)
        kwargs.setdefault("num_image_tokens", 4)
        kwargs.setdefault("prediction_head_size", 16)
        kwargs.setdefault("sliding_window", 2)
        kwargs.setdefault("layer_types", ["sliding_attention", "full_attention"])
        super().__init__(parent, **kwargs)

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor([batch_size, 1, self.num_channels, self.image_size, self.image_size])

    def create_attention_mask(self, input_ids):
        return random_attention_mask(input_ids.shape).to(torch_device)

    def get_vision_config(self):
        config = super().get_vision_config()
        config.vision_use_head = False
        return config

    def prepare_config_and_inputs(self):
        config, inputs_dict = self.prepare_config_and_inputs_for_common()
        sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
        token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
        return (
            config,
            inputs_dict["input_ids"],
            inputs_dict["attention_mask"],
            inputs_dict["pixel_values"],
            sequence_labels,
            token_labels,
        )

    def create_and_check_model(self, config, input_ids, input_mask, pixel_values, sequence_labels, token_labels):
        model = UltraVBertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids=input_ids, attention_mask=input_mask, pixel_values=pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_masked_lm(
        self, config, input_ids, input_mask, pixel_values, sequence_labels, token_labels
    ):
        model = UltraVBertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids=input_ids, attention_mask=input_mask, pixel_values=pixel_values, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, input_mask, pixel_values, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = UltraVBertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids, attention_mask=input_mask, pixel_values=pixel_values, labels=sequence_labels
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, input_mask, pixel_values, sequence_labels, token_labels
    ):
        config.num_labels = self.num_labels
        model = UltraVBertForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids=input_ids, attention_mask=input_mask, pixel_values=pixel_values, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))


@require_torch
class UltraVBertModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = UltraVBertModelTester
    all_model_classes = (
        (
            UltraVBertModel,
            UltraVBertForMaskedLM,
            UltraVBertForSequenceClassification,
            UltraVBertForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": UltraVBertModel,
            "fill-mask": UltraVBertForMaskedLM,
            "text-classification": UltraVBertForSequenceClassification,
            "image-classification": UltraVBertForSequenceClassification,
            "token-classification": UltraVBertForTokenClassification,
            "zero-shot": UltraVBertForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )

    test_mismatched_shapes = False
    skip_test_image_features_output_shape = True  # UltraVBert merges batch_size with num_images in index 0
    model_split_percents = [0.5, 0.8, 0.9]

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    @unittest.skip(reason="SDPA's flash backend doesn't support the mandatory sliding-window attention masks.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="UltraVBert has no generative head; VLMModelTest inherits GenerationTesterMixin.")
    def test_generation_tester_mixin_inheritance(self):
        pass


@require_torch
@require_vision
@slow
class UltraVBertIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """Check the released checkpoint in float32 on a photographed receipt."""

    checkpoint = "hf-internal-testing/testing-dummy-UltraVBERT"
    image_url = (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/receipt_00008.png"
    )

    def load_checkpoint(self, auto_class):
        model = auto_class.from_pretrained(self.checkpoint, dtype=torch.float32)
        processor = AutoProcessor.from_pretrained(self.checkpoint)
        return model.to(torch_device), processor

    def test_inference_masked_lm(self):
        model, processor = self.load_checkpoint(AutoModelForMaskedLM)
        inputs = processor(
            text="<image>Rounding 0 <|mask|> 50,000", images=[load_test_image(self.image_url)], return_tensors="pt"
        ).to(torch_device)
        expected_ids = [16046, 42472, 17647, 9712, 22580]
        expected_probs = Expectations(
            {(None, None): torch.tensor([0.991282, 0.003430, 0.003411, 0.000647, 0.000110])}
        ).get_expectation()

        with torch.no_grad():
            logits = model(**inputs).logits

        self.assertEqual(logits.shape, (1, 1670, 49280))
        masked_logits = logits[inputs.input_ids == processor.tokenizer.mask_token_id]
        top_probs, top_ids = masked_logits.softmax(dim=-1).topk(5)
        self.assertEqual(top_ids[0].tolist(), expected_ids)
        torch.testing.assert_close(top_probs[0].cpu(), expected_probs, rtol=1e-4, atol=1e-4)
        self.assertEqual(processor.tokenizer.decode(top_ids[0, :1]), " Total")

    def test_inference_page_hidden_states(self):
        model, processor = self.load_checkpoint(AutoModel)
        inputs = processor(text="<image>", images=[load_test_image(self.image_url)], return_tensors="pt").to(
            torch_device
        )
        expected_slice = Expectations(
            {
                (None, None): torch.tensor(
                    [
                        [0.954978, -1.569990, 0.402189],
                        [-1.443539, -2.338481, 3.283278],
                        [3.948290, 6.073206, 7.573400],
                        [-0.218822, 0.179303, 0.313176],
                    ]
                )
            }
        ).get_expectation()

        with torch.no_grad():
            output = model(**inputs).last_hidden_state

        self.assertEqual(inputs.pixel_values.shape, (1, 25, 3, 512, 512))
        self.assertEqual(output.shape, (1, 1658, 768))
        torch.testing.assert_close(output[0, [0, 3, 1000, 1657], :3].cpu(), expected_slice, rtol=1e-4, atol=1e-4)
