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
import tempfile
import unittest

from transformers import AutoProcessor, SiglipVisionConfig, UltrabertConfig, UltraVBertConfig, is_torch_available
from transformers.testing_utils import Expectations, require_torch, require_vision, slow, torch_device

from ...test_image_processing_common import load_test_image
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import floats_tensor, ids_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch
    from safetensors.torch import load_file

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
        return torch.ones_like(input_ids)


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

    def test_task_heads(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        batch_size, seq_length = inputs_dict["input_ids"].shape
        vocab_size = config.text_config.vocab_size
        config.num_labels = 3
        label_shapes = {
            UltraVBertForMaskedLM: ((batch_size, seq_length), vocab_size, (batch_size, seq_length, vocab_size)),
            UltraVBertForSequenceClassification: ((batch_size,), 3, (batch_size, 3)),
            UltraVBertForTokenClassification: ((batch_size, seq_length), 3, (batch_size, seq_length, 3)),
        }
        for model_class, (labels_shape, num_classes, logits_shape) in label_shapes.items():
            labels = ids_tensor(list(labels_shape), num_classes)
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict, labels=labels)
            self.assertEqual(tuple(outputs.logits.shape), logits_shape)
            self.assertIsNotNone(outputs.loss)

    def test_masked_lm_saves_ultrabert_head_layout(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = UltraVBertForMaskedLM(config)
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            saved_keys = set(load_file(f"{tmpdirname}/model.safetensors"))
            self.assertTrue({"head.linear1.weight", "head.linear2.weight", "head.decoder.weight"} <= saved_keys)
            self.assertFalse(any(key.startswith(("lm_head.", "projection_head.")) for key in saved_keys))
            reloaded = UltraVBertForMaskedLM.from_pretrained(tmpdirname)
        torch.testing.assert_close(reloaded.lm_head.weight, model.lm_head.weight)
        torch.testing.assert_close(reloaded.projection_head.linear1.weight, model.projection_head.linear1.weight)

    @unittest.skip(reason="SDPA's flash backend doesn't support the mandatory sliding-window attention masks.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="UltraVBert has no generative head; VLMModelTest inherits GenerationTesterMixin.")
    def test_generation_tester_mixin_inheritance(self):
        pass

    @unittest.skip(reason="Vision head's probe has no gradient.")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="Vision head's probe has no gradient.")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Vision head's probe has no gradient.")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass


@require_torch
@require_vision
@slow
class UltraVBertIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """Check the released checkpoint in float32 on a photographed receipt."""

    checkpoint = "athrael-soju/UltraVBERT"
    revision = "1ec67d53f511d246a88887bec939076ad2150905"
    image_url = (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/receipt_00008.png"
    )

    def load_checkpoint(self, auto_class):
        model = auto_class.from_pretrained(self.checkpoint, revision=self.revision, dtype=torch.float32)
        processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)
        return model.to(torch_device).eval(), processor

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
