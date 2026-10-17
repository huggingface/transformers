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
from typing import ClassVar

from huggingface_hub import hf_hub_download
from PIL import Image

from tests.test_configuration_common import ConfigTester
from tests.test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from transformers import is_torch_available
from transformers.models.colultravbert.configuration_colultravbert import ColUltraVBertConfig
from transformers.models.colultravbert.modeling_colultravbert import (
    ColUltraVBertForRetrieval,
    ColUltraVBertForRetrievalOutput,
)
from transformers.models.colultravbert.processing_colultravbert import ColUltraVBertProcessor
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_vision,
    slow,
    torch_device,
)


if is_torch_available():
    import torch


class ColUltraVBertForRetrievalModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_images=2,
        seq_length=7,
        text_config=None,
        is_training=False,
        vision_config=None,
        pixel_shuffle_factor=2,
        embedding_dim=64,
    ):
        if text_config is None:
            text_config = {
                "vocab_size": 99,
                "pad_token_id": 0,
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 64,
                "prediction_head_size": 16,
                "sliding_window": 2,
                "layer_types": ["sliding_attention", "full_attention"],
                "max_position_embeddings": 512,
                "initializer_range": 0.02,
            }
        if vision_config is None:
            vision_config = {
                "image_size": 16,
                "patch_size": 4,
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 32,
                "dropout": 0.1,
                "attention_dropout": 0.1,
                "initializer_range": 0.02,
                "vision_use_head": False,
            }
        self.is_training = is_training
        self.parent = parent
        self.batch_size = batch_size
        self.text_config = text_config
        self.vision_config = vision_config
        self.num_images = num_images
        self.image_size = vision_config["image_size"]
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.image_token_id = self.text_config["vocab_size"] - 1
        self.pad_token_id = text_config["pad_token_id"]
        self.image_seq_length = (
            int(((vision_config["image_size"] // vision_config["patch_size"]) ** 2) / (pixel_shuffle_factor**2))
            * self.num_images
        )
        self.seq_length = seq_length + self.image_seq_length

        self.hidden_size = text_config["hidden_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]

        self.embedding_dim = embedding_dim
        self.vlm_config = {
            "model_type": "ultravbert",
            "text_config": self.text_config,
            "vision_config": self.vision_config,
            "image_token_id": self.image_token_id,
            "pixel_shuffle_factor": self.pixel_shuffle_factor,
        }

    def get_config(self):
        return ColUltraVBertConfig(vlm_config=self.vlm_config, embedding_dim=self.embedding_dim)

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_images, 3, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vlm_config.text_config.vocab_size)

        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_id
        attention_mask = input_ids.ne(1).to(torch_device)
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class ColUltraVBertForRetrievalModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (ColUltraVBertForRetrieval,) if is_torch_available() else ()

    test_resize_embeddings = True
    test_missing_keys = False
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = ColUltraVBertForRetrievalModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ColUltraVBertConfig, has_text_modality=False)

    @unittest.skip(reason="SDPA's flash backend doesn't support the mandatory sliding-window attention masks.")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @require_vision
    def test_colultravbert_forward_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)

            self.assertIsInstance(outputs, ColUltraVBertForRetrievalOutput)

    def test_embeddings_are_normalized_and_masked(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        attention_mask = torch.ones_like(inputs_dict["input_ids"])
        attention_mask[:, -2:] = 0
        inputs_dict["attention_mask"] = attention_mask
        model = ColUltraVBertForRetrieval(config).to(torch_device).eval()

        with torch.no_grad():
            embeddings = model(**inputs_dict).embeddings

        self.assertEqual(embeddings.shape[-1], self.model_tester.embedding_dim)
        norms = embeddings.norm(dim=-1)
        torch.testing.assert_close(norms[:, :-2], torch.ones_like(norms[:, :-2]))
        self.assertEqual(norms[:, -2:].abs().sum().item(), 0.0)


@require_torch
class ColUltraVBertModelIntegrationTest(unittest.TestCase):
    model_name: ClassVar[str] = "athrael-soju/ColUltraVBERT"
    revision: ClassVar[str] = "1af4d07cbbdfdf65b1babfa70980502103e4ed8d"

    def setUp(self):
        self.processor = ColUltraVBertProcessor.from_pretrained(self.model_name, revision=self.revision)
        self.model = (
            ColUltraVBertForRetrieval.from_pretrained(self.model_name, revision=self.revision, dtype=torch.float32)
            .to(torch_device)
            .eval()
        )

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_model_integration_test(self):
        queries = [
            "A paint on the wall",
            "ColModernVBERT matches the performance of models nearly 10x larger on visual document benchmarks.",
        ]
        images = [
            Image.open(hf_hub_download("HuggingFaceTB/SmolVLM", "example_images/rococo.jpg", repo_type="space")),
            Image.open(hf_hub_download("ModernVBERT/colmodernvbert", "table.png", repo_type="model")),
        ]

        batch_queries = self.processor.process_queries(text=queries).to(torch_device)
        batch_images = self.processor.process_images(images=images).to(torch_device)

        with torch.inference_mode():
            image_embeddings = self.model(**batch_images).embeddings
            query_embeddings = self.model(**batch_queries).embeddings

        scores = self.processor.score_retrieval(query_embeddings=query_embeddings, passage_embeddings=image_embeddings)

        self.assertEqual(scores.shape, (len(queries), len(images)))
        self.assertTrue((scores.argmax(dim=1) == torch.arange(len(images), device=scores.device)).all())

        # MaxSim scores from the released checkpoint; CPU and CUDA runs differ by up to 3e-3.
        expected_scores = torch.tensor([[9.2127, 4.0044], [10.1212, 21.3719]], dtype=scores.dtype)
        torch.testing.assert_close(scores, expected_scores, rtol=0, atol=1e-2)
