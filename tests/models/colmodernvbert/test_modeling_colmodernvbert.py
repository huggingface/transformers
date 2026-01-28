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
"""Testing suite for the PyTorch ColModernVBert model."""

import gc
import unittest
from typing import ClassVar

from huggingface_hub import hf_hub_download
from PIL import Image

from tests.test_configuration_common import ConfigTester
from tests.test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from transformers import (
    is_torch_available,
)
from transformers.models.colmodernvbert.configuration_colmodernvbert import ColModernVBertConfig
from transformers.models.colmodernvbert.modeling_colmodernvbert import (
    ColModernVBertForRetrieval,
    ColModernVBertForRetrievalOutput,
)
from transformers.models.colmodernvbert.processing_colmodernvbert import ColModernVBertProcessor
from transformers.testing_utils import (
    backend_empty_cache,
    require_torch,
    require_vision,
    slow,
    torch_device,
)


if is_torch_available():
    import torch


class ColModernVBertForRetrievalModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        num_images=2,
        ignore_index=-100,
        text_config={
            "vocab_size": 99,
            "pad_token_id": 0,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "hidden_activation": "gelu",
            "mlp_dropout": 0.1,
            "attention_dropout": 0.1,
            "embedding_dropout": 0.1,
            "classifier_dropout": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "is_decoder": False,
            "initializer_range": 0.02,
            "reference_compile": False,
        },
        is_training=False,
        vision_config={
            "image_size": 16,
            "patch_size": 4,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        pixel_shuffle_factor=2,
        embedding_dim=64,
    ):
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
        self.seq_length = (
            int(((vision_config["image_size"] // vision_config["patch_size"]) ** 2) / (pixel_shuffle_factor**2))
            * self.num_images
        )

        self.hidden_size = text_config["hidden_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.ignore_index = ignore_index

        self.embedding_dim = embedding_dim
        self.vlm_config = {
            "model_type": "modernvbert",
            "text_config": self.text_config,
            "vision_config": self.vision_config,
            "image_token_id": self.image_token_id,
            "pixel_shuffle_factor": self.pixel_shuffle_factor,
        }

    def get_config(self):
        return ColModernVBertConfig(
            vlm_config=self.vlm_config,
            embedding_dim=self.embedding_dim,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_images, 3, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vlm_config.text_config.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)

        # For simplicity just set the last n tokens to the image token
        n_image_tokens_per_batch = self.seq_length
        input_ids[:, -n_image_tokens_per_batch:] = self.image_token_id
        attention_mask = input_ids.ne(1).to(torch_device)
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class ColModernVBertForRetrievalModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `ColModernVBertForRetrieval`.
    """

    all_model_classes = (ColModernVBertForRetrieval,) if is_torch_available() else ()
    test_resize_embeddings = True

    def setUp(self):
        self.model_tester = ColModernVBertForRetrievalModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ColModernVBertConfig, has_text_modality=False)

    # @slow
    @require_vision
    def test_colmodernvbert_forward_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)

            self.assertIsInstance(outputs, ColModernVBertForRetrievalOutput)

    @unittest.skip(reason="Seems to fail due to device issues.")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Seems to fail due to device issues.")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Seems to fail due to device issues.")
    def test_disk_offload_safetensors(self):
        pass

    # @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    # def test_model_parallelism(self):
    #     pass

    @unittest.skip(reason="The test seems not to be compatible, tries to load the base model through the retrieval.")
    def test_correct_missing_keys(self):
        pass

    @unittest.skip(reason="Error related to ModernBERT model parallelism: self.dtype is broken.")
    def test_multi_gpu_data_parallel_forward(self):
        pass


@require_torch
class ColModernVBertModelIntegrationTest(unittest.TestCase):
    model_name: ClassVar[str] = "ModernVBERT/colmodernvbert-hf"

    def setUp(self):
        self.processor = ColModernVBertProcessor.from_pretrained(self.model_name)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    @slow
    @unittest.skip(reason="Model not available on HF for the moment.")
    def test_model_integration_test(self):
        """
        Test if the model is able to retrieve the correct pages for a small and easy dataset.
        """
        model = ColModernVBertForRetrieval.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map=torch_device,
        ).eval()

        # Load the test dataset
        queries = [
            "A paint on the wall",
            "ColModernVBERT matches the performance of models nearly 10x larger on visual document benchmarks.",
        ]

        images = [
            Image.open(hf_hub_download("HuggingFaceTB/SmolVLM", "example_images/rococo.jpg", repo_type="space")),
            Image.open(hf_hub_download("ModernVBERT/colmodernvbert", "table.png", repo_type="model")),
        ]

        # Preprocess the examples
        batch_images = self.processor(images=images).to(torch_device)
        batch_queries = self.processor(text=queries).to(torch_device)

        # Run inference
        with torch.inference_mode():
            image_embeddings = model(**batch_images).embeddings
            query_embeddings = model(**batch_queries).embeddings

        # Compute retrieval scores
        scores = self.processor.score_retrieval(
            query_embeddings=query_embeddings,
            passage_embeddings=image_embeddings,
        )  # (num_queries, num_passages)

        scores = torch.softmax(scores, dim=-1)

        assert scores.ndim == 2, f"Expected 2D tensor, got {scores.ndim}"
        assert scores.shape == (len(images), len(images)), (
            f"Expected shape {(len(images), len(images))}, got {scores.shape}"
        )

        # Check if the maximum scores per row are in the diagonal of the matrix score
        self.assertTrue((scores.argmax(axis=1) == torch.arange(len(images), device=scores.device)).all())

        # Further validation: fine-grained check, with a hardcoded score from the original implementation
        expected_scores = torch.tensor(
            [[0.95181, 0.048189], [0.00057251, 0.99943]],
            dtype=scores.dtype,
        )

        assert torch.allclose(scores, expected_scores, atol=1e-2), f"Expected scores {expected_scores}, got {scores}"
