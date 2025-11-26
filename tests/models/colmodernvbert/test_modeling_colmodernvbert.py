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

import collections
import gc
import re
import unittest
from typing import ClassVar

import pytest
from datasets import load_dataset

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

    from transformers.pytorch_utils import id_tensor_storage


class ColModernVBertForRetrievalModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=0,
        seq_length=25,
        projection_dim=32,
        text_config={
            "dtype": "float32",
            "hidden_size": 768,
            "intermediate_size": 1152,
            "mlp_bias": False,
            "model_type": "modernvbert_text",
            "num_hidden_layers": 22,
            "num_attention_heads": 12,
            "text_model_name": "jhu-clsp/ettin-encoder-150m",
            "vocab_size": 50368
        },
        is_training=True,
        vision_config={
            "dtype": "float32",
            "embed_dim": 768,
            "image_size": 512,
            "intermediate_size": 3072,
            "model_type": "modernvbert_vision",
            "num_hidden_layers": 12,
            "patch_size": 16,
            "vision_model_name": "google/siglip2-base-patch16-512"
        },
        use_cache=False,
        embedding_dim=128,
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        # `image_token_index` is set to 0 to pass "resize_embeddings" test, do not modify
        self.image_token_index = image_token_index
        self.pad_token_id = 1
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.projection_dim = projection_dim

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.is_training = is_training

        self.batch_size = 3
        self.image_size = vision_config["image_size"]
        self.encoder_seq_length = seq_length
        self.use_cache = use_cache

        self.embedding_dim = embedding_dim
        self.vlm_config = {
            "model_type": "modernvbert",
            "text_config": self.text_config,
            "vision_config": self.vision_config,
            "image_token_id": self.image_token_index,
            "vocab_size": self.vocab_size,
        }

    def get_config(self):
        return ColModernVBertConfig(
            vlm_config=self.vlm_config,
            embedding_dim=self.embedding_dim,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                1,
                3, # num_channels is not in vision_config for ModernVBertVisionConfig, assuming 3
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vlm_config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(torch_device)
        # set the 16 first tokens to be image, and ensure that no other tokens are image tokens
        # do not change this unless you modified image size or patch size
        input_ids[input_ids == config.vlm_config.image_token_id] = self.pad_token_id
        input_ids[:, :16] = config.vlm_config.image_token_id
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
            "token_type_ids": torch.zeros_like(input_ids),
        }
        return config, inputs_dict


@require_torch
class ColModernVBertForRetrievalModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `ColModernVBertForRetrieval`.
    """

    all_model_classes = (ColModernVBertForRetrieval,) if is_torch_available() else ()
    test_resize_embeddings = True
    additional_model_inputs = ["token_type_ids"]

    def setUp(self):
        self.model_tester = ColModernVBertForRetrievalModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ColModernVBertConfig, has_text_modality=False)

    @slow
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

    # ColModernVBert uses a VLM internally which has its state dict keys renames with `conversion_mapping`
    # This test is written assuming that `_tied_weights_keys` are not going to be renamed, thus we
    # overwrite it. NOTE: ColModernVBert inference/save/load works without issues, it is the testcase
    # that makes general assumptions
    def test_tied_weights_keys(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.vlm_config.tie_word_embeddings = True
        for model_class in self.all_model_classes:
            model_tied = model_class(config)

            ptrs = collections.defaultdict(list)
            for name, tensor in model_tied.state_dict().items():
                ptrs[id_tensor_storage(tensor)].append(name)

            # These are all the pointers of shared tensors.
            tied_params = [names for _, names in ptrs.items() if len(names) > 1]

            tied_weight_keys = model_tied._tied_weights_keys if model_tied._tied_weights_keys is not None else []
            # Detect we get a hit for each key
            for key in tied_weight_keys:
                key = key.replace(".language_model", "")  # remove 'language_model' prefix
                is_tied_key = any(re.search(key, p) for group in tied_params for p in group)
                self.assertTrue(is_tied_key, f"{key} is not a tied weight key for {model_class}.")

            # Removed tied weights found from tied params -> there should only be one left after
            for key in tied_weight_keys:
                key = key.replace(".language_model", "")  # remove 'language_model' prefix
                for i in range(len(tied_params)):
                    tied_params[i] = [p for p in tied_params[i] if re.search(key, p) is None]

            tied_params = [group for group in tied_params if len(group) > 1]
            self.assertListEqual(
                tied_params,
                [],
                f"Missing `_tied_weights_keys` for {model_class}: add all of {tied_params} except one.",
            )

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(
        reason="From ModernVBert: Some undefined behavior encountered with test versions of this model. Skip for now."
    )
    def test_model_parallelism(self):
        pass

    # TODO extend valid outputs to include this test @Molbap
    @unittest.skip(reason="ModernVBert has currently one output format.")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="Pass because ColModernVBert requires `attention_mask is not None`")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Pass because ColModernVBert requires `attention_mask is not None`")
    @pytest.mark.torch_compile_test
    def test_sdpa_can_compile_dynamic(self):
        pass


@require_torch
class ColModernVBertModelIntegrationTest(unittest.TestCase):
    model_name: ClassVar[str] = "vidore/colmodernvbert-v1.0-hf"

    def setUp(self):
        self.processor = ColModernVBertProcessor.from_pretrained(self.model_name)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    @slow
    @unittest.skip(reason="Model not yet available on HF Hub")
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
        ds = load_dataset("hf-internal-testing/document-visual-retrieval-test", split="test")

        # Preprocess the examples
        batch_images = self.processor(images=ds["image"]).to(torch_device)
        batch_queries = self.processor(text=ds["query"]).to(torch_device)

        # Run inference
        with torch.inference_mode():
            image_embeddings = model(**batch_images).embeddings
            query_embeddings = model(**batch_queries).embeddings

        # Compute retrieval scores
        scores = self.processor.score_retrieval(
            query_embeddings=query_embeddings,
            passage_embeddings=image_embeddings,
        )  # (num_queries, num_passages)

        assert scores.ndim == 2, f"Expected 2D tensor, got {scores.ndim}"
        assert scores.shape == (len(ds), len(ds)), f"Expected shape {(len(ds), len(ds))}, got {scores.shape}"

        # Check if the maximum scores per row are in the diagonal of the matrix score
        self.assertTrue((scores.argmax(axis=1) == torch.arange(len(ds), device=scores.device)).all())

        # Further validation: fine-grained check, with a hardcoded score from the original implementation
        # expected_scores = torch.tensor(
        #     [
        #         [15.5625, 6.5938, 14.4375],
        #         [12.2500, 16.2500, 11.0000],
        #         [15.0625, 11.7500, 21.0000],
        #     ],
        #     dtype=scores.dtype,
        # )

        # assert torch.allclose(scores, expected_scores, atol=1), f"Expected scores {expected_scores}, got {scores}"
