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
"""Testing suite for the PyTorch ColPali model."""

import collections
import gc
import re
import unittest
from typing import ClassVar

import torch
from datasets import load_dataset

from tests.test_configuration_common import ConfigTester
from tests.test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from transformers import (
    is_torch_available,
)
from transformers.models.colpali.configuration_colpali import ColPaliConfig
from transformers.models.colpali.modeling_colpali import ColPaliForRetrieval, ColPaliForRetrievalOutput
from transformers.models.colpali.processing_colpali import ColPaliProcessor
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


class ColPaliForRetrievalModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=0,
        projector_hidden_act="gelu",
        seq_length=25,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        projection_dim=32,
        text_config={
            "model_type": "gemma",
            "seq_length": 128,
            "is_training": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "intermediate_size": 37,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 1,
        },
        is_training=False,
        vision_config={
            "use_labels": True,
            "image_size": 20,
            "patch_size": 5,
            "num_image_tokens": 4,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        use_cache=False,
        embedding_dim=128,
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        # `image_token_index` is set to 0 to pass "resize_embeddings" test, do not modify
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.projection_dim = projection_dim
        self.pad_token_id = text_config["pad_token_id"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]
        self.encoder_seq_length = seq_length
        self.use_cache = use_cache

        self.embedding_dim = embedding_dim
        self.vlm_config = {
            "model_type": "paligemma",
            "text_config": self.text_config,
            "vision_config": self.vision_config,
            "ignore_index": self.ignore_index,
            "image_token_index": self.image_token_index,
            "projector_hidden_act": self.projector_hidden_act,
            "projection_dim": self.projection_dim,
            "vision_feature_select_strategy": self.vision_feature_select_strategy,
            "vision_feature_layer": self.vision_feature_layer,
        }

    def get_config(self):
        return ColPaliConfig(
            vlm_config=self.vlm_config,
            embedding_dim=self.embedding_dim,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["num_channels"],
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
        input_ids[input_ids == config.vlm_config.image_token_index] = self.pad_token_id
        input_ids[:, :16] = config.vlm_config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }
        return config, inputs_dict


@require_torch
class ColPaliForRetrievalModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `ColPaliForRetrieval`.
    """

    all_model_classes = (ColPaliForRetrieval,) if is_torch_available() else ()
    fx_compatible = False
    test_torchscript = False
    test_pruning = False
    test_resize_embeddings = True
    test_head_masking = False

    def setUp(self):
        self.model_tester = ColPaliForRetrievalModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ColPaliConfig, has_text_modality=False)

    @slow
    @require_vision
    def test_colpali_forward_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)

            self.assertIsInstance(outputs, ColPaliForRetrievalOutput)

    # ColPali uses a VLM internally which has its state dict keys renames with `conversion_mapping`
    # This test is written assuming that `_tied_weights_keys` are not going to be renamed, thus we
    # overwrite it. NOTE: ColPali inference/save/load works without issues, it is the testcase
    # that makes general assumptions
    def test_tied_weights_keys(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.get_text_config().tie_word_embeddings = True
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
        reason="From PaliGemma: Some undefined behavior encountered with test versions of this model. Skip for now."
    )
    def test_model_parallelism(self):
        pass

    @unittest.skip(
        reason="PaliGemma's SigLip encoder uses the same initialization scheme as the Flax original implementation"
    )
    def test_initialization(self):
        pass

    # TODO extend valid outputs to include this test @Molbap
    @unittest.skip(reason="PaliGemma has currently one output format.")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="Pass because ColPali requires `attention_mask is not None`")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Pass because ColPali requires `attention_mask is not None`")
    def test_sdpa_can_compile_dynamic(self):
        pass


@require_torch
class ColPaliModelIntegrationTest(unittest.TestCase):
    model_name: ClassVar[str] = "vidore/colpali-v1.2-hf"

    def setUp(self):
        self.processor = ColPaliProcessor.from_pretrained(self.model_name)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)

    @slow
    def test_model_integration_test(self):
        """
        Test if the model is able to retrieve the correct pages for a small and easy dataset.
        """
        model = ColPaliForRetrieval.from_pretrained(
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
        expected_scores = torch.tensor(
            [
                [15.5625, 6.5938, 14.4375],
                [12.2500, 16.2500, 11.0000],
                [15.0625, 11.7500, 21.0000],
            ],
            dtype=scores.dtype,
        )

        assert torch.allclose(scores, expected_scores, atol=1), f"Expected scores {expected_scores}, got {scores}"
