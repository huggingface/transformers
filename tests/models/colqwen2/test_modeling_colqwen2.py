# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch ColQwen2 model."""

import unittest
from typing import ClassVar

import torch
from datasets import load_dataset

from tests.test_configuration_common import ConfigTester
from tests.test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from transformers import is_torch_available
from transformers.models.colqwen2.configuration_colqwen2 import ColQwen2Config
from transformers.models.colqwen2.modeling_colqwen2 import ColQwen2ForRetrieval, ColQwen2ForRetrievalOutput
from transformers.models.colqwen2.processing_colqwen2 import ColQwen2Processor
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_bitsandbytes,
    require_torch,
    require_vision,
    slow,
    torch_device,
)


if is_torch_available():
    import torch


class ColQwen2ForRetrievalModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        pad_token_id=2,
        projector_hidden_act="gelu",
        seq_length=11,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        projection_dim=32,
        is_training=False,
        use_cache=False,
        vlm_config={
            "_name_or_path": "Qwen/Qwen2-VL-2B-Instruct",
            "bos_token_id": 0,
            "eos_token_id": 1,
            "vision_start_token_id": 3,
            "image_token_id": 4,
            "video_token_id": 5,
            "hidden_size": 64,
            "intermediate_size": 2,
            "max_window_layers": 2,
            "model_type": "qwen2_vl",
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_scaling": {"mrope_section": [4, 6, 6], "rope_type": "default", "type": "default"},
            "sliding_window": 32768,
            "tie_word_embeddings": True,
            "vision_config": {
                "depth": 2,
                "embed_dim": 32,
                "hidden_act": "quick_gelu",
                "hidden_size": 64,
                "mlp_ratio": 4,
                "num_heads": 4,
                "patch_size": 14,
                "in_chans": 3,
                "spatial_merge_size": 1,
                "temporal_patch_size": 2,
            },
            "vision_end_token_id": 151653,
            "vision_token_id": 151654,
            "vocab_size": 99,
        },
        embedding_dim=32,
        initializer_range=0.02,
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.pad_token_id = pad_token_id

        # `image_token_index` is set to 0 to pass "resize_embeddings" test, do not modify
        self.image_token_index = 0

        self.image_token_id = vlm_config["image_token_id"]
        self.video_token_id = vlm_config["video_token_id"]
        self.pad_token_id = vlm_config["eos_token_id"]
        self.vision_start_token_id = vlm_config["vision_start_token_id"]
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        self.image_size = 56
        self.num_image_tokens = 4

        self.seq_length = seq_length + self.num_image_tokens
        self.projection_dim = projection_dim

        self.num_hidden_layers = vlm_config["num_hidden_layers"]
        self.vocab_size = vlm_config["vocab_size"]
        self.hidden_size = vlm_config["hidden_size"]
        self.num_attention_heads = vlm_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = vlm_config["vision_config"]["in_chans"]

        self.encoder_seq_length = self.seq_length
        self.use_cache = use_cache

        self.vlm_config = vlm_config
        self.embedding_dim = embedding_dim
        self.initializer_range = initializer_range

    def get_config(self):
        return ColQwen2Config(
            vlm_config=self.vlm_config,
            embedding_dim=self.embedding_dim,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vlm_config.vision_config.patch_size
        temporal_patch_size = config.vlm_config.vision_config.temporal_patch_size

        # NOTE: Assume all inputs are square images of the same size.
        num_patches = (self.image_size // patch_size) ** 2
        pixel_values = floats_tensor(
            [
                self.batch_size * num_patches,
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )

        # Hardcoded image grid size: do not change unless you modified image size or patch size!
        image_grid_thw = torch.tensor([1, 4, 4]).repeat(self.batch_size, 1)

        # NOTE: The following adjustment ensures correct behavior with DDP on multiple GPUs.
        # Line is copied from `src/transformers/models/colqwen2/processing_colqwen2.py`
        offsets = image_grid_thw[:, 1] * image_grid_thw[:, 2]  # (batch_size,)
        pixel_values = list(
            torch.split(pixel_values, offsets.tolist())
        )  # [(num_patches_image_0, pixel_values), ..., (num_patches_image_n, pixel_values)]
        pixel_values = torch.nn.utils.rnn.pad_sequence(
            pixel_values, batch_first=True
        )  # (batch_size, max_num_patches, pixel_values)

        return config, pixel_values, image_grid_thw

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, image_grid_thw = config_and_inputs
        input_ids = (
            ids_tensor(
                shape=[self.batch_size, self.seq_length],
                vocab_size=config.vlm_config.vocab_size - 1,
            )
            + 1
        )
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[:, -1] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = self.image_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id

        inputs_dict = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }
        return config, inputs_dict


@require_torch
class ColQwen2ForRetrievalModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `ColQwen2ForRetrieval`.
    """

    all_model_classes = (ColQwen2ForRetrieval,) if is_torch_available() else ()
    fx_compatible = False
    test_torchscript = False
    test_pruning = False
    test_resize_embeddings = True
    test_head_masking = False

    def setUp(self):
        self.model_tester = ColQwen2ForRetrievalModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ColQwen2Config, has_text_modality=False)

    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)

            with torch.no_grad():
                model(**inputs)

    # overwrite inputs_embeds tests because we need to delete "pixel values" for LVLMs
    # while some other models require pixel_values to be present
    def test_inputs_embeds_matches_input_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)
            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            self.assertTrue(torch.allclose(out_embeds, out_ids))

    @slow
    @require_vision
    def test_colqwen2_forward_inputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)

            self.assertIsInstance(outputs, ColQwen2ForRetrievalOutput)

    @unittest.skip(reason="Some undefined behavior encountered with test versions of Qwen2-VL. Skip for now.")
    def test_model_parallelism(self):
        pass

    @unittest.skip(reason="Pass because ColQwen2 requires `attention_mask is not None`")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Pass because ColQwen2 requires `attention_mask is not None`")
    def test_sdpa_can_compile_dynamic(self):
        pass


@require_torch
class ColQwen2ModelIntegrationTest(unittest.TestCase):
    model_name: ClassVar[str] = "vidore/colqwen2-v1.0-hf"

    def setUp(self):
        self.processor = ColQwen2Processor.from_pretrained(self.model_name)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @require_bitsandbytes
    @slow
    def test_model_integration_test(self):
        """
        Test if the model is able to retrieve the correct pages for a small and easy dataset.
        """
        model = ColQwen2ForRetrieval.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            load_in_8bit=True,
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

        # Further validation: fine-grained check, with a hardcoded score from the original Hf implementation.
        expectations = Expectations(
            {
                ("cuda", 7): [
                    [15.0938, 8.3203, 15.0391],
                    [9.6328, 16.9062, 10.5312],
                    [15.6562, 12.2656, 20.2969],
                ],
                ("cuda", 8): [
                    [15.0703, 8.7422, 15.0312],
                    [9.5078, 16.8906, 10.6250],
                    [15.6484, 12.3984, 20.4688],
                ],
            }
        )
        expected_scores = torch.tensor(expectations.get_expectation(), dtype=scores.dtype)

        assert torch.allclose(scores, expected_scores, atol=1e-3), f"Expected scores {expected_scores}, got {scores}"
