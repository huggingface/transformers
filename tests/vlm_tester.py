# Copyright 2026 HuggingFace Inc.
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

import copy
import unittest
from inspect import signature

from .multimodal_tester import MultiModalModelTest, MultiModalModelTester
from .test_modeling_common import (
    floats_tensor,
    is_torch_available,
    torch_device,
)


if is_torch_available():
    import torch


class VLMModelTester(MultiModalModelTester):
    vision_config_class = None
    _required_attributes = MultiModalModelTester._required_attributes + ("base_model_class", "vision_config_class")

    @property
    def pipeline_model_mapping(self):
        return {
            "feature-extraction": self.base_model_class,
            "image-text-to-text": self.conditional_generation_class,
        }

    def __init__(self, parent, **kwargs):
        # Overrides of _TEXT_MODEL_TESTER_DEFAULTS
        kwargs.setdefault(
            "seq_length",
            7
            + kwargs.get(
                "num_image_tokens",
                (kwargs.get("image_size", 8) // kwargs.get("patch_size", 4)) ** 2,
            ),
        )
        kwargs.setdefault("pad_token_id", 0)

        # VLM-specific defaults
        kwargs.setdefault("use_token_type_ids", False)
        kwargs.setdefault("hidden_dropout_prob", 0.1)
        kwargs.setdefault("attention_probs_dropout_prob", 0.1)
        kwargs.setdefault("type_vocab_size", 16)
        kwargs.setdefault("type_sequence_label_size", 2)
        kwargs.setdefault("initializer_range", 0.02)
        kwargs.setdefault("num_labels", 3)
        kwargs.setdefault("num_choices", 4)
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("is_decoder", False)
        kwargs.setdefault("image_size", 8)
        kwargs.setdefault("patch_size", 4)
        kwargs.setdefault("num_channels", 3)
        kwargs.setdefault("projection_dim", 32)
        kwargs.setdefault("projector_hidden_act", "gelu")
        kwargs.setdefault("vision_feature_select_strategy", "default")
        kwargs.setdefault("vision_feature_layer", -1)
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("num_image_tokens", (kwargs["image_size"] // kwargs["patch_size"]) ** 2)

        super().__init__(parent, **kwargs)

        # Computed default depending on base-class defaults for hidden_size / num_attention_heads.
        if not hasattr(self, "head_dim"):
            self.head_dim = self.hidden_size // self.num_attention_heads

    # -- Overridable VLM-specific hooks ------------------------------------------------------

    def create_pixel_values(self):
        # Override to 5D for patch-based models
        return floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size], scale=1.0)

    def place_image_tokens(self, input_ids, config):
        # Override if the image tokens shouldn't be placed at the start of the test sequence
        image_token_id = getattr(config, "image_token_id", self.image_token_id)
        # Clear any accidental image tokens first
        input_ids = input_ids.clone()
        input_ids[input_ids == image_token_id] = self.bos_token_id
        # Place image tokens at the start
        input_ids[:, : self.num_image_tokens] = image_token_id
        return input_ids

    # -- Hooks consumed by the shared base ---------------------------------------------------

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.image_token_id}

    def _build_modality_sub_configs(self):
        return {"vision_config": self.get_vision_config()}

    def _prepare_modality_inputs(self, input_ids, config):
        pixel_values = self.create_pixel_values()
        input_ids = self.place_image_tokens(input_ids, config)
        return input_ids, {"pixel_values": pixel_values}

    # -- Vision sub-config construction ------------------------------------------------------

    @property
    def vision_config_args(self):
        return list(signature(self.vision_config_class.__init__).parameters.keys())

    def get_vision_config(self):
        kwargs = self._collect_kwargs(self.vision_config_args, self.vision_config_class)
        return self.vision_config_class(**kwargs)


class VLMModelTest(MultiModalModelTest):
    """
    Base test class for Vision-Language Models.

    Subclasses should set:
    - `model_tester_class`: The tester class (subclass of VLMModelTester)

    Optional:
    - `all_model_classes`: Override if not using default from model_tester
    - `pipeline_model_mapping`: Override if not using default from model_tester
    """

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs throw an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompt has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)
            _ = model(**curr_input_dict)  # successful forward with no modifications

            # Test 1: remove one image but leave the image token in text
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-1:, ...]
            if "image_sizes" in curr_input_dict:
                curr_input_dict["image_sizes"] = curr_input_dict["image_sizes"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 2: simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            # First, take just the first item from each tensor
            curr_input_dict = {key: val[:1] for key, val in curr_input_dict.items()}

            # Double the batch size for all batch-dimension tensors except pixel_values
            # This simulates having 2 prompts (each with image tokens) but only 1 image
            batch_tensors_to_double = ["input_ids", "attention_mask", "token_type_ids"]
            for key in batch_tensors_to_double:
                if key in curr_input_dict and curr_input_dict[key] is not None:
                    curr_input_dict[key] = torch.cat([curr_input_dict[key], curr_input_dict[key]], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 3: two images and two image tokens don't raise an error
            curr_input_dict["pixel_values"] = torch.cat(
                [curr_input_dict["pixel_values"], curr_input_dict["pixel_values"]], dim=0
            )
            if "image_sizes" in curr_input_dict:
                curr_input_dict["image_sizes"] = torch.cat(
                    [curr_input_dict["image_sizes"], curr_input_dict["image_sizes"]], dim=0
                )
            _ = model(**curr_input_dict)

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. "
        "Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass
