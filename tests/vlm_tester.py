# Copyright 2025 HuggingFace Inc.
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

from parameterized import parameterized

from .test_configuration_common import ConfigTester
from .test_modeling_common import (
    GenerationTesterMixin,
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    is_torch_available,
    require_torch,
    torch_device,
)
from .test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch


class VLMModelTester:
    # If the model follows the standard naming conventions, only `base_model_class` needs to be set (the others are
    # inferred from available public classes).
    base_model_class = None
    config_class = None
    text_config_class = None
    vision_config_class = None
    conditional_generation_class = None
    sequence_classification_class = None
    # These attributes are required after the initialization phase of the tester.
    _required_attributes = ("base_model_class", "config_class", "conditional_generation_class")

    # Arguments that should be passed to the config class even if not in its signature
    forced_config_args = ["pad_token_id"]

    @property
    def all_model_classes(self):
        # Models that set `all_model_classes` in their `XXXModelTest` class must have a new class that doesn't fit
        # any of the common classes.
        return [
            model_class
            for model_class in (
                self.base_model_class,
                self.conditional_generation_class,
                self.sequence_classification_class,
            )
            if model_class is not None
        ]

    @property
    def pipeline_model_mapping(self):
        mapping = {
            "feature-extraction": self.base_model_class,
            "image-text-to-text": self.conditional_generation_class,
        }
        return mapping

    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        is_decoder=False,
        scope=None,
        expert_interval=1,
        moe_layer_start_index=0,
        moe_intermediate_size=12,
        shared_expert_intermediate_size=36,
        shared_expert_gate=True,
        moe_num_shared_experts=2,
        num_experts_per_tok=2,
        num_experts=8,
        mamba_n_groups=1,
        mamba_n_heads=16,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=16,
        image_size=8,
        patch_size=4,
        num_channels=3,
        projection_dim=32,
        projector_hidden_act="gelu",
        ignore_index=-100,
        image_token_index=0,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        num_image_tokens=32,
        **kwargs,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.scope = scope
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.is_decoder = is_decoder
        self.expert_interval = expert_interval
        self.moe_layer_start_index = moe_layer_start_index
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.shared_expert_gate = shared_expert_gate
        self.moe_num_shared_experts = moe_num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.mamba_n_groups = mamba_n_groups
        self.mamba_n_heads = mamba_n_heads
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_chunk_size = mamba_chunk_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.projection_dim = projection_dim
        self.projector_hidden_act = projector_hidden_act
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self._base_num_image_tokens = num_image_tokens
        self._base_seq_length = seq_length
        self.tie_word_embeddings = False

        for required_attribute in [
            "base_model_class",
            "config_class",
            "conditional_generation_class",
            "text_config_class",
            "vision_config_class",
        ]:
            if getattr(self, required_attribute) is None:
                raise ValueError(
                    f"You have inherited from VLMModelTester but did not set the {required_attribute} attribute."
                )

    # Because VLMs have some different standards in how they handle image tokens, we need a few methods
    # and properties that can be overridden if required:
    @property
    def num_image_tokens(self):
        return self._base_num_image_tokens

    @property
    def seq_length(self):
        return self._base_seq_length + self.num_image_tokens

    def create_pixel_values(self):
        # Override to 5D for patch-based models
        return floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size], scale=1.0)

    def create_attention_mask(self, input_ids):
        # Override for bidirectional attention models like Gemma3
        return torch.tril(torch.ones_like(input_ids).to(torch_device))

    def place_image_tokens(self, input_ids, config):
        # Override if the image tokens shouldn't be placed at the start of the test sequence
        image_token_index = getattr(config, "image_token_index", self.image_token_index)
        # Clear any accidental image tokens first
        input_ids = input_ids.clone()
        input_ids[input_ids == image_token_index] = self.bos_token_id
        # Place image tokens at the start
        input_ids[:, : self.num_image_tokens] = image_token_index
        return input_ids

    def get_additional_inputs(self, config, input_ids, pixel_values):
        # Override for model-specific inputs like LlavaNext's image_sizes
        return {}

    # End of overridable methods/properties

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = self.create_pixel_values()

        input_mask = None
        if self.use_input_mask:
            input_mask = self.create_attention_mask(input_ids)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, token_type_ids, input_mask, pixel_values = self.prepare_config_and_inputs()

        # Place image tokens in input_ids using template method
        input_ids = self.place_image_tokens(input_ids, config)

        # Recreate attention mask with final input_ids (after image tokens are placed)
        # This is important for models that use padding masks based on token values
        if self.use_input_mask:
            input_mask = self.create_attention_mask(input_ids)

        # Build base inputs dict
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask, "pixel_values": pixel_values}

        # Add model-specific additional inputs using template method
        additional_inputs = self.get_additional_inputs(config, input_ids, pixel_values)
        inputs_dict.update(additional_inputs)

        return config, inputs_dict

    @property
    def config_args(self):
        return list(signature(self.config_class.__init__).parameters.keys())

    @property
    def text_config_args(self):
        args = list(signature(self.text_config_class.__init__).parameters.keys())
        for token_arg in ["pad_token_id", "bos_token_id", "eos_token_id"]:  # Not always explicitly in the sig
            if token_arg not in args:
                args.append(token_arg)
        return args

    @property
    def vision_config_args(self):
        return list(signature(self.vision_config_class.__init__).parameters.keys())

    def get_config(self):
        kwargs = {}
        attribute_map = getattr(self.config_class, "attribute_map", {})
        model_name_to_common_name = {v: k for k, v in attribute_map.items()}
        for k in self.config_args + self.forced_config_args:
            if hasattr(self, k) and k != "self":
                kwargs[k] = getattr(self, k)
            elif k in model_name_to_common_name and hasattr(self, model_name_to_common_name[k]):
                kwargs[k] = getattr(self, model_name_to_common_name[k])
        kwargs["text_config"] = self.get_text_config()
        kwargs["vision_config"] = self.get_vision_config()
        return self.config_class(**kwargs)

    def get_text_config(self):
        kwargs = {}
        attribute_map = getattr(self.text_config_class, "attribute_map", {})
        model_name_to_common_name = {v: k for k, v in attribute_map.items()}
        for k in self.text_config_args:
            if hasattr(self, k) and k != "self":
                kwargs[k] = getattr(self, k)
            elif k in model_name_to_common_name and hasattr(self, model_name_to_common_name[k]):
                kwargs[k] = getattr(self, model_name_to_common_name[k])
        return self.text_config_class(**kwargs)

    def get_vision_config(self):
        kwargs = {}
        attribute_map = getattr(self.vision_config_class, "attribute_map", {})
        model_name_to_common_name = {v: k for k, v in attribute_map.items()}
        for k in self.vision_config_args:
            if hasattr(self, k) and k != "self":
                kwargs[k] = getattr(self, k)
            elif k in model_name_to_common_name and hasattr(self, model_name_to_common_name[k]):
                kwargs[k] = getattr(self, model_name_to_common_name[k])
        return self.vision_config_class(**kwargs)

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = self.base_model_class(config=config)
        model.to(torch_device)
        model.eval()
        model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))


@require_torch
class VLMModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin):
    """
    Base test class for Vision-Language Models.

    Subclasses should set:
    - `model_tester_class`: The tester class (subclass of VLMModelTester)

    Optional:
    - `all_model_classes`: Override if not using default from model_tester
    - `pipeline_model_mapping`: Override if not using default from model_tester
    """

    model_tester_class = None
    all_model_classes = None
    pipeline_model_mapping = None

    # VLMs are always composite
    _is_composite = True

    def setUp(self):
        if self.model_tester_class is None:
            raise ValueError("You have inherited from VLMModelTest but did not set the model_tester_class attribute.")
        self.model_tester = self.model_tester_class(self)
        self.config_tester = ConfigTester(self, config_class=self.model_tester.config_class, has_text_modality=False)

        if self.pipeline_model_mapping is None:
            if self.all_model_classes is not None:
                raise ValueError(
                    "Tests that inherit from `VLMModelTest` and set `all_model_classes` must manually set "
                    "`pipeline_model_mapping`."
                )
            else:
                self.pipeline_model_mapping = self.model_tester.pipeline_model_mapping

        if self.all_model_classes is None:
            self.all_model_classes = self.model_tester.all_model_classes

    def test_config(self):
        """Test config common functionality."""
        self.config_tester.run_common_tests()

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

    @parameterized.expand(
        [
            (-1,),
            ([-1],),
            ([-1, -2],),
        ],
    )
    @unittest.skip(reason="VLMs cannot pass input_embeds without input_ids")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="VLMs cannot pass input_embeds without input_ids")
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. "
        "Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass
