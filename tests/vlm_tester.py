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
            )
            if model_class is not None
        ]

    @property
    def pipeline_model_mapping(self):
        # This is the default pipeline mapping.
        mapping = {
            "feature-extraction": self.base_model_class,
            "image-text-to-text": self.conditional_generation_class,
        }
        return mapping

    def __init__(
        self,
        parent,
        batch_size=13,
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
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
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

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size], scale=1.0)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, pixel_values

    @property
    def config_args(self):
        return list(signature(self.config_class.__init__).parameters.keys())

    @property
    def text_config_args(self):
        return list(signature(self.text_config_class.__init__).parameters.keys())

    @property
    def vision_config_args(self):
        return list(signature(self.vision_config_class.__init__).parameters.keys())

    def get_config(self):
        kwargs = {}
        model_name_to_common_name = {v: k for k, v in self.config_class.attribute_map.items()}
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
        model_name_to_common_name = {v: k for k, v in self.text_config_class.attribute_map.items()}
        for k in self.text_config_args:
            if hasattr(self, k) and k != "self":
                kwargs[k] = getattr(self, k)
            elif k in model_name_to_common_name and hasattr(self, model_name_to_common_name[k]):
                kwargs[k] = getattr(self, model_name_to_common_name[k])
        return self.text_config_class(**kwargs)

    def get_vision_config(self):
        kwargs = {}
        model_name_to_common_name = {v: k for k, v in self.vision_config_class.attribute_map.items()}
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

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, token_type_ids, input_mask, pixel_values = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask, "pixel_values": pixel_values}
        if "image_sizes" in signature(self.base_model_class.forward).parameters:
            image_sizes = torch.tensor([[self.image_size, self.image_size]] * self.batch_size)
            inputs_dict["image_sizes"] = image_sizes
        return config, inputs_dict


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
        Tests that VLMs through an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompr has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)  # in=place modifications further
            _ = model(**curr_input_dict)  # successful forward with no modifications

            # remove one image but leave the image token in text
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-1:, ...]
            curr_input_dict["image_sizes"] = curr_input_dict["image_sizes"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:1]
            image_sizes = curr_input_dict["image_sizes"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(input_ids=input_ids, pixel_values=pixel_values, image_sizes=image_sizes)

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_sizes = torch.cat([image_sizes, image_sizes], dim=0)
            _ = model(input_ids=input_ids, pixel_values=pixel_values, image_sizes=image_sizes)

    @parameterized.expand(
        [
            (-1,),
            ([-1],),
            ([-1, -2],),
        ],
    )
    def test_vision_feature_layers(self, vision_feature_layer):
        """
        Test that we can use either one vision feature layer, or a list of
        vision feature layers.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.vision_feature_layer = vision_feature_layer

        num_feature_layers = 1 if isinstance(vision_feature_layer, int) else len(vision_feature_layer)
        hidden_size = config.vision_config.hidden_size
        expected_features = hidden_size * num_feature_layers

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            # We should have the right number of input features,
            # and should be able to run a forward pass without exploding
            base_model = getattr(model, "model", model)
            assert base_model.multi_modal_projector.linear_1.in_features == expected_features
            model(**input_dict)

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
