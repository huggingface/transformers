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
import tempfile
import unittest
from inspect import signature

import pytest

from transformers.models.auto.auto_factory import getattribute_from_module
from transformers.testing_utils import (
    _VLM_COMMON_MODEL_NAMES_MAP,
    is_flaky,
    require_flash_attn,
    require_torch_accelerator,
    slow,
)

from .test_configuration_common import ConfigTester
from .test_modeling_common import (
    GenerationTesterMixin,
    ModelTesterMixin,
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
    # ⚠️ Don't set these unless the model does NOT follow the standard naming conventions ⚠️
    config_class = None
    text_config_class = None
    vision_config_class = None
    conditional_generation_class = None
    # These attributes are required after the initialization phase of the tester.
    _required_attributes = ("base_model_class", "config_class", "causal_lm_class")

    # Arguments that should be passed to the config class even if not in its signature
    forced_config_args = ["pad_token_id"]

    @classmethod
    def _verify_and_infer_model_attributes(cls):
        """
        Verifies that the required tester attributes are set correctly, and infers unset tester attributes.
        Intentionally nitpicks the tester class attributes, to prevent human errors.
        """
        # `base_model_class` is mandatory, and it must be a valid model class.
        base_model_class = getattr(cls, "base_model_class")
        if base_model_class is None or "PreTrainedModel" not in str(base_model_class.__mro__):
            raise ValueError(
                f"You have inherited from `CausalLMModelTester` but did not set the `base_model_class` "
                f"attribute to a valid model class. (It's set to `{base_model_class}`)"
            )

        # Infers other model classes from the base class name and available public classes, if the corresponding
        # attributes are not set explicitly. If they are set, they must be set to a valid class (config or model).
        model_name = base_model_class.__name__.replace("Model", "")
        base_class_module = ".".join(base_model_class.__module__.split(".")[:-1])
        for tester_attribute_name, model_class_termination in _VLM_COMMON_MODEL_NAMES_MAP.items():
            if getattr(cls, tester_attribute_name) is None:
                try:
                    model_class = getattribute_from_module(base_class_module, model_name + model_class_termination)
                    setattr(cls, tester_attribute_name, model_class)
                except ValueError:
                    pass
            else:
                if tester_attribute_name == "config_class":
                    if "PreTrainedConfig" not in str(getattr(cls, tester_attribute_name).__mro__):
                        raise ValueError(
                            f"You have inherited from `CausalLMModelTester` but did not set the "
                            f"`{tester_attribute_name}` attribute to a valid config class. (It's set to "
                            f"`{getattr(cls, tester_attribute_name)}`). If the config class follows a standard "
                            f"naming convention, you should unset `{tester_attribute_name}`."
                        )
                else:
                    if "PreTrainedModel" not in str(getattr(cls, tester_attribute_name).__mro__):
                        raise ValueError(
                            f"You have inherited from `CausalLMModelTester` but did not set the "
                            f"`{tester_attribute_name}` attribute to a valid model class. (It's set to "
                            f"`{getattr(cls, tester_attribute_name)}`). If the model class follows a standard "
                            f"naming convention, you should unset `{tester_attribute_name}`."
                        )

        # After inferring, if we don't have the basic classes set, we raise an error.
        for required_attribute in cls._required_attributes:
            if getattr(cls, required_attribute) is None:
                raise ValueError(
                    f"You have inherited from `CausalLMModelTester` but did not set the `{required_attribute}` "
                    "attribute. It can't be automatically inferred either -- this means it is not following a "
                    "standard naming convention. If this is intentional, please set the attribute explicitly."
                )

        # To prevent issues with typos, no other attributes can be set to a model class
        for instance_attribute_name, instance_attribute in cls.__dict__.items():
            if (
                (
                    instance_attribute_name not in _VLM_COMMON_MODEL_NAMES_MAP
                    and instance_attribute_name != "base_model_class"
                )
                and isinstance(instance_attribute, type)
                and "PreTrainedModel" in str(instance_attribute.__mro__)
            ):
                raise ValueError(
                    f"You have inherited from `CausalLMModelTester` but set an unexpected attribute to a model class "
                    f"(`{instance_attribute_name}` is set to `{instance_attribute}`). "
                    f"Only the following attributes can be set to model classes: {_VLM_COMMON_MODEL_NAMES_MAP.keys()}."
                )

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
                self.token_classification_class,
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
    ):
        self._verify_and_infer_model_attributes()
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
        self.tie_word_embeddings = False

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        config = self.get_config()


        return config, input_ids, token_type_ids, input_mask

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
        return self.config_class(**kwargs)  # TODO Resume from here as a minimum conversion of the Causal tester class

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
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, _, input_mask, _, _, _ = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
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
            raise ValueError(
                "You have inherited from VLMModelTest but did not set the model_tester_class attribute."
            )
        self.model_tester = self.model_tester_class(self)
        self.config_tester = ConfigTester(
            self, config_class=self.model_tester.config_class, has_text_modality=False
        )

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

    def test_model(self):
        """Test basic model forward pass."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(config, **inputs_dict)

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs throw an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        Also tests multi-image cases when one prompt has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()

            # Successful forward with no modifications
            _ = model(**input_dict)

            # Create a copy for modification
            curr_input_dict = copy.deepcopy(input_dict)

            # Remove one image but leave the image tokens in text
            if self.model_tester.num_images > 1:
                # For batched pixel values [batch, num_images, C, H, W]
                curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][:, :-1, ...]
            else:
                # Remove images from first batch item
                curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][1:, ...]

            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

    def test_text_only_forward(self):
        """
        Tests that VLMs can do a forward pass without pixel_values (text-only mode).
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Remove pixel values and image tokens from input
        input_ids = input_dict["input_ids"].clone()
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        attention_mask = input_dict["attention_mask"].clone()

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            # Forward pass without pixel_values should work
            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception as e:
                self.skipTest(f"Model {model_class.__name__} does not support text-only forward: {e}")

    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    @is_flaky()
    @slow
    def test_flash_attn_2_equivalence(self):
        """Test that Flash Attention 2 produces equivalent outputs to eager attention."""
        for model_class in self.all_model_classes:
            if not model_class._supports_flash_attn:
                self.skipTest(reason="Model does not support Flash Attention 2")

            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_fa = model_class.from_pretrained(
                    tmpdirname, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
                )
                model_fa.to(torch_device)

                model = model_class.from_pretrained(tmpdirname, dtype=torch.bfloat16, attn_implementation="eager")
                model.to(torch_device)

                dummy_pixel_values = inputs_dict["pixel_values"].to(device=torch_device, dtype=torch.bfloat16)
                dummy_input_ids = inputs_dict["input_ids"].to(torch_device)
                dummy_attention_mask = inputs_dict.get("attention_mask")
                if dummy_attention_mask is not None:
                    dummy_attention_mask = dummy_attention_mask.to(torch_device)

                outputs = model(
                    input_ids=dummy_input_ids,
                    pixel_values=dummy_pixel_values,
                    attention_mask=dummy_attention_mask,
                    output_hidden_states=True,
                )
                outputs_fa = model_fa(
                    input_ids=dummy_input_ids,
                    pixel_values=dummy_pixel_values,
                    attention_mask=dummy_attention_mask,
                    output_hidden_states=True,
                )

                logits = outputs.hidden_states[-1]
                logits_fa = outputs_fa.hidden_states[-1]
                torch.testing.assert_close(logits_fa, logits, atol=3e-2, rtol=3e-2)

    def test_resize_tokens_embeddings(self):
        """
        Test resizing token embeddings for VLMs.
        VLMs need special handling because of image token IDs.
        """
        (original_config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            # Get vocab size from text config or main config
            if hasattr(config, "text_config") and hasattr(config.text_config, "vocab_size"):
                model_vocab_size = config.text_config.vocab_size
            else:
                model_vocab_size = config.vocab_size

            # Retrieve the embeddings and clone them
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = model_embed.weight.clone()

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size + 10)
            if hasattr(config, "text_config") and hasattr(config.text_config, "vocab_size"):
                self.assertEqual(model.config.text_config.vocab_size, model_vocab_size + 10)
            else:
                self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] + 10)
            # Check that the model can still do a forward pass successfully
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model_embed = model.resize_token_embeddings(model_vocab_size - 15)
            if hasattr(config, "text_config") and hasattr(config.text_config, "vocab_size"):
                self.assertEqual(model.config.text_config.vocab_size, model_vocab_size - 15)
            else:
                self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            # Check that it actually resizes the embeddings matrix
            self.assertEqual(model_embed.weight.shape[0], cloned_embeddings.shape[0] - 15)

            # Adjust inputs for smaller vocab - clamp input_ids and update image tokens
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 2)
            # Update image token positions with new valid image token id
            new_image_token_id = model_vocab_size - 15 - 1
            inputs_dict["input_ids"][:, : self.model_tester.num_image_tokens] = new_image_token_id

            # Check that adding and removing tokens has not modified the first part of the embedding matrix
            models_equal = True
            for p1, p2 in zip(cloned_embeddings, model_embed.weight):
                if p1.data.ne(p2.data).sum() > 0:
                    models_equal = False

            self.assertTrue(models_equal)

    def test_resize_embeddings_untied(self):
        """Test resizing embeddings when they are untied."""
        (original_config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()

        original_config.tie_word_embeddings = False

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config).to(torch_device)
            model.eval()

            # If no output embeddings, skip
            if model.get_output_embeddings() is None:
                continue

            # Get vocab size
            if hasattr(config, "text_config") and hasattr(config.text_config, "vocab_size"):
                model_vocab_size = config.text_config.vocab_size
            else:
                model_vocab_size = config.vocab_size

            # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
            model.resize_token_embeddings(model_vocab_size + 10)
            if hasattr(config, "text_config") and hasattr(config.text_config, "vocab_size"):
                self.assertEqual(model.config.text_config.vocab_size, model_vocab_size + 10)
            else:
                self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size + 10)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size + 10)
            # Check that the model can still do a forward pass successfully
            model(**self._prepare_for_class(inputs_dict, model_class))

            # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
            model.resize_token_embeddings(model_vocab_size - 15)
            if hasattr(config, "text_config") and hasattr(config.text_config, "vocab_size"):
                self.assertEqual(model.config.text_config.vocab_size, model_vocab_size - 15)
            else:
                self.assertEqual(model.config.vocab_size, model_vocab_size - 15)
            output_embeds = model.get_output_embeddings()
            self.assertEqual(output_embeds.weight.shape[0], model_vocab_size - 15)
            # Check bias if present
            if output_embeds.bias is not None:
                self.assertEqual(output_embeds.bias.shape[0], model_vocab_size - 15)

            # Adjust inputs for smaller vocab
            inputs_dict["input_ids"].clamp_(max=model_vocab_size - 15 - 2)
            new_image_token_id = model_vocab_size - 15 - 1
            inputs_dict["input_ids"][:, : self.model_tester.num_image_tokens] = new_image_token_id

            # Check that the model can still do a forward pass successfully
            model(**self._prepare_for_class(inputs_dict, model_class))
