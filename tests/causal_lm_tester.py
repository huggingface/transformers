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

import tempfile
from inspect import signature

import pytest
from parameterized import parameterized

from transformers import AutoModelForCausalLM, PreTrainedConfig, set_seed
from transformers.models.auto.auto_factory import getattribute_from_module
from transformers.testing_utils import (
    _COMMON_MODEL_NAMES_MAP,
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


class CausalLMModelTester:
    # If the model follows the standard naming conventions, only `base_model_class` needs to be set (the others are
    # inferred from available public classes).
    base_model_class = None
    # ⚠️ Don't set these unless the model does NOT follow the standard naming conventions ⚠️
    config_class = None
    causal_lm_class = None
    question_answering_class = None
    sequence_classification_class = None
    token_classification_class = None
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
        for tester_attribute_name, model_class_termination in _COMMON_MODEL_NAMES_MAP.items():
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
                    instance_attribute_name not in _COMMON_MODEL_NAMES_MAP
                    and instance_attribute_name != "base_model_class"
                )
                and isinstance(instance_attribute, type)
                and "PreTrainedModel" in str(instance_attribute.__mro__)
            ):
                raise ValueError(
                    f"You have inherited from `CausalLMModelTester` but set an unexpected attribute to a model class "
                    f"(`{instance_attribute_name}` is set to `{instance_attribute}`). "
                    f"Only the following attributes can be set to model classes: {_COMMON_MODEL_NAMES_MAP.keys()}."
                )

    @property
    def all_model_classes(self):
        # Models that set `all_model_classes` in their `XXXModelTest` class must have a new class that doesn't fit
        # any of the common classes.
        return [
            model_class
            for model_class in (
                self.base_model_class,
                self.causal_lm_class,
                self.question_answering_class,
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
            "text-generation": self.causal_lm_class,
        }
        if self.question_answering_class is not None:
            mapping["question-answering"] = self.question_answering_class
        if self.sequence_classification_class is not None:
            mapping["text-classification"] = self.sequence_classification_class
        if self.token_classification_class is not None:
            mapping["token-classification"] = self.token_classification_class
        if self.sequence_classification_class is not None:
            mapping["zero-shot"] = self.sequence_classification_class
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

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    @property
    def config_args(self):
        return list(signature(self.config_class.__init__).parameters.keys())

    def get_config(self):
        kwargs = {}
        model_name_to_common_name = {v: k for k, v in self.config_class.attribute_map.items()}
        for k in self.config_args + self.forced_config_args:
            if hasattr(self, k) and k != "self":
                kwargs[k] = getattr(self, k)
            elif k in model_name_to_common_name and hasattr(self, model_name_to_common_name[k]):
                kwargs[k] = getattr(self, model_name_to_common_name[k])
        return self.config_class(**kwargs)

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
class CausalLMModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin):
    model_tester_class = None
    all_model_classes = None
    pipeline_model_mapping = None

    def setUp(self):
        if self.model_tester_class is None:
            raise ValueError(
                "You have inherited from CausalLMModelTest but did not set the model_tester_class attribute."
            )
        self.model_tester = self.model_tester_class(self)
        self.config_tester = ConfigTester(self, config_class=self.model_tester.config_class)

        if self.pipeline_model_mapping is None:
            # If `all_model_classes` is not the default, maybe there are more pipeline mappings to be set.
            if self.all_model_classes is not None:
                raise ValueError(
                    "Testes that inherit from `CausalLMModelTest` and set `all_model_classes` must manually set "
                    "`pipeline_model_mapping`."
                )
            # Otherwise, we know the pipeline mapping is the default.
            else:
                self.pipeline_model_mapping = self.model_tester.pipeline_model_mapping

        if self.all_model_classes is None:
            self.all_model_classes = self.model_tester.all_model_classes

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_sequence_classification_model(self):
        if self.model_tester.sequence_classification_class is None:
            self.skipTest("Model does not support sequence classification")
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = self.model_tester.sequence_classification_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_sequence_classification_model_for_single_label(self):
        if self.model_tester.sequence_classification_class is None:
            self.skipTest("Model does not support sequence classification")
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "single_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor([self.model_tester.batch_size], self.model_tester.type_sequence_label_size)
        model = self.model_tester.sequence_classification_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_sequence_classification_model_for_multi_label(self):
        if self.model_tester.sequence_classification_class is None:
            self.skipTest("Model does not support sequence classification")
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        config.problem_type = "multi_label_classification"
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        sequence_labels = ids_tensor(
            [self.model_tester.batch_size, config.num_labels], self.model_tester.type_sequence_label_size
        ).to(torch.float)
        model = self.model_tester.sequence_classification_class(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=sequence_labels)
        self.assertEqual(result.logits.shape, (self.model_tester.batch_size, self.model_tester.num_labels))

    def test_token_classification_model(self):
        if self.model_tester.token_classification_class is None:
            self.skipTest("Model does not support token classification")
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3
        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        token_labels = ids_tensor([self.model_tester.batch_size, self.model_tester.seq_length], config.num_labels)
        model = self.model_tester.token_classification_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask, labels=token_labels)
        self.assertEqual(
            result.logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.num_labels),
        )

    def test_question_answering_model(self):
        if self.model_tester.question_answering_class is None:
            self.skipTest("Model does not support question answering")
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.num_labels = 3

        input_ids = input_dict["input_ids"]
        attention_mask = input_ids.ne(1).to(torch_device)
        model = self.model_tester.question_answering_class(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=attention_mask)
        self.assertEqual(
            result.start_logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length),
        )
        self.assertEqual(
            result.end_logits.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length),
        )

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        """
        Tests that we can initialize a model with RoPE scaling in the config, that it can run a forward pass, and
        that a few basic model output properties are honored.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        if not _config_supports_rope_scaling(config):
            self.skipTest("This model does not support RoPE scaling")

        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 1.5)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        _set_config_rope_params(
            config,
            {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
                "original_max_position_embeddings": 16384,
            },
        )
        original_model = self.model_tester_class.base_model_class(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        _set_config_rope_params(
            config,
            {
                "rope_type": scaling_type,
                "factor": 10.0,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        scaled_model = self.model_tester_class.base_model_class(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            torch.testing.assert_close(original_short_output, scaled_short_output, rtol=1e-5, atol=1e-5)
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    def test_model_rope_scaling_frequencies(self):
        """Tests the frequency properties of the different RoPE scaling types on the model RoPE layer."""
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        if not _config_supports_rope_scaling(config):
            self.skipTest("This model does not support RoPE scaling")

        # Retrieves the RoPE layer class from the base model class. Uses `.named_modules()` to avoid hardcoding the
        # named location of the RoPE layer class.
        base_model = self.model_tester.base_model_class(config)
        possible_rope_attributes = [
            "pos_emb",
            "rotary_emb",  # most common case
            "global_rotary_emb",
            "local_rotary_emb",
        ]
        for name, module in base_model.named_modules():
            if any(potential_name in name for potential_name in possible_rope_attributes):
                rope_class = type(module)
                break

        scaling_factor = 10
        short_input_length = 10
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        long_input_length = int(config.max_position_embeddings * 1.5)

        # Inputs
        x = torch.randn(
            1, dtype=torch.float32, device=torch_device
        )  # used exclusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Sanity check original RoPE
        _set_config_rope_params(
            config, {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": partial_rotary_factor}
        )
        original_rope = rope_class(config=config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short)
        original_cos_long, original_sin_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        _set_config_rope_params(
            config,
            {
                "rope_type": "linear",
                "factor": scaling_factor,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        linear_scaling_rope = rope_class(config=config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short)
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        _set_config_rope_params(
            config,
            {
                "rope_type": "dynamic",
                "factor": scaling_factor,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        ntk_scaling_rope = rope_class(config=config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short)
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq).all())

        # Sanity check Yarn RoPE scaling
        # Scaling should be over the entire input
        _set_config_rope_params(
            config,
            {
                "rope_type": "yarn",
                "factor": scaling_factor,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        yarn_scaling_rope = rope_class(config=config).to(torch_device)
        yarn_cos_short, yarn_sin_short = yarn_scaling_rope(x, position_ids_short)
        yarn_cos_long, yarn_sin_long = yarn_scaling_rope(x, position_ids_long)
        torch.testing.assert_close(yarn_cos_short, yarn_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(yarn_sin_short, yarn_sin_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_short, original_cos_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_long, original_sin_long)

    @require_flash_attn
    @require_torch_accelerator
    @pytest.mark.flash_attn_test
    @is_flaky()
    @slow
    def test_flash_attn_2_equivalence(self):
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

                dummy_input = inputs_dict[model_class.main_input_name]
                dummy_input = dummy_input.to(torch_device)
                outputs = model(dummy_input, output_hidden_states=True)
                outputs_fa = model_fa(dummy_input, output_hidden_states=True)

                logits = outputs.hidden_states[-1]
                logits_fa = outputs_fa.hidden_states[-1]
                torch.testing.assert_close(logits_fa, logits, atol=3e-2, rtol=3e-2)

    def test_causal_lm_can_accept_training_kwargs(self):
        if not getattr(self.model_tester, "is_training", False):
            self.skipTest(reason="ModelTester is not configured to run training tests")

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        with tempfile.TemporaryDirectory() as tmpdir:
            with torch.device(torch_device):
                model_eager = AutoModelForCausalLM.from_config(config, dtype=torch.float32)

            model_eager.save_pretrained(tmpdir)
            model = AutoModelForCausalLM.from_pretrained(tmpdir, dtype=torch.float32, device_map=torch_device)
            inputs_dict["num_items_in_batch"] = torch.tensor(inputs_dict["input_ids"].shape[0])
            inputs_dict["labels"] = inputs_dict["input_ids"]
            _ = model(**inputs_dict, return_dict=False)


def _config_supports_rope_scaling(config: PreTrainedConfig) -> bool:
    """Returns whether a certain model config supports RoPE scaling parameterization."""
    # Has rope_scaling -> model was designed with rope scaling in mind
    # Has rope_theta (and no rope_scaling) -> probably an older model, but should support rope scaling as well
    main_config_has_rope = hasattr(config, "rope_parameters")
    sub_config_has_rope = any(
        hasattr(getattr(config, sub_config), "rope_parameters") for sub_config in config.sub_configs.keys()
    )
    return main_config_has_rope or sub_config_has_rope


def _set_config_rope_params(config: PreTrainedConfig, rope_params: dict) -> bool:
    """Recursively sets RoPE parameters on configs and subconfigs, by duplicating the same RoPE values."""
    config.rope_parameters = getattr(config, "rope_parameters", {}) or {}
    config.rope_parameters.update(rope_params)

    if any(name in config.__class__.__name__.lower() for name in ["gemma3", "modernbert"]):
        config.rope_parameters = {layer_type: config.rope_parameters.copy() for layer_type in config.layer_types}

    for sub_config in config.sub_configs.keys():
        _set_config_rope_params(getattr(config, sub_config), rope_params)
    return config
