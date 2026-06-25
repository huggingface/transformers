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
import math
from inspect import signature

from parameterized import parameterized

from transformers import set_seed
from transformers.testing_utils import _TEXT_MODEL_TESTER_DEFAULTS

from .causal_lm_tester import _config_supports_rope_scaling, _set_config_rope_params
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


class MultiModalModelTester:
    """Shared tester base for VLM (vision-language) and ALM (audio-language) models.

    Concrete subclasses (e.g. `VLMModelTester`, `ALMModelTester`) supply:
      - the modality-specific sub-config class (`vision_config_class` for VLMs, `audio_config_class` for ALMs, ...),
      - the modality-specific defaults and helper methods,
      - the hooks `_build_modality_sub_configs` and `_prepare_modality_inputs`,
      - optionally an extended `_special_token_ids` and `pipeline_model_mapping`.

    This tester provides shared logic for evaluating and verifying models that combine text with other modalities,
    centering on the needs of vision-language (VLM) and audio-language (ALM) models.
    """

    # If the model follows the standard naming conventions, only `base_model_class` needs to be set
    # (the others are inferred from available public classes).
    base_model_class = None
    config_class = None
    text_config_class = None
    conditional_generation_class = None
    sequence_classification_class = None

    # Required attributes after the initialization phase of the tester. Subclasses extend.
    _required_attributes = ("config_class", "text_config_class", "conditional_generation_class")

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

    def __init__(self, parent, **kwargs):
        self.parent = parent

        # Multimodal-specific overrides of shared defaults (applied before the shared
        # defaults so they take precedence, but after any subclass setdefault calls).
        kwargs.setdefault("batch_size", 3)
        kwargs.setdefault("moe_intermediate_size", 12)

        # Apply shared text-model defaults for anything not already set.
        # Subclasses are expected to `setdefault` their modality-specific kwargs
        # (and any differing values such as `pad_token_id`) *before* calling super.
        for key, default in _TEXT_MODEL_TESTER_DEFAULTS.items():
            kwargs.setdefault(key, default)

        kwargs.setdefault("ignore_index", -100)
        kwargs.setdefault("scope", None)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._check_required_attributes()

    def _check_required_attributes(self):
        for required_attribute in self._required_attributes:
            if getattr(self, required_attribute, None) is None:
                raise ValueError(
                    f"You have inherited from {type(self).__name__} but did not set the {required_attribute} attribute."
                )

    # -- Overridable modality hooks -----------------------------------------------------------

    def create_attention_mask(self, input_ids):
        """Default causal (lower-triangular) attention mask. Override for bidirectional models like Gemma3."""
        return torch.tril(torch.ones_like(input_ids).to(torch_device))

    def get_additional_inputs(self, config, input_ids, modality_inputs):
        """Model-specific extra inputs (e.g. LlavaNext `image_sizes`, Qwen3VL `mm_token_type_ids`).

        ``modality_inputs`` is the full dict returned by ``_prepare_modality_inputs``.
        """
        return {}

    @property
    def _special_token_ids(self):
        """Special token ids that must never appear as random text tokens. Subclasses add modality tokens."""
        return {self.pad_token_id, self.bos_token_id, self.eos_token_id}

    def _build_modality_sub_configs(self):
        """Return the {sub-config-key: sub-config-instance} entries for the main config constructor."""
        raise NotImplementedError

    def _prepare_modality_inputs(self, input_ids, config):
        """Create modality features, place modality placeholder tokens in ``input_ids``, and return:

        (input_ids_with_placeholders, modality_inputs_dict)
        """
        raise NotImplementedError

    # -- End of overridable hooks -------------------------------------------------------------

    def _safe_token_id(self):
        """Smallest token ID that is not a special token. Used to scrub random ids_tensor outputs."""
        special_tokens = self._special_token_ids
        for i in range(self.vocab_size):
            if i not in special_tokens:
                return i
        raise ValueError("vocab_size is too small and there is no token ID that is not a special token!")

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        # Avoid flaky tests by scrubbing any accidental special tokens produced by ids_tensor.
        # Modality placeholder tokens are scrubbed and placed by `_prepare_modality_inputs`.
        safe_token_id = self._safe_token_id()
        for token_id in self._special_token_ids:
            input_ids[input_ids == token_id] = safe_token_id

        input_ids, modality_inputs = self._prepare_modality_inputs(input_ids, config)

        # Create attention mask with final input_ids (after modality placeholders are placed) — important
        # for models that derive padding from token values.
        attention_mask = self.create_attention_mask(input_ids) if self.use_input_mask else None

        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        inputs_dict.update(modality_inputs)
        inputs_dict.update(self.get_additional_inputs(config, input_ids, modality_inputs))
        return config, inputs_dict

    # -- Config construction helpers ----------------------------------------------------------

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

    def _collect_kwargs(self, sig_keys, config_class):
        """Collect kwargs for ``config_class`` by matching ``sig_keys`` (and its ``attribute_map``) against ``self``."""
        attribute_map = getattr(config_class, "attribute_map", {})
        model_name_to_common_name = {v: k for k, v in attribute_map.items()}
        kwargs = {}
        for k in sig_keys:
            if hasattr(self, k) and k != "self":
                kwargs[k] = getattr(self, k)
            elif k in model_name_to_common_name and hasattr(self, model_name_to_common_name[k]):
                kwargs[k] = getattr(self, model_name_to_common_name[k])
        return kwargs

    def get_config(self):
        kwargs = self._collect_kwargs(self.config_args + self.forced_config_args, self.config_class)
        kwargs["text_config"] = self.get_text_config()
        kwargs.update(self._build_modality_sub_configs())
        return self.config_class(**kwargs)

    def get_text_config(self):
        kwargs = self._collect_kwargs(self.text_config_args, self.text_config_class)
        return self.text_config_class(**kwargs)

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
class MultiModalModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin):
    """Shared test-class base for multimodal model families.

    Subclasses must set:
      - ``model_tester_class``: The tester class (subclass of ``MultiModalModelTester``)

    Optional:
      - ``all_model_classes``: override if not using the default from the model tester
      - ``pipeline_model_mapping``: override if not using the default from the model tester
    """

    model_tester_class = None
    all_model_classes = None
    pipeline_model_mapping = None

    # Multimodal models are always composite
    _is_composite = True

    def setUp(self):
        if self.model_tester_class is None:
            raise ValueError(
                f"You have inherited from {type(self).__name__} but did not set the model_tester_class attribute."
            )
        self.model_tester = self.model_tester_class(self)
        self.config_tester = ConfigTester(self, config_class=self.model_tester.config_class, has_text_modality=False)

        if self.pipeline_model_mapping is None:
            if self.all_model_classes is not None:
                raise ValueError(
                    f"Tests that inherit from `{type(self).__name__}` and set `all_model_classes` must manually set "
                    "`pipeline_model_mapping`."
                )
            else:
                self.pipeline_model_mapping = self.model_tester.pipeline_model_mapping

        if self.all_model_classes is None:
            self.all_model_classes = self.model_tester.all_model_classes

    def test_config(self):
        """Test config common functionality."""
        self.config_tester.run_common_tests()

    # RoPE tests are copied from `causal_lm_tester.py` with an only diff being "config -> text_config"
    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        """
        Tests that we can initialize a model with RoPE scaling in the config, that it can run a forward pass, and
        that a few basic model output properties are honored.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        text_config = config.get_text_config()

        if not _config_supports_rope_scaling(text_config):
            self.skipTest("This model does not support RoPE scaling")

        # Factor cannot be smaller than `int(2/head_dim)`, otherwise we'll end up dividing by zero!
        partial_rotary_factor = text_config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
        partial_rotary_factor = max(partial_rotary_factor, math.nextafter(2 / head_dim, 1.0))
        if int(head_dim * partial_rotary_factor) <= 2:
            partial_rotary_factor = math.ceil((3 / head_dim) / 0.05) * 0.05
        partial_rotary_factor = round(partial_rotary_factor, 10)
        text_config.partial_rotary_factor = None # override if present, so it doesn't interfere with test values 

        short_input = ids_tensor([1, 10], text_config.vocab_size)
        long_input = ids_tensor([1, int(text_config.max_position_embeddings * 1.5)], text_config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        _set_config_rope_params(
            text_config,
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
            text_config,
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
        text_config = config.get_text_config()

        if not _config_supports_rope_scaling(text_config):
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
        partial_rotary_factor = text_config.rope_parameters.get("partial_rotary_factor", 1.0)
        long_input_length = int(text_config.max_position_embeddings * 1.5)

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
            text_config,
            {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": partial_rotary_factor},
        )
        original_rope = rope_class(config=text_config, torch_device=torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short)
        original_cos_long, original_sin_long = original_rope(x, position_ids_long)
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        _set_config_rope_params(
            text_config,
            {
                "rope_type": "linear",
                "factor": scaling_factor,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        linear_scaling_rope = rope_class(config=text_config, torch_device=torch_device)
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
            text_config,
            {
                "rope_type": "dynamic",
                "factor": scaling_factor,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        ntk_scaling_rope = rope_class(config=text_config, torch_device=torch_device)
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
            text_config,
            {
                "rope_type": "yarn",
                "factor": scaling_factor,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        yarn_scaling_rope = rope_class(config=text_config, torch_device=torch_device)
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
