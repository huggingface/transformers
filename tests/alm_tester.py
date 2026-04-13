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

import tempfile
import unittest
from inspect import signature

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


class ALMModelTester:
    # If the model follows standard naming conventions, only `config_class` and
    # `conditional_generation_class` need to be set (others are optional).
    config_class = None
    conditional_generation_class = None
    base_model_class = None
    sequence_classification_class = None

    # Key name for the audio sub-config in the main config constructor.
    # Override to "encoder_config" for models like GraniteSpeech.
    audio_config_key = "audio_config"

    # Model attribute name for the audio encoder (used in SDPA dispatch tests).
    # Set to None to skip audio encoder SDPA checking.
    audio_tower_attr = "audio_tower"

    # Arguments that should be passed to the config class even if not in its signature.
    forced_config_args = ["pad_token_id"]

    _required_attributes = ("config_class", "conditional_generation_class")

    @property
    def all_model_classes(self):
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
        return {"any-to-any": self.conditional_generation_class}

    def __init__(self, parent, **kwargs):
        self.parent = parent

        # Standard defaults
        kwargs.setdefault("batch_size", 3)
        kwargs.setdefault("seq_length", 25)
        kwargs.setdefault("feat_seq_length", 60)
        kwargs.setdefault("num_mel_bins", 80)
        kwargs.setdefault("is_training", True)
        kwargs.setdefault("use_labels", True)
        kwargs.setdefault("pad_token_id", 1)
        kwargs.setdefault("bos_token_id", 1)
        kwargs.setdefault("eos_token_id", 2)
        kwargs.setdefault("audio_token_id", 0)
        kwargs.setdefault("audio_token_index", 0)  # Alias for models that use this name
        kwargs.setdefault("ignore_index", -100)
        kwargs.setdefault("scope", None)

        # Text config defaults (small Qwen2-style backbone)
        kwargs.setdefault(
            "text_config",
            {
                "model_type": "qwen2",
                "intermediate_size": 36,
                "initializer_range": 0.02,
                "hidden_size": 32,
                "max_position_embeddings": 52,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 99,
                "pad_token_id": 1,
            },
        )

        # Audio config defaults (small Whisper-style encoder)
        kwargs.setdefault(
            "audio_config",
            {
                "model_type": "qwen2_audio_encoder",
                "d_model": 16,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 16,
                "encoder_layers": 2,
                "num_mel_bins": 80,
                "max_source_positions": 30,
                "initializer_range": 0.02,
            },
        )

        # Optional projector config (e.g. GraniteSpeech uses a Q-Former projector)
        kwargs.setdefault("projector_config", None)

        # Set all kwargs as instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Derived from text config (needed by ModelTesterMixin)
        self.vocab_size = self.text_config.get("vocab_size", 99)
        self.hidden_size = self.text_config.get("hidden_size", 32)
        self.num_hidden_layers = self.text_config.get("num_hidden_layers", 2)
        self.num_attention_heads = self.text_config.get("num_attention_heads", 4)
        self.encoder_seq_length = self.seq_length

        for required_attribute in self._required_attributes:
            if getattr(self, required_attribute) is None:
                raise ValueError(
                    f"You have inherited from ALMModelTester but did not set the {required_attribute} attribute."
                )

    # Because audio-LMs have some different standards in how they handle audio tokens, we need
    # a few methods that can be overridden if required:

    def create_audio_features(self):
        """Create audio feature tensor. Override for different shapes (e.g. [B, T, features])."""
        return floats_tensor([self.batch_size, self.num_mel_bins, self.feat_seq_length])

    def create_attention_mask(self, input_ids):
        """Create text attention mask. Override for models without a padding sentinel."""
        attention_mask = torch.ones_like(input_ids, dtype=torch.long).to(torch_device)
        attention_mask[:, :1] = 0  # Padding sentinel
        return attention_mask

    def get_num_audio_tokens(self, audio_features):
        """Compute number of audio placeholder tokens from features. Override for different subsampling."""
        # Default: 2-stage pooling (common for Whisper-style encoders)
        input_length = (audio_features.shape[-1] - 1) // 2 + 1
        return (input_length - 2) // 2 + 1

    def place_audio_tokens(self, input_ids, config, num_audio_tokens):
        """Place audio placeholder tokens in input_ids. Override for different placement."""
        input_ids = input_ids.clone()
        input_ids[input_ids == self.audio_token_id] = self.pad_token_id
        input_ids[:, 1 : 1 + num_audio_tokens] = self.audio_token_id
        return input_ids

    def get_audio_feature_key(self):
        """Key name for audio features in the inputs dict."""
        return "input_features"

    def get_audio_mask_key(self):
        """Key name for audio attention mask. Return None if no audio mask needed."""
        return None

    def create_audio_mask(self, audio_features):
        """Create audio-level attention mask. Override for bool masks or different shapes."""
        return torch.ones([self.batch_size, self.feat_seq_length], dtype=torch.long).to(torch_device)

    def get_additional_inputs(self, config, input_ids, audio_features):
        """Return dict of model-specific extra inputs (e.g. image_sizes for multi-modal)."""
        return {}

    # End of overridable methods

    @property
    def config_args(self):
        return list(signature(self.config_class.__init__).parameters.keys())

    def get_config(self):
        kwargs = {}
        skip_keys = {"self", "text_config", self.audio_config_key, "projector_config"}
        attribute_map = getattr(self.config_class, "attribute_map", {})
        model_name_to_common_name = {v: k for k, v in attribute_map.items()}
        for k in self.config_args + self.forced_config_args:
            if k in skip_keys:
                continue
            if hasattr(self, k) and k != "self":
                kwargs[k] = getattr(self, k)
            elif k in model_name_to_common_name and hasattr(self, model_name_to_common_name[k]):
                kwargs[k] = getattr(self, model_name_to_common_name[k])
        kwargs["text_config"] = self.text_config
        kwargs[self.audio_config_key] = self.audio_config
        if self.projector_config is not None:
            kwargs["projector_config"] = self.projector_config
        return self.config_class(**kwargs)

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        audio_features = self.create_audio_features()
        num_audio_tokens = self.get_num_audio_tokens(audio_features)

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 2
        input_ids = self.place_audio_tokens(input_ids, config, num_audio_tokens)
        attention_mask = self.create_attention_mask(input_ids)

        inputs_dict = {
            self.get_audio_feature_key(): audio_features,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        audio_mask_key = self.get_audio_mask_key()
        if audio_mask_key is not None:
            inputs_dict[audio_mask_key] = self.create_audio_mask(audio_features)

        inputs_dict.update(self.get_additional_inputs(config, input_ids, audio_features))
        return config, inputs_dict


@require_torch
class ALMModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin):
    """
    Base test class for Audio-Language Models.

    Subclasses should set:
    - `model_tester_class`: The tester class (subclass of ALMModelTester)

    Optional:
    - `all_model_classes`: Override if not using default from model_tester
    - `pipeline_model_mapping`: Override if not using default from model_tester
    """

    model_tester_class = None
    all_model_classes = None
    pipeline_model_mapping = None

    # Audio-LMs are always composite
    _is_composite = True

    def setUp(self):
        if self.model_tester_class is None:
            raise ValueError(
                "You have inherited from ALMModelTest but did not set the model_tester_class attribute."
            )
        self.model_tester = self.model_tester_class(self)
        self.config_tester = ConfigTester(self, config_class=self.model_tester.config_class, has_text_modality=False)

        if self.pipeline_model_mapping is None:
            if self.all_model_classes is not None:
                raise ValueError(
                    "Tests that inherit from `ALMModelTest` and set `all_model_classes` must manually set "
                    "`pipeline_model_mapping`."
                )
            else:
                self.pipeline_model_mapping = self.model_tester.pipeline_model_mapping

        if self.all_model_classes is None:
            self.all_model_classes = self.model_tester.all_model_classes

    def test_config(self):
        """Test config common functionality."""
        self.config_tester.run_common_tests()

    def test_sdpa_can_dispatch_composite_models(self):
        """Verify SDPA toggles propagate correctly to audio and text sub-modules."""
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # SDPA (default)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                text_attn = "sdpa" if model.language_model._supports_sdpa else "eager"

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
                self.assertTrue(model.language_model.config._attn_implementation == text_attn)

                audio_tower_attr = self.model_tester.audio_tower_attr
                if audio_tower_attr is not None:
                    audio_tower = getattr(model, audio_tower_attr)
                    audio_attn = "sdpa" if audio_tower._supports_sdpa else "eager"
                    self.assertTrue(audio_tower.config._attn_implementation == audio_attn)

                # Eager
                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")

                if audio_tower_attr is not None:
                    self.assertTrue(getattr(model_eager, audio_tower_attr).config._attn_implementation == "eager")

                for _, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")

    @unittest.skip("Audio-LMs have no separate base model without a head.")
    def test_model_base_model_prefix(self):
        pass
