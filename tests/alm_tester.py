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
    base_model_class = None, # this should be added for most models when #45534 is merged
    config_class = None
    text_config_class = None
    audio_config_class = None
    conditional_generation_class = None
    sequence_classification_class = None
    # These attributes are required after the initialization phase of the tester.
    _required_attributes = ("config_class", "conditional_generation_class")

    # Arguments that should be passed to the config class even if not in its signature.
    forced_config_args = ["pad_token_id"]

    # Key name for the audio sub-config in the main config constructor.
    # Override to "encoder_config" for models like GraniteSpeech.
    audio_config_key = "audio_config"
    audio_mask_key = None  # to be set if audio-related mask has to be passed to the model's forward

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
        # TODO: @eustlb, we don't have pipeline testing for audio-text-to-text
        mapping = {
            "feature-extraction": self.base_model_class,
            # "audio-text-to-text": self.conditional_generation_class,
        }
        # TODO: should we add automatic-speech-recognition with a special flag?
        return mapping

    def __init__(self, parent, **kwargs):
        self.parent = parent

        # Standard defaults
        kwargs.setdefault("batch_size", 3)

        # TODO: explain here specifically why these values are chosen
        kwargs.setdefault("seq_length", 32)
        kwargs.setdefault("feat_seq_length", 128)

        kwargs.setdefault("num_mel_bins", 80)
        kwargs.setdefault("is_training", True)
        kwargs.setdefault("use_labels", True)
        kwargs.setdefault("pad_token_id", 1)
        kwargs.setdefault("bos_token_id", 1)
        kwargs.setdefault("eos_token_id", 2)
        kwargs.setdefault("audio_token_id", 0)
        kwargs.setdefault("ignore_index", -100)
        kwargs.setdefault("scope", None)
        kwargs.setdefault("vocab_size", 99)
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("num_attention_heads", 2)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("intermediate_size", 32)  # Keep this divisible by 8 for fp16/bf16/fp32 16-bytes alignment
        kwargs.setdefault("hidden_act", "gelu")
        kwargs.setdefault("max_position_embeddings", 512)

        # Set all kwargs as instance attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        for required_attribute in [
            # "base_model_class", # TODO: @eustlb, there is a discrepancy here between ALMs/ VLMs. XXModel and XXForConditionalGeneration
            "config_class",
            "conditional_generation_class",
            "text_config_class",
            "audio_config_class",
        ]:
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
        # TODO: check, this looks strange to force as default behavior
        # Override for bidirectional attention models like Gemma3
        return torch.tril(torch.ones_like(input_ids).to(torch_device))

    def get_audio_embeds_mask(self, audio_embeds_mask):
        """Get audio embeds mask from audio mask. Override for different shapes."""
        raise NotImplementedError("This method should be overridden in the subclass")

    def place_audio_tokens(self, input_ids, config, num_audio_tokens):
        """Place audio placeholder tokens contiguously after BOS. Override for different placement.

        Deterministic placement (position 0 reserved for BOS; audio tokens at [1:1+n]) keeps
        the tail of each sequence text-only, which downstream tests (e.g. resize_token_embeddings
        overwriting column -2) rely on.
        """
        input_ids = input_ids.clone()
        input_ids[input_ids == self.audio_token_id] = self.pad_token_id
        for i in range(input_ids.shape[0]):
            n = num_audio_tokens[i].item() if isinstance(num_audio_tokens, torch.Tensor) else num_audio_tokens
            input_ids[i, 1 : 1 + int(n)] = self.audio_token_id
        return input_ids

    def get_audio_feature_key(self):
        """Key name for audio features in the inputs dict."""
        return "input_features" 

    def create_audio_mask(self):
        """Create audio-level attention mask with contiguous valid regions per batch element.

        Each element gets a random offset and length, producing masks like [0, 0, 1, 1, 1, 0, 0].
        """
        # Sample lengths in [1, feat_seq_length] and offsets in [0, feat_seq_length - length]
        lengths = ids_tensor([self.batch_size], vocab_size=self.feat_seq_length).abs() + 1
        lengths = lengths.clamp(max=self.feat_seq_length)
        offsets = ids_tensor([self.batch_size], vocab_size=self.feat_seq_length).abs()
        offsets = offsets % (self.feat_seq_length - lengths + 1)

        positions = torch.arange(self.feat_seq_length, device=torch_device)[None, :]
        audio_mask = ((positions >= offsets[:, None]) & (positions < offsets[:, None] + lengths[:, None])).long()
        return audio_mask

    def get_additional_inputs(self, config, input_ids, audio_features):
        """Return dict of model-specific extra inputs (e.g. image_sizes for multi-modal)."""
        return {}

    # End of overridable methods

    def prepare_config_and_inputs_for_common(self):
        # TODO: add a clear diagram that explains input prep

        audio_features = self.create_audio_features()
        audio_mask = self.create_audio_mask()
        audio_embeds_mask = self.get_audio_embeds_mask(audio_mask)

        if audio_embeds_mask.shape[1] > self.seq_length:
            raise ValueError(
                f"`audio_embeds_mask` has more tokens per sequence than `seq_length` allows "
                f"({audio_embeds_mask.shape[1]} > {self.seq_length}). "
                "This likely indicates a mismatch between your feature extraction/configuration and your sequence length. "
                "Please ensure `seq_length` is >= the number of audio embedding positions."
            )

        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        special_tokens = [self.pad_token_id, self.bos_token_id, self.eos_token_id, self.audio_token_id]
        for i in range(self.vocab_size):
            if i not in special_tokens:
                safe_token_id = i
                break
        else:
            raise ValueError("vocab_size is too small and there is no token ID that is not a special token!")

        # Avoid flaky tests, clear any special tokens in ids_tensor
        # audio_token_id is handled separately by place_audio_tokens()
        input_ids[input_ids == self.pad_token_id] = safe_token_id
        input_ids[input_ids == self.eos_token_id] = safe_token_id

        config = self.get_config()
        num_audio_tokens = audio_embeds_mask.sum(dim=1)
        input_ids = self.place_audio_tokens(input_ids, config, num_audio_tokens)
        attention_mask = self.create_attention_mask(input_ids)

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            self.get_audio_feature_key(): audio_features,
        }

        if self.audio_mask_key is not None:
            inputs_dict[self.audio_mask_key] = audio_mask

        inputs_dict.update(self.get_additional_inputs(config, input_ids, audio_features))
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
    def audio_config_args(self):
        return list(signature(self.audio_config_class.__init__).parameters.keys())

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
        kwargs[self.audio_config_key] = self.get_audio_config()
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

    def get_audio_config(self):
        kwargs = {}
        attribute_map = getattr(self.audio_config_class, "attribute_map", {})
        model_name_to_common_name = {v: k for k, v in attribute_map.items()}
        for k in self.audio_config_args:
            if hasattr(self, k) and k != "self":
                kwargs[k] = getattr(self, k)
            elif k in model_name_to_common_name and hasattr(self, model_name_to_common_name[k]):
                kwargs[k] = getattr(self, model_name_to_common_name[k])
        return self.audio_config_class(**kwargs)

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
            raise ValueError("You have inherited from ALMModelTest but did not set the model_tester_class attribute.")
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

    # TODO: @eustlb, remove this once #45534 is merged
    @unittest.skip("Audio-LMs have no separate base model without a head.")
    def test_model_base_model_prefix(self):
        pass

    # TODO: @eustlb, add this
    # def test_mismatching_num_audio_tokens(self):
    #     pass
