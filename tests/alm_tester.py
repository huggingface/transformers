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
    ids_tensor,
    is_torch_available,
    torch_device,
)


if is_torch_available():
    import torch


class ALMModelTester(MultiModalModelTester):
    audio_config_class = None
    audio_config_key = "audio_config"
    # Name under which the audio mask is passed to the model's forward (e.g. "feature_attention_mask"
    # for Qwen2Audio). Leave as `None` if the model does not consume a separate audio-level mask;
    # `_prepare_modality_inputs` then skips adding it to the inputs dict.
    audio_mask_key = None
    _required_attributes = MultiModalModelTester._required_attributes + ("audio_config_class",)

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
        # Overrides of _TEXT_MODEL_TESTER_DEFAULTS
        kwargs.setdefault("seq_length", 32)
        kwargs.setdefault("pad_token_id", 1)

        # ALM-specific defaults
        kwargs.setdefault("feat_seq_length", 128)
        kwargs.setdefault("num_mel_bins", 80)
        kwargs.setdefault("audio_token_id", 0)

        super().__init__(parent, **kwargs)

    # -- Overridable ALM-specific hooks ------------------------------------------------------

    def create_audio_features(self):
        """Create audio feature tensor. Override for different shapes (e.g. [B, T, features])."""
        return floats_tensor([self.batch_size, self.num_mel_bins, self.feat_seq_length])

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
            if 1 + int(n) > self.seq_length:
                raise ValueError(
                    f"Cannot place {int(n)} audio tokens after BOS in a sequence of length {self.seq_length}. "
                    "This likely indicates a mismatch between your feature extraction/configuration and your sequence length. "
                    "Please ensure `seq_length` is >= the number of audio embedding positions + 1."
                )
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

    # -- Hooks consumed by the shared base ---------------------------------------------------

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.audio_token_id}

    def _build_modality_sub_configs(self):
        return {self.audio_config_key: self.get_audio_config()}

    def _prepare_modality_inputs(self, input_ids, config):
        audio_features = self.create_audio_features()
        audio_mask = self.create_audio_mask()
        audio_embeds_mask = self.get_audio_embeds_mask(audio_mask)
        num_audio_tokens = audio_embeds_mask.sum(dim=1)
        input_ids = self.place_audio_tokens(input_ids, config, num_audio_tokens)

        modality_inputs = {self.get_audio_feature_key(): audio_features}
        if self.audio_mask_key is not None:
            modality_inputs[self.audio_mask_key] = audio_mask
        return input_ids, modality_inputs

    # -- Audio sub-config construction -------------------------------------------------------

    @property
    def audio_config_args(self):
        return list(signature(self.audio_config_class.__init__).parameters.keys())

    def get_audio_config(self):
        kwargs = self._collect_kwargs(self.audio_config_args, self.audio_config_class)
        return self.audio_config_class(**kwargs)


class ALMModelTest(MultiModalModelTest):
    """
    Base test class for Audio-Language Models.

    Subclasses should set:
    - `model_tester_class`: The tester class (subclass of ALMModelTester)

    Optional:
    - `all_model_classes`: Override if not using default from model_tester
    - `pipeline_model_mapping`: Override if not using default from model_tester
    """

    # TODO: @eustlb, remove this once #45534 is merged
    @unittest.skip("Audio-LMs have no separate base model without a head.")
    def test_model_base_model_prefix(self):
        pass

    def test_mismatching_num_audio_tokens(self):
        """
        Tests that ALMs throw an error with explicit message saying what is wrong
        when number of audios don't match number of audio tokens in the text.
        Also we need to test multi-audio cases when one prompt has multiple audio tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        audio_feature_key = self.model_tester.get_audio_feature_key()
        audio_mask_key = self.model_tester.audio_mask_key

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            curr_input_dict = copy.deepcopy(input_dict)
            _ = model(**curr_input_dict)  # successful forward with no modifications

            # Test 1: remove one audio but leave the audio tokens in the text
            curr_input_dict[audio_feature_key] = curr_input_dict[audio_feature_key][-1:, ...]
            if audio_mask_key is not None:
                curr_input_dict[audio_mask_key] = curr_input_dict[audio_mask_key][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 2: add one audio but leave the audio tokens in the text
            curr_input_dict = copy.deepcopy(input_dict)
            curr_input_dict[audio_feature_key] = torch.cat(
                [curr_input_dict[audio_feature_key], curr_input_dict[audio_feature_key][:1, ...]], dim=0
            )
            if audio_mask_key is not None:
                curr_input_dict[audio_mask_key] = torch.cat(
                    [curr_input_dict[audio_mask_key], curr_input_dict[audio_mask_key][:1, ...]], dim=0
                )
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 3: duplicate the text along the seq dim so each prompt has twice as many
            # audio tokens, while leaving the audio features unchanged -> mismatch
            curr_input_dict = copy.deepcopy(input_dict)
            curr_input_dict["input_ids"] = torch.cat(
                [curr_input_dict["input_ids"], curr_input_dict["input_ids"]], dim=1
            )
            curr_input_dict["attention_mask"] = torch.cat(
                [curr_input_dict["attention_mask"], curr_input_dict["attention_mask"]], dim=1
            )
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 4: multi-audio valid case. A prompt may contain multiple audio segments;
            # all audio segments are concatenated along the batch dim on the audio side.
            # Duplicating input_ids along seq dim (-> [audios, audios] per prompt) and the
            # audio features along batch dim (-> batch_size * 2) must forward successfully.
            curr_input_dict = copy.deepcopy(input_dict)
            curr_input_dict["input_ids"] = torch.cat(
                [curr_input_dict["input_ids"], curr_input_dict["input_ids"]], dim=1
            )
            curr_input_dict["attention_mask"] = torch.cat(
                [curr_input_dict["attention_mask"], curr_input_dict["attention_mask"]], dim=1
            )
            curr_input_dict[audio_feature_key] = torch.cat(
                [curr_input_dict[audio_feature_key], curr_input_dict[audio_feature_key]], dim=0
            )
            if audio_mask_key is not None:
                curr_input_dict[audio_mask_key] = torch.cat(
                    [curr_input_dict[audio_mask_key], curr_input_dict[audio_mask_key]], dim=0
                )
            _ = model(**curr_input_dict)
