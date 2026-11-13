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
import random
import tempfile
from unittest.mock import patch

from huggingface_hub import get_safetensors_metadata

from .multimodal_tester import MultiModalModelTest, MultiModalModelTester
from .test_modeling_common import (
    floats_tensor,
    ids_tensor,
    is_torch_available,
    torch_device,
)


if is_torch_available():
    import torch

    from transformers import PreTrainedConfig
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.utils.auto_docstring import get_checkpoint_from_config_class


class ALMModelTester(MultiModalModelTester):
    audio_config_key = "audio_config"
    # Name under which the audio mask is passed to the model's forward (e.g. "feature_attention_mask"
    # for Qwen2Audio). Leave as `None` if the model does not consume a separate audio-level mask;
    # `_prepare_modality_inputs` then skips adding it to the inputs dict.
    audio_mask_key = None

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

    def create_audio_features(self, batch_size: int | None = None):
        """Create audio feature tensor. Override for different shapes (e.g. [B, T, features])."""
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor([batch_size, self.num_mel_bins, self.feat_seq_length])

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

    def create_audio_mask(self, batch_size: int | None = None):
        """Create audio-level attention mask with contiguous valid regions per batch element.

        Each element gets a random offset and length, producing masks like [0, 0, 1, 1, 1, 0, 0].
        At least one batch index is pinned to a full-length mask.
        """
        # Use a locally-seeded RNG so repeated calls within a test produce the same mask
        rng = random.Random(0)
        batch_size = batch_size if batch_size is not None else self.batch_size
        # Sample lengths in [1, feat_seq_length] and offsets in [0, feat_seq_length - length]
        lengths = ids_tensor([batch_size], vocab_size=self.feat_seq_length, rng=rng).abs() + 1
        lengths = lengths.clamp(max=self.feat_seq_length)

        # Presuming feat_seq_length is set correctly, ensure at least one batch has a full-length mask for valid audio tokens
        lengths[rng.randint(0, batch_size - 1)] = self.feat_seq_length
        offsets = ids_tensor([batch_size], vocab_size=self.feat_seq_length, rng=rng).abs()
        offsets = offsets % (self.feat_seq_length - lengths + 1)

        positions = torch.arange(self.feat_seq_length, device=torch_device)[None, :]
        audio_mask = ((positions >= offsets[:, None]) & (positions < offsets[:, None] + lengths[:, None])).long()
        return audio_mask

    # -- Hooks consumed by the shared base ---------------------------------------------------

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.audio_token_id}

    def _prepare_modality_inputs(self, input_ids, config, batch_size: int | None = None):
        audio_features = self.create_audio_features(batch_size=batch_size)
        audio_mask = self.create_audio_mask(batch_size=batch_size)
        audio_embeds_mask = self.get_audio_embeds_mask(audio_mask)
        num_audio_tokens = audio_embeds_mask.sum(dim=1)
        input_ids = self.place_audio_tokens(input_ids, config, num_audio_tokens)

        modality_inputs = {self.get_audio_feature_key(): audio_features}
        if self.audio_mask_key is not None:
            modality_inputs[self.audio_mask_key] = audio_mask

        additional_inputs = self.get_additional_inputs(
            config, input_ids, modality_inputs=modality_inputs, batch_size=batch_size
        )
        modality_inputs.update(additional_inputs)
        return input_ids, modality_inputs


class ALMModelTest(MultiModalModelTest):
    """
    Base test class for Audio-Language Models.

    Subclasses should set:
    - `model_tester_class`: The tester class (subclass of ALMModelTester)

    Optional:
    - `all_model_classes`: Override if not using default from model_tester
    - `pipeline_model_mapping`: Override if not using default from model_tester
    """

    def test_sdpa_can_dispatch_on_flash(self):
        # `test_sdpa_can_dispatch_on_flash` already pops the attention mask, but we cannot simply pop the
        # audio mask here since it will raise an error in `get_audio_features` (cf. `test_mismatching_num_audio_tokens`).
        # Therefore we substitute a full-ones mask instead.
        def full_ones_mask(batch_size: int | None = None):
            batch_size = batch_size if batch_size is not None else self.model_tester.batch_size
            return torch.ones(
                [batch_size, self.model_tester.feat_seq_length],
                dtype=torch.bool,
                device=torch_device,
            )

        with patch.object(self.model_tester, "create_audio_mask", new=full_ones_mask):
            super().test_sdpa_can_dispatch_on_flash()

    def test_mismatching_num_audio_tokens(self):
        """
        Tests that ALMs throw an error with explicit message saying what is wrong
        when number of audios don't match number of audio tokens in the text.
        Also we need to test multi-audio cases when one prompt has multiple audio tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        audio_feature_key = self.model_tester.get_audio_feature_key()
        audio_mask_key = self.model_tester.audio_mask_key

        # Pick the batch index `create_audio_mask` pinned to full length — guaranteed to
        # contribute > 0 audio tokens even for encoders that aggressively downsample
        # (e.g. GlmAsr), so duplicating it in Test 2 reliably moves the audio-token total.
        audio_token_id = self.model_tester.audio_token_id
        dup_idx = int((input_dict["input_ids"] == audio_token_id).sum(-1).argmax().item())

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
                [curr_input_dict[audio_feature_key], curr_input_dict[audio_feature_key][dup_idx : dup_idx + 1, ...]],
                dim=0,
            )
            if audio_mask_key is not None:
                curr_input_dict[audio_mask_key] = torch.cat(
                    [curr_input_dict[audio_mask_key], curr_input_dict[audio_mask_key][dup_idx : dup_idx + 1, ...]],
                    dim=0,
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

    # TODO: @eustlb, escalate to core? Core `test_reverse_loading_mapping` only checks self-consistency (no reference
    # layout), so it misses #45534. No saved layout is enforced: built from config, all rules are reverted and rule
    # order picks the layout when rules collide (#45534, nemotron3_5_transcribe, inkling); after `from_pretrained`,
    # used rules can still collide (inkling), untested. Escalating = run this check for every model with a mapping.
    def test_save_pretrained_matches_hub_checkpoint(self):
        """
        Model built from config (no `from_pretrained`) must save under its reference checkpoint's tensor names.
        Regression test for #45534: saved `language_model.model.model.*` instead of the checkpoint's
        `language_model.model.*`.

        Test:
            1. reference = repo in config docstring, fetch only its config + safetensors header
            2. build model from that config on meta
            3. run `save_pretrained`, capture the names it writes (file write patched out)
            4. every saved name must be in the checkpoint -> catches #45534
        """
        model_class = self.model_tester.conditional_generation_class
        config_class = model_class.config_class
        repo = get_checkpoint_from_config_class(config_class)
        if repo is None:  # `auto_docstring(custom_intro=...)` drops the link, the sub-configs keep it (e.g. Inkling)
            sub_config_classes = [
                c for c in config_class.sub_configs.values() if isinstance(c, type) and issubclass(c, PreTrainedConfig)
            ]
            repo = next(filter(None, map(get_checkpoint_from_config_class, sub_config_classes)), None)
        if repo is None:
            self.skipTest(f"No reference checkpoint declared for {config_class.__name__}")
        config = config_class.from_pretrained(repo)
        with torch.device("meta"):
            model = model_class(config)
        if not get_model_conversion_mapping(model, add_legacy=False):
            self.skipTest("No checkpoint conversion mapping: the model saves its own parameter names")
        checkpoint_keys = set(get_safetensors_metadata(repo).weight_map)

        # Real `save_pretrained`, only the file write is replaced (meta tensors can't be written)
        saved_keys = set()
        with (
            tempfile.TemporaryDirectory() as tmpdirname,
            patch("transformers.modeling_utils.safe_save_file", lambda shard, *_, **__: saved_keys.update(shard)),
        ):
            model.save_pretrained(tmpdirname)

        self.assertEqual(saved_keys - checkpoint_keys, set(), f"Saved under names that are not in {repo}")
