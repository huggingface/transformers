# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import json
import unittest
from pathlib import Path

from transformers import (
    AutoProcessor,
    LlamaConfig,
    OmniASRConfig,
    OmniASRCTCConfig,
    OmniASREncoderConfig,
    OmniASRForConditionalGeneration,
    OmniASRForCTC,
    OmniASRModel,
    is_datasets_available,
    is_torch_available,
)
from transformers.testing_utils import cleanup, require_torch, slow, torch_device

from ...alm_tester import ALMModelTest, ALMModelTester
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, random_attention_mask


if is_datasets_available():
    from datasets import Audio, load_dataset

if is_torch_available():
    import torch


class OmniASRModelTester(ALMModelTester):
    config_class = OmniASRConfig
    base_model_class = OmniASRModel
    conditional_generation_class = OmniASRForConditionalGeneration
    text_config_class = LlamaConfig
    audio_config_class = OmniASREncoderConfig
    audio_mask_key = "padding_mask"

    def __init__(self, parent, **kwargs):
        # seq_length 20 = BOS + 12 audio placeholders + 1 language placeholder + 6 text, which keeps the tail of
        # each sequence text-only (the resize_token_embeddings test overwrites column -2).
        kwargs.setdefault("seq_length", 20)
        # 80 raw samples through the two convolutions below -> 12 encoder frames.
        kwargs.setdefault("feat_seq_length", 80)
        kwargs.setdefault("conv_dim", [16, 16])
        kwargs.setdefault("conv_kernel", [5, 3])
        kwargs.setdefault("conv_stride", [3, 2])
        kwargs.setdefault("num_conv_pos_embeddings", 8)
        kwargs.setdefault("num_conv_pos_embedding_groups", 2)
        # Low placeholder ids on purpose: `test_resize_tokens_embeddings` clamps `input_ids` from above, which
        # would wipe out placeholders sitting at the end of the table (where the real checkpoints keep them).
        kwargs.setdefault("audio_token_id", 0)
        kwargs.setdefault("language_embedding_token_id", 3)
        kwargs.setdefault("language_token_id", 4)
        kwargs.setdefault("num_language_embeddings", 4)
        # Keeps a training-mode forward deterministic: no language dropout, and no layer dropped by the encoder.
        kwargs.setdefault("language_embedding_probability", 0.0)
        kwargs.setdefault("layerdrop", 0.0)
        # Llama needs head_dim
        kwargs.setdefault("head_dim", 8)
        super().__init__(parent, **kwargs)

    @property
    def _special_token_ids(self):
        # The LID marker is an ordinary token as far as the model is concerned -- only the processor writes it --
        # so only the language placeholder has to be kept out of the random text.
        return super()._special_token_ids | {self.language_embedding_token_id}

    def create_audio_features(self):
        # OmniASR is fed the raw waveform, not mel features.
        return floats_tensor([self.batch_size, self.feat_seq_length])

    def get_audio_feature_key(self):
        return "input_values"

    def get_audio_embeds_mask(self, audio_mask):
        # Mirrors `OmniASRPreTrainedModel._get_feat_extract_output_lengths`.
        lengths = audio_mask.sum(-1)
        for kernel, stride in zip(self.conv_kernel, self.conv_stride):
            lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
        positions = torch.arange(int(lengths.max()), device=audio_mask.device)[None, :]
        return (positions < lengths[:, None]).long()

    def place_audio_tokens(self, input_ids, config, num_audio_tokens):
        """Place the audio placeholders after BOS, then the single language placeholder right behind them.

        OmniASR's prompt is `audio | lid_marker | language | bos`, so every sequence carries exactly one language
        placeholder, which the row of the language embedding table is scattered over.
        """
        input_ids = super().place_audio_tokens(input_ids, config, num_audio_tokens)
        for i in range(input_ids.shape[0]):
            n = num_audio_tokens[i].item() if isinstance(num_audio_tokens, torch.Tensor) else num_audio_tokens
            if 2 + int(n) > self.seq_length:
                raise ValueError(
                    f"Cannot place {int(n)} audio placeholders and a language placeholder after BOS in a sequence "
                    f"of length {self.seq_length}. Please raise `seq_length`."
                )
            input_ids[i, 1 + int(n)] = self.language_embedding_token_id
        return input_ids


@require_torch
class OmniASRForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    model_tester_class = OmniASRModelTester

    @unittest.skip(
        reason="Like other audio LMs (Voxtral, Qwen3 ASR) inputs_embeds corresponding to audio tokens are replaced when input values are provided."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    def test_mismatching_num_audio_tokens(self):
        """Same as the shared test, minus its multi-audio case.

        OmniASR transcribes one audio per prompt: `audio | lid_marker | language | bos` holds exactly one language
        placeholder, so duplicating the prompt along the sequence dim -- what the shared test does to build a
        multi-audio prompt -- asks for two language embeddings per sample and cannot succeed. The mismatch checks
        themselves do apply and are kept.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        audio_feature_key = self.model_tester.get_audio_feature_key()
        audio_mask_key = self.model_tester.audio_mask_key

        # The batch index `create_audio_mask` pinned to full length is guaranteed to carry audio tokens, so
        # duplicating it reliably moves the audio-token total.
        dup = int((input_dict["input_ids"] == self.model_tester.audio_token_id).sum(-1).argmax().item())

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            _ = model(**copy.deepcopy(input_dict))  # successful forward with no modifications

            # Test 1: remove one audio but leave the audio tokens in the text
            curr_input_dict = copy.deepcopy(input_dict)
            for key in (audio_feature_key, audio_mask_key):
                curr_input_dict[key] = curr_input_dict[key][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 2: add one audio but leave the audio tokens in the text
            curr_input_dict = copy.deepcopy(input_dict)
            for key in (audio_feature_key, audio_mask_key):
                curr_input_dict[key] = torch.cat([curr_input_dict[key], curr_input_dict[key][dup : dup + 1]], dim=0)
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # Test 3: duplicate the text along the seq dim so each prompt has twice as many audio tokens, while
            # leaving the audio features unchanged
            curr_input_dict = copy.deepcopy(input_dict)
            for key in ("input_ids", "attention_mask"):
                curr_input_dict[key] = torch.cat([curr_input_dict[key], curr_input_dict[key]], dim=1)
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)


class OmniASRForCTCModelTester:
    def __init__(self, parent, batch_size=3, num_samples=80, vocab_size=32, pad_token_id=0, is_training=False):
        self.parent = parent
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.is_training = is_training

        # 80 raw samples through the two convolutions below -> 12 encoder frames.
        self.conv_dim = [16, 16]
        self.conv_kernel = [5, 3]
        self.conv_stride = [3, 2]
        self.output_seq_length = 12
        self.seq_length = self.output_seq_length
        self.hidden_size = 32
        self.num_hidden_layers = 2
        self.num_attention_heads = 2

    def get_config(self):
        return OmniASRCTCConfig(
            encoder_config=OmniASREncoderConfig(
                hidden_size=self.hidden_size,
                conv_dim=self.conv_dim,
                conv_kernel=self.conv_kernel,
                conv_stride=self.conv_stride,
                num_attention_heads=self.num_attention_heads,
                num_hidden_layers=self.num_hidden_layers,
                intermediate_size=32,
                num_conv_pos_embeddings=8,
                num_conv_pos_embedding_groups=2,
                layerdrop=0.0,
            ),
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs(self):
        input_values = floats_tensor([self.batch_size, self.num_samples])
        attention_mask = random_attention_mask([self.batch_size, self.num_samples])
        return self.get_config(), input_values, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_values, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_values": input_values, "attention_mask": attention_mask}

    def create_and_check_model(self, config, input_values, attention_mask):
        model = OmniASRForCTC(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_values, attention_mask=attention_mask)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.output_seq_length, self.vocab_size))


@require_torch
class OmniASRForCTCModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (OmniASRForCTC,) if is_torch_available() else ()
    all_generative_model_classes = ()  # OmniASRForCTC has a custom `generate`
    _is_composite = True
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = OmniASRForCTCModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OmniASRCTCConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        self.model_tester.create_and_check_model(*self.model_tester.prepare_config_and_inputs())

    @unittest.skip(reason="OmniASRForCTC is an encoder with a CTC head: it takes the waveform, not inputs_embeds.")
    def test_model_get_set_embeddings(self):
        pass


@require_torch
class OmniASRForCTCIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = "bezzam/omniasr-ctc-300m-v2"
        cls.dtype = torch.float32
        cls.processor = AutoProcessor.from_pretrained("bezzam/omniasr-ctc-300m-v2")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def _load_dataset(cls):
        if cls._dataset is None:
            cls._dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            cls._dataset = cls._dataset.cast_column(
                "audio", Audio(sampling_rate=cls.processor.feature_extractor.sampling_rate)
            )

    def _load_datasamples(self, num_samples):
        self._load_dataset()
        ds = self._dataset
        speech_samples = ds.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    @slow
    def test_ctc_300m_v2_model_integration(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-reproducer_ctc-py
        """
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/omniasr/expected_results_single.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["pred_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(1)
        model = OmniASRForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map="auto")

        inputs = self.processor(samples)
        inputs.to(model.device, dtype=self.dtype)
        predicted_ids = model.generate(**inputs)
        torch.testing.assert_close(predicted_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_ctc_300m_v2_model_integration_batched(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-reproducer_ctc_batch-py
        """
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/omniasr/expected_results_batch.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["pred_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(3)
        model = OmniASRForCTC.from_pretrained(self.checkpoint_name, torch_dtype=self.dtype, device_map="auto")

        inputs = self.processor(samples)
        inputs.to(model.device, dtype=self.dtype)
        encoder_lengths = model._get_subsampling_output_length(inputs["attention_mask"].sum(-1))

        predicted_ids = model.generate(**inputs)
        for idx, length in enumerate(encoder_lengths.tolist()):
            torch.testing.assert_close(predicted_ids[idx, :length].cpu(), EXPECTED_TOKEN_IDS[idx, :length])
        predicted_transcripts = self.processor.decode(predicted_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)


@require_torch
class OmniASRForConditionalGenerationIntegrationTest(unittest.TestCase):
    _dataset = None

    @classmethod
    def setUp(cls):
        cls.checkpoint_name = "bezzam/omniasr-llm-300m-v2"
        cls.dtype = torch.float32
        cls.processor = AutoProcessor.from_pretrained("bezzam/omniasr-llm-300m-v2")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @classmethod
    def _load_dataset(cls):
        if cls._dataset is None:
            cls._dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            cls._dataset = cls._dataset.cast_column(
                "audio", Audio(sampling_rate=cls.processor.feature_extractor.sampling_rate)
            )

    def _load_datasamples(self, num_samples):
        self._load_dataset()
        ds = self._dataset
        speech_samples = ds.sort("id")[:num_samples]["audio"]
        return [x["array"] for x in speech_samples]

    @slow
    def test_llm_300m_v2_model_integration(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-reproducer_llm-py
        """
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/omniasr/expected_results_single_llm.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = torch.tensor(raw_data["pred_ids"])
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(1)
        model = OmniASRForConditionalGeneration.from_pretrained(
            self.checkpoint_name, torch_dtype=self.dtype, device_map="auto"
        )

        inputs = self.processor(
            samples,
            return_tensors="pt",
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            language=["eng_Latn"],
        )
        inputs.to(model.device, dtype=self.dtype)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
            )
        # the audio prompt is part of `input_ids`, so `generate` returns it back before the transcription
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]

        torch.testing.assert_close(generated_ids.cpu(), EXPECTED_TOKEN_IDS)
        predicted_transcripts = self.processor.decode(generated_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_llm_300m_v2_model_integration_batched(self):
        """
        reproducer (creates JSON directly in repo): https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-reproducer_llm_batch-py
        """
        RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/omniasr/expected_results_batch_llm.json"
        with open(RESULTS_PATH, "r") as f:
            raw_data = json.load(f)
        EXPECTED_TOKEN_IDS = raw_data["pred_ids"]
        EXPECTED_HYPOTHESIS_LENS = raw_data["hypothesis_lens"]
        EXPECTED_TRANSCRIPTIONS = raw_data["transcriptions"]

        samples = self._load_datasamples(3)
        model = OmniASRForConditionalGeneration.from_pretrained(
            self.checkpoint_name, torch_dtype=self.dtype, device_map="auto"
        )

        inputs = self.processor(
            samples,
            return_tensors="pt",
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            padding=True,
            language=["eng_Latn"] * len(samples),
        )
        inputs.to(model.device, dtype=self.dtype)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
            )
        # the audio prompt is part of `input_ids`, so `generate` returns it back before the transcription
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]

        # The audio context is left-padded, so each sample decodes exactly as it would on its own. Compare each
        # hypothesis over its own length; what follows it is padding.
        for idx, length in enumerate(EXPECTED_HYPOTHESIS_LENS):
            torch.testing.assert_close(
                generated_ids[idx, :length].cpu(), torch.tensor(EXPECTED_TOKEN_IDS[idx][:length])
            )

        predicted_transcripts = self.processor.decode(generated_ids, skip_special_tokens=True)
        self.assertListEqual(predicted_transcripts, EXPECTED_TRANSCRIPTIONS)

    @slow
    def test_llm_300m_v2_generate_is_padding_invariant(self):
        """
        Batching must not change what a sample transcribes to. The shortest and longest clips of the dataset are
        paired so that the short one carries ~6x its own length in padding: without the audio `attention_mask`
        reaching the encoder, and without the audio being left-padded, it decodes to a shorter, degraded hypothesis.
        """
        samples = self._load_datasamples(6)
        shortest, longest = min(samples, key=len), max(samples, key=len)
        model = OmniASRForConditionalGeneration.from_pretrained(
            self.checkpoint_name, torch_dtype=self.dtype, device_map="auto"
        )

        def generate(batch):
            inputs = self.processor(
                batch,
                sampling_rate=self.processor.feature_extractor.sampling_rate,
                language=["eng_Latn"] * len(batch),
                padding=True,
            )
            inputs.to(model.device, dtype=self.dtype)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=200)
            # the audio prompt is part of `input_ids`, so `generate` returns it back before the transcription
            return generated_ids[:, inputs["input_ids"].shape[1] :]

        alone = generate([shortest])[0].cpu()
        batched = generate([shortest, longest])[0].cpu()
        torch.testing.assert_close(batched[: alone.shape[0]], alone)
        # everything past the hypothesis is padding
        self.assertTrue(bool((batched[alone.shape[0] :] == model.config.pad_token_id).all()))
