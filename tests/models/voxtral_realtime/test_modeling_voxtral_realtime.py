# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch VoxtralRealtime model."""

import functools
import unittest

from transformers import (
    AutoProcessor,
    VoxtralRealtimeConfig,
    VoxtralRealtimeForConditionalGeneration,
    is_datasets_available,
    is_torch_available,
)
from transformers.audio_utils import load_audio
from transformers.models.voxtral_realtime.configuration_voxtral_realtime import (
    VoxtralRealtimeEncoderConfig,
    VoxtralRealtimeTextConfig,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...alm_tester import ALMModelTest, ALMModelTester
from ...test_modeling_common import floats_tensor, ids_tensor


if is_datasets_available():
    import datasets

if is_torch_available():
    import torch


class VoxtralRealtimeModelTester(ALMModelTester):
    config_class = VoxtralRealtimeConfig
    conditional_generation_class = VoxtralRealtimeForConditionalGeneration
    text_config_class = VoxtralRealtimeTextConfig
    audio_config_class = VoxtralRealtimeEncoderConfig

    def __init__(self, parent, **kwargs):
        # VoxtralRealtime does additive audio/text fusion: seq_length must equal num_audio_embeds.
        # With audio_length_per_tok=8 (config default), num_audio_embeds = feat_seq_length // 8.
        kwargs.setdefault("seq_length", 32)
        kwargs.setdefault("feat_seq_length", kwargs["seq_length"] * 8)
        # Audio encoder uses RoPE; max position must cover post-conv length (feat_seq_length // 2).
        kwargs.setdefault("max_position_embeddings", kwargs["feat_seq_length"])
        kwargs.setdefault("head_dim", 8)
        kwargs.setdefault("rms_norm_eps", 1e-6)
        kwargs.setdefault("activation_function", "silu")
        kwargs.setdefault("hidden_act", "silu")
        super().__init__(parent, **kwargs)
        self._max_new_tokens = None

    def get_audio_embeds_mask(self, audio_mask):
        # Causal conv2 (stride 2, left-pad 1): post_conv_len = feat_seq_length // 2.
        # Projector reshapes by downsample_factor=4 → post_conv_len // downsample_factor embeds.
        downsample_factor = 4
        effective_feat = self.feat_seq_length + (self._max_new_tokens or 0) * 8
        post_conv_len = effective_feat // 2
        output_length = post_conv_len // downsample_factor
        return torch.ones([self.batch_size, output_length], dtype=torch.long).to(torch_device)

    def create_audio_features(self):
        effective_feat = self.feat_seq_length + (self._max_new_tokens or 0) * 8
        return floats_tensor([self.batch_size, self.num_mel_bins, effective_feat])

    def place_audio_tokens(self, input_ids, config, num_audio_tokens):
        # VoxtralRealtime fuses audio additively over the whole sequence; no placeholder token required.
        input_ids = input_ids.clone()
        input_ids[input_ids == self.audio_token_id] = self.pad_token_id
        return input_ids

    def prepare_config_and_inputs_for_common(self):
        # Custom pipeline: input_ids at seq_length, audio covers seq_length (+ max_new_tokens extras
        # during generation so the model can slice future-token audio per decode step). We do not run
        # the base-class `audio_embeds_mask.shape[1] <= seq_length` invariant because, for this model,
        # audio embeds legitimately exceed input length during generation.
        audio_features = self.create_audio_features()

        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        special_tokens = [self.pad_token_id, self.bos_token_id, self.eos_token_id, self.audio_token_id]
        for safe_id in range(self.vocab_size):
            if safe_id not in special_tokens:
                break
        else:
            raise ValueError("vocab_size too small for a non-special safe token.")
        input_ids[input_ids == self.pad_token_id] = safe_id
        input_ids[input_ids == self.eos_token_id] = safe_id

        config = self.get_config()
        # place_audio_tokens is a no-op for this model; call for symmetry.
        input_ids = self.place_audio_tokens(input_ids, config, torch.tensor([self.seq_length] * self.batch_size))
        attention_mask = self.create_attention_mask(input_ids)

        return config, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": audio_features,
        }


@require_torch
class VoxtralRealtimeForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    """
    Model tester for `VoxtralRealtimeForConditionalGeneration`.
    """

    additional_model_inputs = ["input_features"]
    model_tester_class = VoxtralRealtimeModelTester
    pipeline_model_mapping = {"any-to-any": VoxtralRealtimeForConditionalGeneration} if is_torch_available() else {}

    def _with_max_new_tokens(max_new_tokens):
        def decorator(test_func):
            @functools.wraps(test_func)
            def wrapper(self, *args, **kwargs):
                try:
                    self.model_tester._max_new_tokens = max_new_tokens
                    return test_func(self, *args, **kwargs)
                finally:
                    self.model_tester._max_new_tokens = None

            return wrapper

        return decorator

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        original_feat_seq_length = self.model_tester.feat_seq_length
        try:
            self.model_tester.feat_seq_length += self.max_new_tokens * 8
            config, inputs_dict = super().prepare_config_and_inputs_for_generate(batch_size=batch_size)
        finally:
            self.model_tester.feat_seq_length = original_feat_seq_length
        return config, inputs_dict

    @_with_max_new_tokens(max_new_tokens=10)
    def test_generate_methods_with_logits_to_keep(self):
        super().test_generate_methods_with_logits_to_keep()

    @_with_max_new_tokens(max_new_tokens=5)
    def test_generate_compile_model_forward_fullgraph(self):
        super().test_generate_compile_model_forward_fullgraph()

    @_with_max_new_tokens(max_new_tokens=5)
    def test_generate_with_and_without_position_ids(self):
        super().test_generate_with_and_without_position_ids()

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime: audio tokens are not replaced in inputs_embeds, "
        "audio and text embeddings are summed instead."
    )
    def test_mismatching_num_audio_tokens(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since input_features must be provided along input_ids"
    )
    def test_flash_attention_2_continue_generate_with_position_ids(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since input_features must be provided along input_ids"
    )
    def test_custom_4d_attention_mask(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since input_features must be provided along input_ids"
    )
    def test_flash_attn_2_from_config(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since input_features must be provided along input_ids"
    )
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since input_features must be provided along input_ids"
    )
    def attention_mask_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since input_features must be provided along input_ids"
    )
    def flash_attn_inference_equivalence(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime for now since encoder_past_key_values AND padding_cache are returned by generate"
    )
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since prepare_inputs_for_generation is overwritten"
    )
    def test_prepare_inputs_for_generation_kwargs_forwards(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since input_features must be provided along input_ids"
    )
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(
        reason="VoxtralRealtime does not fall in the paradigm of assisted decoding (at least for the way it is implemented in generate)"
    )
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(
        reason="VoxtralRealtime does not fall in the paradigm of assisted decoding (at least for the way it is implemented in generate)"
    )
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip(
        reason="VoxtralRealtime does not fall in the paradigm of assisted decoding (at least for the way it is implemented in generate)"
    )
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    @unittest.skip(
        reason="This test does not apply to VoxtralRealtime since in only pads input_ids but input_features should also be padded"
    )
    def test_left_padding_compatibility(self):
        pass

    @unittest.skip(
        reason="VoxtralRealtime output contains non-tensor padding_cache state that is incompatible with DataParallel gather"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(
        reason="VoxtralRealtime only supports static and offloaded_static cache implementations, not quantized cache"
    )
    def test_generate_with_quant_cache(self):
        pass


@require_torch
class VoxtralRealtimeForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint_name = "mistralai/Voxtral-Mini-4B-Realtime-2602"
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_name)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_single_longform(self):
        """
        reproducer: https://gist.github.com/eustlb/980bade49311336509985f9a308e80af
        """
        model = VoxtralRealtimeForConditionalGeneration.from_pretrained(self.checkpoint_name, device_map=torch_device)
        audio = load_audio(
            "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
            self.processor.feature_extractor.sampling_rate,
        )

        inputs = self.processor(audio, return_tensors="pt")
        inputs.to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs)
        decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)

        EXPECTED_OUTPUT = [
            " Come on! Dude. You got a tattoo. So did you, dude. No. Oh, dude, what does my tattoo say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude! What does mine say? Sweet! Idiot! Your tattoo says dude. Your tattoo says sweet. Got it? Sorry. Hey, sorry.",
        ]

        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

    @slow
    def test_batched(self):
        """
        reproducer: https://gist.github.com/eustlb/980bade49311336509985f9a308e80af
        """
        model = VoxtralRealtimeForConditionalGeneration.from_pretrained(self.checkpoint_name, device_map=torch_device)

        # Load dataset manually
        ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        speech_samples = ds.sort("id")[:5]["audio"]
        input_speech = [x["array"] for x in speech_samples]

        inputs = self.processor(input_speech, return_tensors="pt")
        inputs.to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs)
        decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)

        EXPECTED_OUTPUT = [
            " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.",
            " nor is mr quilter's manner less interesting than his matter",
            " He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind.",
            " He has grave doubts whether Sir Frederick Leighton's work is really Greek after all, and can discover in it but little of rocky Ithaca.",
            " Linnell's pictures are a sort of up-guards-and-atom paintings, and Mason's exquisite idylls are as national as a jingo poem. Mr. Burkett Foster's landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. And Mr. John Collier gives his sitter a cheerful slap on the back before he says, like a shampooer in a Turkish bath, Next man!",
        ]

        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

    @slow
    def test_batched_longform(self):
        """
        reproducer: https://gist.github.com/eustlb/980bade49311336509985f9a308e80af
        """
        model = VoxtralRealtimeForConditionalGeneration.from_pretrained(self.checkpoint_name, device_map=torch_device)

        audio1 = load_audio(
            "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
            self.processor.feature_extractor.sampling_rate,
        )
        audio2 = load_audio(
            "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama_first_45_secs.mp3",
            self.processor.feature_extractor.sampling_rate,
        )

        inputs = self.processor([audio1, audio2], return_tensors="pt")
        inputs.to(model.device, dtype=model.dtype)

        outputs = model.generate(**inputs)
        decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)

        EXPECTED_OUTPUT = [
            " Come on. Dude. You got a tattoo. So did you, dude. No. Oh, dude, what does my tattoo say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude! What does mine say? Sweet! Idiot! Your tattoo says dude. Your tattoo says sweet. Got it? Sorry. Hey, sorry.",
            " This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners and on distant military outposts, All these conversations are what have kept me honest, kept me inspired, and kept me going. Every day, I learned from you. You made me a better president, and you made me a better man. Over the course of these eight years, I've seen the goodness, the resilience, and the hope of the",
        ]

        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)
