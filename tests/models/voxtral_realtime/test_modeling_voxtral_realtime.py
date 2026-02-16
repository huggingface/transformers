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
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_datasets_available():
    import datasets

if is_torch_available():
    import torch


class VoxtralRealtimeModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        audio_token_id=0,
        seq_length=5,
        feat_seq_length=40,
        text_config={
            "model_type": "voxtral_realtime_text",
            "intermediate_size": 36,
            "initializer_range": 0.02,
            "hidden_size": 32,
            "max_position_embeddings": 52,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "use_labels": True,
            "vocab_size": 99,
            "head_dim": 8,
            "pad_token_id": 1,  # can't be the same as the audio token id
            "hidden_act": "silu",
            "rms_norm_eps": 1e-6,
            "attention_dropout": 0.0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        },
        is_training=True,
        audio_config={
            "model_type": "voxtral_realtime_encoder",
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "encoder_layers": 2,
            "num_mel_bins": 80,
            "max_position_embeddings": 100,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "activation_function": "silu",
            "activation_dropout": 0.0,
            "attention_dropout": 0.0,
            "head_dim": 4,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.audio_token_id = audio_token_id
        self.text_config = text_config
        self.audio_config = audio_config
        self.seq_length = seq_length
        self.feat_seq_length = feat_seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.encoder_seq_length = seq_length
        self._max_new_tokens = None  # this is used to set

    def get_config(self):
        return VoxtralRealtimeConfig(
            text_config=self.text_config,
            audio_config=self.audio_config,
            ignore_index=self.ignore_index,
            audio_token_id=self.audio_token_id,
        )

    def prepare_config_and_inputs(self):
        if self._max_new_tokens is not None:
            feat_seq_length = self.feat_seq_length + self._max_new_tokens * 8
        else:
            feat_seq_length = self.feat_seq_length

        input_features_values = floats_tensor(
            [
                self.batch_size,
                self.audio_config["num_mel_bins"],
                feat_seq_length,
            ]
        )
        config = self.get_config()
        return config, input_features_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_features_values = config_and_inputs
        num_audio_tokens_per_batch_idx = 30

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)
        attention_mask[:, :1] = 0

        input_ids[:, 1 : 1 + num_audio_tokens_per_batch_idx] = config.audio_token_id
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features_values,
        }
        return config, inputs_dict


@require_torch
class VoxtralRealtimeForConditionalGenerationModelTest(
    ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    """
    Model tester for `VoxtralRealtimeForConditionalGeneration`.
    """

    additional_model_inputs = ["input_features"]

    all_model_classes = (VoxtralRealtimeForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"any-to-any": VoxtralRealtimeForConditionalGeneration} if is_torch_available() else {}

    _is_composite = True

    def setUp(self):
        self.model_tester = VoxtralRealtimeModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VoxtralRealtimeConfig, has_text_modality=False)

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

    @unittest.skip(reason="VoxtralRealtime does not have a base model")
    def test_model_base_model_prefix(self):
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
            " Come on! Dude, you got a tattoo. So do you, dude. No. Oh, dude, what does my tattoo say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude! What does mine say? Sweet! Idiot! Your tattoo says dude. Your tattoo says sweet. Got it?",
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
            " Come on! Dude, you got a tattoo. So do you, dude. No. Oh, dude, what does my tattoo say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude, what does mine say? Sweet! What about mine? Dude! What does mine say? Sweet! Idiot! Your tattoo says dude. Your tattoo says sweet. Got it?",
            " This week, I traveled to Chicago to deliver my final farewell address to the nation. Following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye to eye or rarely agreed at all, My conversations with you, the American people, in living rooms, in schools, at farms, and on factory floors, at diners, and on distant military outposts, all these conversations are what have kept me honest, kept me inspired, and kept me going. Every day, I learned from you. You made me a better president, and you made me a better man. Over the course of these eight years, I've seen the goodness, the resilience, and the hope of the American",
        ]

        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)
