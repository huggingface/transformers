# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the IBM Granite Speech model."""

import tempfile
import unittest

import pytest

from transformers import (
    AutoProcessor,
    GraniteSpeechConfig,
    GraniteSpeechForConditionalGeneration,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_sdpa,
    slow,
    torch_device,
)
from transformers.utils import (
    is_datasets_available,
    is_peft_available,
    is_torch_available,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch

if is_datasets_available():
    from datasets import load_dataset


class GraniteSpeechForConditionalGenerationModelTester:
    def __init__(
        self,
        parent,
        seq_length=7,
        encoder_config={
            "model_type": "granite_speech_encoder",
            "context_size": 200,
            "conv_expansion_factor": 2,
            "conv_kernel_size": 15,
            "dim_head": 32,
            "dropout": 0.1,
            "feedforward_mult": 4,
            "hidden_dim": 32,
            "input_dim": 160,
            "num_heads": 4,
            "num_layers": 2,
            "output_dim": 42,
        },
        text_config={
            "model_type": "granite",
            "is_training": True,
            "seq_length": 7,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 580,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 1,
        },
        projector_config={
            "attention_probs_dropout_prob": 0.1,
            "cross_attention_frequency": 1,
            "encoder_hidden_size": 32,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 32,
            "initializer_range": 0.02,
            "intermediate_size": 256,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 2048,
            "model_type": "blip_2_qformer",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "position_embedding_type": "absolute",
            "use_qformer_text_input": False,
            "vocab_size": 30522,
        },
        audio_token_index=0,
        tie_word_embeddings=True,
        initializer_range=0.02,
        has_lora_adapter=True,
        downsample_rate=5,
        window_size=15,
        is_training=True,
    ):
        self.parent = parent
        self.encoder_config = encoder_config
        self.text_config = text_config
        self.projector_config = projector_config
        self.audio_token_index = audio_token_index
        self.tie_word_embeddings = tie_word_embeddings
        self.initializer_range = initializer_range
        self.has_lora_adapater = has_lora_adapter
        self.downsample_rate = downsample_rate
        self.window_size = window_size
        self.is_training = is_training

        # Dims for audio features
        self.sequence_dim = 844
        self.feature_dim = 160
        self.num_attention_heads = text_config["num_attention_heads"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.hidden_size = text_config["hidden_size"]
        self.batch_size = 3
        self.pad_token_id = text_config["pad_token_id"]
        self.seq_len = 7
        self.num_audio_tokens = 2
        self.seq_length = seq_length + self.num_audio_tokens

    def get_config(self):
        return GraniteSpeechConfig(
            encoder_config=self.encoder_config,
            text_config=self.text_config,
            projector_config=self.projector_config,
            audio_token_index=self.audio_token_index,
            tie_word_embeddings=self.tie_word_embeddings,
            initializer_range=self.initializer_range,
            has_lora_adapter=self.has_lora_adapater,
        )

    def prepare_config_and_inputs(self):
        input_features = floats_tensor(
            [self.batch_size, self.sequence_dim, self.feature_dim],
        )
        config = self.get_config()
        return config, input_features

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_features = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 2
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)
        input_ids[input_ids == config.audio_token_index] = self.pad_token_id

        input_ids[:, : self.num_audio_tokens] = config.audio_token_index

        inputs_dict = {
            "input_features": input_features,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict

    def create_and_check_granite_speech_model_fp16_forward(self, config, input_ids, input_features, attention_mask):
        model = GraniteSpeechForConditionalGeneration(config=config)
        model.to(torch_device)
        model.half()
        model.eval()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            return_dict=True,
        )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())

    def create_and_check_granite_speech_model_fp16_autocast_forward(
        self,
        config,
        input_ids,
        input_features,
        attention_mask,
    ):
        config.dtype = torch.float16
        model = GraniteSpeechForConditionalGeneration(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class GraniteSpeechForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `GraniteSpeechForConditionalGeneration`.
    """

    all_model_classes = (GraniteSpeechForConditionalGeneration,) if is_torch_available() else ()
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = GraniteSpeechForConditionalGenerationModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=GraniteSpeechConfig,
            has_text_modality=False,
        )

    def test_inputs_embeds(self):
        # overwrite inputs_embeds tests because we need to delete "input features" for the audio model
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["input_features"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)

            with torch.no_grad():
                model(**inputs)

    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if name == "projector.query":
                    continue
                elif param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        # overwrite because Granite Speech is audio+text model (not vision+text)
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            # NOTE - currently we only enable alternate attention implementations on
            # the encapsulated LLM; in the future, this should be added for the conformer
            # encoder as well.
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                text_attn = "sdpa" if model.language_model._supports_sdpa else "eager"

                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
                self.assertTrue(model.language_model.config._attn_implementation == text_attn)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")

    @pytest.mark.generate
    @require_torch_sdpa
    @slow
    @unittest.skip(reason="Granite Speech doesn't support SDPA for all backbones")
    def test_eager_matches_sdpa_generate(self):
        pass


class GraniteSpeechForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_path = "ibm-granite/granite-speech-3.3-2b"
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.prompt = self._get_prompt(self.processor.tokenizer)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _get_prompt(self, tokenizer):
        chat = [
            {
                "role": "system",
                "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant",
            },
            {
                "role": "user",
                "content": "<|audio|>can you transcribe the speech into a written format?",
            },
        ]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def _load_datasamples(self, num_samples):
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # automatic decoding with librispeech
        speech_samples = ds.sort("id")[:num_samples]["audio"]

        return [x["array"] for x in speech_samples]

    @slow
    @pytest.mark.skipif(not is_peft_available(), reason="Outputs diverge without lora")
    def test_small_model_integration_test_single(self):
        model = GraniteSpeechForConditionalGeneration.from_pretrained(self.model_path).to(torch_device)
        input_speech = self._load_datasamples(1)

        # Verify feature sizes; note that the feature mask refers to the size of
        # features that are masked into the LLM, not the output of the processor,
        # which is why we inspect the mask instead of the `num_features` tensor.
        inputs = self.processor(self.prompt, input_speech, return_tensors="pt").to(torch_device)

        num_computed_features = self.processor.audio_processor._get_num_audio_features(
            [speech_arr.shape[-1] for speech_arr in input_speech],
        )[0]
        num_actual_features = torch.sum(inputs["input_features_mask"]).item()
        assert num_actual_features == num_computed_features

        # verify generation
        output = model.generate(**inputs, max_new_tokens=32)
        EXPECTED_DECODED_TEXT = "systemKnowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant\nusercan you transcribe the speech into a written format?\nassistantmister quilter is the apostle of the middle classes and we are glad to welcome his gospel"  # fmt: skip

        self.assertEqual(
            self.processor.tokenizer.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @pytest.mark.skipif(not is_peft_available(), reason="Outputs diverge without lora")
    def test_small_model_integration_test_batch(self):
        model = GraniteSpeechForConditionalGeneration.from_pretrained(self.model_path).to(torch_device)
        input_speech = self._load_datasamples(2)
        prompts = [self.prompt, self.prompt]

        # Verify feature sizes & padding
        inputs = self.processor(prompts, input_speech, return_tensors="pt").to(model.device)
        num_computed_features = self.processor.audio_processor._get_num_audio_features(
            [speech_arr.shape[-1] for speech_arr in input_speech],
        )
        num_actual_features = torch.sum(inputs["input_features_mask"], dim=-1)
        for e_feats, a_feats in zip(num_computed_features, num_actual_features):
            assert e_feats == a_feats.item()

        # verify generation
        output = model.generate(**inputs, max_new_tokens=32)

        EXPECTED_DECODED_TEXT = [
            "systemKnowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant\nusercan you transcribe the speech into a written format?\nassistantmister quilter is the apostle of the middle classes and we are glad to welcome his gospel",
            "systemKnowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant\nusercan you transcribe the speech into a written format?\nassistantnor is mister quilter's manner less interesting than his matter"
        ]  # fmt: skip

        self.assertEqual(
            self.processor.tokenizer.batch_decode(output, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
