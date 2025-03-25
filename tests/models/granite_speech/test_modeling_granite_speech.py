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

from transformers import (
    GraniteSpeechConfig,
    GraniteSpeechForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    require_torch_sdpa,
    torch_device,
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
            "downsample_rate": 5,
            "encoder_hidden_size": 32,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 32,
            "initializer_range": 0.02,
            "intermediate_size": 256,
            "layer_norm_eps": 1e-12,
            "llm_dim": 32,
            "max_position_embeddings": 2048,
            "model_type": "granite_speech_qformer",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "position_embedding_type": "absolute",
            "use_qformer_text_input": False,
            "vocab_size": 30522,
            "window_size": 15,
        },
        audio_token_index=0,
        tie_word_embeddings=True,
        initializer_range=0.02,
        has_lora_adapter=True,
        is_training=True,
    ):
        self.parent = parent
        self.projector_config = None
        self.encoder_config = encoder_config
        self.text_config = text_config
        self.projector_config = projector_config
        self.audio_token_index = audio_token_index
        self.tie_word_embeddings = tie_word_embeddings
        self.initializer_range = initializer_range
        self.has_lora_adapater = has_lora_adapter
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
        config.torch_dtype = torch.float16
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
