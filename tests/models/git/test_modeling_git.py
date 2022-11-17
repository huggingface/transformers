# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from transformers import GITConfig, is_torch_available
from transformers.models.auto import get_values
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import MODEL_FOR_PRETRAINING_MAPPING, GITForCausalLM, GITModel
    from transformers.models.git.modeling_git import GIT_PRETRAINED_MODEL_ARCHIVE_LIST


class GITModelTester:
    def __init__(
        self,
        parent,
        num_channels=3,
        image_size=32,
        patch_size=16,
        batch_size=13,
        text_seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.text_seq_length = text_seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope

        # for GIT, the sequence length is the sum of the text and patch tokens, + 1 due to the CLS token
        self.seq_length = self.text_seq_length + int((self.image_size / self.patch_size) ** 2) + 1

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.text_seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.text_seq_length])

        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        token_labels = None
        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.text_seq_length], self.num_labels)

        config = self.get_config()

        return config, input_ids, input_mask, pixel_values, token_labels

    def get_config(self):
        """
        Returns a tiny configuration by default.
        """
        return GITConfig(
            vision_config={
                "num_channels": self.num_channels,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
            },
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, input_ids, input_mask, pixel_values, token_labels):
        model = GITModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, pixel_values=pixel_values)

        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

         # TODO support inference without pixel values

    def create_and_check_for_causal_lm(self, config, input_ids, input_mask, pixel_values, token_labels):
        model = GITForCausalLM(config=config)
        model.to(torch_device)
        model.eval()

        # inference
        # TODO support inference without pixel values
        result = model(input_ids, attention_mask=input_mask, pixel_values=pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

        # TODO training
        # result = model(input_ids, attention_mask=input_mask, pixel_values=pixel_values)
        # self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))
        # self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()

        (
            config,
            input_ids,
            input_mask,
            pixel_values,
            token_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "pixel_values": pixel_values,
        }

        return config, inputs_dict


@require_torch
class GITModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):

    all_model_classes = (GITModel, GITForCausalLM) if is_torch_available() else ()
    all_generative_model_classes = (GITForCausalLM,) if is_torch_available() else ()
    fx_compatible = False
    test_torchscript = False

    # special case for ForPreTraining model
    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if return_labels:
            if model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
                inputs_dict["next_sentence_label"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
        return inputs_dict

    def setUp(self):
        self.model_tester = GITModelTester(self)
        self.config_tester = ConfigTester(self, config_class=GITConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in GIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = GITModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class GITModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head_absolute_embedding(self):
        processor = GITProcessor.from_pretrained("nielsr/git-base")
        model = GITForCausalLM.from_pretrained("nielsr/git-base")



        raise NotImplementedError("To do")
