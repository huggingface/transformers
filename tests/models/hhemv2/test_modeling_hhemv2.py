# coding=utf-8
# Copyright 2024 Google HHEMv2 Authors and HuggingFace Inc. team.
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


import os
import unittest

from transformers import HHEMv2Config, is_torch_available
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import cached_property, is_torch_fx_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_fx_available():
    pass


if is_torch_available():
    import torch

    from transformers import (
        HHEMv2Model,
        T5Tokenizer,
    )


class HHEMv2ModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        decoder_seq_length=7,
        # For common tests
        is_training=False,
        use_attention_mask=True,
        use_labels=False,
        hidden_size=32,
        num_hidden_layers=12,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        dropout_rate=0.1,
        initializer_factor=0.002,
        eos_token_id=1,
        pad_token_id=0,
        decoder_start_token_id=0,
        scope=None,
        decoder_layers=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.scope = scope
        self.decoder_layers = decoder_layers

    def get_large_model_config(self):
        return HHEMv2Config.from_pretrained("vectara/hallucination_evaluation_model")

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size).clamp(2)
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)

        config = self.get_config()

        return (
            config,
            input_ids,
            attention_mask,
        )

    def get_pipeline_config(self):
        return HHEMv2Config(
            vocab_size=166,  # hhemv2 forces 100 extra tokens
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

    def get_config(self):
        return HHEMv2Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

    def create_and_check_with_sequence_classification_head(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        # labels = torch.tensor([1] * self.batch_size, dtype=torch.long, device=torch_device)
        labels = torch.tensor(
            [[1] + [0] * (input_ids.shape[1] - 1)] * self.batch_size, dtype=torch.long, device=torch_device
        )
        model = HHEMv2Model(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, config.num_labels))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class HHEMv2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (HHEMv2Model,) if is_torch_available() else ()
    all_generative_model_classes = ()
    all_parallelizable_model_classes = ()
    pipeline_model_mapping = (
        {
            "text-classification": HHEMv2Model,
        }
        if is_torch_available()
        else {}
    )

    test_pruning = False
    test_mismatched_shapes = False
    test_model_parallel = True  # False
    # The small HHEMv2 model needs higher percentages for CPU/MP tests
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = HHEMv2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=HHEMv2Config, d_model=37)

    def test_with_sequence_classification_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_sequence_classification_head(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        model_name = "vectara/hallucination_evaluation_model"
        model = HHEMv2Model.from_pretrained(model_name)
        self.assertIsNotNone(model)


@require_torch
@require_sentencepiece
@require_tokenizers
class HHEMv2ModelIntegrationTests(unittest.TestCase):
    @cached_property
    def local_model_path(self):
        return os.path.join(
            os.path.dirname(
                (os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
            ),
            "hallucination_evaluation_model",
        )

    @cached_property
    def model(self):
        return HHEMv2Model.from_pretrained(self.local_model_path).to(torch_device).eval()

    @cached_property
    def config(self):
        return HHEMv2Config.from_pretrained(self.local_model_path)

    @cached_property
    def tokenizer(self):
        config = self.config
        return T5Tokenizer.from_pretrained(config.foundation)

    @slow
    def test_small_prediction(self):
        model = self.model
        pairs = [("The capital of France is Berlin.", "The capital of France is Paris.")]
        score = model.predict(pairs).item()
        self.assertTrue(round(score, 4) == 0.0111)

    @slow
    def test_prediction(self):
        model = self.model
        pairs = [  # Test data, List[Tuple[str, str]]
            ("The capital of France is Berlin.", "The capital of France is Paris."),  # factual but hallucinated
            ("I am in California", "I am in United States."),  # Consistent
            ("I am in United States", "I am in California."),  # Hallucinated
            ("A person on a horse jumps over a broken down airplane.", "A person is outdoors, on a horse."),
            (
                "A boy is jumping on skateboard in the middle of a red bridge.",
                "The boy skates down the sidewalk on a red bridge",
            ),
            (
                "A man with blond-hair, and a brown shirt drinking out of a public water fountain.",
                "A blond man wearing a brown shirt is reading a book.",
            ),
            ("Mark Wahlberg was a fan of Manny.", "Manny was a fan of Mark Wahlberg."),
        ]
        preds = model.predict(pairs).tolist()
        expected = [0.0111, 0.6474, 0.1290, 0.8969, 0.1846, 0.0050, 0.0543]
        self.assertListEqual(
            expected,
            [round(pred, 4) for pred in preds],
        )
