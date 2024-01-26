# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from transformers import MT5Config, is_torch_available
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        MT5EncoderModel,
        MT5ForTokenClassification,
    )


# Copied from tests.models.t5.test_modeling_t5.T5EncoderOnlyModelTester with T5->MT5
class MT5EncoderOnlyModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        # For common tests
        use_attention_mask=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        is_training=False,
        dropout_rate=0.1,
        initializer_factor=0.002,
        is_encoder_decoder=False,
        eos_token_id=1,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        # For common tests
        self.seq_length = self.encoder_seq_length
        self.use_attention_mask = use_attention_mask
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
        self.is_encoder_decoder = is_encoder_decoder
        self.scope = None
        self.is_training = is_training

    def get_large_model_config(self):
        return MT5Config.from_pretrained("t5-base")

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)

        config = MT5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )

        return (
            config,
            input_ids,
            attention_mask,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        model = MT5EncoderModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        result = model(input_ids=input_ids)
        encoder_output = result.last_hidden_state

        self.parent.assertEqual(encoder_output.size(), (self.batch_size, self.encoder_seq_length, self.hidden_size))

    def create_and_check_model_fp16_forward(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        model = MT5EncoderModel(config=config).to(torch_device).half().eval()
        output = model(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        self.parent.assertFalse(torch.isnan(output).any().item())

    def create_and_check_with_token_classification_head(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        labels = torch.tensor([1] * self.seq_length * self.batch_size, dtype=torch.long, device=torch_device)
        model = MT5ForTokenClassification(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.seq_length, config.num_labels))
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


# Copied from tests.models.t5.test_modeling_t5.T5EncoderOnlyModelTest with T5->MT5
class MT5EncoderOnlyModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (MT5EncoderModel, MT5ForTokenClassification) if is_torch_available() else ()
    test_pruning = False
    test_resize_embeddings = False
    test_model_parallel = True
    pipeline_model_mapping = (
        {
            "token-classification": MT5ForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    all_parallelizable_model_classes = (MT5EncoderModel,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = MT5EncoderOnlyModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MT5Config, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skipIf(torch_device == "cpu", "Cant do half precision")
    def test_model_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(*config_and_inputs)

    def test_with_token_classification_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_token_classification_head(*config_and_inputs)


@require_torch
@require_sentencepiece
@require_tokenizers
class MT5IntegrationTest(unittest.TestCase):
    @slow
    def test_small_integration_test(self):
        """
        For comparision run:
        >>> import t5  # pip install t5==0.7.1
        >>> from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary

        >>> path_to_mtf_small_mt5_checkpoint = '<fill_in>'
        >>> path_to_mtf_small_mt5_spm_model_path = '<fill_in>'
        >>> t5_model = t5.models.MtfModel(model_dir=path_to_mtf_small_mt5_checkpoint, batch_size=1, tpu=None)
        >>> vocab = SentencePieceVocabulary(path_to_mtf_small_mt5_spm_model_path)
        >>> score = t5_model.score(inputs=["Hello there"], targets=["Hi I am"], vocabulary=vocab)
        """

        model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small", return_dict=True).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

        input_ids = tokenizer("Hello there", return_tensors="pt").input_ids
        labels = tokenizer("Hi I am", return_tensors="pt").input_ids

        loss = model(input_ids.to(torch_device), labels=labels.to(torch_device)).loss
        mtf_score = -(labels.shape[-1] * loss.item())

        EXPECTED_SCORE = -84.9127
        self.assertTrue(abs(mtf_score - EXPECTED_SCORE) < 1e-4)
