# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import json
import os
import tempfile
import unittest
from pathlib import Path

from huggingface_hub import hf_hub_download

from transformers import AutoConfig, CharacterBertConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForMaskedLM,
        AutoTokenizer,
        CharacterBertForMaskedLM,
        CharacterBertForQuestionAnswering,
        CharacterBertForSequenceClassification,
        CharacterBertForTokenClassification,
        CharacterBertModel,
    )


class CharacterBertModelTester:
    def __init__(
        self,
        parent,
        batch_size=4,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        max_characters_per_token=12,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        type_vocab_size=2,
        num_labels=3,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.max_characters_per_token = max_characters_per_token
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.type_vocab_size = type_vocab_size
        self.num_labels = num_labels

    def get_config(self):
        return CharacterBertConfig(
            vocab_size=99,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            type_vocab_size=self.type_vocab_size,
            max_characters_per_token=self.max_characters_per_token,
            character_embedding_dim=8,
            character_cnn_filters=((1, 8), (2, 8), (3, 16)),
            num_highway_layers=1,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor(
            [self.batch_size, self.seq_length, self.max_characters_per_token],
            config.character_vocab_size + 1,
        )
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = ids_tensor([self.batch_size], self.num_labels)
        token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
        return config, input_ids, attention_mask, token_type_ids, sequence_labels, token_labels

    def create_and_check_model(self, config, input_ids, attention_mask, token_type_ids):
        model = CharacterBertModel(config=config)
        model.to(torch_device)
        model.eval()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        self.parent.assertEqual(
            outputs.last_hidden_state.shape,
            (self.batch_size, self.seq_length, self.hidden_size),
        )
        self.parent.assertEqual(outputs.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_sequence_classification(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
        sequence_labels,
    ):
        config.num_labels = self.num_labels
        model = CharacterBertForSequenceClassification(config=config)
        model.to(torch_device)
        model.eval()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
        )

        self.parent.assertEqual(outputs.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_masked_lm(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
    ):
        model = CharacterBertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()

        masked_lm_labels = ids_tensor([self.batch_size, self.seq_length], config.vocab_size)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=masked_lm_labels,
        )

        self.parent.assertEqual(outputs.logits.shape, (self.batch_size, self.seq_length, config.vocab_size))

    def create_and_check_for_token_classification(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
        token_labels,
    ):
        config.num_labels = self.num_labels
        model = CharacterBertForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
        )

        self.parent.assertEqual(
            outputs.logits.shape,
            (self.batch_size, self.seq_length, self.num_labels),
        )

    def create_and_check_for_question_answering(
        self,
        config,
        input_ids,
        attention_mask,
        token_type_ids,
        sequence_labels,
    ):
        model = CharacterBertForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )

        self.parent.assertEqual(outputs.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(outputs.end_logits.shape, (self.batch_size, self.seq_length))

    def prepare_config_and_inputs_for_common(self):
        (
            config,
            input_ids,
            attention_mask,
            token_type_ids,
            _,
            _,
        ) = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        return config, inputs_dict


@require_torch
class CharacterBertModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            CharacterBertModel,
            CharacterBertForMaskedLM,
            CharacterBertForQuestionAnswering,
            CharacterBertForSequenceClassification,
            CharacterBertForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    model_split_percents = [0.5, 0.999, 0.9995]
    test_mismatched_shapes = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = CharacterBertModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=CharacterBertConfig,
            hidden_size=37,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, input_ids, attention_mask, token_type_ids, _, _ = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, input_ids, attention_mask, token_type_ids)

    def test_for_sequence_classification(self):
        config, input_ids, attention_mask, token_type_ids, sequence_labels, _ = (
            self.model_tester.prepare_config_and_inputs()
        )
        self.model_tester.create_and_check_for_sequence_classification(
            config,
            input_ids,
            attention_mask,
            token_type_ids,
            sequence_labels,
        )

    def test_for_masked_lm(self):
        config, input_ids, attention_mask, token_type_ids, _, _ = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(
            config,
            input_ids,
            attention_mask,
            token_type_ids,
        )

    def test_character_vocab_size_validation(self):
        with self.assertRaisesRegex(ValueError, "`character_vocab_size` must be 262"):
            CharacterBertConfig(character_vocab_size=300)

    def test_max_characters_per_token_validation(self):
        with self.assertRaisesRegex(ValueError, "`max_characters_per_token` must be at least the width"):
            CharacterBertConfig(max_characters_per_token=6)

    def test_custom_character_cnn_filters_allow_shorter_tokens(self):
        config = CharacterBertConfig(
            vocab_size=99,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=37,
            type_vocab_size=2,
            max_characters_per_token=3,
            character_embedding_dim=8,
            character_cnn_filters=((1, 8), (2, 8), (3, 16)),
            num_highway_layers=1,
        )
        model = CharacterBertModel(config).to(torch_device)
        input_ids = ids_tensor(
            [self.model_tester.batch_size, self.model_tester.seq_length, 3], config.character_vocab_size
        )
        outputs = model(input_ids=input_ids)

        self.assertEqual(
            outputs.last_hidden_state.shape,
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.hidden_size),
        )

    def test_legacy_checkpoint_config_fields_for_masked_lm(self):
        legacy_config = {
            "model_type": "character_bert",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "character_embeddings_dim": 11,
            "cnn_filters": [[1, 5], [2, 7], [3, 9]],
            "max_word_length": 9,
            "mlm_vocab_size": 123,
            "tie_word_embeddings": False,
            "cnn_activation": "relu",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = os.path.join(tmp_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as writer:
                json.dump(legacy_config, writer)

            config = AutoConfig.from_pretrained(tmp_dir)

        self.assertIsInstance(config, CharacterBertConfig)
        self.assertEqual(config.character_embedding_dim, legacy_config["character_embeddings_dim"])
        self.assertEqual(config.character_cnn_filters, tuple(tuple(x) for x in legacy_config["cnn_filters"]))
        self.assertEqual(config.max_characters_per_token, legacy_config["max_word_length"])
        self.assertEqual(config.vocab_size, legacy_config["mlm_vocab_size"])
        self.assertEqual(config.mlm_vocab_size, legacy_config["mlm_vocab_size"])

        model = CharacterBertForMaskedLM(config).to(torch_device)
        self.assertEqual(model.cls.predictions.decoder.out_features, legacy_config["mlm_vocab_size"])

    def test_for_token_classification(self):
        (
            config,
            input_ids,
            attention_mask,
            token_type_ids,
            _,
            token_labels,
        ) = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(
            config,
            input_ids,
            attention_mask,
            token_type_ids,
            token_labels,
        )

    def test_for_question_answering(self):
        config, input_ids, attention_mask, token_type_ids, sequence_labels, _ = (
            self.model_tester.prepare_config_and_inputs()
        )
        self.model_tester.create_and_check_for_question_answering(
            config,
            input_ids,
            attention_mask,
            token_type_ids,
            sequence_labels,
        )

    @unittest.skip(reason="CharacterBERT uses a CharacterCNN input embedder instead of `nn.Embedding`.")
    def test_model_get_set_embeddings(self):
        pass


@require_torch
class CharacterBertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_masked_lm_predictions(self):
        model_id = os.environ.get("CHARACTER_BERT_INTEGRATION_MODEL", "helboukkouri/character-bert-base-uncased")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForMaskedLM.from_pretrained(model_id).to("cpu")
        model.eval()

        inputs = tokenizer("paris is the capital of [MASK].", return_tensors="pt").to("cpu")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
        mask_index = tokens.index(tokenizer.mask_token)

        with torch.no_grad():
            probs = model(**inputs).logits[0, mask_index].float().softmax(dim=-1)

        local_model_path = Path(model_id)
        if local_model_path.is_dir():
            mlm_vocab_path = local_model_path / "mlm_vocab.txt"
        else:
            mlm_vocab_path = Path(hf_hub_download(repo_id=model_id, filename="mlm_vocab.txt"))
        with open(mlm_vocab_path, encoding="utf-8") as reader:
            mlm_vocab = [line.rstrip("\n") for line in reader]

        top_probs, top_indices = torch.topk(probs, k=5)
        top_tokens = [mlm_vocab[i] for i in top_indices.tolist()]
        self.assertEqual(top_tokens, ["france", "monaco", "algeria", "senegal", "haiti"])

        expected_top_probs = torch.tensor([0.5094, 0.0324, 0.0317, 0.0196, 0.0185], dtype=torch.float32)
        torch.testing.assert_close(top_probs, expected_top_probs, rtol=1e-3, atol=1e-4)
