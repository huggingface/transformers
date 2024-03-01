# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import os
import tempfile
import unittest

from transformers import CharacterBertConfig, is_torch_available
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
)
from transformers.testing_utils import CaptureLogger, require_torch, require_torch_gpu, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from torch import nn

    from transformers.models.character_bert.modeling_character_bert import CharacterCnn

    from transformers import (
        MODEL_FOR_PRETRAINING_MAPPING,
        AutoModel,
        AutoModelForSequenceClassification,
        CharacterBertForMaskedLM,
        CharacterBertForMultipleChoice,
        CharacterBertForNextSentencePrediction,
        CharacterBertForPreTraining,
        CharacterBertForQuestionAnswering,
        CharacterBertForSequenceClassification,
        CharacterBertForTokenClassification,
        CharacterBertModel,
        logging,
    )
    from transformers.models.character_bert.modeling_character_bert import CHARACTER_BERT_PRETRAINED_MODEL_ARCHIVE_LIST


class CharacterBertModelTester:
    def __init__(
        self,
        parent,
        max_word_length=50,
        cnn_filters=[[1, 16], [2, 32]],
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        mlm_vocab_size=999,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.max_word_length = max_word_length
        self.cnn_filters = cnn_filters
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.mlm_vocab_size = mlm_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self):
        num_possible_utf8_bytes = 256
        input_ids = ids_tensor([self.batch_size, self.seq_length, self.max_word_length], num_possible_utf8_bytes + 1)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        """
        Returns a tiny configuration by default.
        """
        return CharacterBertConfig(
            vocab_size=self.vocab_size,
            mlm_vocab_size=self.mlm_vocab_size,
            cnn_filters=self.cnn_filters,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs_for_decoder(self):
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()

        config.is_decoder = True
        encoder_hidden_states = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = CharacterBertModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        result = model(input_ids, token_type_ids=token_type_ids)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_model_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        config.add_cross_attention = True
        model = CharacterBertModel(config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            encoder_hidden_states=encoder_hidden_states,
        )
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def create_and_check_for_causal_lm(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        # NOTE: no current support for generation
        return

    def create_and_check_for_masked_lm(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = CharacterBertForMaskedLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.mlm_vocab_size))

    def create_and_check_model_for_causal_lm_as_decoder(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        # NOTE: no current support for generation
        return

    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ):
        # NOTE: no current support for generation
        return

    def create_and_check_for_next_sequence_prediction(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = CharacterBertForNextSentencePrediction(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=sequence_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, 2))

    def create_and_check_for_pretraining(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = CharacterBertForPreTraining(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            labels=token_labels,
            next_sentence_label=sequence_labels,
        )
        self.parent.assertEqual(
            result.prediction_logits.shape, (self.batch_size, self.seq_length, self.mlm_vocab_size)
        )
        self.parent.assertEqual(result.seq_relationship_logits.shape, (self.batch_size, 2))

    def create_and_check_for_question_answering(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = CharacterBertForQuestionAnswering(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids,
            attention_mask=input_mask,
            token_type_ids=token_type_ids,
            start_positions=sequence_labels,
            end_positions=sequence_labels,
        )
        self.parent.assertEqual(result.start_logits.shape, (self.batch_size, self.seq_length))
        self.parent.assertEqual(result.end_logits.shape, (self.batch_size, self.seq_length))

    def create_and_check_for_sequence_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = CharacterBertForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=sequence_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def create_and_check_for_token_classification(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_labels = self.num_labels
        model = CharacterBertForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, token_type_ids=token_type_ids, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def create_and_check_for_multiple_choice(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        config.num_choices = self.num_choices
        model = CharacterBertForMultipleChoice(config=config)
        model.to(torch_device)
        model.eval()
        multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.num_choices, -1, -1).contiguous()
        multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.num_choices, -1).contiguous()
        result = model(
            multiple_choice_inputs_ids,
            attention_mask=multiple_choice_input_mask,
            token_type_ids=multiple_choice_token_type_ids,
            labels=choice_labels,
        )
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_choices))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class CharacterBertModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            CharacterBertModel,
            CharacterBertForMaskedLM,
            CharacterBertForMultipleChoice,
            CharacterBertForNextSentencePrediction,
            CharacterBertForPreTraining,
            CharacterBertForQuestionAnswering,
            CharacterBertForSequenceClassification,
            CharacterBertForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = ()
    pipeline_model_mapping = (
        {
            "feature-extraction": CharacterBertModel,
            "fill-mask": CharacterBertForMaskedLM,
            "question-answering": CharacterBertForQuestionAnswering,
            "text-classification": CharacterBertForSequenceClassification,
            "token-classification": CharacterBertForTokenClassification,
            "zero-shot": CharacterBertForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_resize_embeddings = False  # NOTE: cannot resize CharacterBERT embeddings

    def setUp(self):
        self.model_tester = CharacterBertModelTester(self)
        self.config_tester = ConfigTester(self, config_class=CharacterBertConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_tie_model_weights(self):
        # NOTE: cannot tie input/output embeddings since the input is handled
        # by a CharacterCNN and the output is a standard matrix
        return

    def test_tied_weights_keys(self):
        # NOTE: cannot tie input/output embeddings since the input is handled
        # by a CharacterCNN and the output is a standard matrix
        return

    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    def test_model_various_embeddings(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        for type in ["absolute", "relative_key", "relative_key_query"]:
            config_and_inputs[0].position_embedding_type = type
            self.model_tester.create_and_check_model(*config_and_inputs)

    def test_model_as_decoder(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_model_as_decoder(*config_and_inputs)

    def test_model_as_decoder_with_default_input_mask(self):
        # This regression test was failing with PyTorch < 1.3
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = self.model_tester.prepare_config_and_inputs_for_decoder()

        input_mask = None

        self.model_tester.create_and_check_model_as_decoder(
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            encoder_hidden_states,
            encoder_attention_mask,
        )

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_for_masked_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_masked_lm(*config_and_inputs)

    def test_for_causal_lm_decoder(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_model_for_causal_lm_as_decoder(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_decoder_model_past_with_large_inputs_relative_pos_emb(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_decoder()
        config_and_inputs[0].position_embedding_type = "relative_key"
        self.model_tester.create_and_check_decoder_model_past_large_inputs(*config_and_inputs)

    def test_for_multiple_choice(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_multiple_choice(*config_and_inputs)

    def test_for_next_sequence_prediction(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_next_sequence_prediction(*config_and_inputs)

    def test_for_pretraining(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_pretraining(*config_and_inputs)

    def test_for_question_answering(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_question_answering(*config_and_inputs)

    def test_for_sequence_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_sequence_classification(*config_and_inputs)

    def test_for_token_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    def test_for_warning_if_padding_and_no_attention_mask(self):
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.model_tester.prepare_config_and_inputs()

        # Set pad tokens in the input_ids
        input_ids[0, 0] = config.pad_token_id

        # Check for warnings if the attention_mask is missing.
        logger = logging.get_logger("transformers.modeling_utils")
        # clear cache so we can test the warning is emitted (from `warning_once`).
        logger.warning_once.cache_clear()

        with CaptureLogger(logger) as cl:
            model = CharacterBertModel(config=config)
            model.to(torch_device)
            model.eval()
            model(input_ids, attention_mask=None, token_type_ids=token_type_ids)
        self.assertIn("We strongly recommend passing in an `attention_mask`", cl.out)

    @slow
    def test_model_from_pretrained(self):
        for model_name in CHARACTER_BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = CharacterBertModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    @slow
    @require_torch_gpu
    def test_torchscript_device_change(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            # CharacterBertForMultipleChoice behaves incorrectly in JIT environments.
            if model_class == CharacterBertForMultipleChoice:
                return

            config.torchscript = True
            model = model_class(config=config)

            inputs_dict = self._prepare_for_class(inputs_dict, model_class)
            traced_model = torch.jit.trace(
                model, (inputs_dict["input_ids"].to("cpu"), inputs_dict["attention_mask"].to("cpu"))
            )

            with tempfile.TemporaryDirectory() as tmp:
                torch.jit.save(traced_model, os.path.join(tmp, "character_bert.pt"))
                loaded = torch.jit.load(os.path.join(tmp, "character_bert.pt"), map_location=torch_device)
                loaded(inputs_dict["input_ids"].to(torch_device), inputs_dict["attention_mask"].to(torch_device))

    def _get_input_ids_and_config(self, batch_size=2):
        # NOTE: this override is necessary because otherwise we look at the last
        # axis for sequence length and get word length instead with CharacterBERT
        # Moreover, the attention mask is a 2D tensor and input ids are 3D, which
        # can also cause some issues.
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict[self.input_name]

        # cut to half length & take max batch_size 3
        sequence_length = input_ids.shape[1] // 2
        input_ids = input_ids[:batch_size, :sequence_length]

        # generate max 3 tokens
        max_length = input_ids.shape[1] + 3
        if config.eos_token_id is not None and config.pad_token_id is None:
            # hack to allow generate for models such as GPT2 as is done in `generate()`
            if isinstance(config.eos_token_id, int):
                config.eos_token_id = [config.eos_token_id]
            config.pad_token_id = config.eos_token_id[0]
        # TransfoXL has no attention mask
        if "transfoxl" in config.__class__.__name__.lower():
            attention_mask = None
        else:
            attention_mask = torch.ones_like(input_ids[:, :, 0], dtype=torch.long)[:batch_size, :sequence_length]

        return config, input_ids, attention_mask, max_length

    def test_assisted_decoding_sample(self):
        # NOTE: generation is not supported
        return

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        # NOTE: needed to adapt this for 3D input ids
        inputs_dict = copy.deepcopy(inputs_dict)
        if model_class.__name__ in get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES):
            inputs_dict = {
                k: (
                    v.unsqueeze(1).expand(-1, self.model_tester.num_choices, -1).contiguous()
                    if isinstance(v, torch.Tensor) and v.ndim > 1 and k != "input_ids"
                    else (
                        v.unsqueeze(1).expand(-1, self.model_tester.num_choices, -1, -1).contiguous()
                        if k == "input_ids"
                        else v
                    )
                )
                for k, v in inputs_dict.items()
            }
            if return_labels:
                inputs_dict["labels"] = torch.ones(self.model_tester.batch_size, dtype=torch.long, device=torch_device)
        else:
            inputs_dict = super()._prepare_for_class(
                inputs_dict=inputs_dict, model_class=model_class, return_labels=return_labels
            )

        if return_labels:
            if model_class in get_values(MODEL_FOR_PRETRAINING_MAPPING):
                inputs_dict["labels"] = torch.zeros(
                    (self.model_tester.batch_size, self.model_tester.seq_length), dtype=torch.long, device=torch_device
                )
                inputs_dict["next_sentence_label"] = torch.zeros(
                    self.model_tester.batch_size, dtype=torch.long, device=torch_device
                )
        return inputs_dict

    def test_load_with_mismatched_shapes(self):
        # NOTE: this is because the test using vocabulary does not raise an error as expected due
        # to the model not having a vocabulary anyway.
        if not self.test_mismatched_shapes:
            return
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            if model_class.__name__ not in get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES):
                continue

            with self.subTest(msg=f"Testing {model_class}"):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model = model_class(config)
                    model.save_pretrained(tmp_dir)

                    # Fails when we don't set ignore_mismatched_sizes=True
                    with self.assertRaises(RuntimeError):
                        new_model = AutoModelForSequenceClassification.from_pretrained(tmp_dir, num_labels=42)
                    # with self.assertRaises(RuntimeError):
                    #     new_model_without_prefix = AutoModel.from_pretrained(tmp_dir, vocab_size=10)

                    logger = logging.get_logger("transformers.modeling_utils")

                    with CaptureLogger(logger) as cl:
                        new_model = AutoModelForSequenceClassification.from_pretrained(
                            tmp_dir, num_labels=42, ignore_mismatched_sizes=True
                        )
                    self.assertIn("the shapes did not match", cl.out)
                    new_model.to(torch_device)
                    inputs = self._prepare_for_class(inputs_dict, model_class)
                    logits = new_model(**inputs).logits
                    self.assertEqual(logits.shape[1], 42)

                    with CaptureLogger(logger) as cl:
                        new_model_without_prefix = AutoModel.from_pretrained(
                            tmp_dir, vocab_size=10, ignore_mismatched_sizes=True
                        )
                    # NOTE: no vocabulary so no problem here
                    # self.assertIn("the shapes did not match", cl.out)

                    input_ids = ids_tensor((2, 8, self.model_tester.max_word_length), 10)
                    new_model_without_prefix.to(torch_device)
                    if self.is_encoder_decoder:
                        new_model_without_prefix(input_ids, decoder_input_ids=input_ids)
                    else:
                        new_model_without_prefix(input_ids)

    def test_model_common_attributes(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (CharacterCnn,))
            model.set_input_embeddings(nn.Embedding(10, 10))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))


@require_torch
class CharacterBertModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_no_head_absolute_embedding(self):
        model = CharacterBertModel.from_pretrained("helboukkouri/character-bert-base-uncased")
        input_ids = torch.tensor(
            [
                [
                    [259, 257, 260, 261, 261, 261, 261, 261, 261, 261] + [261] * (model.config.max_word_length - 10),
                    [259, 81, 98, 115, 106, 116, 260, 261, 261, 261] + [261] * (model.config.max_word_length - 10),
                    [259, 106, 116, 260, 261, 261, 261, 261, 261, 261] + [261] * (model.config.max_word_length - 10),
                    [259, 117, 105, 102, 260, 261, 261, 261, 261, 261] + [261] * (model.config.max_word_length - 10),
                    [259, 100, 98, 113, 106, 117, 98, 109, 260, 261] + [261] * (model.config.max_word_length - 10),
                    [259, 112, 103, 260, 261, 261, 261, 261, 261, 261] + [261] * (model.config.max_word_length - 10),
                    [259, 262, 260, 261, 261, 261, 261, 261, 261, 261] + [261] * (model.config.max_word_length - 10),
                    [259, 258, 260, 261, 261, 261, 261, 261, 261, 261] + [261] * (model.config.max_word_length - 10),
                ]
            ]
        )
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)[0]
        expected_shape = torch.Size([1, 8, 768])
        self.assertEqual(output.shape, expected_shape)
        expected_slice = torch.tensor(
            [[[0.0430, -0.3031, 0.2274], [0.1547, -0.2594, 0.3634], [-0.2890, -0.2062, 0.2870]]]
        )

        self.assertTrue(torch.allclose(output[:, 1:4, 1:4], expected_slice, atol=1e-4))
