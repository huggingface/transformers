# Copyright 2026 the HuggingFace Team. All rights reserved.
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

from transformers import AutoConfig, Gliner2Config, is_torch_available
from transformers.testing_utils import require_torch, torch_device
from transformers.utils import is_tokenizers_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import Gliner2ForSchemaExtraction, Gliner2Model


def _tiny_encoder(vocab_size=128, hidden=32):
    return AutoConfig.for_model(
        "deberta-v2",
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        type_vocab_size=0,
    )


class Gliner2ModelTester:
    def __init__(self, parent, batch_size=2, seq_length=8, vocab_size=128, hidden_size=32, architecture="span"):
        self.parent = parent
        self.is_training = False
        self.num_hidden_layers = 1
        self.num_attention_heads = 4
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.architecture = architecture

    def get_config(self):
        encoder = _tiny_encoder(self.vocab_size, self.hidden_size)
        encoder._attn_implementation = "eager"
        boundary = None
        if self.architecture == "boundary":
            boundary = {
                "boundary_dim": 16,
                "pair_dim": 16,
                "boundary_attention_layers": 0,
                "candidate_attention_layers": 0,
                "query_attention_layers": 0,
                "candidate_pool": "per_query",
                "enable_records": False,
                "enable_relations": False,
                "pool_size": 8,
                "candidate_budget": 8,
                "start_top_k": 2,
                "end_top_k": 2,
            }
        return Gliner2Config(
            encoder_config=encoder,
            architecture=self.architecture,
            max_width=4,
            counting_layer="count_lstm",
            boundary_config=boundary,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(self.batch_size, self.seq_length, dtype=torch.long)
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}

    def create_and_check_model(self, config, input_ids, attention_mask):
        model = Gliner2Model(config).to(torch_device).eval()
        result = model(input_ids=input_ids, attention_mask=attention_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))


@require_torch
class Gliner2ModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Gliner2Model, Gliner2ForSchemaExtraction) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": Gliner2Model, "schema-extraction": Gliner2ForSchemaExtraction}
        if is_torch_available()
        else {}
    )
    test_resize_embeddings = False
    test_torchscript = False
    test_pruning = False
    test_head_masking = False
    test_mismatched_shapes = False

    def setUp(self):
        self.model_tester = Gliner2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Gliner2Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(config, input_ids.to(torch_device), attention_mask.to(torch_device))

    def test_span_scores(self):
        config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = Gliner2ForSchemaExtraction(config).eval()
        batch, length = input_ids.shape
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            text_word_indices=torch.tensor([[1, 3]]).expand(batch, -1),
            text_word_mask=torch.ones(batch, 2, dtype=torch.bool),
            query_marker_indices=torch.tensor([[0, 2]]).expand(batch, -1),
            query_marker_mask=torch.ones(batch, 2, dtype=torch.bool),
            query_group_index=torch.zeros(batch, 2, dtype=torch.long),
            cls_marker_indices=torch.zeros(batch, 1, dtype=torch.long),
            cls_marker_mask=torch.zeros(batch, 1, dtype=torch.bool),
            cls_group_index=torch.zeros(batch, 1, dtype=torch.long),
            prompt_marker_indices=torch.zeros(batch, 1, dtype=torch.long),
            prompt_marker_mask=torch.ones(batch, 1, dtype=torch.bool),
            prompt_group_index=torch.zeros(batch, 1, dtype=torch.long),
            task_type_ids=torch.ones(batch, 1, dtype=torch.long),
            group_mask=torch.ones(batch, 1, dtype=torch.bool),
        )
        self.assertEqual(len(out.span_logits), batch)
        self.assertEqual(out.span_logits[0][0].shape[1], 2)
        names = set(model.state_dict())
        self.assertTrue(any(name.startswith("span_rep.span_rep_layer.") for name in names))
        self.assertTrue(any(name.startswith("count_pred.") for name in names))
        self.assertTrue(any(name.startswith("classifier.") for name in names))

    def test_boundary_head_names(self):
        tester = Gliner2ModelTester(self, architecture="boundary")
        config = tester.get_config()
        model = Gliner2ForSchemaExtraction(config)
        names = set(model.state_dict())
        self.assertTrue(any(name.startswith("boundary_head.") for name in names))
        self.assertIsNone(model.record_decoder)


@require_torch
class Gliner2ConfigTest(unittest.TestCase):
    def test_span_defaults(self):
        config = Gliner2Config(encoder_config=_tiny_encoder())
        self.assertEqual(config.architecture, "span")
        self.assertEqual(config.model_type, "gliner2")
        self.assertIsNone(config.boundary_config)

    def test_boundary_subconfig(self):
        config = Gliner2Config(
            encoder_config=_tiny_encoder(), architecture="boundary", boundary_config={"pool_size": 32}
        )
        self.assertEqual(config.boundary_config.pool_size, 32)
        self.assertEqual(config.boundary_config.model_type, "gliner2_boundary")

    def test_published_keys(self):
        config = Gliner2Config(
            encoder_config=_tiny_encoder(),
            architecture="boundary",
            model_type="extractor",
            max_len=None,
            boundary_head={"pool_size": 16, "abstention_loss_weight": 0.2},
            span_head={"span_mode": "markerV0", "max_width": 4},
        )
        self.assertIsNone(config.max_len)
        self.assertEqual(config.boundary_config.pool_size, 16)
        self.assertEqual(config.boundary_config.abstention_loss_weight, 0.2)
        saved = config.to_dict()
        self.assertEqual(saved["boundary_config"]["pool_size"], 16)
        self.assertEqual(saved["span_head"]["span_mode"], "markerV0")
        self.assertEqual(saved["model_type"], "gliner2")
        with self.assertRaises(ValueError):
            Gliner2Config(encoder_config=_tiny_encoder(), use_moe=True)


def _boundary_batch():
    return {
        "input_ids": torch.randint(1, 50, (1, 8)),
        "attention_mask": torch.ones(1, 8, dtype=torch.long),
        "text_word_indices": torch.tensor([[1, 2, 3, 4]]),
        "text_word_mask": torch.ones(1, 4, dtype=torch.bool),
        "query_marker_indices": torch.tensor([[0]]),
        "query_marker_mask": torch.ones(1, 1, dtype=torch.bool),
        "query_group_index": torch.zeros(1, 1, dtype=torch.long),
        "cls_marker_indices": torch.tensor([[5]]),
        "cls_marker_mask": torch.ones(1, 1, dtype=torch.bool),
        "cls_group_index": torch.zeros(1, 1, dtype=torch.long),
        "task_type_ids": torch.tensor([[1]]),
        "group_mask": torch.ones(1, 1, dtype=torch.bool),
    }


def _tiny_boundary_model(candidate_pool="per_query"):
    boundary = {
        "boundary_dim": 16,
        "pair_dim": 16,
        "boundary_attention_layers": 0,
        "candidate_attention_layers": 0,
        "query_attention_layers": 0,
        "candidate_pool": candidate_pool,
        "enable_records": False,
        "enable_relations": False,
        "pool_size": 4,
        "candidate_budget": 4,
        "training_candidate_budget": 4,
        "start_top_k": 2,
        "end_top_k": 2,
        "ends_per_start": 2,
        "starts_per_end": 2,
    }
    config = Gliner2Config(encoder_config=_tiny_encoder(), architecture="boundary", boundary_config=boundary)
    return Gliner2ForSchemaExtraction(config)


@require_torch
class Gliner2ProposalRecallTest(unittest.TestCase):
    def _assert_eval_labels_keep_inference_candidates(self, candidate_pool):
        torch.manual_seed(0)
        model = _tiny_boundary_model(candidate_pool).eval()
        batch = _boundary_batch()
        labels = {
            "mention_pairs": torch.tensor([[[[0, 1], [0, 2]]]]),
            "mention_mask": torch.tensor([[[True, True]]]),
        }
        bare = model(**batch)
        labeled = model(**batch, labels=labels)
        again = model(**batch)
        self.assertIsNone(bare.loss)
        self.assertNotIn("metrics", bare)
        self.assertEqual(set(bare.keys()), set(again.keys()))
        self.assertTrue(set(labeled.keys()) <= set(bare.keys()) | {"loss", "losses"})
        torch.testing.assert_close(bare.boundary.candidates.pair_logits, labeled.boundary.candidates.pair_logits)
        self.assertTrue(torch.equal(bare.boundary.candidates.indices, labeled.boundary.candidates.indices))
        self.assertTrue(torch.equal(bare.boundary.candidates.valid_mask, labeled.boundary.candidates.valid_mask))

    def test_eval_labels_expose_recall_without_changing_candidates(self):
        self._assert_eval_labels_keep_inference_candidates("per_query")

    def test_shared_pool_eval_labels_do_not_change_candidates(self):
        self._assert_eval_labels_keep_inference_candidates("shared")


def _word_tokenizer():
    if not is_tokenizers_available():
        return None
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    from transformers import PreTrainedTokenizerFast

    specials = [
        "[PAD]",
        "[UNK]",
        "[SEP_STRUCT]",
        "[SEP_TEXT]",
        "[P]",
        "[C]",
        "[E]",
        "[R]",
        "[L]",
        "[EXAMPLE]",
        "[OUTPUT]",
        "[DESCRIPTION]",
        "(",
        ")",
        ",",
        "|",
    ]
    words = ["ada", "lovelace", "wrote", "notes", "person", "place", "label"]
    vocab = {token: index for index, token in enumerate(specials + words)}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


@unittest.skipUnless(is_tokenizers_available() and is_torch_available(), "tokenizers and torch are required")
class Gliner2ProcessorTest(unittest.TestCase):
    def test_entity_ids_and_postprocess(self):
        from transformers import Gliner2ForSchemaExtraction, Gliner2Processor

        tokenizer = _word_tokenizer()
        processor = Gliner2Processor(tokenizer, word_splitter="whitespace")
        schema = {"entities": {"person": {}}}
        encoded = processor("Ada Lovelace wrote notes", schema=schema, return_tensors="pt")
        self.assertIn("input_ids", encoded)
        self.assertGreater(int(encoded["text_word_mask"].sum()), 0)
        self.assertTrue(bool(encoded["prompt_marker_mask"].any()))
        config = Gliner2Config(encoder_config=_tiny_encoder(vocab_size=tokenizer.vocab_size), max_width=4)
        model = Gliner2ForSchemaExtraction(config).eval()
        model_inputs = {key: value for key, value in encoded.items() if key != "metadata"}
        outputs = model(**model_inputs)
        decoded = processor.post_process_extraction(outputs, encoded["metadata"], threshold=0.5)
        self.assertEqual(len(decoded), 1)
        self.assertIn("entities", decoded[0])
