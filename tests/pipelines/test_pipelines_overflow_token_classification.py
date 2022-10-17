"""Tests covering the overflow token classification NER pipeline"""
from typing import Dict, List
import unittest
from unittest.mock import patch

from parameterized import parameterized

from transformers.pipelines.overflow_token_classification import (
    OverflowTokenClassificationPipeline,
    ReconstitutionStrategy,
)
from transformers.testing_utils import nested_simplify, require_torch_gpu
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER
from transformers import (
    BertForTokenClassification,
    BertTokenizerFast
)

from .test_pipelines_common import PipelineTestCaseMeta

MODEL_NAME = "hf-internal-testing/tiny-bert-for-token-classification"
MODEL = BertForTokenClassification.from_pretrained(MODEL_NAME)
TOKENIZER = BertTokenizerFast.from_pretrained(MODEL_NAME)

def get_test_pipeline():
        pipeline = OverflowTokenClassificationPipeline(
            model=MODEL, tokenizer=TOKENIZER, framework="pt"
        )
        return pipeline

class OverflowTokenClassificationPipelineTests(unittest.TestCase, metaclass=PipelineTestCaseMeta):
    @require_torch_gpu
    def test_gpu(self):
        sentence = "This is dummy sentence"
        ner = OverflowTokenClassificationPipeline(
            model=MODEL,
            tokenizer=TOKENIZER,
            device=0,
        )

        output = ner(sentence)
        self.assertEqual(nested_simplify(output), [])

    def test_empty(self):
        """Test the results when passing an empty sentence"""
        pipeline = get_test_pipeline()
        output = pipeline("")
        self.assertEqual(output, [])
        output = pipeline([""])
        self.assertEqual(output, [[]])
        with self.assertRaises(ValueError):
            pipeline([])

    def test_simple_str(self, overflow_short: str, overflow_short_output):
        """Test the expected tokens with simple string inputs"""
        pipeline = get_test_pipeline()
        outputs = pipeline(overflow_short)
        self.assertEqual(nested_simplify(outputs), overflow_short_output)

    def test_simple_list(
        self,
        overflow_short: str,
        overflow_short_output,
        overflow_batch,
        overflow_batch_output,
    ):
        """Test the expected tokens with simple list inputs"""
        pipeline = get_test_pipeline()
        outputs = pipeline([overflow_short])
        self.assertEqual(nested_simplify(outputs), [overflow_short_output])
        outputs = pipeline(overflow_batch)
        self.assertEqual(nested_simplify(outputs), overflow_batch_output)

    def test_batching(self, overflow_batch):
        """Assert that batching sentences does not change the output shape or
        results"""
        pipeline = get_test_pipeline()
        single_output = pipeline(overflow_batch, batch_size=1)
        batched_output = pipeline(overflow_batch, batch_size=2)
        self.assertEqual(single_output, batched_output)

    def test_simple_reconstruction(self, overflow_long: str):
        """Assert that truncated sentences are reconstructed into original
        input sentences"""
        expected_tokens = [token.lower() for token in overflow_long.split()]
        num_tokens = len(expected_tokens)
        max_seq_len = num_tokens - 1
        ner = OverflowTokenClassificationPipeline(
            model=MODEL,
            tokenizer=TOKENIZER,
            max_seq_len=max_seq_len,
            stride=2,
        )
        outputs = ner(overflow_long)
        self.assertEqual(len(outputs), num_tokens)
        output_words = [token["word"].lower() for token in outputs]
        self.assertEqual(output_words, expected_tokens)
        # Check index, start and end tokens are all ascending
        for i, token in enumerate(outputs):
            self.assertEqual(token["index"], i + 1)
            try:
                next_token = outputs[i + 1]
                self.assertGreater(next_token["start"], token["start"])
                self.assertGreater(next_token["end"], token["end"])
            except IndexError:
                break

    @parameterized.expand(
        [
            (
                [{"entity": "O", "score": 0.5}],
                [{"entity": "B-PER", "score": 0.6}],
                ReconstitutionStrategy.FIRST,
                "",
                [{"entity": "O", "score": 0.5}],
            ),
            (
                [{"entity": "O", "score": 0.6}],
                [{"entity": "B-PER", "score": 0.5}],
                ReconstitutionStrategy.ENTITIES,
                "",
                [{"entity": "B-PER", "score": 0.5}],
            ),
            (
                [{"entity": "B-LOC", "score": 0.5}],
                [{"entity": "B-PER", "score": 0.6}],
                ReconstitutionStrategy.ENTITIES,
                "",
                [{"entity": "B-PER", "score": 0.6}],
            ),
            (
                [{"entity": "B-LOC", "score": 0.5}],
                [{"entity": "B-PER", "score": 0.5}],
                ReconstitutionStrategy.ENTITIES,
                "",
                [{"entity": "B-LOC", "score": 0.5}],
            ),
            (
                [{"entity": "O", "score": 0.6}],
                [{"entity": "B-PER", "score": 0.5}],
                ReconstitutionStrategy.FIRST_ENTITIES,
                "",
                [{"entity": "B-PER", "score": 0.5}],
            ),
            (
                [{"entity": "B-LOC", "score": 0.5}],
                [{"entity": "B-PER", "score": 0.6}],
                ReconstitutionStrategy.FIRST_ENTITIES,
                "",
                [{"entity": "B-LOC", "score": 0.5}],
            ),
            (
                [{"entity": "I-PER", "score": 0.5}],
                [{"entity": "B-LOC", "score": 0.6}],
                ReconstitutionStrategy.ENTITIES,
                "B-PER",
                [{"entity": "I-PER", "score": 0.5}],
            ),
            # TODO This feels like it should not occur, or should be treated as the
            #  exception to the continuation rule as the first segment has more
            #  prior context
            (
                [{"entity": "B-LOC", "score": 0.5}],
                [{"entity": "I-PER", "score": 0.6}],
                ReconstitutionStrategy.FIRST_ENTITIES,
                "B-PER",
                [{"entity": "I-PER", "score": 0.6}],
            ),
            (
                [{"entity": "B-PER", "score": 0.5}],
                [{"entity": "O", "score": 0.6}],
                ReconstitutionStrategy.MAX_SCORE,
                "",
                [{"entity": "O", "score": 0.6}],
            ),
            (
                [{"entity": "B-PER", "score": 0.5}, {"entity": "I-PER", "score": 0.5}],
                [{"entity": "O", "score": 0.6}, {"entity": "B-LOC", "score": 0.6}],
                ReconstitutionStrategy.ENTITIES,
                "",
                [{"entity": "B-PER", "score": 0.5}, {"entity": "I-PER", "score": 0.5}],
            ),
            (
                [{"entity": "I-PER", "score": 0.5}, {"entity": "I-PER", "score": 0.5}],
                [{"entity": "B-LOC", "score": 0.6}, {"entity": "I-LOC", "score": 0.6}],
                ReconstitutionStrategy.ENTITIES,
                "I-PER",
                [{"entity": "I-PER", "score": 0.5}, {"entity": "I-PER", "score": 0.5}],
            ),
            (
                [{"entity": "O", "score": 0.6}, {"entity": "B-LOC", "score": 0.5}],
                [{"entity": "B-PER", "score": 0.5}, {"entity": "I-PER", "score": 0.5}],
                ReconstitutionStrategy.FIRST_ENTITIES,
                "",
                [{"entity": "B-PER", "score": 0.5}, {"entity": "I-PER", "score": 0.5}],
            ),
        ],
    )
    def test_reconstitution_strategies(
        self,
        first: List[Dict],
        second: List[Dict],
        strategy: ReconstitutionStrategy,
        prev_ent: str,
        expected_result: List[Dict],
    ):
        """Assert that different strategies return different tokens
        This only uses the minimal token values required to perform
        reconstitution, rather than full entity dicts which would contain
        duplicated values"""
        pipeline = get_test_pipeline()
        result = pipeline.combine_overlapping_tokens(first, second, strategy, prev_ent)
        self.assertEqual(result, expected_result)

    def test_invalid_prev_entity(self):
        """Assert that a previous entity tag that cannot be found when combining
        segments that have no previous entity token"""
        pipeline = get_test_pipeline()
        first_segment = [
            {
                "entity": "O",
                "score": 0.6,
                "start": 0,
                "end": 5,
                "word": "hello",
                "index": 1,
            }
        ]
        second_segment = [
            {
                "entity": "B-PER",
                "score": 0.7,
                "start": 0,
                "end": 5,
                "word": "hello",
                "index": 1,
            }
        ]
        expected_result = [
            {
                "entity": "B-PER",
                "score": 0.7,
                "start": 0,
                "end": 5,
                "word": "hello",
                "index": 1,
            }
        ]
        result = pipeline.reconstitute_segments(
            first_segment,
            second_segment,
            reconstitution_strategy=ReconstitutionStrategy.ENTITIES,
        )
        self.assertEqual(result, expected_result)

    @unittest.skip("Ignore labels not implemented yet")
    def test_ignore_labels(self):
        """Assert that ignored labels are not included in the output"""

    def test_offset_mapping(self):
        """Assert that an overloaded offset mapping results in the correct
        character indices"""
        pipeline = get_test_pipeline()
        # Overload offset_mapping
        outputs = pipeline(
            "This is a test !",
            offset_mapping=[(0, 0), (0, 1), (0, 2), (0, 0), (0, 0), (0, 0), (0, 0)],
        )
        self.assertEqual(nested_simplify(outputs), [
            {
                "entity": "I-MISC",
                "score": 0.115,
                "index": 1,
                "word": "this",
                "start": 0,
                "end": 1,
            },
            {
                "entity": "I-MISC",
                "score": 0.115,
                "index": 2,
                "word": "is",
                "start": 0,
                "end": 2,
            },
            {
                "entity": "O",
                "score": 0.116,
                "index": 3,
                "word": "a",
                "start": 0,
                "end": 0,
            },
            {
                "entity": "O",
                "score": 0.116,
                "index": 4,
                "word": "test",
                "start": 0,
                "end": 0,
            },
            {
                "entity": "O",
                "score": 0.116,
                "index": 5,
                "word": "!",
                "start": 0,
                "end": 0,
            },
        ])


class TestTruncatedTokenClassificationArguments(unittest.TestCase):
    @parameterized.expand(
        [
            (
                {"max_seq_len": 128, "stride": 8},
                False,
            ),
            (
                {"max_seq_len": 128},
                False,
            ),
            (
                {"max_seq_len": 128},
                True,
            ),
            (
                {"stride": 8},
                False,
            ),
        ],
    )
    def test_simple(
        self,
        kwargs,
        mock_model_max_len,
    ):
        """Assert that standard arguments are accepted"""
        if mock_model_max_len:
            patch.object(TOKENIZER, "model_max_length", None)
        pipeline = OverflowTokenClassificationPipeline(
            model=MODEL,
            tokenizer=TOKENIZER,
            **kwargs,
        )
        if "max_seq_len" in kwargs:
            self.assertEqual(pipeline.max_length, kwargs["max_seq_len"])
        if "stride" in kwargs:
            self.assertEqual(pipeline.stride, kwargs["stride"])

    @parameterized.expand(
        [
            ({"max_seq_len": 10, "stride": 10},),
            ({"max_seq_len": 8, "stride": 12},),
            ({"stride": 520},),
        ],
    )
    def test_stride_to_len_ratio(self, kwargs):
        """Assert that the stride must be less than the max length"""
        with self.assertRaises(ValueError):
            OverflowTokenClassificationPipeline(
                model=MODEL, tokenizer=TOKENIZER, **kwargs
            )

    @patch.object(TOKENIZER, "model_max_length", None)
    def test_no_stride_to_len_ratio_check(self):
        """Assert that the stride to length check is not performed if truncation
        is not being used"""
        
        pipeline = OverflowTokenClassificationPipeline(
            model=MODEL,
            tokenizer=TOKENIZER,
        )
        self.assertTrue(pipeline.truncation)

    @patch.object(TOKENIZER, "model_max_length", VERY_LARGE_INTEGER)
    def test_model_without_max_len(self):
        """Assert that model max length overrides the supplied max length"""
        
        pipeline = OverflowTokenClassificationPipeline(
            model=MODEL,
            tokenizer=TOKENIZER,
        )
        self.assertTrue(pipeline.truncation)
        self.assertEqual(pipeline.max_length, 0)

    def test_model_max_len(self):
        """Assert that model max length overrides the supplied max length"""
        expected_max_len = TOKENIZER.model_max_length
        pipeline = OverflowTokenClassificationPipeline(
            model=MODEL,
            tokenizer=TOKENIZER,
        )
        self.assertEqual(pipeline.max_length, expected_max_len)

    def test_no_tokenizer(self):
        """Assert that passing no tokenizer raises an error"""
        with self.assertRaises(ValueError):
            OverflowTokenClassificationPipeline(model=MODEL)

    def test_invalid_reconstitution_strategy(self, overflow_short):
        pipeline = get_test_pipeline()
        """Assert that passing an unknown reconstitution strategy raises an error"""
        strategy = "my-strategy"
        with self.assertRaises(ValueError):
            pipeline(overflow_short, reconstitution_strategy=strategy)
        with self.assertRaises(ValueError):
            pipeline.combine_overlapping_tokens([], [], strategy)  # noqa
