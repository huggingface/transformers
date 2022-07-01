"""Tests covering the overflow token classification NER pipeline"""
from typing import Dict, List

import pytest
from pytest_mock import MockerFixture
from transformers.testing_utils import nested_simplify, require_torch_gpu
from transformers.tokenization_utils_base import VERY_LARGE_INTEGER

from transformers.pipelines.overflow_token_classification import (
    OverflowTokenClassificationPipeline,
    ReconstitutionStrategy,
)


# File-specific fixtures
@pytest.fixture(scope="class")
def pipeline(ner_model, ner_tokenizer):
    pipeline = OverflowTokenClassificationPipeline(
        model=ner_model, tokenizer=ner_tokenizer, framework="pt"
    )
    yield pipeline


# Tests
class TestTruncatedTokenClassificationPipeline:
    @require_torch_gpu
    def test_gpu(self, ner_model, ner_tokenizer):
        sentence = "This is dummy sentence"
        ner = OverflowTokenClassificationPipeline(
            model=ner_model,
            tokenizer=ner_tokenizer,
            device=0,
        )

        output = ner(sentence)
        assert nested_simplify(output) == []

    def test_empty(self, pipeline):
        """Test the results when passing an empty sentence"""
        output = pipeline("")
        assert output == []
        output = pipeline([""])
        assert output == [[]]
        with pytest.raises(ValueError):
            pipeline([])

    def test_simple_str(self, pipeline, overflow_short: str, overflow_short_output):
        """Test the expected tokens with simple string inputs"""
        outputs = pipeline(overflow_short)
        assert nested_simplify(outputs) == overflow_short_output

    def test_simple_list(
        self,
        pipeline,
        overflow_short: str,
        overflow_short_output,
        overflow_batch,
        overflow_batch_output,
    ):
        """Test the expected tokens with simple list inputs"""
        outputs = pipeline([overflow_short])
        assert nested_simplify(outputs) == [overflow_short_output]
        outputs = pipeline(overflow_batch)
        assert nested_simplify(outputs) == overflow_batch_output

    def test_batching(self, pipeline, overflow_batch):
        """Assert that batching sentences does not change the output shape or
        results"""
        single_output = pipeline(overflow_batch, batch_size=1)
        batched_output = pipeline(overflow_batch, batch_size=2)
        assert single_output == batched_output

    def test_simple_reconstruction(self, ner_model, ner_tokenizer, overflow_long: str):
        """Assert that truncated sentences are reconstructed into original
        input sentences"""
        expected_tokens = [token.lower() for token in overflow_long.split()]
        num_tokens = len(expected_tokens)
        max_seq_len = num_tokens - 1
        ner = OverflowTokenClassificationPipeline(
            model=ner_model,
            tokenizer=ner_tokenizer,
            max_seq_len=max_seq_len,
            stride=2,
        )
        outputs = ner(overflow_long)
        assert len(outputs) == num_tokens
        output_words = [token["word"].lower() for token in outputs]
        assert output_words == expected_tokens
        # Check index, start and end tokens are all ascending
        for i, token in enumerate(outputs):
            assert token["index"] == i + 1
            try:
                next_token = outputs[i + 1]
                assert next_token["start"] > token["start"]
                assert next_token["end"] > token["end"]
            except IndexError:
                break

    @pytest.mark.parametrize(
        ["first", "second", "strategy", "prev_ent", "expected_result"],
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
        pipeline,
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
        result = pipeline.combine_overlapping_tokens(first, second, strategy, prev_ent)
        assert result == expected_result

    def test_invalid_prev_entity(self, pipeline):
        """Assert that a previous entity tag that cannot be found when combining
        segments that have no previous entity token"""
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
        assert result == expected_result

    @pytest.mark.skip("Ignore labels not implemented yet")
    def test_ignore_labels(self, pipeline):
        """Assert that ignored labels are not included in the output"""

    def test_offset_mapping(self, pipeline):
        """Assert that an overloaded offset mapping results in the correct
        character indices"""
        # Overload offset_mapping
        outputs = pipeline(
            "This is a test !",
            offset_mapping=[(0, 0), (0, 1), (0, 2), (0, 0), (0, 0), (0, 0), (0, 0)],
        )
        assert nested_simplify(outputs) == [
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
        ]


class TestTruncatedTokenClassificationArguments:
    @pytest.mark.parametrize(
        ["kwargs", "mock_model_max_len"],
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
        ner_model,
        ner_tokenizer,
        kwargs,
        mock_model_max_len,
        mocker: MockerFixture,
    ):
        """Assert that standard arguments are accepted"""
        if mock_model_max_len:
            mocker.patch.object(ner_tokenizer, "model_max_length", None)
        ner = OverflowTokenClassificationPipeline(
            model=ner_model,
            tokenizer=ner_tokenizer,
            **kwargs,
        )
        if "max_seq_len" in kwargs:
            assert ner.max_length == kwargs["max_seq_len"]
        if "stride" in kwargs:
            assert ner.stride == kwargs["stride"]

    @pytest.mark.parametrize(
        ["kwargs"],
        [
            ({"max_seq_len": 10, "stride": 10},),
            ({"max_seq_len": 8, "stride": 12},),
            ({"stride": 520},),
        ],
    )
    def test_stride_to_len_ratio(self, ner_model, ner_tokenizer, kwargs):
        """Assert that the stride must be less than the max length"""
        with pytest.raises(ValueError):
            OverflowTokenClassificationPipeline(
                model=ner_model, tokenizer=ner_tokenizer, **kwargs
            )

    def test_no_stride_to_len_ratio_check(
        self, ner_model, ner_tokenizer, mocker: MockerFixture
    ):
        """Assert that the stride to length check is not performed if truncation
        is not being used"""
        mocker.patch.object(ner_tokenizer, "model_max_length", None)
        ner = OverflowTokenClassificationPipeline(
            model=ner_model,
            tokenizer=ner_tokenizer,
        )
        assert not ner.truncation

    def test_model_without_max_len(
        self, ner_model, ner_tokenizer, mocker: MockerFixture
    ):
        """Assert that model max length overrides the supplied max length"""
        mocker.patch.object(ner_tokenizer, "model_max_length", VERY_LARGE_INTEGER)
        ner = OverflowTokenClassificationPipeline(
            model=ner_model,
            tokenizer=ner_tokenizer,
        )
        assert not ner.truncation
        assert ner.max_length == 0

    def test_model_max_len(self, ner_model, ner_tokenizer):
        """Assert that model max length overrides the supplied max length"""
        expected_max_len = ner_tokenizer.model_max_length
        ner = OverflowTokenClassificationPipeline(
            model=ner_model,
            tokenizer=ner_tokenizer,
        )
        assert ner.max_length == expected_max_len

    def test_no_tokenizer(self, ner_model):
        """Assert that passing no tokenizer raises an error"""
        with pytest.raises(ValueError):
            OverflowTokenClassificationPipeline(model=ner_model)

    def test_invalid_reconstitution_strategy(self, pipeline, overflow_short):
        """Assert that passing an unknown reconstitution strategy raises an error"""
        strategy = "my-strategy"
        with pytest.raises(ValueError):
            pipeline(overflow_short, reconstitution_strategy=strategy)
        with pytest.raises(ValueError):
            pipeline.combine_overlapping_tokens([], [], strategy)  # noqa
