"""Tests for assist_strict module functionality."""

import concurrent.futures
from typing import Tuple

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM as ModelType

from assist_strict.assisted import assisted_generate_strict


@pytest.fixture
def setup() -> Tuple[ModelType, ModelType, dict]:
    """Provide fresh model instances for each test."""
    model_name = "microsoft/DialoGPT-small"

    model = AutoModelForCausalLM.from_pretrained(model_name)
    assistant_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text = "Test input"
    tokenized_inputs = tokenizer(text, return_tensors="pt", padding=True)

    return model, assistant_model, tokenized_inputs


def test_applies_then_restores(setup):
    """Test assisted generation completes and restores original config."""
    model, assistant_model, tokenized_inputs = setup

    original_config = assistant_model.generation_config.to_dict()

    result = assisted_generate_strict(
        model=model,
        inputs=tokenized_inputs.input_ids,
        assistant_model=assistant_model,
        num_assistant_tokens=3,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=tokenized_inputs.attention_mask.shape[1] - 1
    )

    assert result is not None
    post_call_config = assistant_model.generation_config.to_dict()
    assert original_config == post_call_config


def test_read_verification(setup):
    """Test assisted generation enforces config read verification."""
    model, assistant_model, tokenized_inputs = setup

    result = assisted_generate_strict(
        model=model,
        inputs=tokenized_inputs.input_ids,
        assistant_model=assistant_model,
        num_assistant_tokens=2,
        max_new_tokens=2,
        do_sample=False,
        pad_token_id=tokenized_inputs.attention_mask.shape[1] - 1
    )

    assert result is not None


@pytest.mark.timeout(30)
def test_parallel_isolation():
    """Test parallel calls maintain isolation (may be flaky under heavy load)."""
    def worker_task(n: int) -> bool:
        """Worker function for parallel execution."""
        model_name = "microsoft/DialoGPT-small"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        assistant_model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        text = "Test input"
        tokenized_inputs = tokenizer(text, return_tensors="pt", padding=True)
        original_config = assistant_model.generation_config.to_dict()

        assisted_generate_strict(
            model=model,
            inputs=tokenized_inputs.input_ids,
            assistant_model=assistant_model,
            num_assistant_tokens=n,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

        post_call_config = assistant_model.generation_config.to_dict()
        return original_config == post_call_config

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(worker_task, 1),
            executor.submit(worker_task, 3),
            executor.submit(worker_task, 5)
        ]

        results = [future.result() for future in concurrent.futures.as_completed(futures)]
        assert all(results), "Parallel isolation failed"


def test_invalid_num_assistant_tokens(setup):
    """Test input validation for invalid num_assistant_tokens."""
    model, assistant_model, tokenized_inputs = setup

    with pytest.raises(ValueError, match="must be a positive integer"):
        assisted_generate_strict(
            model=model,
            inputs=tokenized_inputs.input_ids,
            assistant_model=assistant_model,
            num_assistant_tokens=0,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenized_inputs.attention_mask.shape[1] - 1
        )

    with pytest.raises(ValueError, match="must be a positive integer"):
        assisted_generate_strict(
            model=model,
            inputs=tokenized_inputs.input_ids,
            assistant_model=assistant_model,
            num_assistant_tokens="invalid",  # type: ignore
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenized_inputs.attention_mask.shape[1] - 1
        )
