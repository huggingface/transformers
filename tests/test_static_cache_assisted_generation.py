"""
Integration tests for StaticCache.crop() with assisted (speculative) generation.

These tests verify that the combination of StaticCache + assisted generation works
end-to-end, which was previously blocked by a ValueError in generation/utils.py.
Also verifies that the assistant model automatically receives static cache when
the main model uses cache_implementation="static" (auto-propagation logic).
"""

import copy

import pytest
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.candidate_generator import AssistedCandidateGenerator
from transformers.generation.configuration_utils import ALL_STATIC_CACHE_IMPLEMENTATIONS


MODEL_ID = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def assistant_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    return model


class TestStaticCacheAssistedGeneration:
    @pytest.mark.slow
    def test_assisted_generation_with_static_cache(self, model_and_tokenizer, assistant_model):
        """Assisted generation with cache_implementation='static' should not raise ValueError."""
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("Hello, world", return_tensors="pt")

        output = model.generate(
            **inputs,
            assistant_model=assistant_model,
            cache_implementation="static",
            max_new_tokens=20,
            do_sample=False,
        )
        assert output.shape[1] > inputs["input_ids"].shape[1]

    @pytest.mark.slow
    def test_assisted_generation_static_cache_matches_dynamic(self, model_and_tokenizer, assistant_model):
        """Static cache assisted generation should produce the same output as dynamic cache."""
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("The quick brown fox", return_tensors="pt")

        output_dynamic = model.generate(
            **inputs,
            assistant_model=assistant_model,
            max_new_tokens=20,
            do_sample=False,
        )

        output_static = model.generate(
            **inputs,
            assistant_model=assistant_model,
            cache_implementation="static",
            max_new_tokens=20,
            do_sample=False,
        )

        assert torch.equal(output_dynamic, output_static), (
            f"Static and dynamic cache produced different outputs.\n"
            f"Dynamic: {tokenizer.decode(output_dynamic[0])}\n"
            f"Static:  {tokenizer.decode(output_static[0])}"
        )

    @pytest.mark.slow
    def test_assisted_generation_static_cache_multiple_rounds(self, model_and_tokenizer, assistant_model):
        """Multiple sequential generate calls should work (cache gets re-initialized each time)."""
        model, tokenizer = model_and_tokenizer

        for prompt in ["Hello", "World", "Testing"]:
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(
                **inputs,
                assistant_model=assistant_model,
                cache_implementation="static",
                max_new_tokens=10,
                do_sample=False,
            )
            assert output.shape[1] > inputs["input_ids"].shape[1]

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile test requires CUDA")
    def test_assisted_generation_static_cache_with_compile(self, assistant_model):
        """torch.compile + static cache + assisted generation should work together."""
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).cuda()
        assistant = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).cuda()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.forward = torch.compile(model.forward, mode="reduce-overhead")

        inputs = tokenizer("Hello, world", return_tensors="pt").to("cuda")

        output = model.generate(
            **inputs,
            assistant_model=assistant,
            cache_implementation="static",
            max_new_tokens=20,
            do_sample=False,
        )
        assert output.shape[1] > inputs["input_ids"].shape[1]


class TestAssistantModelCachePropagation:
    """Tests that the assistant model's cache_implementation is correctly set based on the main model."""

    def test_assistant_receives_static_cache_implementation(self, model_and_tokenizer, assistant_model):
        """When main model uses static cache, the AssistedCandidateGenerator should propagate it to the assistant."""
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("Hello", return_tensors="pt")

        # Create a generation config with static cache
        gen_config = copy.deepcopy(model.generation_config)
        gen_config.cache_implementation = "static"
        gen_config.max_new_tokens = 10
        gen_config.max_length = inputs["input_ids"].shape[1] + 10

        # Build the candidate generator directly
        candidate_gen = AssistedCandidateGenerator(
            input_ids=inputs["input_ids"],
            assistant_model=assistant_model,
            generation_config=gen_config,
            model_kwargs={"attention_mask": inputs["attention_mask"]},
        )

        # The assistant's generation_config.cache_implementation should be "static"
        assert candidate_gen.generation_config.cache_implementation == "static", (
            f"Expected assistant cache_implementation='static', "
            f"got '{candidate_gen.generation_config.cache_implementation}'"
        )

    def test_assistant_receives_dynamic_cache_by_default(self, model_and_tokenizer, assistant_model):
        """When main model uses dynamic cache, the assistant should get 'dynamic_full'."""
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("Hello", return_tensors="pt")

        gen_config = copy.deepcopy(model.generation_config)
        gen_config.cache_implementation = None  # default / dynamic
        gen_config.max_new_tokens = 10
        gen_config.max_length = inputs["input_ids"].shape[1] + 10

        candidate_gen = AssistedCandidateGenerator(
            input_ids=inputs["input_ids"],
            assistant_model=assistant_model,
            generation_config=gen_config,
            model_kwargs={"attention_mask": inputs["attention_mask"]},
        )

        # Should fall back to "dynamic_full" when main model doesn't use static cache
        assert candidate_gen.generation_config.cache_implementation == "dynamic_full", (
            f"Expected assistant cache_implementation='dynamic_full', "
            f"got '{candidate_gen.generation_config.cache_implementation}'"
        )

    def test_all_static_implementations_propagated(self, model_and_tokenizer, assistant_model):
        """All static cache implementations should be propagated to the assistant model."""
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("Hello", return_tensors="pt")

        for static_impl in ALL_STATIC_CACHE_IMPLEMENTATIONS:
            gen_config = copy.deepcopy(model.generation_config)
            gen_config.cache_implementation = static_impl
            gen_config.max_new_tokens = 10
            gen_config.max_length = inputs["input_ids"].shape[1] + 10

            candidate_gen = AssistedCandidateGenerator(
                input_ids=inputs["input_ids"],
                assistant_model=assistant_model,
                generation_config=gen_config,
                model_kwargs={"attention_mask": inputs["attention_mask"]},
            )

            assert candidate_gen.generation_config.cache_implementation == static_impl, (
                f"Expected assistant cache_implementation='{static_impl}', "
                f"got '{candidate_gen.generation_config.cache_implementation}'"
            )

    @pytest.mark.slow
    def test_end_to_end_assistant_uses_static_cache(self, model_and_tokenizer, assistant_model):
        """End-to-end test: when main model uses static cache, the assistant model should
        also generate with static cache and produce valid output."""
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("The quick brown", return_tensors="pt")

        # Generate with static cache for the main model
        output = model.generate(
            **inputs,
            assistant_model=assistant_model,
            cache_implementation="static",
            max_new_tokens=15,
            do_sample=False,
        )

        # The output should have new tokens
        assert output.shape[1] > inputs["input_ids"].shape[1], (
            "Static cache assisted generation should produce new tokens"
        )
