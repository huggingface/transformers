# Copyright 2025 The HuggingFace Team Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import gc
import itertools
import os
import unittest
from typing import Any
from unittest.mock import patch

import torch
from parameterized import parameterized

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    CompileConfig,
    ContinuousBatchingConfig,
    GenerationConfig,
    GenerationMixin,
    StaticCache,
)
from transformers.generation.continuous_batching.cache import (
    PagedAttentionCache,
    PagedAttentionMemoryHandler,
    SlidingAttentionCacheAllocator,
    group_layers_by_attn_type,
)
from transformers.generation.continuous_batching.cache_manager import FullAttentionCacheAllocator
from transformers.generation.continuous_batching.continuous_api import OutputRouter
from transformers.generation.continuous_batching.distributed import DistributedHelper
from transformers.generation.continuous_batching.input_outputs import build_attention_mask
from transformers.generation.continuous_batching.offloading_manager import OffloadingManager
from transformers.generation.continuous_batching.requests import GenerationOutput, RequestStatus
from transformers.testing_utils import (
    require_deterministic_for_xpu,
    require_flash_attn,
    require_flash_attn_3,
    require_kernels,
    require_torch_accelerator,
    require_torch_gpu,
    require_torch_multi_accelerator,
    slow,
    torch_device,
)
from transformers.utils import (
    is_flash_attn_2_available,
    is_kernels_available,
    is_torch_xpu_available,
)
from transformers.utils.generic import is_flash_attention_requested

from ..test_tensor_parallel_mixin import _init_distributed


# Constants for tests
_DEFAULT_USER_MESSAGES = [
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    "A basket contains 25 oranges among which 1 is bad, 20% are unripe, 2 are sour and the rest are good. How many oranges are good?",
]  # fmt: skip


# Helper functions
def flush_memory(flush_compile: bool = True) -> None:
    """Flushes the memory of the current device and, if the flush_compile flag is True, all data related to
    torch.compile."""
    gc.collect()
    # If needed, flush everything related to torch.compile
    if flush_compile:
        # Dynamo resets
        torch._dynamo.reset()
        torch._dynamo.reset_code_caches()
        if hasattr(torch._inductor, "codecache"):
            # Clear FX graph cache
            if hasattr(torch._inductor.codecache, "FxGraphCache"):
                torch._inductor.codecache.FxGraphCache.clear()
            # Clear PyCodeCache
            if hasattr(torch._inductor.codecache, "PyCodeCache"):
                torch._inductor.codecache.PyCodeCache.cache_clear()
            # Clear TritonFuture cache (for async compilation)
            if hasattr(torch._inductor.codecache, "TritonFuture"):
                if hasattr(torch._inductor.codecache.TritonFuture, "_compile_cache"):
                    torch._inductor.codecache.TritonFuture._compile_cache.clear()
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
    gc.collect()


def get_tokenizer_and_model(
    model_id: str, attn_implementation: str, device: str, dtype: str | torch.dtype = "auto"
) -> tuple[AutoTokenizer, GenerationMixin]:
    """Returns a tokenizer and a model for the given model ID. Attributes to setup the models (attn_implementation,
    dtype and device) are needed as arguments."""
    # Get tokenizer, with a padding token if not present
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if not hasattr(tokenizer, "pad_token") and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token
    # Load model on CPU
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation=attn_implementation, torch_dtype=dtype)
    model = model.to(device).eval()
    return tokenizer, model


def with_flush_memory(func):
    """Decorator that ensures flush_memory is called after the test, even if it fails."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Determine flush_compile value from continuous_batching_config or generation_config
        cb_config = kwargs.get("continuous_batching_config")
        generation_config = kwargs.get("generation_config")
        if isinstance(cb_config, ContinuousBatchingConfig):
            flush_compile = (
                cb_config.use_default_compile_configs
                or cb_config.varlen_compile_config is not None
                or cb_config.decode_compile_config is not None
            )
        elif isinstance(generation_config, GenerationConfig):
            flush_compile = generation_config.compile_config is not None
        else:
            flush_compile = False
        # Run the test and always flush memory
        try:
            return func(*args, **kwargs)
        finally:
            flush_memory(flush_compile=flush_compile)

    return wrapper


def get_generation_inputs(
    user_messages: list[str], tokenizer: AutoTokenizer, for_continuous_batching: bool = False
) -> Any:
    """Returns the tokenized inputs for batched or non-batched generation."""
    chats = [[{"role": "user", "content": user_message}] for user_message in user_messages]
    if for_continuous_batching:
        tokenized = [tokenizer.apply_chat_template(chat, add_generation_prompt=True) for chat in chats]
        input_ids = [(x if isinstance(x, list) else x["input_ids"]) for x in tokenized]
        return input_ids
    else:
        inputs = tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
            return_attention_mask=True,
        )
        return inputs


def regular_generate(
    model: GenerationMixin,
    tokenizer: AutoTokenizer,
    user_messages: list[str],
    **generate_kwargs,
) -> tuple[list[list[int]], list[list[float]]]:
    # Run generation
    inputs = get_generation_inputs(user_messages, tokenizer, for_continuous_batching=False)
    generate_outputs = model.generate(**inputs.to(model.device), return_dict_in_generate=True, **generate_kwargs)

    # Keep only generated tokens
    all_generated_tokens = []
    num_input_tokens = inputs.input_ids.shape[1]
    for i in range(len(user_messages)):
        # Remove left-side input and padding tokens
        generated_toks = generate_outputs.sequences[i, num_input_tokens:].tolist()
        # Remove right-side padding tokens
        while generated_toks[-1] == model.generation_config.pad_token_id:
            generated_toks.pop()
        all_generated_tokens.append(generated_toks)

    # Retrieve logprobs if the scores were requested
    per_prompt_logprobs = []
    if generate_kwargs.get("output_scores", False):
        # Loop over prompts
        for i in range(len(user_messages)):
            logprobs = []
            tokens_for_prompt = generate_outputs.sequences[i, num_input_tokens:].tolist()
            for score, token in zip(generate_outputs.scores, tokens_for_prompt):
                # Scores already have logits processors applied (including temperature)
                probs = torch.nn.functional.softmax(score[i], dim=-1)
                logprobs.append(probs[token].log().item())
            per_prompt_logprobs.append(logprobs)
    # Otherwise, return an empty list
    else:
        per_prompt_logprobs = []
    return all_generated_tokens, per_prompt_logprobs


# Class for all continuous batching tests that do not require any accelerator. Usualy those test are faster to run.
class ContinuousBatchingNoAcceleratorTest(unittest.TestCase):
    @parameterized.expand(
        [
            (None, None, "0"),
            (None, 4096, "0"),
            ("f", None, "0"),
            ("ffff", None, "0000"),
            ("sssss", 4096, "00000"),
            ("fs", 4096, "01"),
            ("ssfssf", 4096, "001221"),
            ("ssssf", 4096, "01234"),
            ("fffsffs", 4096, "0123456"),
        ]
    )
    def test_group_layers(
        self,
        layer_types_str: str | None,
        sliding_window: int | None,
        expected_groups: str,
    ) -> None:
        """Test the layer grouping algorithm of the hybrid allocator."""
        # Take a config and change the layer_types attribute to the mix we want
        config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM-1.7B")

        if layer_types_str is not None:
            layer_types = [{"f": "full_attention", "s": "sliding_window"}[char] for char in layer_types_str]
        else:
            layer_types = None
            config.num_hidden_layers = len(expected_groups)

        config.layer_types = layer_types
        config.sliding_window = sliding_window

        expected_lg = {}
        for i, group in enumerate(expected_groups):
            group = int(group)
            expected_lg[group] = expected_lg.get(group, []) + [i]
        expected_layer_groups = [expected_lg[i] for i in sorted(expected_lg.keys())]

        # Test layer groups formation
        layer_groups, group_types = group_layers_by_attn_type(config)
        self.assertEqual(
            sorted(expected_layer_groups),
            sorted(layer_groups),
            f"Test failed for: {layer_types_str = }, {sliding_window = }, {expected_layer_groups = }, {layer_groups = }",
        )

        # If layer_types is provided, check that group_types matches the type of the all layers in each group
        if layer_types is not None:
            for layer_group, group_type in zip(layer_groups, group_types):
                layer_types = [config.layer_types[i] for i in layer_group]
                self.assertEqual(layer_types, [group_type] * len(layer_types))
        # If layer_types is None, all groups should be of the same type
        else:
            for group_type in group_types:
                sliding_window = getattr(config, "sliding_window", None)
                expected_group_type = "sliding_attention" if sliding_window is not None else "full_attention"
                self.assertEqual(
                    group_type,
                    expected_group_type,
                    f"Test failed for: {layer_types_str = }, {sliding_window = }, {group_types = }",
                )

    @parameterized.expand(
        [
            ([0, 4], [0, 4], 1, ["1000", "1100", "1110", "1111"]),
            ([0, 4], [0, 4], 2, ["1000", "1100", "0110", "0011"]),
            ([0, 3], [0, 5], 1, ["11100", "11110", "11111"]),
            ([0, 3], [0, 5], 3, ["11100", "01110", "00111"]),
            ([0, 3, 6], [0, 3, 6], 1, ["100000", "110000", "111000", "000100", "000110", "000111"]),
            ([0, 3, 6], [0, 3, 6], 2, ["100000", "110000", "011000", "000100", "000110", "000011"]),
        ]
    )
    def test_attention_mask(
        self,
        cumulative_seqlens_q: list[int],
        cumulative_seqlens_k: list[int],
        sliding_window: int,  # the sliding window size, 1 means no sliding window
        str_expected_mask_lines: list[str],  # the attention mask, broken down by line as a string of 0s and 1s
    ) -> None:
        """Tests the correctness of the attention mask used in the continuous batching API."""
        # Build expected mask
        minus_inf = torch.finfo(torch.float32).min
        expected_mask = torch.empty((cumulative_seqlens_q[-1], cumulative_seqlens_k[-1]), dtype=torch.float32)
        for i, line in enumerate(str_expected_mask_lines):
            expected_mask[i, :] = torch.tensor([minus_inf if c == "0" else 0 for c in line])
        # Build actual mask
        actual_mask = torch.full_like(expected_mask, minus_inf)  # function modifies in place
        build_attention_mask(actual_mask, cumulative_seqlens_q, cumulative_seqlens_k, sliding_window)
        # Check that the actual mask matches the expected mask
        matches = (expected_mask == actual_mask).all()
        # If it doesn't match, print the masks in a readable form and fail the test
        if not matches:
            str_mask = [
                "".join("1" if x == 0 else "0" for x in token_attn_vector) for token_attn_vector in actual_mask
            ]
            str_mask = "\n".join(str_mask)
            str_expected_mask = "\n".join(str_expected_mask_lines)
            self.fail(
                f"Test failed for: {cumulative_seqlens_q = }, {cumulative_seqlens_k = }, {sliding_window = }\n"
                f"Expected mask:\n{str_expected_mask}\n"
                f"Actual mask:\n{str_mask}"
            )

    @parameterized.expand(
        [
            # Case 1: Only full attention groups, allocation succeeds
            # needed_blocks = 2 * 1 = 2, free_blocks = 10 -> 2 <= 10 = True
            (2, 0, 1, 0, 0, 10, True),
            # Case 2: Only full attention groups, allocation fails
            # needed_blocks = 5 * 2 = 10, free_blocks = 5 -> 10 <= 5 = False
            (5, 0, 2, 0, 0, 5, False),
            # Case 3: Mixed attention, sliding window not yet full
            # needed_blocks = 2 * 1 + min(4 - 0, 2) * 1 = 2 + 2 = 4, free_blocks = 10 -> 4 <= 10 = True
            (2, 0, 1, 1, 4, 10, True),
            # Case 4: Mixed attention, sliding window partially filled
            # needed_blocks = 3 * 1 + min(4 - 2, 3) * 1 = 3 + 2 = 5, free_blocks = 5 -> 5 <= 5 = True
            (3, 2, 1, 1, 4, 5, True),
            # Case 5: Mixed attention, sliding window already full (allocated_blocks >= max_sliding)
            # blocks_left = max(4 - 5, 0) = 0, needed_blocks = 3 * 1 + 0 = 3, free_blocks = 5 -> 3 <= 5 = True
            (3, 5, 1, 1, 4, 5, True),
            # Case 6: Mixed attention, sliding window full, allocation fails due to full attention
            # blocks_left = max(4 - 4, 0) = 0, needed_blocks = 6 * 1 + 0 = 6, free_blocks = 5 -> 6 <= 5 = False
            (6, 4, 1, 1, 4, 5, False),
            # Case 7: Multiple full attention groups
            # needed_blocks = 3 * 2 = 6, free_blocks = 6 -> 6 <= 6 = True
            (3, 0, 2, 0, 0, 6, True),
            # Case 8: Multiple sliding attention groups, not full
            # needed_blocks = 2 * 1 + min(4 - 1, 2) * 2 = 2 + 4 = 6, free_blocks = 6 -> 6 <= 6 = True
            (2, 1, 1, 2, 4, 6, True),
            # Case 9: Edge case - requesting 0 blocks always succeeds
            # needed_blocks = 0, free_blocks = 0 -> 0 <= 0 = True
            (0, 0, 1, 1, 4, 0, True),
            # Case 10: Edge case - exactly enough blocks
            # needed_blocks = 2 * 1 + min(3 - 0, 2) * 1 = 2 + 2 = 4, free_blocks = 4 -> 4 <= 4 = True
            (2, 0, 1, 1, 3, 4, True),
        ]
    )
    def test_continuous_batching_will_allocation_be_successful(
        self,
        num_requested_blocks: int,
        allocated_blocks: int,
        num_full_attention_groups: int,
        num_sliding_attention_groups: int,
        max_sliding_window_blocks_per_request: int,
        num_free_blocks: int,
        expected_result: bool,
    ) -> None:
        """Test the will_allocation_be_successful method of PagedAttentionCache, overloading the relevant attributes of
        a dummy cache."""

        if torch_device is None:  # this check which should always pass and helps with type checking
            raise ValueError(f"This requires a torch accelerator, yet {torch_device = } and the test was not skipped.")

        # Create the cache
        cache = PagedAttentionCache(
            config=AutoConfig.from_pretrained("HuggingFaceTB/SmolLM-1.7B", attn_implementation="sdpa"),
            continuous_batching_config=ContinuousBatchingConfig(block_size=16, num_blocks=8, max_batch_tokens=8),
            device=torch_device,
            tp_plan={},
            distributed_helper=DistributedHelper(device_mesh=None, cpu_group_timeout=300),
        )

        # Overload cache parameters to match test scenario
        cache.num_full_attention_groups = num_full_attention_groups
        cache.num_sliding_attention_groups = num_sliding_attention_groups
        cache.max_sliding_window_blocks_per_request = max_sliding_window_blocks_per_request

        # Overload the cache get_num_free_blocks method
        cache.get_num_free_blocks = lambda: num_free_blocks

        # Test the method
        result = cache.will_allocation_be_successful(num_requested_blocks, allocated_blocks)

        self.assertEqual(
            result,
            expected_result,
            f"Failed for: {num_requested_blocks=}, {allocated_blocks=}, {num_full_attention_groups=}, "
            f"{num_sliding_attention_groups=}, {max_sliding_window_blocks_per_request=}, {num_free_blocks=}. "
            f"Expected {expected_result}, got {result}",
        )

    @parameterized.expand(
        [
            # (block_size, block_table, past_length, query_length)
            # Contiguous blocks
            (32, [0, 1, 2], 0, 16),
            (32, [0, 1, 2], 0, 64),
            (32, [0, 1, 2], 16, 16),
            (32, [0, 1, 2], 31, 2),
            # Non-contiguous blocks
            (32, [0, 3, 6], 0, 64),
            (32, [2, 5, 8], 60, 10),
            # Different block sizes
            (16, [0, 1, 2, 3], 14, 4),
            (64, [0, 1], 60, 10),
        ]
    )
    def test_full_attention_get_indices(
        self,
        block_size: int,
        block_table: list[int],
        past_length: int,
        query_length: int,
    ) -> None:
        """Test FullAttentionCacheAllocator.get_read_indices and get_write_indices return correct physical indices."""

        def reference_indices(start: int, end: int) -> list[int]:
            """Reference implementation: converts logical indices to physical indices."""
            return [block_table[i // block_size] * block_size + i % block_size for i in range(start, end)]

        allocator = FullAttentionCacheAllocator(index=0, block_size=block_size, allow_block_sharing=False)
        allocator.block_table["req"] = block_table

        # Test read indices (from 0 to past_length + query_length)
        expected_read = reference_indices(0, past_length + query_length)
        self.assertEqual(allocator.get_read_indices("req", past_length, query_length), expected_read)

        # Test write indices (from past_length to past_length + query_length)
        expected_write = reference_indices(past_length, past_length + query_length)
        self.assertEqual(allocator.get_write_indices("req", past_length, query_length), expected_write)

    @slow
    def test_continuous_batching_no_accelerators(self) -> None:
        """Test continuous batching generation when no accelerator is available. It uses a simulated CPU-only PyTorch
        environment by mocking all acceleratoravailability checks to return False"""
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # Mock all accelerator availability checks to simulate CPU-only PyTorch
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("transformers.utils.is_torch_xpu_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            # Verify patches work
            self.assertFalse(torch.cuda.is_available())
            self.assertFalse(is_torch_xpu_available())
            self.assertFalse(torch.backends.mps.is_available())

            tokenizer, model = get_tokenizer_and_model(model_id, "sdpa", "cpu")
            user_messages = _DEFAULT_USER_MESSAGES[:1]
            input_ids = get_generation_inputs(user_messages, tokenizer, for_continuous_batching=True)

            model.generation_config.max_new_tokens = 10
            model.generation_config.do_sample = False

            continuous_batching_config = ContinuousBatchingConfig(use_cuda_graph=False, use_async_batching=False)

            # This should not crash even with all accelerators unavailable
            outputs = model.generate_batch(
                inputs=input_ids,
                generation_config=model.generation_config,
                continuous_batching_config=continuous_batching_config,
            )

            # Verify we got outputs
            self.assertEqual(len(outputs), len(input_ids))
            for output in outputs.values():
                self.assertIsNotNone(output.generated_tokens)
                self.assertGreater(len(output.generated_tokens), 0)

    def test_output_router_deliver_to_queue(self):
        """Test that OutputRouter.deliver places outputs on the queue when no handler is registered."""
        router = OutputRouter()
        output = GenerationOutput(request_id="req_0", status=RequestStatus.FINISHED)
        router.deliver(output)
        result = router.output_queue.get_nowait()
        self.assertEqual(result.request_id, "req_0")
        self.assertTrue(router.output_queue.empty())

    def test_output_router_deliver_to_handler(self):
        """Test that OutputRouter.deliver forwards to a registered handler instead of the queue."""
        router = OutputRouter()
        received = []
        loop = unittest.mock.Mock()

        with router._lock:
            router.result_handlers["req_0"] = (lambda out: received.append(out), loop)

        output = GenerationOutput(request_id="req_0", status=RequestStatus.DECODING)
        router.deliver(output)

        loop.call_soon_threadsafe.assert_called_once()
        self.assertTrue(router.output_queue.empty())

    def test_distributed_helper_no_dist(self) -> None:
        """Test that DistributedHelper falls back to a single-rank, TP-driver setup when distributed is not on."""
        helper = DistributedHelper(device_mesh=None, cpu_group_timeout=300)
        self.assertFalse(helper.dist_on)
        self.assertEqual(helper.global_rank, 0)
        self.assertEqual(helper.world_size, 1)
        self.assertEqual(helper.tp_size, 1)
        self.assertEqual(helper.tp_local_rank, 0)
        self.assertEqual(helper.dp_rank, 0)
        self.assertEqual(helper.dp_size, 1)
        self.assertTrue(helper.is_tp_driver)
        self.assertIsNone(helper.tp_group)
        self.assertIsNone(helper.cpu_comm_group)

        # Tensor and object broadcasts should be no-ops without a TP group
        tensor = torch.tensor([1.0, 2.0])
        self.assertTrue(torch.equal(helper.tp_broadcast_from_rank_0(tensor), tensor))
        obj = {"some_request": "payload"}
        self.assertIs(helper.tp_broadcast_object_from_rank_0(obj), obj)

        # All-reduce-min should be a no-op without a TP group
        reduce_tensor = torch.tensor([7, 3], dtype=torch.int64)
        self.assertIs(helper.tp_all_reduce_min(reduce_tensor), reduce_tensor)
        self.assertTrue(torch.equal(reduce_tensor, torch.tensor([7, 3], dtype=torch.int64)))

    def test_distributed_helper_set_tp_seed_no_dist(self) -> None:
        """Test that set_tp_seed sets a torch seed without distributed initialized, both with and without a user seed."""
        helper = DistributedHelper(device_mesh=None, cpu_group_timeout=300)

        # Explicit seed: torch RNG state must be reproducible across calls
        helper.set_tp_seed(seed=42, model_device=torch.device("cpu"))
        first = torch.randint(0, 2**31 - 1, (4,))
        helper.set_tp_seed(seed=42, model_device=torch.device("cpu"))
        second = torch.randint(0, 2**31 - 1, (4,))
        self.assertTrue(torch.equal(first, second))

        # No seed: should not raise and should still set a torch seed
        helper.set_tp_seed(seed=None, model_device=torch.device("cpu"))

    def test_continuous_batching_config_disables_nccl_graph_mixing(self) -> None:
        """Test that ContinuousBatchingConfig sets NCCL_GRAPH_MIXING_SUPPORT=0 only under a distributed launch
        (WORLD_SIZE > 1) and respects the disable_nccl_graph_mixing flag."""
        original_nccl = os.environ.pop("NCCL_GRAPH_MIXING_SUPPORT", None)
        original_ws = os.environ.pop("WORLD_SIZE", None)
        try:
            # Single-GPU launch (no WORLD_SIZE): env var is left untouched
            ContinuousBatchingConfig()
            self.assertNotIn("NCCL_GRAPH_MIXING_SUPPORT", os.environ)

            # Distributed launch (WORLD_SIZE > 1): env var is set to "0"
            os.environ["WORLD_SIZE"] = "2"
            ContinuousBatchingConfig()
            self.assertEqual(os.environ.get("NCCL_GRAPH_MIXING_SUPPORT"), "0")

            # Explicitly disabled flag: env var is left untouched even under a distributed launch
            os.environ.pop("NCCL_GRAPH_MIXING_SUPPORT", None)
            ContinuousBatchingConfig(disable_nccl_graph_mixing=False)
            self.assertNotIn("NCCL_GRAPH_MIXING_SUPPORT", os.environ)

            # setdefault semantics: a pre-existing value is preserved
            os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = "1"
            ContinuousBatchingConfig()
            self.assertEqual(os.environ.get("NCCL_GRAPH_MIXING_SUPPORT"), "1")
        finally:
            if original_nccl is None:
                os.environ.pop("NCCL_GRAPH_MIXING_SUPPORT", None)
            else:
                os.environ["NCCL_GRAPH_MIXING_SUPPORT"] = original_nccl
            if original_ws is None:
                os.environ.pop("WORLD_SIZE", None)
            else:
                os.environ["WORLD_SIZE"] = original_ws


@require_torch_accelerator
class ContinuousBatchingWithAcceleratorTest(unittest.TestCase):
    # -----------------------------------------------Parity tests----------------------------------------------- #
    #         Ensure continuous batching and non-continuous batching generation produce the same outputs         #
    # ---------------------------------------------------------------------------------------------------------- #
    @require_deterministic_for_xpu
    @with_flush_memory
    def _test_continuous_batching_parity(
        self,
        model_id: str,
        continuous_batching_config: ContinuousBatchingConfig,
        attn_implementation: str,
        max_new_tokens: int = 20,
        num_repeat_prompts: int = 1,
    ) -> None:
        """Tests the parity between continuous batching and non-continuous batching generation."""

        # Skip the test if Flash Attention is required but not available
        is_fa = is_flash_attention_requested(requested_attention_implementation=attn_implementation)
        if is_fa and not (is_flash_attn_2_available() or is_kernels_available()):
            self.skipTest("Flash Attention is not available and neither is the kernels library. Skipping test.")
        # Skip the test if cuda graph is on but the device is not CUDA
        if continuous_batching_config.use_cuda_graph and torch_device != "cuda":
            self.skipTest("CUDA graph is only supported on CUDA devices. Skipping test.")

        # If the config turns on compile, change the generation config to use the default mode instead of
        # max-autotune-no-cudagraphs which can change the kernels between generate_batch and generate
        if continuous_batching_config.use_default_compile_configs:
            fullgraph = not is_flash_attention_requested(requested_attention_implementation=attn_implementation)
            compile_config = CompileConfig(mode="default", fullgraph=fullgraph, dynamic=True)
            continuous_batching_config.varlen_compile_config = compile_config

        # Eager and SDPA implementations get a precision boost to account for the fact that an attention mask is used in
        # continuous batching but not in generate
        dtype = "auto" if is_fa else torch.float32

        # Prepare inputs
        tokenizer, model = get_tokenizer_and_model(model_id, attn_implementation, torch_device, dtype)
        if (
            attn_implementation == "flash_attention_2"
            and torch_device == "cpu"
            and getattr(model.config, "sliding_window", None) is not None
            and model.config.sliding_window > 0
        ):
            self.skipTest("Flash Attention 2 with sliding window attention is not supported on CPU. Skipping test.")

        user_messages = _DEFAULT_USER_MESSAGES * num_repeat_prompts
        input_ids = get_generation_inputs(user_messages, tokenizer, for_continuous_batching=True)

        model.generation_config.max_new_tokens = max_new_tokens
        model.generation_config.do_sample = False

        # Generation with continuous batching
        continuous_batching_outputs = model.generate_batch(
            inputs=input_ids,
            generation_config=model.generation_config,
            continuous_batching_config=continuous_batching_config,
        )

        # Prepare non-continuous batching inputs and model
        inputs = get_generation_inputs(user_messages, tokenizer, for_continuous_batching=False)
        num_input_tokens = inputs.input_ids.shape[1]

        # Generation without continuous batching (reload model to avoid any state contamination)
        _, model = get_tokenizer_and_model(model_id, attn_implementation, torch_device, dtype)
        model.generation_config.max_new_tokens = max_new_tokens
        model.generation_config.do_sample = False
        model.generation_config.use_cuda_graph = continuous_batching_config.use_cuda_graph
        model.generation_config.compile_config = continuous_batching_config.varlen_compile_config

        # Create a static cache if compile_config is set, because regular generate requires a compileable cache
        past_key_values = None
        if model.generation_config.compile_config is not None:
            max_cache_len = num_input_tokens + max_new_tokens
            past_key_values = StaticCache(config=model.config, max_cache_len=max_cache_len)

        generate_outputs = model.generate(
            **inputs.to(torch_device), generation_config=model.generation_config, past_key_values=past_key_values
        )

        for i, user_message in enumerate(user_messages):
            # Find the corresponding request in the continuous batching outputs
            input_tokens = inputs.input_ids[i][inputs.attention_mask[i] == 1].tolist()
            key_to_pop = None
            for key, state in continuous_batching_outputs.items():
                if state.prompt_ids == input_tokens:
                    key_to_pop = key
                    break
            if key_to_pop is None:
                self.fail(f"Request {i} not found in continuous batching outputs")
            continuous_batching_output = continuous_batching_outputs.pop(key_to_pop).generated_tokens

            generate_output = generate_outputs[i][num_input_tokens:].tolist()
            while generate_output[-1] == model.generation_config.pad_token_id:
                generate_output.pop()

            if continuous_batching_output != generate_output:
                decoded_continuous_batching_output = tokenizer.decode(continuous_batching_output)
                decoded_generate_output = tokenizer.decode(generate_output)
                msg = f"Test failed for {model_id = } {continuous_batching_config = }, {attn_implementation = }\n"
                msg += f"User message              : {repr(user_message)}\n"
                msg += f"Continuous batching output: {repr(decoded_continuous_batching_output)}\n"
                msg += f"Generate output           : {repr(decoded_generate_output)}"
                self.fail(msg)

    @parameterized.expand(
        list(
            itertools.product(
                [False, True],
                ["eager", "sdpa", "flash_attention_2"],
                [False, True],
            )
        )
    )
    @slow
    def test_continuous_batching_config_combinations_no_compile(
        self,
        allow_block_sharing: bool,
        attn_implementation: str,
        use_cuda_graph: bool,
    ) -> None:
        # Compiling adds a lot of overhead, so it's better not to include here (2*3*2=12 tests because of cross product)
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        continuous_batching_config = ContinuousBatchingConfig(
            allow_block_sharing=allow_block_sharing,
            use_cuda_graph=use_cuda_graph,
            use_default_compile_configs=False,
        )
        self._test_continuous_batching_parity(
            model_id=model_id,
            continuous_batching_config=continuous_batching_config,
            attn_implementation=attn_implementation,
        )

    @parameterized.expand([("eager", False), ("sdpa", False), ("sdpa", True), ("flash_attention_2", True)])
    @slow
    def test_continuous_batching_config_combinations_with_compile(
        self,
        attn_implementation: str,
        use_cuda_graph: bool,
    ) -> None:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        continuous_batching_config = ContinuousBatchingConfig(
            use_cuda_graph=use_cuda_graph,
            use_default_compile_configs=True,
        )
        self._test_continuous_batching_parity(
            model_id=model_id,
            continuous_batching_config=continuous_batching_config,
            attn_implementation=attn_implementation,
        )

    # FIXME: Qwen2.5-0.5B-Instruct is not here because it's  broken (it uses a repetition penalty logits processor)
    # TODO: replace gemma2 with a tiny version of GPT-OSS? That way we can test sliding window AND attention sink
    @parameterized.expand(
        list(
            itertools.product(
                ["google/gemma-2-2b-it"],
                [False, True],
                [False, True],
            )
        )
    )
    @slow
    def test_continuous_batching_diverse_models(self, model_id: str, use_cuda_graph: bool, use_compile: bool) -> None:
        continuous_batching_config = ContinuousBatchingConfig(
            use_cuda_graph=use_cuda_graph, use_default_compile_configs=use_compile
        )
        self._test_continuous_batching_parity(
            model_id=model_id,
            continuous_batching_config=continuous_batching_config,
            attn_implementation="flash_attention_2",
        )

    @parameterized.expand([(True, False), (False, True)])
    @require_flash_attn_3
    @slow
    def test_continuous_batching_tuple_cuda_graph(self, varlen_cg: bool, decode_cg: bool) -> None:
        """Tests that use_cuda_graph can be a tuple to independently control varlen and decode paths."""
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        continuous_batching_config = ContinuousBatchingConfig(
            use_cuda_graph=(varlen_cg, decode_cg),
            use_async_batching=False,
        )
        self._test_continuous_batching_parity(
            model_id=model_id,
            continuous_batching_config=continuous_batching_config,
            attn_implementation="flash_attention_3",
        )

    def test_continuous_batching_fast(self) -> None:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        continuous_batching_config = ContinuousBatchingConfig(
            use_cuda_graph=False,
            allow_block_sharing=False,
            use_async_batching=False,
            use_default_compile_configs=False,
        )
        self._test_continuous_batching_parity(
            model_id=model_id,
            continuous_batching_config=continuous_batching_config,
            attn_implementation="sdpa",
        )

    def test_continuous_batching_long_generate(self) -> None:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        continuous_batching_config = ContinuousBatchingConfig(
            use_cuda_graph=True, allow_block_sharing=True, use_async_batching=False, use_default_compile_configs=True
        )
        self._test_continuous_batching_parity(
            model_id=model_id,
            continuous_batching_config=continuous_batching_config,
            attn_implementation="sdpa",
            max_new_tokens=80,
        )

    @parameterized.expand([(False, False), (False, True), (True, False), (True, True)])
    @slow
    def test_continuous_batching_log_probs(self, use_cuda_graph: bool, use_async_batching: bool) -> None:
        """Test that log probabilities match between continuous batching and regular generate."""
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # Retrieve tokenizer, model and eos_token_id (required otherwise logits will be misaligned)
        tokenizer, model = get_tokenizer_and_model(model_id, "sdpa", torch_device, torch.float32)
        eos_token_id = model.config.eos_token_id  # type: ignore[attr-defined]

        # Run CB generation
        user_messages = ["What is 2+2?", "Hello world"]
        input_ids = get_generation_inputs(user_messages, tokenizer, for_continuous_batching=True)
        gen_config = GenerationConfig(max_new_tokens=10, do_sample=False, eos_token_id=eos_token_id)
        continuous_batching_config = ContinuousBatchingConfig(
            use_cuda_graph=use_cuda_graph,
            use_async_batching=use_async_batching,
            return_logprobs=True,
        )
        cb_outputs = model.generate_batch(
            inputs=input_ids, generation_config=gen_config, continuous_batching_config=continuous_batching_config
        )

        # Load fresh model for regular generate
        tokenizer, model = get_tokenizer_and_model(model_id, "sdpa", torch_device, torch.float32)
        # Run regular generate
        regular_outputs, regular_logprobs = regular_generate(
            model=model,
            tokenizer=tokenizer,
            user_messages=user_messages,
            max_new_tokens=10,
            do_sample=False,
            output_scores=True,
            eos_token_id=eos_token_id,
        )

        # Compare log_probs for each request, matching by prompt_ids
        for i, cb_output in enumerate(cb_outputs.values()):
            # Compare Cb and regular generate outputs
            cb_output_ids = cb_output.generated_tokens
            regular_output_ids = regular_outputs[i]
            self.assertEqual(len(cb_output_ids), len(regular_output_ids))
            self.assertEqual(cb_output_ids, regular_output_ids)

            # Retrieve logprobs from CB and regular generate
            cb_logprobs = cb_output.logprobs
            expected_logprobs = regular_logprobs[i]

            # Because of padding, we need to truncate to the same length
            min_len = min(len(cb_logprobs), len(expected_logprobs))
            cb_logprobs = cb_logprobs[:min_len]
            expected_logprobs = expected_logprobs[:min_len]
            self.assertEqual(len(cb_logprobs), len(expected_logprobs))

            # Compare with tolerance for floating point differences (because of padding, tol is higher for cuda graphs)
            delta = 2e-5 if use_cuda_graph else 1e-5
            for j, (cb_lp, exp_lp) in enumerate(zip(cb_logprobs, expected_logprobs)):
                error_msg = f"logprob mismatch at position {j} for request {i}: CB={cb_lp}, expected={exp_lp}"
                self.assertAlmostEqual(cb_lp, exp_lp, delta=delta, msg=error_msg)

    def test_continuous_batching_few_blocks(self) -> None:
        """This test verifies that generation works with a very small number of blocks, ie. small enough that we need to
        offload a request at some point. To add more complexity, we repeat the same prompt 4 times and enable prefix
        sharing."""
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        continuous_batching_config = ContinuousBatchingConfig(
            use_cuda_graph=True, allow_block_sharing=True, use_async_batching=False, num_blocks=4, block_size=32
        )

        # Patch offload_one_request to verify it's called at least once
        original_offload = OffloadingManager.offload_one_request
        with patch.object(
            OffloadingManager, "offload_one_request", autospec=True, side_effect=original_offload
        ) as mock_offload:
            self._test_continuous_batching_parity(
                model_id=model_id,
                continuous_batching_config=continuous_batching_config,
                attn_implementation="sdpa",
                max_new_tokens=30,
                num_repeat_prompts=4,
            )
            self.assertTrue(mock_offload.called, "Offload method was not called.")

    # ---------------------------------------Streaming tests--------------------------------------- #
    #           Ensures the requests have the right behavior with and without streaming             #
    # --------------------------------------------------------------------------------------------- #
    def _test_streaming_or_not_request(self, with_streaming: bool, with_non_streaming: bool) -> None:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        max_new_tokens = 3

        tokenizer, model = get_tokenizer_and_model(model_id, "sdpa", torch_device)
        manager = model.init_continuous_batching()
        manager.logit_processor.clear()
        manager.start()

        user_messages = ["What is the Transformers library known for?"]
        inputs = get_generation_inputs(user_messages, tokenizer, for_continuous_batching=True)[0]

        # Test with non-streaming
        if with_non_streaming:
            request_id = manager.add_request(inputs, max_new_tokens=max_new_tokens, streaming=False)

            # In non-streaming mode, the total number of generated tokens is equal to the max new tokens
            chunk = next(manager.request_id_iter(request_id))
            self.assertEqual(len(chunk.generated_tokens), max_new_tokens)

        # Test with streaming
        if with_streaming:
            request_id = manager.add_request(inputs, max_new_tokens=max_new_tokens, streaming=True)

            # In streaming mode, the total number of generated tokens is incremented by 1 on each iteration
            chunk_1 = next(manager.request_id_iter(request_id))
            self.assertEqual(len(chunk_1.generated_tokens), 1)

            chunk_2 = next(manager.request_id_iter(request_id))
            self.assertEqual(len(chunk_2.generated_tokens), 2)

            chunk_3 = next(manager.request_id_iter(request_id))
            self.assertEqual(len(chunk_3.generated_tokens), 3)

        manager.stop(block=True)

    def test_streaming_request(self) -> None:
        self._test_streaming_or_not_request(with_streaming=True, with_non_streaming=False)

    def test_non_streaming_request(self) -> None:
        self._test_streaming_or_not_request(with_streaming=False, with_non_streaming=True)

    def test_streaming_and_non_streaming_requests_can_alternate(self) -> None:
        self._test_streaming_or_not_request(with_streaming=True, with_non_streaming=True)

    def test_register_result_handler(self) -> None:
        """Test that register_result_handler receives streaming outputs through the OutputRouter."""
        import asyncio

        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        max_new_tokens = 3

        tokenizer, model = get_tokenizer_and_model(model_id, "sdpa", torch_device)
        manager = model.init_continuous_batching()
        manager.logit_processor.clear()
        manager.start()

        user_messages = ["What is the Transformers library known for?"]
        inputs = get_generation_inputs(user_messages, tokenizer, for_continuous_batching=True)[0]

        async def collect_results():
            token_counts = []
            future = asyncio.get_running_loop().create_future()

            def on_result(output):
                token_counts.append(len(output.generated_tokens))
                if output.is_finished():
                    future.set_result(True)

            request_id = manager.add_request(inputs, max_new_tokens=max_new_tokens, streaming=True)
            manager.register_result_handler(request_id, on_result)

            await asyncio.wait_for(future, timeout=30)
            return token_counts

        token_counts = asyncio.run(collect_results())

        # Streaming via handler: incremental token count, same as request_id_iter
        self.assertEqual(token_counts, [1, 2, 3])
        # Queue should be empty — everything went through the handler
        self.assertTrue(manager.output_router.output_queue.empty())

        manager.stop(block=True)

    # -----------------------------------------Misc. tests----------------------------------------- #
    #                     Various tests that don't fit into the other categories                    #
    # --------------------------------------------------------------------------------------------- #
    def _test_block_sharing(self, model_id: str, expected_layer_types: dict[str, int], input_msg: str) -> None:
        # Use float32 for SDPA to handle precision differences from attention masks (same as parity test)
        tokenizer, model = get_tokenizer_and_model(model_id, "sdpa", torch_device, dtype=torch.float32)

        # Configure generation for parity: disable processors not supported by CB (like repetition_penalty)
        model.generation_config.max_new_tokens = 32
        model.generation_config.do_sample = False
        model.generation_config.repetition_penalty = None

        # Get expected output from regular generate for parity check
        expected_output_tokens, _ = regular_generate(model, tokenizer, [input_msg])

        cb_context_manager = model.continuous_batching_context_manager(
            generation_config=model.generation_config,
            continuous_batching_config=ContinuousBatchingConfig(block_size=32),
        )
        with cb_context_manager as manager:
            # Create a request with at least 32 tokens but less than 64 so prefill only generates one complete block
            inputs = get_generation_inputs([input_msg], tokenizer, for_continuous_batching=True)[0]
            self.assertGreaterEqual(len(inputs), 32, f"Input length is {len(inputs)} instead of at least 32")
            self.assertLess(len(inputs), 64, f"Input length is {len(inputs)} instead of less than 64")

            # First request, which populates the cache w/ 2 complete blocks for each full attention layer group
            request_id = manager.add_request(inputs, max_new_tokens=32)
            chunk_no_reuse = next(manager.request_id_iter(request_id))

            num_fa = expected_layer_types["full_attention"]
            num_sw = expected_layer_types["sliding_window"]

            if manager.batch_processor is None:
                raise RuntimeError("Batch processor is None even after a request was added.")

            hash_table = manager.batch_processor.cache._block_manager._hash_to_id
            self.assertEqual(
                len(hash_table),
                2 * num_fa,  # 2 = 1 for prefill + 1 for decode
                f"There should be {2 * num_fa} blocks, 2 for each full attention layer group, but {len(hash_table) = }",
            )
            total_prefix_length = manager.batch_processor.cache._total_prefix_length
            self.assertEqual(
                total_prefix_length, 0, f"Expected total prefix length to be 0, got {total_prefix_length}"
            )

            # Assert the number of layer groups and their types are the expected ones
            layer_groups = manager.batch_processor.cache.group_cache_managers
            self.assertEqual(
                len(layer_groups),
                num_fa + num_sw,
                f"There should be {num_fa + num_sw} layer groups, but {len(layer_groups) = }",
            )

            layer_group_types = {"full_attention": 0, "sliding_window": 0}
            for cm in layer_groups:
                if isinstance(cm, FullAttentionCacheAllocator):
                    layer_group_types["full_attention"] += 1
                elif isinstance(cm, SlidingAttentionCacheAllocator):
                    layer_group_types["sliding_window"] += 1
                else:
                    raise ValueError(f"Invalid layer group type: {type(cm)}")

            self.assertEqual(
                layer_group_types,
                expected_layer_types,
                f"The expected layer group types are\n{expected_layer_types}\nbut got\n{layer_group_types}",
            )

            # Second request, which should reuse the same blocks for the full attention layer groups
            request_id = manager.add_request(inputs, max_new_tokens=32)
            chunk_with_reuse = next(manager.request_id_iter(request_id))

            # There should only still be two blocks in the hash table because of block reuse
            self.assertEqual(
                len(hash_table),
                2 * num_fa,
                f"Because of block reuse, there should still be two blocks in the hash table, but {len(hash_table) = }",
            )

            # Check that the whole prefill was matched if there are only full attention layers
            if expected_layer_types["sliding_window"] == 0:
                expected_total_prefix_length = 32
            else:
                expected_total_prefix_length = 0
            total_prefix_length = manager.batch_processor.cache._total_prefix_length
            self.assertEqual(
                total_prefix_length,
                expected_total_prefix_length,
                f"Expected total prefix length to be {expected_total_prefix_length}, but got {total_prefix_length = }",
            )

        # Check the outputs were the same (block sharing should produce identical results)
        self.assertEqual(chunk_no_reuse.generated_tokens, chunk_with_reuse.generated_tokens)

        # Verify parity with regular generate
        self.assertEqual(chunk_no_reuse.generated_tokens, expected_output_tokens[0])

    def test_prefix_sharing(self) -> None:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        num_layer_groups = {"full_attention": 1, "sliding_window": 0}
        input_msg = "What is the Transformers library known for?"
        return self._test_block_sharing(model_id, num_layer_groups, input_msg)

    def test_block_sharing_with_hybrid_model(self) -> None:
        model_id = "google/gemma-3-1b-it"
        num_layer_groups = {"full_attention": 2, "sliding_window": 11}
        input_msg = "I am a software engineer looking to use open source software to build a new AI agent. What is the Transformers library known for?"
        return self._test_block_sharing(model_id, num_layer_groups, input_msg)

    @parameterized.expand([True, False])
    @require_flash_attn  # otherwise the test can fail because attention bias has a very slight impact on SDPA and eager
    def test_num_return_sequences(self, allow_block_sharing: bool) -> None:
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer, model = get_tokenizer_and_model(model_id, "flash_attention_2", torch_device)
        user_messages = _DEFAULT_USER_MESSAGES[:1]
        input_ids = get_generation_inputs(user_messages, tokenizer, for_continuous_batching=True)

        model.generation_config.max_new_tokens = 30
        model.generation_config.do_sample = False

        # Generation with continuous batching
        cb_context_manager = model.continuous_batching_context_manager(
            continuous_batching_config=ContinuousBatchingConfig(allow_block_sharing=allow_block_sharing),
            block=True,
            timeout=5,
        )
        # Main loop
        results = []
        with cb_context_manager as manager:
            manager.num_return_sequences = 2
            manager.add_requests(inputs=input_ids, max_new_tokens=30)
            requests_left = 2
            while requests_left:
                result = manager.get_result(timeout=1)
                if result and result.is_finished():
                    results.append(result)
                    requests_left -= 1
                else:
                    if not manager.is_running():
                        break

        self.assertEqual(len(results), 2, f"Expected 2 results, but got {len(results) = }")
        self.assertEqual(results[0].generated_tokens, results[1].generated_tokens)

    # ----------------------------------Additional features tests---------------------------------- #
    #               Tests to check addtional features of CB do not change its results               #
    # --------------------------------------------------------------------------------------------- #
    @parameterized.expand(
        [
            # SDPA: basic features or full features
            ("sdpa", False, False),
            ("sdpa", True, True),
            # FA2: full coverage
            ("flash_attention_2", False, False),
            ("flash_attention_2", False, True),
            ("flash_attention_2", True, False),
            ("flash_attention_2", True, True),
            # FA3: always turn on CUDA graphs
            ("flash_attention_3", True, False),
            ("flash_attention_3", True, True),
        ]
    )
    @slow
    def test_continuous_batching_async(
        self, attn_implementation: str, use_cuda_graph: bool, use_compile: bool
    ) -> None:
        # Again, we try to not overly use_compile because it adds a lot of overhead
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self._test_continuous_batching_parity(
            model_id=model_id,
            continuous_batching_config=ContinuousBatchingConfig(
                allow_block_sharing=True,
                use_cuda_graph=use_cuda_graph,
                use_async_batching=True,
                use_default_compile_configs=use_compile,
            ),
            attn_implementation=attn_implementation,
        )

    @parameterized.expand([(False, False), (False, True), (True, False), (True, True)])
    @slow
    @require_kernels
    def test_flash_attn_with_kvcache_parity(self, use_cuda_graph: bool, use_async: bool) -> None:
        """Test that paged flash_attn3 (flash_attn_with_kvcache path) produces same outputs as varlen."""

        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer, model = get_tokenizer_and_model(
            model_id, "paged|kernels-community/flash-attn3", torch_device, torch.bfloat16
        )
        user_messages = _DEFAULT_USER_MESSAGES[:]
        input_ids = get_generation_inputs(user_messages, tokenizer, for_continuous_batching=True)

        gen_config = GenerationConfig(do_sample=False, max_new_tokens=20)
        continuous_batching_config = ContinuousBatchingConfig(
            block_size=256,
            num_blocks=64,
            max_batch_tokens=16,
            use_cuda_graph=use_cuda_graph,
            use_async_batching=use_async,
        )

        # Generate with varlen path only
        continuous_batching_config.max_blocks_per_request = 0
        outputs_varlen = model.generate_batch(
            inputs=input_ids, generation_config=gen_config, continuous_batching_config=continuous_batching_config
        )

        # Generate with flash_attn_with_kvcache path for decode
        continuous_batching_config.max_blocks_per_request = 16
        # This context manager ensures that the varlen path is used
        og_get_block_table_key = PagedAttentionCache.get_block_table_key
        with patch.object(
            PagedAttentionCache, "get_block_table_key", autospec=True, side_effect=og_get_block_table_key
        ) as mock_get_block_table_key:
            outputs_kvcache = model.generate_batch(
                inputs=input_ids, generation_config=gen_config, continuous_batching_config=continuous_batching_config
            )
            self.assertTrue(mock_get_block_table_key.called, "get_block_table_key method was not called.")

        self.assertEqual(len(outputs_varlen), len(outputs_kvcache))
        for (_, out_fa2), (_, out_fa3) in zip(outputs_varlen.items(), outputs_kvcache.items()):
            text_fa2 = tokenizer.decode(out_fa2.generated_tokens, skip_special_tokens=True)
            text_fa3 = tokenizer.decode(out_fa3.generated_tokens, skip_special_tokens=True)
            self.assertEqual(text_fa2, text_fa3, f"Mismatch:\nFA2: {text_fa2}\nFA3: {text_fa3}")

    @slow
    @require_kernels
    def test_decode_fast_path_wide_batch_parity(self) -> None:
        """Decode-fast-path output must match varlen when more requests decode concurrently than
        `max_blocks_per_request` (regression test for the `pad_to_pow2` cap truncating the decode batch)."""
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer, model = get_tokenizer_and_model(
            model_id, "paged|kernels-community/flash-attn3", torch_device, torch.bfloat16
        )
        # 12 requests but only 4 blocks per request: the decode batch is wider than max_blocks_per_request
        input_ids = get_generation_inputs(_DEFAULT_USER_MESSAGES * 4, tokenizer, for_continuous_batching=True)
        gen_config = GenerationConfig(do_sample=False, max_new_tokens=20)
        # CUDA graphs enable input padding, which is where the truncation happened
        cb_config = ContinuousBatchingConfig(block_size=256, num_blocks=64, use_cuda_graph=True)

        cb_config.max_blocks_per_request = 0  # varlen reference
        outputs_varlen = model.generate_batch(
            inputs=input_ids, generation_config=gen_config, continuous_batching_config=cb_config
        )
        cb_config.max_blocks_per_request = 4  # decode fast path, narrower than the batch
        outputs_fast = model.generate_batch(
            inputs=input_ids, generation_config=gen_config, continuous_batching_config=cb_config
        )

        self.assertEqual(len(outputs_varlen), len(outputs_fast))
        for (_, out_varlen), (_, out_fast) in zip(outputs_varlen.items(), outputs_fast.items()):
            text_varlen = tokenizer.decode(out_varlen.generated_tokens, skip_special_tokens=True)
            text_fast = tokenizer.decode(out_fast.generated_tokens, skip_special_tokens=True)
            self.assertEqual(text_varlen, text_fast, f"Mismatch:\nvarlen: {text_varlen}\nfast: {text_fast}")

    @parameterized.expand([(False, False), (False, True), (True, False), (True, True)])
    @slow
    def test_per_request_logits_processors(self, use_cuda_graph: bool, use_async_batching: bool) -> None:
        """Tests that per-request logits processor kwargs (temperature, top_k, top_p) work correctly in generation."""
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        max_new_tokens = 10
        temperatures = [1.0, 1.0]
        top_ks = [10, 50]
        top_ps = [0.9, 0.99]

        tokenizer, model = get_tokenizer_and_model(model_id, "flash_attention_2", torch_device)
        eos_token_id = model.config.eos_token_id  # type: ignore[attr-defined]

        # Same prompt for both requests
        user_messages = ["Write a random number:"]
        input_ids = get_generation_inputs(user_messages, tokenizer, for_continuous_batching=True)[0]

        # Use the context manager to add requests with different per-request kwargs
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=max(temperatures) + 1,  # enables temperature warping
            top_k=max(top_ks) + 1,
            top_p=min(top_ps) - 0.01,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        continuous_batching_config = ContinuousBatchingConfig(
            use_cuda_graph=use_cuda_graph,
            use_async_batching=use_async_batching,
            per_request_processors=True,
            return_logprobs=True,
            q_padding_interval_size=16,  # allows for exact comparison between CB and regular generation
        )
        manager = model.init_continuous_batching(
            generation_config=generation_config,
            continuous_batching_config=continuous_batching_config,
        )

        # Trick to have temperature, top-k, top-p ... without randomness: diable sampling after manager creation
        manager.generation_config.do_sample = False

        manager.start()
        try:
            # Request 0: low temperature (more deterministic)
            req0_id = manager.add_request(
                input_ids, max_new_tokens=max_new_tokens, temperature=temperatures[0], top_k=top_ks[0], top_p=top_ps[0]
            )
            # Request 1: high temperature (more random)
            req1_id = manager.add_request(
                input_ids, max_new_tokens=max_new_tokens, temperature=temperatures[1], top_k=top_ks[1], top_p=top_ps[1]
            )
            # Collect results
            results = {}
            while len(results) < 2:
                result = manager.get_result(timeout=1)
                if result is not None and result.is_finished():
                    results[result.request_id] = result
                elif not manager.is_running():
                    break
        finally:
            manager.stop(block=True)

        # Both requests should complete and have logprobs
        self.assertEqual(len(results), 2, f"Expected 2 results, got {len(results)}")
        self.assertGreater(len(results[req0_id].logprobs), 0)
        self.assertGreater(len(results[req1_id].logprobs), 0)
        # Also ensure the logprobs were not the same
        self.assertNotEqual(results[req0_id].logprobs, results[req1_id].logprobs)

        # Compare each request with regular generation
        # Build logits processor with do_sample=True (so temperature is included), then set do_sample=False for
        # deterministic generation, which is the same trick that CB uses
        delta = 2e-5 if use_cuda_graph else 1e-5
        for i, req_id in enumerate([req0_id, req1_id]):
            tokenizer, model = get_tokenizer_and_model(model_id, "flash_attention_2", torch_device)
            gen_config = GenerationConfig(
                do_sample=True,
                temperature=temperatures[i],
                top_k=top_ks[i],
                top_p=top_ps[i],
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
            )
            logits_processor = model._get_logits_processor(gen_config)
            gen_config.do_sample = False
            regular_generated_tokens, regular_logprobs = regular_generate(
                model=model,
                tokenizer=tokenizer,
                user_messages=user_messages,
                logits_processor=logits_processor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                eos_token_id=eos_token_id,
            )
            self.assertEqual(results[req_id].generated_tokens, regular_generated_tokens[0])
            for j, (cb_lp, exp_lp) in enumerate(zip(results[req_id].logprobs, regular_logprobs[0])):
                error_msg = f"Request {i}: logprob mismatch at position {j}: CB={cb_lp}, expected={exp_lp}"
                self.assertAlmostEqual(cb_lp, exp_lp, delta=delta, msg=error_msg)

    # ---------------------------------- CPU offloading tests ---------------------------------- #

    @require_torch_accelerator
    def test_cpu_offloading_parity(self) -> None:
        """Test that CPU offloading produces the same results as the legacy soft-reset path, and that it is actually
        called at least once. Uses a very small cache (few blocks) to force offloading."""
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        continuous_batching_config = ContinuousBatchingConfig(
            use_cuda_graph=True,
            allow_block_sharing=True,
            use_async_batching=False,
            num_blocks=4,
            block_size=32,
            cpu_offload_space=1.0,
        )

        original_offload = OffloadingManager._offload_to_cpu
        with patch.object(
            OffloadingManager, "_offload_to_cpu", autospec=True, side_effect=original_offload
        ) as mock_offload:
            self._test_continuous_batching_parity(
                model_id=model_id,
                continuous_batching_config=continuous_batching_config,
                attn_implementation="sdpa",
                max_new_tokens=30,
                num_repeat_prompts=4,
            )
            self.assertTrue(mock_offload.called, "_offload_to_cpu was not called despite few blocks being available.")

    @require_torch_accelerator
    def test_cpu_offloading_disabled_when_zero(self) -> None:
        """Test that cpu_offload_space=0 produces the same output as the legacy path."""
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        continuous_batching_config = ContinuousBatchingConfig(
            use_cuda_graph=True,
            allow_block_sharing=True,
            use_async_batching=False,
            num_blocks=4,
            block_size=32,
            cpu_offload_space=0.0,
        )
        # Should work identically to the existing test_continuous_batching_few_blocks
        self._test_continuous_batching_parity(
            model_id=model_id,
            continuous_batching_config=continuous_batching_config,
            attn_implementation="sdpa",
            max_new_tokens=30,
            num_repeat_prompts=4,
        )


@require_torch_gpu
class TestMemoryHandlerPrediction(unittest.TestCase):
    """Verifies that ``PagedAttentionMemoryHandler.compute_memory_footprint`` matches real GPU memory usage.

    For each configuration we allocate tensors at the *idealized* sizes modeled by the handler (same shapes, same
    dtypes, no alignment padding or extra blocks) and compare the CUDA memory delta to the handler's prediction.
    """

    # (block_size, page_size, num_groups, group_size, peak_act, num_attn_masks, max_bpr, logprobs, cache_dtype, use_async_batching)
    CONFIGS = [
        (32, 256, 1, 22, 34048, 1, 0, False, torch.float16, False),  # sdpa-like, 1 attn mask
        (256, 256, 1, 22, 34048, 0, 0, False, torch.float16, False),  # flash-like, no attn mask
        (32, 256, 2, 14, 34048, 2, 0, False, torch.bfloat16, False),  # hybrid model, 2 groups + 2 masks
        (32, 128, 1, 16, 8192, 1, 8, True, torch.float16, False),  # with block_table + logprobs
        (32, 128, 1, 16, 8192, 1, 8, True, torch.float16, True),  # with block_table + logprobs + async batching
    ]

    NUM_BLOCKS = 4
    MAX_BATCH_TOKENS = 64

    @parameterized.expand(CONFIGS)
    def test_memory_prediction(
        self,
        block_size: int,
        page_size: int,
        num_groups: int,
        group_size: int,
        peak_act: int,
        num_attn_masks: int,
        max_bpr: int,
        logprobs: bool,
        cache_dtype: torch.dtype,
        use_async_batching: bool,
    ) -> None:
        cb_config = ContinuousBatchingConfig(
            max_blocks_per_request=max_bpr,
            return_logprobs=logprobs,
            use_async_batching=use_async_batching,
            block_size=block_size,
        )

        handler = PagedAttentionMemoryHandler(
            continuous_batching_config=cb_config,
            page_size=page_size,
            num_groups=num_groups,
            group_size=group_size,
            activation_peaks=[(0, peak_act)],
            num_attention_masks=num_attn_masks,
        )

        N = self.NUM_BLOCKS * block_size  # num_pages
        M = self.MAX_BATCH_TOKENS
        predicted = handler.compute_memory_footprint(self.NUM_BLOCKS, M, cache_dtype)
        num_output_rows = 2 if logprobs else 1
        act_dtype = handler._activation_dtype
        i32 = handler._input_dtype

        # -- Allocate tensors at the exact idealized sizes the handler models --
        device = "cuda"
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)

        k = handler.io_multiplier  # 1 sync, 2 async -- scales IO tensors only
        tensors = []
        # kv_cache: 2 * group_size tensors of [N, page_size] (not scaled by k)
        for _ in range(group_size):
            tensors.append(torch.empty((N, page_size), dtype=cache_dtype, device=device))
            tensors.append(torch.empty((N, page_size), dtype=cache_dtype, device=device))
        # activation peak: flat tensor of peak_act * M elements (not scaled by k)
        tensors.append(torch.empty(peak_act * M, dtype=act_dtype, device=device))
        # IO tensors below are allocated k times (once per IO instance)
        for _ in range(k):
            # bulk_input: [7, M]
            tensors.append(torch.empty((7, M), dtype=i32, device=device))
            # output_ids: [num_output_rows, M]
            tensors.append(torch.empty((num_output_rows, M), dtype=i32, device=device))
            # attention_mask: [1, 1, M, N + M] per mask type
            for _ in range(num_attn_masks):
                tensors.append(torch.empty((1, 1, M, N + M), dtype=act_dtype, device=device))
            # block_table: [num_groups, M, max_bpr] (empty when max_bpr == 0)
            if max_bpr > 0:
                tensors.append(torch.empty((num_groups, M, max_bpr), dtype=i32, device=device))
            # write_index: [num_groups, M]
            tensors.append(torch.empty((num_groups, M), dtype=torch.int64, device=device))
            # read_index: [num_groups, N + M]
            tensors.append(torch.empty((num_groups, N + M), dtype=torch.int64, device=device))

        actual_cuda = torch.cuda.memory_allocated(device) - baseline
        expected_nbytes = sum(t.nbytes for t in tensors)
        num_allocations = len(tensors)

        del tensors
        torch.cuda.empty_cache()

        # 1) Exact check: prediction must equal the sum of tensor nbytes. This validates the polynomial
        #    coefficients against the tensor shapes, with zero tolerance.
        self.assertEqual(
            predicted,
            expected_nbytes,
            f"Prediction ({predicted}) != sum of tensor nbytes ({expected_nbytes})",
        )

        # 2) GPU memory check: CUDA's caching allocator rounds each allocation up (typically to 512 bytes).
        #    We allow up to 512 bytes of overhead per allocation.
        max_cuda_overhead = num_allocations * 512
        self.assertLessEqual(
            abs(actual_cuda - predicted),
            max_cuda_overhead,
            f"CUDA delta ({actual_cuda}) too far from prediction ({predicted}), "
            f"allowed overhead = {max_cuda_overhead} ({num_allocations} allocs × 512B)",
        )


# Worker functions for the TP continuous batching tests, spawned through `_init_distributed`.
def _tp_continuous_batching_worker(
    rank: int,
    model_id: str,
    attn_implementation: str,
    max_new_tokens: int,
    do_sample: bool,
    seed: int,
    use_cuda_graph: bool,
    use_async_batching: bool,
) -> None:
    """Loads `model_id` with `tp_plan="auto"`, checks three TP-specific paths in the same process: (a) direct
    broadcasts via `DistributedHelper`, (b) per-rank parity of CB-generated tokens via `dist.all_gather_object`, and
    (c) reproducibility across two CB runs sharing the same seed. Rank 0 owns all the assertions; the other ranks
    only need to participate in the collectives."""
    import torch
    import torch.distributed as dist

    from transformers.generation.continuous_batching.distributed import DistributedHelper

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if not hasattr(tokenizer, "pad_token") and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation=attn_implementation, tp_plan="auto", dtype=torch.float32
    ).eval()

    # Direct broadcast tests: only rank 0's value should propagate to every TP rank
    helper = DistributedHelper(device_mesh=model._device_mesh, cpu_group_timeout=300)

    received_obj = helper.tp_broadcast_object_from_rank_0({"src_rank": rank})
    assert received_obj == {"src_rank": 0}, f"tp_broadcast_object: rank {rank} got {received_obj}"

    sent_tensor = torch.tensor([float(rank)], device=model.device)
    helper.tp_broadcast_from_rank_0(sent_tensor)
    assert sent_tensor.item() == 0.0, f"tp_broadcast_from_rank_0: rank {rank} got {sent_tensor.item()}"

    # CB runs: same seed twice, assert reproducibility AND cross-rank parity
    user_messages = [
        "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?"
    ]
    chats = [[{"role": "user", "content": m}] for m in user_messages]
    tokenized = [tokenizer.apply_chat_template(chat, add_generation_prompt=True) for chat in chats]
    input_ids = [(x if isinstance(x, list) else x["input_ids"]) for x in tokenized]

    cb_config_kwargs = {"use_cuda_graph": use_cuda_graph, "use_async_batching": use_async_batching, "seed": seed}
    gen_config = GenerationConfig(do_sample=do_sample, max_new_tokens=max_new_tokens)
    first_outputs = model.generate_batch(
        inputs=input_ids,
        generation_config=gen_config,
        continuous_batching_config=ContinuousBatchingConfig(**cb_config_kwargs),
    )
    second_outputs = model.generate_batch(
        inputs=input_ids,
        generation_config=gen_config,
        continuous_batching_config=ContinuousBatchingConfig(**cb_config_kwargs),
    )

    # Cross-rank parity: every TP rank must produce the same tokens, otherwise the seed broadcast / TP collectives are
    # diverging silently. Gather the first run's tokens onto all ranks and let rank 0 compare.
    local_tokens = [out.generated_tokens for out in first_outputs.values()]
    gathered_tokens = [None] * helper.tp_size
    dist.all_gather_object(gathered_tokens, local_tokens, group=helper.tp_group)

    if rank != 0:
        return

    assert len(first_outputs) == len(input_ids), f"Expected {len(input_ids)} CB outputs, got {len(first_outputs)}"
    for i, (_, output) in enumerate(first_outputs.items()):
        assert len(output.generated_tokens) > 0, f"Request {i} got no generated tokens"

    for src_rank, src_tokens in enumerate(gathered_tokens):
        if src_tokens != gathered_tokens[0]:
            raise AssertionError(
                f"TP continuous batching diverges across ranks: rank {src_rank} got {src_tokens}, rank 0 got "
                f"{gathered_tokens[0]}"
            )

    second_tokens = [out.generated_tokens for out in second_outputs.values()]
    if local_tokens != second_tokens:
        raise AssertionError(
            f"TP continuous batching is not reproducible across runs with the same seed\n"
            f"First run : {local_tokens}\n"
            f"Second run: {second_tokens}"
        )


def _tp_cancellation_worker(
    rank: int,
    model_id: str,
    attn_implementation: str,
    use_cuda_graph: bool = False,
    use_async_batching: bool = False,
) -> None:
    """Loads `model_id` with `DistributedConfig(tp_size=...)`, submits a long-running streaming request, and cancels it mid-flight.
    The cancellation goes through the cancel-queue + `tp_broadcast_object` path: if the broadcast were broken, the
    non-driver rank's scheduler would not learn about the cancellation and the test would hang or crash on the next
    TP forward pass. Rank 0 owns the assertions."""
    import time

    import torch

    cb_config = ContinuousBatchingConfig(use_cuda_graph=use_cuda_graph, use_async_batching=use_async_batching)

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if not hasattr(tokenizer, "pad_token") and hasattr(tokenizer, "eos_token"):
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation=attn_implementation, tp_plan="auto", dtype=torch.float32
    ).eval()

    chat = [{"role": "user", "content": "Tell me a long story about a robot exploring the galaxy."}]
    tokenized = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
    inputs = tokenized if isinstance(tokenized, list) else tokenized["input_ids"]

    max_new_tokens = 200
    cancel_after_n_chunks = 3

    manager = model.init_continuous_batching(continuous_batching_config=cb_config)
    manager.logit_processor.clear()
    # Warm up synchronously so CUDA-graph capture doesn't eat the streaming-loop deadline below
    manager.warmup()
    manager.start()
    try:
        request_id = manager.add_request(inputs, max_new_tokens=max_new_tokens, streaming=True)
        chunks_seen = 0
        cancelled = False
        deadline = time.time() + 60
        while time.time() < deadline:
            chunk = manager.get_result(request_id=request_id, timeout=2.0)
            if chunk is None:
                # No new chunks for 2s after cancel — cancellation took effect on every rank
                break
            chunks_seen += 1
            if chunks_seen >= cancel_after_n_chunks and not cancelled:
                manager.cancel_request(request_id)
                cancelled = True
        if rank == 0:
            assert cancelled, "Test setup did not reach the cancel call"
            assert chunks_seen < max_new_tokens, (
                f"Cancellation did not stop generation early: saw {chunks_seen} chunks "
                f"for max_new_tokens={max_new_tokens}"
            )
    finally:
        manager.stop(block=True)


@require_torch_multi_accelerator
class ContinuousBatchingTensorParallelTest(unittest.TestCase):
    """Integration tests for continuous batching with tensor parallelism. Each test spawns a TP-sized process group
    via `_init_distributed` (see `tests/test_tensor_parallel_mixin.py`) with the NCCL backend."""

    @property
    def tp_size(self) -> int:
        return min(torch.cuda.device_count(), 2)

    def _run_cb_worker(self, max_new_tokens: int = 20, **worker_kwargs) -> None:
        """Spawn `_tp_continuous_batching_worker` on `tp_size` NCCL processes with sensible defaults."""
        defaults = {
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "attn_implementation": "sdpa",
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "seed": 42,
            "use_cuda_graph": False,
            "use_async_batching": False,
        }
        defaults.update(worker_kwargs)
        _init_distributed(tp=self.tp_size, backend="nccl")(_tp_continuous_batching_worker)(**defaults)

    def test_continuous_batching_tp_fast(self) -> None:
        """Test that continuous batching with `DistributedConfig(tp_size=...)` produces non-empty, reproducible greedy outputs and
        that all TP ranks agree on the generated tokens."""
        self._run_cb_worker(max_new_tokens=4)

    @slow
    def test_continuous_batching_tp_greedy(self) -> None:
        """Test that continuous batching with `DistributedConfig(tp_size=...)` produces non-empty, reproducible greedy outputs and
        that all TP ranks agree on the generated tokens."""
        self._run_cb_worker()

    @slow
    def test_continuous_batching_tp_with_sampling(self) -> None:
        """Test that continuous batching with TP and sampling is reproducible across runs with the same seed and that
        all TP ranks agree on the sampled tokens — implicitly validating the seed broadcast from rank 0."""
        self._run_cb_worker(do_sample=True, seed=123)

    @slow
    def test_continuous_batching_tp_with_cuda_graph(self) -> None:
        """Test that continuous batching with TP and CUDA graphs is reproducible across runs and that all TP ranks
        agree on the generated tokens — captured-graph collectives must stay in sync across ranks."""
        self._run_cb_worker(use_cuda_graph=True)

    @slow
    def test_continuous_batching_tp_with_cuda_graph_and_async(self) -> None:
        """Test that continuous batching with TP, CUDA graphs, and async batching is reproducible across runs and
        that all TP ranks agree on the generated tokens — the toughest combination, exercising both captured-graph
        collectives and the async producer/consumer split."""
        self._run_cb_worker(use_cuda_graph=True, use_async_batching=True)

    @slow
    def test_continuous_batching_tp_cancellation(self) -> None:
        """Test that `cancel_request` propagates across the TP group: the driver enqueues the cancellation, broadcasts
        it to non-driver ranks via `tp_broadcast_object`, and generation stops well before `max_new_tokens`."""
        _init_distributed(tp=self.tp_size, backend="nccl")(_tp_cancellation_worker)(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            attn_implementation="sdpa",
        )

    @slow
    def test_continuous_batching_tp_cancellation_realistic(self) -> None:
        """Test that `cancel_request` propagates across the TP group: the driver enqueues the cancellation, broadcasts
        it to non-driver ranks via `tp_broadcast_object`, and generation stops well before `max_new_tokens`."""
        _init_distributed(tp=self.tp_size, backend="nccl")(_tp_cancellation_worker)(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            attn_implementation="sdpa",
            use_async_batching=True,
            use_cuda_graph=True,
        )
