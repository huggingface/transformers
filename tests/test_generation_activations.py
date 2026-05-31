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
"""Tests for the GenerationActivations utility."""

import unittest

import torch
from parameterized import parameterized

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.activations import GenerationActivations
from transformers.testing_utils import require_torch_accelerator, slow


# Small model used for fast CI-friendly tests.  Must support
# output_hidden_states in generate() and be small enough to load on CPU.
_CI_MODEL = "Qwen/Qwen3-0.6B"


def _generate_and_extract(
    model_name: str = _CI_MODEL,
    prompt: str = "The capital of France is",
    max_new_tokens: int = 8,
    **generate_kwargs,
) -> tuple:
    """Helper: generate one completion and return (gen_out, tokenizer, inputs)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_out = model.generate(
        **inputs,
        output_hidden_states=True,
        return_dict_in_generate=True,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        **generate_kwargs,
    )
    return gen_out, tokenizer, inputs


@require_torch_accelerator
@slow
class GenerationActivationsTest(unittest.TestCase):
    """Core tests for GenerationActivations."""

    def test_from_generate_output_basic(self):
        """Shape sanity: [num_layers, total_tokens, hidden_dim]."""
        gen_out, tokenizer, inputs = _generate_and_extract(max_new_tokens=4)

        acts = GenerationActivations.from_generate_output(gen_out)

        self.assertIsInstance(acts, GenerationActivations)
        self.assertEqual(acts.hidden_states.ndim, 3)  # [L, T, D]
        L, T, D = acts.hidden_states.shape
        self.assertEqual(L, acts.num_layers, "num_layers should match the first dim")
        self.assertEqual(D, acts.hidden_dim, "hidden_dim should match the last dim")
        self.assertGreater(T, acts.prompt_len, "total > prompt")
        # batch_size should be None for batch=1
        self.assertIsNone(acts.batch_size)
        self.assertIsNone(acts.attention_mask)

    def test_prompt_vs_generated_split(self):
        """prompt_hidden_states and generated_hidden_states have correct sizes."""
        gen_out, _, inputs = _generate_and_extract(max_new_tokens=6)
        acts = GenerationActivations.from_generate_output(gen_out)

        prompt = acts.prompt_hidden_states
        generated = acts.generated_hidden_states

        # Prompt: [L, prompt_len, D]
        self.assertEqual(prompt.shape[0], acts.num_layers)
        self.assertEqual(prompt.shape[1], acts.prompt_len)
        self.assertEqual(prompt.shape[2], acts.hidden_dim)

        # Generated: [L, gen_len, D]
        gen_len = acts.total_len - acts.prompt_len
        self.assertEqual(generated.shape[0], acts.num_layers)
        self.assertEqual(generated.shape[1], gen_len)
        self.assertEqual(generated.shape[2], acts.hidden_dim)

        # Concatenating prompt + generated should give the full tensor
        recon = torch.cat([prompt, generated], dim=1)
        self.assertTrue(
            torch.equal(recon, acts.hidden_states),
            "prompt + generated should reconstruct full hidden_states",
        )

    def test_total_len_property(self):
        """total_len matches the token axis."""
        gen_out, _, _ = _generate_and_extract(max_new_tokens=3)
        acts = GenerationActivations.from_generate_output(gen_out)
        self.assertEqual(
            acts.total_len,
            acts.hidden_states.shape[1],
            "total_len should be the token-axis size",
        )

    def test_hidden_states_none_raises(self):
        """ValueError when gen_output.hidden_states is None."""
        gen_out, _, _ = _generate_and_extract(max_new_tokens=1)
        # Manually null out the field to simulate missing flag
        gen_out.hidden_states = None
        with self.assertRaises(ValueError) as ctx:
            GenerationActivations.from_generate_output(gen_out)
        self.assertIn("output_hidden_states=True", str(ctx.exception))

    def test_empty_raw_raises(self):
        """ValueError when raw_hidden_states is an empty tuple."""
        with self.assertRaises(ValueError) as ctx:
            GenerationActivations._stack((), None)
        self.assertIn("empty tuple", str(ctx.exception))

    def test_from_generate_dict(self):
        """from_generate_dict accepts the expected dictionary format."""
        gen_out, _, _ = _generate_and_extract(max_new_tokens=2)
        dummy_dict = {"decoder_hidden_states": gen_out.hidden_states}
        acts = GenerationActivations.from_generate_dict(dummy_dict)
        self.assertEqual(acts.num_layers, len(gen_out.hidden_states[0]))
        self.assertIsNone(acts.batch_size)

    def test_from_generate_dict_missing_key_raises(self):
        """ValueError when the dict is missing 'decoder_hidden_states'."""
        with self.assertRaises(ValueError) as ctx:
            GenerationActivations.from_generate_dict({})
        self.assertIn("decoder_hidden_states", str(ctx.exception))

    def test_pool_layers(self):
        """pool_layers reduces the layer axis via adaptive avg pooling."""
        gen_out, _, _ = _generate_and_extract(max_new_tokens=4)
        acts = GenerationActivations.from_generate_output(gen_out)

        target = acts.num_layers // 2  # e.g. 29 → 14 on Qwen3-0.6B
        if target < 1:
            target = 1

        pooled = acts.pool_layers(target)
        self.assertEqual(pooled.shape[0], target)
        self.assertEqual(pooled.shape[1], acts.total_len)
        self.assertEqual(pooled.shape[2], acts.hidden_dim)

    def test_pool_layers_identity(self):
        """pool_layers with target == num_layers returns identical tensor."""
        gen_out, _, _ = _generate_and_extract(max_new_tokens=2)
        acts = GenerationActivations.from_generate_output(gen_out)
        pooled = acts.pool_layers(acts.num_layers)
        self.assertTrue(
            torch.equal(pooled, acts.hidden_states),
            "Identity pooling should return the same tensor",
        )

    def test_pool_layers_too_large_raises(self):
        """ValueError when target_layers > num_layers."""
        gen_out, _, _ = _generate_and_extract(max_new_tokens=1)
        acts = GenerationActivations.from_generate_output(gen_out)
        with self.assertRaises(ValueError) as ctx:
            acts.pool_layers(acts.num_layers + 10)
        self.assertIn("must be ≤ num_layers", str(ctx.exception))

    def test_to_device(self):
        """to() moves tensors correctly."""
        gen_out, _, _ = _generate_and_extract(max_new_tokens=2)
        acts = GenerationActivations.from_generate_output(gen_out)

        # Move to CPU (may already be there)
        cpu_acts = acts.to("cpu")
        self.assertEqual(cpu_acts.hidden_states.device.type, "cpu")
        self.assertEqual(cpu_acts.prompt_len, acts.prompt_len)
        self.assertEqual(cpu_acts.num_layers, acts.num_layers)

    def test_repr(self):
        """__repr__ is informative and non-crashing."""
        gen_out, _, _ = _generate_and_extract(max_new_tokens=1)
        acts = GenerationActivations.from_generate_output(gen_out)
        r = repr(acts)
        self.assertIn("GenerationActivations", r)
        self.assertIn("shape=", r)
        self.assertIn("num_layers=", r)


@require_torch_accelerator
@slow
class GenerationActivationsBatchTest(unittest.TestCase):
    """Tests for batched inputs (batch_size > 1)."""

    def test_single_batch_has_no_batch_size(self):
        """batch=1 → batch_size=None, squeezed tensor."""
        gen_out, _, _ = _generate_and_extract(max_new_tokens=3)
        acts = GenerationActivations.from_generate_output(gen_out)
        self.assertIsNone(acts.batch_size)
        self.assertEqual(acts.hidden_states.ndim, 3)  # [L, T, D]

    @parameterized.expand(
        [
            (2, "left"),
            (2, "right"),
        ]
    )
    def test_multi_batch_shapes(self, batch_size: int, padding_side: str):
        """Batch > 1 returns [B, L, T, D] with batch_size set."""
        model_name = _CI_MODEL
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.padding_side = padding_side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        prompts = [
            "The capital of France is",
            "The largest ocean on Earth is",
        ][:batch_size]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        gen_out = model.generate(
            **inputs,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_new_tokens=5,
            do_sample=False,
        )

        acts = GenerationActivations.from_generate_output(
            gen_out,
            attention_mask=inputs.get("attention_mask"),
        )

        self.assertEqual(acts.batch_size, batch_size)
        # Shape should be [B, L, T, D]
        self.assertEqual(acts.hidden_states.shape[0], batch_size)
        self.assertEqual(acts.hidden_states.shape[1], acts.num_layers)
        self.assertEqual(acts.hidden_states.shape[3], acts.hidden_dim)

        # Attention mask should be present for left-padded batches
        if padding_side == "left" and inputs.get("attention_mask") is not None:
            self.assertIsNotNone(acts.attention_mask)
        else:
            # right-padding: attention_mask may or may not be set depending on
            # whether the tokenizer includes it
            pass

    def test_multi_batch_prompt_split(self):
        """prompt_hidden_states is [B, L, prompt_len, D] for batched."""
        tokenizer = AutoTokenizer.from_pretrained(_CI_MODEL)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(_CI_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()

        prompts = ["The capital of France is", "What is 2+2?"]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        gen_out = model.generate(
            **inputs,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_new_tokens=4,
            do_sample=False,
        )

        acts = GenerationActivations.from_generate_output(gen_out, attention_mask=inputs["attention_mask"])

        prompt = acts.prompt_hidden_states
        self.assertEqual(prompt.shape[0], 2)  # batch
        self.assertEqual(prompt.shape[1], acts.num_layers)
        self.assertEqual(prompt.shape[2], acts.prompt_len)
        self.assertEqual(prompt.shape[3], acts.hidden_dim)

        generated = acts.generated_hidden_states
        self.assertEqual(generated.shape[0], 2)
        self.assertEqual(generated.shape[1], acts.num_layers)
        self.assertEqual(generated.shape[3], acts.hidden_dim)

    def test_multi_batch_pool_layers(self):
        """pool_layers works correctly for batched activations."""
        tokenizer = AutoTokenizer.from_pretrained(_CI_MODEL)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(_CI_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()

        prompts = ["Hello world", "What is AI?"]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        gen_out = model.generate(
            **inputs,
            output_hidden_states=True,
            return_dict_in_generate=True,
            max_new_tokens=3,
            do_sample=False,
        )

        acts = GenerationActivations.from_generate_output(gen_out)
        target = max(1, acts.num_layers // 4)
        pooled = acts.pool_layers(target)
        self.assertEqual(pooled.shape[0], 2)  # batch
        self.assertEqual(pooled.shape[1], target)  # pooled layers
        self.assertEqual(pooled.shape[3], acts.hidden_dim)


class GenerationActivationsProjectionTest(unittest.TestCase):
    """Tests for per-layer and per-token projection properties."""

    def test_layer_index(self):
        """layer(n) returns [T, D] for the correct layer."""
        gen_out, _, _ = _generate_and_extract(max_new_tokens=4)
        acts = GenerationActivations.from_generate_output(gen_out)
        l0 = acts.layer(0)
        self.assertEqual(l0.ndim, 2)
        self.assertEqual(l0.shape, (acts.total_len, acts.hidden_dim))
        l_last = acts.layer(acts.num_layers - 1)
        self.assertEqual(l_last.shape, (acts.total_len, acts.hidden_dim))

    def test_last_layer(self):
        """last_layer equals layer(-1)."""
        gen_out, _, _ = _generate_and_extract(max_new_tokens=3)
        acts = GenerationActivations.from_generate_output(gen_out)
        self.assertTrue(torch.equal(acts.last_layer, acts.layer(acts.num_layers - 1)))

    def test_last_token(self):
        """last_token returns [L, D]."""
        gen_out, _, _ = _generate_and_extract(max_new_tokens=2)
        acts = GenerationActivations.from_generate_output(gen_out)
        lt = acts.last_token
        self.assertEqual(lt.ndim, 2)
        self.assertEqual(lt.shape, (acts.num_layers, acts.hidden_dim))
        # Verify it matches the last position of the full tensor
        self.assertTrue(torch.equal(lt, acts.hidden_states[:, -1, :]))

    def test_mean_pool_tokens(self):
        """mean_pool_tokens returns [L, D]."""
        gen_out, _, _ = _generate_and_extract(max_new_tokens=3)
        acts = GenerationActivations.from_generate_output(gen_out)
        mp = acts.mean_pool_tokens()
        self.assertEqual(mp.ndim, 2)
        self.assertEqual(mp.shape, (acts.num_layers, acts.hidden_dim))
        # Verify it equals manual mean over token axis
        self.assertTrue(torch.allclose(mp, acts.hidden_states.mean(dim=1)))
