# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Runtime tests for exported artifacts — running them in real inference settings.

`test_export.py` checks that models *export* across backends (and that each component runs and returns
the right number of outputs). This file is the complement: it takes exported artifacts and exercises
them the way a deployment would — real inputs, real loops, on the actual runtimes (`torch.export`
`module()`, ONNX Runtime, the ExecuTorch runtime) — checking the results match eager. That's the
behaviour a count-only smoke test can't see.

Current coverage — the generation `decode` component:

- **query axis stays dynamic** — the exported multi-token decode runs at query lengths other than the
  captured one;
- **cache mutates in place** — driving the decode against a fixed-size `StaticCache` carries the cache
  across steps in place and matches eager: `torch.export` via `USER_INPUT_MUTATION`, and ONNX Runtime
  via `CudaSession` buffer sharing on the max-performance path (device-resident buffers, in-place input
  updates, no per-step allocations or host round-trips).
"""

import copy
import unittest

import pytest

from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.exporters.utils import decompose_for_generation
from transformers.testing_utils import (
    require_onnxruntime,
    require_onnxscript,
    require_torch,
    require_torch_gpu,
    slow,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


MAX_CACHE_LEN = 16


def _causal_mask(positions, cache_len):
    """Boolean SDPA mask `[1, 1, len(positions), cache_len]`: the query token at absolute
    `positions[i]` attends to cache slots `0..positions[i]` (and nothing ahead)."""
    return (torch.arange(cache_len)[None, :] <= positions[:, None])[None, None]


@slow
@require_torch
class ExportedDecodeRuntimeTest(unittest.TestCase):
    def _tiny_model(self):
        config = LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=64,
            max_position_embeddings=128,
        )
        return LlamaForCausalLM(config).eval()

    def _decompose_static_decode(self, model, prompt):
        """Capture the multi-token `decode` component against a fixed-size `StaticCache`."""
        inputs = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}
        gen_config = GenerationConfig(cache_implementation="static", max_cache_len=MAX_CACHE_LEN, do_sample=False)
        return decompose_for_generation(
            model, copy.deepcopy(inputs), generation_config=gen_config, multi_token_decode=True
        )["decode"]

    # ──────────────────── torch.export (Dynamo) ────────────────────

    @pytest.mark.torch_export_test
    def test_decode_accepts_variable_query_length(self):
        """The multi-token decode's query axis stays dynamic: the exported graph runs at query lengths
        other than the one it was captured with, returning logits for every query position."""
        from transformers.exporters import DynamoConfig, DynamoExporter

        torch.manual_seed(0)
        model = self._tiny_model()
        decode_model, decode_inputs = self._decompose_static_decode(model, torch.randint(0, 64, (1, 4)))
        decode = (
            DynamoExporter()
            .export(decode_model, copy.deepcopy(decode_inputs), config=DynamoConfig(dynamic=True))
            .module()
        )

        for query_len in (1, 2, 4):
            with self.subTest(query_len=query_len):
                past_key_values = copy.deepcopy(decode_inputs["past_key_values"])
                past_key_values.reset()
                positions = torch.arange(query_len)
                with torch.no_grad():
                    out = decode(
                        input_ids=torch.randint(0, 64, (1, query_len)),
                        attention_mask=_causal_mask(positions, MAX_CACHE_LEN),
                        position_ids=positions[None],
                        past_key_values=past_key_values,
                    )
                logits = out.logits if hasattr(out, "logits") else out[0]
                self.assertEqual(logits.shape[:2], (1, query_len))

    @pytest.mark.torch_export_test
    def test_static_cache_mutated_in_place_dynamo(self):
        """The exported decode mutates the passed `StaticCache` in place (a `USER_INPUT_MUTATION`): the
        same cache reused across calls advances its per-layer position counter, so state carries from
        step to step without threading a cache in and out."""
        from transformers.exporters import DynamoConfig, DynamoExporter

        torch.manual_seed(0)
        model = self._tiny_model()
        decode_model, decode_inputs = self._decompose_static_decode(model, torch.randint(0, 64, (1, 4)))
        decode = (
            DynamoExporter()
            .export(decode_model, copy.deepcopy(decode_inputs), config=DynamoConfig(dynamic=True))
            .module()
        )

        past_key_values = copy.deepcopy(decode_inputs["past_key_values"])
        past_key_values.reset()

        def run(input_ids, positions):
            with torch.no_grad():
                decode(
                    input_ids=input_ids,
                    attention_mask=_causal_mask(positions, MAX_CACHE_LEN),
                    position_ids=positions[None],
                    past_key_values=past_key_values,
                )

        self.assertEqual(int(past_key_values.get_seq_length()), 0)
        run(torch.randint(0, 64, (1, 4)), torch.arange(4))  # prefill 4 tokens
        self.assertEqual(int(past_key_values.get_seq_length()), 4)
        run(torch.randint(0, 64, (1, 1)), torch.tensor([4]))  # one decode step
        self.assertEqual(int(past_key_values.get_seq_length()), 5)

    # ──────────────────────── ONNX Runtime ────────────────────────

    @require_torch_gpu
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    def test_static_cache_mutated_in_place_onnx(self):
        """Run the exported decode on ONNX Runtime and check it matches eager while carrying the cache in
        place. The decode graph exposes the cache as matched `input.<name>` / `output.<name>` pairs;
        ORT's `CudaSession.set_buffer_sharing` binds each pair to one device buffer, so the cache updates
        in place. Max-performance path: the shared cache and the step inputs are device-resident buffers
        reused across steps (passed in `feed_dict` by pointer, written in place — no host round-trips),
        and `CudaSession` binds the logits output. Teacher-forced, so the check is on the logits, not a
        greedy argmax a random model can flip on near-ties."""
        import onnxruntime as ort
        from onnxruntime.transformers.io_binding_helper import CudaSession

        from transformers.exporters import OnnxConfig, OnnxExporter

        torch.manual_seed(0)
        model = self._tiny_model()
        prompt = torch.randint(0, 64, (1, 4))
        decode_model, decode_inputs = self._decompose_static_decode(model, prompt)
        onnx_program = OnnxExporter().export(
            decode_model, copy.deepcopy(decode_inputs), config=OnnxConfig(dynamic=True, external_data=False)
        )

        # cache exposed as matched `input.<name>` / `output.<name>` pairs (graph-input order lines up
        # with the `StaticCache` pytree leaves below)
        cache_names = [
            node.name[len("input.") :]
            for node in onnx_program.model_proto.graph.input
            if node.name.startswith("input.")
        ]
        self.assertTrue(cache_names, "decode graph exposes no cache inputs")
        counter_name = next(name for name in cache_names if name.endswith("cumulative_length"))
        vocab_size = model.config.vocab_size

        session = ort.InferenceSession(
            onnx_program.model_proto.SerializeToString(), providers=["CUDAExecutionProvider"]
        )
        cuda = CudaSession(session, torch.device("cuda"))
        for name in cache_names:
            cuda.set_buffer_sharing(f"input.{name}", f"output.{name}")

        # shared cache buffers (device, zeroed = empty cache), passed in `feed_dict` each step so
        # `CudaSession` binds each `input.<name>` and its `output.<name>` to the same buffer → in place
        cache_tensors = [
            t for t in torch.utils._pytree.tree_leaves(decode_inputs["past_key_values"]) if isinstance(t, torch.Tensor)
        ]
        cache = {name: torch.zeros_like(t, device="cuda") for name, t in zip(cache_names, cache_tensors)}
        cache_feed = {f"input.{name}": buf for name, buf in cache.items()}

        # eager reference: a fresh StaticCache fed the same tokens across the whole trajectory
        eager_cache = copy.deepcopy(decode_inputs["past_key_values"])
        eager_cache.reset()

        def eager(input_ids, positions):
            with torch.no_grad():
                return decode_model(
                    input_ids=input_ids,
                    attention_mask=_causal_mask(positions, MAX_CACHE_LEN),
                    position_ids=positions[None],
                    past_key_values=eager_cache,
                ).logits

        # prefill the whole prompt in one multi-token forward (populates the shared cache in place)
        prompt_len = prompt.shape[1]
        positions = torch.arange(prompt_len)
        cuda.allocate_buffers({"logits": (1, prompt_len, vocab_size)})
        out = cuda.infer(
            {
                "input_ids": prompt.cuda(),
                "attention_mask": _causal_mask(positions, MAX_CACHE_LEN).cuda(),
                "position_ids": positions[None].cuda(),
                **cache_feed,
            }
        )
        torch.testing.assert_close(out["logits"].cpu(), eager(prompt, positions), atol=1e-3, rtol=1e-3)
        self.assertEqual(int(cache[counter_name].cpu().item()), prompt_len)

        # decode loop: fixed query=1 device buffers allocated once and updated in place each step
        cuda.allocate_buffers({"logits": (1, 1, vocab_size)})
        input_ids = torch.empty((1, 1), dtype=torch.long, device="cuda")
        position_ids = torch.empty((1, 1), dtype=torch.long, device="cuda")
        attention_mask = torch.empty((1, 1, 1, MAX_CACHE_LEN), dtype=torch.bool, device="cuda")
        slots = torch.arange(MAX_CACHE_LEN, device="cuda")
        for position in range(prompt_len, prompt_len + 2):
            input_ids.fill_(7)  # teacher-forced with a fixed token on both sides
            position_ids.fill_(position)
            attention_mask[0, 0, 0].copy_(slots <= position)
            out = cuda.infer(
                {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids, **cache_feed}
            )
            torch.testing.assert_close(
                out["logits"].cpu(), eager(torch.tensor([[7]]), torch.tensor([position])), atol=1e-3, rtol=1e-3
            )
            self.assertEqual(int(cache[counter_name].cpu().item()), position + 1)
