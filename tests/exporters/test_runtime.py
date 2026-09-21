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
  across steps in place and matches eager: `torch.export` via `USER_INPUT_MUTATION`, ONNX Runtime via the
  shared cache buffers [`OnnxModelRunner`] binds (device-resident, no per-step allocation or host
  round-trip);
- **a saved export runs** — `save_pretrained` then `AutoExportedModel.from_pretrained` generates what the
  in-memory export generated, and a single-graph export forwards like the model it came from.

Everything here drives artifacts through the shipped runners rather than hand-built sessions: the runtime
layer is what a deployment uses, so a test that reimplemented it could pass while the shipped path was
broken.
"""

import copy
import unittest

import pytest

from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.exporters.base import ModelRunner
from transformers.exporters.cache import mask_width
from transformers.exporters.decompose import decompose_for_generation
from transformers.exporters.generator import ExportedGenerator
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


@require_torch
class RuntimeFeedTest(unittest.TestCase):
    """What the runtime hands a graph, decided without exporting anything.

    Each case here is a bug that reached a model sweep and read as a flake for weeks, because the question
    it gets wrong ("how wide is the mask", "does this graph take this kwarg") is only asked while driving a
    real export. They are plain functions, so they can be asked directly.
    """

    class _Graph:
        """A stand-in for a runner: what it declares, and what the trace recorded about it."""

        def __init__(self, input_names, kwargs=None, cache_input="past_key_values"):
            from transformers.exporters.metadata import ExportMetadata

            self.input_names = tuple(input_names)
            self.export_metadata = ExportMetadata.from_dict({"kwargs": kwargs or {}})
            self.cache_input = cache_input
            self.device = "cpu"

        declares = ModelRunner.declares

    def test_declares_a_pytree_kwarg_its_backend_flattened(self):
        """A backend that flattens a kwarg declares only its leaves, so the kwarg's own name is absent —
        the rule that left t5gemma's per-type decoder masks unfed on ONNX."""
        graph = self._Graph(["input.encoder_outputs.last_hidden_state", "decoder_attention_mask.full_attention"])
        self.assertTrue(graph.declares("encoder_outputs", {"last_hidden_state": torch.zeros(1)}))
        self.assertTrue(graph.declares("decoder_attention_mask", {"full_attention": torch.zeros(1)}))
        # A plain tensor has to be named outright: a graph taking `input_features_mask` does not take
        # `input_features`.
        self.assertFalse(graph.declares("input_features", torch.zeros(1)))

    def test_mask_width_comes_from_the_layer_the_mask_belongs_to(self):
        """`get_max_length()` answers for the *longest* layer, which on mllama is the vision-sized
        cross-attention one — the mask belongs to the self-attention layer beside it."""

        class _Cache:
            def get_mask_sizes(self, query_length, layer_idx):
                return 256, 0

            def get_max_length(self):
                return 904

        self.assertEqual(mask_width(_Cache(), query_length=7), 256)

    def test_mask_is_built_at_the_rank_the_graph_was_traced_with(self):
        """`generate` omits the mask when nothing is padded. Rebuilding it as the 4-D causal mask for a
        graph traced on the 2-D padding mask puts the head axis where the width belongs, which the graph's
        own comparison then rejects (`input_ids.size()[1] <= attention_mask.size()[1]`)."""
        two_dimensional = self._Graph(["attention_mask"], {"attention_mask": {"rank": 2, "dtype": "bool"}})
        four_dimensional = self._Graph(["attention_mask"], {"attention_mask": {"rank": 4, "dtype": "float32"}})
        generator = ExportedGenerator.__new__(ExportedGenerator)
        generator._device = torch.device("cpu")
        positions = torch.arange(3)[None]

        flat = generator._mask_feed(two_dimensional, None, positions, cache_len=3)["attention_mask"]
        self.assertEqual(flat.shape, (1, 3))
        self.assertEqual(flat.dtype, torch.bool)

        causal = generator._mask_feed(four_dimensional, None, positions, cache_len=3)["attention_mask"]
        self.assertEqual(causal.dim(), 4)


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
        decode = decompose_for_generation(
            model, copy.deepcopy(inputs), generation_config=gen_config, multi_token_decode=True
        )["decode"]
        return decode.module, decode.inputs

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
            .runner()
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
                logits = out["logits"]
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
            .runner()
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
        """Run the exported decode on ONNX Runtime through its shipped runner and check it matches eager
        while carrying the cache in place.

        The graph exposes the cache as matched `input.<name>` / `output.<name>` pairs, which is what lets
        [`OnnxModelRunner`] bind each pair to one device buffer: the updated K/V are written straight back
        into the tensors passed in, so no full-size cache output is allocated per step and nothing goes
        through the host. Teacher-forced, so the check is on the logits, not a greedy argmax a random model
        can flip on near-ties."""
        from transformers.exporters import OnnxConfig, OnnxExporter

        torch.manual_seed(0)
        model = self._tiny_model()
        prompt = torch.randint(0, 64, (1, 4))
        decode_model, decode_inputs = self._decompose_static_decode(model, prompt)
        exported = OnnxExporter().export(
            decode_model, copy.deepcopy(decode_inputs), config=OnnxConfig(dynamic=True, external_data=False)
        )

        # The pairing is a property of what the exporter emitted, so assert it on the graph itself: without
        # it there is nothing for the runner to share a buffer between.
        graph_inputs = {node.name for node in exported.artifact.model_proto.graph.input}
        graph_outputs = {node.name for node in exported.artifact.model_proto.graph.output}
        cache_names = {name[len("input.") :] for name in graph_inputs if name.startswith("input.")}
        self.assertTrue(cache_names, "decode graph exposes no cache inputs")
        self.assertTrue(
            {f"output.{name}" for name in cache_names} <= graph_outputs,
            "cache inputs have no matching outputs, so they cannot share a buffer",
        )

        decode = exported.runner(device="cuda")
        # Device-resident, because that is what makes the update in place: the runner binds what it is given
        # by pointer, but moves a tensor that lives elsewhere first — and writes would then land in the copy.
        past_key_values = torch.utils._pytree.tree_map(
            lambda leaf: leaf.cuda() if isinstance(leaf, torch.Tensor) else leaf,
            copy.deepcopy(decode_inputs["past_key_values"]),
        )
        past_key_values.reset()

        def eager(input_ids, positions, cache):
            with torch.no_grad():
                return decode_model(
                    input_ids=input_ids,
                    attention_mask=_causal_mask(positions, MAX_CACHE_LEN),
                    position_ids=positions[None],
                    past_key_values=cache,
                ).logits

        eager_cache = copy.deepcopy(decode_inputs["past_key_values"])
        eager_cache.reset()

        # prefill the whole prompt in one multi-token call, then two single-token steps over the same cache
        steps = [(prompt, torch.arange(prompt.shape[1]))]
        steps += [
            (torch.tensor([[7]]), torch.tensor([position])) for position in range(prompt.shape[1], prompt.shape[1] + 2)
        ]
        for input_ids, positions in steps:
            outputs = decode(
                input_ids=input_ids.cuda(),
                attention_mask=_causal_mask(positions, MAX_CACHE_LEN).cuda(),
                position_ids=positions[None].cuda(),
                past_key_values=past_key_values,
            )
            expected = eager(input_ids, positions, eager_cache)
            torch.testing.assert_close(outputs["logits"].cpu(), expected, atol=1e-3, rtol=1e-3)
            # The cache carried the step: its own counter advanced to the last position written.
            self.assertEqual(int(past_key_values.get_seq_length()), int(positions[-1]) + 1)

    # ──────────────────────── save / load ────────────────────────

    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    def test_saved_export_generates_like_the_in_memory_one(self):
        """An export that has been through disk generates what it generated in memory.

        This is the deployment path end to end: `export_for_generation` -> `save_pretrained` ->
        `AutoExportedModel.from_pretrained` -> `generate`. It is also what proves the manifest carries
        enough — the graphs describe their own shapes, but which file is which component, what precision
        each computes in, and which cache they were traced against only survive because the save records
        them."""
        import tempfile

        from transformers.exporters import AutoExportedModel, OnnxConfig, OnnxExporter

        torch.manual_seed(0)
        model = self._tiny_model()
        model.generation_config.pad_token_id = 0
        prompt = torch.randint(0, 64, (1, 4))
        inputs = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}

        exported = OnnxExporter().export_for_generation(
            model, copy.deepcopy(inputs), config=OnnxConfig(dynamic=True, external_data=False)
        )
        in_memory = exported.runtime(device="cpu").generate(**inputs, max_new_tokens=4, do_sample=False)

        with tempfile.TemporaryDirectory() as directory:
            exported.save_pretrained(directory)
            loaded = AutoExportedModel.from_pretrained(directory, device="cpu")
            from_disk = loaded.generate(**inputs, max_new_tokens=4, do_sample=False)

        self.assertListEqual(from_disk.tolist(), in_memory.tolist())

    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    def test_saved_single_graph_matches_eager(self):
        """A non-generative export loads as an [`ExportedModel`] and forwards like the model it came from —
        the shape most exports are (a classifier, an encoder), and the one `generate` has no part in."""
        import tempfile

        from transformers.exporters import AutoExportedModel, OnnxConfig, OnnxExporter

        torch.manual_seed(0)
        model = self._tiny_model()
        prompt = torch.randint(0, 64, (1, 4))
        inputs = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}
        with torch.no_grad():
            expected = model(**copy.deepcopy(inputs)).logits

        exported = OnnxExporter().export(
            model, copy.deepcopy(inputs), config=OnnxConfig(dynamic=True, external_data=False)
        )
        with tempfile.TemporaryDirectory() as directory:
            exported.save_pretrained(directory)
            loaded = AutoExportedModel.from_pretrained(directory, device="cpu")
            self.assertIsNot(type(loaded).__name__, "ExportedGenerator")
            outputs = loaded(**inputs)

        torch.testing.assert_close(outputs.logits, expected, atol=1e-3, rtol=1e-3)
