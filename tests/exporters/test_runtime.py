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
"""Runtime tests for exported artifacts: the exported `decode` steps a static cache in place at any query length
(`torch.export` and ONNX Runtime), an exported draft assists generation, and a saved export generates and forwards
like the in-memory one. Everything runs through the shipped runners, the layer a deployment uses.
"""

import copy
import functools
import tempfile
import unittest

import pytest

from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.exporters import (
    AutoExportedModel,
    DynamoConfig,
    DynamoExporter,
    OnnxConfig,
    OnnxExporter,
)
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

        def __init__(self, input_names, kwargs=None):
            from transformers.exporters.metadata import ExportMetadata

            self.input_names = tuple(input_names)
            self.export_metadata = ExportMetadata.from_dict({"kwargs": kwargs or {}})
            self.cache_inputs = ("past_key_values",)
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

    def _assert_decode_steps_like_eager(self, decode, decode_model, decode_inputs, prompt, device):
        """Prefill `prompt`, then a 2-token and a 1-token step, on one cache the exported `decode` mutates in place:
        each step matches eager, returns the last position's logits, and advances the cache it was handed."""
        past_key_values = torch.utils._pytree.tree_map(
            lambda leaf: leaf.to(device) if isinstance(leaf, torch.Tensor) else leaf,
            copy.deepcopy(decode_inputs["past_key_values"]),
        )
        past_key_values.reset()
        eager_cache = copy.deepcopy(decode_inputs["past_key_values"])
        eager_cache.reset()

        start = prompt.shape[1]
        steps = [(prompt, torch.arange(start)), (torch.tensor([[7, 8]]), torch.arange(start, start + 2))]
        steps.append((torch.tensor([[9]]), torch.tensor([start + 2])))
        for input_ids, positions in steps:
            mask = _causal_mask(positions, MAX_CACHE_LEN)
            with torch.no_grad():
                outputs = decode(
                    input_ids=input_ids.to(device),
                    attention_mask=mask.to(device),
                    position_ids=positions[None].to(device),
                    past_key_values=past_key_values,
                )
                expected = decode_model(
                    input_ids=input_ids,
                    attention_mask=mask,
                    position_ids=positions[None],
                    past_key_values=eager_cache,
                    logits_to_keep=decode_inputs["logits_to_keep"],
                ).logits
            self.assertEqual(outputs["logits"].shape[:2], (1, 1))
            torch.testing.assert_close(outputs["logits"].cpu(), expected, atol=1e-3, rtol=1e-3)
            self.assertEqual(int(past_key_values.get_seq_length()), int(positions[-1]) + 1)

    @pytest.mark.torch_export_test
    def test_decode_steps_in_place_dynamo(self):
        """The multi-token decode keeps its query axis dynamic, and mutates the `StaticCache` it is passed (a
        `USER_INPUT_MUTATION`), so state carries from step to step."""
        torch.manual_seed(0)
        prompt = torch.randint(0, 64, (1, 4))
        decode_model, decode_inputs = self._decompose_static_decode(self._tiny_model(), prompt)
        exported = DynamoExporter().export(
            decode_model, copy.deepcopy(decode_inputs), config=DynamoConfig(dynamic=True)
        )
        self._assert_decode_steps_like_eager(exported.runtime(), decode_model, decode_inputs, prompt, "cpu")

    @require_torch_gpu
    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    def test_decode_steps_in_place_onnx(self):
        """The same on ONNX Runtime: the graph exposes the cache as matched `input.<name>` / `output.<name>`
        pairs, which [`OnnxModelRunner`] binds to one device buffer, so the update lands in the tensors passed in."""
        torch.manual_seed(0)
        prompt = torch.randint(0, 64, (1, 4))
        decode_model, decode_inputs = self._decompose_static_decode(self._tiny_model(), prompt)
        exported = OnnxExporter().export(decode_model, copy.deepcopy(decode_inputs), config=OnnxConfig(dynamic=True))
        graph = exported.artifact.model_proto.graph
        cache_names = {node.name[len("input.") :] for node in graph.input if node.name.startswith("input.")}
        self.assertTrue(cache_names, "decode graph exposes no cache inputs")
        self.assertLessEqual({f"output.{name}" for name in cache_names}, {node.name for node in graph.output})
        # Device-resident: the runner binds by pointer, and would copy (then write into the copy) a host tensor.
        self._assert_decode_steps_like_eager(
            exported.runtime(device="cuda").runner, decode_model, decode_inputs, prompt, "cuda"
        )

    # ──────────────────────── save / load ────────────────────────

    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    def test_saved_export_generates_like_the_in_memory_one(self):
        """`export_for_generation` -> `save_pretrained` -> `AutoExportedModel.from_pretrained` -> `generate` gives
        what the in-memory export generated: the manifest carries the components, precisions and cache."""
        torch.manual_seed(0)
        model = self._tiny_model()
        model.generation_config.pad_token_id = 0
        prompt = torch.randint(0, 64, (1, 4))
        inputs = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}

        exported = OnnxExporter().export_for_generation(model, copy.deepcopy(inputs), config=OnnxConfig(dynamic=True))
        in_memory = exported.runtime(device="cpu").generate(**inputs, max_new_tokens=4, do_sample=False)

        with tempfile.TemporaryDirectory() as directory:
            exported.save_pretrained(directory)
            loaded = AutoExportedModel.from_pretrained(directory, device="cpu")
            from_disk = loaded.generate(**inputs, max_new_tokens=4, do_sample=False)

        self.assertListEqual(from_disk.tolist(), in_memory.tolist())

    @pytest.mark.torch_export_test
    def test_exported_draft_assists_generation(self):
        """An `assistant_model` passed to `export_for_generation` is exported with the target and saved beside it,
        and its runtime drafts for the target's `generate` like any `assistant_model`. A draft with other weights
        disagrees with the target, so candidates are rejected and the caches rolled back, and greedy assisted
        decoding must still match plain greedy."""
        torch.manual_seed(0)
        target = self._tiny_model()
        draft = self._tiny_model()
        prompt = torch.randint(0, 64, (1, 6))
        inputs = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}
        generate_kwargs = {"do_sample": False, "max_new_tokens": 8, "min_new_tokens": 8}
        expected = target.generate(**inputs, **generate_kwargs)
        self.assertFalse(torch.equal(draft.generate(**inputs, **generate_kwargs), expected))

        exported = DynamoExporter().export_for_generation(
            target, copy.deepcopy(inputs), config=DynamoConfig(dynamic=True), assistant_model=draft
        )

        def assisted_generate(runtime, draft_runtime):
            # Count the draft's steps, keeping the signature `generate` reads
            forward = draft_runtime.forward
            calls = []
            draft_runtime.forward = functools.wraps(forward)(lambda **kwargs: calls.append(1) or forward(**kwargs))
            return runtime.generate(**inputs, **generate_kwargs, assistant_model=draft_runtime), len(calls)

        in_memory, draft_steps = assisted_generate(exported.runtime(), exported.assistant.runtime())
        self.assertGreater(draft_steps, 0)
        self.assertListEqual(in_memory.tolist(), expected.tolist())

        with tempfile.TemporaryDirectory() as directory:
            exported.save_pretrained(directory)
            loaded = AutoExportedModel.from_pretrained(directory)
            loaded_draft = AutoExportedModel.from_pretrained(directory, subfolder="assistant")
            from_disk, draft_steps = assisted_generate(loaded, loaded_draft)
        self.assertGreater(draft_steps, 0)
        self.assertListEqual(from_disk.tolist(), expected.tolist())

    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    def test_saved_single_graph_matches_eager(self):
        """A non-generative export loads as an [`ExportedModel`] and forwards like the model it came from —
        the shape most exports are (a classifier, an encoder)."""
        torch.manual_seed(0)
        model = self._tiny_model()
        prompt = torch.randint(0, 64, (1, 4))
        inputs = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}
        with torch.no_grad():
            expected = model(**copy.deepcopy(inputs)).logits

        exported = OnnxExporter().export(model, copy.deepcopy(inputs), config=OnnxConfig(dynamic=True))
        with tempfile.TemporaryDirectory() as directory:
            exported.save_pretrained(directory)
            loaded = AutoExportedModel.from_pretrained(directory, device="cpu")
            self.assertNotIsInstance(loaded, ExportedGenerator)
            outputs = loaded(**inputs)

        torch.testing.assert_close(outputs.logits, expected, atol=1e-3, rtol=1e-3)
