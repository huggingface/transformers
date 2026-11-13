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
  via `CudaSession` buffer sharing. That contract belongs to the exported graph rather than to a
  device, so the ONNX check runs in host memory everywhere and upgrades itself to device-resident
  buffers on CUDA (the max-performance path: in-place input updates, no per-step allocations or host
  round-trips) — a provider that copies instead of aliasing device memory only shows up there.

Saving and loading exports is tested in `test_save_load.py`.
"""

import copy
import os
import tempfile
import unittest

import pytest

from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM
from transformers.exporters.utils import decompose_for_generation
from transformers.testing_utils import (
    require_executorch,
    require_onnxruntime,
    require_onnxscript,
    require_openvino,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


MAX_CACHE_LEN = 16


def _causal_mask(positions, cache_len):
    """Boolean SDPA mask `[1, 1, len(positions), cache_len]`: the query token at absolute
    `positions[i]` attends to cache slots `0..positions[i]` (and nothing ahead)."""
    return (torch.arange(cache_len)[None, :] <= positions[:, None])[None, None]


def _tiny_model():
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


def _decompose_static_decode(model, prompt):
    """Capture the multi-token `decode` component against a fixed-size `StaticCache`."""
    inputs = {"input_ids": prompt, "attention_mask": torch.ones_like(prompt)}
    gen_config = GenerationConfig(cache_implementation="static", max_cache_len=MAX_CACHE_LEN, do_sample=False)
    return decompose_for_generation(
        model, copy.deepcopy(inputs), generation_config=gen_config, multi_token_decode=True
    )["decode"]


@slow
@require_torch
class ExportedDecodeRuntimeTest(unittest.TestCase):
    # ──────────────────── torch.export (Dynamo) ────────────────────

    @pytest.mark.torch_export_test
    def test_decode_accepts_variable_query_length(self):
        """The multi-token decode's query axis stays dynamic: the exported graph runs at query lengths
        other than the one it was captured with, returning logits for every query position."""
        from transformers.exporters import DynamoConfig, DynamoExporter

        torch.manual_seed(0)
        model = _tiny_model()
        decode_model, decode_inputs = _decompose_static_decode(model, torch.randint(0, 64, (1, 4)))
        decode = DynamoExporter().export(decode_model, copy.deepcopy(decode_inputs), config=DynamoConfig(dynamic=True))

        for query_len in (1, 2, 4):
            with self.subTest(query_len=query_len):
                cache = copy.deepcopy(decode_inputs["cache"])
                cache.reset()
                positions = torch.arange(query_len)
                with torch.no_grad():
                    out = decode(
                        input_ids=torch.randint(0, 64, (1, query_len)),
                        attention_mask=_causal_mask(positions, MAX_CACHE_LEN),
                        position_ids=positions[None],
                        cache=cache,
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
        model = _tiny_model()
        decode_model, decode_inputs = _decompose_static_decode(model, torch.randint(0, 64, (1, 4)))
        decode = DynamoExporter().export(decode_model, copy.deepcopy(decode_inputs), config=DynamoConfig(dynamic=True))

        cache = copy.deepcopy(decode_inputs["cache"])
        cache.reset()

        def run(input_ids, positions):
            with torch.no_grad():
                decode(
                    input_ids=input_ids,
                    attention_mask=_causal_mask(positions, MAX_CACHE_LEN),
                    position_ids=positions[None],
                    cache=cache,
                )

        self.assertEqual(int(cache.get_seq_length()), 0)
        run(torch.randint(0, 64, (1, 4)), torch.arange(4))  # prefill 4 tokens
        self.assertEqual(int(cache.get_seq_length()), 4)
        run(torch.randint(0, 64, (1, 1)), torch.tensor([4]))  # one decode step
        self.assertEqual(int(cache.get_seq_length()), 5)

    # ──────────────────────── ONNX Runtime ────────────────────────

    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    def test_static_cache_mutated_in_place_onnx(self):
        """Run the exported decode on ONNX Runtime and check it matches eager while carrying the cache in
        place. The decode graph exposes the cache as matched `input.<name>` / `output.<name>` pairs;
        ORT's `CudaSession.set_buffer_sharing` binds each pair to one buffer, so the cache updates in
        place. The shared cache and the step inputs are buffers reused across steps (passed in
        `feed_dict` by pointer, written in place — no per-step allocations), and `CudaSession` binds the
        logits output. Teacher-forced, so the check is on the logits, not a greedy argmax a random model
        can flip on near-ties.

        Nothing here is tied to a device: it runs in host memory wherever ONNX Runtime is installed, and
        on a CUDA box with a matching build it upgrades itself to the max-performance path, where the
        buffers are device resident and a provider that copies instead of aliasing them would show up."""
        import onnx_ir
        import onnxruntime as ort
        from onnxruntime.transformers.io_binding_helper import CudaSession

        from transformers.exporters import OnnxConfig, OnnxExporter

        if torch_device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            provider, buffer_device = "CUDAExecutionProvider", "cuda"
        else:
            provider, buffer_device = "CPUExecutionProvider", "cpu"

        torch.manual_seed(0)
        model = _tiny_model()
        prompt = torch.randint(0, 64, (1, 4))
        decode_model, decode_inputs = _decompose_static_decode(model, prompt)
        onnx_program = (
            OnnxExporter().export(decode_model, copy.deepcopy(decode_inputs), config=OnnxConfig(dynamic=True)).program
        )

        # cache exposed as matched `input.<name>` / `output.<name>` pairs (graph-input order lines up
        # with the `StaticCache` pytree leaves below)
        cache_names = [
            value.name[len("input.") :] for value in onnx_program.graph.inputs if value.name.startswith("input.")
        ]
        self.assertTrue(cache_names, "decode graph exposes no cache inputs")
        counter_name = next(name for name in cache_names if name.endswith("cumulative_length"))
        vocab_size = model.config.vocab_size

        session = ort.InferenceSession(onnx_ir.to_proto(onnx_program).SerializeToString(), providers=[provider])
        binding = CudaSession(session, torch.device(buffer_device))
        for name in cache_names:
            binding.set_buffer_sharing(f"input.{name}", f"output.{name}")

        # shared cache buffers (device, zeroed = empty cache), passed in `feed_dict` each step so
        # `CudaSession` binds each `input.<name>` and its `output.<name>` to the same buffer → in place
        cache_tensors = [
            t for t in torch.utils._pytree.tree_leaves(decode_inputs["cache"]) if isinstance(t, torch.Tensor)
        ]
        cache = {name: torch.zeros_like(t, device=buffer_device) for name, t in zip(cache_names, cache_tensors)}
        cache_feed = {f"input.{name}": buf for name, buf in cache.items()}

        # eager reference: a fresh StaticCache fed the same tokens across the whole trajectory
        eager_cache = copy.deepcopy(decode_inputs["cache"])
        eager_cache.reset()

        def eager(input_ids, positions):
            with torch.no_grad():
                return decode_model(
                    input_ids=input_ids,
                    attention_mask=_causal_mask(positions, MAX_CACHE_LEN),
                    position_ids=positions[None],
                    cache=eager_cache,
                ).logits

        # prefill the whole prompt in one multi-token forward (populates the shared cache in place)
        prompt_len = prompt.shape[1]
        positions = torch.arange(prompt_len)
        binding.allocate_buffers({"logits": (1, prompt_len, vocab_size)})
        out = binding.infer(
            {
                "input_ids": prompt.to(buffer_device),
                "attention_mask": _causal_mask(positions, MAX_CACHE_LEN).to(buffer_device),
                "position_ids": positions[None].to(buffer_device),
                **cache_feed,
            }
        )
        torch.testing.assert_close(out["logits"].cpu(), eager(prompt, positions), atol=1e-3, rtol=1e-3)
        self.assertEqual(int(cache[counter_name].cpu().item()), prompt_len)

        # decode loop: fixed query=1 buffers allocated once and updated in place each step
        binding.allocate_buffers({"logits": (1, 1, vocab_size)})
        input_ids = torch.empty((1, 1), dtype=torch.long, device=buffer_device)
        position_ids = torch.empty((1, 1), dtype=torch.long, device=buffer_device)
        attention_mask = torch.empty((1, 1, 1, MAX_CACHE_LEN), dtype=torch.bool, device=buffer_device)
        slots = torch.arange(MAX_CACHE_LEN, device=buffer_device)
        for position in range(prompt_len, prompt_len + 2):
            input_ids.fill_(7)  # teacher-forced with a fixed token on both sides
            position_ids.fill_(position)
            attention_mask[0, 0, 0].copy_(slots <= position)
            out = binding.infer(
                {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids, **cache_feed}
            )
            torch.testing.assert_close(
                out["logits"].cpu(), eager(torch.tensor([[7]]), torch.tensor([position])), atol=1e-3, rtol=1e-3
            )
            self.assertEqual(int(cache[counter_name].cpu().item()), position + 1)


@slow
@require_torch
class ExportedModelTest(unittest.TestCase):
    """A single-graph export runs like its source model on every backend: built from the export, and loaded from disk
    (`ExportedModel.from_pretrained`)."""

    def _model_and_inputs(self):
        from transformers import BertConfig, BertModel

        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=4, intermediate_size=37
        )
        inputs = {"input_ids": torch.randint(0, 99, (2, 7)), "attention_mask": torch.ones(2, 7, dtype=torch.long)}
        return BertModel(config).eval(), inputs

    def _check_runs_like_eager(self, exporter, config, atol, device=None):
        from transformers.exporters import ExportedModel

        model, inputs = self._model_and_inputs()
        with torch.no_grad():
            expected = model(**inputs)
        exported = exporter.export(model, copy.deepcopy(inputs), config)
        with tempfile.TemporaryDirectory() as directory:
            exported.save_pretrained(directory)
            for name, runtime in (
                ("fresh", exported.to(device)),
                ("from_pretrained", ExportedModel.from_pretrained(directory, device=device)),
            ):
                with self.subTest(name):
                    # Inputs the trace never saw are dropped, as when passing a processor's whole output.
                    outputs = runtime(**inputs, token_type_ids=torch.zeros(2, 7, dtype=torch.long))
                    self.assertIs(type(outputs), type(expected))
                    for key in expected:
                        torch.testing.assert_close(outputs[key].cpu(), expected[key], atol=atol, rtol=atol)
                    self.assertEqual(runtime.dtype, torch.float32)
                    # Copied by default: the next call leaves them as they are.
                    runtime(**{**inputs, "input_ids": inputs["input_ids"].flip(-1)})
                    torch.testing.assert_close(
                        outputs["last_hidden_state"].cpu(), expected["last_hidden_state"], atol=atol, rtol=atol
                    )
                    # Shared outputs are right until the next call.
                    shared = runtime(**inputs, share_inputs=False, share_outputs=True)
                    torch.testing.assert_close(
                        shared["last_hidden_state"].cpu(), expected["last_hidden_state"], atol=atol, rtol=atol
                    )
                    with tempfile.TemporaryDirectory() as resaved:
                        runtime.save_pretrained(resaved)
                        self.assertEqual(sorted(os.listdir(resaved)), sorted(os.listdir(directory)))

    @pytest.mark.torch_export_test
    def test_dynamo(self):
        from transformers.exporters import DynamoConfig, DynamoExporter

        self._check_runs_like_eager(DynamoExporter(), DynamoConfig(dynamic=True), atol=1e-4)

    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    def test_onnx(self):
        from transformers.exporters import OnnxConfig, OnnxExporter

        self._check_runs_like_eager(OnnxExporter(), OnnxConfig(dynamic=True), atol=1e-3, device="cpu")

    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    def test_onnx_runner_options(self):
        import onnxruntime as ort

        from transformers.exporters import ExportedModel, OnnxConfig, OnnxExporter

        model, inputs = self._model_and_inputs()
        with torch.no_grad():
            expected = model(**inputs).last_hidden_state
        exported = OnnxExporter().export(model, copy.deepcopy(inputs), OnnxConfig(dynamic=True))
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        provider = "CUDAExecutionProvider" if torch_device == "cuda" else "CPUExecutionProvider"
        providers = [(provider, {"arena_extend_strategy": "kSameAsRequested"})]
        with tempfile.TemporaryDirectory() as directory:
            exported.save_pretrained(directory)
            loaded = ExportedModel.from_pretrained(directory, providers=providers, session_options=session_options)
            torch.testing.assert_close(loaded(**inputs).last_hidden_state.cpu(), expected, atol=1e-3, rtol=1e-3)
            session = next(iter(loaded.runners.values())).session
            self.assertEqual(session.get_providers()[0], provider)
            self.assertEqual(session.get_session_options().intra_op_num_threads, 1)
            self.assertEqual(
                session.get_session_options().graph_optimization_level, ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            with self.assertRaisesRegex(TypeError, "revison"):
                ExportedModel.from_pretrained(directory, revison="main")
            with self.assertRaisesRegex(ValueError, "either `device` or `providers`"):
                ExportedModel.from_pretrained(directory, device="cpu", providers=providers)

    @require_onnxscript
    @require_onnxruntime
    @require_torch_gpu
    @pytest.mark.onnx_export_test
    def test_onnx_cuda_graph(self):
        """A static export run with `enable_cuda_graph` replays a CUDA graph: each call's inputs are copied into the
        buffers it was captured with, and its outputs copied out, so an earlier call's outputs survive the next."""
        from transformers.exporters import OnnxConfig, OnnxExporter

        model = _tiny_model().to("cuda")
        calls = [
            {
                "input_ids": torch.randint(0, 64, (2, 8), device="cuda"),
                "attention_mask": torch.ones(2, 8, dtype=torch.long, device="cuda"),
            }
            for _ in range(3)
        ]
        exported = OnnxExporter().export(model, dict(calls[0]), OnnxConfig())
        exported.to(providers=[("CUDAExecutionProvider", {"device_id": 0, "enable_cuda_graph": "1"})])
        runner = exported.runners["model"]
        self.assertTrue(runner.replays_cuda_graph)
        with torch.no_grad():
            outputs = [exported(**inputs).logits for inputs in calls]
            for output, inputs in zip(outputs, calls):
                torch.testing.assert_close(output, model(**inputs).logits, atol=1e-3, rtol=1e-3)
        self.assertIsNotNone(runner.cuda_graph)
        with self.assertRaisesRegex(ValueError, "captured with"):
            exported(**{name: tensor[:1] for name, tensor in calls[0].items()})

    @require_openvino
    @pytest.mark.openvino_export_test
    def test_openvino(self):
        from transformers.exporters import OpenVINOConfig, OpenVINOExporter

        self._check_runs_like_eager(OpenVINOExporter(), OpenVINOConfig(dynamic=True), atol=1e-3, device="cpu")

    @require_openvino
    @pytest.mark.openvino_export_test
    def test_openvino_bfloat16_view_buffers(self):
        """A bf16 model whose buffers are views exports and runs: a loaded checkpoint holds such views (EmbeddingGemma2's
        0-d clipping bounds, each viewing a 1-element tensor), which the trace's f32 upcast must not replay."""
        from transformers.exporters import OpenVINOConfig, OpenVINOExporter

        model = _tiny_model().to(torch.bfloat16)
        base = torch.ones(1, dtype=torch.bfloat16)
        model.model.norm.register_buffer("input_min", base.squeeze())
        inputs = {"input_ids": torch.randint(0, 64, (1, 6)), "attention_mask": torch.ones(1, 6, dtype=torch.long)}
        with torch.no_grad():
            expected = model(**inputs).logits
        exported = OpenVINOExporter().export(model, copy.deepcopy(inputs), OpenVINOConfig())
        self.assertIs(model.model.norm.input_min._base, base)
        # The graph takes and returns float32, the trace's, and runs in the model's own precision.
        self.assertEqual(exported.components["model"].metadata.export_dtype, "float32")
        for option in ("INFERENCE_PRECISION_HINT", "KV_CACHE_PRECISION"):
            self.assertEqual(exported.program.get_rt_info(["runtime_options", option]).astype(str), "bf16")
        # OpenVINO hands a bf16 model's outputs back in fp32.
        torch.testing.assert_close(exported.to("cpu")(**inputs).logits, expected.float(), atol=1e-2, rtol=1e-2)

    @require_executorch
    @pytest.mark.executorch_export_test
    def test_executorch(self):
        from transformers.exporters import ExecutorchConfig, ExecutorchExporter

        self._check_runs_like_eager(ExecutorchExporter(), ExecutorchConfig(), atol=1e-3)

    def _check_loads_cache_output(self, exporter, config):
        """An export whose output holds a cache loads in a process that never exported, where the cache isn't a
        pytree node yet."""
        from transformers import DynamicCache, LlamaConfig, LlamaModel
        from transformers.exporters import ExportedModel

        llama_config = LlamaConfig(
            vocab_size=99,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=37,
        )
        model = LlamaModel(llama_config).eval()
        inputs = {"input_ids": torch.randint(0, 99, (1, 7))}
        exported = exporter.export(model, inputs, config)
        with tempfile.TemporaryDirectory() as directory:
            exported.save_pretrained(directory)
            torch.utils._pytree._deregister_pytree_node(DynamicCache)
            outputs = ExportedModel.from_pretrained(directory, device="cpu")(**inputs)
        self.assertIsInstance(outputs.cache, DynamicCache)

    @pytest.mark.torch_export_test
    def test_dynamo_loads_cache_output(self):
        from transformers.exporters import DynamoConfig, DynamoExporter

        self._check_loads_cache_output(DynamoExporter(), DynamoConfig())

    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    def test_onnx_loads_cache_output(self):
        from transformers.exporters import OnnxConfig, OnnxExporter

        self._check_loads_cache_output(OnnxExporter(), OnnxConfig())

    def _check_runs_non_tensor_outputs(self, exporter, config):
        """Outputs that are no graph outputs come back as the model returned them, also once saved and loaded: the
        `None`s of hidden states asked for by layer index (`output_hidden_states=[2]`), MaskFormer-Swin's spatial
        sizes."""
        from transformers.exporters import ExportedModel
        from transformers.modeling_outputs import BaseModelOutput

        class NonTensorOutputs(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.full((1,), 2.0))

            def forward(self, hidden_states):
                hidden_states = hidden_states * self.scale
                return BaseModelOutput(
                    last_hidden_state=hidden_states, hidden_states=(None, (4, 5), hidden_states + 1)
                )

        model, inputs = NonTensorOutputs().eval(), {"hidden_states": torch.randn(4, 5)}
        with torch.no_grad():
            expected = model(**inputs)
        exported = exporter.export(model, dict(inputs), config)
        with tempfile.TemporaryDirectory() as directory:
            exported.save_pretrained(directory)
            for outputs in (
                exported.to("cpu")(**inputs),
                ExportedModel.from_pretrained(directory, device="cpu")(**inputs),
            ):
                self.assertEqual(outputs.hidden_states[:2], (None, (4, 5)))
                torch.testing.assert_close(outputs.hidden_states[2], expected.hidden_states[2])
                torch.testing.assert_close(outputs.last_hidden_state, expected.last_hidden_state)

    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    def test_onnx_non_tensor_outputs(self):
        from transformers.exporters import OnnxConfig, OnnxExporter

        self._check_runs_non_tensor_outputs(OnnxExporter(), OnnxConfig())

    @require_openvino
    @pytest.mark.openvino_export_test
    def test_openvino_non_tensor_outputs(self):
        from transformers.exporters import OpenVINOConfig, OpenVINOExporter

        self._check_runs_non_tensor_outputs(OpenVINOExporter(), OpenVINOConfig())

    def test_unsaveable_outputs_refused(self):
        from transformers.exporters.metadata import ExportMetadata, record_output_structure

        with self.assertRaisesRegex(ValueError, "can't save"):
            record_output_structure(ExportMetadata(), {"logits": torch.ones(1), "device": torch.device("cpu")})

    @require_onnxscript
    @require_onnxruntime
    @pytest.mark.onnx_export_test
    def test_onnx_bool_and_passthrough_outputs(self):
        """A bool output comes back as bool, and an output returning an input as is leaves the input its name, also
        without the optimizer (ProphetNet's encoder states)."""
        from transformers.exporters import OnnxConfig, OnnxExporter

        class BoolAndPassthrough(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.full((1,), 2.0))

            def forward(self, hidden_states, encoder_states):
                return {
                    "logits": hidden_states * self.scale,
                    "mask": hidden_states > 0,
                    "encoder_last_hidden_state": encoder_states,
                }

        model = BoolAndPassthrough().eval()
        inputs = {"hidden_states": torch.randn(2, 3), "encoder_states": torch.randn(2, 4)}
        exported = OnnxExporter().export(model, dict(inputs), OnnxConfig(optimize=False))
        self.assertEqual([value.name for value in exported.program.graph.inputs], list(inputs))
        outputs = exported.to("cpu")(**inputs)
        self.assertEqual(outputs["mask"].dtype, torch.bool)
        torch.testing.assert_close(outputs["mask"], inputs["hidden_states"] > 0)
        torch.testing.assert_close(outputs["encoder_last_hidden_state"], inputs["encoder_states"])

    def test_call_refuses_several_components(self):
        from transformers.exporters import ExportedComponent, ExportedModel, ExportMetadata

        component = ExportedComponent("program", ExportMetadata())
        exported = ExportedModel({"prefill": component, "decode": component}, "dynamo")
        with self.assertRaisesRegex(ValueError, "2 components"):
            exported(input_ids=None)
