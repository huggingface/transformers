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
"""Distributed loading, sharding, and compiled static-cache decode checks."""

import os
import re
import tempfile
from unittest import mock

import torch

from transformers import CompileConfig
from transformers.testing_utils import (
    cleanup,
    require_kernels,
    require_torch_multi_accelerator,
    set_model_for_less_flaky_test,
    slow,
    torch_device,
)
from transformers.utils.quantization_config import Fp8Config, Mxfp4Config

from ..test_tensor_parallel_mixin import _init_distributed
from .block_scaled_checkpoint_tester import _checkpoint_bytes
from .block_scaled_tester import FormatSpec, _cheap_tuning, _require_kernel_capability


def _check_static_cache_decode(model, ids):
    from torch._dynamo.backends.registry import lookup_backend

    generation_kwargs = {
        "input_ids": ids.to(model.device),
        "attention_mask": torch.ones_like(ids, device=model.device),
        "max_new_tokens": 3,
        "do_sample": False,
        "eos_token_id": None,
        "pad_token_id": 0,
        "cache_implementation": "static",
        "output_logits": True,
        "return_dict_in_generate": True,
    }
    # Match the generation compile tests' norm stabilization for tiny random models.
    set_model_for_less_flaky_test(model)
    eager = model.generate(**generation_kwargs, disable_compile=True)
    graphs = []

    def compile_backend(graph, inputs, **kwargs):
        graphs.append(graph)
        return lookup_backend("inductor")(graph, inputs, **kwargs)

    compiled = model.generate(
        **generation_kwargs,
        compile_config=CompileConfig(backend=compile_backend, fullgraph=True, dynamic=False, mode="default"),
    )
    assert graphs, "Static-cache generation did not compile the decode forward"
    torch.testing.assert_close(compiled.sequences, eager.sequences)
    for actual, expected in zip(compiled.logits, eager.logits):
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def _check_sharded_load_paths(rank, model_dir, model_cls, modes, out_dir, compile_decode):
    from transformers.distributed import DistributedConfig

    world = torch.distributed.get_world_size()
    ids = torch.arange(16, dtype=torch.long).unsqueeze(0)
    for mode in modes:
        model = model_cls.from_pretrained(
            model_dir,
            dtype="auto",
            attn_implementation="eager",
            distributed_config=DistributedConfig(tp_size=world, ep_size=world if mode == "ep" else None),
        ).eval()
        experts = next(m for n, m in model.named_modules() if n.endswith("mlp.experts"))
        weight = getattr(experts, experts.gate_up_name)
        local = weight.to_local() if hasattr(weight, "to_local") else weight
        payload = {"expert_local": tuple(local.shape)}
        with torch.no_grad():
            if compile_decode:
                _check_static_cache_decode(model, ids)
            else:
                payload["logits"] = model(ids.to(model.device)).logits.float().cpu()
        if rank == 0:
            torch.save(payload, os.path.join(out_dir, mode + ".pt"))
        del model
        torch.accelerator.empty_cache()


def _checkpoint_expert_shape(model_dir):
    """Return the unsharded `(experts, rows)` of the stacked gate/up projection, read from the checkpoint.

    Checkpoints ship experts fused as `experts.gate_up_proj`, or one per expert as `experts.0.gate_proj.weight`.
    DeepSeek names the per-expert projections `w1` and `w3`.
    """
    import glob
    import json

    from safetensors import safe_open

    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as fh:
            weight_map = json.load(fh)["weight_map"]
        shards = sorted({os.path.join(model_dir, f) for f in weight_map.values()})

    experts: set[int] = set()
    rows = 0
    for shard in shards:
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                shape = None
                if re.search(r"\.experts\.gate_up_proj$", key):
                    shape = handle.get_slice(key).get_shape()
                    if len(shape) == 3:
                        return (shape[0], shape[1])
                if re.search(r"\.experts\.(\d+)\.(gate_proj|up_proj|w1|w3)\.weight$", key):
                    experts.add(int(key.rsplit(".experts.", 1)[1].split(".", 1)[0]))
                    rows = max(rows, handle.get_slice(key).get_shape()[0])
    if not experts or not rows:
        raise AssertionError(f"no expert projection in {model_dir} to size the shard against")
    return (len(experts), 2 * rows)  # the module stacks gate and up along the rows


class LoadPathEquivalenceTesterMixin:
    """Check that expert parallelism and intra-expert tensor parallelism match the `device_map` baseline.

    `device_map` places modules across devices without splitting a tensor. It needs no process group, DTensor or
    collective, so it gives the unsharded answer.

    Each model is built in the format it ships, since the conversion path differs by format and checkpoint layout.

    Three properties let this test fail:
    - The fixture is pre-quantized. On-the-fly quantization derives scales from each rank's own shard, so a plan
      that shards no scale would still give the right answer.
    - The model's plan must name experts. A model with an empty `base_model_tp_plan` shards nothing under TP, so
      that mode is not run.
    - Every mode reports its local expert shard, so a mode that placed nothing fails.

    A format's test file sets `spec`. The fixtures are its `distributed_models`.
    """

    spec: FormatSpec

    @staticmethod
    def _model_table():
        """Return `{label: (config_cls, model_cls, config_kwargs, quantization_config)}`.

        Each model uses the quantization config it ships with. Dims are multiples of the 128 block, so a 2-way split
        stays block-aligned. The expert count divides the mesh.
        """
        from transformers import (
            DeepseekV3Config,
            DeepseekV3ForCausalLM,
            DeepseekV4Config,
            DeepseekV4ForCausalLM,
            GptOssConfig,
            GptOssForCausalLM,
            Mistral4Config,
            Mistral4ForCausalLM,
        )

        return {
            "deepseek_v3-fp8": (
                DeepseekV3Config,
                DeepseekV3ForCausalLM,
                {
                    "vocab_size": 64,
                    "hidden_size": 256,
                    "intermediate_size": 256,
                    "moe_intermediate_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                    "n_shared_experts": 1,
                    "n_group": 1,
                    "topk_group": 1,
                    "first_k_dense_replace": 0,
                    "max_position_embeddings": 32,
                    "q_lora_rank": None,
                    "kv_lora_rank": 32,
                    "qk_nope_head_dim": 32,
                    "qk_rope_head_dim": 16,
                    "v_head_dim": 32,
                },
                Fp8Config(weight_block_size=(128, 128)),
            ),
            # The only shipped static scheme. Weights are per-tensor, with one calibrated activation scale per module.
            # A MoE gets one activation scale per expert. Ministral-3 is the dense version of this export.
            "mistral4-fp8_tensor_static": (
                Mistral4Config,
                Mistral4ForCausalLM,
                {
                    "vocab_size": 64,
                    "hidden_size": 256,
                    "intermediate_size": 256,
                    "moe_intermediate_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                    "first_k_dense_replace": 0,
                    "max_position_embeddings": 32,
                },
                Fp8Config(weight_block_size=None, activation_scheme="static"),
            ),
            # Interleaved, transposed rows (`is_concatenated=False`) with expert biases that the EP plan names
            "gpt_oss-mxfp4": (
                GptOssConfig,
                GptOssForCausalLM,
                {
                    "vocab_size": 64,
                    "hidden_size": 256,
                    "intermediate_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "num_local_experts": 4,
                    "num_experts_per_tok": 2,
                    "max_position_embeddings": 32,
                },
                Mxfp4Config(activation_format="bf16"),
            ),
            # Mixed. `expert_dtype` gives MXFP4 experts next to block-FP8 layers with UE8M0 scales.
            "deepseek_v4-fp4_experts+fp8_dense": (
                DeepseekV4Config,
                DeepseekV4ForCausalLM,
                {
                    "vocab_size": 64,
                    "hidden_size": 256,
                    "intermediate_size": 256,
                    "moe_intermediate_size": 256,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "n_routed_experts": 4,
                    "num_experts_per_tok": 2,
                    "n_shared_experts": 1,
                    "first_k_dense_replace": 0,
                    "max_position_embeddings": 32,
                    "expert_dtype": "fp4",
                },
                Fp8Config(weight_block_size=(128, 128), scale_fmt="ue8m0"),
            ),
        }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmp = cls.enterClassContext(tempfile.TemporaryDirectory())
        cls._fixtures = {}
        cls.enterClassContext(_cheap_tuning())
        cls.enterClassContext(
            mock.patch.dict(
                os.environ, {"TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR", os.path.join(cls._tmp, "triton"))}
            )
        )

    def setUp(self):
        super().setUp()
        # Below 8.9 a pre-quantized fixture dequantizes on load, so the modes would compare bf16 models
        _require_kernel_capability(self)

    def tearDown(self):
        super().tearDown()
        cleanup(torch_device, gc_collect=True)

    def _fixture(self, label):
        """Return a pre-quantized checkpoint of one tiny model, in the format that model ships."""
        if label in self._fixtures:
            return self._fixtures[label]
        config_cls, model_cls, kwargs, quantization_config = self._model_table()[label]
        config = config_cls(**kwargs)

        bf16_dir = os.path.join(self._tmp, label, "bf16")
        quant_dir = os.path.join(self._tmp, label, "quantized")
        torch.manual_seed(0)
        # Every model here ships in bf16, so save in bf16 and let `dtype="auto"` reload it
        model_cls(config).to(torch.bfloat16).save_pretrained(bf16_dir, safe_serialization=True)
        quantized = model_cls.from_pretrained(
            bf16_dir,
            dtype="auto",
            attn_implementation="eager",
            quantization_config=quantization_config,
            device_map=f"{torch_device}:0",
        )
        quantized.save_pretrained(quant_dir, safe_serialization=True)

        # A reload saved again must give the same bytes. Each layout op's reverse restores the checkpoint.
        reloaded = model_cls.from_pretrained(quant_dir, dtype="auto", device_map=f"{torch_device}:0")
        round_trip = os.path.join(self._tmp, label, "round_trip")
        reloaded.save_pretrained(round_trip, safe_serialization=True)
        saved, resaved = _checkpoint_bytes(quant_dir), _checkpoint_bytes(round_trip)
        mismatched = sorted(key for key in saved.keys() | resaved.keys() if saved.get(key) != resaved.get(key))
        self.assertEqual(mismatched, [], f"{label}: save did not round-trip")
        self._fixtures[label] = quant_dir, config, model_cls
        return self._fixtures[label]

    def _logits(self, model_dir, model_cls, **load_kwargs):
        model = model_cls.from_pretrained(model_dir, dtype="auto", attn_implementation="eager", **load_kwargs).eval()
        ids = torch.arange(16, dtype=torch.long, device=model.device).unsqueeze(0)
        with torch.no_grad():
            logits = model(ids).logits.float().cpu()
        return logits

    def _rounding_floor(self, model_dir, model_cls, reference):
        """Return how far the logits move when every sharded block gets BF16-rounding noise.

        Sharding cannot be bit-exact, because a rowwise all-reduce sums in a different order. How far that error
        travels depends on the model, and a chain of MoE layers can amplify it a lot. Injecting the same noise on one
        device gives each model a budget that matches its own conditioning.
        """
        model = model_cls.from_pretrained(
            model_dir, dtype="auto", attn_implementation="eager", device_map=f"{torch_device}:0"
        ).eval()
        # Perturb every block a sharded reduction passes through. Rounding accumulates down the stack.
        blocks = [
            m
            for n, m in model.named_modules()
            if n.endswith((".self_attn", ".mlp", ".block_sparse_moe")) and n.count(".layers.") == 1
        ]
        generator = torch.Generator(device=model.device).manual_seed(0)

        def perturb(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor):
                return output
            ulp = torch.finfo(tensor.dtype).eps * tensor.abs().max()
            noise = torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype) * ulp
            tensor = tensor + noise
            return (tensor,) + output[1:] if isinstance(output, tuple) else tensor

        handles = [b.register_forward_hook(perturb) for b in blocks]
        ids = torch.arange(16, dtype=torch.long, device=model.device).unsqueeze(0)
        with torch.no_grad():
            perturbed = model(ids).logits.float().cpu()
        for handle in handles:
            handle.remove()
        return (perturbed - reference).abs().max().item()

    def _sharded(self, model_dir, model_cls, modes, compile_decode=False):
        """Return `{mode: payload}` from two ranks, through the real EP and TP load paths."""
        out = os.path.join(self._tmp, "out")
        os.makedirs(out, exist_ok=True)
        _init_distributed(tp=2)(_check_sharded_load_paths)(model_dir, model_cls, modes, out, compile_decode)
        return {m: torch.load(os.path.join(out, m + ".pt")) for m in modes}

    @slow
    @require_kernels
    @require_torch_multi_accelerator
    def test_every_load_path_agrees(self):
        for label in self.spec.distributed_models:
            with self.subTest(model=label):
                model_dir, config, model_cls = self._fixture(label)
                # `max_memory` spreads the baseline across both devices. `auto` alone fits it on one.
                reference = self._logits(model_dir, model_cls, device_map="auto", max_memory={0: "120MiB", 1: "40GiB"})
                noise_floor = self._rounding_floor(model_dir, model_cls, reference)

                modes = self._parallel_modes(config)
                self.assertTrue(modes, f"{label}: no plan shards experts")
                whole = _checkpoint_expert_shape(model_dir)
                for mode, payload in self._sharded(model_dir, model_cls, modes).items():
                    with self.subTest(model=label, mode=mode):
                        expected = (whole[0] // 2, whole[1]) if mode == "ep" else (whole[0], whole[1] // 2)
                        self.assertEqual(payload["expert_local"][:2], expected, f"{label}/{mode}: wrong shard axis")
                        # A wrong shard axis lands far above the model's rounding floor
                        budget = max(2e-2, noise_floor)
                        gap = (payload["logits"] - reference).abs().max().item()
                        self.assertLessEqual(
                            gap,
                            budget,
                            f"{label}/{mode}: sharded logits differ by {gap:.4g}, beyond this "
                            f"model's own rounding budget {budget:.4g} — that is a sharding bug, "
                            f"not reduction order",
                        )
                        if noise_floor <= 2e-2:  # rounding alone does not move the argmax
                            self.assertTrue(
                                torch.equal(payload["logits"].argmax(-1), reference.argmax(-1)),
                                f"{label}/{mode}: argmax diverged from the unsharded model",
                            )

    @staticmethod
    def _parallel_modes(config):
        owners = [config] + [
            c for name in getattr(type(config), "sub_configs", {}) if (c := getattr(config, name, None))
        ]
        return [
            mode
            for mode, attr in (("ep", "base_model_ep_plan"), ("tp", "base_model_tp_plan"))
            if any(".experts." in key for owner in owners for key in (getattr(owner, attr, None) or {}))
        ]

    @slow
    @require_kernels
    @require_torch_multi_accelerator
    def test_static_cache_decode_compiles(self):
        supported = [
            label for label in self.spec.distributed_models if self._model_table()[label][1]._can_compile_fullgraph
        ]
        if not supported:
            self.skipTest("These model architectures do not support fullgraph compilation")
        for label in supported:
            with self.subTest(model=label):
                model_dir, config, model_cls = self._fixture(label)
                modes = self._parallel_modes(config)
                self.assertTrue(modes, f"{label}: no plan shards experts")
                self._sharded(model_dir, model_cls, modes, compile_decode=True)
