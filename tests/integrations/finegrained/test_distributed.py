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
"""The sharded load paths: expert parallelism and intra-expert tensor parallelism must both reproduce
what `device_map` computes, for every format a shipped model quantizes with, and a checkpoint quantized
on the fly must save and reload to the same bytes."""

import os
import re
import socket
import subprocess
import tempfile

import torch

from transformers.testing_utils import (
    TestCasePlus,
    cleanup,
    require_torch_multi_accelerator,
    torch_device,
)

from .test_models import _checkpoint_bytes


_SHARDING_WORKER = """
import importlib, os, sys, torch
from transformers.distributed import DistributedConfig

model_dir, out_dir, modes, cls_path = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
module_name, cls_name = cls_path.split(":")
model_cls = getattr(importlib.import_module(module_name), cls_name)
world = int(os.environ["WORLD_SIZE"])
ids = torch.arange(16, dtype=torch.long).unsqueeze(0)

# both modes in ONE launch: they share the mesh, so the process group is created once and only
# the plan differs -- a second torchrun would pay another interpreter + CUDA start
for mode in modes:
    model = model_cls.from_pretrained(
        model_dir,
        dtype="auto",
        attn_implementation="eager",
        distributed_config=DistributedConfig(tp_size=world, enable_expert_parallel=(mode == "ep")),
    ).eval()
    experts = next(m for n, m in model.named_modules() if n.endswith("mlp.experts"))
    weight = getattr(experts, "gate_up_proj", None)
    if weight is None:
        weight = experts.up_proj
    local = weight.to_local() if hasattr(weight, "to_local") else weight
    with torch.no_grad():
        logits = model(ids.to(model.device)).logits.float().cpu()
    if int(os.environ["RANK"]) == 0:
        torch.save({"logits": logits, "expert_local": tuple(local.shape)},
                   os.path.join(out_dir, mode + ".pt"))
    del model
    torch.accelerator.empty_cache()
"""


def _checkpoint_expert_shape(model_dir):
    """The UNSHARDED `(experts, rows)` of the stacked gate|up projection the module holds.

    Read from the checkpoint, which ships the experts either already fused (`experts.gate_up_proj`)
    or one per expert (`experts.0.gate_proj.weight`, or DeepSeek's `w1`/`w3`, whose rows are gate and
    up separately). Taking the baseline from whichever leg ran first would compare one mode to
    another, and a mode that placed nothing would still look sharded.
    """
    import glob
    import json
    import os

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
    return (len(experts), 2 * rows)  # the module stacks gate and up into one row extent


@require_torch_multi_accelerator
class FineGrainedLoadPathEquivalenceTest(TestCasePlus):
    """Expert parallelism and intra-expert tensor parallelism must both reproduce what
    `device_map` gives, which is the baseline because it places modules across devices without
    splitting a single tensor — no process group, no DTensor, no collectives. So it computes the
    unsharded answer, on the multi-GPU path people actually deploy.

    Each model is built in the format it actually ships, because the conversion path differs by
    format and by checkpoint layout — a per-expert FP8 checkpoint and a fused NVFP4 one reach the
    experts through different converters, and bugs have hidden in exactly that gap.

    Three properties make this able to fail:
      * the fixture is PRE-QUANTIZED. Quantizing on the fly derives each rank's scales from the
        shard it already holds, so they come out correctly sized whatever the plan says and a
        plan that shards no scale at all still produces the right answer.
      * the model's plan must carry expert entries. A model whose `base_model_tp_plan` is empty
        (GPT-OSS, DeepSeek-V4) shards nothing under TP, so that leg is skipped EXPLICITLY rather
        than passing vacuously.
      * every leg reports the local expert shard, so a leg that placed nothing fails loudly.
    """

    @staticmethod
    def _model_table():
        """`{label: (config_cls, model_cls, config_kwargs, quantization_config)}` — each model
        in the format it actually ships, with the quantization config that format arrives under,
        not just its name: block-FP8 carries a `weight_block_size`, and an NVFP4 checkpoint comes
        from modelopt under `quant_algo`, which is remapped on construction. Dims are multiples
        of the 128 block so a 2-way split stays block-aligned, and the expert count divides the
        mesh."""
        from transformers import (
            DeepseekV3Config,
            DeepseekV3ForCausalLM,
            DeepseekV4Config,
            DeepseekV4ForCausalLM,
            Glm4vMoeConfig,
            Glm4vMoeForConditionalGeneration,
            Glm4vMoeTextConfig,
            Glm4vMoeVisionConfig,
            GptOssConfig,
            GptOssForCausalLM,
            MiniMaxM3SparseForConditionalGeneration,
            MiniMaxM3VLConfig,
            Mistral4Config,
            Mistral4ForCausalLM,
        )
        from transformers.utils.quantization_config import FineGrainedConfig

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
                # DeepSeek-V3 ships block-FP8: 128x128 weight blocks, activations quantized
                # per token at run time
                FineGrainedConfig(quant_method="fp8", weight_block_size=(128, 128)),
            ),
            # the only shipped STATIC scheme: per-TENSOR weights (no `weight_block_size`) and a
            # calibrated activation scale per quantized module, which for a MoE is one per expert.
            # Its conversion script asserts `qscheme_act == "TENSOR"`; Ministral-3 is the dense
            # counterpart of the same export.
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
                FineGrainedConfig(quant_method="fp8", weight_block_size=None, activation_scheme="static"),
            ),
            # MULTIMODAL: the experts' plans live on `text_config`, not on the config the
            # quantizer is handed — whose `base_model_ep_plan` is None. Reading only the outer
            # one adds no companion while the weights still shard, which is how a real
            # multimodal MoE ends up with whole scales against sharded weights.
            "glm4v_moe-nvfp4": (
                Glm4vMoeConfig,
                Glm4vMoeForConditionalGeneration,
                {
                    "text_config": Glm4vMoeTextConfig(
                        vocab_size=64,
                        hidden_size=256,
                        intermediate_size=256,
                        moe_intermediate_size=256,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        max_position_embeddings=32,
                        rope_parameters={"type": "default", "mrope_section": [16, 8, 8], "partial_rotary_factor": 1.0},
                        rope_theta=10000,
                        tie_word_embeddings=True,
                        bos_token_id=0,
                        eos_token_id=0,
                        pad_token_id=0,
                        n_routed_experts=4,
                        n_shared_experts=1,
                        n_group=1,
                        topk_group=1,
                        num_experts_per_tok=2,
                        first_k_dense_replace=0,
                    ),
                    "vision_config": Glm4vMoeVisionConfig(
                        depth=2,
                        num_heads=4,
                        hidden_size=64,
                        out_hidden_size=256,
                        intermediate_size=64,
                        patch_size=14,
                        spatial_merge_size=1,
                        temporal_patch_size=2,
                    ),
                },
                # the GLM NVFP4 checkpoints are modelopt exports: `quant_algo` names the format
                # and `FineGrainedConfig` remaps it to nvfp4 at construction
                FineGrainedConfig(quant_method="modelopt", quant_algo="NVFP4"),
            ),
            # interleaved rows (`is_concatenated=False`), transposed, with expert biases — and
            # an EP plan that already names those biases, so the companion rules meet entries
            # the model wrote itself
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
                # GPT-OSS ships weight-only: raw bf16 activations against packed fp4 weights
                FineGrainedConfig(quant_method="mxfp4", activation_format="bf16"),
            ),
            # MIXED precision, and the only entry whose expert format is not the quantization
            # config's: `expert_dtype` is a model-config side-channel that makes the EXPERTS
            # mxfp4 while the dense and attention paths stay block-FP8 — with scales in UE8M0
            # containers rather than fp32, the other `scale_fmt`
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
                FineGrainedConfig(quant_method="fp8", weight_block_size=(128, 128), scale_fmt="ue8m0"),
            ),
            # a second MULTIMODAL nesting, in the group-32 MX format. The sub-config shape
            # follows this model's own tester; only the MoE dims are raised to a multiple of
            # the 128 block so a 2-way split stays block-aligned.
            "minimax_m3_vl-mxfp8": (
                MiniMaxM3VLConfig,
                MiniMaxM3SparseForConditionalGeneration,
                {
                    "text_config": {
                        "hidden_size": 256,
                        # 512 so the 2-way split leaves 256: the experts' down projection
                        # contracts over this, and a sharded 128 has no 128-wide swizzled tile
                        "intermediate_size": 512,
                        "dense_intermediate_size": 256,
                        "shared_intermediate_size": 256,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 4,
                        "head_dim": 64,
                        "rotary_dim": 32,
                        "vocab_size": 64,
                        "max_position_embeddings": 32,
                        "bos_token_id": 0,
                        "eos_token_id": 1,
                        "pad_token_id": 2,
                        "num_local_experts": 4,
                        "num_experts_per_tok": 2,
                        "n_shared_experts": 1,
                        "moe_layer_freq": [0, 1],
                        "layer_types": ["full_attention", "minimax_m3_sparse"],
                        "tie_word_embeddings": False,
                        "index_n_heads": 2,
                        "index_head_dim": 16,
                        "index_block_size": 8,
                        "index_topk_blocks": 4,
                        "index_local_blocks": 1,
                    },
                    "vision_config": {
                        # 256 so the 2-way SPLIT is still 128-aligned: an MXFP8 weight with
                        # pre-swizzled scales is read in 128-wide K tiles, and a sharded 128 dim
                        # leaves 64 — which has none to offer, and the tuner has no config at all
                        "hidden_size": 256,
                        "intermediate_size": 256,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 4,
                        "num_channels": 3,
                        "image_size": 14,
                        "patch_size": 14,
                        "temporal_patch_size": 2,
                        "spatial_merge_size": 1,
                    },
                    "image_token_index": 4,
                    "video_token_index": 5,
                    "projector_hidden_size": 256,
                    "pad_token_id": 2,
                },
                FineGrainedConfig(quant_method="mxfp8"),
            ),
        }

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        # the kernels autotune per shape from a cold cache, which dwarfs everything else here
        cls._env = {
            "FINEGRAINED_AUTOTUNE_TRIALS": "1",
            "TRITON_CACHE_DIR": os.path.join(cls._tmp.name, "triton"),
        }
        os.environ.update(cls._env)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def tearDown(self):
        super().tearDown()
        cleanup(torch_device, gc_collect=True)

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def _fixture(self, label):
        """A PRE-QUANTIZED checkpoint of one tiny model, in the format that model ships."""

        config_cls, model_cls, kwargs, quantization_config = self._model_table()[label]
        config = config_cls(**kwargs)

        bf16_dir = os.path.join(self._tmp.name, label, "bf16")
        quant_dir = os.path.join(self._tmp.name, label, "quantized")
        torch.manual_seed(0)
        # BF16 is what every model here ships, and the dtype decides which kernel arms run: a
        # float32 checkpoint (torch's default for a freshly built model, which `dtype="auto"`
        # then faithfully reloads) sends the weight-only formats down arms `tl.dot_scaled`
        # cannot serve at all, so the suite would exercise a dtype nobody deploys and leave the
        # real one uncovered. Saving in bf16 makes `"auto"` mean bf16 for every load below.
        model_cls(config).to(torch.bfloat16).save_pretrained(bf16_dir, safe_serialization=True)
        quantized = model_cls.from_pretrained(
            bf16_dir,
            dtype="auto",
            attn_implementation="eager",
            quantization_config=quantization_config,
            device_map=f"{torch_device}:0",
        )
        quantized.save_pretrained(quant_dir, safe_serialization=True)

        # SAVE must restore the checkpoint's own layout: the layout ops (gate|up interleave,
        # scale container, swizzle) each have a reverse, and a quantized model reloaded and
        # written again has to land on the same bytes. A broken reverse writes a
        # corrupt checkpoint silently — the module still reads back whatever it wrote.
        reloaded = model_cls.from_pretrained(quant_dir, dtype="auto", device_map=f"{torch_device}:0")
        round_trip = os.path.join(self._tmp.name, label, "round_trip")
        reloaded.save_pretrained(round_trip, safe_serialization=True)
        saved, resaved = _checkpoint_bytes(quant_dir), _checkpoint_bytes(round_trip)
        mismatched = sorted(key for key in saved.keys() | resaved.keys() if saved.get(key) != resaved.get(key))
        self.assertEqual(mismatched, [], f"{label}: save did not round-trip")
        return quant_dir, config, model_cls

    def _logits(self, model_dir, model_cls, **load_kwargs):
        model = model_cls.from_pretrained(model_dir, dtype="auto", attn_implementation="eager", **load_kwargs).eval()
        ids = torch.arange(16, dtype=torch.long, device=model.device).unsqueeze(0)
        with torch.no_grad():
            logits = model(ids).logits.float().cpu()
        return logits

    def _rounding_floor(self, model_dir, model_cls, reference):
        """How far this model's logits move when every sharded block is perturbed by BF16 rounding.

        Sharding cannot be bit-exact: a rowwise all-reduce sums the same terms in a different
        order, so the first sharded op differs by an ULP. How far that travels is a property of
        the MODEL, not of the sharding — a chain of MoE layers can amplify it a hundredfold while
        a dense stack barely moves. Injecting the same magnitude on ONE device measures that
        amplification directly, giving each model a budget its own conditioning earns.
        """

        model = model_cls.from_pretrained(
            model_dir, dtype="auto", attn_implementation="eager", device_map=f"{torch_device}:0"
        ).eval()
        # Perturb EVERY block a sharded reduction passes through, not one: TP re-orders the sum
        # in each of them, so the rounding accumulates down the stack instead of cancelling.
        # Perturbing a single site with random noise measured LESS movement than sharding caused
        # even at 14x the magnitude, which is what accumulation looks like from the wrong model.
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

    def _sharded(self, model_dir, model_cls, modes):
        """`{mode: payload}` from one 2-rank `torchrun`, through the real EP / TP load path."""

        script = os.path.join(self._tmp.name, "sharding_worker.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(_SHARDING_WORKER)
        out = os.path.join(self._tmp.name, "out")
        os.makedirs(out, exist_ok=True)
        subprocess.run(
            [
                "torchrun",
                "--nproc_per_node=2",
                f"--master_port={self._free_port()}",
                script,
                model_dir,
                out,
                ",".join(modes),
                # the model's own class: `AutoModelForCausalLM` cannot resolve a multimodal
                # `ForConditionalGeneration` from its config
                f"{model_cls.__module__}:{model_cls.__name__}",
            ],
            check=True,
            env={**os.environ, **self._env, "TOKENIZERS_PARALLELISM": "false"},
        )
        return {m: torch.load(os.path.join(out, m + ".pt")) for m in modes}

    def test_every_load_path_agrees(self):
        for label in self._model_table():
            with self.subTest(model=label):
                model_dir, config, model_cls = self._fixture(label)
                # the baseline, spread across both devices: `max_memory` forces a real split,
                # since `auto` alone fits this whole model on one GPU and would quietly compare
                # the sharded legs against a single-device load
                reference = self._logits(model_dir, model_cls, device_map="auto", max_memory={0: "120MiB", 1: "40GiB"})
                noise_floor = self._rounding_floor(model_dir, model_cls, reference)

                # Only the modes this model's OWN plans shard experts under. A mode whose plan
                # has no expert entry (GPT-OSS and DeepSeek-V4 ship no TP plan at all) places
                # nothing, so running it would pass without being evidence of anything. A
                # multimodal model keeps these on a sub-config.
                owners = [config] + [
                    c for name in getattr(type(config), "sub_configs", {}) if (c := getattr(config, name, None))
                ]
                modes = [
                    mode
                    for mode, attr in (("ep", "base_model_ep_plan"), ("tp", "base_model_tp_plan"))
                    if any(".experts." in key for owner in owners for key in (getattr(owner, attr, None) or {}))
                ]
                self.assertTrue(modes, f"{label}: no plan shards experts, so nothing is under test")
                whole = _checkpoint_expert_shape(model_dir)
                for mode, payload in self._sharded(model_dir, model_cls, modes).items():
                    with self.subTest(model=label, mode=mode):
                        local = payload["expert_local"]
                        # against the CHECKPOINT's own shape, never another mode's shard: seeding
                        # this from the first leg made the second leg check EP against TP, so a
                        # mode that placed nothing still looked sharded (glm4v's TP left its
                        # experts replicated and passed).
                        self.assertLess(
                            local[0] * local[1],
                            whole[0] * whole[1],
                            f"{label}/{mode}: experts were not sharded ({local} of {whole}) — "
                            f"this leg cannot catch a bad axis",
                        )
                        # Sharding reorders reductions, so the legs differ by BF16 rounding
                        # whatever the axes — and how much that shows at the logits is the
                        # model's own business (the NVFP4 fixture amplifies ~100x, the dense
                        # ones barely). Hence each model's measured floor, not a fixed
                        # tolerance. A wrong shard axis lands orders of magnitude above it.
                        budget = max(2e-2, noise_floor)
                        gap = (payload["logits"] - reference).abs().max().item()
                        self.assertLessEqual(
                            gap,
                            budget,
                            f"{label}/{mode}: sharded logits differ by {gap:.4g}, beyond this "
                            f"model's own rounding budget {budget:.4g} — that is a sharding bug, "
                            f"not reduction order",
                        )
                        if noise_floor <= 2e-2:
                            # only meaningful where rounding does NOT already move the argmax
                            self.assertTrue(
                                torch.equal(payload["logits"].argmax(-1), reference.argmax(-1)),
                                f"{label}/{mode}: argmax diverged from the unsharded model",
                            )
