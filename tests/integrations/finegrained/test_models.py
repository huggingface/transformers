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
"""The shipped checkpoint layouts end to end: tiny random copies of the real checkpoints must load
cleanly, compute what their dequantized weights compute and the logits recorded for them, and save back
to their own layout."""

import os
import re

import torch
from parameterized import parameterized

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.testing_utils import (
    Expectations,
    TestCasePlus,
    cleanup,
    require_kernels,
    require_torch_accelerator,
    slow,
    torch_device,
)


def _checkpoint_bytes(path):
    """`{key: (dtype, shape, raw bytes)}` of a saved checkpoint, for comparing one save against another."""
    import glob

    from safetensors import safe_open

    tensors = {}
    for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
                tensors[key] = (tensor.dtype, tuple(tensor.shape), raw)
    return tensors


# Tiny random copies of the shipped checkpoints, shrunk from the real configs and written in their layouts.
# Each repo's `reference/` subfolder holds the same weights dequantized to a plain bf16 model; the budget is
# each format's own rounding against it, measured, with headroom.
TINY_CHECKPOINTS = {
    "IlyasMoutawwakil/tiny-random-Qwen3-FP8": 0.1,
    "IlyasMoutawwakil/tiny-random-Mistral3-FP8-static": 0.1,
    "IlyasMoutawwakil/tiny-random-Mistral4-FP8-static": 0.15,
    "IlyasMoutawwakil/tiny-random-Llama-NVFP4": 0.25,
    "IlyasMoutawwakil/tiny-random-DeepseekV3-FP8": 0.25,
    "IlyasMoutawwakil/tiny-random-DeepseekV4-Flash": 0.25,
    "IlyasMoutawwakil/tiny-random-GptOss-MXFP4": 0.05,
    "IlyasMoutawwakil/tiny-random-GlmMoeDsa-NVFP4": 0.25,
    "IlyasMoutawwakil/tiny-random-GlmMoeDsa-NVFP4-all-linears": 0.6,
    "IlyasMoutawwakil/tiny-random-MiniMaxM3-MXFP8": 0.3,
}


# The logits `[0, :3, :8]` each checkpoint gives on `torch.arange(3, 35)`, per device, as the model integration
# tests record them: bit-stable across tuner caches, so a tight tolerance holds.
EXPECTED_SLICES = {
    "IlyasMoutawwakil/tiny-random-Qwen3-FP8": Expectations(
        {
            ("cuda", (10, 0)): [
                [0.6094, -0.6055, -0.3262, 0.1396, -0.1270, -0.0420, 0.3750, -0.1592],
                [0.1689, -0.4961, 0.1621, 0.0752, -0.1895, 0.0688, 0.3320, -0.0234],
                [0.1069, -0.3027, 0.1050, 0.1553, 0.1699, -0.1572, 0.1699, 0.2598],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-Mistral3-FP8-static": Expectations(
        {
            ("cuda", (10, 0)): [
                [-0.1992, -0.4141, -0.1650, -0.1826, 0.1309, 0.4375, -0.2832, 0.2480],
                [-0.1367, -0.2422, -0.1138, -0.3418, 0.2910, 0.3965, -0.0293, -0.0330],
                [-0.1436, -0.1055, -0.3320, -0.3203, 0.3633, 0.2480, 0.1523, -0.1621],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-Mistral4-FP8-static": Expectations(
        {
            ("cuda", (10, 0)): [
                [-0.0850, 0.1816, -0.2344, 0.5547, -0.0747, 0.6875, -0.2949, 0.1982],
                [-0.0056, -0.0065, 0.0530, 0.0996, 0.1436, 0.2217, -0.3281, 0.1846],
                [0.0055, 0.2559, -0.1045, -0.0031, 0.0608, 0.0449, 0.0962, 0.1416],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-Llama-NVFP4": Expectations(
        {
            ("cuda", (10, 0)): [
                [0.7617, -0.5938, -0.2832, 0.2559, -0.1021, 0.0082, 0.3418, -0.1699],
                [0.6523, -0.7148, 0.2930, 0.3750, -0.5039, 0.0835, 0.2793, -0.0153],
                [0.3457, -0.4785, 0.1924, 0.4102, -0.2715, -0.0835, 0.2070, 0.1602],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-DeepseekV3-FP8": Expectations(
        {
            ("cuda", (10, 0)): [
                [-0.1260, -0.2051, -0.1836, -0.6523, 0.4395, -0.0771, -0.7461, 0.3613],
                [-0.1025, -0.0640, -0.0132, -0.6602, 0.3672, -0.1270, -0.9023, -0.0786],
                [-0.0874, 0.0552, 0.0442, -0.4824, 0.4375, -0.4121, -0.4980, -0.1836],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-DeepseekV4-Flash": Expectations(
        {
            ("cuda", (10, 0)): [
                [-0.3984, 0.3633, 0.1348, -0.0476, -0.1118, -0.0518, -0.0952, -0.4590],
                [-0.4336, 0.1836, 0.0049, -0.1514, -0.0449, -0.2383, -0.1113, -0.4766],
                [-0.2559, 0.0693, -0.0664, -0.2207, -0.2500, -0.1177, -0.1562, -0.2949],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-GptOss-MXFP4": Expectations(
        {
            ("cuda", (10, 0)): [
                [-0.1309, 0.2061, 0.2559, 0.5156, -0.2715, 0.5469, 0.4004, -0.0854],
                [-0.2490, 0.6406, 0.1289, 0.4277, -0.3809, 0.3125, 0.3516, -0.0811],
                [-0.2334, 0.1367, 0.2930, 0.6875, -0.1836, 0.4844, 0.1953, -0.2363],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-GlmMoeDsa-NVFP4": Expectations(
        {
            ("cuda", (10, 0)): [
                [0.1709, -0.4199, -0.4199, -0.2402, -0.3125, 0.6016, 0.3008, 0.1914],
                [-0.2051, -0.6328, -0.5391, 0.3184, 0.1299, 0.2773, 0.4824, 0.0938],
                [0.0447, -0.5078, -0.3672, -0.1992, 0.1553, 0.6523, 0.4141, -0.0038],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-GlmMoeDsa-NVFP4-all-linears": Expectations(
        {
            ("cuda", (10, 0)): [
                [0.2266, -0.2793, -0.3906, -0.1357, 0.0170, 0.4980, 0.1035, 0.2041],
                [-0.0649, -0.2188, -0.5547, 0.2041, 0.1201, 0.3750, 0.4355, 0.1099],
                [0.0908, -0.4492, -0.6094, -0.1836, 0.3203, 0.4453, 0.4199, 0.1172],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-MiniMaxM3-MXFP8": Expectations(
        {
            ("cuda", (10, 0)): [
                [0.2334, -0.0771, -0.3633, 0.1670, -0.2832, 0.1611, 0.4609, 0.0574],
                [0.4199, -0.1011, -0.2031, -0.2334, -0.2832, 0.1182, 0.5586, 0.1201],
                [0.1836, -0.0981, -0.6836, 0.2969, 0.0684, 0.0942, 0.5625, -0.0635],
            ],
        }
    ),  # fmt: skip
}


@slow
@require_kernels
@require_torch_accelerator
class FineGrainedTinyCheckpointTest(TestCasePlus):
    """Each shipped layout, from the hub: it must load with every key accounted for, compute what
    its dequantized weights compute within the format's own rounding, and save back to its own
    layout. The checkpoints come from the formats' own producers (modelopt, OpenAI's and DeepSeek's
    quantizers), so a converter is checked against what the format really is, not against what
    this code believes it to be."""

    @classmethod
    def setUpClass(cls):
        # the kernels autotune per shape from a cold cache, which dwarfs everything else here
        os.environ["FINEGRAINED_AUTOTUNE_TRIALS"] = "1"

    def setUp(self):
        super().setUp()
        # below that the quantizer dequantizes instead, which is not what is under test
        if torch_device == "cuda" and torch.cuda.get_device_capability() < (8, 9):
            self.skipTest("fine-grained quantized compute needs compute capability >= 8.9")

    def tearDown(self):
        super().tearDown()
        cleanup(torch_device, gc_collect=True)

    @staticmethod
    def _load(repo, **kwargs):
        config = AutoConfig.from_pretrained(repo, **kwargs)
        model_cls = AutoModelForImageTextToText if "vision_config" in config.to_dict() else AutoModelForCausalLM
        return model_cls.from_pretrained(repo, device_map=torch_device, output_loading_info=True, **kwargs)

    @staticmethod
    def _unbuilt_mtp_keys(model, keys):
        """The multi-token-prediction layer the checkpoint ships and the model does not build: it
        sits at index `num_hidden_layers`, which the models' own ignore rule only names at the
        full-size index."""
        mtp = model.config.get_text_config().num_hidden_layers
        return {key for key in keys if re.search(rf"(^|\.)layers\.{mtp}\.", key)}

    def _not_loaded(self, model, keys):
        """Checkpoint keys the model does not load, so a save does not write them back: the unbuilt MTP
        layer, and what the load declares it ignores (an FP8 KV cache's scales)."""
        ignored = model._keys_to_ignore_on_load_unexpected or []
        return self._unbuilt_mtp_keys(model, keys) | {k for k in keys if any(re.search(p, k) for p in ignored)}

    @parameterized.expand(TINY_CHECKPOINTS.items())
    def test_logits_match_the_dequantized_reference(self, repo, budget):
        model, info = self._load(repo)
        unexpected = set(info["unexpected_keys"]) - self._unbuilt_mtp_keys(model, info["unexpected_keys"])
        self.assertEqual((set(info["missing_keys"]), unexpected, set(info["mismatched_keys"])), (set(), set(), set()))

        reference, _ = self._load(repo, subfolder="reference", dtype=torch.bfloat16)
        ids = torch.arange(3, 35, dtype=torch.long, device=model.device).unsqueeze(0)
        with torch.no_grad():
            got, want = model(ids).logits.float(), reference(ids).logits.float()
        error = ((got - want).norm() / want.norm()).item()
        self.assertLessEqual(error, budget, f"{repo}: logits {error:.4f} away from the dequantized reference")

    @parameterized.expand(TINY_CHECKPOINTS)
    def test_logits_match_the_expected_slice(self, repo):
        model, _ = self._load(repo)
        ids = torch.arange(3, 35, dtype=torch.long, device=model.device).unsqueeze(0)
        with torch.no_grad():
            logits = model(ids).logits[0, :3, :8].float().cpu()
        expected = torch.tensor(EXPECTED_SLICES[repo].get_expectation())
        torch.testing.assert_close(logits, expected, rtol=1e-2, atol=5e-3)

    @parameterized.expand(TINY_CHECKPOINTS)
    def test_save_writes_the_checkpoint_layout_back(self, repo):
        """A pre-quantized model saved again lands on its checkpoint's keys, dtypes and bytes. The
        one exception is by design: NVFP4's gate and up `input_scale` share one activation global,
        which a save writes back to both."""
        from huggingface_hub import snapshot_download

        model, _ = self._load(repo)
        saved = os.path.join(self.get_auto_remove_tmp_dir(), "saved")
        model.save_pretrained(saved)
        source = _checkpoint_bytes(snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"]))
        source = {key: value for key, value in source.items() if key not in self._not_loaded(model, source)}
        written = _checkpoint_bytes(saved)
        self.assertEqual(sorted(written), sorted(source))
        for key, (dtype, shape, raw) in written.items():
            with self.subTest(key=key):
                self.assertEqual((dtype, shape), source[key][:2])
                if not re.search(r"\.(gate|up)_proj\.input_scale$", key):
                    self.assertEqual(raw, source[key][2])
