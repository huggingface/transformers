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
"""Accuracy and serialization checks against independently generated tiny checkpoints."""

import os
import re

import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText
from transformers.testing_utils import (
    Expectations,
    cleanup,
    get_device_properties,
    require_kernels,
    require_torch_accelerator,
    slow,
    torch_device,
)

from .block_scaled_tester import FormatSpec, _cheap_tuning, _require_kernel_capability


def _checkpoint_bytes(path):
    """Return `{key: (dtype, shape, raw bytes)}` for a saved checkpoint."""
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


# Logits `[0, :3, :8]` of each checkpoint on `torch.arange(3, 35)`, per device.
# They are bit-stable across tuner caches, so a tight tolerance holds.
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
                [-0.1089, -0.1553, -0.2207, -0.6562, 0.4219, -0.1221, -0.7578, 0.3145],
                [-0.0967, -0.0444, -0.0253, -0.6523, 0.3828, -0.1660, -0.9258, -0.0471],
                [-0.0791, 0.0947, 0.0309, -0.4688, 0.4316, -0.4414, -0.5039, -0.1777],
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
                [0.3086, -0.2773, -0.4883, -0.1797, -0.0073, 0.4531, 0.2158, 0.2754],
                [0.0295, -0.2598, -0.5586, 0.2432, -0.0106, 0.3496, 0.4805, 0.1279],
                [0.1533, -0.4551, -0.5742, -0.1973, 0.2139, 0.4141, 0.3574, 0.1338],
            ],
        }
    ),  # fmt: skip
    "IlyasMoutawwakil/tiny-random-Muse-Spark-NVFP4": Expectations(
        {
            ("cuda", (10, 0)): [
                [0.0430, -0.0131, -0.0304, 0.0415, -0.0082, -0.0029, 0.0042, -0.0498],
                [0.0105, 0.0723, 0.0615, -0.0175, 0.0522, 0.0015, 0.0023, 0.0304],
                [0.0089, -0.0251, 0.0131, 0.0234, 0.0243, -0.0283, -0.0703, 0.0664],
            ],
        }
    ),  # fmt: skip
}


class TinyCheckpointTesterMixin:
    """Tests on a format's tiny checkpoints.

    The checkpoints are random copies of shipped ones, written by their producers' quantizers.
    Each must load every key, match its dequantized reference and recorded logits, and save back to its own layout.
    This checks converters against the real format, not against our reading of it.
    A format's test file sets `spec`.
    """

    spec: FormatSpec

    def setUp(self):
        super().setUp()
        self.enterContext(_cheap_tuning())
        _require_kernel_capability(self)

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
        """Return the keys of the multi-token-prediction layer that the model does not build.

        The layer sits at index `num_hidden_layers`. The models' own ignore rule only names the full-size index.
        """
        mtp = model.config.get_text_config().num_hidden_layers
        return {key for key in keys if re.search(rf"(^|\.)layers\.{mtp}\.", key)}

    def _not_loaded(self, model, keys):
        """Return the checkpoint keys the model does not load, so a save does not write them back.

        These are the unbuilt MTP layer and the keys the load ignores, like FP8 KV cache scales.
        """
        ignored = model._keys_to_ignore_on_load_unexpected or []
        return self._unbuilt_mtp_keys(model, keys) | {k for k in keys if any(re.search(p, k) for p in ignored)}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._reference_logits = {}

    def _reference_error(self, model, repo):
        """Return the logits and their relative error against the dequantized reference, computed once per repo."""
        ids = torch.arange(3, 35, dtype=torch.long, device=model.device).unsqueeze(0)
        if repo not in self._reference_logits:
            reference, _ = self._load(repo, subfolder="reference", dtype=torch.bfloat16)
            with torch.no_grad():
                self._reference_logits[repo] = reference(ids).logits.float().cpu()
            del reference
        with torch.no_grad():
            got = model(ids).logits.float()
        want = self._reference_logits[repo].to(got.device)
        return ((got - want).norm() / want.norm()).item(), got

    @staticmethod
    def _recorded_slice(repo):
        """Return the slice recorded on this exact device, if any. Other devices' kernels round differently."""
        return dict(EXPECTED_SLICES[repo].unpacked()).get(get_device_properties())

    @slow
    @require_kernels
    @require_torch_accelerator
    def test_dequantized_load_matches_the_reference(self):
        """Check that a `dequantize=True` load matches the dequantized reference."""
        for repo, budget in self.spec.tiny_checkpoints.items():
            with self.subTest(repo=repo):
                model, info = self._load(repo, quantization_config=self.spec.config_cls(dequantize=True))
                unexpected = set(info["unexpected_keys"]) - self._unbuilt_mtp_keys(model, info["unexpected_keys"])
                self.assertEqual((set(info["missing_keys"]), unexpected), (set(), set()))
                # Nothing stays quantized. fp32 parameters like DeepSeek-V4's hyper-connections are expected.
                quantized = {torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.int8, torch.uint8}
                self.assertEqual([name for name, param in model.named_parameters() if param.dtype in quantized], [])
                error, _ = self._reference_error(model, repo)
                self.assertLessEqual(error, budget, f"{repo}: logits {error:.4f} away from the dequantized reference")

    @slow
    @require_kernels
    @require_torch_accelerator
    def test_load_matches_the_reference_and_saves_the_checkpoint_layout_back(self):
        """Check that every key loads and the logits match the reference and the recorded slice.

        Saving the model again must reproduce the checkpoint's keys, dtypes and bytes.
        """
        from huggingface_hub import snapshot_download

        for repo, budget in self.spec.tiny_checkpoints.items():
            with self.subTest(repo=repo):
                model, info = self._load(repo)
                unexpected = set(info["unexpected_keys"]) - self._unbuilt_mtp_keys(model, info["unexpected_keys"])
                self.assertEqual(
                    (set(info["missing_keys"]), unexpected, set(info["mismatched_keys"])), (set(), set(), set())
                )
                error, logits = self._reference_error(model, repo)
                self.assertLessEqual(error, budget, f"{repo}: logits {error:.4f} away from the dequantized reference")
                expected = self._recorded_slice(repo)
                if expected is not None:
                    torch.testing.assert_close(logits[0, :3, :8].cpu(), torch.tensor(expected), rtol=1e-2, atol=5e-3)

                saved = os.path.join(self.get_auto_remove_tmp_dir(), "saved")
                model.save_pretrained(saved)
                source = _checkpoint_bytes(snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"]))
                source = {key: value for key, value in source.items() if key not in self._not_loaded(model, source)}
                written = _checkpoint_bytes(saved)
                self.assertEqual(sorted(written), sorted(source))
                for key, (dtype, shape, raw) in written.items():
                    with self.subTest(repo=repo, key=key):
                        source_dtype, source_shape, source_raw = source[key]
                        self.assertEqual(shape, source_shape)
                        if dtype != source_dtype:
                            # Loading keeps initialized dtypes, like FP32 norms and BF16 MiniMax routers.
                            # FP8 scales normalize to FP32. Float32 UE8M0 containers save as native E8M0.
                            self.assertTrue(
                                (
                                    dtype == torch.float32
                                    and key.endswith("norm.weight")
                                    and source_dtype == torch.bfloat16
                                )
                                or (
                                    dtype == torch.bfloat16
                                    and source_dtype == torch.float32
                                    and key.endswith(
                                        ("gate.weight", "e_score_correction_bias", "patch_embedding.weight")
                                    )
                                )
                                or (
                                    dtype == torch.float32
                                    and source_dtype == torch.bfloat16
                                    and key.endswith("_scale_inv")
                                    and written.get(key.removesuffix("_scale_inv"), (None,))[0] == torch.float8_e4m3fn
                                )
                                or (
                                    dtype == torch.float8_e8m0fnu
                                    and source_dtype == torch.float32
                                    and key.endswith((".scale", "_scale_inv"))
                                ),
                                f"{repo}: {key} unexpectedly changed dtype from {source_dtype} to {dtype}",
                            )
                            expected = torch.frombuffer(bytearray(source_raw), dtype=source_dtype).to(dtype)
                            self.assertEqual(raw, expected.view(torch.uint8).numpy().tobytes())
                        else:
                            self.assertEqual(raw, source_raw)
