# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Testing suite for the PyTorch NemotronH_Omni model.

`NemotronH_Omni_Reasoning_V3` is a bespoke multimodal model (RADIO vision tower +
optional Parakeet audio + NemotronH language model) whose forward always requires coupled
image inputs (`pixel_values` + `image_flags` + image-context tokens). The generic
`ModelTesterMixin` common tests assume text-only conventions and do not apply, so this file
provides targeted tests instead. `all_model_classes` is declared so the repo-consistency
check (`utils/check_repo.py`) recognizes the model as tested.
"""

import tempfile
import unittest

from transformers import NemotronH_Omni_Reasoning_V3_Config, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_modeling_common import floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import NemotronH_Omni_Reasoning_V3


class NemotronHOmniModelTester:
    """Builds a tiny NemotronH_Omni model and coupled multimodal inputs.

    The image branch is sized so a single image yields exactly one `img_context` token after
    the RADIO patch-embed + pixel-shuffle:
        num_image_token = (force_image_size // patch_size) ** 2 * downsample_ratio ** 2
                        = (32 // 16) ** 2 * 0.5 ** 2 = 1
    so each sequence must contain exactly one `img_context_token_id`.
    """

    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=8,
        force_image_size=32,
        patch_size=16,
        downsample_ratio=0.5,
        vit_hidden_size=32,
        projector_hidden_size=64,
        img_context_token_id=1,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.force_image_size = force_image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.vit_hidden_size = vit_hidden_size
        self.projector_hidden_size = projector_hidden_size
        self.img_context_token_id = img_context_token_id

        self.vocab_size = 99
        self.hidden_size = 32
        self.llm_config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "layers_block_type": ["linear_attention", "moe", "full_attention", "moe"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 40,
            "moe_intermediate_size": 40,
            "moe_shared_expert_intermediate_size": 40,
            "mlp_hidden_act": "relu2",
            "mamba_hidden_act": "silu",
            "ssm_state_size": 16,
            "mamba_num_heads": 8,
            "mamba_n_groups": 2,
            "mamba_head_dim": 8,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
            "mamba_chunk_size": 8,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "use_mamba_kernels": False,
        }
        self.vision_config = {
            "hidden_size": self.vit_hidden_size,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "mlp_ratio": 2.0,
            "patch_size": self.patch_size,
            "image_size": self.force_image_size,
            "max_img_size": 64,
            "num_channels": 3,
            # >= 2 cls tokens so the default summary_idxs=[0, 1] is in-bounds
            "num_cls_tokens": 2,
            "num_registers": 1,
        }
        self.num_image_token = int((force_image_size // patch_size) ** 2 * (downsample_ratio**2))

    def get_config(self):
        return NemotronH_Omni_Reasoning_V3_Config(
            vision_config=self.vision_config,
            llm_config=self.llm_config,
            sound_config=None,
            force_image_size=self.force_image_size,
            patch_size=self.patch_size,
            downsample_ratio=self.downsample_ratio,
            vit_hidden_size=self.vit_hidden_size,
            projector_hidden_size=self.projector_hidden_size,
            img_context_token_id=self.img_context_token_id,
            attn_implementation="eager",
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        # ids in [3, vocab) so they never collide with img_context_token_id (1)
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 3) + 3
        input_ids[:, 1 : 1 + self.num_image_token] = self.img_context_token_id
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        attention_mask[:, : 1 + self.num_image_token] = 1  # keep image tokens unmasked
        pixel_values = floats_tensor([self.batch_size, 3, self.force_image_size, self.force_image_size])
        image_flags = torch.ones(self.batch_size, 1, dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_flags": image_flags,
        }, config


@require_torch
class NemotronHOmniModelTest(unittest.TestCase):
    all_model_classes = (NemotronH_Omni_Reasoning_V3,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = NemotronHOmniModelTester(self)

    def _inputs_and_model(self):
        # The model hardcodes bfloat16 for the projected vision features (see
        # `_extract_feature_single`), so it must run in bfloat16 end-to-end.
        inputs, config = self.model_tester.prepare_config_and_inputs()
        model = NemotronH_Omni_Reasoning_V3(config).to(torch_device, torch.bfloat16).eval()
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return inputs, model

    def test_main_input_name(self):
        self.assertEqual(NemotronH_Omni_Reasoning_V3.main_input_name, "pixel_values")

    def test_model_forward(self):
        inputs, model = self._inputs_and_model()
        with torch.no_grad():
            out = model(**inputs)
        self.assertEqual(
            tuple(out.logits.shape),
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.vocab_size),
        )

    def test_forward_with_labels_returns_loss(self):
        inputs, model = self._inputs_and_model()
        inputs["labels"] = inputs["input_ids"].clone()
        with torch.no_grad():
            out = model(**inputs)
        self.assertIsNotNone(out.loss)
        self.assertEqual(out.loss.dim(), 0)

    def test_save_load_roundtrip(self):
        inputs, model = self._inputs_and_model()
        with torch.no_grad():
            logits_a = model(**inputs).logits
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            # attn_implementation isn't persisted in the saved config (defaults back to flash,
            # which is unavailable on CPU), so request eager explicitly on reload.
            reloaded = (
                NemotronH_Omni_Reasoning_V3.from_pretrained(
                    tmp, attn_implementation="eager", dtype=torch.bfloat16
                )
                .to(torch_device)
                .eval()
            )
        with torch.no_grad():
            logits_b = reloaded(**inputs).logits
        torch.testing.assert_close(logits_a, logits_b, atol=1e-4, rtol=1e-4)
