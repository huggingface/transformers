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
import math
import unittest

import numpy as np

from transformers import Gr00tProcessor
from transformers.testing_utils import require_torch, require_torchvision, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

if is_vision_available():
    from transformers import AutoTokenizer, Gr00tImageProcessor


HORIZON = 4
EMBODIMENT = "test_robot"


def joint_stats(low, high, margin=1.0):
    # The percentiles are `low`/`high`, the min/max are `margin` further out
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    return {
        "min": (low - margin).tolist(),
        "max": (high + margin).tolist(),
        "q01": low.tolist(),
        "q99": high.tolist(),
        "mean": ((low + high) / 2).tolist(),
        "std": ((high - low) / 4).tolist(),
    }


@require_vision
@require_torch
@require_torchvision
class Gr00tProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Gr00tProcessor

    @classmethod
    def _setup_tokenizer(cls):
        return AutoTokenizer.from_pretrained("hf-internal-testing/tiny-processor-qwen3_vl")

    @classmethod
    def _setup_image_processor(cls):
        # The tiny tokenizer carries `patch_size=4` and pixel bounds that the processor forwards to the image processor
        return Gr00tImageProcessor(
            image_resolution=32, patch_size=4, size={"shortest_edge": 28 * 28, "longest_edge": 56 * 56}
        )

    @staticmethod
    def prepare_processor_dict():
        relative_eef_low = [-0.1, -0.1, -0.1, 0.9, -0.1, -0.1, -0.1, 0.9, -0.1]
        relative_eef_high = [0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1, 1.0, 0.1]
        return {
            "modality_configs": {
                EMBODIMENT: {
                    "video": {"delta_indices": [0], "modality_keys": ["front"]},
                    "state": {"delta_indices": [0], "modality_keys": ["eef", "arm", "gripper"]},
                    "action": {
                        "delta_indices": list(range(HORIZON)),
                        "modality_keys": ["eef", "arm", "gripper"],
                        "action_configs": [
                            {"rep": "RELATIVE", "type": "EEF", "format": "XYZ_ROT6D", "state_key": "eef"},
                            {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "arm"},
                            {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "gripper"},
                        ],
                    },
                    "language": {"delta_indices": [0], "modality_keys": ["instruction"]},
                }
            },
            "statistics": {
                EMBODIMENT: {
                    "state": {
                        "eef": joint_stats([-1] * 9, [1] * 9),
                        "arm": joint_stats([0.0, -2.0, 5.0], [2.0, 2.0, 5.0]),
                        "gripper": joint_stats([0.0], [1.0]),
                    },
                    "action": {
                        "eef": joint_stats([-1] * 9, [1] * 9),
                        "arm": joint_stats([0.0, -2.0, 5.0], [2.0, 2.0, 5.0]),
                        "gripper": joint_stats([0.0], [10.0]),
                    },
                    "relative_action": {
                        "eef": {
                            key: [value] * HORIZON
                            for key, value in joint_stats(relative_eef_low, relative_eef_high, margin=0).items()
                        },
                        "arm": {
                            key: [value] * HORIZON
                            for key, value in joint_stats([-1.0] * 3, [1.0] * 3, margin=0).items()
                        },
                    },
                }
            },
            "embodiment_id_mapping": {EMBODIMENT: 3},
            "max_state_dim": 16,
            "max_action_horizon": HORIZON,
        }

    def get_gr00t_processor(self):
        components = self.prepare_components()
        return self.processor_class(
            **components, chat_template=components["tokenizer"].chat_template, **self.prepare_processor_dict()
        )

    @staticmethod
    def identity_eef(batch_size, translation=(0.0, 0.0, 0.0)):
        return np.tile(np.array([*translation, 1, 0, 0, 0, 1, 0], dtype=np.float32), (batch_size, 1))

    def test_instruction_is_formatted_into_prompt(self):
        processor = self.get_gr00t_processor()
        image = self.prepare_images_inputs()
        raw = processor(images=[[image]], text="Pick up the CUP, then stop!", return_tensors="pt")
        clean = processor(images=[[image]], text="pick up the cup then stop", return_tensors="pt")
        torch.testing.assert_close(raw["input_ids"], clean["input_ids"])
        prompt = processor.tokenizer.decode(raw["input_ids"][0])
        self.assertTrue(prompt.startswith("<|im_start|>user\n<|vision_start|>"))
        self.assertIn("pick up the cup then stop<|im_end|>", prompt)
        num_image_tokens = raw["image_grid_thw"][0].prod().item() // processor.image_processor.merge_size**2
        self.assertEqual((raw["input_ids"] == processor.image_token_id).sum().item(), num_image_tokens)

    def test_batch_is_left_padded(self):
        processor = self.get_gr00t_processor()
        image = self.prepare_images_inputs()
        inputs = processor(images=[[image], [image]], text=["go", "go to the far left corner"], return_tensors="pt")
        self.assertEqual(inputs["attention_mask"][0, 0].item(), 0)
        self.assertEqual(inputs["attention_mask"][0, -1].item(), 1)

    def test_state_normalization(self):
        processor = self.get_gr00t_processor()
        state = {
            "eef": self.identity_eef(2),
            # The third joint has q01 == q99 and is mapped to 0; out-of-range values are clipped
            "arm": np.array([[0.0, -2.0, 5.0], [1.0, 4.0, 5.0]], dtype=np.float32),
            "gripper": np.array([[0.5], [1.0]], dtype=np.float32),
        }
        inputs = processor(
            images=[[self.prepare_images_inputs()]] * 2,
            text=["a", "b"],
            state=state,
            embodiment_tag=EMBODIMENT,
            return_tensors="pt",
        )
        self.assertEqual(tuple(inputs["state"].shape), (2, 16))
        torch.testing.assert_close(
            inputs["state"][:, 9:13], torch.tensor([[-1.0, -1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        )
        self.assertTrue(inputs["state"][:, 13:].eq(0).all())
        self.assertEqual(inputs["embodiment_id"].tolist(), [3, 3])

    def test_unknown_embodiment_raises(self):
        processor = self.get_gr00t_processor()
        with self.assertRaisesRegex(ValueError, "Unknown embodiment"):
            processor(images=[[self.prepare_images_inputs()]], text="a", embodiment_tag="other_robot")

    def test_post_process_actions(self):
        processor = self.get_gr00t_processor()
        batch_size = 2
        state = {
            "eef": self.identity_eef(batch_size, translation=(1.0, 2.0, 3.0)),
            "arm": np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], dtype=np.float32),
            "gripper": np.zeros((batch_size, 1), dtype=np.float32),
        }
        actions = torch.zeros(batch_size, HORIZON + 2, 16)
        # Normalized relative eef action: identity rotation with a 0.1 translation along x. The relative actions are
        # scaled with the min/max of the relative-action statistics
        actions[..., 0] = 1.0
        actions[..., 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        # Relative arm action of +1 on every joint, absolute gripper action at the middle of its range
        actions[..., 9:12] = 1.0
        outputs = processor.post_process_actions(actions, EMBODIMENT, state=state)

        self.assertEqual(sorted(outputs), ["arm", "eef", "gripper"])
        self.assertEqual(tuple(outputs["eef"].shape), (batch_size, HORIZON, 9))
        expected_eef = torch.tensor([1.1, 2.0, 3.0, 1, 0, 0, 0, 1, 0], dtype=torch.float64)
        torch.testing.assert_close(outputs["eef"], expected_eef.expand(batch_size, HORIZON, 9))
        expected_arm = torch.tensor([[1.5] * 3, [2.0] * 3], dtype=torch.float64)[:, None].expand(-1, HORIZON, -1)
        torch.testing.assert_close(outputs["arm"], expected_arm)
        torch.testing.assert_close(outputs["gripper"], torch.full((batch_size, HORIZON, 1), 5.0, dtype=torch.float64))

    def test_relative_eef_rotation_is_composed(self):
        processor = self.get_gr00t_processor()
        angle = math.pi / 2
        # Reference pose rotated by 90 degrees around z, relative action translating by 0.1 along its x axis
        reference_rows = [math.cos(angle), -math.sin(angle), 0, math.sin(angle), math.cos(angle), 0]
        state = {
            "eef": np.array([[0.0, 0.0, 0.0, *reference_rows]], dtype=np.float32),
            "arm": np.zeros((1, 3), dtype=np.float32),
            "gripper": np.zeros((1, 1), dtype=np.float32),
        }
        actions = torch.zeros(1, HORIZON, 16)
        actions[..., 0] = 1.0
        actions[..., 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        eef = processor.post_process_actions(actions, EMBODIMENT, state=state)["eef"][0, 0]
        torch.testing.assert_close(eef[:3], torch.tensor([0.0, 0.1, 0.0], dtype=torch.float64), atol=1e-6, rtol=0)
        torch.testing.assert_close(eef[3:], torch.tensor(reference_rows, dtype=torch.float64), atol=1e-6, rtol=0)

    def test_relative_actions_require_state(self):
        processor = self.get_gr00t_processor()
        with self.assertRaisesRegex(ValueError, "`state` is required"):
            processor.post_process_actions(torch.zeros(1, HORIZON, 16), EMBODIMENT)

    def test_post_process_actions_with_disabled_group(self):
        processor_dict = self.prepare_processor_dict()
        # Some embodiments disable a joint group: its statistics are empty, but its relative statistics are not
        processor_dict["statistics"][EMBODIMENT]["action"]["eef"] = {key: [] for key in joint_stats([0], [1])}
        processor = self.processor_class(**self.prepare_components(), **processor_dict)
        state = {"eef": self.identity_eef(1), "arm": np.zeros((1, 3)), "gripper": np.zeros((1, 1))}
        outputs = processor.post_process_actions(torch.zeros(1, HORIZON, 16), EMBODIMENT, state=state)
        self.assertEqual(tuple(outputs["eef"].shape), (1, HORIZON, 0))
        self.assertEqual(tuple(outputs["arm"].shape), (1, HORIZON, 3))
