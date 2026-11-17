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
"""Tests of the checks of the GR00T conversion script, on small original-format configs."""

import copy
import json
import os
import tempfile
import unittest

from transformers.testing_utils import require_torch


@require_torch
class Gr00tConversionTest(unittest.TestCase):
    # The relevant fields of the original `nvidia/GR00T-N1.7-3B` config
    original_config = {
        "architectures": ["Gr00tN1d7"],
        "model_type": "Gr00tN1d7",
        "model_name": "nvidia/Cosmos-Reason2-2B",
        "select_layer": 16,
        "backbone_embedding_dim": 2048,
        "hidden_size": 1024,
        "max_state_dim": 132,
        "max_action_dim": 132,
        "max_num_embodiments": 32,
        "use_alternate_vl_dit": True,
        "use_vl_self_attention": True,
        "use_vlln": True,
        "add_pos_embed": True,
        "exclude_state": False,
        "diffusion_model_cfg": {
            "attention_head_dim": 48,
            "dropout": 0.2,
            "final_dropout": True,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_attention_heads": 32,
            "num_layers": 32,
            "output_dim": 1024,
            "positional_embeddings": None,
        },
        "vl_self_attention_cfg": {
            "attention_head_dim": 64,
            "dropout": 0.2,
            "final_dropout": True,
            "num_attention_heads": 32,
            "num_layers": 4,
            "positional_embeddings": None,
        },
    }
    processor_kwargs = {
        "max_state_dim": 132,
        "max_action_dim": 132,
        "use_albumentations": True,
        "use_percentiles": True,
        "use_relative_action": True,
        "apply_sincos_state_encoding": False,
        "use_mean_std": False,
        "exclude_state": False,
        "modality_configs": {
            "test_robot": {
                "state": {"delta_indices": [0], "modality_keys": ["eef", "joints"]},
                "action": {
                    "delta_indices": [0, 1],
                    "modality_keys": ["eef", "gripper", "joints"],
                    "action_configs": [
                        {"rep": "RELATIVE", "type": "EEF", "format": "XYZ_ROT6D", "state_key": "eef"},
                        {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT"},
                        {"rep": "RELATIVE", "type": "NON_EEF", "format": "DEFAULT", "state_key": "joints"},
                    ],
                },
            }
        },
    }

    @classmethod
    def setUpClass(cls):
        from transformers.models.gr00t import convert_gr00t_to_hf

        cls.convert = convert_gr00t_to_hf

    def assert_rejected(self, function, *args, message):
        with self.assertRaises(ValueError) as context:
            function(*args)
        self.assertIn(message, str(context.exception))

    def test_original_config_is_accepted(self):
        self.convert.validate_original_config(copy.deepcopy(self.original_config))

    def test_unsupported_architectures_are_rejected(self):
        validate = self.convert.validate_original_config
        cases = {
            "only the original `Gr00tN1d7` architecture": {"architectures": ["Gr00tN1d7Icl"]},
            "without the VL refiner": {"use_vl_self_attention": False, "vl_self_attention_cfg": None},
            "strict_vl_padding_mask=True": {"strict_vl_padding_mask": True},
            "use_tactile=True": {"use_tactile": True},
        }
        for message, changes in cases.items():
            with self.subTest(message):
                self.assert_rejected(validate, {**copy.deepcopy(self.original_config), **changes}, message=message)

        pruned = copy.deepcopy(self.original_config)
        pruned["diffusion_model_cfg"].update(num_layers=2, block_indices=[0, 5])
        self.assert_rejected(validate, pruned, message="pruned DiTs")

        refiner_dropout = copy.deepcopy(self.original_config)
        refiner_dropout["vl_self_attention_cfg"]["dropout"] = 0.1
        self.assert_rejected(validate, refiner_dropout, message="single `dropout`")

        # A missing refiner `positional_embeddings` takes the original default ("sinusoidal")
        refiner_positions = copy.deepcopy(self.original_config)
        del refiner_positions["vl_self_attention_cfg"]["positional_embeddings"]
        self.assert_rejected(validate, refiner_positions, message="no positional embeddings")

    def test_original_processor_is_accepted(self):
        kwargs = copy.deepcopy(self.processor_kwargs)
        # `DELTA` groups are treated as absolute by the original processor, as by `Gr00tProcessor`
        kwargs["modality_configs"]["test_robot"]["action"]["action_configs"][1]["rep"] = "DELTA"
        # Per-key sin/cos keys are only used with `apply_sincos_state_encoding`
        kwargs["modality_configs"]["test_robot"]["state"]["sin_cos_embedding_keys"] = ["joints"]
        self.convert.validate_original_processor(kwargs, {"test_robot": 10}, self.original_config)

    def test_unsupported_processor_settings_are_rejected(self):
        validate = self.convert.validate_original_processor

        def with_changes(update):
            kwargs = copy.deepcopy(self.processor_kwargs)
            update(kwargs, kwargs["modality_configs"]["test_robot"])
            return kwargs

        cases = {
            "unknown action config REL_XYZ_ROT6D": lambda k, m: m["action"]["action_configs"][0].update(
                rep="REL_XYZ_ROT6D"
            ),
            "relative EEF actions in XYZ_ROTVEC": lambda k, m: m["action"]["action_configs"][0].update(
                format="XYZ_ROTVEC"
            ),
            "per-key mean_std_embedding_keys": lambda k, m: m["state"].update(mean_std_embedding_keys=["eef"]),
            "per-key pass_through_keys": lambda k, m: m["state"].update(pass_through_keys=["joints"]),
            "use_albumentations=False": lambda k, m: k.pop("use_albumentations"),
            "apply_sincos_state_encoding=True": lambda k, m: k.update(apply_sincos_state_encoding=True),
            "the processor pads to 64": lambda k, m: k.update(max_state_dim=64),
        }
        for message, update in cases.items():
            with self.subTest(message):
                self.assert_rejected(
                    validate, with_changes(update), {"test_robot": 10}, self.original_config, message=message
                )
        self.assert_rejected(
            validate, copy.deepcopy(self.processor_kwargs), {"test_robot": 40}, self.original_config, message="outside"
        )

        # Relative groups are converted only with `use_relative_action`: otherwise any format is treated as absolute
        absolute = with_changes(lambda k, m: m["action"]["action_configs"][0].update(format="XYZ_ROTVEC"))
        absolute["use_relative_action"] = False
        validate(absolute, {"test_robot": 10}, self.original_config)

    def test_missing_processor_settings_take_original_defaults(self):
        self.assertFalse(self.convert.processor_kwarg({}, "use_percentiles"))
        self.assertFalse(self.convert.processor_kwarg({}, "use_relative_action"))
        self.assertTrue(self.convert.processor_kwarg({"use_relative_action": None}, "clip_outliers"))

    def test_embodiment_mapping_adds_original_tags(self):
        mapping = self.convert.convert_embodiment_id_mapping({"libero_sim": 5, "test_robot": 12})
        self.assertEqual(mapping["libero_sim"], 5)  # the checkpoint's mapping takes precedence
        self.assertEqual(mapping["test_robot"], 12)
        self.assertEqual(mapping["new_embodiment"], 10)
        self.assertEqual(self.convert.convert_embodiment_id_mapping(None)["new_embodiment"], 10)

    def test_processor_files_in_subfolder_are_found(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                self.convert.find_processor_dir(directory)
            os.makedirs(os.path.join(directory, "processor"))
            with open(os.path.join(directory, "processor", "processor_config.json"), "w", encoding="utf-8") as f:
                json.dump({"processor_kwargs": self.processor_kwargs}, f)
            self.assertEqual(self.convert.find_processor_dir(directory), os.path.join(directory, "processor"))
