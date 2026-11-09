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

import unittest

from ..alm_tester import ALMModelTester
from ..causal_lm_tester import CausalLMModelTester
from ..multimodal_tester import MultiModalModelTester
from ..vlm_tester import VLMModelTester


class SmallConfigDefaultsTest(unittest.TestCase):
    def _tester_class(self, base):
        class Tester(base):
            @staticmethod
            def get_small_config_defaults():
                return {
                    "hidden_size": 64,
                    "image_size": 16,
                    "patch_size": 4,
                    "num_mel_bins": 24,
                    "no_rope_layers": [1, 0],
                }

            def _check_required_attributes(self):
                # These tests exercise constructor defaults without choosing model classes.
                pass

        return Tester

    def test_model_overrides_shared_defaults_and_caller_wins(self):
        for base in (MultiModalModelTester, VLMModelTester, ALMModelTester):
            with self.subTest(base=base.__name__):
                tester = self._tester_class(base)
                self.assertEqual(tester(self).hidden_size, 64)
                self.assertEqual(tester(self, hidden_size=128).hidden_size, 128)
                self.assertEqual(tester(self).intermediate_size, 32)

    def test_vision_derived_defaults_use_model_and_caller_overrides(self):
        tester = self._tester_class(VLMModelTester)
        model_defaults = tester(self)
        self.assertEqual(model_defaults.num_image_tokens, 16)
        self.assertEqual(model_defaults.seq_length, 23)
        caller_defaults = tester(self, image_size=8)
        self.assertEqual(caller_defaults.num_image_tokens, 4)
        self.assertEqual(caller_defaults.seq_length, 11)
        self.assertEqual(tester(self, num_image_tokens=2).seq_length, 9)

    def test_audio_model_overrides_modality_defaults(self):
        tester = self._tester_class(ALMModelTester)
        self.assertEqual(tester(self).num_mel_bins, 24)
        self.assertEqual(tester(self, num_mel_bins=16).num_mel_bins, 16)

    def test_defaults_are_fresh_for_each_instance(self):
        tester = self._tester_class(VLMModelTester)
        first, second = tester(self), tester(self)
        first.no_rope_layers.append(0)
        self.assertEqual(second.no_rope_layers, [1, 0])
        self.assertEqual(tester.get_small_config_defaults()["no_rope_layers"], [1, 0])

    def test_inherited_hook_keeps_existing_vision_defaults(self):
        class LegacyTester(VLMModelTester):
            def _check_required_attributes(self):
                pass

        tester = LegacyTester(self)
        self.assertEqual(tester.hidden_size, 32)
        self.assertEqual(tester.image_size, 8)
        self.assertEqual(tester.num_image_tokens, 4)
        self.assertEqual(tester.seq_length, 11)


class CausalLMSmallConfigDefaultsTest(unittest.TestCase):
    def _tester_class(self, defaults):
        class Tester(CausalLMModelTester):
            @staticmethod
            def get_small_config_defaults():
                return dict(defaults)

            @classmethod
            def _verify_and_infer_model_attributes(cls):
                # These tests exercise constructor defaults without choosing model classes.
                pass

        return Tester

    def test_model_overrides_shared_and_causal_defaults_and_caller_wins(self):
        tester = self._tester_class({"hidden_size": 64, "hidden_dropout_prob": 0.0, "mamba_n_heads": 4})
        defaults = tester(self)
        self.assertEqual(defaults.hidden_size, 64)
        self.assertEqual(defaults.hidden_dropout_prob, 0.0)
        self.assertEqual(defaults.mamba_n_heads, 4)
        self.assertEqual(defaults.intermediate_size, 32)
        self.assertEqual(defaults.attention_probs_dropout_prob, 0.1)
        caller = tester(self, hidden_size=128, hidden_dropout_prob=0.2)
        self.assertEqual(caller.hidden_size, 128)
        self.assertEqual(caller.hidden_dropout_prob, 0.2)

    def test_head_dim_derived_from_final_widths(self):
        tester = self._tester_class({"hidden_size": 64, "num_attention_heads": 4})
        self.assertEqual(tester(self).head_dim, 16)
        self.assertEqual(tester(self, hidden_size=32).head_dim, 8)
        self.assertEqual(self._tester_class({"head_dim": 32})(self).head_dim, 32)

    def test_inherited_hook_keeps_existing_defaults(self):
        tester = self._tester_class({})(self)
        self.assertEqual(tester.hidden_size, 32)
        self.assertEqual(tester.head_dim, 16)
        self.assertEqual(tester.batch_size, 13)
        self.assertEqual(tester.mamba_chunk_size, 16)
        self.assertFalse(tester.tie_word_embeddings)
        self.assertIsNone(tester.scope)
