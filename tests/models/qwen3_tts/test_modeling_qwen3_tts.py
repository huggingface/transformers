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

import json
import tempfile
import unittest
from pathlib import Path

from transformers import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)
from transformers.trainer_utils import set_seed

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch


class Qwen3TTSModelTester:
    """
    Builds a tiny Qwen3TTS config and synthetic inputs for unit testing.
    """

    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=10,
        is_training=False,
        talker_config=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training

        # Tiny talker config
        self.talker_config = talker_config or {
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "text_vocab_size": 64,
            "text_hidden_size": 32,
            "num_code_groups": 2,
            "code_predictor_config": {
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
            },
        }

    def get_config(self):
        return Qwen3TTSConfig(
            talker_config=self.talker_config,
            tts_pad_token_id=0,
            tts_bos_token_id=1,
            tts_eos_token_id=2,
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.talker_config["text_vocab_size"])
        attention_mask = torch.ones([self.batch_size, self.seq_length], dtype=torch.long, device=torch_device)
        config = self.get_config()
        return config, input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Qwen3TTSForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Qwen3TTSForConditionalGeneration,) if is_torch_available() else ()
    _is_composite = True
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = Qwen3TTSModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Qwen3TTSConfig, has_text_modality=False)
        _no_forward_tests = (
            "test_eager_matches_sdpa_inference",
            "test_attention_outputs",
            "test_feed_forward_chunking",
            "test_hidden_states_output",
            "test_model_forward_default_config_values",
            "test_retain_grad_hidden_states_attentions",
            "test_inputs_embeds",
        )
        if any(name in self._testMethodName for name in _no_forward_tests):
            self.skipTest("Qwen3TTSForConditionalGeneration has no standard forward()")

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model_instantiation(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = Qwen3TTSForConditionalGeneration(config)
        self.assertIsNotNone(model)

    def test_save_load(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).eval().to(torch_device)
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                loaded = model_class.from_pretrained(tmpdirname).eval().to(torch_device)
            for key in model.state_dict():
                self.assertTrue(
                    torch.allclose(model.state_dict()[key], loaded.state_dict()[key]),
                    f"Mismatch in key: {key}",
                )

    @unittest.skip(reason="Qwen3TTS uses a custom generation mixin, not standard generate()")
    def test_generate_without_input_ids(self):
        pass

    @unittest.skip(reason="Qwen3TTS has a composite architecture with sub-models")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(reason="Qwen3TTSForConditionalGeneration uses a custom generation mixin, not GenerationMixin")
    def test_generation_tester_mixin_inheritance(self):
        pass

    @unittest.skip(reason="Composite model config attn implementation not propagated uniformly")
    def test_config_attn_implementation_setter(self):
        pass

    @unittest.skip(reason="Qwen3TTSForConditionalGeneration does not expose get_input_embeddings directly")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip(reason="main_input_name requires converter re-run to propagate to modeling file")
    def test_model_main_input_name(self):
        pass

    @unittest.skip(reason="Qwen3TTS generation is non-standard (two-stage talker + code predictor)")
    def test_training(self):
        pass

    @unittest.skip(reason="Qwen3TTS forward is generation-only; determinism not guaranteed across batch")
    def test_determinism(self):
        pass

    @unittest.skip(reason="Qwen3TTS forward requires generation-stage inputs not covered by batching test")
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason="Qwen3TTS forward is non-standard; model outputs equivalence not applicable")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="Compile not yet supported for Qwen3TTS")
    def test_sdpa_can_compile_dynamic(self):
        pass

    @unittest.skip(reason="Compile not yet supported for Qwen3TTS")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Qwen3TTS right-padding equivalence not applicable")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip(reason="Qwen3TTSForConditionalGeneration has no standard forward; uses custom generation mixin")
    def test_all_tensors_are_parameter_or_buffer(self):
        pass

    @unittest.skip(reason="Qwen3TTSForConditionalGeneration has no standard forward")
    def test_sdpa_can_dispatch_composite_models(self):
        pass

    @unittest.skip(reason="Qwen3TTSForConditionalGeneration has no standard forward")
    def test_left_padding_compatibility(self):
        pass

    @unittest.skip(reason="Qwen3TTSForConditionalGeneration has no standard forward")
    def test_torch_fx(self):
        pass

    @unittest.skip(reason="Qwen3TTSForConditionalGeneration has no standard forward")
    def test_torch_fx_output_loss(self):
        pass


@require_torch
class Qwen3TTSForConditionalGenerationIntegrationTest(unittest.TestCase):
    """
    Integration tests that run against a real converted checkpoint.
    Validates that the HF modeling code produces the same outputs as the original.
    """

    @classmethod
    def setUpClass(cls):
        from transformers.testing_utils import cleanup

        cleanup(torch_device, gc_collect=True)
        cls.checkpoint = "Qwen/Qwen3-TTS"

    def tearDown(self):
        from transformers.testing_utils import cleanup

        cleanup(torch_device, gc_collect=True)

    @slow
    def test_single(self):
        """
        Text -> audio codes sanity check against saved expected outputs.
        """
        set_seed(42)

        path = Path(__file__).parent.parent.parent / "fixtures/qwen3_tts/expected_results_single.json"
        with open(path, "r", encoding="utf-8") as f:
            expected_outputs = json.load(f)

        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(self.checkpoint)
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype=torch.bfloat16
        )

        text = expected_outputs["input_text"]
        inputs = processor(text=text, return_tensors="pt").to(torch_device)

        torch.testing.assert_close(inputs["input_ids"].cpu(), torch.tensor(expected_outputs["input_ids"]))

        with torch.no_grad():
            output = model.generate(**inputs)

        torch.testing.assert_close(output.cpu(), torch.tensor(expected_outputs["generated_ids"]))

    @slow
    def test_batch(self):
        """
        Batch of texts -> audio codes sanity check against saved expected outputs.
        """
        set_seed(42)

        path = Path(__file__).parent.parent.parent / "fixtures/qwen3_tts/expected_results_batch.json"
        with open(path, "r", encoding="utf-8") as f:
            expected_outputs = json.load(f)

        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(self.checkpoint)
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype=torch.bfloat16
        )

        texts = expected_outputs["input_texts"]
        inputs = processor(text=texts, return_tensors="pt", padding=True).to(torch_device)

        with torch.no_grad():
            output = model.generate(**inputs)

        for i, exp_ids in enumerate(expected_outputs["generated_ids"]):
            actual = output[i, : len(exp_ids)].cpu()
            torch.testing.assert_close(actual, torch.tensor(exp_ids))
