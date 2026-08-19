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
    # `generate` is provided by `Qwen3TTSGenerationMixin`, whose signature takes a list of prompts plus
    # `languages`/`speakers` and returns codec codes rather than token ids, so the generic `generate`
    # tests do not apply to it.
    all_generative_model_classes = ()
    _is_composite = True
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    # base_model (the talker text encoder) carries a sub-config, not the composite Qwen3TTSConfig,
    # so it cannot round-trip through Qwen3TTSForConditionalGeneration.from_pretrained.
    test_missing_keys = False

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
            "test_capture_outputs_decorator",
        )
        if any(name in self._testMethodName for name in _no_forward_tests):
            self.skipTest(
                "`forward` requires `past_hidden` from the preceding generation step, which the common "
                "tester does not provide"
            )

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

    # `forward` runs one step of the talker loop and expects `past_hidden` from the previous step, which
    # only the generation loop produces; the common testers call it with standard inputs, so it raises on
    # `torch.cat((past_hidden, last_id_hidden))` with `past_hidden=None`.
    _forward_needs_generation_state = (
        "`forward` requires `past_hidden` from the preceding generation step, which the common tester "
        "does not provide"
    )

    @unittest.skip(reason=_forward_needs_generation_state)
    def test_all_tensors_are_parameter_or_buffer(self):
        pass

    @unittest.skip(reason=_forward_needs_generation_state)
    def test_batching_equivalence(self):
        pass

    @unittest.skip(reason=_forward_needs_generation_state)
    def test_determinism(self):
        pass

    @unittest.skip(reason=_forward_needs_generation_state)
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(
        reason="`attn_implementation` set on Qwen3TTSConfig is not propagated to `talker_config`, so the "
        "sub-config reports None instead of the requested value"
    )
    def test_config_attn_implementation_setter(self):
        pass


def _build_assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


@require_torch
class Qwen3TTSForConditionalGenerationIntegrationTest(unittest.TestCase):
    """
    Integration tests that run against a real converted checkpoint.

    The fixtures hold the codes produced by the original Qwen3-TTS implementation for the same
    prompts and generation settings, so these tests assert that the port reproduces the reference
    exactly rather than merely reproducing itself. Regenerate them with:

        python reproduce_qwen3_tts_from_original.py \
            --original_repo ../Qwen3-TTS \
            --original_weights qwen3_tts_original \
            --processor_checkpoint qwen3_tts_converted

    The generation settings below must stay in step with `GENERATE_KWARGS` in that script: greedy
    decoding makes the codes reproducible, and the short horizon keeps them clear of the repeated
    tail that greedy decoding falls into on longer runs.
    """

    @classmethod
    def setUpClass(cls):
        from transformers.testing_utils import cleanup

        cleanup(torch_device, gc_collect=True)
        cls.checkpoint = "shahvandit/qwen3-tts-base-hf"

    def tearDown(self):
        from transformers.testing_utils import cleanup

        cleanup(torch_device, gc_collect=True)

    @slow
    def test_single(self):
        set_seed(42)

        path = Path(__file__).parent.parent.parent / "fixtures/qwen3_tts/expected_results_single.json"
        with open(path, "r", encoding="utf-8") as f:
            expected = json.load(f)

        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(self.checkpoint)
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype=torch.float32
        )

        formatted = _build_assistant_text(expected["input_text"])
        input_ids = [processor(text=formatted, return_tensors="pt")["input_ids"].to(torch_device)]

        torch.testing.assert_close(input_ids[0].cpu(), torch.tensor(expected["input_ids"]))

        with torch.no_grad():
            talker_codes_list, _ = model.generate(
                input_ids=input_ids,
                languages=["Auto"],
                do_sample=False,
                max_new_tokens=50,
                repetition_penalty=1.05,
                subtalker_dosample=False,
            )

        torch.testing.assert_close(
            talker_codes_list[0].cpu(),
            torch.tensor(expected["generated_codes"]),
        )

    @slow
    def test_batch(self):
        set_seed(42)

        path = Path(__file__).parent.parent.parent / "fixtures/qwen3_tts/expected_results_batch.json"
        with open(path, "r", encoding="utf-8") as f:
            expected = json.load(f)

        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(self.checkpoint)
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map=torch_device, dtype=torch.float32
        )

        input_ids = [
            processor(text=_build_assistant_text(t), return_tensors="pt")["input_ids"].to(torch_device)
            for t in expected["input_texts"]
        ]
        languages = ["Auto"] * len(expected["input_texts"])

        with torch.no_grad():
            talker_codes_list, _ = model.generate(
                input_ids=input_ids,
                languages=languages,
                do_sample=False,
                max_new_tokens=50,
                repetition_penalty=1.05,
                subtalker_dosample=False,
            )

        for i, exp_codes in enumerate(expected["generated_codes"]):
            torch.testing.assert_close(
                talker_codes_list[i].cpu(),
                torch.tensor(exp_codes),
            )
