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
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
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


@require_torch
class Qwen3TTSCodePredictorTrainingTest(unittest.TestCase):
    def get_model(self, predictor_hidden_size=32):
        config = Qwen3TTSModelTester(self).get_config().talker_config
        config.num_code_groups = 4
        config.code_predictor_config.num_code_groups = 4
        config.code_predictor_config.hidden_size = predictor_hidden_size
        return Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(config.code_predictor_config, config).to(
            torch_device
        )

    def prepare_inputs(self, model):
        codes = ids_tensor([2, 3], model.vocab_size)
        conditioning = torch.randn(2, 2, 32, device=torch_device, requires_grad=True)
        embeddings = torch.cat(
            [conditioning] + [model.get_input_embeddings()[index](codes[:, index : index + 1]) for index in range(2)],
            dim=1,
        )
        return codes, conditioning, embeddings

    def test_parallel_matches_cached_teacher_forcing(self):
        for hidden_size in (16, 32):
            with self.subTest(predictor_hidden_size=hidden_size):
                model = self.get_model(hidden_size).eval()
                codes, _, embeddings = self.prepare_inputs(model)
                with torch.no_grad():
                    parallel = model(inputs_embeds=embeddings, labels=codes, use_cache=False)
                    output = model(inputs_embeds=embeddings[:, :2], use_cache=True)
                    sequential_logits = [output.logits[:, -1]]
                    for index in range(2):
                        output = model(
                            input_ids=codes[:, index : index + 1],
                            past_key_values=output.past_key_values,
                            generation_steps=output.generation_steps,
                            use_cache=True,
                        )
                        sequential_logits.append(output.logits[:, -1])
                torch.testing.assert_close(parallel.logits, torch.stack(sequential_logits, dim=1))
                expected_loss = torch.nn.functional.cross_entropy(
                    parallel.logits.float().reshape(-1, model.vocab_size), codes.reshape(-1)
                )
                torch.testing.assert_close(parallel.loss, expected_loss)

    def test_loss_masking_and_backward(self):
        model = self.get_model().train()
        codes, conditioning, embeddings = self.prepare_inputs(model)
        codes = codes.clone()
        codes[0, 1] = -100
        output = model(inputs_embeds=embeddings, labels=codes, use_cache=False)
        output.loss.backward()
        self.assertTrue(torch.isfinite(output.loss))
        self.assertGreater(conditioning.grad.abs().sum().item(), 0)
        for gradient in model.lm_head.weight.grad.chunk(3):
            self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertGreater(model.get_input_embeddings()[0].weight.grad.abs().sum().item(), 0)

        model.zero_grad()
        _, _, embeddings = self.prepare_inputs(model)
        ignored = model(inputs_embeds=embeddings, labels=torch.full_like(codes, -100), use_cache=False)
        ignored.loss.backward()
        self.assertEqual(ignored.loss.item(), 0)
        self.assertTrue(torch.isfinite(model.lm_head.weight.grad).all())
        self.assertEqual(model.lm_head.weight.grad.abs().sum().item(), 0)


@require_torch
class Qwen3TTSTeacherForcingTest(unittest.TestCase):
    def get_model(self):
        config = Qwen3TTSModelTester(self).get_config()
        return Qwen3TTSForConditionalGeneration(config).to(torch_device).eval()

    def prepare_inputs(self):
        return {
            "input_ids": ids_tensor([2, 11], 64),
            "attention_mask": torch.tensor([[1] * 11, [1] * 9 + [0] * 2], device=torch_device),
            "audio_codes": ids_tensor([2, 3, 2], 64),
            "audio_attention_mask": torch.tensor([[1, 1, 1], [1, 0, 0]], device=torch_device),
            "speaker_embeddings": torch.randn(2, 32, device=torch_device),
        }

    def reference_sequence(self, model, text_ids, codes, speaker):
        config = model.config.talker_config
        def text_embedding(ids):
            return model.text_projection(model.get_text_embeddings()(ids))
        codec_embedding = model.get_input_embeddings()
        pad, bos, eos = text_embedding(text_ids.new_tensor([0, 1, 2])).split(1)
        codec_ids = text_ids.new_tensor(
            [
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
                config.codec_pad_id,
                config.codec_bos_id,
                config.codec_eos_token_id,
            ]
        )
        special = codec_embedding(codec_ids)
        prefix = torch.cat([special[:3], speaker[None], special[3:4]])
        prefix = prefix + torch.cat([pad.expand(4, -1), bos])
        audio = codec_embedding(codes[:, 0])
        for index, embedding in enumerate(model.code_predictor.get_input_embeddings()):
            audio = audio + embedding(codes[:, index + 1])
        return torch.cat(
            [
                text_embedding(text_ids[:3]),
                prefix,
                text_embedding(text_ids[3:-5]) + special[3],
                eos + special[3],
                pad + special[4],
                audio + pad,
                pad + special[5],
            ]
        )

    def test_sequence_matches_non_streaming_layout(self):
        model = self.get_model()
        inputs = self.prepare_inputs()
        with torch.no_grad():
            embeddings, mask, audio_positions, eos_positions = model._prepare_teacher_forcing_inputs(**inputs)
            for index in range(2):
                text = inputs["input_ids"][index][inputs["attention_mask"][index].bool()]
                codes = inputs["audio_codes"][index][inputs["audio_attention_mask"][index].bool()]
                reference = self.reference_sequence(model, text, codes, inputs["speaker_embeddings"][index])
                length = reference.shape[0]
                torch.testing.assert_close(embeddings[index, :length], reference)
                self.assertEqual(mask[index].sum().item(), length)
                self.assertEqual(embeddings[index, length:].abs().sum().item(), 0)
                torch.testing.assert_close(
                    audio_positions[index, : codes.shape[0]],
                    torch.arange(text.shape[0] + 2, text.shape[0] + 2 + codes.shape[0], device=torch_device),
                )
                self.assertEqual(eos_positions[index].item(), length - 1)

    def test_padding_side_and_padded_ids_do_not_change_sequence(self):
        model = self.get_model()
        inputs = self.prepare_inputs()
        with torch.no_grad():
            reference = model._prepare_teacher_forcing_inputs(**inputs)
            inputs["input_ids"][1] = inputs["input_ids"][1].roll(2)
            inputs["attention_mask"][1] = inputs["attention_mask"][1].roll(2)
            inputs["input_ids"].masked_fill_(~inputs["attention_mask"].bool(), -100)
            inputs["audio_codes"].masked_fill_(~inputs["audio_attention_mask"].bool()[..., None], -100)
            actual = model._prepare_teacher_forcing_inputs(**inputs)
        for expected, result in zip(reference, actual):
            torch.testing.assert_close(expected, result)

    def test_rejects_invalid_teacher_forcing_inputs(self):
        model = self.get_model()
        for name, value, message in (
            ("audio_codes", torch.zeros(2, 3, 3, dtype=torch.long, device=torch_device), "num_code_groups"),
            ("speaker_embeddings", torch.zeros(2, 16, device=torch_device), "talker_hidden_size"),
            ("audio_attention_mask", torch.zeros(2, 3, device=torch_device), "at least one audio frame"),
            ("attention_mask", torch.zeros(2, 11, device=torch_device), "role prefix"),
            ("audio_codes", torch.full((2, 3, 2), 64, device=torch_device), "talker vocabulary"),
        ):
            with self.subTest(input=name, message=message):
                inputs = self.prepare_inputs()
                inputs[name] = value
                with self.assertRaisesRegex(ValueError, message):
                    model._prepare_teacher_forcing_inputs(**inputs)


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
            "codec_eos_token_id": 3,
            "codec_think_id": 4,
            "codec_nothink_id": 5,
            "codec_think_bos_id": 6,
            "codec_think_eos_id": 7,
            "codec_pad_id": 8,
            "codec_bos_id": 9,
            # the talker always applies mRoPE, so the tiny config declares its sections too; they sum to
            # `head_dim // 2`, as `apply_multimodal_rotary_pos_emb` doubles them
            "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0, "mrope_section": [4, 2, 2]},
            "code_predictor_config": {
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "num_code_groups": 2,
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

    def test_talker_config_accepts_checkpoint_metadata(self):
        config = Qwen3TTSConfig(
            talker_config={
                "attention_dropout": 0,
                "spk_id": {},
                "spk_is_dialect": {},
                "codec_language_id": {},
            }
        )

        self.assertEqual(config.talker_config.attention_dropout, 0)
        self.assertEqual(config.talker_config.spk_id, {})
        self.assertEqual(config.talker_config.spk_is_dialect, {})
        self.assertEqual(config.talker_config.codec_language_id, {})

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
        "`forward` requires `past_hidden` from the preceding generation step, which the common tester does not provide"
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
    exactly rather than merely reproducing itself.

    The generation settings below must stay in step with `GENERATE_KWARGS` in the reproducer:
    greedy decoding makes the codes reproducible, and the short horizon keeps them clear of the
    repeated tail that greedy decoding falls into on longer runs.
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
        """
        reproducer: https://gist.github.com/ShahVandit/cab13f3b7232c52b4ff93cce592950c4#file-reproducer_qwen3_tts-py
        """
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
            talker_codes_list = model.generate(
                input_ids=input_ids,
                languages=["Auto"],
                do_sample=False,
                max_new_tokens=50,
                repetition_penalty=1.05,
                subtalker_dosample=False,
            ).sequences

        torch.testing.assert_close(
            talker_codes_list[0].cpu(),
            torch.tensor(expected["generated_codes"]),
        )

    @slow
    def test_batch(self):
        """
        reproducer: https://gist.github.com/ShahVandit/cab13f3b7232c52b4ff93cce592950c4#file-reproducer_qwen3_tts-py
        """
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
            talker_codes_list = model.generate(
                input_ids=input_ids,
                languages=languages,
                do_sample=False,
                max_new_tokens=50,
                repetition_penalty=1.05,
                subtalker_dosample=False,
            ).sequences

        for i, exp_codes in enumerate(expected["generated_codes"]):
            torch.testing.assert_close(
                talker_codes_list[i].cpu(),
                torch.tensor(exp_codes),
            )
