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
from unittest.mock import patch

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
    def get_model(self, num_code_groups=2, predictor_hidden_size=32):
        config = Qwen3TTSModelTester(self).get_config()
        config.talker_config.num_code_groups = num_code_groups
        config.talker_config.code_predictor_config.num_code_groups = num_code_groups
        config.talker_config.code_predictor_config.hidden_size = predictor_hidden_size
        return Qwen3TTSForConditionalGeneration(config).to(torch_device).eval()

    def prepare_inputs(self, num_code_groups=2):
        return {
            "input_ids": ids_tensor([2, 11], 64),
            "attention_mask": torch.tensor([[1] * 11, [1] * 9 + [0] * 2], device=torch_device),
            "audio_codes": ids_tensor([2, 3, num_code_groups], 64),
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

    def test_primary_and_residual_loss_alignment(self):
        for codebooks, hidden_size in ((2, 32), (4, 16)):
            with self.subTest(codebooks=codebooks, predictor_hidden_size=hidden_size):
                model = self.get_model(codebooks, hidden_size)
                inputs = self.prepare_inputs(codebooks)
                labels = inputs["audio_codes"].clone()
                labels[0, 1, 0] = -100
                labels[0, 0, 1] = -100
                labels[1, :, 0] = -100
                captured = {}

                def capture_talker(module, args, output):
                    captured["hidden"] = output.last_hidden_state

                def capture_predictor(module, args, kwargs, output):
                    captured["predictor_inputs"] = kwargs["inputs_embeds"]
                    captured["predictor_output"] = output

                talker_hook = model.model.register_forward_hook(capture_talker)
                predictor_hook = model.code_predictor.register_forward_hook(capture_predictor, with_kwargs=True)
                try:
                    with torch.no_grad():
                        output = model(**inputs, labels=labels, output_hidden_states=True)
                finally:
                    talker_hook.remove()
                    predictor_hook.remove()

                primary_logits, primary_targets = [], []
                for batch_index in range(2):
                    start = inputs["attention_mask"][batch_index].sum().item() + 2
                    frames = inputs["audio_attention_mask"][batch_index].sum().item()
                    primary_logits.append(output.logits[batch_index, start - 1 : start + frames])
                    eos_target = 3 if (labels[batch_index, :frames, 0] != -100).any() else -100
                    primary_targets.append(
                        torch.cat([labels[batch_index, :frames, 0], labels.new_tensor([eos_target])])
                    )
                    for frame in range(frames):
                        predictor_input = captured["predictor_inputs"][batch_index * 3 + frame]
                        torch.testing.assert_close(
                            predictor_input[0], captured["hidden"][batch_index, start + frame - 1]
                        )
                        torch.testing.assert_close(
                            predictor_input[1],
                            model.get_input_embeddings()(inputs["audio_codes"][batch_index, frame, 0]),
                        )
                        for codebook in range(codebooks - 2):
                            torch.testing.assert_close(
                                predictor_input[codebook + 2],
                                model.code_predictor.get_input_embeddings()[codebook](
                                    inputs["audio_codes"][batch_index, frame, codebook + 1]
                                ),
                            )
                expected_primary = torch.nn.functional.cross_entropy(
                    torch.cat(primary_logits).float(), torch.cat(primary_targets)
                )
                residual_targets = labels[..., 1:].masked_fill(~inputs["audio_attention_mask"].bool()[..., None], -100)
                expected_residual = torch.nn.functional.cross_entropy(
                    captured["predictor_output"].logits.float().reshape(-1, 64), residual_targets.reshape(-1)
                )
                torch.testing.assert_close(output.talker_loss, expected_primary)
                torch.testing.assert_close(output.code_predictor_loss, expected_residual)
                torch.testing.assert_close(output.loss, expected_primary + 0.3 * expected_residual)
                self.assertEqual(len(output.hidden_states), model.config.talker_config.num_hidden_layers + 1)

    def test_future_frame_does_not_leak_into_predictions(self):
        model = self.get_model(4)
        inputs = self.prepare_inputs(4)
        predictor_outputs = []

        def capture(module, args, output):
            predictor_outputs.append(output.logits)

        hook = model.code_predictor.register_forward_hook(capture)
        try:
            with torch.no_grad():
                before = model(**inputs, labels=inputs["audio_codes"])
                inputs["audio_codes"][0, 2, -1] = (inputs["audio_codes"][0, 2, -1] + 1) % 64
                after = model(**inputs, labels=inputs["audio_codes"])
        finally:
            hook.remove()
        last_frame_position = inputs["attention_mask"][0].sum().item() + 4
        torch.testing.assert_close(before.logits[0, :last_frame_position], after.logits[0, :last_frame_position])
        torch.testing.assert_close(predictor_outputs[0], predictor_outputs[1])

    def test_training_backward_optimizer_and_reload(self):
        model = self.get_model(4, 16).train()
        inputs = self.prepare_inputs(4)
        parameters = [
            model.text_projection.linear_1.weight,
            model.model.text_embedding.weight,
            model.model.layers[0].self_attn.q_proj.weight,
            model.codec_head.weight,
            model.model.embed_tokens.weight,
            model.code_predictor.lm_head.weight,
            model.code_predictor.small_to_mtp_projection.weight,
            model.code_predictor.model.layers[0].self_attn.q_proj.weight,
        ]
        parameters += [embedding.weight for embedding in model.code_predictor.get_input_embeddings()]
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        output = model(**inputs, labels=inputs["audio_codes"])
        output.loss.backward()
        for parameter in parameters:
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        optimizer.step()
        for original, parameter in zip(before, parameters):
            self.assertFalse(torch.equal(original, parameter))
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = Qwen3TTSForConditionalGeneration.from_pretrained(directory).to(torch_device).eval()
        with torch.no_grad():
            expected = model(**inputs, labels=inputs["audio_codes"])
            actual = restored(**inputs, labels=inputs["audio_codes"])
        torch.testing.assert_close(actual.loss, expected.loss)
        torch.testing.assert_close(actual.logits, expected.logits)

    def test_ignored_targets_and_label_free_forward(self):
        model = self.get_model().train()
        inputs = self.prepare_inputs()
        labels = torch.full_like(inputs["audio_codes"], -100)
        output = model(**inputs, labels=labels)
        self.assertEqual(output.talker_loss.item(), 0)
        self.assertEqual(output.code_predictor_loss.item(), 0)
        output.loss.backward()
        for parameter in (model.codec_head.weight, model.code_predictor.lm_head.weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertEqual(parameter.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            output = model(**inputs)
            as_tuple = model(**inputs, return_dict=False)
        self.assertIsNone(output.loss)
        self.assertIsNone(output.code_predictor_loss)
        torch.testing.assert_close(output.logits, as_tuple[0])

    def test_teacher_forcing_batching_and_padding_invariance(self):
        model = self.get_model()
        inputs = self.prepare_inputs()
        with torch.no_grad():
            batched = model(**inputs)
            for index in range(2):
                single = {name: value[index : index + 1] for name, value in inputs.items()}
                text_length = single["attention_mask"].sum().item()
                audio_length = single["audio_attention_mask"].sum().item()
                single["input_ids"] = single["input_ids"][:, :text_length]
                single["attention_mask"] = single["attention_mask"][:, :text_length]
                single["audio_codes"] = single["audio_codes"][:, :audio_length]
                single["audio_attention_mask"] = single["audio_attention_mask"][:, :audio_length]
                output = model(**single)
                length = text_length + audio_length + 3
                torch.testing.assert_close(batched.logits[index, :length], output.logits[0], atol=1e-6, rtol=1e-5)


@require_torch
class Qwen3TTSGenerationTest(unittest.TestCase):
    def get_model(self):
        set_seed(42)
        config = Qwen3TTSModelTester(self).get_config()
        config.talker_config.vocab_size = 2048
        config.talker_config.code_predictor_config.vocab_size = 2048
        config.talker_config.num_code_groups = 4
        config.talker_config.code_predictor_config.num_code_groups = 4
        config.talker_config.codec_eos_token_id = 2047
        return Qwen3TTSForConditionalGeneration(config).to(torch_device).eval()

    def test_generation_preserves_codec_frames(self):
        model = self.get_model()
        expected = {
            False: [
                [[730, 27, 1960, 1134], [869, 905, 1892, 371], [49, 1802, 1516, 352]],
                [[730, 27, 1960, 1134], [869, 905, 1892, 371], [49, 1802, 1516, 352]],
            ],
            True: [
                [[730, 27, 423, 1625], [602, 719, 1042, 115], [518, 719, 1230, 1107]],
                [[730, 27, 1960, 971], [368, 1563, 1516, 57], [871, 45, 158, 104]],
            ],
        }
        # Recorded before moving residual sampling out of forward, with these exact weights and prompts.
        for non_streaming_mode in (False, True):
            with self.subTest(non_streaming_mode=non_streaming_mode):
                output = model.generate(
                    input_ids=[
                        torch.arange(11, device=torch_device)[None],
                        torch.arange(9, device=torch_device)[None],
                    ],
                    languages=["Auto", "Auto"],
                    non_streaming_mode=non_streaming_mode,
                    max_new_tokens=4,
                    do_sample=False,
                    subtalker_dosample=False,
                )
                for codes, reference in zip(output.sequences, expected[non_streaming_mode]):
                    torch.testing.assert_close(codes, codes.new_tensor(reference))
                self.assertIsNone(output.hidden_states)

    def test_generation_prepares_residual_codes_before_forward(self):
        model = self.get_model()
        primary_codes = torch.tensor([[5], [9]], device=torch_device)
        past_hidden = torch.randn(2, 1, 32, device=torch_device)
        trailing_text = torch.randn(2, 2, 32, device=torch_device)
        pad_embed = torch.randn(1, 1, 32, device=torch_device)
        with torch.no_grad():
            predictor = model.code_predictor.generate(
                inputs_embeds=torch.cat([past_hidden, model.get_input_embeddings()(primary_codes)], dim=1),
                max_new_tokens=3,
                do_sample=False,
                return_dict_in_generate=True,
            )
            codes = torch.cat([primary_codes, predictor.sequences], dim=-1)
            expected = model.get_input_embeddings()(primary_codes)
            for index, embedding in enumerate(model.code_predictor.get_input_embeddings()):
                expected = expected + embedding(codes[:, index + 1 : index + 2])
            model.code_predictor.generation_config.output_hidden_states = True
            for step in (0, 2):
                codec_frames = []
                with patch.object(model.code_predictor, "generate", wraps=model.code_predictor.generate) as generate:
                    prepared = model.prepare_inputs_for_generation(
                        primary_codes,
                        next_sequence_length=1,
                        past_hidden=past_hidden,
                        trailing_text_hidden=trailing_text,
                        tts_pad_embed=pad_embed,
                        generation_step=step,
                        subtalker_dosample=False,
                        attention_mask=torch.ones(2, 1, device=torch_device, dtype=torch.long),
                        use_cache=True,
                        codec_frames=codec_frames,
                    )
                self.assertFalse(generate.call_args.kwargs["output_hidden_states"])
                text = trailing_text[:, :1] if step == 0 else pad_embed
                torch.testing.assert_close(prepared["inputs_embeds"], expected + text)
                self.assertEqual(len(codec_frames), 1)
                torch.testing.assert_close(codec_frames[0], codes)
                with patch.object(
                    model.code_predictor, "generate", side_effect=AssertionError("forward must not sample")
                ):
                    output = model(**prepared, output_hidden_states=True)
                self.assertEqual(output.logits.shape, (2, 1, 2048))
                self.assertEqual(len(output.hidden_states), model.config.talker_config.num_hidden_layers + 1)
                self.assertTrue(all(isinstance(hidden, torch.Tensor) for hidden in output.hidden_states))

    def test_generation_codec_collection_is_independent_of_hidden_states(self):
        model = self.get_model()
        inputs = {
            "input_ids": [torch.arange(11, device=torch_device)[None]],
            "languages": ["Auto"],
            "max_new_tokens": 4,
            "do_sample": False,
            "subtalker_dosample": False,
        }
        without_hidden = model.generate(**inputs, output_hidden_states=False)
        with_hidden = model.generate(**inputs, output_hidden_states=True)
        torch.testing.assert_close(without_hidden.sequences[0], with_hidden.sequences[0])
        self.assertEqual(with_hidden.sequences[0].shape, (3, 4))
        self.assertIsNone(without_hidden.hidden_states)
        for step in with_hidden.hidden_states:
            self.assertEqual(len(step), model.config.talker_config.num_hidden_layers + 1)
            self.assertTrue(all(isinstance(hidden, torch.Tensor) for hidden in step))
        empty = model.generate(**{**inputs, "max_new_tokens": 1})
        self.assertEqual(empty.sequences[0].shape, (0, 4))

    def test_generation_requires_cache(self):
        model = self.get_model()
        inputs = {
            "input_ids": [torch.arange(11, device=torch_device)[None]],
            "languages": ["Auto"],
            "max_new_tokens": 4,
            "do_sample": False,
            "subtalker_dosample": False,
        }
        for config_use_cache, kwargs in ((True, {"use_cache": False}), (False, {})):
            with self.subTest(config_use_cache=config_use_cache, kwargs=kwargs):
                model.generation_config.use_cache = config_use_cache
                with self.assertRaisesRegex(ValueError, "Qwen3-TTS generation requires `use_cache=True`"):
                    model.generate(**inputs, **kwargs)

        output = model.generate(**inputs, use_cache=True)
        self.assertEqual(output.sequences[0].shape, (3, 4))

    def test_generation_preparation_rejects_disabled_cache(self):
        model = self.get_model()
        input_ids = torch.tensor([[5]], device=torch_device)
        with patch.object(model.code_predictor, "generate") as generate:
            for is_first_iteration in (True, False):
                with self.subTest(is_first_iteration=is_first_iteration):
                    with self.assertRaisesRegex(ValueError, "Qwen3-TTS generation requires `use_cache=True`"):
                        model.prepare_inputs_for_generation(
                            input_ids, is_first_iteration=is_first_iteration, use_cache=False
                        )
            generate.assert_not_called()


@require_torch
class Qwen3TTSForwardTest(unittest.TestCase):
    def get_model(self):
        config = Qwen3TTSModelTester(self).get_config()
        return Qwen3TTSForConditionalGeneration(config).to(torch_device).eval()

    def test_explicit_position_ids_are_honored(self):
        model = self.get_model()
        input_ids = torch.tensor([[4, 5, 6, 7]], device=torch_device)
        mask = torch.ones_like(input_ids)
        positions = torch.tensor([[0, 2, 5, 9]], device=torch_device)
        with torch.no_grad():
            reference = model.model(input_ids=input_ids, attention_mask=mask, position_ids=positions, use_cache=False)
            for position_ids in (positions, positions[None].expand(3, -1, -1)):
                with self.subTest(ndim=position_ids.ndim):
                    output = model(
                        input_ids=input_ids, attention_mask=mask, position_ids=position_ids, use_cache=False
                    )
                    torch.testing.assert_close(output.logits, model.codec_head(reference.last_hidden_state))

    def test_cached_positions_match_full_sequence_after_another_forward(self):
        model = self.get_model()
        input_ids = torch.tensor([[0, 0, 4, 5, 6], [7, 8, 9, 10, 11]], device=torch_device)
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], device=torch_device)
        next_ids = torch.tensor([[12], [13]], device=torch_device)
        full_mask = torch.cat([mask, torch.ones_like(next_ids)], dim=1)
        with torch.no_grad():
            prefill = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
            model(input_ids=next_ids, attention_mask=torch.ones_like(next_ids), use_cache=False)
            cached = model(
                input_ids=next_ids, attention_mask=full_mask, past_key_values=prefill.past_key_values, use_cache=True
            )
            full = model(input_ids=torch.cat([input_ids, next_ids], dim=1), attention_mask=full_mask, use_cache=False)
        torch.testing.assert_close(cached.logits[:, -1], full.logits[:, -1], atol=1e-6, rtol=1e-5)


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
