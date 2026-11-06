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

import copy
import json
import tempfile
import unittest
from pathlib import Path

from transformers import D1OmniConfig, is_torch_available
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import (
        AutoConfig,
        AutoModelForDecision,
        D1OmniAudioEncoder,
        D1OmniForDecision,
        D1OmniModel,
        D1OmniTextModel,
        Trainer,
        TrainingArguments,
    )


def tiny_config():
    return D1OmniConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "layer_types": ["conv", "full_attention"],
            "block_multiple_of": 16,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_patches": 4,
            "patch_size": 16,
            "vision_use_head": False,
        },
        audio_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_mel_bins": 16,
            "subsampling_conv_channels": 8,
            "dropout": 0.0,
            "dropout_positions": 0.0,
            "layerdrop": 0.0,
            "activation_dropout": 0.0,
            "scale_input": False,
        },
        projector_hidden_size=32,
        audio_residual_width=16,
        head_layers=2,
    )


def reference_weights(model):
    """Deterministic random-like weights used to produce the publisher golden outputs."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            phase = sum(name.encode()) / 100
            values = (torch.arange(parameter.numel()).reshape(parameter.shape).float() * 0.13 + phase).sin() * 0.02
            if parameter.ndim == 1 and name.endswith("weight"):
                values += 1.0
            parameter.copy_(values)


def batch():
    return {
        "input_ids": torch.tensor([[1, 4, 5, 6, 7, 8], [1, 6, 8, 5, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool),
        "marker_positions": torch.tensor([[1, 3, 5], [1, 3, 0]]),
        "marker_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        "question_type_ids": torch.tensor([0, 2]),
    }


@require_torch
class D1OmniForDecisionTest(unittest.TestCase):
    all_model_classes = (
        (D1OmniForDecision, D1OmniModel, D1OmniTextModel, D1OmniAudioEncoder) if is_torch_available() else ()
    )

    def test_publisher_encoder_head_audio_and_gradients(self):
        fixture = json.loads(Path(__file__).with_name("publisher_reference.json").read_text())
        model = D1OmniForDecision(tiny_config()).eval()
        reference_weights(model)
        valid = torch.tensor(fixture["valid"], dtype=torch.bool)
        prefix = torch.tensor(fixture["prefix"])
        values = torch.tensor(fixture["inputs"], requires_grad=True)
        outputs = model.encoder(inputs_embeds=values, attention_mask=valid, media_lengths=prefix).last_hidden_state
        torch.testing.assert_close(outputs, torch.tensor(fixture["hidden"]), atol=3e-6, rtol=3e-5)
        lengths = valid.sum(-1) - prefix
        text = torch.nn.utils.rnn.pad_sequence(
            [row[o : o + n] for row, o, n in zip(outputs, prefix, lengths)], batch_first=True
        )
        text_mask = torch.arange(text.shape[1])[None] < lengths[:, None]
        logits = model.head(
            text,
            text_mask,
            torch.tensor(fixture["positions"]),
            torch.tensor(fixture["markers"], dtype=torch.bool),
            torch.tensor([0, 1, 2]),
        )
        torch.testing.assert_close(logits, torch.tensor(fixture["logits"]), atol=3e-6, rtol=3e-5)
        logits.backward(torch.tensor(fixture["upstream"]))
        torch.testing.assert_close(values.grad, torch.tensor(fixture["input_gradient"]), atol=3e-6, rtol=3e-5)
        for name, parameter in model.named_parameters():
            if name in fixture["gradient_stats"]:
                stats = torch.stack([parameter.grad.sum(), parameter.grad.square().sum()])
                torch.testing.assert_close(
                    stats, torch.tensor(fixture["gradient_stats"][name]), atol=3e-6, rtol=3e-5, msg=name
                )
        features = torch.tensor(fixture["audio_inputs"])
        mask = torch.arange(features.shape[1])[None] < torch.tensor(fixture["audio_lengths"])[:, None]
        audio = model.audio.encoder(features, attention_mask=mask).last_hidden_state
        torch.testing.assert_close(audio, torch.tensor(fixture["audio_outputs"]), atol=3e-6, rtol=3e-5)

    def test_base_model(self):
        model = D1OmniModel(tiny_config()).eval()
        inputs = batch()
        output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        self.assertEqual(output.last_hidden_state.shape, (2, 6, 64))
        self.assertTrue(torch.equal(output.attention_mask, inputs["attention_mask"]))
        output.last_hidden_state.sum().backward()
        self.assertGreater(model.get_input_embeddings().weight.grad.abs().sum(), 0)

    def test_native_loss_and_padding(self):
        model = D1OmniForDecision(tiny_config()).eval()
        inputs = batch()
        labels = torch.tensor([2, 0])
        output = model(**inputs, labels=labels)
        expected = torch.nn.functional.cross_entropy(
            output.logits.masked_fill(~inputs["marker_mask"], -torch.inf), labels
        )
        torch.testing.assert_close(output.loss, expected)
        self.assertEqual(model.loss_function.__name__, "D1OmniForDecisionLoss")
        self.assertEqual(output.logits[1, 2], -10000)
        output.loss.backward()
        self.assertGreater(model.encoder.embed_tokens.weight.grad.abs().sum(), 0)
        self.assertGreater(model.head.layers[0].self_attn.q_proj.weight.grad.abs().sum(), 0)
        ignored = model(**inputs, labels=torch.full((2,), -100))
        self.assertEqual(ignored.loss.item(), 0)
        self.assertTrue(torch.isfinite(ignored.loss))
        with self.assertRaises(ValueError):
            model(**inputs, labels=torch.tensor([0, 2]))

    def test_question_count_accumulation(self):
        model = D1OmniForDecision(tiny_config()).eval()
        other = copy.deepcopy(model)
        inputs = batch()
        labels = torch.tensor([2, 0])
        model(**inputs, labels=labels).loss.backward()
        for index in range(2):
            other(
                **{key: value[index : index + 1] for key, value in inputs.items()},
                labels=labels[index : index + 1],
                num_items_in_batch=2,
            ).loss.backward()
        for left, right in zip(model.parameters(), other.parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, atol=3e-7, rtol=3e-5)

    def test_save_load_and_publisher_conversion(self):
        model = D1OmniForDecision(tiny_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            # Test both native saving and the inverse publisher QKV/audio conversion.
            for original_format in (False, True):
                model.save_pretrained(directory, save_original_format=original_format)
                self.assertIsInstance(AutoConfig.from_pretrained(directory), D1OmniConfig)
                loaded, info = AutoModelForDecision.from_pretrained(directory, output_loading_info=True)
                self.assertFalse(info["missing_keys"])
                self.assertFalse(info["unexpected_keys"])
                torch.testing.assert_close(loaded(**batch()).logits, model(**batch()).logits)

    def test_media_prefix_and_input_validation(self):
        model = D1OmniForDecision(tiny_config()).eval()
        inputs = batch()
        prefix = torch.randn(2, 3, 64, requires_grad=True)
        output = model(**inputs, media_prefix=prefix, media_lengths=torch.tensor([3, 1]), labels=torch.tensor([0, 1]))
        output.loss.backward()
        self.assertGreater(prefix.grad.abs().sum(), 0)
        self.assertEqual(prefix.grad[1, 1:].abs().sum(), 0)
        inputs["marker_positions"][1, 1] = 4
        with self.assertRaises(ValueError):
            model(**inputs)

    def test_trainer_update(self):
        model = D1OmniForDecision(tiny_config())
        before = model.head.scorer.out_proj.weight.detach().clone()
        inputs = batch()
        dataset = [
            {**{key: value[index] for key, value in inputs.items()}, "labels": torch.tensor(index)}
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                train_dataset=dataset,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    use_cpu=True,
                    report_to=[],
                    disable_tqdm=True,
                ),
            )
            trainer.train()
        self.assertFalse(torch.equal(before, model.head.scorer.out_proj.weight))
