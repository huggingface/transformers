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

import tempfile
import unittest
from pathlib import Path

import yaml

from transformers import AutoConfig, AutoTokenizer, JevaConfig, is_torch_available, pipeline
from transformers.testing_utils import require_peft, require_torch

from ...test_decision_head_common import render, tiny_qwen2


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import AutoModelForDecision, JevaForDecision, Trainer, TrainingArguments
    from transformers.models.jeva.modeling_jeva import _log_softmax


@require_torch
class JevaForDecisionTest(unittest.TestCase):
    all_model_classes = (JevaForDecision,) if is_torch_available() else ()

    def setUp(self):
        self.config = JevaConfig(
            text_config=AutoConfig.for_model(
                "qwen2",
                vocab_size=128,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
            ),
            projection_dim=16,
            decision_num_heads=2,
        )
        self.inputs = {
            "input_ids": torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
            "option_input_ids": torch.tensor([[2, 3, 0], [4, 5, 6], [7, 8, 9], [3, 4, 5], [6, 7, 0]]),
            "option_attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0]]),
            "num_options": torch.tensor([2, 3]),
            "option_mask": torch.tensor([[True, True, False], [True, True, True]]),
            "labels": torch.tensor([[0.2, 0.3, 0.5, 0.0], [0.0, 0.1, 0.3, 0.6]]),
            "allow_null": torch.tensor([True, False]),
        }

    def test_loss_and_save_load(self):
        for model_type in ("qwen2", "llama"):
            with self.subTest(model_type=model_type):
                self.config.text_config = AutoConfig.for_model(
                    model_type, **{k: v for k, v in self.config.text_config.to_dict().items() if k != "model_type"}
                )
                model = JevaForDecision(self.config).eval()
                output = model(**self.inputs)
                candidates = self.inputs["labels"][:, 1:]
                candidate_loss = -(candidates * output.logits.masked_fill(~self.inputs["option_mask"], 0)).sum(-1)
                coverage_loss = F.binary_cross_entropy_with_logits(
                    output.coverage_logits, candidates.sum(-1), reduction="none"
                )
                expected = (candidate_loss + coverage_loss * self.inputs["allow_null"]).mean()
                torch.testing.assert_close(output.loss, expected)
                torch.testing.assert_close(output.logits.exp().sum(-1), torch.ones(2))
                self.assertTrue(torch.isneginf(output.logits[0, 2]))
                output.loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assertGreater(model.core.tau_cov.raw.grad.abs().sum().item(), 0)
                with tempfile.TemporaryDirectory() as tmp:
                    model.save_pretrained(tmp)
                    loaded = AutoModelForDecision.from_pretrained(tmp).eval()
                result = loaded(**self.inputs)
                torch.testing.assert_close(result.loss, output.loss)
                torch.testing.assert_close(result.logits, output.logits)
                torch.testing.assert_close(result.coverage_logits, output.coverage_logits)
                self.assertIsNone(model(**{k: v for k, v in self.inputs.items() if k != "labels"}).loss)
                torch.testing.assert_close(model(**self.inputs, return_dict=False)[0], output.loss)

    def test_chunking_and_accumulation(self):
        model = JevaForDecision(self.config).eval()
        output = model(**self.inputs)
        chunked = model(**self.inputs, option_chunk_size=2)
        torch.testing.assert_close(chunked.loss, output.loss)
        output.loss.backward()
        reference = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad()
        offset = 0
        for index, count in enumerate(self.inputs["num_options"].tolist()):
            sample = {
                key: value[offset : offset + count]
                if key in {"option_input_ids", "option_attention_mask"}
                else value[index : index + 1]
                for key, value in self.inputs.items()
            }
            model(**sample, num_items_in_batch=2).loss.backward()
            offset += count
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter.grad, reference[name], rtol=1e-4, atol=1e-5)

    def test_trainer(self):
        model = JevaForDecision(self.config)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=tmp,
                    use_cpu=True,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    report_to="none",
                    remove_unused_columns=False,
                    save_strategy="no",
                ),
                train_dataset=[0, 1],
                data_collator=lambda features: self.inputs,
            )
            result = trainer.train()
        self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
        self.assertEqual(trainer.state.global_step, 1)

    def test_decision_pipeline(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        encoder = tiny_qwen2(tokenizer).model
        config = JevaConfig(text_config=encoder.config, projection_dim=16, decision_num_heads=2)
        model = JevaForDecision(config, model=encoder).eval()
        request = {
            "state": "My card has been charged twice.",
            "questions": {
                "intent": {
                    "type": "choice",
                    "instructions": "Which intent?",
                    "criteria": {"refund": None, "delivery": None},
                }
            },
        }
        prompt = render(tokenizer, request, model.decision_config)[0]
        questions = prompt.questions
        scores = model.decide(**prompt.inputs, decision_questions=questions)
        states = tokenizer(
            ["Decision type: choice\nQuestion: Which intent?\nState:\n" + request["state"]],
            padding=True,
            return_tensors="pt",
        )
        options = tokenizer(
            ["Option for choice: refund", "Option for choice: delivery"], padding=True, return_tensors="pt"
        )
        direct = model(
            **states,
            **{f"option_{key}": value for key, value in options.items()},
            num_options=torch.tensor([2]),
            option_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        torch.testing.assert_close(scores[0], direct.logits[0])
        expected = tokenizer.decode_decision(scores, questions)
        decide = pipeline("decision", model=model, tokenizer=tokenizer, device="cpu")
        self.assertEqual(decide(request), expected)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            tokenizer.save_pretrained(tmp)
            decide = pipeline("decision", model=tmp, model_kwargs={"attn_implementation": "eager"}, device="cpu")
            answer = decide(request)
            self.assertEqual(answer["intent"]["choice"], expected["intent"]["choice"])
            torch.testing.assert_close(answer["intent"]["probabilities"], expected["intent"]["probabilities"])

    def test_typed_decisions(self):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        encoder = tiny_qwen2(tokenizer).model
        model = JevaForDecision(
            JevaConfig(text_config=encoder.config, projection_dim=16, decision_num_heads=2), model=encoder
        ).eval()
        request = {
            "state": "A refund request",
            "questions": {
                "intent": {"type": "choice", "criteria": {str(i): "An intent" for i in range(32)}, "allow_null": True},
                "truth": {"type": "noul", "instructions": "Is this a refund?"},
                "quality": {"type": "score", "criteria": ["bad", "good"], "values": [1, 5]},
            },
        }
        prompts = render(tokenizer, request, model.decision_config)
        scores = [model.decide(**prompt.inputs, decision_questions=prompt.questions)[0] for prompt in prompts]
        self.assertEqual([len(score) for score in scores], [33, 2, 2])
        answers = pipeline("decision", model=model, tokenizer=tokenizer, device="cpu")(request)
        intent = answers["intent"]
        self.assertAlmostEqual(sum(intent["probabilities"].values()) + intent["null_probability"], 1, places=6)
        best = max(intent["probabilities"], key=intent["probabilities"].get)
        self.assertEqual(
            intent["choice"], None if intent["null_probability"] >= intent["probabilities"][best] else best
        )
        probabilities = scores[2].softmax(-1)
        self.assertAlmostEqual(
            answers["quality"]["score"], (probabilities * torch.tensor([1, 5])).sum().item(), places=6
        )
        self.assertAlmostEqual(answers["truth"]["noul"], scores[1].softmax(-1)[0].item(), places=6)

    def test_router_aux_loss(self):
        self.config.text_config = AutoConfig.for_model(
            "qwen3_moe",
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            moe_intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=2,
            num_experts_per_tok=1,
        )
        self.config.router_aux_loss_coef = 0.1
        model = JevaForDecision(self.config)
        output = model(**self.inputs, option_chunk_size=2)
        self.assertGreater(output.aux_loss.item(), 0)
        output.loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    @require_peft
    def test_checkpoint_conversion(self):
        from peft import LoraConfig, get_peft_model

        from transformers.models.jeva.convert_jeva_weights_to_hf import convert_checkpoint

        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        model = JevaForDecision(self.config).eval()
        with tempfile.TemporaryDirectory() as checkpoint, tempfile.TemporaryDirectory() as converted:
            path = Path(checkpoint)
            model.model.save_pretrained(path / "backbone")
            tokenizer.save_pretrained(path / "tokenizer")
            model.model = get_peft_model(model.model, LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))
            for name, parameter in model.model.named_parameters():
                if "lora_B" in name:
                    torch.nn.init.normal_(parameter, std=0.01)
            model.model.save_pretrained(path / "backbone_adapter")
            settings = {"projection_dim": 16, "decision_num_heads": 2}
            (path / "config.yaml").write_text(yaml.safe_dump({"model": settings}))
            head = {
                name: {
                    key: value.squeeze() if value.numel() == 1 else value
                    for key, value in model.get_submodule(name).state_dict().items()
                }
                for name in ("context_proj", "option_proj", "core")
            }
            torch.save(head, path / "head.pt")
            expected = model(**self.inputs)
            convert_checkpoint(checkpoint, converted)
            loaded = AutoModelForDecision.from_pretrained(converted).eval()
            output = loaded(**self.inputs)
        torch.testing.assert_close(output.logits, expected.logits)
        torch.testing.assert_close(output.coverage_logits, expected.coverage_logits)
        torch.testing.assert_close(output.loss, expected.loss)

    def test_confident_gradients(self):
        logits = torch.tensor([[9.0, -9.0, -9.0], [2.0, 1.0, -torch.inf]], requires_grad=True)
        reference = logits.detach().double().requires_grad_(True)
        (-_log_softmax(logits)[:, 0].sum()).backward()
        (-reference.log_softmax(-1)[:, 0].sum()).backward()
        torch.testing.assert_close(logits.grad.double(), reference.grad, rtol=1e-6, atol=1e-14)
        self.assertLess(logits.grad[0, 0].item(), 0)
