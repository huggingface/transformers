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

from types import SimpleNamespace
from unittest.mock import patch

import torch
from parameterized import parameterized

from transformers import LlamaConfig, LlamaForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM, Trainer, TrainingArguments
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.testing_utils import TestCasePlus, require_torch


@require_torch
class TrainerMoELossTest(TestCasePlus):
    def get_trainer(self, *, dense=False, compute_loss_func=None, **args):
        torch.manual_seed(17)
        config_args = {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "max_position_embeddings": 128,
            "use_cache": False,
            "pad_token_id": 0,
        }
        if dense:
            model = LlamaForCausalLM(LlamaConfig(**config_args))
        else:
            model = Qwen3MoeForCausalLM(
                Qwen3MoeConfig(
                    **config_args,
                    moe_intermediate_size=64,
                    num_experts=4,
                    num_experts_per_tok=1,
                    router_aux_loss_coef=0.01,
                    output_router_logits=True,
                )
            )
        return Trainer(
            model=model,
            args=TrainingArguments(output_dir=self.get_auto_remove_tmp_dir(), use_cpu=True, report_to="none", **args),
            compute_loss_func=compute_loss_func,
        )

    def check_partitioned_loss(self, trainer, *, tp_size=1, **forward_kwargs):
        # Each half has the same routing distribution, so the local auxiliary
        # losses should match the full batch without any distributed reduction.
        rows = [(torch.arange(9) * (3 + i) + 92 + i * 11) % 125 + 3 for i in range(4)]
        input_ids = torch.stack(rows).repeat_interleave(2, dim=0)
        batch = {"input_ids": input_ids, "labels": input_ids.clone(), "attention_mask": torch.ones_like(input_ids)}

        def loss_and_gradients(world_size):
            trainer.model.zero_grad(set_to_none=True)
            inputs = {key: value[::world_size] for key, value in batch.items()}
            with patch.object(trainer.accelerator.state, "num_processes", world_size * tp_size):
                loss = trainer.compute_loss(trainer.model, {**inputs, **forward_kwargs}, num_items_in_batch=64)
            loss.backward()
            gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in trainer.model.named_parameters()
                if parameter.grad is not None
            }
            return loss.detach(), gradients

        full_loss, full_gradients = loss_and_gradients(1)
        local_loss, local_gradients = loss_and_gradients(2)
        torch.testing.assert_close(local_loss, full_loss)
        self.assertEqual(local_gradients.keys(), full_gradients.keys())
        for name in full_gradients:
            with self.subTest(parameter=name):
                torch.testing.assert_close(local_gradients[name], full_gradients[name], atol=1e-6, rtol=1e-4)

    @parameterized.expand(
        [
            ("model_output", {}),
            ("tuple", {"return_dict": False}),
            ("tuple_default_router_flag", {"return_dict": False, "output_router_logits": None}),
            ("router_disabled", {"output_router_logits": False}),
            ("tuple_router_disabled", {"return_dict": False, "output_router_logits": False}),
        ]
    )
    def test_model_loss_matches_full_batch(self, _, forward_kwargs):
        self.check_partitioned_loss(self.get_trainer(), **forward_kwargs)

    @parameterized.expand([(0.0,), (0.07,)])
    def test_uses_model_auxiliary_coefficient(self, coefficient):
        trainer = self.get_trainer()
        trainer.model.router_aux_loss_coef = coefficient
        self.check_partitioned_loss(trainer)

    def test_custom_loss(self):
        def compute_loss(outputs, labels, num_items_in_batch):
            return ForCausalLMLoss(outputs.logits, labels, vocab_size=128, num_items_in_batch=num_items_in_batch)

        self.check_partitioned_loss(self.get_trainer(compute_loss_func=compute_loss))

    def test_label_smoothing(self):
        self.check_partitioned_loss(self.get_trainer(label_smoothing_factor=0.1))

    def test_dense_model(self):
        self.check_partitioned_loss(self.get_trainer(dense=True))

    def test_tensor_parallel_ranks_do_not_increase_loss_scale(self):
        trainer = self.get_trainer()
        parallelism_config = SimpleNamespace(tp_size=2, sp_backend=None)
        with patch.object(trainer.accelerator.state, "parallelism_config", parallelism_config):
            self.check_partitioned_loss(trainer, tp_size=2)

    @parameterized.expand([("averaging_disabled", False, 64), ("no_token_count", True, None)])
    def test_unscaled_model_loss(self, _, average_tokens, num_items):
        trainer = self.get_trainer(average_tokens_across_devices=average_tokens)
        input_ids = torch.arange(18).reshape(2, 9) + 3
        inputs = {"input_ids": input_ids, "labels": input_ids.clone(), "attention_mask": torch.ones_like(input_ids)}
        expected = trainer.model(**inputs, num_items_in_batch=num_items).loss
        with patch.object(trainer.accelerator.state, "num_processes", 2):
            loss, outputs = trainer.compute_loss(
                trainer.model, inputs, return_outputs=True, num_items_in_batch=num_items
            )
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(outputs.loss, expected)
