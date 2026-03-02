# Copyright 2018 the HuggingFace Inc. team.
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

"""
Trainer optimizer and LR scheduler tests: custom optimizers, LR scheduler kwargs, cosine-with-min-lr,
reduce-on-plateau, Adafactor, bitsandbytes (RMSProp, AdEMAMix), LOMO, GrokAdamW, schedule-free,
GaLore, Apollo, Stable AdamW, Liger kernel, optimizer choice resolution, factory pattern detection,
and model parameter inspection.
"""

import tempfile

import numpy as np
from parameterized import parameterized

from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    is_torch_available,
)
from transformers.testing_utils import (
    TestCasePlus,
    require_apollo_torch,
    require_bitsandbytes,
    require_galore_torch,
    require_grokadamw,
    require_lomo,
    require_schedulefree,
    require_torch,
    require_torch_accelerator,
    require_torch_optimi,
)
from transformers.trainer_utils import check_target_module_exists

from .trainer_test_utils import (
    BasicTextGenerationModel,
    RegressionDataset,
    RegressionModel,
    RepeatDataset,
    TorchTracemalloc,
    TrainerIntegrationCommon,
    TstLayer,
    bytes2megabytes,
    get_regression_trainer,
)


if is_torch_available():
    import torch
    from torch import nn

_ATTN_MLP_TARGET_MODULES = [r".*attn.*", r".*mlp.*"]


@require_torch
class TrainerOptimizerIntegrationTest(TestCasePlus, TrainerIntegrationCommon):
    def setUp(self):
        super().setUp()
        args = TrainingArguments("..")
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    def _get_llama_and_dataset(self):
        config = LlamaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=3, num_attention_heads=4)
        model = LlamaForCausalLM(config)
        train_dataset = RepeatDataset(torch.randint(0, 100, (128,)))
        return model, train_dataset

    def _get_gpt2_and_dataset(self):
        config = GPT2Config(vocab_size=100, n_positions=128, n_embd=32, n_layer=3, n_head=4)
        model = GPT2LMHeadModel(config)
        train_dataset = RepeatDataset(torch.randint(0, 100, (128,)))
        return model, train_dataset

    def _train_with_llama(self, optim, optim_target_modules=None, **extra_kwargs):
        """Smoke-test: tiny Llama + RepeatDataset with the given optimizer."""
        tiny_llama, train_dataset = self._get_llama_and_dataset()
        kwargs = {"learning_rate": 1e-9, "logging_steps": 5, "optim": optim}
        if optim_target_modules is not None:
            kwargs["optim_target_modules"] = optim_target_modules
        kwargs.update(extra_kwargs)
        args = TrainingArguments(self.get_auto_remove_tmp_dir(), **kwargs)
        trainer = Trainer(tiny_llama, args, train_dataset=train_dataset)
        trainer.train()
        return trainer

    def _check_lr_display_without_scheduler(self, optim, optim_target_modules):
        """Verify that LR is correctly reported without an LR scheduler."""
        tiny_llama, train_dataset = self._get_llama_and_dataset()
        learning_rate = 1e-9
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=learning_rate,
            logging_steps=5,
            optim=optim,
            optim_target_modules=optim_target_modules,
        )
        trainer = Trainer(tiny_llama, args, train_dataset=train_dataset)
        trainer.create_optimizer_and_scheduler(num_training_steps=10)
        self.assertEqual(trainer.get_learning_rates(), [learning_rate, learning_rate])

    def _check_lr_display_with_scheduler(self, optim, optim_target_modules, num_train_epochs=2):
        """Verify warmup + cosine LR schedule: increases then decreases."""
        tiny_llama, train_dataset = self._get_llama_and_dataset()
        learning_rate = 2e-4
        num_warmup_steps = 5
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            warmup_steps=num_warmup_steps,
            lr_scheduler_type="cosine",
            logging_steps=1,
            optim=optim,
            optim_target_modules=optim_target_modules,
        )
        trainer = Trainer(tiny_llama, args, train_dataset=train_dataset)
        trainer.train()
        logs = trainer.state.log_history[1:-1]

        self.assertTrue(logs[num_warmup_steps - 1]["learning_rate"] == learning_rate)
        self.assertTrue(np.allclose(logs[-1]["learning_rate"], 0, atol=5e-6))

        increasing_lrs = [
            logs[i]["learning_rate"] < logs[i + 1]["learning_rate"]
            for i in range(len(logs))
            if i < num_warmup_steps - 1
        ]
        decreasing_lrs = [
            logs[i]["learning_rate"] > logs[i + 1]["learning_rate"]
            for i in range(len(logs) - 1)
            if i >= num_warmup_steps - 1
        ]

        self.assertTrue(all(increasing_lrs))
        self.assertTrue(all(decreasing_lrs))
        self.assertTrue(len(decreasing_lrs) > len(increasing_lrs))

    # ---------------------------------------------------------------------------
    # adafactor optmizer test
    # ---------------------------------------------------------------------------

    def test_adafactor_lr_none(self):
        # test the special case where lr=None, since Trainer can't not have lr_scheduler

        from transformers.optimization import Adafactor, AdafactorSchedule

        train_dataset = RegressionDataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(tmp_dir)
            model = RegressionModel()
            optimizer = Adafactor(
                model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None
            )
            lr_scheduler = AdafactorSchedule(optimizer)
            trainer = Trainer(model, args, train_dataset=train_dataset, optimizers=(optimizer, lr_scheduler))
            trainer.train()

            # Train a default model to compare against
            default_trainer = get_regression_trainer(learning_rate=0.1, output_dir=tmp_dir)
            default_trainer.train()

            self.assertFalse(torch.allclose(trainer.model.a, default_trainer.model.a))
            self.assertFalse(torch.allclose(trainer.model.b, default_trainer.model.b))
            self.assertGreater(trainer.optimizer.state_dict()["param_groups"][0]["lr"], 0)

    # ---------------------------------------------------------------------------
    # BNB optimizer tests
    # ---------------------------------------------------------------------------

    @parameterized.expand(["rmsprop_bnb", "ademamix", "ademamix_8bit", "rmsprop_bnb_8bit", "rmsprop_bnb_32bit"])
    @require_bitsandbytes
    def test_bnb_optim(self, optim):
        tiny_gpt2, train_dataset = self._get_gpt2_and_dataset()
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e-9,
            logging_steps=5,
            logging_nan_inf_filter=False,
            optim=optim,
        )
        Trainer(tiny_gpt2, args, train_dataset=train_dataset).train()

    @require_bitsandbytes
    def test_bnb_8bit_optimizer_skip_embedding(self):
        model = BasicTextGenerationModel(8, 4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            for name_optim in ["rmsprop_bnb_8bit", "adamw_8bit"]:
                args = TrainingArguments(
                    output_dir=tmp_dir,
                    optim=name_optim,
                )
                trainer = Trainer(model=model, args=args)
                optimizer = trainer.create_optimizer()
                modules = optimizer.mng.module_weight_config_triple
                self.assertNotEqual(len(modules), 0)
                module, name, config = modules[0]
                self.assertIsInstance(module, torch.nn.Embedding)
                self.assertEqual(name, "weight")
                self.assertDictEqual(config, {"optim_bits": 32})

    # ---------------------------------------------------------------------------
    # LOMO tests
    # ---------------------------------------------------------------------------

    @require_lomo
    @require_torch_accelerator
    def test_lomo(self):
        tiny_llama, train_dataset = self._get_llama_and_dataset()
        previous_params = {n: p.clone() for n, p in tiny_llama.named_parameters()}

        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(), learning_rate=1e-2, logging_steps=5, optim="lomo", max_steps=20
        )
        Trainer(tiny_llama, args, train_dataset=train_dataset).train()

        for name, param in tiny_llama.named_parameters():
            self.assertFalse(torch.allclose(param, previous_params[name].to(param.device), rtol=1e-12, atol=1e-12))

    @require_lomo
    @require_torch_accelerator
    def test_adalomo(self):
        self._train_with_llama("adalomo")

    # ---------------------------------------------------------------------------
    # GrokAdamW test
    # ---------------------------------------------------------------------------

    @require_grokadamw
    @require_torch_accelerator
    def test_grokadamw(self):
        self._train_with_llama("grokadamw", learning_rate=2e-5, max_steps=20)

    # ---------------------------------------------------------------------------
    # Schedule-free tests
    # ---------------------------------------------------------------------------

    @parameterized.expand([("schedule_free_adamw",), ("schedule_free_radam",)])
    @require_schedulefree
    @require_torch_accelerator
    def test_schedulefree(self, optim):
        self._train_with_llama(optim, lr_scheduler_type="constant")

    # ---------------------------------------------------------------------------
    # GaLore tests
    # ---------------------------------------------------------------------------

    def test_galore_matched_modules(self):
        regex_patterns = [r".*.attn.*", r".*.mlp.*"]

        module_names = [
            "model.transformer.h.0.ln_1",
            "model.transformer.h.0.attn.q_proj",
            "model.lm_head",
            "model.transformer.h.0.mlp.up_proj",
        ]
        expected_values = [False, True, False, True]

        for expected_value, module_name in zip(expected_values, module_names):
            is_module_matched, is_regex = check_target_module_exists(regex_patterns, module_name, return_is_regex=True)
            self.assertTrue(is_module_matched == expected_value)
            if is_module_matched:
                self.assertTrue(is_regex)

        exact_patterns = ["q_proj", "up_proj"]

        module_names = [
            "model.transformer.h.0.ln_1",
            "model.transformer.h.0.attn.q_proj",
            "model.lm_head",
            "model.transformer.h.0.mlp.up_proj",
        ]
        expected_values = [False, True, False, True]

        for expected_value, module_name in zip(expected_values, module_names):
            is_module_matched, is_regex = check_target_module_exists(exact_patterns, module_name, return_is_regex=True)
            self.assertTrue(is_module_matched == expected_value)
            if is_module_matched:
                self.assertFalse(is_regex)

        simple_regex = r".*.attn.*"

        module_names = [
            "model.transformer.h.0.ln_1",
            "model.transformer.h.0.attn.q_proj",
            "model.lm_head",
            "model.transformer.h.0.mlp.up_proj",
        ]
        expected_values = [False, True, False, False]

        for expected_value, module_name in zip(expected_values, module_names):
            is_module_matched, is_regex = check_target_module_exists(simple_regex, module_name, return_is_regex=True)
            self.assertTrue(is_module_matched == expected_value)
            if is_module_matched:
                self.assertTrue(is_regex)

        simple_regex = "model.transformer.h.0.attn.q_proj"

        module_names = [
            "model.transformer.h.0.ln_1",
            "model.transformer.h.0.attn.q_proj",
            "model.lm_head",
            "model.transformer.h.0.mlp.up_proj",
        ]
        expected_values = [False, True, False, False]

        for expected_value, module_name in zip(expected_values, module_names):
            is_module_matched, is_regex = check_target_module_exists(simple_regex, module_name, return_is_regex=True)
            self.assertTrue(is_module_matched == expected_value)
            if is_module_matched:
                self.assertFalse(is_regex)

        target_modules = ["attn", "mlp"]

        module_names = [
            "model.transformer.h.0.ln_1",
            "model.transformer.h.0.attn.q_proj",
            "model.lm_head",
            "model.transformer.h.0.mlp.up_proj",
        ]
        expected_values = [False, True, False, True]

        for expected_value, module_name in zip(expected_values, module_names):
            is_module_matched, is_regex = check_target_module_exists(target_modules, module_name, return_is_regex=True)
            self.assertTrue(is_module_matched == expected_value)
            if is_module_matched:
                self.assertFalse(is_regex)

    @parameterized.expand([("galore_adamw",), ("galore_adamw_layerwise",), ("galore_adamw_8bit",)])
    @require_galore_torch
    @require_torch_accelerator
    def test_galore(self, optim):
        self._train_with_llama(optim, optim_target_modules=_ATTN_MLP_TARGET_MODULES)

    @require_galore_torch
    @require_torch_accelerator
    def test_galore_extra_args(self):
        self._train_with_llama(
            "galore_adamw",
            optim_target_modules=_ATTN_MLP_TARGET_MODULES,
            optim_args="rank=64, update_proj_gap=100, scale=0.10",
        )

    @require_galore_torch
    @require_torch_accelerator
    def test_galore_layerwise_with_scheduler(self):
        self._train_with_llama(
            "galore_adamw_layerwise",
            optim_target_modules=_ATTN_MLP_TARGET_MODULES,
            lr_scheduler_type="cosine",
        )

    @parameterized.expand(
        [
            (_ATTN_MLP_TARGET_MODULES,),
            (["q_proj", "k_proj", "v_proj"],),
            ("all-linear",),
        ]
    )
    @require_galore_torch
    @require_torch_accelerator
    def test_galore_adafactor(self, optim_target_modules):
        upper_bound_pm = 700
        lower_bound_pm = 650
        tiny_llama, train_dataset = self._get_llama_and_dataset()

        with tempfile.TemporaryDirectory() as tmpdir, TorchTracemalloc() as tracemalloc:
            args = TrainingArguments(
                tmpdir,
                learning_rate=1e-9,
                logging_steps=5,
                optim="galore_adafactor",
                optim_target_modules=optim_target_modules,
            )
            Trainer(tiny_llama, args, train_dataset=train_dataset).train()

        galore_peak_memory = tracemalloc.peaked + bytes2megabytes(tracemalloc.begin)
        self.assertTrue(galore_peak_memory < upper_bound_pm)
        self.assertTrue(lower_bound_pm < galore_peak_memory)

    @require_galore_torch
    @require_torch_accelerator
    def test_galore_lr_display_without_scheduler(self):
        self._check_lr_display_without_scheduler("galore_adamw", _ATTN_MLP_TARGET_MODULES)

    @require_galore_torch
    @require_torch_accelerator
    def test_galore_lr_display_with_scheduler(self):
        self._check_lr_display_with_scheduler("galore_adamw", _ATTN_MLP_TARGET_MODULES)

    # ---------------------------------------------------------------------------
    # Apollo tests
    # ---------------------------------------------------------------------------

    @parameterized.expand([("apollo_adamw",), ("apollo_adamw_layerwise",)])
    @require_apollo_torch
    @require_torch_accelerator
    def test_apollo(self, optim):
        self._train_with_llama(optim, optim_target_modules=_ATTN_MLP_TARGET_MODULES)

    @require_apollo_torch
    @require_torch_accelerator
    def test_apollo_extra_args(self):
        self._train_with_llama(
            "apollo_adamw",
            optim_target_modules=_ATTN_MLP_TARGET_MODULES,
            optim_args="proj=random,scale_type=tensor,rank=1,update_proj_gap=100,scale=128.0",
        )

    @require_apollo_torch
    @require_torch_accelerator
    def test_apollo_layerwise_with_scheduler(self):
        self._train_with_llama(
            "apollo_adamw_layerwise",
            optim_target_modules=_ATTN_MLP_TARGET_MODULES,
            lr_scheduler_type="cosine",
        )

    @require_apollo_torch
    @require_torch_accelerator
    def test_apollo_lr_display_without_scheduler(self):
        self._check_lr_display_without_scheduler("apollo_adamw", _ATTN_MLP_TARGET_MODULES)

    @require_apollo_torch
    @require_torch_accelerator
    def test_apollo_lr_display_with_scheduler(self):
        self._check_lr_display_with_scheduler("apollo_adamw", _ATTN_MLP_TARGET_MODULES, num_train_epochs=10)

    # ---------------------------------------------------------------------------
    # Stable AdamW tests
    # ---------------------------------------------------------------------------

    @require_torch_optimi
    @require_torch_accelerator
    def test_stable_adamw(self):
        self._train_with_llama("stable_adamw", optim_target_modules=_ATTN_MLP_TARGET_MODULES)

    @require_torch_optimi
    @require_torch_accelerator
    def test_stable_adamw_extra_args(self):
        self._train_with_llama(
            "stable_adamw",
            optim_target_modules=_ATTN_MLP_TARGET_MODULES,
            optim_args="decouple_lr=True,max_lr=1e-3,kahan_sum=True",
        )

    @require_torch_optimi
    @require_torch_accelerator
    def test_stable_adamw_trainer_adamw_args(self):
        tiny_llama, train_dataset = self._get_llama_and_dataset()
        args = TrainingArguments(
            self.get_auto_remove_tmp_dir(),
            learning_rate=1e-9,
            logging_steps=5,
            weight_decay=0.001,
            adam_beta1=0.89,
            adam_beta2=0.98,
            adam_epsilon=1e-8,
            optim="stable_adamw",
            optim_target_modules=_ATTN_MLP_TARGET_MODULES,
        )
        trainer = Trainer(tiny_llama, args, train_dataset=train_dataset)
        trainer.create_optimizer_and_scheduler(num_training_steps=10)

        # check StableAdamW optimizer is created with the correct parameters
        self.assertEqual(trainer.optimizer.defaults["beta1"], args.adam_beta1)
        self.assertEqual(trainer.optimizer.defaults["beta2"], args.adam_beta2)
        self.assertEqual(trainer.optimizer.defaults["eps"], args.adam_epsilon)
        self.assertEqual(trainer.optimizer.defaults["weight_decay"], args.weight_decay)

    @require_torch_optimi
    @require_torch_accelerator
    def test_stable_adamw_lr_display_without_scheduler(self):
        self._check_lr_display_without_scheduler("stable_adamw", _ATTN_MLP_TARGET_MODULES)

    @require_torch_optimi
    @require_torch_accelerator
    def test_stable_adamw_lr_display_with_scheduler(self):
        self._check_lr_display_with_scheduler("stable_adamw", _ATTN_MLP_TARGET_MODULES, num_train_epochs=10)

    # ---------------------------------------------------------------------------
    # Misc optimizer tests
    # ---------------------------------------------------------------------------

    def test_optimizer_factory_pattern(self):
        """Test that is_optimizer_factory correctly identifies factory classes vs optimizer classes."""
        from transformers.trainer_optimizer import is_optimizer_factory

        # Create a mock optimizer class
        class MockComplexOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=1e-3):
                defaults = {"lr": lr}
                super().__init__(params, defaults)

            def step(self, closure=None):
                pass

        # Create a factory class (simulates Muon/Dion pattern)
        class MockOptimizerFactory:
            def __call__(self, opt_model, **optimizer_kwargs):
                all_params = list(opt_model.parameters())
                return MockComplexOptimizer(all_params, **optimizer_kwargs)

        # Verify is_optimizer_factory correctly identifies factories vs optimizer classes
        self.assertFalse(is_optimizer_factory(MockComplexOptimizer))  # Optimizer class should return False
        self.assertTrue(is_optimizer_factory(MockOptimizerFactory))  # Factory class should return True

    # ---------------------------------------------------------------------------
    # Optimizer group and learning rate inspection tests
    # ---------------------------------------------------------------------------

    def test_get_optimizer_group(self):
        model = nn.Sequential(nn.Linear(128, 64))
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(model=model, args=TrainingArguments(output_dir=tmp_dir))
            # ValueError is raised if optimizer is None
            with self.assertRaises(ValueError):
                trainer.get_optimizer_group()
            trainer.create_optimizer()
            # Get groups
            num_groups = len(trainer.get_optimizer_group())
            self.assertEqual(num_groups, 2)
            # Get group of parameter
            param = next(model.parameters())
            group = trainer.get_optimizer_group(param)
            self.assertIn(param, group["params"])


# ---------------------------------------------------------------------------
# Custom optimizer and LR scheduler tests
# ---------------------------------------------------------------------------


class TrainerOptimizerTest(TestCasePlus):
    def test_get_optimizer_group(self):
        model = nn.Sequential(nn.Linear(128, 64))
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(model=model, args=TrainingArguments(output_dir=tmp_dir))
            # ValueError is raised if optimizer is None
            with self.assertRaises(ValueError):
                trainer.get_optimizer_group()
            trainer.create_optimizer()
            # Get groups
            num_groups = len(trainer.get_optimizer_group())
            self.assertEqual(num_groups, 2)
            # Get group of parameter
            param = next(model.parameters())
            group = trainer.get_optimizer_group(param)
            self.assertIn(param, group["params"])

    def test_optimizer_factory_pattern(self):
        """Test that is_optimizer_factory correctly identifies factory classes vs optimizer classes."""
        from transformers.trainer_optimizer import is_optimizer_factory

        # Create a mock optimizer class
        class MockComplexOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr=1e-3):
                defaults = {"lr": lr}
                super().__init__(params, defaults)

            def step(self, closure=None):
                pass

        # Create a factory class (simulates Muon/Dion pattern)
        class MockOptimizerFactory:
            def __call__(self, opt_model, **optimizer_kwargs):
                all_params = list(opt_model.parameters())
                return MockComplexOptimizer(all_params, **optimizer_kwargs)

        # Verify is_optimizer_factory correctly identifies factories vs optimizer classes
        self.assertFalse(is_optimizer_factory(MockComplexOptimizer))  # Optimizer class should return False
        self.assertTrue(is_optimizer_factory(MockOptimizerFactory))  # Factory class should return True

    def test_custom_optimizer(self):
        train_dataset = RegressionDataset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(tmp_dir)
            model = RegressionModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
            trainer = Trainer(model, args, train_dataset=train_dataset, optimizers=(optimizer, lr_scheduler))
            trainer.train()

            # Train a default model to compare against
            default_trainer = get_regression_trainer(learning_rate=0.1, output_dir=tmp_dir)
            default_trainer.train()

            self.assertFalse(torch.allclose(trainer.model.a, default_trainer.model.a))
            self.assertFalse(torch.allclose(trainer.model.b, default_trainer.model.b))
            self.assertEqual(trainer.optimizer.state_dict()["param_groups"][0]["lr"], 1.0)

    # ---------------------------------------------------------------------------
    # Weight decay parameter groups
    # ---------------------------------------------------------------------------

    def test_no_wd_param_group(self):
        model = nn.Sequential(TstLayer(128), nn.ModuleList([TstLayer(128), TstLayer(128)]))
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(model=model, args=TrainingArguments(output_dir=tmp_dir))
            trainer.create_optimizer_and_scheduler(10)
            wd_names = ['0.linear1.weight', '0.linear2.weight', '1.0.linear1.weight', '1.0.linear2.weight', '1.1.linear1.weight', '1.1.linear2.weight']  # fmt: skip
            wd_params = [p for n, p in model.named_parameters() if n in wd_names]
            no_wd_params = [p for n, p in model.named_parameters() if n not in wd_names]
            self.assertListEqual(trainer.optimizer.param_groups[0]["params"], wd_params)
            self.assertListEqual(trainer.optimizer.param_groups[1]["params"], no_wd_params)


@require_torch
class TrainerLRTest(TestCasePlus):
    def test_get_learning_rates(self):
        model = nn.Sequential(nn.Linear(128, 64))
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(model=model, args=TrainingArguments(output_dir=tmp_dir))
            with self.assertRaises(ValueError):
                trainer.get_learning_rates()
            trainer.create_optimizer()
            self.assertEqual(trainer.get_learning_rates(), [5e-05, 5e-05])

    def test_lr_scheduler_kwargs(self):
        from transformers import get_polynomial_decay_schedule_with_warmup

        # test scheduler kwargs passed via TrainingArguments
        train_dataset = RegressionDataset()
        model = RegressionModel()
        num_steps, num_warmup_steps = 10, 2
        extra_kwargs = {"power": 5.0, "lr_end": 1e-5}  # Non-default arguments
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                lr_scheduler_type="polynomial",
                lr_scheduler_kwargs=extra_kwargs,
                learning_rate=0.2,
                warmup_steps=num_warmup_steps,
            )
            trainer = Trainer(model, args, train_dataset=train_dataset)
            trainer.create_optimizer_and_scheduler(num_training_steps=num_steps)

            # Checking that the scheduler was created
            self.assertIsNotNone(trainer.lr_scheduler)

            # Checking that the correct args were passed
            sched1 = trainer.lr_scheduler
            sched2 = get_polynomial_decay_schedule_with_warmup(
                trainer.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_steps, **extra_kwargs
            )
            self.assertEqual(sched1.lr_lambdas[0].args, sched2.lr_lambdas[0].args)
            self.assertEqual(sched1.lr_lambdas[0].keywords, sched2.lr_lambdas[0].keywords)

    def test_cosine_with_min_lr_scheduler(self):
        train_dataset = RegressionDataset()
        model = RegressionModel()
        num_steps, num_warmup_steps = 10, 2
        extra_kwargs = {"min_lr": 1e-5}  # Non-default arguments
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                lr_scheduler_type="cosine_with_min_lr",
                lr_scheduler_kwargs=extra_kwargs,
                learning_rate=0.2,
                warmup_steps=num_warmup_steps,
            )
            trainer = Trainer(model, args, train_dataset=train_dataset)
            trainer.create_optimizer_and_scheduler(num_training_steps=num_steps)

            # Checking that the scheduler was created
            self.assertIsNotNone(trainer.lr_scheduler)

            # Check the last learning rate
            for _ in range(num_steps):
                trainer.lr_scheduler.step()
            self.assertEqual(trainer.lr_scheduler.get_last_lr()[0], 1e-5)

    def test_cosine_with_min_lr_schedule_with_warmup_lr_rate(self):
        train_dataset = RegressionDataset()
        model = RegressionModel()
        num_steps, num_warmup_steps = 10, 2
        extra_kwargs = {"min_lr": 1e-5}  # Non-default arguments
        args = TrainingArguments(
            "./regression",
            lr_scheduler_type="cosine_warmup_with_min_lr",
            lr_scheduler_kwargs=extra_kwargs,
            learning_rate=0.2,
            warmup_steps=num_warmup_steps,
        )
        trainer = Trainer(model, args, train_dataset=train_dataset)
        trainer.create_optimizer_and_scheduler(num_training_steps=num_steps)

        # Checking that the scheduler was created
        self.assertIsNotNone(trainer.lr_scheduler)

        # Check the last learning rate
        step_lrs = []
        for _ in range(num_steps):
            step_lrs.append(trainer.optimizer.param_groups[0]["lr"])
            trainer.lr_scheduler.step()
        self.assertEqual(step_lrs[0], 0.1)
        self.assertEqual(step_lrs[1], 0.2)
        self.assertEqual(step_lrs[-1], 1e-05)

    def test_reduce_lr_on_plateau_args(self):
        # test passed arguments for a custom ReduceLROnPlateau scheduler
        train_dataset = RegressionDataset(length=64)
        eval_dataset = RegressionDataset(length=64)
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                eval_strategy="epoch",
                metric_for_best_model="eval_loss",
            )
            model = RegressionModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5, cooldown=2)
            trainer = Trainer(
                model,
                args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                optimizers=(optimizer, lr_scheduler),
            )
            trainer.train()

            self.assertIsInstance(trainer.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            self.assertEqual(trainer.lr_scheduler.factor, 0.2)
            self.assertEqual(trainer.lr_scheduler.patience, 5)
            self.assertEqual(trainer.lr_scheduler.cooldown, 2)

    def test_reduce_lr_on_plateau(self):
        # test the ReduceLROnPlateau scheduler

        class TrainerWithLRLogs(Trainer):
            def log(self, logs):
                # the LR is computed after metrics and does not exist for the first epoch
                if hasattr(self.lr_scheduler, "_last_lr"):
                    logs["learning_rate"] = self.lr_scheduler._last_lr[0]
                super().log(logs)

        train_dataset = RegressionDataset(length=64)
        eval_dataset = RegressionDataset(length=64)

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                tmp_dir,
                lr_scheduler_type="reduce_lr_on_plateau",
                eval_strategy="epoch",
                metric_for_best_model="eval_loss",
                num_train_epochs=10,
                learning_rate=0.2,
            )
            model = RegressionModel()
            trainer = TrainerWithLRLogs(model, args, train_dataset=train_dataset, eval_dataset=eval_dataset)
            trainer.train()

            self.assertIsInstance(trainer.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            patience = trainer.lr_scheduler.patience

            logs = trainer.state.log_history[1:]
            best_loss = logs[0]["eval_loss"]
            bad_epochs = 0
            for i, log in enumerate(logs[:-1]):  # Compare learning rate to next epoch's
                loss = log["eval_loss"]
                just_decreased = False
                if loss > best_loss:
                    bad_epochs += 1
                    if bad_epochs > patience:
                        self.assertLess(logs[i + 1]["learning_rate"], log["learning_rate"])
                        just_decreased = True
                        bad_epochs = 0
                else:
                    best_loss = loss
                    bad_epochs = 0
                if not just_decreased:
                    self.assertEqual(logs[i + 1]["learning_rate"], log["learning_rate"])
