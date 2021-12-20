# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch S4 model. """


import random
import unittest

from transformers import S4Config, is_torch_available
from transformers.testing_utils import (
    require_einops,
    require_opt_einsum,
    require_pykeops,
    require_scipy,
    require_torch,
    torch_device,
)

from .test_configuration_common import ConfigTester
from .test_generation_utils import GenerationTesterMixin
from .test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import S4LMHeadModel, S4Model


class S4ModelTester:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 14
        self.vocab_size = 99
        self.d_embed = 32
        self.embedding_dropout_prob = 0.25
        self.div_val = 2
        self.cutoffs = [7, 47, 77]
        self.tie_weights = True
        self.tie_projs = [True, True, True]
        self.initializer_scale = 0.5
        self.bias_scale = 1.0
        self.input_dropout_prob = 0.0
        self.hidden_dropout_prob = 0.25
        self.pre_norm = True
        self.transposed = True
        self.d_model = 32
        self.num_hidden_layers = 5
        self.residual_connections = True
        self.pool_size = 1
        self.pool_expand = 1
        self.normalize_type = "layer"
        self.d_state = 64
        self.measure = "legs"
        self.rank = 1
        self.dt_min = 0.001
        self.dt_max = 0.1
        self.trainable_s4_params = {"A": 1, "B": 2, "C": 1, "dt": 1}
        self.learning_rate_s4_params = {"A": 0.0005, "B": 0.0005, "C": None, "dt": 0.0005}
        self.cache = False
        self.weight_decay = 0.0
        self.l_max = 7
        self.activation_function = "gelu"
        self.post_activation_function = "glu"
        self.s4_dropout = 0.25
        self.ff_expand = 4
        self.ff_activation_function = "gelu"
        self.ff_dropout = 0.25
        self.softmax_dropout_prob = 0.25
        self.initializer_range = 0.02
        self.seed = 1
        self.use_labels = True
        self.is_training = False

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.l_max], self.vocab_size)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.l_max], self.vocab_size)

        config = self.get_config()

        return config, input_ids, lm_labels

    def get_config(self):
        return S4Config(
            vocab_size=self.vocab_size,
            d_embed=self.d_embed,
            embedding_dropout_prob=self.embedding_dropout_prob,
            div_val=self.div_val,
            cutoffs=self.cutoffs,
            tie_weights=self.tie_weights,
            tie_projs=self.tie_projs,
            initializer_scale=self.initializer_scale,
            bias_scale=self.bias_scale,
            input_dropout_prob=self.input_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob,
            pre_norm=self.pre_norm,
            transposed=self.transposed,
            d_model=self.d_model,
            num_hidden_layers=self.num_hidden_layers,
            residual_connections=self.residual_connections,
            pool_size=self.pool_size,
            pool_expand=self.pool_expand,
            normalize_type=self.normalize_type,
            d_state=self.d_state,
            measure=self.measure,
            rank=self.rank,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            trainable_s4_params=self.trainable_s4_params,
            learning_rate_s4_params=self.learning_rate_s4_params,
            cache=self.cache,
            weight_decay=self.weight_decay,
            l_max=self.l_max,
            activation_function=self.activation_function,
            post_activation_function=self.post_activation_function,
            s4_dropout=self.s4_dropout,
            ff_expand=self.ff_expand,
            ff_activation_function=self.ff_activation_function,
            ff_dropout=self.ff_dropout,
            softmax_dropout_prob=self.softmax_dropout_prob,
            initializer_range=self.initializer_range,
        )

    def set_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def create_s4_model(self, config, input_ids, lm_labels):
        model = S4Model(config)
        model.to(torch_device)
        model.eval()

        outputs = model(input_ids)
        outputs = {
            "hidden_states": outputs["last_hidden_state"],
        }
        return outputs

    def check_s4_model_output(self, result):
        self.parent.assertEqual(result["hidden_states"].shape, (self.batch_size, self.l_max, self.d_model))

    def create_s4_lm_head(self, config, input_ids, lm_labels):
        model = S4LMHeadModel(config)
        model.to(torch_device)
        model.eval()

        lm_logits = model(input_ids)["prediction_scores"]
        outputs = model(input_ids, labels=lm_labels)

        outputs = {"loss": outputs["losses"], "lm_logits": lm_logits}
        return outputs

    def check_s4_lm_head_output(self, result):
        self.parent.assertEqual(result["loss"].shape, ())
        self.parent.assertEqual(result["lm_logits"].shape, (self.batch_size, self.l_max, self.cutoffs[0] + 3))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            lm_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict


@require_torch
@require_einops
@require_opt_einsum
@require_pykeops
@require_scipy
class S4ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            S4Model,
            S4LMHeadModel,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (S4LMHeadModel,) if is_torch_available() else ()
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = True
    test_mismatched_shape = False

    def check_cutoffs_and_n_token(
        self, copied_cutoffs, layer, model_embed, model, model_class, resized_value, vocab_size
    ):
        # Check that the cutoffs were modified accordingly
        for i in range(len(copied_cutoffs)):
            if i < layer:
                self.assertEqual(model_embed.cutoffs[i], copied_cutoffs[i])
                if model_class == S4LMHeadModel:
                    self.assertEqual(model.crit.cutoffs[i], copied_cutoffs[i])
                if i < len(model.config.cutoffs):
                    self.assertEqual(model.config.cutoffs[i], copied_cutoffs[i])
            else:
                self.assertEqual(model_embed.cutoffs[i], copied_cutoffs[i] + resized_value)
                if model_class == S4LMHeadModel:
                    self.assertEqual(model.crit.cutoffs[i], copied_cutoffs[i] + resized_value)
                if i < len(model.config.cutoffs):
                    self.assertEqual(model.config.cutoffs[i], copied_cutoffs[i] + resized_value)

        self.assertEqual(model_embed.n_token, vocab_size + resized_value)
        if model_class == S4LMHeadModel:
            self.assertEqual(model.crit.n_token, vocab_size + resized_value)

    def setUp(self):
        self.model_tester = S4ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=S4Config, d_embed=37)

    def test_config(self):
        self.config_tester.create_and_test_config_common_properties = lambda: None
        self.config_tester.run_common_tests()

    def test_s4_model(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        output_result = self.model_tester.create_s4_model(*config_and_inputs)
        self.model_tester.check_s4_model_output(output_result)

    def test_s4_lm_head(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        output_result = self.model_tester.create_s4_lm_head(*config_and_inputs)
        self.model_tester.check_s4_lm_head_output(output_result)

    def test_retain_grad_hidden_states_attentions(self):
        # s4 no attentions outputs
        return

    def test_attention_outputs(self):
        # s4 no attention outputs
        return
