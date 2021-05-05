# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import random
import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, require_torch_multi_gpu, slow, torch_device

from .test_configuration_common import ConfigTester
from .test_generation_utils import GenerationTesterMixin
from .test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import TransfoXLConfig, TransfoXLForSequenceClassification, TransfoXLLMHeadModel, TransfoXLModel
    from transformers.models.transfo_xl.modeling_transfo_xl import TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST


class TransfoXLModelTester:
    def __init__(
        self,
        parent,
    ):
        self.parent = parent
        self.batch_size = 14
        self.seq_length = 7
        self.mem_len = 30
        self.key_length = self.seq_length + self.mem_len
        self.clamp_len = 15
        self.is_training = False
        self.use_labels = True
        self.vocab_size = 99
        self.cutoffs = [10, 50, 80]
        self.hidden_size = 32
        self.d_embed = 32
        self.num_attention_heads = 4
        self.d_head = 8
        self.d_inner = 128
        self.div_val = 2
        self.num_hidden_layers = 5
        self.scope = None
        self.seed = 1
        self.eos_token_id = 0
        self.num_labels = 3
        self.pad_token_id = self.vocab_size - 1

    def prepare_config_and_inputs(self):
        input_ids_1 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids_2 = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = TransfoXLConfig(
            vocab_size=self.vocab_size,
            mem_len=self.mem_len,
            clamp_len=self.clamp_len,
            cutoffs=self.cutoffs,
            d_model=self.hidden_size,
            d_embed=self.d_embed,
            n_head=self.num_attention_heads,
            d_head=self.d_head,
            d_inner=self.d_inner,
            div_val=self.div_val,
            n_layer=self.num_hidden_layers,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

        return (config, input_ids_1, input_ids_2, lm_labels)

    def set_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)

    def create_transfo_xl_model(self, config, input_ids_1, input_ids_2, lm_labels):
        model = TransfoXLModel(config)
        model.to(torch_device)
        model.eval()

        outputs1 = model(input_ids_1)
        outputs2 = model(input_ids_2, outputs1["mems"])
        outputs = {
            "hidden_states_1": outputs1["last_hidden_state"],
            "mems_1": outputs1["mems"],
            "hidden_states_2": outputs2["last_hidden_state"],
            "mems_2": outputs2["mems"],
        }
        return outputs

    def check_transfo_xl_model_output(self, result):
        self.parent.assertEqual(result["hidden_states_1"].shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertEqual(result["hidden_states_2"].shape, (self.batch_size, self.seq_length, self.hidden_size))
        self.parent.assertListEqual(
            [mem.shape for mem in result["mems_1"]],
            [(self.mem_len, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
        )
        self.parent.assertListEqual(
            [mem.shape for mem in result["mems_2"]],
            [(self.mem_len, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
        )

    def create_transfo_xl_lm_head(self, config, input_ids_1, input_ids_2, lm_labels):
        model = TransfoXLLMHeadModel(config)
        model.to(torch_device)
        model.eval()

        lm_logits_1 = model(input_ids_1)["prediction_scores"]
        outputs1 = model(input_ids_1, labels=lm_labels)
        lm_logits_2 = model(input_ids_2, mems=outputs1["mems"])["prediction_scores"]
        outputs2 = model(input_ids_2, labels=lm_labels, mems=outputs1["mems"])

        outputs = {
            "loss_1": outputs1["losses"],
            "mems_1": outputs1["mems"],
            "lm_logits_1": lm_logits_1,
            "loss_2": outputs2["losses"],
            "mems_2": outputs2["mems"],
            "lm_logits_2": lm_logits_2,
        }
        return outputs

    def check_transfo_xl_lm_head_output(self, result):
        self.parent.assertEqual(result["loss_1"].shape, (self.batch_size, self.seq_length - 1))
        self.parent.assertEqual(result["lm_logits_1"].shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertListEqual(
            [mem.shape for mem in result["mems_1"]],
            [(self.mem_len, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
        )

        self.parent.assertEqual(result["loss_2"].shape, (self.batch_size, self.seq_length - 1))
        self.parent.assertEqual(result["lm_logits_2"].shape, (self.batch_size, self.seq_length, self.vocab_size))
        self.parent.assertListEqual(
            [mem.shape for mem in result["mems_2"]],
            [(self.mem_len, self.batch_size, self.hidden_size)] * self.num_hidden_layers,
        )

    def create_and_check_transfo_xl_for_sequence_classification(self, config, input_ids_1, input_ids_2, lm_labels):
        config.num_labels = self.num_labels
        model = TransfoXLForSequenceClassification(config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids_1)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids_1, input_ids_2, lm_labels) = config_and_inputs
        inputs_dict = {"input_ids": input_ids_1}
        return config, inputs_dict


@require_torch
class TransfoXLModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (
        (TransfoXLModel, TransfoXLLMHeadModel, TransfoXLForSequenceClassification) if is_torch_available() else ()
    )
    all_generative_model_classes = (TransfoXLLMHeadModel,) if is_torch_available() else ()
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = True

    def check_cutoffs_and_n_token(
        self, copied_cutoffs, layer, model_embed, model, model_class, resized_value, vocab_size
    ):
        # Check that the cutoffs were modified accordingly
        for i in range(len(copied_cutoffs)):
            if i < layer:
                self.assertEqual(model_embed.cutoffs[i], copied_cutoffs[i])
                if model_class == TransfoXLLMHeadModel:
                    self.assertEqual(model.crit.cutoffs[i], copied_cutoffs[i])
                if i < len(model.config.cutoffs):
                    self.assertEqual(model.config.cutoffs[i], copied_cutoffs[i])
            else:
                self.assertEqual(model_embed.cutoffs[i], copied_cutoffs[i] + resized_value)
                if model_class == TransfoXLLMHeadModel:
                    self.assertEqual(model.crit.cutoffs[i], copied_cutoffs[i] + resized_value)
                if i < len(model.config.cutoffs):
                    self.assertEqual(model.config.cutoffs[i], copied_cutoffs[i] + resized_value)

        self.assertEqual(model_embed.n_token, vocab_size + resized_value)
        if model_class == TransfoXLLMHeadModel:
            self.assertEqual(model.crit.n_token, vocab_size + resized_value)

    def setUp(self):
        self.model_tester = TransfoXLModelTester(self)
        self.config_tester = ConfigTester(self, config_class=TransfoXLConfig, d_embed=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_transfo_xl_model(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        output_result = self.model_tester.create_transfo_xl_model(*config_and_inputs)
        self.model_tester.check_transfo_xl_model_output(output_result)

    def test_transfo_xl_lm_head(self):
        self.model_tester.set_seed()
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        output_result = self.model_tester.create_transfo_xl_lm_head(*config_and_inputs)
        self.model_tester.check_transfo_xl_lm_head_output(output_result)

    def test_transfo_xl_sequence_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_transfo_xl_for_sequence_classification(*config_and_inputs)

    def test_retain_grad_hidden_states_attentions(self):
        # xlnet cannot keep gradients in attentions or hidden states
        return

    @require_torch_multi_gpu
    def test_multi_gpu_data_parallel_forward(self):
        # Opt-out of this test.
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = TransfoXLModel.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_resize_tokens_embeddings(self):
        (original_config, inputs_dict) = self.model_tester.prepare_config_and_inputs_for_common()
        if not self.test_resize_embeddings:
            return

        for model_class in self.all_model_classes:
            config = copy.deepcopy(original_config)
            model = model_class(config)
            model.to(torch_device)

            if self.model_tester.is_training is False:
                model.eval()

            model_vocab_size = config.vocab_size
            # Retrieve the embeddings and clone theme
            model_embed = model.resize_token_embeddings(model_vocab_size)
            cloned_embeddings = [emb.weight.clone() for emb in model_embed.emb_layers]
            # Retrieve the cutoffs and copy them
            copied_cutoffs = copy.copy(model_embed.cutoffs)

            test_layers = [x for x in range(config.div_val)]
            for layer in test_layers:
                # Check that resizing the token embeddings with a larger vocab size increases the model's vocab size
                model_embed = model.resize_token_embeddings(model_vocab_size + 10, layer)
                self.assertEqual(model.config.vocab_size, model_vocab_size + 10)
                # Check that it actually resizes the embeddings matrix
                self.assertEqual(model_embed.emb_layers[layer].weight.shape[0], cloned_embeddings[layer].shape[0] + 10)
                # Check that the cutoffs were modified accordingly
                self.check_cutoffs_and_n_token(
                    copied_cutoffs, layer, model_embed, model, model_class, 10, model_vocab_size
                )

                # Check that the model can still do a forward pass successfully (every parameter should be resized)
                model(**inputs_dict)

                # Check that resizing the token embeddings with a smaller vocab size decreases the model's vocab size
                model_embed = model.resize_token_embeddings(model_vocab_size - 5, layer)
                self.assertEqual(model.config.vocab_size, model_vocab_size - 5)
                # Check that it actually resizes the embeddings matrix
                self.assertEqual(model_embed.emb_layers[layer].weight.shape[0], cloned_embeddings[layer].shape[0] - 5)
                # Check that the cutoffs were modified accordingly
                self.check_cutoffs_and_n_token(
                    copied_cutoffs, layer, model_embed, model, model_class, -5, model_vocab_size
                )

                # Check that the model can still do a forward pass successfully (every parameter should be resized)
                # Input ids should be clamped to the maximum size of the vocabulary
                inputs_dict["input_ids"].clamp_(max=model_vocab_size - 5 - 1)
                model(**inputs_dict)

                # Check that adding and removing tokens has not modified the first part of the embedding matrix.
                models_equal = True
                for p1, p2 in zip(cloned_embeddings[layer], model_embed.emb_layers[layer].weight):
                    if p1.data.ne(p2.data).sum() > 0:
                        models_equal = False

                self.assertTrue(models_equal)

                # Reset model embeddings to original size
                model.resize_token_embeddings(model_vocab_size, layer)
                self.assertEqual(model_vocab_size, model.config.vocab_size)
                self.assertEqual(model_embed.emb_layers[layer].weight.shape[0], cloned_embeddings[layer].shape[0])

    def test_resize_embeddings_untied(self):
        # transfo-xl requires special resize for lm-head
        return

    def _check_attentions_for_generate(
        self, batch_size, attentions, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, tuple) for iter_attentions in attentions], [True] * len(attentions)
        )
        self.assertEqual(len(attentions), (max_length - min_length) * num_beam_groups)

        for idx, iter_attentions in enumerate(attentions):
            tgt_len = min_length if idx == 0 else (min_length - 2)
            src_len = (min_length + config.mem_len) if idx == 0 else (min_length + config.mem_len - 2)

            expected_shape = (
                batch_size * num_beam_groups,
                config.num_attention_heads,
                tgt_len,
                src_len,
            )

            # check attn size
            self.assertListEqual(
                [layer_attention.shape for layer_attention in iter_attentions], [expected_shape] * len(iter_attentions)
            )

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, min_length, max_length, config, use_cache=False, num_beam_groups=1
    ):
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (max_length - min_length) * num_beam_groups)

        for idx, iter_hidden_states in enumerate(hidden_states):
            seq_len = min_length if idx == 0 else min_length - 2
            expected_shape = (batch_size * num_beam_groups, seq_len, config.hidden_size)
            # check hidden size
            self.assertListEqual(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states],
                [expected_shape] * len(iter_hidden_states),
            )

    # overwrite from test_modeling_common
    def _mock_init_weights(self, module):
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data.fill_(3)
        if hasattr(module, "cluster_weight") and module.cluster_weight is not None:
            module.cluster_weight.data.fill_(3)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.fill_(3)
        if hasattr(module, "cluster_bias") and module.cluster_bias is not None:
            module.cluster_bias.data.fill_(3)

        if hasattr(module, "emb_projs"):
            for i in range(len(module.emb_projs)):
                if module.emb_projs[i] is not None:
                    torch.nn.init.constant_(module.emb_projs[i], 0.0003)
        if hasattr(module, "out_projs"):
            for i in range(len(module.out_projs)):
                if module.out_projs[i] is not None:
                    torch.nn.init.constant_(module.out_projs[i], 0.0003)

        for param in ["r_emb", "r_w_bias", "r_r_bias", "r_bias"]:
            if hasattr(module, param) and getattr(module, param) is not None:
                weight = getattr(module, param)
                weight.data.fill_(3)


@require_torch
class TransfoXLModelLanguageGenerationTest(unittest.TestCase):
    @slow
    def test_lm_generate_transfo_xl_wt103(self):
        model = TransfoXLLMHeadModel.from_pretrained("transfo-xl-wt103")
        model.to(torch_device)
        input_ids = torch.tensor(
            [
                [
                    33,
                    1297,
                    2,
                    1,
                    1009,
                    4,
                    1109,
                    11739,
                    4762,
                    358,
                    5,
                    25,
                    245,
                    22,
                    1706,
                    17,
                    20098,
                    5,
                    3215,
                    21,
                    37,
                    1110,
                    3,
                    13,
                    1041,
                    4,
                    24,
                    603,
                    490,
                    2,
                    71477,
                    20098,
                    104447,
                    2,
                    20961,
                    1,
                    2604,
                    4,
                    1,
                    329,
                    3,
                    6224,
                    831,
                    16002,
                    2,
                    8,
                    603,
                    78967,
                    29546,
                    23,
                    803,
                    20,
                    25,
                    416,
                    5,
                    8,
                    232,
                    4,
                    277,
                    6,
                    1855,
                    4601,
                    3,
                    29546,
                    54,
                    8,
                    3609,
                    5,
                    57211,
                    49,
                    4,
                    1,
                    277,
                    18,
                    8,
                    1755,
                    15691,
                    3,
                    341,
                    25,
                    416,
                    693,
                    42573,
                    71,
                    17,
                    401,
                    94,
                    31,
                    17919,
                    2,
                    29546,
                    7873,
                    18,
                    1,
                    435,
                    23,
                    11011,
                    755,
                    5,
                    5167,
                    3,
                    7983,
                    98,
                    84,
                    2,
                    29546,
                    3267,
                    8,
                    3609,
                    4,
                    1,
                    4865,
                    1075,
                    2,
                    6087,
                    71,
                    6,
                    346,
                    8,
                    5854,
                    3,
                    29546,
                    824,
                    1400,
                    1868,
                    2,
                    19,
                    160,
                    2,
                    311,
                    8,
                    5496,
                    2,
                    20920,
                    17,
                    25,
                    15097,
                    3,
                    24,
                    24,
                    0,
                ]
            ],
            dtype=torch.long,
            device=torch_device,
        )
        #  In 1991 , the remains of Russian Tsar Nicholas II and his family
        #  ( except for Alexei and Maria ) are discovered .
        #  The voice of Nicholas's young son , Tsarevich Alexei Nikolaevich , narrates the
        #  remainder of the story . 1883 Western Siberia ,
        #  a young Grigori Rasputin is asked by his father and a group of men to perform magic .
        #  Rasputin has a vision and denounces one of the men as a horse thief . Although his
        #  father initially slaps him for making such an accusation , Rasputin watches as the
        #  man is chased outside and beaten . Twenty years later , Rasputin sees a vision of
        #  the Virgin Mary , prompting him to become a priest . Rasputin quickly becomes famous ,
        #  with people , even a bishop , begging for his blessing . <eod> </s> <eos>

        expected_output_ids = [
            33,
            1297,
            2,
            1,
            1009,
            4,
            1109,
            11739,
            4762,
            358,
            5,
            25,
            245,
            22,
            1706,
            17,
            20098,
            5,
            3215,
            21,
            37,
            1110,
            3,
            13,
            1041,
            4,
            24,
            603,
            490,
            2,
            71477,
            20098,
            104447,
            2,
            20961,
            1,
            2604,
            4,
            1,
            329,
            3,
            6224,
            831,
            16002,
            2,
            8,
            603,
            78967,
            29546,
            23,
            803,
            20,
            25,
            416,
            5,
            8,
            232,
            4,
            277,
            6,
            1855,
            4601,
            3,
            29546,
            54,
            8,
            3609,
            5,
            57211,
            49,
            4,
            1,
            277,
            18,
            8,
            1755,
            15691,
            3,
            341,
            25,
            416,
            693,
            42573,
            71,
            17,
            401,
            94,
            31,
            17919,
            2,
            29546,
            7873,
            18,
            1,
            435,
            23,
            11011,
            755,
            5,
            5167,
            3,
            7983,
            98,
            84,
            2,
            29546,
            3267,
            8,
            3609,
            4,
            1,
            4865,
            1075,
            2,
            6087,
            71,
            6,
            346,
            8,
            5854,
            3,
            29546,
            824,
            1400,
            1868,
            2,
            19,
            160,
            2,
            311,
            8,
            5496,
            2,
            20920,
            17,
            25,
            15097,
            3,
            24,
            24,
            0,
            33,
            1,
            142,
            1298,
            188,
            2,
            29546,
            113,
            8,
            3654,
            4,
            1,
            1109,
            7136,
            833,
            3,
            13,
            1645,
            4,
            29546,
            11,
            104,
            7,
            1,
            1109,
            532,
            7129,
            2,
            10,
            83507,
            2,
            1162,
            1123,
            2,
            6,
            7245,
            10,
            2,
            5,
            11,
            104,
            7,
            1,
            1109,
            532,
            7129,
            2,
            10,
            24,
            24,
            10,
            22,
            10,
            13,
            770,
            5863,
            4,
            7245,
            10,
        ]
        #  In 1991, the remains of Russian Tsar Nicholas II and his family ( except for
        #  Alexei and Maria ) are discovered. The voice of young son, Tsarevich Alexei
        #  Nikolaevich, narrates the remainder of the story. 1883 Western Siberia, a young
        #  Grigori Rasputin is asked by his father and a group of men to perform magic.
        #  Rasputin has a vision and denounces one of the men as a horse thief. Although
        #  his father initially slaps him for making such an accusation, Rasputin watches
        #  as the man is chased outside and beaten. Twenty years later, Rasputin sees a
        #  vision of the Virgin Mary, prompting him to become a priest. Rasputin quickly
        #  becomes famous, with people, even a bishop, begging for his blessing. In the
        #  early 20th century, Rasputin became a symbol of the Russian Orthodox Church.
        #  The image of Rasputin was used in the Russian national anthem, " Nearer, My God,
        #  to Heaven ", and was used in the Russian national anthem, " " ( " The Great Spirit
        #  of Heaven "

        output_ids = model.generate(input_ids, max_length=200, do_sample=False)
        self.assertListEqual(output_ids[0].tolist(), expected_output_ids)
