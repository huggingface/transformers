# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import unittest
import json
import random
import shutil
import pytest

import torch

from pytorch_pretrained_bert import (XLNetConfig, XLNetModel, XLNetLMHeadModel)
from pytorch_pretrained_bert.modeling_xlnet import PRETRAINED_MODEL_ARCHIVE_MAP

class XLNetModelTest(unittest.TestCase):
    class XLNetModelTester(object):

        def __init__(self,
                     parent,
                     batch_size=13,
                     seq_length=7,
                     mem_len=10,
                     clamp_len=-1,
                     reuse_len=15,
                     is_training=True,
                     use_labels=True,
                     vocab_size=99,
                     cutoffs=[10, 50, 80],
                     d_model=32,
                     n_head=4,
                     d_inner=128,
                     n_layer=5,
                     max_position_embeddings=10,
                     untie_r=True,
                     bi_data=False,
                     same_length=False,
                     seed=1,
                     type_vocab_size=2):
            self.parent = parent
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.mem_len = mem_len
            self.clamp_len = clamp_len
            self.reuse_len = reuse_len
            self.is_training = is_training
            self.use_labels = use_labels
            self.vocab_size = vocab_size
            self.cutoffs = cutoffs
            self.d_model = d_model
            self.n_head = n_head
            self.d_inner = d_inner
            self.n_layer = n_layer
            self.max_position_embeddings = max_position_embeddings
            self.bi_data = bi_data
            self.untie_r = untie_r
            self.same_length = same_length
            self.seed = seed
            self.type_vocab_size = type_vocab_size

        def prepare_config_and_inputs(self):
            input_ids_1 = XLNetModelTest.ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            input_ids_2 = XLNetModelTest.ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
            segment_ids = XLNetModelTest.ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

            input_ids_q = XLNetModelTest.ids_tensor([self.batch_size, self.seq_length + 1], self.vocab_size)
            perm_mask = torch.zeros(self.batch_size, self.seq_length + 1, self.seq_length + 1, dtype=torch.float)
            perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
            target_mapping = torch.zeros(self.batch_size, 1, self.seq_length + 1, dtype=torch.float)
            target_mapping[:, 0, -1] = 1.0  # predict last token
            inp_q = target_mapping[:, 0, :].clone()  # predict last token

            # inp_k: int32 Tensor in shape [bsz, len], the input token IDs.
            # token_type_ids: int32 Tensor in shape [bsz, len], the input segment IDs.
            # input_mask: float32 Tensor in shape [bsz, len], the input mask.
            #     0 for real tokens and 1 for padding.
            # mems: a list of float32 Tensors in shape [bsz, mem_len, d_model], memory
            #     from previous batches. The length of the list equals n_layer.
            #     If None, no memory is used.
            # perm_mask: float32 Tensor in shape [bsz, len, len].
            #     If perm_mask[k, i, j] = 0, i attend to j in batch k;
            #     if perm_mask[k, i, j] = 1, i does not attend to j in batch k.
            #     If None, each position attends to all the others.
            # target_mapping: float32 Tensor in shape [bsz, num_predict, len].
            #     If target_mapping[k, i, j] = 1, the i-th predict in batch k is
            #     on the j-th token.
            #     Only used during pretraining for partial prediction.
            #     Set to None during finetuning.
            # inp_q: float32 Tensor in shape [bsz, len].
            #     1 for tokens with losses and 0 for tokens without losses.
            #     Only used during pretraining for two-stream attention.
            #     Set to None during finetuning.

            lm_labels = None
            if self.use_labels:
                lm_labels = XLNetModelTest.ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

            config = XLNetConfig(
                vocab_size_or_config_json_file=self.vocab_size,
                d_model=self.d_model,
                n_head=self.n_head,
                d_inner=self.d_inner,
                n_layer=self.n_layer,
                untie_r=self.untie_r,
                max_position_embeddings=self.max_position_embeddings,
                mem_len=self.mem_len,
                clamp_len=self.clamp_len,
                same_length=self.same_length,
                reuse_len=self.reuse_len,
                bi_data=self.bi_data)

            return (config, input_ids_1, input_ids_2, input_ids_q, perm_mask, target_mapping, inp_q, segment_ids, lm_labels)

        def set_seed(self):
            random.seed(self.seed)
            torch.manual_seed(self.seed)

        def create_transfo_xl_lm_head(self, config, input_ids_1, input_ids_2, input_ids_q, perm_mask, target_mapping, inp_q, segment_ids, lm_labels):
            model = XLNetLMHeadModel(config)
            model.eval()

            loss_1, all_logits_1, mems_1 = model(input_ids_1, token_type_ids=segment_ids, labels=lm_labels)

            loss_2, all_logits_2, mems_2 = model(input_ids_2, token_type_ids=segment_ids, labels=lm_labels, mems=mems_1)

            logits, _ = model(input_ids_q, perm_mask=perm_mask, target_mapping=target_mapping, inp_q=inp_q)

            outputs = {
                "loss_1": loss_1,
                "mems_1": mems_1,
                "all_logits_1": all_logits_1,
                "loss_2": loss_2,
                "mems_2": mems_2,
                "all_logits_2": all_logits_2,
            }
            return outputs

        def check_transfo_xl_lm_head_output(self, result):
            self.parent.assertListEqual(
                list(result["loss_1"].size()),
                [])
            self.parent.assertListEqual(
                list(result["all_logits_1"].size()),
                [self.batch_size, self.seq_length, self.vocab_size])
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_1"]),
                [[self.seq_length, self.batch_size, self.d_model]] * self.n_layer)

            self.parent.assertListEqual(
                list(result["loss_2"].size()),
                [])
            self.parent.assertListEqual(
                list(result["all_logits_2"].size()),
                [self.batch_size, self.seq_length, self.vocab_size])
            self.parent.assertListEqual(
                list(list(mem.size()) for mem in result["mems_2"]),
                [[self.mem_len, self.batch_size, self.d_model]] * self.n_layer)

    def test_default(self):
        self.run_tester(XLNetModelTest.XLNetModelTester(self))

    def test_config_to_json_string(self):
        config = XLNetConfig(vocab_size_or_config_json_file=96, d_model=16*4)
        obj = json.loads(config.to_json_string())
        self.assertEqual(obj["n_token"], 96)
        self.assertEqual(obj["d_model"], 16*4)

    def test_config_to_json_file(self):
        config_first = XLNetConfig(vocab_size_or_config_json_file=96, d_model=16*4)
        json_file_path = "/tmp/config.json"
        config_first.to_json_file(json_file_path)
        config_second = XLNetConfig.from_json_file(json_file_path)
        os.remove(json_file_path)
        self.assertEqual(config_second.to_dict(), config_first.to_dict())

    @pytest.mark.slow
    def test_model_from_pretrained(self):
        cache_dir = "/tmp/pytorch_pretrained_bert_test/"
        for model_name in list(PRETRAINED_MODEL_ARCHIVE_MAP.keys())[:1]:
            model = XLNetModel.from_pretrained(model_name, cache_dir=cache_dir)
            shutil.rmtree(cache_dir)
            self.assertIsNotNone(model)

    def run_tester(self, tester):
        config_and_inputs = tester.prepare_config_and_inputs()

        tester.set_seed()
        output_result = tester.create_transfo_xl_lm_head(*config_and_inputs)
        tester.check_transfo_xl_lm_head_output(output_result)

    @classmethod
    def ids_tensor(cls, shape, vocab_size, rng=None, name=None):
        """Creates a random int32 tensor of the shape within the vocab size."""
        if rng is None:
            rng = random.Random()

        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(rng.randint(0, vocab_size - 1))

        return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()

    @classmethod
    def mask_tensor(cls, shape, vocab_size, rng=None, name=None):
        """Creates a tensor with padding on the right (0.0 for )."""
        if rng is None:
            rng = random.Random()

        total_dims = 1
        for dim in shape:
            total_dims *= dim

        values = []
        for _ in range(total_dims):
            values.append(rng.randint(0, vocab_size - 1))

        return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


if __name__ == "__main__":
    unittest.main()
