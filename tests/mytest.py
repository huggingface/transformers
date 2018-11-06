import unittest
import json
import random

import torch
import numpy as np

import modeling
import convert_tf_checkpoint_to_pytorch

import grouch


class MyTest(unittest.TestCase):
    def test_loading_and_running(self):
        bertpath = "../../grouch/data/bert/bert-base/"
        configpath = bertpath + "bert_config.json"
        ckptpath = bertpath + "bert_model.ckpt"
        m = convert_tf_checkpoint_to_pytorch.convert(configpath, ckptpath)
        m.eval()
        # print(m)

        input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        all_y, pool_y = m(input_ids, token_type_ids, input_mask)
        print(pool_y.shape)
        # np.save("_bert_ref_pool_out.npy", pool_y.detach().numpy())
        # np.save("_bert_ref_all_out.npy", torch.stack(all_y, 0).detach().numpy())

        config = grouch.TransformerBERT.load_config(configpath)
        gm = grouch.TransformerBERT.init_from_config(config)
        gm.load_weights_from_tf_checkpoint(ckptpath)
        gm.eval()

        g_all_y, g_pool_y = gm(input_ids, token_type_ids, input_mask)
        print(g_pool_y.shape)

        # check embeddings
        # print(m.embeddings)
        # print(gm.emb)
        # hugging_emb = m.embeddings(input_ids, token_type_ids)
        # grouch_emb = gm.emb(input_ids, token_type_ids)

        print((all_y[0] - g_all_y[0]).norm())
        # print(all_y[0][:, :, :10] - g_all_y[0][:, :, :10])
        self.assertTrue(np.allclose(all_y[0].detach().numpy(), g_all_y[0].detach().numpy(), atol=1e-7))
        print("embeddings good")

        print(m.encoder.layer[0])
        print(gm.encoder.layers[0])
        print("norm of diff at layer 1", (all_y[1] - g_all_y[1]).norm())
        # print(all_y[1][:, :, :10] - g_all_y[1][:, :, :10])
        self.assertTrue(np.allclose(all_y[1].detach().numpy(), g_all_y[1].detach().numpy(), atol=1e-6))

        # hugging_layer = m.encoder.layer[0]
        # grouch_layer = gm.encoder.layers[0]
        # print("comparing weights")
        # print((hugging_layer.attention.self.query.weight - grouch_layer.slf_attn.q_proj.weight).norm())
        # print((hugging_layer.attention.self.query.bias - grouch_layer.slf_attn.q_proj.bias).norm())
        # print((hugging_layer.attention.self.key.weight - grouch_layer.slf_attn.k_proj.weight).norm())
        # print((hugging_layer.attention.self.key.bias - grouch_layer.slf_attn.k_proj.bias).norm())
        # print((hugging_layer.attention.self.value.weight - grouch_layer.slf_attn.v_proj.weight).norm())
        # print((hugging_layer.attention.self.value.bias - grouch_layer.slf_attn.v_proj.bias).norm())
        # print((hugging_layer.attention.output.dense.weight - grouch_layer.slf_attn.vw_proj.weight).norm())
        # print((hugging_layer.attention.output.dense.bias - grouch_layer.slf_attn.vw_proj.bias).norm())

        print("norm of diff at last layer", (all_y[-1] - g_all_y[-1]).norm())
        # print(all_y[-1][:, :, :10] - g_all_y[-1][:, :, :10])
        self.assertTrue(np.allclose(all_y[-1].detach().numpy(), g_all_y[-1].detach().numpy(), atol=1e-4))