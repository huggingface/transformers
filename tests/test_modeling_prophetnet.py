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


import unittest

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

if is_torch_available():
    import torch
    from transformers import (
        ProphetNetConfig,
        ProphetNetModel,
        ProphetNetForConditionalGeneration,
        ProphetNetTokenizer
    )
    from transformers.modeling_prophetnet import _relative_positions_bucket
    from transformers.modeling_prophetnet import PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST


class ProphetNetModelIntegrationTest(unittest.TestCase):

    @slow
    def test_prophetnet_pretrained_checkpoint_hidden_states(self):
        model = ProphetNetForConditionalGeneration.from_pretrained(
            'microsoft/prophetnet-large-uncased')

        # encoder-decoder outputs
        encoder_ids = torch.tensor(
            [[102, 2871, 102, 2048, 3176, 2780, 1997, 2871, 26727, 2169, 2097,
                     12673, 1996, 8457, 2006, 2049, 8240, 2859, 2799, 1012, 2023,
                     6512, 2038, 2174, 13977, 2195, 25962, 1012, 102]])  # notice the bos of encoder_ids will be removed in the model

        decoder_prev_ids = torch.tensor([[ 102, 2129, 2116, 2372, 2024, 2006, 2169, 1997, 2122, 2048, 2780, 1029]])
        output = model(input_ids=encoder_ids, attention_mask=None, encoder_outputs=None,
                       decoder_input_ids=decoder_prev_ids)
        output_predited_logis = output[0]
        expected_shape = torch.Size((1, 12, 30522))
        self.assertEqual(output_predited_logis.shape, expected_shape)
        expected_slice = torch.tensor(
            [[[-7.6213, -7.9008, -7.9979], [-7.6834, -7.8467, -8.2187], [-7.5326, -7.4762, -8.1914]]]
        )
        self.assertTrue(torch.allclose(output_predited_logis[:, :3, :3], expected_slice, atol=1e-4))

        # encoder outputs
        encoder_outputs = model.model.encoder(encoder_ids)[0]
        expected_encoder_outputs_slice = torch.tensor(
            [[[-0.2526, -0.1951, -0.2185], [-0.8923, 0.2992, -0.4623], [-0.4585, 0.0165, -0.6652]]]
        )
        expected_shape_encoder = torch.Size((1, 28, 1024))
        self.assertEqual(encoder_outputs.shape, expected_shape_encoder)
        self.assertTrue(torch.allclose(encoder_outputs[:, :3, :3], expected_encoder_outputs_slice, atol=1e-4))

        # decoder outputs
        decoder_outputs = model.model.decoder(decoder_prev_ids, encoder_hidden_states=encoder_outputs,
                                              encoder_padding_mask=None)
        predicting_streams = decoder_outputs[0][1:]
        predicting_streams_logits = [model.output_layer(x) for x in predicting_streams]
        next_first_stream_logits = predicting_streams_logits[0]
        self.assertTrue(torch.allclose(next_first_stream_logits[:, :3, :3], expected_slice, atol=1e-4))

    @slow
    def test_xprophetnet_pretrained_checkpoint_hidden_states(self):
        model = ProphetNetForConditionalGeneration.from_pretrained(
            'microsoft/xprophetnet-large-wiki100-cased')

        # encoder-decoder outputs
        encoder_ids = torch.tensor(
            [[2, 17, 96208, 103471, 2]])  # notice the bos of encoder_ids will be removed in the model
        decoder_prev_ids = torch.tensor([[2, 250, 9953, 34, 69489, 1620, 32, 118424, 624, 210, 105, 2913, 1032, 351]])
        output = model(input_ids=encoder_ids, attention_mask=None, encoder_outputs=None,
                       decoder_input_ids=decoder_prev_ids)
        output_predited_logis = output[0]
        expected_shape = torch.Size((1, 14, 250012))
        self.assertEqual(output_predited_logis.shape, expected_shape)
        expected_slice = torch.tensor(
            [[[-6.6042, -8.3838, 12.4717], [-6.4426, -8.1994, 12.4542], [-6.0851, -7.8209, 12.9493]]]
        )
        self.assertTrue(torch.allclose(output_predited_logis[:, :3, :3], expected_slice, atol=1e-4))

        # encoder outputs
        encoder_outputs = model.model.encoder(encoder_ids)[0]
        expected_encoder_outputs_slice = torch.tensor(
            [[[-1.4260, -0.7628, 0.8453], [-1.4719, -0.1391, 0.7807], [-1.7678, 0.0114, 0.4646]]]
        )
        expected_shape_encoder = torch.Size((1, 4, 1024))
        self.assertEqual(encoder_outputs.shape, expected_shape_encoder)
        self.assertTrue(torch.allclose(encoder_outputs[:, :3, :3], expected_encoder_outputs_slice, atol=1e-4))

        # decoder outputs
        decoder_outputs = model.model.decoder(decoder_prev_ids, encoder_hidden_states=encoder_outputs,
                                              encoder_padding_mask=None)
        predicting_streams = decoder_outputs[0][1:]
        predicting_streams_logits = [model.output_layer(x) for x in predicting_streams]
        next_first_stream_logits = predicting_streams_logits[0]
        self.assertTrue(torch.allclose(next_first_stream_logits[:, :3, :3], expected_slice, atol=1e-4))

    @slow
    def test_xprophetnet_ntg_hidden_states(self):
        model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/xprophetnet-large-wiki100-cased-xglue-ntg')

        encoder_ids = torch.tensor([[2, 17, 96208, 103471, 2]]) # notice the bos of encoder_ids will be removed in the model
        decoder_prev_ids = torch.tensor([[2, 250,  9953, 34, 69489, 1620, 32, 118424, 624, 210, 105, 2913, 1032, 351]])
        output = model(input_ids=encoder_ids, attention_mask=None, encoder_outputs=None, decoder_input_ids=decoder_prev_ids)
        output_predited_logis = output[0]
        expected_shape = torch.Size((1, 14, 250012))
        self.assertEqual(output_predited_logis.shape, expected_shape)
        # compare the actual values for a slice.
        expected_slice = torch.tensor(
            [[[-8.8815, -9.2996, -4.4506], [-6.7202, -7.8944, -0.9402], [-8.6890, -7.4528, -1.9437]]]
        )

        self.assertTrue(torch.allclose(output_predited_logis[:, :3, :3], expected_slice, atol=1e-4))

    @slow
    def test_xprophetnet_ntg_inference(self):
        model = ProphetNetForConditionalGeneration.from_pretrained(
            'microsoft/xprophetnet-large-wiki100-cased-xglue-ntg')
        tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/xprophetnet-large-wiki100-cased-xglue-ntg')
        EN_SENTENCE = "Microsoft Corporation intends to officially end free support for the Windows 7 operating system after January 14, 2020, according to the official portal of the organization. From that day, users of this system will not be able to receive security updates, which could make their computers vulnerable to cyber attacks."
        RU_SENTENCE = "орпорация Microsoft намерена официально прекратить бесплатную поддержку операционной системы Windows 7 после 14 января 2020 года, сообщается на официальном портале организации . С указанного дня пользователи этой системы не смогут получать обновления безопасности, из-за чего их компьютеры могут стать уязвимыми к кибератакам."
        ZH_SENTENCE = "根据该组织的官方门户网站，微软公司打算在2020年1月14日之后正式终止对Windows 7操作系统的免费支持。从那时起，该系统的用户将无法接收安全更新，这可能会使他们的计算机容易受到网络攻击。"
        inputs = tokenizer([EN_SENTENCE, RU_SENTENCE, ZH_SENTENCE], padding=True, max_length=256, return_tensors='pt')
        summary_ids = model.generate(
            inputs['input_ids'],
            num_beams=10,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True)
        generated_titles = [tokenizer.decode(g, skip_special_tokens=True) for g in summary_ids]
        EXPECTED_TITLE_EN = "Microsoft to end Windows 7 free support after January 14, 2020"
        EXPECTED_TITLE_RU = "Microsoft намерена прекратить бесплатную поддержку Windows 7 после 14 января 2020 года"
        EXPECTED_TITLE_ZH = "微软打算终止对Windows 7操作系统的免费支持"
        self.assertListEqual(
            [EXPECTED_TITLE_EN, EXPECTED_TITLE_RU, EXPECTED_TITLE_ZH],
            generated_titles,
        )

        summary_ids_beam1 = model.generate(
            inputs['input_ids'],
            num_beams=1,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True)
        generated_titles_beam1_tok  = [tokenizer.convert_ids_to_tokens(g, skip_special_tokens=True) for g in summary_ids_beam1]
        EXPECTED_TITLE_EN_BEAM1_TOK = "▁Microsoft ▁to ▁end ▁free ▁support ▁for ▁Windows ▁7".split(' ')
        EXPECTED_TITLE_RU_BEAM1_TOK = '▁Microsoft ▁намерен а ▁прекрати ть ▁бес плат ную ▁поддержку ▁Windows ▁7 ▁после ▁14 ▁января ▁2020 ▁года'.split(' ')
        EXPECTED_TITLE_ZH_BEAM1_TOK = '微软 公司 打算 终止 对 Windows ▁7 操作 系统的 免费 支持'.split(' ')
        print([EXPECTED_TITLE_EN_BEAM1_TOK, EXPECTED_TITLE_RU_BEAM1_TOK, EXPECTED_TITLE_ZH_BEAM1_TOK])
        print(generated_titles_beam1_tok)
        self.assertListEqual(
            [EXPECTED_TITLE_EN_BEAM1_TOK, EXPECTED_TITLE_RU_BEAM1_TOK, EXPECTED_TITLE_ZH_BEAM1_TOK],
            generated_titles_beam1_tok,
        )

    @slow
    def test_prophetnet_cnndm_inference(self):
        model = ProphetNetForConditionalGeneration.from_pretrained('microsoft/prophetnet-large-uncased-cnndm')
        tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased-cnndm')
        ARTICLE_TO_SUMMARIZE = "USTC was founded in Beijing by the Chinese Academy of Sciences (CAS) in September 1958. The Director of CAS, Mr. Guo Moruo was appointed the first president of USTC. USTC's founding mission was to develop a high-level science and technology workforce, as deemed critical for development of China's economy, defense, and science and technology education. The establishment was hailed as \"A Major Event in the History of Chinese Education and Science.\" CAS has supported USTC by combining most of its institutes with the departments of the university. USTC is listed in the top 16 national key universities, becoming the youngest national key university.".lower()
        inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=512, return_tensors='pt')
        summary_ids = model.generate(
            inputs['input_ids'],
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True)
        EXPECTED_SUMMARIZE_512 = 'us ##tc was founded in beijing in 1958 by the chinese academy of sciences . [X_SEP] us ##tc is listed in the top 16 national key universities .'
        generated_titles = [' '.join(tokenizer.convert_ids_to_tokens(g, skip_special_tokens=True)) for g in summary_ids]
        self.assertListEqual(
            [EXPECTED_SUMMARIZE_512],
            generated_titles,
        )
        inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=100, return_tensors='pt')
        # actually 98 tokens are used. max_length=100 contains bos and eos.
        # print(' '.join(tokenizer.tokenize(ARTICLE_TO_SUMMARIZE)[:98]))
        summary_ids = model.generate(
            inputs['input_ids'],
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True)
        EXPECTED_SUMMARIZE_100 = r'us ##tc was founded in beijing by the chinese academy of sciences ( cas ) in 1958 . [X_SEP] its mission was to develop a high - ##lev ##el science and technology workforce . [X_SEP] the establishment was hailed as " a major event in the history of chinese education "'
        generated_titles = [' '.join(tokenizer.convert_ids_to_tokens(g, skip_special_tokens=True)) for g in summary_ids]
        self.assertListEqual(
            [EXPECTED_SUMMARIZE_100],
            generated_titles,
        )








