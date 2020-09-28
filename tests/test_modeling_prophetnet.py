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
from transformers.testing_utils import slow, torch_device


if is_torch_available():
    import torch

    from transformers import ProphetNetForConditionalGeneration, ProphetNetTokenizer


class ProphetNetModelIntegrationTest(unittest.TestCase):
    @slow
    def test_pretrained_checkpoint_hidden_states(self):
        model = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased")
        model.to(torch_device)

        # encoder-decoder outputs
        encoder_ids = torch.tensor(
            [
                [
                    2871,
                    102,
                    2048,
                    3176,
                    2780,
                    1997,
                    2871,
                    26727,
                    2169,
                    2097,
                    12673,
                    1996,
                    8457,
                    2006,
                    2049,
                    8240,
                    2859,
                    2799,
                    1012,
                    2023,
                    6512,
                    2038,
                    2174,
                    13977,
                    2195,
                    25962,
                    1012,
                    102,
                ]
            ]
        ).to(torch_device)

        decoder_prev_ids = torch.tensor([[102, 2129, 2116, 2372, 2024, 2006, 2169, 1997, 2122, 2048, 2780, 1029]]).to(
            torch_device
        )
        output = model(
            input_ids=encoder_ids, attention_mask=None, encoder_outputs=None, decoder_input_ids=decoder_prev_ids
        )
        output_predited_logis = output[0]
        expected_shape = torch.Size((1, 12, 30522))
        self.assertEqual(output_predited_logis.shape, expected_shape)
        expected_slice = torch.tensor(
            [[[-7.6213, -7.9008, -7.9979], [-7.6834, -7.8467, -8.2187], [-7.5326, -7.4762, -8.1914]]]
        ).to(torch_device)
        self.assertTrue(torch.allclose(output_predited_logis[:, :3, :3], expected_slice, atol=1e-4))

        # encoder outputs
        encoder_outputs = model.model.encoder(encoder_ids)[0]
        expected_encoder_outputs_slice = torch.tensor(
            [[[-0.2526, -0.1951, -0.2185], [-0.8923, 0.2992, -0.4623], [-0.4585, 0.0165, -0.6652]]]
        ).to(torch_device)
        expected_shape_encoder = torch.Size((1, 28, 1024))
        self.assertEqual(encoder_outputs.shape, expected_shape_encoder)
        self.assertTrue(torch.allclose(encoder_outputs[:, :3, :3], expected_encoder_outputs_slice, atol=1e-4))

        # decoder outputs
        decoder_outputs = model.model.decoder(
            decoder_prev_ids, encoder_hidden_states=encoder_outputs, encoder_padding_mask=None
        )
        predicting_streams = decoder_outputs[0][1:]
        predicting_streams_logits = [model.output_layer(x) for x in predicting_streams]
        next_first_stream_logits = predicting_streams_logits[0]
        self.assertTrue(torch.allclose(next_first_stream_logits[:, :3, :3], expected_slice, atol=1e-4))

    @slow
    def test_cnndm_inference(self):
        model = ProphetNetForConditionalGeneration.from_pretrained("microsoft/prophetnet-large-uncased-cnndm")
        model.to(torch_device)

        tokenizer = ProphetNetTokenizer.from_pretrained("microsoft/prophetnet-large-uncased-cnndm")

        ARTICLE_TO_SUMMARIZE = "USTC was founded in Beijing by the Chinese Academy of Sciences (CAS) in September 1958. The Director of CAS, Mr. Guo Moruo was appointed the first president of USTC. USTC's founding mission was to develop a high-level science and technology workforce, as deemed critical for development of China's economy, defense, and science and technology education. The establishment was hailed as \"A Major Event in the History of Chinese Education and Science.\" CAS has supported USTC by combining most of its institutes with the departments of the university. USTC is listed in the top 16 national key universities, becoming the youngest national key university.".lower()
        input_ids = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=511, return_tensors="pt").input_ids

        input_ids = input_ids.to(torch_device)

        summary_ids = model.generate(
            input_ids, num_beams=4, length_penalty=1.0, no_repeat_ngram_size=3, early_stopping=True
        )
        EXPECTED_SUMMARIZE_512 = "us ##tc was founded by the chinese academy of sciences ( cas ) in 1958 . [X_SEP] us ##tc is listed in the top 16 national key universities ."
        generated_titles = [
            " ".join(tokenizer.convert_ids_to_tokens(g, skip_special_tokens=True)) for g in summary_ids
        ]
        self.assertListEqual(
            [EXPECTED_SUMMARIZE_512],
            generated_titles,
        )
        input_ids = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=99, return_tensors="pt").input_ids
        input_ids = input_ids.to(torch_device)
        # actually 98 tokens are used. max_length=100 contains bos and eos.
        # print(' '.join(tokenizer.tokenize(ARTICLE_TO_SUMMARIZE)[:98]))
        summary_ids = model.generate(
            input_ids, num_beams=4, length_penalty=1.0, no_repeat_ngram_size=3, early_stopping=True
        )
        EXPECTED_SUMMARIZE_100 = (
            r"us ##tc was founded in beijing by the chinese academy of sciences ( cas ) in 1958 . [X_SEP] us ##tc "
            "'"
            ' s founding mission was to develop a high - level science and technology workforce . [X_SEP] establishment hailed as " a major event in the history of chinese education and science "'
        )
        generated_titles = [
            " ".join(tokenizer.convert_ids_to_tokens(g, skip_special_tokens=True)) for g in summary_ids
        ]
        self.assertListEqual(
            [EXPECTED_SUMMARIZE_100],
            generated_titles,
        )
