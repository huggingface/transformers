# Copyright 2021 The HuggingFace Team. All rights reserved.
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

import json
import os
import shutil
import tempfile
import unittest
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from transformers.file_utils import FEATURE_EXTRACTOR_NAME, is_pyctcdecode_available
from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.tokenization_wav2vec2 import VOCAB_FILES_NAMES
from transformers.testing_utils import require_pyctcdecode

from .test_feature_extraction_wav2vec2 import floats_list


if is_pyctcdecode_available():
    from pyctcdecode import BeamSearchDecoderCTC
    from transformers.models.wav2vec2_with_lm import Wav2Vec2ProcessorWithLM


@require_pyctcdecode
class Wav2Vec2ProcessorWithLMTest(unittest.TestCase):
    def setUp(self):
        vocab = "| <pad> <unk> <s> </s> a b c d e f g h i j k".split()
        vocab_tokens = dict(zip(vocab, range(len(vocab))))

        self.add_kwargs_tokens_map = {
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
        }
        feature_extractor_map = {
            "feature_size": 1,
            "padding_value": 0.0,
            "sampling_rate": 16000,
            "return_attention_mask": False,
            "do_normalize": True,
        }

        self.tmpdirname = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.feature_extraction_file = os.path.join(self.tmpdirname, FEATURE_EXTRACTOR_NAME)
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")

        with open(self.feature_extraction_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(feature_extractor_map) + "\n")

        # load decoder from hub
        self.decoder_name = "hf-internal-testing/ngram-beam-search-decoder"

    def get_tokenizer(self, **kwargs_init):
        kwargs = self.add_kwargs_tokens_map.copy()
        kwargs.update(kwargs_init)
        return Wav2Vec2CTCTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_feature_extractor(self, **kwargs):
        return Wav2Vec2FeatureExtractor.from_pretrained(self.tmpdirname, **kwargs)

    def get_decoder(self, **kwargs):
        return BeamSearchDecoderCTC.load_from_hf_hub(self.decoder_name, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()
        feature_extractor = self.get_feature_extractor()
        decoder = self.get_decoder()

        processor = Wav2Vec2ProcessorWithLM(tokenizer=tokenizer, feature_extractor=feature_extractor, decoder=decoder)

        processor.save_pretrained(self.tmpdirname)
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(self.tmpdirname)

        # tokenizer
        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertIsInstance(processor.tokenizer, Wav2Vec2CTCTokenizer)

        # feature extractor
        self.assertEqual(processor.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(processor.feature_extractor, Wav2Vec2FeatureExtractor)

        # decoder
        self.assertEqual(processor.decoder._alphabet.labels, decoder._alphabet.labels)
        self.assertEqual(
            processor.decoder.model_container[decoder._model_key]._unigram_set,
            decoder.model_container[decoder._model_key]._unigram_set,
        )
        self.assertIsInstance(processor.decoder, BeamSearchDecoderCTC)

    def test_save_load_pretrained_additional_features(self):
        processor = Wav2Vec2ProcessorWithLM(
            tokenizer=self.get_tokenizer(), feature_extractor=self.get_feature_extractor(), decoder=self.get_decoder()
        )
        processor.save_pretrained(self.tmpdirname)

        # make sure that error is thrown when decoder alphabet doesn't match
        processor = Wav2Vec2ProcessorWithLM.from_pretrained(
            self.tmpdirname, alpha=5.0, beta=3.0, score_boundary=-7.0, unk_score_offset=3
        )

        # decoder
        self.assertEqual(processor.language_model.alpha, 5.0)
        self.assertEqual(processor.language_model.beta, 3.0)
        self.assertEqual(processor.language_model.score_boundary, -7.0)
        self.assertEqual(processor.language_model.unk_score_offset, 3)

    def test_load_decoder_tokenizer_mismatch_content(self):
        tokenizer = self.get_tokenizer()
        # add token to trigger raise
        tokenizer.add_tokens(["xx"])
        with self.assertRaisesRegex(ValueError, "include"):
            Wav2Vec2ProcessorWithLM(
                tokenizer=tokenizer, feature_extractor=self.get_feature_extractor(), decoder=self.get_decoder()
            )

    def test_feature_extractor(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        decoder = self.get_decoder()

        processor = Wav2Vec2ProcessorWithLM(tokenizer=tokenizer, feature_extractor=feature_extractor, decoder=decoder)

        raw_speech = floats_list((3, 1000))

        input_feat_extract = feature_extractor(raw_speech, return_tensors="np")
        input_processor = processor(raw_speech, return_tensors="np")

        for key in input_feat_extract.keys():
            self.assertAlmostEqual(input_feat_extract[key].sum(), input_processor[key].sum(), delta=1e-2)

    def test_tokenizer(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        decoder = self.get_decoder()

        processor = Wav2Vec2ProcessorWithLM(tokenizer=tokenizer, feature_extractor=feature_extractor, decoder=decoder)

        input_str = "This is a test string"

        with processor.as_target_processor():
            encoded_processor = processor(input_str)

        encoded_tok = tokenizer(input_str)

        for key in encoded_tok.keys():
            self.assertListEqual(encoded_tok[key], encoded_processor[key])

    def _get_dummy_logits(self, shape=(2, 10, 16), seed=77):
        np.random.seed(seed)
        return np.random.rand(*shape)

    def test_decoder(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        decoder = self.get_decoder()

        processor = Wav2Vec2ProcessorWithLM(tokenizer=tokenizer, feature_extractor=feature_extractor, decoder=decoder)

        logits = self._get_dummy_logits(shape=(10, 16), seed=13)

        decoded_processor = processor.decode(logits).text

        decoded_decoder = decoder.decode_beams(logits)[0][0]

        self.assertEqual(decoded_decoder, decoded_processor)
        self.assertEqual("</s> <s> </s>", decoded_processor)

    def test_decoder_batch(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        decoder = self.get_decoder()

        processor = Wav2Vec2ProcessorWithLM(tokenizer=tokenizer, feature_extractor=feature_extractor, decoder=decoder)

        logits = self._get_dummy_logits()

        decoded_processor = processor.batch_decode(logits).text

        logits_list = [array for array in logits]
        pool = get_context("fork").Pool()
        decoded_decoder = [d[0][0] for d in decoder.decode_beams_batch(pool, logits_list)]
        pool.close()

        self.assertListEqual(decoded_decoder, decoded_processor)
        self.assertListEqual(["<s> <s> </s>", "<s> <s> <s>"], decoded_processor)

    def test_decoder_with_params(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        decoder = self.get_decoder()

        processor = Wav2Vec2ProcessorWithLM(tokenizer=tokenizer, feature_extractor=feature_extractor, decoder=decoder)

        logits = self._get_dummy_logits()

        beam_width = 20
        beam_prune_logp = -20.0
        token_min_logp = -4.0

        decoded_processor_out = processor.batch_decode(
            logits,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
        )
        decoded_processor = decoded_processor_out.text

        logits_list = [array for array in logits]
        pool = get_context("fork").Pool()
        decoded_decoder_out = decoder.decode_beams_batch(
            pool,
            logits_list,
            beam_width=beam_width,
            beam_prune_logp=beam_prune_logp,
            token_min_logp=token_min_logp,
        )
        pool.close()

        decoded_decoder = [d[0][0] for d in decoded_decoder_out]

        self.assertListEqual(decoded_decoder, decoded_processor)
        self.assertListEqual(["<s> </s> </s>", "<s> <s> </s>"], decoded_processor)

    def test_decoder_with_params_of_lm(self):
        feature_extractor = self.get_feature_extractor()
        tokenizer = self.get_tokenizer()
        decoder = self.get_decoder()

        processor = Wav2Vec2ProcessorWithLM(tokenizer=tokenizer, feature_extractor=feature_extractor, decoder=decoder)

        logits = self._get_dummy_logits()

        alpha = 2.0
        beta = 5.0
        unk_score_offset = -20.0
        lm_score_boundary = True

        decoded_processor_out = processor.batch_decode(
            logits,
            alpha=alpha,
            beta=beta,
            unk_score_offset=unk_score_offset,
            lm_score_boundary=lm_score_boundary,
        )
        decoded_processor = decoded_processor_out.text

        logits_list = [array for array in logits]
        decoder.reset_params(
            alpha=alpha,
            beta=beta,
            unk_score_offset=unk_score_offset,
            lm_score_boundary=lm_score_boundary,
        )
        pool = get_context("fork").Pool()
        decoded_decoder_out = decoder.decode_beams_batch(
            pool,
            logits_list,
        )
        pool.close()

        decoded_decoder = [d[0][0] for d in decoded_decoder_out]

        self.assertListEqual(decoded_decoder, decoded_processor)
        self.assertListEqual(["<s> </s> <s> </s> </s>", "</s> </s> <s> </s> </s>"], decoded_processor)
        lm_model = processor.decoder.model_container[processor.decoder._model_key]
        self.assertEqual(lm_model.alpha, 2.0)
        self.assertEqual(lm_model.beta, 5.0)
        self.assertEqual(lm_model.unk_score_offset, -20.0)
        self.assertEqual(lm_model.score_boundary, True)

    def test_decoder_download_ignores_files(self):
        processor = Wav2Vec2ProcessorWithLM.from_pretrained("hf-internal-testing/processor_with_lm")

        language_model = processor.decoder.model_container[processor.decoder._model_key]
        path_to_cached_dir = Path(language_model._kenlm_model.path.decode("utf-8")).parent.parent.absolute()

        downloaded_decoder_files = os.listdir(path_to_cached_dir)
        expected_decoder_files = ["alphabet.json", "language_model"]

        downloaded_decoder_files.sort()
        expected_decoder_files.sort()

        # test that only decoder relevant files from
        # https://huggingface.co/hf-internal-testing/processor_with_lm/tree/main
        # are downloaded and none of the rest (e.g. README.md, ...)
        self.assertListEqual(downloaded_decoder_files, expected_decoder_files)
