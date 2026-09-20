# Copyright 2018 Google T5 Authors and HuggingFace Inc. team.
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

from transformers import T5Tokenizer
from transformers.testing_utils import get_tests_dir, require_sentencepiece, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")


@require_sentencepiece
@require_tokenizers
class T5TokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "google-t5/t5-small"
    tokenizer_class = T5Tokenizer

    integration_expected_tokens = ['▁This', '▁is', '▁', 'a', '▁test', '▁', '😊', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fal', 's', 'é', '.', '▁', '生活的真谛是', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '▁', '<', 's', '>', '▁hi', '<', 's', '>', 'there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encode', 'd', ':', '▁Hello', '.', '▁But', '▁', 'i', 'r', 'd', '▁and', '▁', 'ปี', '▁', 'i', 'r', 'd', '▁', 'ด', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_token_ids = [100, 19, 3, 9, 794, 3, 2, 27, 47, 2170, 16, 668, 13527, 6, 11, 48, 19, 12553, 7, 154, 5, 3, 2, 2018, 8774, 2018, 8774, 8774, 3, 2, 7, 3155, 7102, 2, 7, 3155, 12137, 37, 826, 6108, 225, 36, 3085, 23734, 26, 10, 8774, 5, 299, 3, 23, 52, 26, 11, 3, 2, 3, 23, 52, 26, 3, 2, 9459, 149, 33, 25, 692]  # fmt: skip
    expected_tokens_from_ids = ['▁This', '▁is', '▁', 'a', '▁test', '▁', '<unk>', '▁I', '▁was', '▁born', '▁in', '▁9', '2000', ',', '▁and', '▁this', '▁is', '▁fal', 's', 'é', '.', '▁', '<unk>', '▁Hi', '▁Hello', '▁Hi', '▁Hello', '▁Hello', '▁', '<unk>', 's', '>', '▁hi', '<unk>', 's', '>', 'there', '▁The', '▁following', '▁string', '▁should', '▁be', '▁properly', '▁encode', 'd', ':', '▁Hello', '.', '▁But', '▁', 'i', 'r', 'd', '▁and', '▁', '<unk>', '▁', 'i', 'r', 'd', '▁', '<unk>', '▁Hey', '▁how', '▁are', '▁you', '▁doing']  # fmt: skip
    integration_expected_decoded_text = "This is a test <unk> I was born in 92000, and this is falsé. <unk> Hi Hello Hi Hello Hello <unk>s> hi<unk>s>there The following string should be properly encoded: Hello. But ird and <unk> ird <unk> Hey how are you doing"

    def test_load_sentencepiece_without_precompiled_charsmap(self):
        """spm models with an empty/missing precompiled_charsmap must still load (e.g. google/umt5-xxl)."""
        import tempfile
        from pathlib import Path

        from transformers.convert_slow_tokenizer import SentencePieceExtractor, import_protobuf

        model_pb2 = import_protobuf()
        proto = model_pb2.ModelProto()
        with open(SAMPLE_VOCAB, "rb") as f:
            proto.ParseFromString(f.read())
        proto.normalizer_spec.ClearField("precompiled_charsmap")
        self.assertEqual(len(proto.normalizer_spec.precompiled_charsmap), 0)

        # Root cause: protobuf returns b"" for unset bytes fields; extractor must treat that as absent.
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "spiece.model"
            model_path.write_bytes(proto.SerializeToString())
            extracted = SentencePieceExtractor(str(model_path)).extract(self.tokenizer_class.model)
            self.assertIsNone(extracted.get("_spm_precompiled_charsmap"))

            hub = Path(tmp) / "tok"
            hub.mkdir()
            (hub / "spiece.model").write_bytes(model_path.read_bytes())
            (hub / "tokenizer_config.json").write_text('{"extra_ids": 0}')
            (hub / "special_tokens_map.json").write_text(
                '{"eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"}'
            )
            tokenizer = self.tokenizer_class.from_pretrained(str(hub))
            encoded = tokenizer("hello world")
            self.assertIn("input_ids", encoded)
            self.assertGreater(len(encoded["input_ids"]), 0)

            # save/reload round-trip
            save_dir = Path(tmp) / "saved"
            tokenizer.save_pretrained(save_dir)
            reloaded = self.tokenizer_class.from_pretrained(str(save_dir))
            self.assertEqual(reloaded("hello world")["input_ids"], encoded["input_ids"])
