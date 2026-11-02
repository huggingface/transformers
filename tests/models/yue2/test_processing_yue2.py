# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import tempfile
import unittest

from transformers import AddedToken, Qwen2Tokenizer, YuE2Processor
from transformers.convert_slow_tokenizer import bytes_to_unicode
from transformers.testing_utils import require_torch, slow
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers import YuE2AcousticTokenizerConfig, YuE2AcousticTokenizerModel


def get_tiny_tokenizer():
    """A byte-level tokenizer without merges (one token per byte), with YuE2's named special tokens."""
    vocab = {character: index for index, character in enumerate(bytes_to_unicode().values())}
    tokenizer = Qwen2Tokenizer(vocab=vocab, merges=[], bos_token="<|endoftext|>", padding_side="left")
    tokenizer.add_tokens(
        [AddedToken(token, special=True) for token in ["<abc>", "</abc>", "<music>"]], special_tokens=True
    )
    return tokenizer


@require_torch
class YuE2ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.processor = YuE2Processor(get_tiny_tokenizer())
        self.tokenizer = self.processor.tokenizer
        self.bos, self.abc_start, self.abc_end, self.music_start = self.tokenizer.convert_tokens_to_ids(
            ["<|endoftext|>", "<abc>", "</abc>", "<music>"]
        )

    def _encode(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _text(self, cot, lyrics, style):
        return f"{self.processor.instructions[cot]}\n[Tags]\n{style}\n[Lyrics]\n{lyrics}\n"

    def test_prompts(self):
        lyrics, style, abc = "[verse]\nla la", "pop", "X:1\nC D |]"
        for cot, score, suffix in [
            ("full", None, [self.abc_start]),
            ("melody", None, [self.abc_start]),
            ("off", None, [self.abc_start, self.abc_end, self.music_start]),
            ("full", abc, [self.abc_start] + self._encode(abc) + [self.abc_end, self.music_start]),
        ]:
            inputs = self.processor(lyrics=lyrics, style=style, cot=cot, abc=score)
            expected = [self.bos] + self._encode(self._text(cot, lyrics, style)) + suffix
            self.assertEqual(inputs["input_ids"][0].tolist(), expected)
            self.assertTrue(inputs["attention_mask"].bool().all())
            # the negative prompt is the instruction alone, which `generate` completes with the score
            self.assertEqual(
                inputs["negative_input_ids"][0].tolist(), [self.bos] + self._encode(self.processor.instructions[cot])
            )

    def test_batch_left_padding(self):
        inputs = self.processor(lyrics=["a", "a much longer verse"], style="pop")
        lengths = inputs["attention_mask"].sum(-1).tolist()
        self.assertLess(lengths[0], lengths[1])
        # left padding: every prompt ends on the open score
        self.assertTrue((inputs["input_ids"][:, -1] == self.abc_start).all())
        self.assertTrue(
            (inputs["input_ids"][0, : inputs["input_ids"].shape[1] - lengths[0]] == self.tokenizer.pad_token_id).all()
        )
        single = self.processor(lyrics="a", style="pop")
        self.assertEqual(inputs["input_ids"][0, -lengths[0] :].tolist(), single["input_ids"][0].tolist())

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.processor(lyrics="a", style="pop", cot="chords")
        with self.assertRaises(ValueError):
            self.processor(lyrics="a", style="pop", cot="off", abc="X:1")
        with self.assertRaises(ValueError):
            self.processor(lyrics=["a", "b"], style=["pop", "rock", "jazz"])

    def test_save_load(self):
        processor = YuE2Processor(self.tokenizer, instructions={"off": "Off.", "melody": "Melody.", "full": "Full."})
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor.save_pretrained(tmp_dir)
            reloaded = YuE2Processor.from_pretrained(tmp_dir)
        self.assertEqual(reloaded.instructions, processor.instructions)
        self.assertEqual(
            reloaded(lyrics="a", style="pop")["input_ids"].tolist(),
            processor(lyrics="a", style="pop")["input_ids"].tolist(),
        )

    def test_decode(self):
        config = YuE2AcousticTokenizerConfig(
            hidden_size=4, channel_multipliers=[1, 2], downsampling_ratios=[2, 3], latent_dim=6
        )
        audio_tokenizer = YuE2AcousticTokenizerModel(config).eval()
        processor = YuE2Processor(self.tokenizer, audio_tokenizer=audio_tokenizer)
        audio_latents = torch.randn(2, 5, config.latent_dim)
        audio_latents_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        waveforms = processor.decode(audio_latents, audio_latents_mask)
        with torch.no_grad():
            expected = audio_tokenizer.decode(audio_latents[1:, :3]).audio_values[0].clamp(-1, 1)
        torch.testing.assert_close(waveforms[1], expected)
        self.assertGreater(waveforms[0].shape[-1], waveforms[1].shape[-1])
        self.assertTrue(all(waveform.abs().max() <= 1 for waveform in waveforms))


@slow
@require_torch
class YuE2ProcessorIntegrationTest(unittest.TestCase):
    """
    Prompts of the original implementation (`token_prefixes` of the `yue2_infer` 0.1.5 wheel shipped with
    https://huggingface.co/m-a-p/YuE2-3B):

    ```python
    from huggingface_hub import hf_hub_download
    from yue2.protocol import SongRequest, token_prefixes
    from yue2.tokenization_yue2 import YuE2TextTokenizer

    tokenizer = YuE2TextTokenizer(hf_hub_download("m-a-p/YuE2-3B", "qwen.tiktoken"))
    request = SongRequest(style="Mandarin funk, nu-disco", lyrics="[verse]\\n今晚不眠\\nTonight we dance\\n", cot="full")
    print(token_prefixes(request, tokenizer))
    ```
    """

    checkpoint = "bezzam/yue2-3b"

    def test_prompt(self):
        processor = YuE2Processor.from_pretrained(self.checkpoint)
        inputs = processor(lyrics="[verse]\n今晚不眠\nTonight we dance\n", style="Mandarin funk, nu-disco", cot="full")
        # fmt: off
        EXPECTED_INPUT_IDS = [151643, 31115, 264, 43221, 12, 3401, 657, 19360, 45840, 11, 1221, 6923, 4627, 448, 34647, 11211, 504, 279, 2661, 4682, 624, 58, 15930, 921, 44, 437, 42740, 68292, 11, 10908, 1737, 9464, 198, 58, 47312, 6198, 921, 58, 4450, 921, 108779, 16530, 101519, 198, 89143, 582, 15254, 271, 151847]
        # fmt: on
        self.assertEqual(inputs["input_ids"][0].tolist(), EXPECTED_INPUT_IDS)
