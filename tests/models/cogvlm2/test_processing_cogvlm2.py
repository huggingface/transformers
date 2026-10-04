# Copyright 2026 The HuggingFace Team. All rights reserved.

import unittest

import torch

from transformers import CogVLM2Processor, CogVLM2VideoProcessor, PreTrainedTokenizer
from transformers.testing_utils import require_torch, require_torchvision


class TinyCogVLM2Tokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :?.,!\\n")
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        for char in chars:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        self.inverse_vocab = {value: key for key, value in self.vocab.items()}
        super().__init__(pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>")

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab)

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab["<unk>"])

    def _convert_id_to_token(self, index):
        return self.inverse_vocab.get(index, "<unk>")

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()


@require_torch
@require_torchvision
class CogVLM2ProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TinyCogVLM2Tokenizer()
        self.video_processor = CogVLM2VideoProcessor()
        self.processor = CogVLM2Processor(
            video_processor=self.video_processor,
            tokenizer=self.tokenizer,
        )

    def test_builds_66_visual_tokens_per_frame(self):
        video = torch.randint(0, 256, (2, 3, 240, 320), dtype=torch.uint8)
        outputs = self.processor(text="What?", videos=video, return_tensors="pt")

        self.assertEqual(int((outputs.token_type_ids == 1).sum()), 132)
        self.assertEqual(outputs.pixel_values_videos.shape, (2, 3, 224, 224))

    def test_base_prompt_matches_config_default(self):
        video = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8)
        outputs = self.processor(text="What?", videos=video, template_version="chat", return_tensors="pt")

        text_start = 1 + 66
        decoded = self.tokenizer.decode(outputs.input_ids[0, text_start:], skip_special_tokens=True)
        self.assertEqual(decoded, "What?")

    def test_chat_prompt_matches_reference_format_and_adds_time_index(self):
        video = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8)
        outputs = self.processor(
            text="What?",
            videos=video,
            template_version="chat",
            return_tensors="pt",
        )

        language_ids = outputs.input_ids[0][outputs.token_type_ids[0] == 0]
        decoded = self.tokenizer.decode(language_ids, skip_special_tokens=True)
        self.assertEqual(decoded, "0Question: What? Answer:")

    def test_default_template_matches_checkpoint_base_mode(self):
        video = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8)
        outputs = self.processor(text="What?", videos=video, return_tensors="pt")

        text_start = 1 + 66
        decoded = self.tokenizer.decode(outputs.input_ids[0, text_start:], skip_special_tokens=True)
        self.assertEqual(decoded, "What?")

    def test_answer_labels_only_supervise_answer(self):
        video = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.uint8)
        outputs = self.processor(text="What?", videos=video, answer="OK", return_tensors="pt")

        self.assertTrue((outputs.labels[0, :-3] == -100).all())
        self.assertEqual(int((outputs.labels != -100).sum()), 3)

    def test_default_does_not_insert_time_indices(self):
        video = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
        outputs = self.processor(text="X", videos=video, return_tensors="pt")
        expected_prefix_length = 1 + 2 * 66

        first_text_id = self.tokenizer.encode("X", add_special_tokens=False)[0]
        self.assertEqual(outputs.input_ids[0, expected_prefix_length].item(), first_text_id)


if __name__ == "__main__":
    unittest.main()
