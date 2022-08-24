import unittest

from transformers import AutoTokenizer
from transformers.testing_utils import require_tokenizers


@require_tokenizers
class OPTTokenizationTest(unittest.TestCase):
    def test_serialize_deserialize_fast_opt(self):
        # More context:
        # https://huggingface.co/wjmcat/opt-350m-paddle/discussions/1
        # https://huggingface.slack.com/archives/C01N44FJDHT/p1653511495183519
        # https://github.com/huggingface/transformers/pull/17088#discussion_r871246439

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", from_slow=True)
        text = "A photo of a cat"

        tokens_ids = tokenizer.encode(
            text,
        )
        self.assertEqual(tokens_ids, [2, 250, 1345, 9, 10, 4758])
        tokenizer.save_pretrained("test_opt")

        tokenizer = AutoTokenizer.from_pretrained("./test_opt")
        tokens_ids = tokenizer.encode(
            text,
        )
        self.assertEqual(tokens_ids, [2, 250, 1345, 9, 10, 4758])
