import unittest

from datasets import load_dataset

from transformers import AutoTokenizer
from transformers.testing_utils import require_tokenizers


# TO PORT ON transformers.test_utils ?


@require_tokenizers
class BigScienceTokenizationTest(unittest.TestCase):
    """
    The goal here is to compare some text that has been tokenized by a model trained
    using Megatron-LM. For now:
        - Read the tokenized text (.bin file) + the raw text
        - Load a tokenizer from the hub (fast tokenizer or python tokenizer)
    You need to install tokenizers following this readme:
        - https://huggingface.co/bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles

    Tokenizer used during training:
        - https://huggingface.co/bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles
    Tokenizer that has been pushed to the hub:
        - https://huggingface.co/bigscience/tokenizer/ -> We do not use it


    This code is not device agnostic --> figure out what to do? / for now I am adding some decorators to check whether the data/tokenizers exists

    # TODO change the script (or just add skip) when building the env with tokenizers 0.12.0
    """

    def setUp(self):
        super().setUp()
        self.path_tokenizer = "bigscience-catalogue-data-dev/byte-level-bpe-tokenizer-no-norm-250k-whitespace-and-eos-regex-alpha-v3-dedup-lines-articles"
        self.NB_SENTENCES = 2

    def test_load_tokenizer(self):
        """
        Assert that we can correctly load the tokenizer that is available on the Hub
        """
        try:
            _ = AutoTokenizer.from_pretrained(self.path_tokenizer)
        except BaseException:
            self.fail("Failed loading tokenizer")

    def test_encodings_from_sample_data(self):
        """
        Assert that the created tokens are the same than the hard-coded ones
        """
        tokenizer = AutoTokenizer.from_pretrained(self.path_tokenizer)

        INPUT_SENTENCES = ["The quick brown fox</s>", "jumps over the lazy dog</s>"]
        TARGET_TOKENS = [[2175, 23714, 73173, 144252, 2], [77, 132619, 3478, 368, 109586, 35433, 2]]

        computed_tokens = tokenizer.batch_encode_plus(INPUT_SENTENCES)["input_ids"]
        self.assertListEqual(TARGET_TOKENS, computed_tokens)

        decoded_tokens = tokenizer.batch_decode(computed_tokens)
        self.assertListEqual(decoded_tokens, INPUT_SENTENCES)

    def test_encodings_from_xnli_dataset(self):
        """
        Tests the tokenizer downloaded from here:
            - https://huggingface.co/bigscience/tokenizer/
        On the bin data (local test data from Thomas Wolf)
        """
        tokenizer = AutoTokenizer.from_pretrained(self.path_tokenizer)
        ds = load_dataset("xnli", "all_languages", split="test", streaming=True)

        sample_data = next(iter(ds))["premise"]  # pick up one data
        input_text = list(sample_data.values())

        output_tokens = list(map(tokenizer.encode, input_text))
        predicted_text = list(map(lambda x: tokenizer.decode(x, clean_up_tokenization_spaces=False), output_tokens))
        self.assertListEqual(predicted_text, input_text)

    @unittest.skip(
        reason="skipping this test bc of env issues (see slack) - You have to install tokenizers 0.12.0 before - Needs to be tested after the release 0.12.1"
    )
    def test_encodings_on_xlmi(self):
        tokenizer = AutoTokenizer.from_pretrained(self.path_tokenizer)
        local_tokenizer = AutoTokenizer.from_pretrained(self.path_local_tokenizer)

        ds = load_dataset("xnli", "all_languages", split="test", streaming=True)

        sample_data = next(iter(ds))["premise"]  # pick up one data
        input_text = list(sample_data.values())

        tokens = tokenizer.batch_encode_plus(input_text)
        tokens_local = local_tokenizer.batch_encode_plus(input_text)

        self.assertListEqual(tokens, tokens_local)


if __name__ == "__main__":
    unittest.main()
