from transformers import UdopTokenizerFast


# from transformers.testing_utils import get_tests_dir


# SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")

tokenizer = UdopTokenizerFast.from_pretrained("t5-base")
tokenizer.save_pretrained(".", legacy_format=False)


new_tokenizer = UdopTokenizerFast.from_pretrained(".")
