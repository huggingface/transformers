from transformers import UdopTokenizer, UdopTokenizerFast
from transformers.testing_utils import get_tests_dir


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")

slow_tokenizer = UdopTokenizer(SAMPLE_VOCAB)

slow_tokenizer.save_pretrained(".")

slow_tokenizer = UdopTokenizer.from_pretrained(".")

fast_tokenizer = UdopTokenizerFast.from_pretrained(".")
