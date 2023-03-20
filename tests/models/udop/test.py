from transformers import UdopTokenizer, UdopTokenizerFast
from transformers.testing_utils import get_tests_dir


SAMPLE_VOCAB = get_tests_dir("fixtures/test_sentencepiece.model")

slow_tokenizer = UdopTokenizer(SAMPLE_VOCAB)

print("------SAVING SLOW TOKENIZER----------")
slow_tokenizer.save_pretrained(".")

print("------INSTANTIATING FAST TOKENIZER----------")
# slow_tokenizer = UdopTokenizer.from_pretrained(".")

fast_tokenizer = UdopTokenizerFast.from_pretrained(".")
