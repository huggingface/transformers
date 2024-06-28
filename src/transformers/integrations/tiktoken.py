from transformers.convert_slow_tokenizer import TikTokenConverter


def convert_tiktoken_tokenizer(vocab_file):
    converter = TikTokenConverter(vocab_file)
    fast_tokenizer = converter.converted()
    return fast_tokenizer
