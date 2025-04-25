from transformers import PreTrainedTokenizerFast
from transformers.models.llama.tokenization_spm import SPMTokenizer
from transformers.convert_slow_tokenizer import convert_slow_tokenizer


def load_spm_tokenizer(model_path: str) -> SPMTokenizer:
    """
    Load a slow SentencePiece tokenizer from the specified model path.
    """
    return SPMTokenizer.from_pretrained(
        model_path,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )


def load_fast_spm_tokenizer(model_path: str) -> PreTrainedTokenizerFast:
    """
    Load a fast tokenizer using the slow SPMTokenizer and convert it.
    """
    slow_tokenizer = SPMTokenizer.from_pretrained(
        model_path,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        do_lower_case=False,
        add_bos_token=True,
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=convert_slow_tokenizer(slow_tokenizer)
    )


def compare_tokenizers(sp_tokenizer, fast_tokenizer, text: str):
    """
    Assert that tokenization and decoding results are identical between slow and fast tokenizers.
    """
    sp_tokens = sp_tokenizer.tokenize(text)
    fast_tokens = fast_tokenizer.tokenize(text)
    assert sp_tokens == fast_tokens, (
        f"\nToken mismatch for input: {repr(text)}\n"
        f"SPM tokens : {sp_tokens}\n"
        f"Fast tokens: {fast_tokens}"
    )

    sp_ids = sp_tokenizer.encode(text)
    fast_ids = fast_tokenizer.encode(text)
    assert sp_ids == fast_ids, (
        f"\nID mismatch for input: {repr(text)}\n"
        f"SPM IDs : {sp_ids}\n"
        f"Fast IDs: {fast_ids}"
    )

    sp_decoded = sp_tokenizer.decode(sp_ids)
    fast_decoded = fast_tokenizer.decode(fast_ids)
    assert sp_decoded == fast_decoded, (
        f"\nDecoded output mismatch for input: {repr(text)}\n"
        f"SPM decoded : {sp_decoded}\n"
        f"Fast decoded: {fast_decoded}"
    )


TEST_STRINGS = [
    "Hey<eos>. \t\t \n\nyou  é  @#😈  🤗!       , 1234 15 5,61",
    "The following string should be properly encoded: Hello.",
    "But ird and ปี   ird   ด",
    "This is a test.",
    "Hello world!   Multiple spaces here.",
    "Hi  Hello with double space.",
    "   Leading spaces.",
    "Trailing spaces",
    "<s>Special token at start",
    "Text with <s> special token in the middle",
    "Text ending with special token <s>",
    "<s> Special token with spaces",
    "<s>I immediately after special token",
    "Hello, <s>, with commas",
    "生活的真谛是 Chinese characters",
    "áéíóúñ Accented characters",
    "ا العربية Arabic text",
    "Numbers 12345 and symbols !@#$%^&*()",
    "Line with\nmultiple\nbreaks",
]


def main():
    model_path = "../../../local-gemma-7b/tokenizer.model"  # Adjust to your local path
    sp_tokenizer = load_spm_tokenizer(model_path)
    fast_tokenizer = load_fast_spm_tokenizer(model_path)

    for text in TEST_STRINGS:
        compare_tokenizers(sp_tokenizer, fast_tokenizer, text)

    print("All tokenizer outputs match ✔️")


if __name__ == "__main__":
    main()
