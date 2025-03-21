from transformers import PreTrainedTokenizerFast


class ArlowTokenizer(PreTrainedTokenizerFast):
    vocab_files_names = {"tokenizer_file": "tokenizer.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        tokenizer_file=None,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        mask_token="<|mask|>",
        additional_special_tokens=None,
        **kwargs
    ):
        if additional_special_tokens is None:
            additional_special_tokens = ["<|im_start|>", "<|im_end|>"]

        super().__init__(
            tokenizer_file=tokenizer_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self.model_type = "ArlowGPT"


# --- Test code below ---

if __name__ == "__main__":
    tokenizer = ArlowTokenizer(tokenizer_file="tokenizer.json")
    print("Instantiated ArlowTokenizer successfully!")

    # Basic encode-decode test
    text = "hello world"
    enc = tokenizer.encode(text)
    dec = tokenizer.decode(enc)
    print("Text:", text)
    print("Encoded IDs:", enc)
    print("Decoded string:", dec)
