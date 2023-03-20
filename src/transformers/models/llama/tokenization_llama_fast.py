from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils.versions import require_version


require_version("tokenizers>=0.13.3")


class LlamaTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding.
    """

    def __init__(
        self,
        *args,
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        super().__init__(*args, clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)
