from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from transformers import LlamaTokenizerFast
from transformers.convert_slow_tokenizer import bytes_to_unicode


class MistralConverter:
    """
    A general tiktoken converter.
    """

    def __init__(
        self,
        vocab=None,
        pattern=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        add_prefix_space=False,
        additional_special_tokens=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args)
        self.vocab = vocab
        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.additional_special_tokens = additional_special_tokens

    def extract_vocab_merges_from_model(self, vocab: str):
        bpe_ranks = vocab
        byte_encoder = bytes_to_unicode()

        def token_bytes_to_string(b):
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        merges = []
        vocab = {}
        for idx, (token, rank) in enumerate(bpe_ranks.items()):
            if token not in self.additional_special_tokens:
                vocab[token_bytes_to_string(token)] = idx
                if len(token) == 1:
                    continue
                local = []
                for index in range(1, len(token)):
                    piece_l, piece_r = token[:index], token[index:]
                    if piece_l in bpe_ranks and piece_r in bpe_ranks and (piece_l + piece_r) in bpe_ranks:
                        local.append((piece_l, piece_r, rank))
                local = sorted(local, key=lambda x: (bpe_ranks[x[0]], bpe_ranks[x[1]]), reverse=False)
                merges.extend(local)
            else:
                vocab[token] = idx
        merges = sorted(merges, key=lambda val: val[2], reverse=False)
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in merges]
        return vocab, merges

    def tokenizer(self):
        vocab_scores, merges = self.extract_vocab_merges_from_model(self.vocab)
        tokenizer = Tokenizer(BPE(vocab_scores, merges, fuse_unk=False))
        if hasattr(tokenizer.model, "ignore_merges"):
            tokenizer.model.ignore_merges = True
        return tokenizer

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(self.pattern), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=self.add_prefix_space, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens(self.additional_special_tokens)

        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        return tokenizer


def convert_tekken_tokenizer(tokenizer_file: str):
    """Convert a "tekken" tokenizer to a fast Tokenizer."""
    # Tekken format -- need to use the Converter

    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    # Load directly using their lib
    mistral_tokenizer = MistralTokenizer.from_file(tokenizer_file)

    # Extract vocab and special tokens
    vocab = mistral_tokenizer.instruct_tokenizer.tokenizer._tekken_token2id_nospecial
    all_special = [
        token.value if hasattr(token, "value") else token
        for token in mistral_tokenizer.instruct_tokenizer.tokenizer._all_special_tokens
    ]
    specials_tokens = {token: all_special.index(token) for token in all_special}
    specials_tokens.update(vocab)
    vocab = specials_tokens

    # Convert
    tokenizer = LlamaTokenizerFast(
        tokenizer_object=MistralConverter(vocab=vocab, additional_special_tokens=all_special).converted(),
    )

    # Post-process
    tokenizer.add_special_tokens({"additional_special_tokens": all_special})

    return tokenizer
