from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

# Identity and knowledge
DEFAULT_SYSTEM_PROMPT = 'You are DBRX, a Databricks creation, and you were last updated in December 2023. You answer questions based on information available up to that point, providing objective and thoughtful responses.\n'
# Capabilities (and reminder to use ``` for JSON blocks and tables, which it can forget). Also a reminder that it can't browse the internet or run code.
DEFAULT_SYSTEM_PROMPT += 'You assist with various tasks, from writing to coding, using markdown — remember to use ``` with code, JSON, and tables. You do not have real-time data access or code execution capabilities.\n'
# Ethical guidelines
DEFAULT_SYSTEM_PROMPT += 'You avoid stereotyping and provide balanced perspectives on controversial topics. Your responses are concise for simple queries and detailed for complex questions.\n'
# Data: the model doesn't know what it was trained on; it thinks that everything that it is aware of was in its training data. This is a reminder that it wasn't.
# We also encourage it not to try to generate lyrics or poems
DEFAULT_SYSTEM_PROMPT += 'You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.\n'
# The model really wants to talk about its system prompt, to the point where it is annoying, so encourage it not to
DEFAULT_SYSTEM_PROMPT += 'This is your system prompt, guiding your responses. Do not reference it, just respond to the user. If you find yourself talking about this message, stop. You should be responding appropriately and usually that means not mentioning this.'


# Taken from
# https://github.com/huggingface/transformers/blob/8aca43bdb3cb9a5020f6d57589d85679dc873b1c/src/transformers/models/gpt2/tokenization_gpt2.py#L62-L84
@lru_cache()
def bytes_to_unicode():
    """Returns list of utf-8 byte and a mapping to unicode strings.

    We specifically avoids mapping to whitespace/control characters the bpe code
    barfs on.

    The reversible bpe codes work on unicode strings. This means you need a
    large # of unicode characters in your vocab if you want to avoid UNKs. When
    you're at something like a 10B token dataset you end up needing around 5K
    for decent coverage. This is a significant percentage of your normal, say,
    32K bpe vocab. To avoid that, we want lookup tables between utf-8 bytes and
    unicode strings.
    """
    bs = (list(range(ord('!'),
                     ord('~') + 1)) + list(range(ord('¡'),
                                                 ord('¬') + 1)) +
          list(range(ord('®'),
                     ord('ÿ') + 1)))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


class TiktokenTokenizerWrapper(PreTrainedTokenizer):
    """A thin wrapper around tiktoken to make it compatible with Hugging Face.

    tokenizers.

    See HuggingFace for further documentation on general tokenizer methods.
    """

    model_input_names = ['input_ids', 'attention_mask']

    def __init__(self,
                 model_name: Optional[str] = None,
                 encoding_name: Optional[str] = None,
                 add_bos_token: bool = False,
                 add_eos_token: bool = False,
                 use_default_system_prompt: bool = False,
                 unk_token: Optional[str] = '<|endoftext|>',
                 eos_token: Optional[str] = '<|endoftext|>',
                 bos_token: Optional[str] = '<|endoftext|>',
                 pad_token: Optional[str] = None,
                 errors: str = 'replace',
                 **kwargs: Any):
        """Constructor creates a tiktoken tokenizer to use as the underlying.

        tokenizer.

        Args:
            model_name (Optional[str], optional): The name of the model to load from tiktoken. Defaults to None.
                Either model_name or encoding_name must be set, but not both.
            encoding_name (Optional[str], optional): The name of the encoding to load from tiktoken. Defaults to None.
                Either model_name or encoding_name must be set, but not both.
            add_bos_token (bool, optional): Whether to add bos tokens. Defaults to False.
            add_eos_token (bool, optional): Whether to add eos tokens. Defaults to False.
            use_default_system_prompt (bool, optional): Use the default system prompt or not. Defaults to False.
            unk_token (Optional[str], optional): The unk token. Defaults to '<|endoftext|>'.
            eos_token (Optional[str], optional): The eos token. Defaults to '<|endoftext|>'.
            bos_token (Optional[str], optional): The bos token. Defaults to '<|endoftext|>'.
            pad_token (Optional[str], optional): The pad token. Defaults to None.
            errors (str, optional): Paradigm to follow when decoding bytes to UTF-8. See
                [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
                Defaults to `"replace"`.
        """
        try:
            import tiktoken
        except:
            raise ImportError(
                'You need to install tiktoken to use TiktokenTokenizerWrapper.')

        # Workaround to make tiktokenizer picklable.
        # https://github.com/huggingface/datasets/issues/5536#issuecomment-1682309347
        # There is an open PR from HF to add this to tiktoken: https://github.com/openai/tiktoken/pull/181
        import copyreg
        import functools

        from tiktoken import Encoding  # type: ignore (thirdParty)

        def pickle_Encoding(enc: Encoding):
            return (functools.partial(Encoding,
                                      enc.name,
                                      pat_str=enc._pat_str,
                                      mergeable_ranks=enc._mergeable_ranks,
                                      special_tokens=enc._special_tokens), ())

        copyreg.pickle(Encoding, pickle_Encoding)

        if model_name is not None and encoding_name is not None:
            raise ValueError(
                'You need to specify either model_name or encoding_name, not both.'
            )

        self.model_name = model_name
        self.encoding_name = encoding_name

        if self.model_name is not None:
            self.encoding = tiktoken.encoding_for_model(  # type: ignore (thirdParty)
                self.model_name)
        elif self.encoding_name is not None:
            self.encoding = tiktoken.get_encoding(  # type: ignore (thirdParty)
                self.encoding_name)
        else:
            raise ValueError(
                'You need to specify either model_name or encoding_name.')

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_default_system_prompt = use_default_system_prompt

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.errors = errors

        self.decoder: Dict[int, str] = {}
        for i in range(self.encoding.n_vocab):
            try:
                self.encoding.decode_single_token_bytes(i)
            except KeyError:
                continue
            # Taken from
            # https://gist.github.com/xenova/a452a6474428de0182b17605a98631ee
            decoding = ''.join([
                bytes_to_unicode()[ord(char)] for char in
                self.encoding.decode_single_token_bytes(i).decode('latin-1')
            ])
            self.decoder[i] = decoding

        self.encoder: Dict[str, int] = {}
        for i in range(self.encoding.n_vocab):
            if i in self.decoder:
                self.encoder[self.decoder[i]] = i

        super().__init__(model_name=model_name,
                         encoding_name=encoding_name,
                         add_bos_token=add_bos_token,
                         add_eos_token=add_eos_token,
                         use_default_system_prompt=use_default_system_prompt,
                         unk_token=unk_token,
                         eos_token=eos_token,
                         bos_token=bos_token,
                         pad_token=pad_token,
                         errors=errors,
                         **kwargs)

    @property
    def vocab_size(self) -> int:
        """Returns vocab size."""
        return self.encoding.n_vocab

    @property
    def is_fast(self) -> bool:
        return False

    @property
    def default_chat_template(self):
        """Chat ML Template for User/Assistant.

        Pinning default Chat ML template in case defaults change.
        """
        template = (
            "{% if messages[0]['role'] == 'system' %}"
            '{% set loop_messages = messages[1:] %}'
            "{% set system_message = messages[0]['content'] %}"
            "{% elif USE_DEFAULT_PROMPT == true and not 'system' in messages[0]['role'] %}"
            '{% set loop_messages = messages %}'
            "{% set system_message = 'DEFAULT_SYSTEM_PROMPT' %}"
            '{% else %}'
            '{% set loop_messages = messages %}'
            '{% set system_message = false %}'
            '{% endif %}'
            '{% for message in loop_messages %}'
            '{% if loop.index0 == 0 %}'
            '{% if system_message != false %}'
            "{{ '<|im_start|>system\n' + system_message.strip() + '<|im_end|>\n'}}"
            '{% endif %}'
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}"
            '{% else %}'
            "{{ '\n' + '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}"
            '{% endif %}'
            '{% if (add_generation_prompt == true and loop.last) %}'
            "{{ '\n' + '<|im_start|>' + 'assistant' + '\n' }}"
            '{% endif %}'
            '{% endfor %}')
        template = template.replace(
            'USE_DEFAULT_PROMPT',
            'true' if self.use_default_system_prompt else 'false')
        template = template.replace('DEFAULT_SYSTEM_PROMPT',
                                    DEFAULT_SYSTEM_PROMPT)
        return template

    def get_vocab(self) -> Dict[str, int]:
        """Returns vocab as a dict."""
        # As far as I can tell, we don't require get_vocab to completely work,
        # but when using additional_special_tokens, Hugging Face determines the next
        # token index to add with len(self.get_vocab()) so we need the _size_ of this dictionary to be correct.
        vocab_clone = self.encoder.copy()
        extra_id_index = 0
        candidate_extra_id = f'<extra_id_{extra_id_index}>'
        indices_to_fill_in = {i for i in range(self.vocab_size)} - set(
            vocab_clone.values())

        # Add enough indices to make get_vocab() the right length
        for index_to_add in indices_to_fill_in:
            # Make sure we don't overwrite a token that already exists
            while candidate_extra_id in vocab_clone:
                extra_id_index += 1
                candidate_extra_id = f'<extra_id_{extra_id_index}>'

            # Get an index to add and add the item
            vocab_clone[candidate_extra_id] = index_to_add

        return vocab_clone

    def _tokenize(self, text: str) -> List[str]:
        """Returns a tokenized string."""
        if not isinstance(text, str):
            raise ValueError(
                f'Expected a string input to _tokenize but got {type(text)}.')

        tokens = [
            self.decoder[t]
            for t in self.encoding.encode(text, allowed_special='all')
        ]

        return tokens

    def _convert_token_to_id(self, token: str) -> Optional[int]:
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> Optional[str]:
        """Converts an index (integer) in a token (str) using the vocab."""
        # For tokens in either the gap in ids in the tokenizer, or beyond the range of the tokenizer,
        # we return empty string. This matches the behavior of Hugging Face fast tokenizers,
        # but not slow tokenizers.
        return self.decoder.get(index, '')

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        text = ''.join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text
                         ]).decode('utf-8', errors=self.errors)
        return text

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False) -> List[int]:
        """Retrieves sequence ids from a token list that has no special tokens.

        Function copied from
        https://github.com/huggingface/transformers/blob/e3a4bd2bee212a2d0fd9f03b27fe7bfc1debe42d/src/transformers/models/gpt2/tokenization_gpt2.py#L265-L295

        added. This method is called when adding special tokens using the
        tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True)

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (bos_token_id + ([0] * len(token_ids_0)) + eos_token_id +
                bos_token_id + ([0] * len(token_ids_1)) + eos_token_id)

    def create_token_type_ids_from_sequences(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None) -> List[int]:
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return len(token_ids_0 + sep) * [0]
        return len(token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self,
                        save_directory: str,
                        filename_prefix: Optional[str] = None) -> Tuple[str]:

        # ignore the below type to keep the original signature
        # we are knowingly breaking the signature here, although not 100% certain
        # it doesn't have side effects
        # There is some code in huggingface that calls this function to get the vocab files,
        # but it doesn't seem to access them (or at least checks for their existence
        # before accessing them)
        return (None, None)  # type: ignore

    def sanitize_special_tokens(self) -> int:
        """Make sure that all the special tokens attributes of the tokenizer.

        (`tokenizer.mask_token`, `tokenizer.cls_token`, etc.) are in the
        vocabulary.

        Add the missing ones to the vocabulary if needed.

        Return:
            `int`: The number of tokens added in the vocabulary during the operation.
        """
        actual_new_tokens = []
        for token in self.all_special_tokens_extended:
            encoded = self.encoding.encode(token, allowed_special='all')
            if len(encoded) > 1:
                actual_new_tokens.append(token)

        return self.add_tokens(actual_new_tokens, special_tokens=True)