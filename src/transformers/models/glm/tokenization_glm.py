# coding=utf-8
# Copyright 2024 GLM & ZhipuAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tokenization classes for GLM."""

import json
import os
from functools import lru_cache
from typing import Optional, Tuple, Type

import regex as re

from ...tokenization_utils import AddedToken, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETOKENIZE_REGEX = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"


@lru_cache()
# Copied from transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


# Copied from transformers.models.gpt2.tokenization_gpt2.get_pairs
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class GLMTokenizer(PreTrainedTokenizer):
    """
    Construct a GLM tokenizer. Based on byte-level Byte-Pair-Encoding.

    Same with GPT2Tokenizer, this tokenizer has been trained to treat spaces like parts of the tokens so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import GLMTokenizer

    >>> tokenizer = GLMTokenizer.from_pretrained("THUDM/GLM-tokenizer")
    >>> tokenizer("Hello world")["input_ids"]
    [9703, 1879]

    >>> tokenizer(" Hello world")["input_ids"]
    [21873, 1879]
    ```
    This is expected.

    You should not use GPT2Tokenizer instead, because of the different pretokenization rules.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*):
            The beginning of sequence token. Not applicable for this tokenizer.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The token used for padding, for example when batching sequences of different lengths.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not the model should cleanup the spaces that were added when splitting the input text during the
            tokenization process. Not applicable to this tokenizer, since tokenization does not add spaces.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            Whether or not the default system prompt for Cohere tokenizer should be used.
        split_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not the special tokens should be split during the tokenization process. The default behavior is
            to not split special tokens. This means that if `<|endoftext|>` is the `eos_token`, then `tokenizer.tokenize("<|endoftext|>") =
            ['<|endoftext|>`]. Otherwise, if `split_special_tokens=True`, then `tokenizer.tokenize("<|endoftext|>")` will be give `['<',
            '|', 'endo', 'ft', 'ext', '|', '>']`. This argument is only supported for `slow` tokenizers for the moment.
        spaces_between_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to add spaces between special tokens.
        add_prefix_space (`bool`, *optional*, defaults to `True`):
            Whether or not to add a space to the beginning of the text. This allows to treat the leading word just as any other word.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        clean_up_tokenization_spaces=False,
        use_default_system_prompt=False,
        split_special_tokens=False,
        spaces_between_special_tokens=False,
        add_prefix_space=True,
        **kwargs,
    ):
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.add_prefix_space = add_prefix_space
        self.use_default_system_prompt = use_default_system_prompt
        self.spaces_between_special_tokens = spaces_between_special_tokens

        bpe_merges = []
        with open(merges_file, encoding="utf-8") as merges_handle:
            for i, line in enumerate(merges_handle):
                line = line.strip()
                if (i == 0 and line.startswith("#version:")) or not line:
                    continue
                bpe_merges.append(tuple(line.split()))
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        # NOTE: the cache can grow without bound and will get really large for long running processes
        # (esp. for texts of language that do not use space between word, e.g. Chinese); technically
        # not a memory leak but appears as one.
        # GPT2Tokenizer has the same problem, so let's be consistent.

        self.cache = {}
        self.pat = re.compile(PRETOKENIZE_REGEX)
        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            use_default_system_prompt=use_default_system_prompt,
            split_special_tokens=split_special_tokens,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.encoder)

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.get_vocab
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.bpe
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._tokenize
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._convert_token_to_id
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.convert_tokens_to_string
    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Type[tuple] | tuple[str, str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return Tuple[None]

        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"],
        )
        merge_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["merges_file"],
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning(
                        f"Saving vocabulary to {merge_file}: BPE merge indices are not consecutive."
                        " Please check that the tokenizer is not corrupted!"
                    )
                    index = token_index
                writer.write(" ".join(bpe_tokens) + "\n")
                index += 1

        return vocab_file, merge_file

    @property
    def default_chat_template(self):
        """
        GLM uses [gMASK] and <sop> to indicate user messages. The system message is included as part of the first user
        message. The assistant messages do not have special tokens, as they can be identified by their order.

        We add a system prompt to make GLM-4 can be used in Function Calling and GLM All Tools capability.

        Here is an example of output：

        [gMASK]<sop><|system|>\nSystemPrompt<|user|>\nPrompt<|assistant|>n\\Answer<|user|>\nPrompt<|assistant|>\nAnswer<|user|>

        """
        template = "[gMASK]<sop>{% for item in messages %}{% if item['tools'] is defined %}<|system|>\n你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。\n\n# 可用工具{% set tools = item['tools'] %}{% for tool in tools %}{% if tool['type'] == 'function' %}\n\n## {{ tool['function']['name'] }}\n\n{{ tool['function'] | tojson(indent=4) }}\n在调用上述函数时，请使用 Json 格式表示调用的参数。{% elif tool['type'] == 'python' %}\n\n## python\n\n当你向 `python` 发送包含 Python 代码的消息时，该代码将会在一个有状态的 Jupyter notebook 环境中执行。\n`python` 返回代码执行的输出，或在执行 60 秒后返回超时。\n`/mnt/data` 将会持久化存储你的文件。在此会话中，`python` 无法访问互联网。不要使用 `python` 进行任何网络请求或者在线 API 调用，这些在线内容的访问将不会成功。{% elif tool['type'] == 'simple_browser' %}\n\n## simple_browser\n\n你可以使用 `simple_browser` 工具。该工具支持以下函数：\n`search(query: str, recency_days: int)`：使用搜索引擎进行查询并显示结果，可以使用 `recency_days` 参数控制搜索内容的时效性。\n`mclick(ids: list[int])`：获取一系列指定 id 的页面内容。每次调用时，须选择3-10个页面。选择多个角度的页面，同时尽可能选择可信任的信息来源。考虑到部分页面是无法加载的，你也可以多打开一些可能有用的页面而不用担心内容过多。\n`open_url(url: str)`：打开指定的 URL。\n\n使用 `【{引用 id}†{引用文本}】` 来引用内容。\n\n操作步骤：1. 使用 `search` 来获得信息列表; 2. 使用 `mclick` 来获取指定 ID 页面的内容; 3. 根据获得的内容进行回复。在回复中应当引用信息来源。\n 如果用户提供了 URL，也可以用 `open_url` 直接打开页面。\n如果初次搜索结果没有找到合适的信息，也可以再次使用 `search` 进行搜索。{% elif tool['type'] == 'cogview' %}\n\n## cogview\n\n如果用户的请求中包含了对图像的描述，你可以使用 `cogview` 来生成图像并展示给用户。你需要向 `cogview` 发送图像描述，规则：\n- 发送给 `cogview` 的消息必须使用英语。用户的中文描述必须完全翻译为英语。\n- 应当尽可能详细地描述图像生成的需求，需求描述约 100 英文单词。\n- 保持用户原始描述的意图。不要虚构内容或者没见过的人物。\n- 如无特殊说明，所在地为中国，持有中国立场并遵循中国社会主义价值观。{% endif %}{% endfor %}{% endif %}{% if item['content'] %}<|{{ item['role'] }}|>{{ item['metadata'] }}\n{{ item['content'] }}{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"
        template = template.replace("USE_DEFAULT_PROMPT", "true" if self.use_default_system_prompt else "false")
        return template
