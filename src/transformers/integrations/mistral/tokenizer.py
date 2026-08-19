# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
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

"""Conversion between Mistral tekken tokenizers and HuggingFace tokenizer formats."""

import base64
import json
import os
from functools import lru_cache
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING

from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from ...convert_slow_tokenizer import bytes_to_unicode
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import cached_file, logging, requires_backends
from ...utils.hub import CHAT_TEMPLATE_FILE, LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE
from ...utils.import_utils import is_mistral_common_available
from .constants import TEKKEN_VOCAB_FILE, is_tekken_vocab_filename


if is_mistral_common_available():
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer


if TYPE_CHECKING:
    from ...models.pixtral.processing_pixtral import PixtralProcessor


logger = logging.get_logger(__name__)


def _resolve_chat_template(tekken_file: str | os.PathLike, chat_template: str | None) -> str | None:
    """Resolve the chat template to attach during tekken to HF conversion.

    Applies a fixed precedence order:

    1. `chat_template` argument (if not `None`) — returned unchanged.
    2. Sibling `chat_template.jinja` file in the same directory as *tekken_file*.
    3. Sibling `chat_template.json` file — value of its `"chat_template"` key.
    4. Automatic generation via `mistral_common.integrations.chat_templates` (lazy import).
    5. `None` if none of the above succeed.

    Args:
        tekken_file (`str` or `os.PathLike`): Path to the `tekken.json` file.
        chat_template (`str` or `None`): Explicit chat template string. Only `None`
            triggers the lookup cascade; an empty string is returned as-is.

    Returns:
        Resolved chat template string, or `None`.

    Raises:
        KeyError: If `chat_template.json` exists but does not contain a `"chat_template"` key.
    """
    # Precedence 1: explicit arg wins (including empty string).
    if chat_template is not None:
        return chat_template

    parent = Path(tekken_file).parent

    # Precedence 2: sibling chat_template.jinja.
    jinja_path = parent / CHAT_TEMPLATE_FILE
    if jinja_path.is_file():
        return jinja_path.read_text(encoding="utf-8")

    # Precedence 3: sibling chat_template.json — KeyError propagates on missing key.
    json_path = parent / LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE
    if json_path.is_file():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        return data["chat_template"]

    # Precedence 4: generate via mistral-common (lazy import inside branch).
    if is_mistral_common_available():
        try:
            from mistral_common.integrations.chat_templates.chat_templates import (
                convert_tokenizer_to_chat_template,
            )

            return convert_tokenizer_to_chat_template(tekken_file)
        except Exception as exc:
            logger.warning_once(
                f"Failed to generate chat template from '{tekken_file}': {exc}. Falling back to no chat template."
            )
            return None

    # Precedence 5: no template available.
    return None


def _probe_file(
    pretrained_model_name_or_path: str | os.PathLike,
    filename: str,
    **cache_kwargs,
) -> str | None:
    """Return the resolved path for *filename* inside *pretrained_model_name_or_path*, or `None`.

    Exceptions for missing entries and connection errors are suppressed so that
    callers can treat a `None` return as "file not found / not reachable".

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`): Model id or local directory.
        filename (`str`): File name to look up inside the checkpoint.
        **cache_kwargs: Forwarded to [`~utils.cached_file`].

    Returns:
        Resolved local path string, or `None` if the file could not be found.
    """
    return cached_file(
        pretrained_model_name_or_path,
        filename,
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
        **cache_kwargs,
    )


def resolve_mistral_format(
    pretrained_model_name_or_path: str | os.PathLike,
    mistral_format: bool | None = None,
    **cache_kwargs,
) -> tuple[bool, str | None]:
    """Resolve whether to use `MistralCommonBackend` for tokenization.

    Probes for `tekken.json` in the checkpoint directory and checks whether
    `mistral-common` is installed to determine the appropriate tokenizer backend.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model hosted on huggingface.co.
            - a path to a *directory* containing model files.
        mistral_format (`bool`, *optional*):
            Tri-state control for tokenizer backend selection:

            - `True` — force `MistralCommonBackend` (raises `ImportError` if `mistral-common`
              is not installed, raises `OSError` if `tekken.json` is not found).
            - `False` — force standard HuggingFace tokenizer.
            - `None` — auto-detect: selects native (`MistralCommonBackend`) when
              `mistral-common` is installed and a `tekken.json` is found, regardless of
              whether HF-format files are also present. Otherwise falls back to the
              standard HuggingFace tokenizer.
        **cache_kwargs:
            Forwarded to [`~utils.cached_file`] (e.g. `cache_dir`, `force_download`,
            `local_files_only`, `revision`, `token`).

    Returns:
        `tuple[bool, str | None]`: A tuple of `(use_mistral_format, tekken_file_path)` where
        `use_mistral_format` indicates whether `MistralCommonBackend` should be used, and
        `tekken_file_path` is the resolved path to `tekken.json` (or `None`).

    Raises:
        ImportError: If `mistral_format=True` and `mistral-common` is not installed.
        OSError: If `mistral_format=True` and `tekken.json` cannot be found.
    """
    if mistral_format is False:
        return (False, None)

    # These are forced below; drop any caller-provided copies (e.g. from AutoProcessor's
    # cached_file_kwargs) to avoid "multiple values for keyword argument" errors.
    cache_kwargs.pop("_raise_exceptions_for_missing_entries", None)
    cache_kwargs.pop("_raise_exceptions_for_connection_errors", None)

    if mistral_format is True:
        if not is_mistral_common_available():
            raise ImportError(
                "mistral_format=True requires `mistral-common`. Install it with: pip install mistral-common"
            )
        # Hub/directory discovery uses strict equality against TEKKEN_VOCAB_FILE (mirroring
        # mistral-common's own discovery rule).
        tekken_file = _probe_file(pretrained_model_name_or_path, TEKKEN_VOCAB_FILE, **cache_kwargs)
        if tekken_file is None:
            raise OSError(
                f"Cannot find '{TEKKEN_VOCAB_FILE}' at '{pretrained_model_name_or_path}'. "
                "Set `mistral_format=False` to use standard HuggingFace files instead."
            )
        return (True, tekken_file)

    # mistral_format is None: auto-detect
    if not is_mistral_common_available():
        return (False, None)

    tekken_file = _probe_file(pretrained_model_name_or_path, TEKKEN_VOCAB_FILE, **cache_kwargs)
    return (tekken_file is not None, tekken_file)


_MAP_SPECIALS = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
}


def _derive_tekken_specials(raw: dict) -> tuple[list[str], int]:
    """Derive the expected special-token strings and special-token count from a parsed tekken.json.

    Single source of truth for the special-token layout, shared by
    `MistralConverter._parse_tekken_file` (to build the vocab) and
    `_check_tekken_vocab_unchanged` (to compare against a live tokenizer), so the two cannot
    silently drift apart.

    Args:
        raw (`dict`): A parsed tekken.json, i.e. the result of `json.load` on the file.

    Returns:
        `tuple[list[str], int]`: The special-token strings, ordered by rank (the real entries
        from *raw* followed by any `<SPECIAL_i>` filler needed to reach `num_special_tokens`),
        and the resolved `num_special_tokens`.

    Raises:
        ImportError: If *raw* has no top-level `"special_tokens"` key (old tekken format) and
            `mistral-common` is not installed.
    """
    config = raw["config"]
    filler_template = "<SPECIAL_{id}>"

    special_tokens_dicts = raw.get("special_tokens")
    if special_tokens_dicts is None:
        # Old tekken format has no special_tokens key; use mistral-common's defaults.
        requires_backends(MistralConverter, ["mistral-common"])
        filler_template = getattr(Tekkenizer, "SPECIAL_TOKEN_TEMPLATE", filler_template)
        special_tokens_dicts = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)

    # Only the string order is returned below, not the real ranks: ids are re-derived by
    # enumerating that order, which requires ranks to be contiguous starting at 0.
    declared_ranks = sorted(entry["rank"] for entry in special_tokens_dicts)
    if declared_ranks != list(range(len(declared_ranks))):
        raise ValueError(f"tekken.json special_tokens ranks must be contiguous starting at 0, got {declared_ranks}")

    special_strings = [
        token_str
        for _, token_str in sorted(
            (entry["rank"], str(getattr(entry["token_str"], "value", entry["token_str"])))
            for entry in special_tokens_dicts
        )
    ]

    num_special_tokens = config.get("default_num_special_tokens")
    if num_special_tokens is None:
        num_special_tokens = len(special_strings)

    filler_strings = [filler_template.format(id=i) for i in range(len(special_strings), num_special_tokens)]
    return special_strings + filler_strings, num_special_tokens


class MistralConverter:
    """Converter from Mistral tekken BPE vocab to a HuggingFace `tokenizers.Tokenizer`."""

    def __init__(self, vocab_file: str, add_prefix_space: bool = False, **kwargs):
        """Parse a raw `tekken.json` file into a ready-to-use converter.

        Matches `mistral_common`'s `Tekkenizer.from_file` vocab and special-token layout.

        Args:
            vocab_file (`str`): Path to a `tekken.json` file.
            add_prefix_space (`bool`): Whether to add a leading space during tokenization.
        """
        self._parse_tekken_file(vocab_file, add_prefix_space)

    def _parse_tekken_file(self, vocab_file: str, add_prefix_space: bool) -> None:
        """Parse a tekken.json file and set all instance attributes.

        Args:
            vocab_file (`str`): Path to a `tekken.json` file.
            add_prefix_space (`bool`): Whether to add a leading space during tokenization.
        """
        with open(vocab_file, encoding="utf-8") as f:
            untyped = json.load(f)

        config = untyped["config"]
        pattern = config["pattern"]
        vocab_size = config.get("default_vocab_size")

        special_token_strings, num_special_tokens = _derive_tekken_specials(untyped)

        additional_special_tokens = [AddedToken(token_str, special=True) for token_str in special_token_strings]

        # Drop padded vocab: keep only the real tokens (matches mistral-common).
        bpe_ranks_raw = untyped["vocab"]
        if vocab_size is not None:
            inner_vocab_size = vocab_size - num_special_tokens
            bpe_ranks_raw = bpe_ranks_raw[:inner_vocab_size]

        bpe_ranks = [base64.b64decode(k["token_bytes"]) for k in bpe_ranks_raw]
        bpe_ranks_dict = {token: rank for rank, token in enumerate(bpe_ranks)}

        vocab, merges = self._extract_merges(bpe_ranks_dict)

        vocab = {k: v + num_special_tokens for k, v in vocab.items()}
        for rank, token_str in enumerate(special_token_strings):
            vocab[token_str] = rank

        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.additional_special_tokens = additional_special_tokens
        self._precomputed_vocab = vocab
        self._precomputed_merges = merges

    @staticmethod
    def _extract_merges(bpe_ranks: dict[bytes, int]) -> tuple[dict[str, int], list[tuple[str, str]]]:
        """Extract a unicode vocab and ordered BPE merge list from byte-level BPE ranks."""
        byte_encoder = bytes_to_unicode()

        @lru_cache
        def token_bytes_to_string(b: bytes) -> str:
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        vocab: dict[str, int] = {}
        all_merges: list[tuple[bytes, bytes, int]] = []

        for token, rank in bpe_ranks.items():
            vocab[token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            local = []
            for index in range(1, len(token)):
                piece_l, piece_r = token[:index], token[index:]
                if piece_l in bpe_ranks and piece_r in bpe_ranks and (piece_l + piece_r) in bpe_ranks:
                    local.append((piece_l, piece_r, rank))
            local = sorted(local, key=lambda x: (bpe_ranks[x[0]], bpe_ranks[x[1]]))
            all_merges.extend(local)

        all_merges = sorted(all_merges, key=lambda val: val[2])
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in all_merges]
        return vocab, merges

    def tokenizer(self) -> Tokenizer:
        """Build a raw `tokenizers.Tokenizer` with BPE model (no pre/post-processing)."""
        tokenizer = Tokenizer(BPE(self._precomputed_vocab, self._precomputed_merges, fuse_unk=False))
        if hasattr(tokenizer.model, "ignore_merges"):
            tokenizer.model.ignore_merges = True
        return tokenizer

    def converted(self) -> Tokenizer:
        """Build a fully configured `tokenizers.Tokenizer` with pre-tokenizer and decoder."""
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


def _resolve_tekken_source(tokenizer: TokenizersBackend) -> str:
    """Locate the on-disk `tekken.json` a tokenizer was built from.

    Called once by `save_tekken_format`, which passes the resolved path on to
    `_check_tekken_vocab_unchanged` to know what to diff against.

    Args:
        tokenizer (`TokenizersBackend`): The tokenizer to resolve a source file for. Must
            have been loaded or converted from a native Mistral tekken vocabulary file,
            i.e. its `vocab_file` must still point at an existing file whose name identifies
            it as tekken (see `is_tekken_vocab_filename`).

    Returns:
        `str`: The validated path to the tokenizer's source tekken vocabulary file.

    Raises:
        OSError: If the tokenizer's `vocab_file` does not resolve to an existing file
            identified as tekken (see `is_tekken_vocab_filename`).
    """
    vocab_file = tokenizer.init_kwargs.get("vocab_file") or getattr(tokenizer, "vocab_file", None)
    if not (isinstance(vocab_file, str) and is_tekken_vocab_filename(vocab_file) and os.path.isfile(vocab_file)):
        raise OSError(
            "Cannot save in 'mistral' format: the original tekken.json is not available. "
            "Load a native checkpoint (that still contains tekken.json) or install "
            "mistral-common to use MistralCommonBackend."
        )
    return vocab_file


def _check_tekken_vocab_unchanged(tokenizer: TokenizersBackend, vocab_file: str) -> None:
    """Compare *tokenizer*'s vocab against the `tekken.json` it was converted from.

    `save_format="mistral"` writes a native `tekken.json` by copying the source file
    directly, so any in-session mutation (`add_tokens`, `add_special_tokens`) would be
    silently dropped, and `tekken.json` cannot represent arbitrary added tokens anyway.
    This guard is called only from `save_tekken_format`, right before that lossy copy, to
    turn a silent drop into a raised error.

    Only two things are compared: the total vocab size, and the *set* of added-token
    content strings (via `tokenizer.added_tokens_decoder`). This does not check token ids
    or other `AddedToken` attributes (e.g. `lstrip`, `rstrip`, `normalized`).

    Args:
        tokenizer (`TokenizersBackend`): The tokenizer to check. Never mutated.
        vocab_file (`str`): Path to the source `tekken.json` to compare against.

    Raises:
        ValueError: If the vocab size or the added-token set diverges from the source
            `tekken.json`.
        OSError: If `vocab_file` cannot be opened.
    """
    with open(vocab_file, encoding="utf-8") as f:
        raw = json.load(f)

    config = raw["config"]
    expected_special_tokens_list, num_special_tokens = _derive_tekken_specials(raw)
    expected_special_tokens = set(expected_special_tokens_list)

    expected_vocab_size = config.get("default_vocab_size")
    if expected_vocab_size is None:
        expected_vocab_size = len(raw["vocab"]) + num_special_tokens

    observed_vocab_size = len(tokenizer)
    observed_added_tokens = {token.content for token in tokenizer.added_tokens_decoder.values()}

    unexpected_tokens = sorted(observed_added_tokens - expected_special_tokens)
    missing_tokens = sorted(expected_special_tokens - observed_added_tokens)
    size_diverges = observed_vocab_size != expected_vocab_size

    if unexpected_tokens or missing_tokens:
        lines = [
            "Cannot save in 'mistral' format: the tokenizer vocab state has diverged from its source "
            "tekken.json, which cannot represent arbitrary added tokens."
        ]
        if unexpected_tokens:
            lines.append(f"Unexpected added tokens: {unexpected_tokens}")
        if missing_tokens:
            lines.append(f"Missing expected special tokens: {missing_tokens}")
        if size_diverges:
            lines.append(f"Vocab size: observed={observed_vocab_size}, expected={expected_vocab_size}")
        lines.append("Use save_format='hf' instead to preserve these changes.")
        raise ValueError("\n".join(lines))

    if size_diverges:
        raise ValueError(
            "Cannot save in 'mistral' format: the tokenizer reports a vocab size of "
            f"{observed_vocab_size}, but its source tekken.json declares {expected_vocab_size} "
            "(via default_vocab_size/default_num_special_tokens). Either the vocab was resized "
            "in this session, in which case use save_format='hf' to preserve the change, or the "
            "source tekken.json is itself inconsistent and should be regenerated with matching "
            "vocab and config sizes."
        )


def save_tekken_format(tokenizer: TokenizersBackend, save_directory: str | os.PathLike) -> tuple[str, ...]:
    """Write a tokenizer back out in native Mistral format by copying its source tekken file.

    Used by `TokenizersBackend.save_pretrained` for `save_format="mistral"`. The tokenizer's
    original tekken vocabulary file (resolved via `vocab_file`) is copied byte-for-byte into
    *save_directory* as `tekken.json`, regardless of the source's own basename (e.g. a source
    named `tekken_240911.json` is normalized to `tekken.json` on save, so the output directory
    stays discoverable by both `resolve_mistral_format` and mistral-common.

    Resaving into the same directory the tokenizer was loaded from is a no-op copy: the
    source and destination are the same file, so the copy is skipped instead of raising
    `shutil.SameFileError`. The divergence guard still runs first, so an in-place resave of
    a mutated tokenizer is still rejected.

    Args:
        tokenizer (`TokenizersBackend`): The tokenizer to save. Must have been loaded or
            converted from a native Mistral tekken vocabulary file, i.e. its `vocab_file`
            must still point at an existing file identified as tekken (see
            `is_tekken_vocab_filename`), and it must not have diverged from that source
            file (see `_check_tekken_vocab_unchanged`).
        save_directory (`str` or `os.PathLike`): The path to a directory where the tekken
            vocabulary file will be written.

    Returns:
        `tuple[str, ...]`: A 1-tuple containing the path of the saved tekken vocabulary file.

    Raises:
        OSError: If the tokenizer's `vocab_file` does not resolve to an existing file
            identified as tekken (see `is_tekken_vocab_filename`).
        ValueError: If the tokenizer has diverged from its source tekken vocabulary file
            (added tokens, changed vocab size). See `_check_tekken_vocab_unchanged`.
    """
    vocab_file = _resolve_tekken_source(tokenizer)
    _check_tekken_vocab_unchanged(tokenizer, vocab_file)

    os.makedirs(save_directory, exist_ok=True)

    # The source may carry a non-canonical name (e.g. tekken_240911.json, accepted by
    # is_tekken_vocab_filename), but the output is always written as TEKKEN_VOCAB_FILE.
    dest = os.path.join(save_directory, TEKKEN_VOCAB_FILE)
    if not (os.path.exists(dest) and os.path.samefile(vocab_file, dest)):
        copyfile(vocab_file, dest)
    return (dest,)


def convert_tekken_tokenizer(
    tokenizer_file: str,
    chat_template: str | None = None,
) -> TokenizersBackend:
    """Build a `TokenizersBackend` from a Mistral `tekken.json` file.

    The chat template is resolved via a fixed precedence order (see
    `_resolve_chat_template`): explicit `chat_template` argument → sibling
    `chat_template.jinja` → sibling `chat_template.json` → auto-generation via
    `mistral-common` → `None`.

    Args:
        tokenizer_file (`str`): Path to the `tekken.json` vocabulary file.
        chat_template (`str`, *optional*): Explicit Jinja2 chat template string.
            When not provided (`None`), the template is resolved automatically
            from sibling files or generated via `mistral-common` if available.

    Returns:
        Configured `TokenizersBackend` with BPE model, special token mappings, and
        an attached chat template (or `None` when none could be resolved).
    """
    resolved = _resolve_chat_template(tokenizer_file, chat_template)
    converter = MistralConverter(vocab_file=tokenizer_file, add_prefix_space=False)
    fast = TokenizersBackend(
        tokenizer_object=converter.converted(),
        vocab_file=tokenizer_file,
        chat_template=resolved,
        **_MAP_SPECIALS,
    )
    return fast


def convert_tekken_image_processor(
    tokenizer_file: str,
    params_file: str,
    chat_template: str | None = None,
) -> "PixtralProcessor":
    """Build a `PixtralProcessor` from a tekken tokenizer file and a native `params.json`.

    Args:
        tokenizer_file (`str`): Path to the `tekken.json` vocabulary file.
        params_file (`str`): Path to the native `params.json` config file.
        chat_template (`str`, *optional*): Explicit Jinja2 chat template string.
            When not provided (`None`), the template is resolved automatically
            from sibling files or generated via `mistral-common` if available
            (see `_resolve_chat_template` for the full precedence order).

    Returns:
        Configured `PixtralProcessor` with tokenizer and image processor.

    Raises:
        ValueError: If `params_file` does not contain a `vision_encoder` key.
    """
    with open(params_file, encoding="utf-8") as f:
        params = json.load(f)

    vision_config = params.get("vision_encoder")
    if vision_config is None:
        raise ValueError(
            f"'vision_encoder' key not found in {params_file}. "
            "This model does not appear to be a vision-language model and does not need a processor. "
            "Use `convert_tekken_tokenizer` for text-only models instead."
        )

    # Lazy imports: processing_pixtral imports from integrations.mistral at module level,
    # so importing it here avoids a circular dependency. Placed after validation to avoid
    # triggering heavy imports (torchvision) for text-only models that will fail anyway.
    from ...models.pixtral.image_processing_pixtral import PixtralImageProcessor
    from ...models.pixtral.processing_pixtral import PixtralProcessor

    patch_size = vision_config["patch_size"]
    max_image_size = vision_config.get("max_image_size", vision_config["image_size"])
    spatial_merge_size = vision_config.get("spatial_merge_size", 1)

    if is_mistral_common_available():
        from ...tokenization_mistral_common import MistralCommonBackend

        tokenizer = MistralCommonBackend(tokenizer_path=tokenizer_file)
    else:
        tokenizer = convert_tekken_tokenizer(tokenizer_file)

    chat_template = _resolve_chat_template(tokenizer_file, chat_template)

    image_processor = PixtralImageProcessor(
        patch_size=patch_size,
        size={"longest_edge": max_image_size},
    )

    processor = PixtralProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token="[IMG]",
        image_break_token="[IMG_BREAK]",
        image_end_token="[IMG_END]",
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        chat_template=chat_template,
    )

    return processor
