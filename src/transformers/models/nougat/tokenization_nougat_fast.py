# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""
Fast tokenizer class for Nougat.
"""
import re
from functools import partial
from multiprocessing import Pool
from typing import List, Union

import numpy as np

from transformers.tokenization_utils_base import INIT_TOKENIZER_DOCSTRING
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import add_end_docstrings

from ...utils import is_levenshtein_available, is_nltk_available, requires_backends


if is_levenshtein_available():
    from Levenshtein import ratio

if is_nltk_available():
    from nltk.corpus import words


INIT_TOKENIZER_DOCSTRING += """
        tokenizer_object ([`tokenizers.Tokenizer`]):
            A [`tokenizers.Tokenizer`] object from 🤗 tokenizers to instantiate from. See [Using tokenizers from 🤗
            tokenizers](../fast_tokenizers) for more information.
        tokenizer_file ([`str`]):
            A path to a local JSON file representing a previously serialized [`tokenizers.Tokenizer`] object from 🤗
            tokenizers.
"""


PRETRAINED_VOCAB_FILES_MAP = {
    "tokenizer_file": {
        "nielsr/nougat": "https://huggingface.co/nielsr/nougat/tokenizer/blob/main/tokenizer.json",
    },
}

VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}


def markdown_compatible(text: str) -> str:
    """
    Make text compatible with Markdown formatting.

    This function makes various text formatting adjustments to make it compatible with Markdown.

    Args:
        text (`str`):
            The input text to be made Markdown-compatible.

    Returns:
        `str`: The Markdown-compatible text.
    """
    # equation tag
    text = re.textub(r"^\(([\d.]+[a-zA-Z]?)\) \\\[(.+?)\\\]$", r"\[\2 \\tag{\1}\]", text, flags=re.M)
    text = re.sub(r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\)$", r"\[\1 \\tag{\2}\]", text, flags=re.M)
    text = re.sub(
        r"^\\\[(.+?)\\\] \(([\d.]+[a-zA-Z]?)\) (\\\[.+?\\\])$",
        r"\[\1 \\tag{\2}\] \3",
        text,
        flags=re.M,
    )  # multi line
    text = text.replace(r"\. ", ". ")
    # bold formatting
    text = text.replace(r"\bm{", r"\mathbf{").replace(r"{\\bm ", r"\mathbf{")
    text = re.sub(r"\\mbox{ ?\\boldmath\$(.*?)\$}", r"\\mathbf{\1}", text)
    # urls
    text = re.sub(
        r"((?:http|ftp|https):\/\/(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-]))",
        r"[\1](\1)",
        text,
    )
    # algorithms
    text = re.sub(r"```\s*(.+?)\s*```", r"```\n\1\n```", text, flags=re.S)

    return text


def find_next_punctuation(s: str, start_inx=0):
    """
    Find the index of the next punctuation mark

    Args:
        s: String to examine
        start_inx: Index where to start
    """

    for i in range(start_inx, len(s)):
        if s[i] in [".", "?", "!", "\n"]:
            return i

    return None


def find_last_punctuation(s: str, start_inx=0):
    """
    Find the index of the last punctuation mark before start_inx

    Args:
        s: String to examine
        start_inx: Index where to look before
    """

    for i in range(start_inx - 1, 0, -1):
        if s[i] in [".", "?", "!", "\n"]:
            return i

    return None


def truncate_repetitions(s: str, min_len=30):
    """
    Attempt to truncate repeating segments in the input string.

    This function looks for the longest repeating substring at the end of the input string and truncates it to appear
    only once. To be considered for removal, repetitions need to be continuous.

    Args:
        s (str): The input raw prediction to be truncated.
        min_len (int): The minimum length of the repeating segment.

    Returns:
        str: The input string with repeated segments truncated.
    """
    s_lower = s.lower()
    s_len = len(s_lower)

    if s_len < 2 * min_len:
        return s

    # try to find a length at which the tail is repeating
    max_rep_len = None
    for rep_len in range(min_len, int(s_len / 2)):
        # check if there is a repetition at the end
        same = True
        for i in range(0, rep_len):
            if s_lower[s_len - rep_len - i - 1] != s_lower[s_len - i - 1]:
                same = False
                break

        if same:
            max_rep_len = rep_len

    if max_rep_len is None:
        return s

    lcs = s_lower[-max_rep_len:]

    # remove all but the last repetition
    st = s
    st_lower = s_lower
    while st_lower.endswith(lcs):
        st = st[:-max_rep_len]
        st_lower = st_lower[:-max_rep_len]

    # this is the tail with the repetitions
    repeating_tail = s_lower[len(st_lower) :]

    # add until next punctuation and make sure last sentence is not repeating
    st_lower_out = st_lower
    while True:
        sentence_end = find_next_punctuation(s_lower, len(st_lower_out))
        sentence_start = find_last_punctuation(s_lower, len(st_lower_out))
        if sentence_end and sentence_start:
            sentence = s_lower[sentence_start:sentence_end]
            st_lower_out = s_lower[: sentence_end + 1]
            if sentence in repeating_tail:
                break
        else:
            break

    s_out = s[: len(st_lower_out)]

    return s_out


def remove_numbers(lines):
    def _clean(s):
        return re.sub(r"(?:[\d_]|\*\*)", "", s).strip()

    if type(lines) is str:
        return _clean(lines)
    out = []
    for l in lines:
        out.append(_clean(l))
    return out


def get_slices(lines, clean_lines):
    """
    Get slices of text based on specific criteria within the lines.

    This function identifies and returns slices of text from the input lines based on certain conditions.

    These conditions were chosen by the Nougat authors:
    - The slice is less than 200 characters long.
    - The slice is more than 3 characters long.
    - The slice does not start with "[MISSING_PAGE".
    - The slice is either the same as the next slice or the ratio of the two in terms of Levensthein distance is
      greater than 0.9.

    Args:
        lines (`List[str]`):
            The list of lines containing the text.
        clean_lines (`List[str]`):
            A cleaned version of the text (without numbers).

    Returns:
        `List[tuple]`: A list of tuples representing the start and end indices of text slices.
    """
    inds = np.zeros(len(lines))
    for i in range(len(lines) - 1):
        j = i + 1
        while not clean_lines[j] and j < len(lines) - 1:
            j += 1
        if (
            len(clean_lines[i]) < 200
            and len(clean_lines[i]) > 3
            and len(clean_lines[j]) < 200
            and len(clean_lines[j]) > 3
            and not clean_lines[i].startswith("[MISSING_PAGE")
            and (clean_lines[i] == clean_lines[j] or ratio(clean_lines[i], clean_lines[j]) > 0.9)
        ):
            inds[i:j] = 1
    ids = np.where(inds)[0]
    slices = []
    if len(ids) == 0:
        return slices
    j0 = 0
    for j, x in enumerate(np.diff(ids) > 3):
        if x:
            slices.append((ids[j0], ids[j] + 2))
            j0 = j + 1
    slices.append((ids[j0], ids[-1] + 2))
    return [sli for sli in slices if sli[1] - sli[0] > 15]


def remove_slice_from_lines(lines, clean_text, slice) -> str:
    """
    Remove a slice of text from the lines based on specific criteria.

    This function identifies a slice of text within the lines and removes it based on certain conditions.

    Args:
        lines (list of str): The list of lines containing the text.
        clean_text (list of str): A cleaned version of the text (without numbers).
        slice (tuple): A tuple representing the start and end indices of the slice to be removed.

    Returns:
        str: The removed slice of text as a single string.
    """
    base = clean_text[slice[0]]
    section = list(slice)
    check_start_flag = False
    # backwards pass, at most 5 lines
    for line_idx in range(max(0, slice[0] - 1), max(0, slice[0] - 5), -1):
        if not lines[line_idx]:
            continue
        if lines[line_idx] == "## References":
            section[0] = line_idx
            break
        elif ratio(base, remove_numbers(lines[line_idx])) < 0.9:
            section[0] = line_idx + 1
            potential_ref = remove_numbers(lines[max(0, line_idx - 1)].partition("* [")[-1])
            if len(potential_ref) >= 0.75 * len(base) and ratio(base, potential_ref) < 0.9:
                section[0] = line_idx
            check_start_flag = True
            break
    # forward pass, at most 5 lines
    for line_idx in range(min(len(lines), slice[1]), min(len(lines), slice[1] + 5)):
        if ratio(base, remove_numbers(lines[line_idx])) < 0.9:
            section[1] = line_idx
            break
    if len(lines) <= section[1]:
        section[1] = len(lines) - 1
    to_delete = "\n".join(lines[section[0] : section[1] + 1])
    # cut off next page content
    itera, iterb = enumerate(lines[section[1] - 1]), enumerate(lines[section[1]])
    while True:
        try:
            (ia, a) = next(itera)
            while a.isnumeric():
                (ia, a) = next(itera)
            (ib, b) = next(iterb)
            while b.isnumeric():
                (ib, b) = next(iterb)
            if a != b:
                break
        except StopIteration:
            break
    if check_start_flag and "* [" in to_delete:
        to_delete = "* [" + to_delete.partition("* [")[-1]
    try:
        delta = len(lines[section[1]]) - ib - 1
        if delta > 0:
            to_delete = to_delete[:-delta]
    except UnboundLocalError:
        pass

    return to_delete.strip()


@add_end_docstrings(INIT_TOKENIZER_DOCSTRING)
class NougatTokenizerFast(PreTrainedTokenizerFast):
    """
    Fast tokenizer for Nougat (backed by HuggingFace tokenizers library).

    Inherits from [`~tokenization_utils_base.PreTrainedTokenizerFast`] with additional Nougat-specific postprocessing.
    Users should refer to this superclass for more information regarding those methods.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    def remove_hallucinated_references(self, text: str) -> str:
        """
        Remove hallucinated or missing references from the text.

        This function identifies and removes references that are marked as missing or hallucinated from the input text.

        Args:
            text (str): The input text containing references.

        Returns:
            str: The text with hallucinated references removed.
        """
        lines = text.split("\n")
        if len(lines) == 0:
            return ""
        clean_lines = remove_numbers(lines)
        slices = get_slices(lines, clean_lines)
        to_delete = []
        for slice in slices:
            to_delete.append(remove_slice_from_lines(lines, clean_lines, slice))
        for to_delete in reversed(to_delete):
            text = text.replace(to_delete, "\n\n[MISSING_PAGE_POST]\n\n")
        text = re.sub(
            r"## References\n+\[MISSING_PAGE_POST(:\d+)?\]",
            "\n\n[MISSING_PAGE_POST\\1]",
            text,
        )
        return text

    def post_process_single(self, generation: str, fix_markdown: bool = True) -> str:
        """
        Postprocess a single generated text.

        Args:
            generation (str): The generated text to be postprocessed.
            fix_markdown (bool, optional): Whether to perform Markdown formatting fixes. Default is True.

        Returns:
            str: The postprocessed text.
        """
        generation = re.sub(
            r"(?:\n|^)#+ \d*\W? ?(.{100,})", r"\n\1", generation
        )  # too long section titles probably are none
        generation = generation.strip()
        generation = generation.replace("\n* [leftmargin=*]\n", "\n")
        generation = re.sub(r"^#+ (?:\.?(?:\d|[ixv])+)*\s*(?:$|\n\s*)", "", generation, flags=re.M)
        # most likely hallucinated titles
        lines = generation.split("\n")
        if lines[-1].startswith("#") and lines[-1].lstrip("#").startswith(" ") and len(lines) > 1:
            print("INFO: likely hallucinated title at the end of the page: " + lines[-1])
            generation = "\n".join(lines[:-1])
        # obvious repetition detection
        generation = truncate_repetitions(generation)
        # Reference corrections
        generation = self.remove_hallucinated_references(generation)
        generation = re.sub(r"^\* \[\d+\](\s?[A-W]\.+\s?){10,}.*$", "", generation, flags=re.M)
        generation = re.sub(r"^(\* \[\d+\])\[\](.*)$", r"\1\2", generation, flags=re.M)
        generation = re.sub(r"(^\w\n\n|\n\n\w$)", "", generation)
        # pmc math artifact correction
        generation = re.sub(
            r"([\s.,()])_([a-zA-Z0-9])__([a-zA-Z0-9]){1,3}_([\s.,:()])",
            r"\1\(\2_{\3}\)\4",
            generation,
        )
        generation = re.sub(r"([\s.,\d])_([a-zA-Z0-9])_([\s.,\d;])", r"\1\(\2\)\3", generation)
        # footnote mistakes
        generation = re.sub(
            r"(\nFootnote .*?:) (?:footnotetext|thanks):\W*(.*(?:\n\n|$))",
            r"\1 \2",
            generation,
        )
        # TODO Come up with footnote formatting inside a table
        generation = re.sub(r"\[FOOTNOTE:.+?\](.*?)\[ENDFOOTNOTE\]", "", generation)
        # itemize post processing
        for match in reversed(
            list(
                re.finditer(
                    r"(?:^)(-|\*)?(?!-|\*) ?((?:\d|[ixv])+ )?.+? (-|\*) (((?:\d|[ixv])+)\.(\d|[ixv]) )?.*(?:$)",
                    generation,
                    flags=re.I | re.M,
                )
            )
        ):
            start, stop = match.span()
            delim = match.group(3) + " "
            splits = match.group(0).split(delim)
            replacement = ""
            if match.group(1) is not None:
                splits = splits[1:]
                delim1 = match.group(1) + " "
            else:
                delim1 = ""
                # too many false positives
                continue
            pre, post = generation[:start], generation[stop:]
            for i, item in enumerate(splits):
                level = 0
                potential_numeral, _, rest = item.strip().partition(" ")
                if not rest:
                    continue
                if re.match(r"^[\dixv]+((?:\.[\dixv])?)+$", potential_numeral, flags=re.I | re.M):
                    level = potential_numeral.count(".")

                replacement += (
                    ("\n" if i > 0 else "")
                    + ("\t" * level)
                    + (delim if i > 0 or start == 0 else delim1)
                    + item.strip()
                )
            if post == "":
                post = "\n"
            generation = pre + replacement + post

        if generation.endswith((".", "}")):
            generation += "\n\n"
        if re.match(r"[A-Z0-9,;:]$", generation):
            # add space in case it there is a comma or word ending
            generation += " "
        elif generation.startswith(("#", "**", "\\begin")):
            generation = "\n\n" + generation
        elif generation.split("\n")[-1].startswith(("#", "Figure", "Table")):
            generation = generation + "\n\n"
        else:
            try:
                last_word = generation.split(" ")[-1]
                if last_word in words.words():
                    generation += " "
            except LookupError:
                # add space just in case. Will split words but better than concatenating them
                generation += " "
                # download for the next time
                import nltk

                nltk.download("words")
        # table corrections
        # remove obvious wrong tables
        for l in generation.split("\n"):
            if l.count("\\begin{tabular}") > 15 or l.count("\\multicolumn") > 60 or l.count("&") > 400:
                generation = generation.replace(l, "")
        # whitespace corrections
        generation = generation.replace("\\begin{table} \\begin{tabular}", "\\begin{table}\n\\begin{tabular}")
        generation = generation.replace("\\end{tabular} \\end{table}", "\\end{tabular}\n\\end{table}")
        generation = generation.replace("\\end{table} Tab", "\\end{table}\nTab")
        generation = re.sub(r"(^.+)\\begin{tab", r"\1\n\\begin{tab", generation, flags=re.M)

        generation = generation.replace(r"\begin{tabular}{l l}  & \\ \end{tabular}", "").replace(
            "\\begin{tabular}{}\n\n\\end{tabular}", ""
        )
        generation = generation.replace("\\begin{array}[]{", "\\begin{array}{")
        generation = re.sub(
            r"\\begin{tabular}{([clr ]){2,}}\s*[& ]*\s*(\\\\)? \\end{tabular}",
            "",
            generation,
        )
        generation = re.sub(r"(\*\*S\. A\. B\.\*\*\n+){2,}", "", generation)
        generation = re.sub(r"^#+( [\[\d\w])?$", "", generation, flags=re.M)
        generation = re.sub(r"^\.\s*$", "", generation, flags=re.M)
        generation = re.sub(r"\n{3,}", "\n\n", generation)
        if fix_markdown:
            return markdown_compatible(generation)
        else:
            return generation

    def post_process_generation(
        self,
        generation: Union[str, List[str]],
        fix_markdown: bool = True,
        num_workers: int = None,
    ) -> Union[str, List[str]]:
        """
        Postprocess a generated text or a list of generated texts.

        This function can be used to perform postprocessing on generated text, such as fixing Markdown formatting.

        Args:
            generation (Union[str, List[str]]):
                The generated text or a list of generated texts.
            fix_markdown (`bool`, *optional*, defaults to `True`):
                Whether to perform Markdown formatting fixes.
            num_workers (`int`, *optional*):
                Optional number of workers to pass to leverage multiprocessing (postprocessing several texts in
                parallel).

        Returns:
            Union[str, List[str]]: The postprocessed text or list of postprocessed texts.
        """
        requires_backends(self, ["nltk", "levenshtein"])

        if isinstance(generation, list):
            if num_workers is not None and isinstance(num_workers, int):
                with Pool(num_workers) as p:
                    return p.map(partial(self.post_process_single, fix_markdown=fix_markdown), generation)
            else:
                return [self.post_process_single(s, fix_markdown=fix_markdown) for s in generation]
        else:
            return self.post_process_single(generation, fix_markdown=fix_markdown)
