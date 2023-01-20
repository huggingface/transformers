# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Processor class for Donut.
"""
import re
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Tuple, Union

from ...processing_utils import ProcessorMixin


class NoClosingTagError(ValueError):
    pass


class DonutProcessor(ProcessorMixin):
    r"""
    Constructs a Donut processor which wraps a Donut image processor and an XLMRoBERTa tokenizer into a single
    processor.

    [`DonutProcessor`] offers all the functionalities of [`DonutImageProcessor`] and
    [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]. See the [`~DonutProcessor.__call__`] and
    [`~DonutProcessor.decode`] for more information.

    Args:
        image_processor ([`DonutImageProcessor`]):
            An instance of [`DonutImageProcessor`]. The image processor is a required input.
        tokenizer ([`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]):
            An instance of [`XLMRobertaTokenizer`/`XLMRobertaTokenizerFast`]. The tokenizer is a required input.
    """
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if "feature_extractor" in kwargs:
            warnings.warn(
                (
                    "The `feature_extractor` argument is deprecated and will be removed in v5, use `image_processor`"
                    " instead."
                ),
                FutureWarning,
            )
            feature_extractor = kwargs.pop("feature_extractor")

        image_processor = image_processor if image_processor is not None else feature_extractor
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)
        self.current_processor = self.image_processor
        self._in_target_context_manager = False

    def __call__(self, *args, **kwargs):
        """
        When used in normal mode, this method forwards all its arguments to AutoImageProcessor's
        [`~AutoImageProcessor.__call__`] and returns its output. If used in the context
        [`~DonutProcessor.as_target_processor`] this method forwards all its arguments to DonutTokenizer's
        [`~DonutTokenizer.__call__`]. Please refer to the doctsring of the above two methods for more information.
        """
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(*args, **kwargs)

        images = kwargs.pop("images", None)
        text = kwargs.pop("text", None)
        if len(args) > 0:
            images = args[0]
            args = args[1:]

        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        if images is not None:
            inputs = self.image_processor(images, *args, **kwargs)
        if text is not None:
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to DonutTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to DonutTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to the
        docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning TrOCR.
        """
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your images inputs, or in a separate call."
        )
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.image_processor
        self._in_target_context_manager = False

    def token2json(
        self,
        tokens: str,
        is_inner_value: bool = False,
        added_vocab: Dict[str, int] = None,
    ) -> Union[List[Any], Dict[str, Any]]:
        """
        Convert a (generated) token sequence into an ordered JSON format.

        This keeps the string length of tokens constant throughout allowing for future extensions that make use of
        indices in the `tokens` string.

        Requires special tokens (except categorical tokens) to be in the format <s_{key}-{object_type}> and
        </s_{key}-{object_type}> for opening and closing tokens, respectively. The object type can be 'list', 'dict',
        or 'str'. E.g.: <s_name-str>

        Args:
            tokens (str):
                The generated text sequence
            is_inner_value (bool):
                Whether the current value is contained by another dict or list
            added_vocab (Dict[str, int]):
                Dict of added vocabulary. If it's not set, it will be derived via `tokenizer.get_added_vocab()`
        """
        if added_vocab is None:
            added_vocab = self.tokenizer.get_added_vocab()

        end_of_sequence = re.search(r"</s>$", tokens)
        if end_of_sequence:
            tokens = self._white_out_str(tokens, end_of_sequence.span())

        remaining_tokens = tokens
        out = {}
        processed_token = False  # Whether we processed any tokens at all

        while True:
            next_open_token = re.search(r"^\s*<s_(?P<group_key>.*?)-(?P<obj_type>.*?)>", remaining_tokens)
            if not next_open_token:
                break
            processed_token = True
            # if we have dict object at this level, we add them to out one by one
            remaining_tokens = self._white_out_str(remaining_tokens, next_open_token.span())
            try:
                closing_token_span = self._find_closing_tag(
                    remaining_tokens, next_open_token["group_key"], next_open_token["obj_type"]
                )
            except NoClosingTagError as e:
                warnings.warn(str(e) + "Discarding all remaining model output")
                # This effectively discards all remaining text as we cannot trust it
                closing_token_span = (next_open_token.span()[1], len(remaining_tokens))
            next_text = self._white_out_str(
                tokens, [(0, next_open_token.span()[1]), (closing_token_span[0], len(tokens))]
            )
            if next_open_token["obj_type"] == "dict":
                out[next_open_token["group_key"]] = self.token2json(
                    tokens=next_text,
                    is_inner_value=True,
                    added_vocab=added_vocab,
                )
            elif next_open_token["obj_type"] == "list":
                list_item_indices = self._get_list_item_indices(next_text)
                out[next_open_token["group_key"]] = [
                    self.token2json(
                        tokens=self._white_out_str(next_text, [(0, vi[0]), (vi[1], len(next_text))]),
                        is_inner_value=True,
                        added_vocab=added_vocab,
                    )
                    for vi in list_item_indices
                ]
            elif next_open_token["obj_type"] == "str":
                value = re.match(r"^\s*(.*?)\s*$", next_text)
                v = value.group(1)
                if v in added_vocab and v[0] == "<" and v[-2:] == "/>":
                    v = v[1:-2]  # for categorical special tokens
                out[next_open_token["group_key"]] = v
            else:
                raise RuntimeError(f"Unknown object type: {next_open_token['obj_type']}")
            # remove the part we just processed from the remaining text
            remaining_tokens = self._white_out_str(
                remaining_tokens, (next_open_token.span()[0], closing_token_span[1])
            )

        if not processed_token:
            # This happens when we process individual items in a list that are just string values
            value = re.match(r"^\s*(.*?)\s*$", tokens)
            v = value.group(1)
            if v in added_vocab and v[0] == "<" and v[-2:] == "/>":
                v = v[1:-2]  # for categorical special tokens
            return v

        if not re.match(r"^\s*$", remaining_tokens):
            warnings.warn(f"Text remaining: {remaining_tokens}")

        if not out and is_inner_value:
            # we failed to extract anything
            return {"text_sequence": tokens}
        return out

    @property
    def feature_extractor_class(self):
        warnings.warn(
            "`feature_extractor_class` is deprecated and will be removed in v5. Use `image_processor_class` instead.",
            FutureWarning,
        )
        return self.image_processor_class

    @property
    def feature_extractor(self):
        warnings.warn(
            "`feature_extractor` is deprecated and will be removed in v5. Use `image_processor` instead.",
            FutureWarning,
        )
        return self.image_processor

    @staticmethod
    def _white_out_str(text: str, span: Union[Tuple[int, int], List[Tuple[int, int]]]) -> str:
        """
        Replace span indices in text with equal-length whitespace

        Args:
            text (str):
                Text to blot something out in
            span (Union[Tuple[int, int], List[Tuple[int, int]]]):
                (List of) tuple(s) of integers of length 2 between which to replace the text with spaces
        Returns:
            str: Equal-length text with spaces in the span
        """
        if not isinstance(span, list):
            span = [span]
        for s in span:
            text = text[: s[0]] + " " * (s[1] - s[0]) + text[s[1] :]
        return text

    @staticmethod
    def _find_closing_tag(text: str, group_key: str, obj_type: Literal["str", "dict", "list"]) -> Tuple[int, int]:
        """
        Find the span of the closing tag that closes the group belonging to the group_key

        The starting tag must already be blotted out from text

        Args:
            text (str):
                Text with starting text replaced with whitespace
        group_key (str):
            Group key for which to find the closing tag
        obj_type (Literal["str", "dict", "list"]):
            Type of the object we're closing
        Returns:
            Tuple[int, int]: Span/indices of the closing tag in text
        """
        currently_open = [(group_key, obj_type)]  # collect open group keys here
        for next_token in re.finditer(r"<(?P<closing>/)?s_(?P<group_key>.*?)-(?P<obj_type>.*?)>", text):
            if next_token["closing"]:
                if (next_token["group_key"], next_token["obj_type"]) == currently_open[-1]:
                    currently_open.pop(-1)
                    if not currently_open:
                        # we just closed the remaining (outermost) group, so this is our answer
                        return next_token.span()
                else:
                    warnings.warn(
                        f"Closing token not currently open: {next_token['group_key']} type {next_token['obj_type']} "
                        "Switching to fallback mode"
                    )
                    if (next_token["obj_type"] == "str") & (currently_open[-1][1] == "str"):
                        # assume we're just closing with an incorrect token and try to move on
                        currently_open.pop(-1)
                        if not currently_open:
                            warnings.warn(
                                (
                                    f"Fallback mode failed. Discarding remaining text text='{text}', "
                                    f"group_key='{group_key}', obj_type='{obj_type}'"
                                ),
                            )
                            break  # this will get us out of the for-loop and raise NoClosingTagError
            else:
                currently_open.append((next_token["group_key"], next_token["obj_type"]))
        raise NoClosingTagError(f"Cannot find closing token for {group_key} and type {obj_type} in {text}")

    def _get_list_item_indices(self, text: str):
        # remove everything between open and close tags to make sure we don't pick up on lists nested further down
        stripped_text = text
        while True:
            open_tag = re.search(r"<s_(?P<group_key>.*?)-(?P<obj_type>.*?)>", stripped_text)
            if not open_tag:
                break
            # white-out any group we come across at this level, so we can check for remaining separating tokens at this
            # level
            stripped_text = self._white_out_str(stripped_text, open_tag.span())
            try:
                closing_token_span = self._find_closing_tag(stripped_text, open_tag["group_key"], open_tag["obj_type"])
            except NoClosingTagError as e:
                warnings.warn(str(e) + "Discarding all remaining model output")
                # discard all remaining text as we cannot trust it
                closing_token_span = (None, len(text))
            stripped_text = self._white_out_str(stripped_text, (open_tag.span()[0], closing_token_span[1]))
        # now split the current object along the separating tags if they exist
        split_points = list(re.finditer(r"<sep/>", stripped_text))
        value_indices = zip(
            [0] + [sp.span()[1] for sp in split_points], [sp.span()[0] for sp in split_points] + [len(text)]
        )
        return value_indices
