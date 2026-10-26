# Copyright 2026 H Company and the HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the NeoMME processor."""

import tempfile
import unittest

import numpy as np
from jinja2.exceptions import TemplateError
from parameterized import parameterized

from transformers.testing_utils import require_tokenizers, require_torch, require_vision
from transformers.utils import is_tokenizers_available, is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_tokenizers_available():
    from tokenizers import Tokenizer, models, pre_tokenizers

if is_vision_available():
    from PIL import Image

    from transformers import NeoMMEImageProcessor, NeoMMEProcessor, PreTrainedTokenizerFast

if is_torch_available():
    import torch


@require_torch
@require_vision
@require_tokenizers
class NeoMMEProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = NeoMMEProcessor if is_vision_available() else None
    patch_size = 4
    chat_template = """
{%- if task is not defined -%}
    {{- raise_exception("NeoMME chat templates require task='query' or task='document'.") -}}
{%- endif -%}
{%- if task not in ['query', 'document'] -%}
    {{- raise_exception("task=" ~ task ~ " is not supported: expected 'query' or 'document'.") -}}
{%- endif -%}
{%- if messages is not defined or not messages -%}
    {{- raise_exception("NeoMME chat conversations must contain at least one message.") -}}
{%- endif -%}

{%- set state = namespace(text='', has_text=false, image_count=0) -%}
{%- for message in messages -%}
    {%- set content = message.content -%}
    {%- set items = [{'type': 'text', 'text': content}] if content is string else content -%}
    {%- for item in items -%}
        {%- if item.type == 'text' -%}
            {%- if image_token in item.text -%}
                {{- raise_exception(image_token ~ " is reserved for image documents.") -}}
            {%- endif -%}
            {%- set state.has_text = true -%}
            {%- set state.text = state.text + item.text -%}
        {%- elif item.type == 'image' -%}
            {%- if item.image is not defined or item.image is none or item.image == '' -%}
                {{- raise_exception("NeoMME image content must provide an image source.") -}}
            {%- endif -%}
            {%- set state.image_count = state.image_count + 1 -%}
        {%- elif item.type == 'image_url' -%}
            {%- if item.image_url is not defined or not item.image_url -%}
                {{- raise_exception("NeoMME image_url content must provide an image source.") -}}
            {%- endif -%}
            {%- set state.image_count = state.image_count + 1 -%}
        {%- else -%}
            {{- raise_exception("NeoMME chat templates do not support content type " ~ item.type ~ ".") -}}
        {%- endif -%}
    {%- endfor -%}
{%- endfor -%}

{%- if state.image_count and state.has_text -%}
    {{- raise_exception("NeoMME cannot encode text and images in the same conversation.") -}}
{%- endif -%}
{%- if state.image_count > 1 -%}
    {{- raise_exception("NeoMME accepts one image document per conversation.") -}}
{%- endif -%}
{%- if state.image_count and task != 'document' -%}
    {{- raise_exception("NeoMME image content must use task='document'.") -}}
{%- endif -%}

{%- set content = image_token if state.image_count else state.text -%}
{%- if task == 'query' -%}
    {{- query_token + content + mask_token * 10 -}}
{%- else -%}
    {{- document_token + content -}}
{%- endif -%}
"""
    # Each token's ID must equal its index in this list.
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "<mask>", "<doc>", "<img>", "<query>", "<row>"]

    @classmethod
    def _setup_tokenizer(cls, specials: list[str] | None = None) -> "PreTrainedTokenizerFast":
        specials = specials if specials is not None else cls.special_tokens
        vocab_words = ["hello", "world", "a", "document", "query", "text", "lower", "newer"]
        vocabulary = {token: index for index, token in enumerate(specials)}
        for word in vocab_words:
            vocabulary[word] = len(vocabulary)

        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
            backend.save(handle.name)
            return PreTrainedTokenizerFast(
                tokenizer_file=handle.name,
                pad_token="<pad>",
                eos_token="<eos>",
                unk_token="<unk>",
                mask_token="<mask>",
                # Passing a missing marker here would add it to the vocabulary.
                extra_special_tokens={
                    name: token
                    for name, token in {
                        "document_token": "<doc>",
                        "image_token": "<img>",
                        "query_token": "<query>",
                        "row_token": "<row>",
                    }.items()
                    if token in vocabulary
                },
            )

    @classmethod
    def setUpClass(cls):
        cls.tmpdirname = tempfile.mkdtemp()
        processor = cls.processor_class(
            image_processor=NeoMMEImageProcessor(
                patch_size=cls.patch_size, size={"min_pixels": 10, "max_pixels": 200}
            ),
            tokenizer=cls._setup_tokenizer(),
            chat_template=cls.chat_template,
        )
        cls._setup_test_attributes(processor)
        processor.save_pretrained(cls.tmpdirname)

    @property
    def marker_ids(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.special_tokens)}

    @unittest.skip(reason="NeoMME image batches require matching image placeholders")
    def test_processor_with_multiple_inputs(self):
        pass

    @unittest.skip(reason="NeoMME chat templates must declare the retrieval task")
    def test_apply_chat_template_assistant_mask(self):
        pass

    @unittest.skip(reason="NeoMME chat templates must declare the retrieval task")
    def test_chat_template_jinja_kwargs(self):
        pass

    @unittest.skip("tiny model has too little tokens and collapses everything to UNK which is not defined")
    def test_replacement_offsets(self):
        pass

    def _set_retrieval_chat_template(self, processor):
        processor.chat_template = self.chat_template

    @staticmethod
    def prepare_processor_dict():
        return {}

    def prepare_text_inputs(self, batch_size: int | None = None, modalities: str | list | None = None):
        if isinstance(modalities, str):
            modalities = [modalities]

        batch_size = batch_size if batch_size is not None else 1
        if modalities is not None and ("image" in modalities or "images" in modalities):
            return ["<doc><img>"] * batch_size
        else:
            return ["<doc> lower newer"] * batch_size

    def _apply_text(self, processor, text, task="query", **processor_kwargs):
        text = [text] if isinstance(text, str) else text
        messages = [[{"role": "user", "content": value}] for value in text]
        processor_kwargs.setdefault("padding", "longest")
        processor_kwargs.setdefault("return_tensors", "pt")
        return processor.apply_chat_template(
            messages,
            task=task,
            tokenize=True,
            return_dict=True,
            processor_kwargs=processor_kwargs,
        )

    def _apply_images(self, processor, images, **processor_kwargs):
        images = images if isinstance(images, (list, tuple)) else [images]
        messages = [[{"role": "user", "content": [{"type": "image", "image": image}]}] for image in images]
        processor_kwargs.setdefault("padding", "longest")
        processor_kwargs.setdefault("return_tensors", "pt")
        return processor.apply_chat_template(
            messages,
            task="document",
            tokenize=True,
            return_dict=True,
            processor_kwargs=processor_kwargs,
        )

    def test_apply_chat_template_query(self):
        processor = self.get_processor()
        self._set_retrieval_chat_template(processor)
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

        inputs = processor.apply_chat_template(
            messages, task="query", tokenize=True, return_dict=True, return_tensors="pt"
        )
        ids = inputs["input_ids"][0].tolist()

        self.assertEqual(ids.count(self.marker_ids["<query>"]), 1)
        self.assertIn(processor.tokenizer.convert_tokens_to_ids("hello"), ids)
        self.assertEqual(ids[-10:], [self.marker_ids["<mask>"]] * 10)

    def test_apply_chat_template_text_document(self):
        processor = self.get_processor()
        self._set_retrieval_chat_template(processor)
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

        inputs = processor.apply_chat_template(
            messages, task="document", tokenize=True, return_dict=True, return_tensors="pt"
        )
        ids = inputs["input_ids"][0].tolist()

        self.assertEqual(ids.count(self.marker_ids["<doc>"]), 1)
        self.assertIn(processor.tokenizer.convert_tokens_to_ids("hello"), ids)
        self.assertNotIn(self.marker_ids["<mask>"], ids)

    def test_apply_chat_template_preserves_processing_kwargs(self):
        processor = self.get_processor()
        self._set_retrieval_chat_template(processor)
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello world"}]}]

        inputs = processor.apply_chat_template(
            messages,
            task="document",
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"max_length": 2, "padding": "max_length", "truncation": True},
        )

        hello_id = processor.tokenizer.convert_tokens_to_ids("hello")
        self.assertEqual(inputs["input_ids"][0].tolist(), [self.marker_ids["<doc>"], hello_id])

    @parameterized.expand([(1, "pt"), (2, "pt")])
    def test_apply_chat_template_image(self, batch_size, return_tensors):
        processor = self.get_processor()
        self._set_retrieval_chat_template(processor)
        image = Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))
        messages = [[{"role": "user", "content": [{"type": "image", "image": image}]}] for _ in range(batch_size)]

        inputs = processor.apply_chat_template(
            messages, task="document", tokenize=True, return_dict=True, return_tensors=return_tensors
        )

        self.assertEqual(inputs["input_ids"].shape[0], batch_size)
        self.assertTrue(torch.all(inputs["input_ids"][:, 0] == self.marker_ids["<doc>"]))
        self.assertEqual(inputs["position_ids"].shape[1], batch_size)
        self.assertNotIn("image_grid_hw", inputs)
        self.assertIn("pixel_values", inputs)

        direct = processor(images=[image] * batch_size, return_tensors=return_tensors)
        torch.testing.assert_close(inputs["input_ids"], direct["input_ids"])
        torch.testing.assert_close(inputs["position_ids"], direct["position_ids"])
        torch.testing.assert_close(inputs["pixel_values"], direct["pixel_values"])

    def test_apply_chat_template_rejects_invalid_task(self):
        processor = self.get_processor()
        self._set_retrieval_chat_template(processor)
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

        with self.assertRaisesRegex(TemplateError, "expected 'query' or 'document'"):
            processor.apply_chat_template(messages, task="invalid", tokenize=True)

        processor.chat_template = "{{ task and 'hello' }}"
        with self.assertRaisesRegex(ValueError, "leading task marker"):
            processor.apply_chat_template(messages, task="query", tokenize=True)

        processor.chat_template = "{{ messages[0].content[0].text }}"
        with self.assertRaisesRegex(ValueError, "leading task marker"):
            processor.apply_chat_template(messages, task="query", tokenize=True)

        processor.chat_template = "{{ messages[0].content[0].text + query_token }}"
        with self.assertRaisesRegex(ValueError, "leading task marker"):
            processor.apply_chat_template(messages, task="query", tokenize=True)

        for chat_template in ("{{ query_token if task }}", "{{ query_token + messages[0].content[0].text * 2 }}"):
            processor.chat_template = chat_template
            with self.assertRaisesRegex(ValueError, "render the text content exactly once"):
                processor.apply_chat_template(messages, task="query", tokenize=True)

    def test_apply_chat_template_does_not_require_task(self):
        processor = self.get_processor()
        processor.chat_template = "{{ messages[0].content }}"
        messages = [{"role": "user", "content": "hello"}]

        self.assertEqual(processor.apply_chat_template(messages, tokenize=False), "hello")

    def test_apply_chat_template_rejects_unsupported_inputs(self):
        processor = self.get_processor()
        self._set_retrieval_chat_template(processor)
        image = Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))
        image_messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        cases = [
            (
                "mixed content",
                [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": image}, {"type": "text", "text": "hello"}],
                    }
                ],
                "document",
                "cannot encode text and images in the same conversation",
            ),
            ("image query", image_messages, "query", "must use task='document'"),
            (
                "multiple images",
                [{"role": "user", "content": [{"type": "image", "image": image}] * 2}],
                "document",
                "one image document per conversation",
            ),
            (
                "video",
                [{"role": "user", "content": [{"type": "video", "video": "example.mp4"}]}],
                "document",
                "do not support content type video",
            ),
            (
                "missing image source",
                [{"role": "user", "content": [{"type": "image"}]}],
                "document",
                "must provide an image source",
            ),
        ]

        for name, messages, task, error in cases:
            with self.subTest(name=name), self.assertRaisesRegex(TemplateError, error):
                processor.apply_chat_template(messages, task=task, tokenize=True)

    def test_processor_text_has_no_visual(self):
        processor = self.get_processor()
        self._set_retrieval_chat_template(processor)
        image = Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))
        messages = [
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            [{"role": "user", "content": [{"type": "image", "image": image}]}],
        ]

        inputs = processor.apply_chat_template(
            messages,
            task="document",
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": "longest"},
        )

        self.assertEqual(inputs["input_ids"].shape[0], 2)
        self.assertTrue(torch.all(inputs["input_ids"][:, 0] == self.marker_ids["<doc>"]))
        self.assertEqual(inputs["position_ids"].shape[1], 2)
        self.assertIn("pixel_values", inputs)

        direct_inputs = processor(
            text=[
                processor.tokenizer.document_token + "hello",
                processor.tokenizer.document_token + processor.image_token,
            ],
            images=[[], [image]],
            padding=True,
            return_tensors="pt",
        )
        for key in ("input_ids", "attention_mask", "pixel_values", "position_ids"):
            torch.testing.assert_close(inputs[key], direct_inputs[key])

    def test_apply_chat_template_rejects_assistant_mask(self):
        processor = self.get_processor()
        self._set_retrieval_chat_template(processor)
        image = Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))

        for messages in (
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": [{"type": "image", "image": image}]}],
        ):
            with (
                self.subTest(messages=messages),
                self.assertRaisesRegex(ValueError, "do not support `return_assistant_tokens_mask`"),
            ):
                processor.apply_chat_template(
                    messages, task="document", tokenize=True, return_assistant_tokens_mask=True
                )

    def test_image_token_is_reserved_and_required(self):
        processor = self.get_processor()
        placeholder = processor.image_token

        with self.assertRaisesRegex(TemplateError, "reserved"):
            processor.apply_chat_template(
                [{"role": "user", "content": f"hello {placeholder}"}],
                task="query",
                tokenize=False,
            )

        image = Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))
        messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        processor.chat_template = (
            "{% if task == 'document' %}{{ document_token }}{% else %}{{ query_token }}{% endif %}"
        )
        with self.assertRaisesRegex(ValueError, "image prompts"):
            processor.apply_chat_template(messages, task="document", tokenize=True)

        processor.chat_template = "{% if task %}{{ document_token + image_token + row_token }}{% endif %}"
        with self.assertRaisesRegex(ValueError, "invalid or truncated token layout"):
            processor.apply_chat_template(messages, task="document", tokenize=True)

    def test_zero_query_expansion_template(self):
        processor = self.get_processor()
        processor.chat_template = self.chat_template.replace("mask_token * 10", "mask_token * 0")

        inputs = self._apply_text(processor, ["hello"], task="query", return_tensors="pt")
        hello_id = processor.tokenizer.convert_tokens_to_ids("hello")
        self.assertEqual(inputs["input_ids"][0].tolist(), [self.marker_ids["<query>"], hello_id])

    def test_model_input_names(self):
        processor = self.get_processor()
        image_inputs = self._apply_images(processor, self.prepare_images_inputs())
        self.assertSetEqual(set(image_inputs.keys()), set(processor.model_input_names))

        # Text queries must not include vision inputs.
        query_inputs = self._apply_text(processor, ["hello"], task="query")
        self.assertSetEqual(set(query_inputs.keys()), {"input_ids", "attention_mask"})

    def test_padding_and_return_tensors(self):
        """Padding and `return_tensors` used to be dropped; only `max_length` survived the merge."""
        processor = self.get_processor()

        padded = self._apply_text(
            processor,
            ["hello world", "a"],
            task="document",
            padding="max_length",
            max_length=32,
        )
        self.assertEqual(padded["input_ids"].shape, (2, 32))
        self.assertEqual(int(padded["attention_mask"][1].sum()), 2)

        for return_tensors, expected in (("np", np.ndarray), ("pt", torch.Tensor)):
            with self.subTest(return_tensors=return_tensors):
                batch = self._apply_text(
                    processor,
                    ["hello world"],
                    task="query",
                    return_tensors=return_tensors,
                )
                self.assertIsInstance(batch["input_ids"], expected)

        ragged = self._apply_text(
            processor,
            ["hello world", "a"],
            task="document",
            padding=False,
            return_tensors=None,
        )
        self.assertIsInstance(ragged["input_ids"], list)
        self.assertNotEqual(len(ragged["input_ids"][0]), len(ragged["input_ids"][1]))

        multiple_of_8 = self._apply_text(processor, ["hello"], task="query", pad_to_multiple_of=8)
        self.assertEqual(multiple_of_8["input_ids"].shape[1], 16)

    def test_query_marker_and_expansion(self):
        processor = self.get_processor()
        batch = self._apply_text(processor, ["hello world", "a"], task="query")
        first = batch["input_ids"][0].tolist()

        self.assertEqual(first[0], self.marker_ids["<query>"])
        self.assertEqual(first[-10:], [self.marker_ids["<mask>"]] * 10)
        self.assertEqual(len(first), 1 + 2 + 10)
        # The shorter query is right-padded and its padding is masked out.
        self.assertEqual(int(batch["attention_mask"][1].sum()), 1 + 1 + 10)

    @parameterized.expand([("right",), ("left",)])
    def test_query_truncation_keeps_markers(self, truncation_side):
        processor = self.get_processor()
        processor.tokenizer.truncation_side = truncation_side
        ids = self._apply_text(
            processor,
            ["hello world text"],
            task="query",
            max_length=12,
            truncation=True,
        )["input_ids"][0].tolist()

        word = processor.tokenizer.convert_tokens_to_ids
        kept_word = "hello" if truncation_side == "right" else "text"
        self.assertEqual(ids, [self.marker_ids["<query>"], word(kept_word)] + [self.marker_ids["<mask>"]] * 10)

        # Text that fits is kept whole on either side.
        ids = self._apply_text(processor, ["hello world"], task="query", max_length=13, truncation=True)
        self.assertEqual(
            ids["input_ids"][0].tolist(),
            [self.marker_ids["<query>"], word("hello"), word("world")] + [self.marker_ids["<mask>"]] * 10,
        )

    def test_special_token_strings_in_text_stay_literal(self):
        processor = self.get_processor()
        markers = {self.marker_ids[token] for token in self.special_tokens if token not in ("<pad>", "<unk>")}

        query = self._apply_text(processor, ["hello <mask> <doc> <query> <row> <img>"], task="query")
        query_ids = query["input_ids"][0].tolist()
        self.assertEqual(query_ids[0], self.marker_ids["<query>"])
        self.assertEqual(query_ids[-10:], [self.marker_ids["<mask>"]] * 10)
        self.assertFalse(markers & set(query_ids[1:-10]))

        document = self._apply_text(processor, ["hello <img> <query> <mask>"], task="document")
        document_ids = document["input_ids"][0].tolist()
        self.assertEqual(document_ids[0], self.marker_ids["<doc>"])
        self.assertFalse(markers & set(document_ids[1:]))

    @parameterized.expand([(None,), ("np",), ("pt",)])
    def test_text_image_token_does_not_take_image_expansion(self, return_tensors):
        processor = self.get_processor()
        image = Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))
        mixed = processor.apply_chat_template(
            [
                [{"role": "user", "content": "hello <img>"}],
                [{"role": "user", "content": [{"type": "image", "image": image}]}],
            ],
            task="document",
            tokenize=True,
            return_dict=True,
            return_tensors=return_tensors,
            processor_kwargs={"padding": "longest", "padding_side": "left"},
        )
        image_only = self._apply_images(processor, [image])

        # The image row is the longest, so left padding leaves it untouched.
        self.assertNotIn(self.marker_ids["<img>"], np.asarray(mixed["input_ids"])[0].tolist())
        np.testing.assert_array_equal(np.asarray(mixed["input_ids"])[1], image_only["input_ids"][0].numpy())
        np.testing.assert_array_equal(
            np.asarray(mixed["position_ids"])[:, 1], image_only["position_ids"][:, 0].numpy()
        )
        np.testing.assert_allclose(np.asarray(mixed["pixel_values"]), image_only["pixel_values"].numpy())

    def test_query_truncation_with_no_room_for_text(self):
        processor = self.get_processor()
        processor.tokenizer.truncation_side = "left"
        ids = self._apply_text(processor, ["hello world"], task="query", max_length=11, truncation=True)
        self.assertEqual(ids["input_ids"][0].tolist(), [self.marker_ids["<query>"]] + [self.marker_ids["<mask>"]] * 10)

        with self.assertRaisesRegex(ValueError, "too short"):
            self._apply_text(processor, ["hello"], task="query", max_length=10, truncation=True)

        # A query without text must not lose its expansion tokens to truncation either.
        empty_query = [{"role": "user", "content": []}]
        ids = processor.apply_chat_template(
            empty_query, task="query", tokenize=True, processor_kwargs={"max_length": 11, "truncation": True}
        )
        self.assertEqual(ids[0], [self.marker_ids["<query>"]] + [self.marker_ids["<mask>"]] * 10)
        with self.assertRaisesRegex(ValueError, "too short"):
            processor.apply_chat_template(
                empty_query, task="query", tokenize=True, processor_kwargs={"max_length": 10, "truncation": True}
            )

    def test_max_length_truncates_unless_padding_to_max_length(self):
        processor = self.get_processor()
        texts = ["hello world text"]
        document_length = 1 + len(texts[0].split())  # <doc> + words
        truncated = self._apply_text(
            processor, texts, task="document", max_length=2, padding=False, return_tensors=None
        )["input_ids"][0]
        self.assertEqual(len(truncated), 2)

        padded = self._apply_text(processor, texts, task="document", max_length=2, padding="max_length")["input_ids"]
        self.assertEqual(padded.shape[1], document_length)

    def test_text_is_joined_across_messages_and_items(self):
        processor = self.get_processor()
        word = processor.tokenizer.convert_tokens_to_ids

        def encode(conversation):
            return processor.apply_chat_template(
                conversation, task="document", tokenize=True, return_dict=True, return_tensors="pt"
            )["input_ids"][0].tolist()

        self.assertEqual(
            encode([{"role": "system", "content": "hello "}, {"role": "user", "content": "world"}]),
            [self.marker_ids["<doc>"], word("hello"), word("world")],
        )
        # Texts are joined with nothing in between, so a word split across messages stays one word.
        self.assertEqual(
            encode([{"role": "system", "content": "hel"}, {"role": "user", "content": "lo"}]),
            [self.marker_ids["<doc>"], word("hello")],
        )
        split_marker = [
            {"role": "user", "content": [{"type": "text", "text": "a <im"}, {"type": "text", "text": "g> a"}]}
        ]
        self.assertNotIn(self.marker_ids["<img>"], encode(split_marker))
        self.assertEqual(encode([{"role": "user", "content": []}]), [self.marker_ids["<doc>"]])

    def test_left_padding_without_attention_mask(self):
        processor = self.get_processor()
        ids = self._apply_text(processor, ["hello", "hello world"], padding_side="left", return_attention_mask=False)
        self.assertEqual(ids["input_ids"][0, :2].tolist(), [self.marker_ids["<pad>"], self.marker_ids["<query>"]])

    def test_retrieval_placeholder_is_unk_token(self):
        processor = self.get_processor()
        # Unknown words in the test vocabulary also map to `<unk>`, so check the placeholder string directly.
        conversation, texts = processor._replace_texts_with_placeholder([{"role": "user", "content": "hello"}])
        self.assertEqual(conversation[0]["content"], processor.tokenizer.unk_token)
        self.assertEqual(texts, ["hello"])

        # The placeholder does not need a pad token.
        processor.tokenizer.pad_token = None
        ids = processor.apply_chat_template([{"role": "user", "content": "hello"}], task="document", tokenize=True)
        self.assertEqual(ids, [[self.marker_ids["<doc>"], processor.tokenizer.convert_tokens_to_ids("hello")]])

    def test_retrieval_accepts_tuple_batch(self):
        processor = self.get_processor()
        conversations = [[{"role": "user", "content": "hello"}], [{"role": "user", "content": "world"}]]
        as_list = processor.apply_chat_template(conversations, task="document", tokenize=True)
        as_tuple = processor.apply_chat_template(tuple(conversations), task="document", tokenize=True)
        self.assertEqual(as_tuple, as_list)

    def test_retrieval_requires_unk_token(self):
        processor = self.get_processor()
        processor.tokenizer.unk_token = None
        with self.assertRaisesRegex(ValueError, "unk_token"):
            self._apply_text(processor, ["hello"], task="document")

    def test_retrieval_rejects_per_token_outputs(self):
        processor = self.get_processor()
        with self.assertRaisesRegex(ValueError, "return_offsets_mapping"):
            self._apply_text(processor, ["hello"], task="document", return_offsets_mapping=True)

    def test_nested_and_flat_processor_kwargs_match(self):
        """sentence-transformers passes processing options nested per modality."""
        processor = self.get_processor()
        inputs = processor.apply_chat_template(
            [[{"role": "user", "content": "hello world"}]],
            task="query",
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={
                "text_kwargs": {"padding": "max_length", "truncation": True, "max_length": 32},
                "images_kwargs": {},
                "common_kwargs": {},
            },
        )
        self.assertEqual(inputs["input_ids"].shape, (1, 32))
        self.assertEqual(int(inputs["attention_mask"][0].sum()), 1 + 2 + 10)

        # Flat kwargs must not drop the user text.
        flat = processor.apply_chat_template(
            [[{"role": "user", "content": "hello world"}]],
            task="query",
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
        )
        torch.testing.assert_close(flat["input_ids"], inputs["input_ids"])

    def test_document_marker(self):
        processor = self.get_processor()
        batch = self._apply_text(processor, ["hello world", ""], task="document")
        first = batch["input_ids"][0].tolist()

        self.assertEqual(first[0], self.marker_ids["<doc>"])
        self.assertNotIn(self.marker_ids["<mask>"], first)
        self.assertEqual(int(batch["attention_mask"][1].sum()), 1)

    def test_document_truncation(self):
        processor = self.get_processor()
        ids = self._apply_text(
            processor,
            ["hello world text"],
            task="document",
            max_length=2,
            truncation=True,
        )["input_ids"][0].tolist()
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], self.marker_ids["<doc>"])
        self.assertNotIn(self.marker_ids["<mask>"], ids)

    def test_document_truncation_uses_tokenizer_limit(self):
        processor = self.get_processor()
        processor.tokenizer.model_max_length = 5
        ids = self._apply_text(
            processor,
            ["hello world a document query text"],
            task="document",
            truncation=True,
        )["input_ids"][0].tolist()

        self.assertEqual(len(ids), processor.tokenizer.model_max_length)
        self.assertEqual(ids[0], self.marker_ids["<doc>"])

    def test_generic_processing_does_not_require_retrieval_template(self):
        processor = self.processor_class(**self.prepare_components())
        self.assertIsNone(processor.chat_template)

        text_batch = processor(text=["hello world"], return_tensors="pt")
        text_ids = text_batch["input_ids"][0].tolist()
        self.assertEqual(
            text_ids,
            processor.tokenizer("hello world", add_special_tokens=False)["input_ids"],
        )
        self.assertFalse(
            {
                self.marker_ids["<query>"],
                self.marker_ids["<doc>"],
                self.marker_ids["<mask>"],
            }
            & set(text_ids)
        )

        image = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        image_batch = processor(images=[image], padding="longest", return_tensors="pt")
        self.assertEqual(image_batch["input_ids"][0, 0], self.marker_ids["<doc>"])
        self.assertIn("pixel_values", image_batch)

        with self.assertRaisesRegex(ValueError, "does not have a chat template"):
            processor.apply_chat_template([{"role": "user", "content": "hello"}], task="document")

    def test_apply_chat_template_forwards_image_kwargs(self):
        processor = self.get_processor()
        image = Image.fromarray(np.random.randint(0, 255, (8, 12, 3), dtype=np.uint8))
        inputs = self._apply_images(processor, [image], max_side=4)
        direct = processor(images=[image], max_side=4, return_tensors="pt")
        torch.testing.assert_close(inputs["input_ids"], direct["input_ids"])
        torch.testing.assert_close(inputs["pixel_values"], direct["pixel_values"])

    def test_process_images_uses_standard_hook(self):
        processor = self.get_processor()
        image = Image.fromarray(np.random.randint(0, 255, (8, 12, 3), dtype=np.uint8))

        image_inputs, replacements = processor._process_images([image], return_tensors="pt")

        self.assertSetEqual(set(image_inputs), {"pixel_values", "image_grid_hw"})
        self.assertEqual(len(replacements), 1)
        grid_height, grid_width = image_inputs["image_grid_hw"][0].tolist()
        row = processor.image_token * grid_width + processor.tokenizer.row_token
        self.assertEqual(replacements[0], processor.image_token + row * grid_height)

    def test_image_layout(self):
        processor = self.get_processor()
        grid_height, grid_width = 2, 3
        patch_size = self.patch_size
        image = Image.fromarray(
            np.random.randint(0, 255, (grid_height * patch_size, grid_width * patch_size, 3), dtype=np.uint8)
        )
        batch = self._apply_images(processor, [image])
        ids = batch["input_ids"][0].tolist()
        positions = batch["position_ids"][:, 0]

        expected = [self.marker_ids["<doc>"], self.marker_ids["<img>"]]
        for _ in range(grid_height):
            expected += [self.marker_ids["<img>"]] * grid_width + [self.marker_ids["<row>"]]
        self.assertEqual(ids, expected)
        self.assertEqual(batch["pixel_values"].shape, (grid_height * grid_width, 3 * patch_size**2))
        self.assertNotIn("image_grid_hw", batch)

        # The document and image markers precede the grid at (2, 2).
        self.assertEqual(positions[:, 0].tolist(), [0, 0])
        self.assertEqual(positions[:, 1].tolist(), [1, 1])
        self.assertEqual(positions[:, 2].tolist(), [2, 2])
        self.assertEqual(positions[:, 2 + grid_width].tolist(), [2, 2 + grid_width])
        self.assertEqual(positions[:, 2 + grid_width + 1].tolist(), [3, 2])

        with self.assertRaisesRegex(ValueError, "require `return_attention_mask=True`"):
            self._apply_images(processor, [image], return_attention_mask=False)

        second_image = Image.fromarray(np.random.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "require padding"):
            self._apply_images(
                processor,
                [image, second_image],
                padding=False,
                return_tensors=None,
            )

    def test_per_image_position_ids(self):
        processor = self.get_processor()
        patch_size = self.patch_size
        images = [
            Image.fromarray(np.random.randint(0, 255, (2 * patch_size, 3 * patch_size, 3), dtype=np.uint8)),
            Image.fromarray(np.random.randint(0, 255, (patch_size, patch_size, 3), dtype=np.uint8)),
        ]
        batch = self._apply_images(processor, images)

        # Each image's positions restart instead of continuing across the batch.
        self.assertEqual(batch["position_ids"][:, 1, 0].tolist(), [0, 0])
        self.assertEqual(batch["position_ids"][:, 1, 1].tolist(), [1, 1])
        self.assertEqual(batch["pixel_values"].shape[0], 2 * 3 + 1)
        self.assertEqual(int(batch["attention_mask"][1].sum()), 2 + 1 * (1 + 1))
