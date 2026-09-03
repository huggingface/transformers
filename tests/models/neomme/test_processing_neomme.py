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
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
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
            image_processor=NeoMMEImageProcessor(patch_size=cls.patch_size),
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

    def _set_retrieval_chat_template(self, processor):
        processor.chat_template = self.chat_template

    @staticmethod
    def prepare_processor_dict():
        return {}

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

        self.assertEqual(inputs["input_ids"][0, 0], self.marker_ids["<doc>"])
        self.assertNotIn(self.marker_ids["<mask>"], inputs["input_ids"][0].tolist())

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

    def test_apply_chat_template_supports_mixed_document_batch(self):
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

    def test_tokenizer_defaults_preserved_by_kwargs(self):
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component(
            "tokenizer", max_length=self.image_text_kwargs_max_length, padding="max_length"
        )
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        inputs = self._apply_text(processor, self.prepare_text_inputs(), return_tensors="pt")
        self.assertEqual(inputs[self.text_input_name].shape[-1], self.image_text_kwargs_max_length)

    def test_kwargs_overrides_default_tokenizer_kwargs(self):
        processor_components = self.prepare_components()
        processor_components["tokenizer"] = self.get_component("tokenizer", padding="longest")
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        inputs = self._apply_text(
            processor,
            self.prepare_text_inputs(),
            return_tensors="pt",
            max_length=self.image_text_kwargs_override_max_length,
            padding="max_length",
        )
        self.assertEqual(inputs[self.text_input_name].shape[-1], self.image_text_kwargs_override_max_length)

    def test_unstructured_kwargs(self):
        processor = self.processor_class(**self.prepare_components())
        self.skip_processor_without_typed_kwargs(processor)

        inputs = self._apply_text(
            processor,
            self.prepare_text_inputs(),
            return_tensors="pt",
            padding="max_length",
            max_length=self.image_unstructured_max_length,
        )
        self.assertEqual(inputs[self.text_input_name].shape[-1], self.image_unstructured_max_length)

    def test_structured_kwargs_nested(self):
        processor = self.processor_class(**self.prepare_components())
        self.skip_processor_without_typed_kwargs(processor)

        inputs = self._apply_text(
            processor,
            self.prepare_text_inputs(),
            common_kwargs={"return_tensors": "pt"},
            text_kwargs={"padding": "max_length", "max_length": self.image_unstructured_max_length},
        )
        self.assertEqual(inputs[self.text_input_name].shape[-1], self.image_unstructured_max_length)

    def test_structured_kwargs_nested_from_dict(self):
        processor = self.processor_class(**self.prepare_components())
        self.skip_processor_without_typed_kwargs(processor)

        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "text_kwargs": {"padding": "max_length", "max_length": self.image_unstructured_max_length},
        }
        inputs = self._apply_text(processor, self.prepare_text_inputs(), **all_kwargs)
        self.assertEqual(inputs[self.text_input_name].shape[-1], self.image_unstructured_max_length)

    def test_flat_kwarg_applied_when_modality_dict_lacks_it(self):
        """A flat `return_tensors` still applies when `text_kwargs` omits it (regression #46192)."""
        processor = self.get_processor()
        self.skip_processor_without_typed_kwargs(processor)

        inputs = self._apply_text(
            processor,
            self.prepare_text_inputs(),
            text_kwargs={"padding": "longest"},
            return_tensors="np",
        )
        self.assertIsInstance(inputs[self.text_input_name], np.ndarray)

    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        """A negative mean confirms that `rescale_factor=-1.0` was preserved."""
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=-1.0
        )
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        inputs = self._apply_images(processor, self.prepare_image_inputs(), return_tensors="pt")
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_kwargs_overrides_default_image_processor_kwargs(self):
        processor_components = self.prepare_components()
        processor_components["image_processor"] = self.get_component(
            "image_processor", do_rescale=True, rescale_factor=1
        )
        processor = self.processor_class(**processor_components)
        self.skip_processor_without_typed_kwargs(processor)

        inputs = self._apply_images(
            processor,
            self.prepare_image_inputs(),
            do_rescale=True,
            rescale_factor=-1.0,
            return_tensors="pt",
        )
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_unstructured_kwargs_batched(self):
        processor = self.processor_class(**self.prepare_components())
        self.skip_processor_without_typed_kwargs(processor)

        inputs = self._apply_images(
            processor,
            self.prepare_image_inputs(batch_size=2),
            return_tensors="pt",
            do_rescale=True,
            rescale_factor=-1.0,
        )
        self.assertLessEqual(inputs[self.images_input_name][0][0].mean(), 0)

    def test_doubly_passed_kwargs(self):
        processor = self.processor_class(**self.prepare_components())
        self.skip_processor_without_typed_kwargs(processor)

        image_input = self.prepare_image_inputs()
        with self.assertRaises(ValueError):
            self._apply_images(
                processor,
                image_input,
                images_kwargs={"do_rescale": True, "rescale_factor": -1.0},
                do_rescale=True,
                return_tensors="pt",
            )

    def test_model_input_names(self):
        processor = self.get_processor()
        image_inputs = self._apply_images(processor, self.prepare_image_inputs())
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

    def test_query_marker_and_expansion(self):
        processor = self.get_processor()
        batch = self._apply_text(processor, ["hello world", "a"], task="query")
        first = batch["input_ids"][0].tolist()

        self.assertEqual(first[0], self.marker_ids["<query>"])
        self.assertEqual(first[-10:], [self.marker_ids["<mask>"]] * 10)
        self.assertEqual(len(first), 1 + 2 + 10)
        # The shorter query is right-padded and its padding is masked out.
        self.assertEqual(int(batch["attention_mask"][1].sum()), 1 + 1 + 10)

    def test_query_truncation_rejects_removed_expansion(self):
        processor = self.get_processor()
        with self.assertRaisesRegex(ValueError, "removed NeoMME query expansion tokens"):
            self._apply_text(
                processor,
                ["hello world text"],
                task="query",
                max_length=12,
                truncation=True,
            )

        processor.tokenizer.truncation_side = "left"
        with self.assertRaisesRegex(ValueError, "leading task marker"):
            self._apply_text(
                processor,
                ["hello world text"],
                task="query",
                max_length=12,
                truncation=True,
            )

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
