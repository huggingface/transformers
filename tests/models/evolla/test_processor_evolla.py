# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import shutil
import tempfile
import unittest
import random

import numpy as np

from transformers import (
    AutoProcessor,
    EsmTokenizer,
    EvollaProcessor,
    LlamaTokenizerFast,
    PreTrainedTokenizerFast,
    AutoTokenizer
)
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


EVOLLA_VALID_AA = list("ACDEFGHIKLMNPQRSTVWY#")
EVOLLA_VALID_FS = list("pynwrqhgdlvtmfsaeikc#")

@require_torch
class EvollaProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = EvollaProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        protein_tokenizer = EsmTokenizer.from_pretrained("/zhouxibin/workspaces/ProteinQA/Models/SaProt_35M_AF2")
        tokenizer = LlamaTokenizerFast.from_pretrained("/zhouxibin/workspaces/ProteinQA/Models/meta-llama_Meta-Llama-3-8B-Instruct")
        print(type(tokenizer))

        processor = EvollaProcessor(protein_tokenizer, tokenizer)

        processor.save_pretrained(self.tmpdirname)

        self.input_keys = ["protein_input_ids", "protein_attention_mask", "text_input_ids", "text_attention_mask"]

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_protein_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).protein_tokenizer

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_inputs_single(self):
        proteins = {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=100)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=100)),
            }
        return proteins

    def prepare_inputs_pair(self):
        proteins = [
            {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=100)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=100)),
            },
            {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=100)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=100)),
            },
        ]
        return proteins

    def prepare_inputs_long(self):
        proteins = [
            {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=100)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=100)),
            },
            {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=2000)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=2000)),
            },
        ]
        return proteins

    def prepare_inputs_short(self):
        proteins = [
            {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=1)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=1)),
            },
            {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=100)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=100)),
            },
        ]
        return proteins

    def prepare_inputs_empty(self):
        proteins = [
            {
                "aa_seq": "",
                "foldseek": "",
            },
            {
                "aa_seq": "".join(random.choices(EVOLLA_VALID_AA, k=100)),
                "foldseek": "".join(random.choices(EVOLLA_VALID_FS, k=100)),
            },
        ]
        return proteins

    def prepare_inputs(self, protein_types="pair"):
        r"""
        Prepare inputs for the test.

        Args:
            protein_types (`str`): the types of proteins to prepare.
                - "single": a single correct protein.
                - "pair": a pair of correct proteins.
                - "long": a long sequence of correct proteins and a correct protein.
                - "short": a short sequence of correct proteins (only have 1 aa) and a correct protein.
                - "empty": an empty sequence of proteins and a correct protein.
        """
        if protein_types == "single":
            proteins = self.prepare_inputs_single()
        elif protein_types == "pair":
            proteins = self.prepare_inputs_pair()
        elif protein_types == "long":
            proteins = self.prepare_inputs_long()
        elif protein_types == "short":
            proteins = self.prepare_inputs_short()
        elif protein_types == "empty":
            proteins = self.prepare_inputs_empty()
        else:
            raise ValueError(f"protein_types should be one of 'single', 'pair', 'long','short', 'empty', but got {protein_types}")
        
        questions = ["What is the function of the protein?"] * len(proteins)
        messages_list = []
        for question in questions:
            messages = [
                {"role": "system", "content": "You are an AI expert that can answer any questions about protein."},
                {"role": "user", "content": question},
            ]
            messages_list.append(messages)
        return proteins, messages_list

    def test_processor(self):
        protein_tokenizer = self.get_protein_tokenizer()
        tokenizer = self.get_tokenizer()

        processor = EvollaProcessor(tokenizer=tokenizer, protein_tokenizer=protein_tokenizer)

        proteins, messages_list = self.prepare_inputs()

        # test that all prompts succeeded
        input_processor = processor(messages_list=messages_list, proteins=proteins, return_tensors="pt", padding="longest")
        for key in self.input_keys:
            assert torch.is_tensor(input_processor[key])

    def test_tokenizer_decode(self):
        protein_tokenizer = self.get_protein_tokenizer()
        tokenizer = self.get_tokenizer()

        processor = EvollaProcessor(tokenizer=tokenizer, protein_tokenizer=protein_tokenizer, return_tensors="pt")

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_model_input_names(self):
        protein_tokenizer = self.get_protein_tokenizer()
        tokenizer = self.get_tokenizer()

        processor = EvollaProcessor(tokenizer=tokenizer, protein_tokenizer=protein_tokenizer)
        proteins, messages_list = self.prepare_inputs()

        inputs = processor(messages_list=messages_list, proteins=proteins, padding="longest", return_tensors="pt")

        # For now the processor supports only ['pixel_values', 'input_ids', 'attention_mask']
        self.assertSetEqual(set(inputs.keys()), set(self.input_keys))

@require_torch
@require_vision
class EvollaProcessorTest2(ProcessorTesterMixin, unittest.TestCase):
    processor_class = EvollaProcessor

    def setUp(self):
        self.tmpdirname = tempfile.mkdtemp()

        image_processor = EvollaProteinProcessor(return_tensors="pt")
        tokenizer = LlamaTokenizerFast.from_pretrained("/zhouxibin/workspaces/ProteinQA/Models/meta-llama_Meta-Llama-3-8B-Instruct")

        processor = EvollaProcessor(image_processor, tokenizer)

        processor.save_pretrained(self.tmpdirname)

        self.input_keys = ["pixel_values", "input_ids", "attention_mask", "image_attention_mask"]

    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    def get_image_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).image_processor

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def prepare_prompts(self):
        """This function prepares a list of PIL images"""

        num_images = 2
        images = [np.random.randint(255, size=(3, 30, 400), dtype=np.uint8) for x in range(num_images)]
        images = [Image.fromarray(np.moveaxis(x, 0, -1)) for x in images]

        # print([type(x) for x in images])
        # die

        prompts = [
            # text and 1 image
            [
                "User:",
                images[0],
                "Describe this image.\nAssistant:",
            ],
            # text and images
            [
                "User:",
                images[0],
                "Describe this image.\nAssistant: An image of two dogs.\n",
                "User:",
                images[1],
                "Describe this image.\nAssistant:",
            ],
            # only text
            [
                "User:",
                "Describe this image.\nAssistant: An image of two kittens.\n",
                "User:",
                "Describe this image.\nAssistant:",
            ],
            # only images
            [
                images[0],
                images[1],
            ],
        ]

        return prompts

    def test_save_load_pretrained_additional_features(self):
        processor = EvollaProcessor(tokenizer=self.get_tokenizer(), image_processor=self.get_image_processor())
        processor.save_pretrained(self.tmpdirname)

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")
        image_processor_add_kwargs = self.get_image_processor(do_normalize=False, padding_value=1.0)

        processor = EvollaProcessor.from_pretrained(
            self.tmpdirname, bos_token="(BOS)", eos_token="(EOS)", do_normalize=False, padding_value=1.0
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())
        self.assertIsInstance(processor.tokenizer, PreTrainedTokenizerFast)

        self.assertEqual(processor.image_processor.to_json_string(), image_processor_add_kwargs.to_json_string())
        self.assertIsInstance(processor.image_processor, EvollaProteinProcessor)

    def test_processor(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = EvollaProcessor(tokenizer=tokenizer, image_processor=image_processor)

        prompts = self.prepare_prompts()

        # test that all prompts succeeded
        input_processor = processor(text=prompts, return_tensors="pt", padding="longest")
        for key in self.input_keys:
            assert torch.is_tensor(input_processor[key])

    def test_tokenizer_decode(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = EvollaProcessor(tokenizer=tokenizer, image_processor=image_processor, return_tensors="pt")

        predicted_ids = [[1, 4, 5, 8, 1, 0, 8], [3, 4, 3, 1, 1, 8, 9]]

        decoded_processor = processor.batch_decode(predicted_ids)
        decoded_tok = tokenizer.batch_decode(predicted_ids)

        self.assertListEqual(decoded_tok, decoded_processor)

    def test_tokenizer_padding(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer(padding_side="right")

        processor = EvollaProcessor(tokenizer=tokenizer, image_processor=image_processor, return_tensors="pt")

        predicted_tokens = [
            "<s> Describe this image.\nAssistant:<unk><unk><unk><unk><unk><unk><unk><unk><unk>",
            "<s> Describe this image.\nAssistant:<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk>",
        ]
        predicted_attention_masks = [
            ([1] * 10) + ([0] * 9),
            ([1] * 10) + ([0] * 10),
        ]
        prompts = [[prompt] for prompt in self.prepare_prompts()[2]]

        max_length = processor(text=prompts, padding="max_length", truncation=True, max_length=20, return_tensors="pt")
        longest = processor(text=prompts, padding="longest", truncation=True, max_length=30, return_tensors="pt")

        decoded_max_length = processor.tokenizer.decode(max_length["input_ids"][-1])
        decoded_longest = processor.tokenizer.decode(longest["input_ids"][-1])

        self.assertEqual(decoded_max_length, predicted_tokens[1])
        self.assertEqual(decoded_longest, predicted_tokens[0])

        self.assertListEqual(max_length["attention_mask"][-1].tolist(), predicted_attention_masks[1])
        self.assertListEqual(longest["attention_mask"][-1].tolist(), predicted_attention_masks[0])

    def test_tokenizer_left_padding(self):
        """Identical to test_tokenizer_padding, but with padding_side not explicitly set."""
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = EvollaProcessor(tokenizer=tokenizer, image_processor=image_processor)

        predicted_tokens = [
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><s> Describe this image.\nAssistant:",
            "<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><s> Describe this image.\nAssistant:",
        ]
        predicted_attention_masks = [
            ([0] * 9) + ([1] * 10),
            ([0] * 10) + ([1] * 10),
        ]
        prompts = [[prompt] for prompt in self.prepare_prompts()[2]]
        max_length = processor(text=prompts, padding="max_length", truncation=True, max_length=20)
        longest = processor(text=prompts, padding="longest", truncation=True, max_length=30)

        decoded_max_length = processor.tokenizer.decode(max_length["input_ids"][-1])
        decoded_longest = processor.tokenizer.decode(longest["input_ids"][-1])

        self.assertEqual(decoded_max_length, predicted_tokens[1])
        self.assertEqual(decoded_longest, predicted_tokens[0])

        self.assertListEqual(max_length["attention_mask"][-1].tolist(), predicted_attention_masks[1])
        self.assertListEqual(longest["attention_mask"][-1].tolist(), predicted_attention_masks[0])

    def test_model_input_names(self):
        image_processor = self.get_image_processor()
        tokenizer = self.get_tokenizer()

        processor = EvollaProcessor(tokenizer=tokenizer, image_processor=image_processor)
        prompts = self.prepare_prompts()

        inputs = processor(text=prompts, padding="longest", return_tensors="pt")

        # For now the processor supports only ['pixel_values', 'input_ids', 'attention_mask']
        self.assertSetEqual(set(inputs.keys()), set(self.input_keys))

    # Override the following tests as Evolla image processor does not accept do_rescale and rescale_factor
    @require_torch
    @require_vision
    def test_image_processor_defaults_preserved_by_image_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", image_size=234)
        tokenizer = self.get_component("tokenizer", max_length=117)

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input)
        self.assertEqual(len(inputs["pixel_values"][0][0][0]), 234)

    @require_torch
    @require_vision
    def test_kwargs_overrides_default_image_processor_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor", image_size=234)
        tokenizer = self.get_component("tokenizer", max_length=117)

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        inputs = processor(text=input_str, images=image_input, image_size=224)
        self.assertEqual(len(inputs["pixel_values"][0][0][0]), 224)

    @require_torch
    @require_vision
    def test_unstructured_kwargs(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            image_size=214,
            padding="max_length",
            max_length=76,
        )

        self.assertEqual(inputs["pixel_values"].shape[3], 214)
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_unstructured_kwargs_batched(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs(batch_size=2)
        image_input = self.prepare_image_inputs(batch_size=2)
        inputs = processor(
            text=input_str,
            images=image_input,
            return_tensors="pt",
            image_size=214,
            padding="longest",
            max_length=76,
        )

        self.assertEqual(inputs["pixel_values"].shape[3], 214)
        self.assertEqual(len(inputs["input_ids"][0]), 8)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")
        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)

        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"image_size": 214},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.skip_processor_without_typed_kwargs(processor)
        self.assertEqual(inputs["pixel_values"].shape[3], 214)
        self.assertEqual(len(inputs["input_ids"][0]), 76)

    @require_torch
    @require_vision
    def test_structured_kwargs_nested_from_dict(self):
        if "image_processor" not in self.processor_class.attributes:
            self.skipTest(f"image_processor attribute not present in {self.processor_class}")

        image_processor = self.get_component("image_processor")
        tokenizer = self.get_component("tokenizer")

        processor = self.processor_class(tokenizer=tokenizer, image_processor=image_processor)
        self.skip_processor_without_typed_kwargs(processor)
        input_str = self.prepare_text_inputs()
        image_input = self.prepare_image_inputs()

        # Define the kwargs for each modality
        all_kwargs = {
            "common_kwargs": {"return_tensors": "pt"},
            "images_kwargs": {"image_size": 214},
            "text_kwargs": {"padding": "max_length", "max_length": 76},
        }

        inputs = processor(text=input_str, images=image_input, **all_kwargs)
        self.assertEqual(inputs["pixel_values"].shape[3], 214)
        self.assertEqual(len(inputs["input_ids"][0]), 76)
