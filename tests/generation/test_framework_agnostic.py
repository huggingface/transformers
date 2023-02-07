"""
Framework agnostic tests for generate()-related methods.
"""

import numpy as np

from transformers import AutoTokenizer
from transformers.testing_utils import torch_device


class GenerationIntegrationTestsMixin:
    # To be populated by the child classes
    framework_dependent_parameters = {
        "AutoModelForCausalLM": None,
        "AutoModelForSeq2SeqLM": None,
        "LogitsProcessorList": None,
        "MinLengthLogitsProcessor": None,
        "create_tensor_fn": None,
        "return_tensors": None,
    }

    def test_validate_generation_inputs(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        create_tensor_fn = self.framework_dependent_parameters["create_tensor_fn"]

        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-t5")

        encoder_input_str = "Hello world"
        input_ids = tokenizer(encoder_input_str, return_tensors=return_tensors).input_ids

        # typos are quickly detected (the correct argument is `do_sample`)
        with self.assertRaisesRegex(ValueError, "do_samples"):
            model.generate(input_ids, do_samples=True)

        # arbitrary arguments that will not be used anywhere are also not accepted
        with self.assertRaisesRegex(ValueError, "foo"):
            fake_model_kwargs = {"foo": "bar"}
            model.generate(input_ids, **fake_model_kwargs)

        # however, valid model_kwargs are accepted
        valid_model_kwargs = {"attention_mask": create_tensor_fn(np.zeros_like(input_ids))}
        model.generate(input_ids, **valid_model_kwargs)

    def test_custom_logits_processor(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        logits_processor_list_cls = self.framework_dependent_parameters["LogitsProcessorList"]
        min_length_logits_processor_cls = self.framework_dependent_parameters["MinLengthLogitsProcessor"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]

        bart_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_model = model_cls.from_pretrained("hf-internal-testing/tiny-random-bart", min_length=1)
        input_ids = bart_tokenizer(article, return_tensors=return_tensors).input_ids

        logits_processor = logits_processor_list_cls()
        logits_processor.append(min_length_logits_processor_cls(min_length=10, eos_token_id=0))
        # it should not be allowed to both define `min_length` via config and `logits_processor` list
        with self.assertRaises(ValueError):
            bart_model.generate(input_ids, logits_processor=logits_processor)

        bart_model.config.min_length = None
        bart_model.generate(input_ids, logits_processor=logits_processor)

    def test_max_new_tokens_encoder_decoder(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        bart_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")

        bart_model = model_cls.from_pretrained("hf-internal-testing/tiny-random-bart")
        input_ids = bart_tokenizer(article, return_tensors=return_tensors).input_ids
        if is_pt:
            bart_model = bart_model.to(torch_device)
            input_ids = input_ids.to(torch_device)

        self.assertEqual(list(input_ids.shape), [1, 29])

        max_new_tokens = 3
        bart_model.config.max_length = 20
        bart_model.config.eos_token_id = None

        # Encoder decoder call
        outputs = bart_model.generate(input_ids, max_new_tokens=max_new_tokens)
        # 1 BOS + 3 new tokens
        self.assertEqual(list(outputs.shape), [1, 4])

        # Decoder only call
        outputs = bart_model.generate(decoder_input_ids=input_ids, max_new_tokens=max_new_tokens)
        # 29 + 3 new tokens
        self.assertEqual(list(outputs.shape), [1, 32])

        # Encoder decoder call > 20
        outputs = bart_model.generate(max_new_tokens=max_new_tokens + 20)

        # 1 BOS + 20 + 3 new tokens
        self.assertEqual(list(outputs.shape), [1, 24])

    def test_max_new_tokens_decoder_only(self):
        model_cls = self.framework_dependent_parameters["AutoModelForCausalLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        article = """Justin Timberlake."""
        gpt2_tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        gpt2_model = model_cls.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        input_ids = gpt2_tokenizer(article, return_tensors=return_tensors).input_ids
        if is_pt:
            gpt2_model = gpt2_model.to(torch_device)
            input_ids = input_ids.to(torch_device)

        self.assertEqual(list(input_ids.shape), [1, 9])

        max_new_tokens = 3
        gpt2_model.config.max_length = 20

        # call < 20
        outputs = gpt2_model.generate(input_ids, max_new_tokens=max_new_tokens)

        # 9 input_ids + 3 new tokens
        self.assertEqual(list(outputs.shape), [1, 12])

        # call > 20
        outputs = gpt2_model.generate(max_new_tokens=max_new_tokens + 20)

        # 1 BOS token + 23 new tokens
        self.assertEqual(list(outputs.shape), [1, 24])

    def test_encoder_decoder_generate_with_inputs_embeds(self):
        model_cls = self.framework_dependent_parameters["AutoModelForSeq2SeqLM"]
        return_tensors = self.framework_dependent_parameters["return_tensors"]
        is_pt = not model_cls.__name__.startswith("TF")

        article = """Justin Timberlake and Jessica Biel, welcome to parenthood."""
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart")
        model = model_cls.from_pretrained("hf-internal-testing/tiny-random-bart", max_length=5)
        model.config.eos_token_id = None
        input_ids = tokenizer(article, return_tensors=return_tensors).input_ids
        if is_pt:
            model = model.to(torch_device)
            input_ids = input_ids.to(torch_device)

        inputs_embeds = model.get_input_embeddings()(input_ids)

        output_sequences = model.generate(inputs_embeds=inputs_embeds)

        # make sure model generated correctly until `max_length`
        self.assertEqual(output_sequences.shape, (1, 5))
