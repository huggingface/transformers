"""
Framework agnostic tests for generate()-related methods.
"""

import numpy as np

from transformers import AutoTokenizer


class GenerationIntegrationTestsMixin:

    # To be populated by the child classes
    framework_dependent_parameters = {
        "AutoModelForSeq2SeqLM": None,
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
