import unittest

import torch.nn.functional as F
from parameterized import parameterized

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)


@require_torch
class ContrastiveSearchTest(unittest.TestCase):
    @parameterized.expand(
        [
            (False, 4, 0.6),
            (True, 6, 0.4),
        ]
    )
    @slow
    def test_batch_contrastive_search_gpt2(self, low_memory, top_k, penalty_alpha):
        # Load the pre-trained GPT-2 model and tokenizer
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        model.to(torch_device)
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2", clean_up_tokenization_spaces=True)

        # Set the tokenizer to left-pad the sequences
        tokenizer.padding_side = "left"

        # Define the PAD token as the EOS token
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        # Configure the model's low_memory setting for generation
        model.generation_config.low_memory = low_memory

        # Define the input prompt
        prompt_text = "The whispered legends of the haunted mansion spoke"

        # Tokenize the input prompt
        encoded_prompt = tokenizer(prompt_text, return_tensors="pt", padding=True)
        input_ids = encoded_prompt.input_ids.to(torch_device)
        attention_mask = encoded_prompt.attention_mask.to(torch_device)

        # Define the padding length to add to the input IDs and attention mask
        padding_length = 10

        # Generate text without padding
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            penalty_alpha=penalty_alpha,
            top_k=top_k,
            max_length=128,
        )
        outputs = outputs[:, :-padding_length]
        generated_text_no_padding = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Pad the input IDs and attention mask on the left
        padded_input_ids = F.pad(
            input_ids, (padding_length, 0), "constant", value=model.generation_config.pad_token_id
        )
        padded_attention_mask = F.pad(attention_mask, (padding_length, 0), "constant", value=0)

        # Generate text with padded inputs
        outputs_with_padding = model.generate(
            input_ids=padded_input_ids,
            attention_mask=padded_attention_mask,
            do_sample=False,
            penalty_alpha=penalty_alpha,
            top_k=top_k,
            max_length=128,
        )
        outputs_with_padding = outputs_with_padding[:, padding_length:]
        generated_text_with_padding = tokenizer.batch_decode(outputs_with_padding, skip_special_tokens=True)

        # Assert that the generated texts are identical for padded and non-padded inputs
        self.assertListEqual(generated_text_no_padding, generated_text_with_padding)
