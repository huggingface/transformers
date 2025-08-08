import time
import unittest

from parameterized import parameterized

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.testing_utils import require_flash_attn, require_torch_gpu, slow


_TEST_PROMPTS = [
    "A man is a walking his dog down the street, and a the turn he sees",
    "Describe a fruit that is of orange color and round. It is a sweet fruit and a great source of Vitamine C. The fruit I'm thinking of is an",
    "A plane is flying high in the sky, out of the window are clouds and mountains. Where could the plane be located?",
    "Please fill in the form to",
    "For safety reasons, the train is stopped in the middle of the",
]

_EXPECTED_OUTPUTS = [
    "a woman standing on the sidewalk, looking at him. He is immediately drawn to her and feels a strong attraction. He walks up to her and strikes up a conversation, and they quickly discover that they have a lot in common. They exchange numbers and",
    "orange.\n\n## Step 1: Identify the key characteristics of the fruit\nThe fruit is described as being orange in color and round in shape.\n\n## Step 2: Determine the taste and nutritional value of the fruit\nThe fruit is described as sweet",
    "This riddle is a classic example of a lateral thinking puzzle, which requires the test-taker to think creatively and consider multiple possibilities. The answer is not a straightforward one, and it requires some lateral thinking to arrive at the correct solution.",
    "get in touch with us. We will respond to your message as soon as possible.\n\n[Your Name]\n[Your Email]\n[Your Phone Number]\n[Your Message]\n\nWe are looking forward to hearing from you!\n\n[Insert Contact Information]\n\nNote:",
    "track. The train is stopped for 30 minutes. The train is moving at a speed of 60 km/h. How many kilometers does the train travel in 30 minutes?\n## Step 1: Convert the speed from km/h to km/min",
]


@slow
@require_flash_attn
@require_torch_gpu
class TestBatchGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3b-Instruct", dtype="bfloat16", device_map="auto"
        ).eval()

        cls.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3b-Instruct", padding_side="left")

        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
            cls.model.config.pad_token_id = cls.model.config.eos_token_id

        cls.model.use_cache = False

    @parameterized.expand(
        [
            ("eager_paged", 64, 128, 64),
            ("sdpa_paged", 32, 256, 128),
            ("paged_attention", 16, 512, 256),
            ("flex_paged", 64, 128, 64),
        ]
    )
    def test_generate_batch_consistency(self, attn_impl, num_blocks, block_size, max_batch_tokens):
        self.model.config.attn_implementation = attn_impl

        generation_config = GenerationConfig(
            max_new_tokens=50,
            top_k=0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False,
            num_blocks=num_blocks,
            block_size=block_size,
            max_batch_tokens=max_batch_tokens,
        )

        tokenized = self.tokenizer(_TEST_PROMPTS, truncation=True, max_length=512)
        batch_inputs = list(tokenized["input_ids"])

        start = time.time()
        batch_outputs = self.model.generate_batch(
            inputs=batch_inputs,
            generation_config=generation_config,
        )
        end = time.time()
        print(
            f"\n[{attn_impl}] Batch took {end - start:.2f}s with config: blocks={num_blocks}, block_size={block_size}, max_batch_tokens={max_batch_tokens}"
        )

        for i, req_id in enumerate(batch_outputs):
            generated = self.tokenizer.decode(batch_outputs[req_id].static_outputs, skip_special_tokens=False).strip()
            expected = _EXPECTED_OUTPUTS[i].strip()
            self.assertTrue(
                generated.startswith(expected),
                msg=f"[{attn_impl}] Mismatch in request {i}:\nExpected start: {expected}\nGot: {generated}",
            )
