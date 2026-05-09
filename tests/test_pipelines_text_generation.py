"""
Regression test for:
https://github.com/huggingface/transformers/issues/45854

Bug: pipeline("text-generation") ignores return_full_text=False when
     input is a chat (list of dicts) processed via a chat template.

Fix: postprocess() now slices by token position (input_ids.shape[-1])
     instead of character length of the decoded prompt string.
"""

import unittest

from transformers import AutoTokenizer, pipeline


class TestReturnFullTextChatTemplate(unittest.TestCase):
    """
    Uses sshleifer/tiny-gpt2 (tiny, no download required beyond CI cache)
    with a manually injected chat template to reproduce the bug without
    needing a large instruction-tuned model.
    """

    # Minimal Jinja chat template — mirrors what real instruct models use
    CHAT_TEMPLATE = (
        "{% for message in messages %}"
        "{{ '<|' + message['role'] + '|>\\n' + message['content'] + '\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|assistant|>\\n' }}"
        "{% endif %}"
    )

    MODEL = "sshleifer/tiny-gpt2"

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)
        self.tokenizer.chat_template = self.CHAT_TEMPLATE
        # tiny-gpt2 has no pad token by default
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.pipe = pipeline(
            "text-generation",
            model=self.MODEL,
            tokenizer=self.tokenizer,
        )

    # ------------------------------------------------------------------
    # Core regression: return_full_text=False with chat input
    # ------------------------------------------------------------------

    def test_return_full_text_false_with_chat_template(self):
        """
        return_full_text=False must return ONLY generated tokens,
        not the formatted chat prompt.
        """
        messages = [{"role": "user", "content": "What is 2+2?"}]
        result = self.pipe(messages, return_full_text=False, max_new_tokens=10)

        self.assertIsInstance(result, list)
        generated = result[0]["generated_text"]

        # With return_full_text=False the output must be a string (new text only),
        # NOT a list of message dicts (which would mean the prompt was included).
        self.assertIsInstance(
            generated,
            str,
            "return_full_text=False should return a plain string, not a chat list",
        )

        # The prompt content must NOT appear in the output
        self.assertNotIn(
            "What is 2+2?",
            generated,
            "return_full_text=False must not include the original prompt text",
        )

        # The chat template role tokens must NOT appear either
        self.assertNotIn("<|user|>", generated)
        self.assertNotIn("<|assistant|>", generated)

    # ------------------------------------------------------------------
    # Sanity: return_full_text=True (default) still works with chat input
    # ------------------------------------------------------------------

    def test_return_full_text_true_with_chat_template(self):
        """
        Default behaviour (return_full_text=True) must return the full
        chat list including the original messages + assistant reply.
        """
        messages = [{"role": "user", "content": "What is 2+2?"}]
        result = self.pipe(messages, return_full_text=True, max_new_tokens=10)

        generated = result[0]["generated_text"]

        # Full-text mode with a chat returns a list of message dicts
        self.assertIsInstance(generated, list)
        roles = [m["role"] for m in generated]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    # ------------------------------------------------------------------
    # Sanity: plain string input still works correctly
    # ------------------------------------------------------------------

    def test_return_full_text_false_plain_string(self):
        """
        return_full_text=False must still work for plain string inputs
        (non-chat) — ensure the fix didn't regress this path.
        """
        result = self.pipe(
            "The capital of France is",
            return_full_text=False,
            max_new_tokens=10,
        )
        generated = result[0]["generated_text"]

        self.assertIsInstance(generated, str)
        self.assertNotIn("The capital of France is", generated)

    # ------------------------------------------------------------------
    # Edge case: empty assistant prefill with return_full_text=False
    # ------------------------------------------------------------------

    def test_return_full_text_false_with_assistant_prefill(self):
        """
        When the chat ends with a partial assistant message (prefill),
        return_full_text=False should still return only the new tokens.
        """
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is"},
        ]
        result = self.pipe(messages, return_full_text=False, max_new_tokens=10)
        generated = result[0]["generated_text"]

        self.assertIsInstance(generated, str)
        self.assertNotIn("What is 2+2?", generated)


if __name__ == "__main__":
    unittest.main()
