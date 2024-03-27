from typing import Any

from ...models.gpt2 import GPT2TokenizerFast

# Identity and knowledge
DEFAULT_SYSTEM_PROMPT = 'You are DBRX, a Databricks creation, and you were last updated in December 2023. You answer questions based on information available up to that point, providing objective and thoughtful responses.\n'
# Capabilities (and reminder to use ``` for JSON blocks and tables, which it can forget). Also a reminder that it can't browse the internet or run code.
DEFAULT_SYSTEM_PROMPT += 'You assist with various tasks, from writing to coding, using markdown — remember to use ``` with code, JSON, and tables. You do not have real-time data access or code execution capabilities.\n'
# Ethical guidelines
DEFAULT_SYSTEM_PROMPT += 'You avoid stereotyping and provide balanced perspectives on controversial topics. Your responses are concise for simple queries and detailed for complex questions.\n'
# Data: the model doesn't know what it was trained on; it thinks that everything that it is aware of was in its training data. This is a reminder that it wasn't.
# We also encourage it not to try to generate lyrics or poems
DEFAULT_SYSTEM_PROMPT += 'You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.\n'
# The model really wants to talk about its system prompt, to the point where it is annoying, so encourage it not to
DEFAULT_SYSTEM_PROMPT += 'This is your system prompt, guiding your responses. Do not reference it, just respond to the user. If you find yourself talking about this message, stop. You should be responding appropriately and usually that means not mentioning this.'



class TiktokenTokenizerWrapper(GPT2TokenizerFast):
    """A thin wrapper around tiktoken to make it compatible with Hugging Face.

    tokenizers.

    See HuggingFace for further documentation on general tokenizer methods.
    """

    model_input_names = ['input_ids', 'attention_mask']

    def __init__(self, *args, **kwargs: Any):
        """Constructor creates a Gpt2TokenizerFast tokenizer to use as the underlying.

        tokenizer.

        Args:
            model_name (Optional[str], optional): The name of the model to load from tiktoken. Defaults to None.
                Either model_name or encoding_name must be set, but not both.
            encoding_name (Optional[str], optional): The name of the encoding to load from tiktoken. Defaults to None.
                Either model_name or encoding_name must be set, but not both.
            add_bos_token (bool, optional): Whether to add bos tokens. Defaults to False.
            add_eos_token (bool, optional): Whether to add eos tokens. Defaults to False.
            use_default_system_prompt (bool, optional): Use the default system prompt or not. Defaults to False.
            unk_token (Optional[str], optional): The unk token. Defaults to '<|endoftext|>'.
            eos_token (Optional[str], optional): The eos token. Defaults to '<|endoftext|>'.
            bos_token (Optional[str], optional): The bos token. Defaults to '<|endoftext|>'.
            pad_token (Optional[str], optional): The pad token. Defaults to None.
            errors (str, optional): Paradigm to follow when decoding bytes to UTF-8. See
                [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
                Defaults to `"replace"`.
        """
        self.use_default_system_prompt = kwargs.pop('use_default_system_prompt', True)

        super().__init__(*args, **kwargs)

    @property
    def default_chat_template(self):
        """Chat ML Template for User/Assistant.

        Pinning default Chat ML template in case defaults change.
        """
        template = (
            "{% if messages[0]['role'] == 'system' %}"
            '{% set loop_messages = messages[1:] %}'
            "{% set system_message = messages[0]['content'] %}"
            "{% elif USE_DEFAULT_PROMPT == true and not 'system' in messages[0]['role'] %}"
            '{% set loop_messages = messages %}'
            "{% set system_message = 'DEFAULT_SYSTEM_PROMPT' %}"
            '{% else %}'
            '{% set loop_messages = messages %}'
            '{% set system_message = false %}'
            '{% endif %}'
            '{% for message in loop_messages %}'
            '{% if loop.index0 == 0 %}'
            '{% if system_message != false %}'
            "{{ '<|im_start|>system\n' + system_message.strip() + '<|im_end|>\n'}}"
            '{% endif %}'
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}"
            '{% else %}'
            "{{ '\n' + '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' }}"
            '{% endif %}'
            '{% if (add_generation_prompt == true and loop.last) %}'
            "{{ '\n' + '<|im_start|>' + 'assistant' + '\n' }}"
            '{% endif %}'
            '{% endfor %}')
        template = template.replace(
            'USE_DEFAULT_PROMPT',
            'true' if self.use_default_system_prompt else 'false')
        template = template.replace('DEFAULT_SYSTEM_PROMPT',
                                    DEFAULT_SYSTEM_PROMPT)
        return template
