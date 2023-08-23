import uuid
from typing import Any, Dict, List, Union

from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_tf_available():
    import tensorflow as tf

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class Conversation:
    """
    Utility class containing a conversation and its history. This class is meant to be used as an input to the
    [`ConversationalPipeline`]. The conversation contains several utility functions to manage the addition of new user
    inputs and generated model responses.

    Arguments:
        messages (Union[str, List[Dict[str, str]]], *optional*):
            The initial messages to start the conversation, either a string, or a list of dicts containing "role" and
            "content" keys. If a string is passed, it is interpreted as a single message with the "user" role.
        conversation_id (`uuid.UUID`, *optional*):
            Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the
            conversation.
        past_user_inputs (`List[str]`, *optional*):
            This is a legacy argument that should not be used in new code. It is kept only for backward compatibility.
        generated_responses (`List[str]`, *optional*):
            This is a legacy argument that should not be used in new code. It is kept only for backward compatibility.

    Usage:

    ```python
    conversation = Conversation("Going to the movies tonight - any suggestions?")
    conversation.add_message({"role": "assistant", "content": "The Big lebowski."})
    conversation.add_message({"role": "user", "content": "Is it good?"})
    ```"""

    def __init__(
        self,
        messages: Union[str, List[Dict[str, str]]] = None,
        conversation_id: uuid.UUID = None,
        past_user_inputs=None,
        generated_responses=None,
    ):
        if not conversation_id:
            conversation_id = uuid.uuid4()
        if messages is None:
            messages = []
        elif isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # This block deals with the legacy args - new code should just totally
        # avoid past_user_inputs and generated_responses
        if generated_responses is not None and past_user_inputs is None:
            raise ValueError("generated_responses cannot be passed without past_user_inputs!")
        if past_user_inputs is not None:
            legacy_messages = []
            if generated_responses is None:
                generated_responses = []
            # We structure it this way instead of using zip() because the lengths may differ by 1
            for i in range(max([len(past_user_inputs), len(generated_responses)])):
                if i < len(past_user_inputs):
                    legacy_messages.append({"role": "user", "content": past_user_inputs[i]})
                if i < len(generated_responses):
                    legacy_messages.append({"role": "assistant", "content": generated_responses[i]})
            messages = legacy_messages + messages

        self.uuid = conversation_id
        self.messages = messages

    def __eq__(self, other):
        if not isinstance(other, Conversation):
            return False
        return self.uuid == other.uuid or self.messages == other.messages

    def add_message(self, message: Dict[str, str]):
        if not set(message.keys()) == {"role", "content"}:
            raise ValueError("Message should contain only 'role' and 'content' keys!")
        if message["role"] not in ("user", "assistant", "system"):
            raise ValueError("Only 'user', 'assistant' and 'system' roles are supported for now!")
        self.messages.append(message)

    def __iter__(self):
        for message in self.messages:
            yield message

    def __getitem__(self, item):
        return self.messages[item]

    def __setitem__(self, key, value):
        self.messages[key] = value

    def __len__(self):
        return len(self.messages)

    def __repr__(self):
        """
        Generates a string representation of the conversation.

        Returns:
            `str`:

        Example:
            Conversation id: 7d15686b-dc94-49f2-9c4b-c9eac6a1f114 user: Going to the movies tonight - any suggestions?
            bot: The Big Lebowski
        """
        output = f"Conversation id: {self.uuid}\n"
        for message in self.messages:
            output += f"{message['role']}: {message['content']}\n"
        return output

    def iter_texts(self):
        # Legacy method to keep the old methods happy for now
        # TODO Remove this method and move the old _build_conversation methods to using the proper iterator
        for message in self.messages:
            yield message["role"] == "user", message["content"]

    @property
    def past_user_inputs(self):
        # This is a legacy property for backwards compatibility. It is recommended to just directly access
        # conversation.messages instead.
        return [message["content"] for message in self.messages if message["role"] == "user"]

    @property
    def generated_responses(self):
        # This is a legacy property for backwards compatibility. It is recommended to just directly access
        # conversation.messages instead.
        return [message["content"] for message in self.messages if message["role"] == "assistant"]


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        min_length_for_response (`int`, *optional*, defaults to 32):
            The minimum length (in number of tokens) for a response.
        minimum_tokens (`int`, *optional*, defaults to 10):
            The minimum length of tokens to leave for a response.
    """,
)
class ConversationalPipeline(Pipeline):
    """
    Multi-turn conversational pipeline.

    Example:

    ```python
    >>> from transformers import pipeline, Conversation

    >>> chatbot = pipeline(model="microsoft/DialoGPT-medium")
    >>> conversation = Conversation("Going to the movies tonight - any suggestions?")
    >>> conversation = chatbot(conversation)
    >>> conversation.generated_responses[-1]
    'The Big Lebowski'

    >>> conversation.add_user_input("Is it an action movie?")
    >>> conversation = chatbot(conversation)
    >>> conversation.generated_responses[-1]
    "It's a comedy."
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This conversational pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"conversational"`.

    The models that this pipeline can use are models that have been fine-tuned on a multi-turn conversational task,
    currently: *'microsoft/DialoGPT-small'*, *'microsoft/DialoGPT-medium'*, *'microsoft/DialoGPT-large'*. See the
    up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=conversational).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _sanitize_parameters(
        self, min_length_for_response=None, minimum_tokens=None, clean_up_tokenization_spaces=None, **generate_kwargs
    ):
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}

        if min_length_for_response is not None:
            preprocess_params["min_length_for_response"] = min_length_for_response
        if minimum_tokens is not None:
            forward_params["minimum_tokens"] = minimum_tokens

        if "max_length" in generate_kwargs:
            forward_params["max_length"] = generate_kwargs["max_length"]
            # self.max_length = generate_kwargs.get("max_length", self.model.config.max_length)
        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        if generate_kwargs:
            forward_params.update(generate_kwargs)
        return preprocess_params, forward_params, postprocess_params

    def __call__(self, conversations: Union[Conversation, List[Conversation]], num_workers=0, **kwargs):
        r"""
        Generate responses for the conversation(s) given as inputs.

        Args:
            conversations (a [`Conversation`] or a list of [`Conversation`]):
                Conversations to generate responses for.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Returns:
            [`Conversation`] or a list of [`Conversation`]: Conversation(s) with updated generated responses for those
            containing a new user input.
        """
        # XXX: num_workers==0 is required to be backward compatible
        # Otherwise the threads will require a Conversation copy.
        # This will definitely hinder performance on GPU, but has to be opted
        # in because of this BC change.
        outputs = super().__call__(conversations, num_workers=num_workers, **kwargs)
        if isinstance(outputs, list) and len(outputs) == 1:
            return outputs[0]
        return outputs

    def preprocess(self, conversation: Conversation, min_length_for_response=32) -> Dict[str, Any]:
        if not isinstance(conversation, Conversation):
            raise ValueError("ConversationalPipeline expects Conversation as inputs")
        if hasattr(self.tokenizer, "_build_conversation_input_ids"):
            input_ids = self.tokenizer._build_conversation_input_ids(conversation)
        else:
            # If the tokenizer cannot handle conversations, we default to only the old version
            input_ids = self._legacy_parse_and_tokenize(conversation)

        if self.framework == "pt":
            input_ids = torch.LongTensor([input_ids])
        elif self.framework == "tf":
            input_ids = tf.constant([input_ids])
        return {"input_ids": input_ids, "conversation": conversation}

    def _forward(self, model_inputs, minimum_tokens=10, **generate_kwargs):
        max_length = generate_kwargs.get("max_length", self.model.config.max_length)

        n = model_inputs["input_ids"].shape[1]
        if max_length - minimum_tokens < n:
            logger.warning(f"Conversation input is to long ({n}), trimming it to ({max_length} - {minimum_tokens})")
            trim = max_length - minimum_tokens
            model_inputs["input_ids"] = model_inputs["input_ids"][:, -trim:]
            if "attention_mask" in model_inputs:
                model_inputs["attention_mask"] = model_inputs["attention_mask"][:, -trim:]
        conversation = model_inputs.pop("conversation")
        generate_kwargs["max_length"] = max_length
        output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        if self.model.config.is_encoder_decoder:
            start_position = 1
        else:
            start_position = n
        return {"output_ids": output_ids[:, start_position:], "conversation": conversation}

    def postprocess(self, model_outputs, clean_up_tokenization_spaces=True):
        output_ids = model_outputs["output_ids"]
        answer = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        conversation = model_outputs["conversation"]
        conversation.add_message({"role": "assistant", "content": answer})
        return conversation

    def _legacy_parse_and_tokenize(self, conversation: Conversation) -> Dict:
        eos_token_id = self.tokenizer.eos_token_id
        input_ids = []
        for is_user, text in conversation.iter_texts():
            if eos_token_id is not None:
                input_ids.extend(self.tokenizer.encode(text, add_special_tokens=False) + [eos_token_id])
            else:
                input_ids.extend(self.tokenizer.encode(text, add_special_tokens=False))

        if len(input_ids) > self.tokenizer.model_max_length:
            input_ids = input_ids[-self.tokenizer.model_max_length :]
        return input_ids
