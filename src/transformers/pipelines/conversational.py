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

    Usage:

    ```python
    conversation = Conversation("Going to the movies tonight - any suggestions?")
    conversation.add_message({"role": "assistant", "content": "The Big lebowski."})
    conversation.add_message({"role": "user", "content": "Is it good?"})
    ```"""

    def __init__(
        self, messages: Union[str, List[Dict[str, str]]] = None, conversation_id: uuid.UUID = None, **deprecated_kwargs
    ):
        if not conversation_id:
            conversation_id = uuid.uuid4()

        if messages is None:
            text = deprecated_kwargs.pop("text", None)
            if text is not None:
                messages = [{"role": "user", "content": text}]
            else:
                messages = []
        elif isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # This block deals with the legacy args - new code should just totally
        # avoid past_user_inputs and generated_responses
        self._num_processed_user_inputs = 0
        generated_responses = deprecated_kwargs.pop("generated_responses", None)
        past_user_inputs = deprecated_kwargs.pop("past_user_inputs", None)
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

    def add_user_input(self, text: str, overwrite: bool = False):
        """
        Add a user input to the conversation for the next round. This is a legacy method that assumes that inputs must
        alternate user/assistant/user/assistant, and so will not add multiple user messages in succession. We recommend
        just using `add_message` with role "user" instead.
        """
        if len(self) > 0 and self[-1]["role"] == "user":
            if overwrite:
                logger.warning(
                    f'User input added while unprocessed input was existing: "{self[-1]["content"]}" was overwritten '
                    f'with: "{text}".'
                )
                self[-1]["content"] = text
            else:
                logger.warning(
                    f'User input added while unprocessed input was existing: "{self[-1]["content"]}" new input '
                    f'ignored: "{text}". Set `overwrite` to True to overwrite unprocessed user input'
                )
        else:
            self.messages.append({"role": "user", "content": text})

    def append_response(self, response: str):
        """
        This is a legacy method. We recommend just using `add_message` with an appropriate role instead.
        """
        self.messages.append({"role": "assistant", "content": response})

    def mark_processed(self):
        """
        This is a legacy method, as the Conversation no longer distinguishes between processed and unprocessed user
        input. We set a counter here to keep behaviour mostly backward-compatible, but in general you should just read
        the messages directly when writing new code.
        """
        self._num_processed_user_inputs = len(self._user_messages)

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
        # This is a legacy method for backwards compatibility. It is recommended to just directly access
        # conversation.messages instead.
        for message in self.messages:
            yield message["role"] == "user", message["content"]

    @property
    def _user_messages(self):
        # This is a legacy property for backwards compatibility. It is recommended to just directly access
        # conversation.messages instead.
        return [message["content"] for message in self.messages if message["role"] == "user"]

    @property
    def past_user_inputs(self):
        # This is a legacy property for backwards compatibility. It is recommended to just directly access
        # conversation.messages instead. The modern class does not care about which messages are "processed"
        # or not.
        if not self._user_messages:
            return []
        # In the past, the most recent user message had to be mark_processed() before being included
        # in past_user_messages. The class essentially had a single-message buffer, representing messages that
        # had not yet been replied to. This is no longer the case, but we mimic the behaviour in this property
        # for backward compatibility.
        if self.messages[-1]["role"] != "user" or self._num_processed_user_inputs == len(self._user_messages):
            return self._user_messages

        return self._user_messages[:-1]

    @property
    def generated_responses(self):
        # This is a legacy property for backwards compatibility. It is recommended to just directly access
        # conversation.messages instead.
        return [message["content"] for message in self.messages if message["role"] == "assistant"]

    @property
    def new_user_input(self):
        # This is a legacy property for backwards compatibility. It is recommended to just directly access
        # conversation.messages instead.
        return self._user_messages[-1]


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
    # Any model with a chat template can be used in a ConversationalPipeline.

    >>> chatbot = pipeline(model="facebook/blenderbot-400M-distill")
    >>> # Conversation objects initialized with a string will treat it as a user message
    >>> conversation = Conversation("I'm looking for a movie - what's your favourite one?")
    >>> conversation = chatbot(conversation)
    >>> conversation.messages[-1]["content"]
    ' I don't really have a favorite movie, but I do like action movies. What about you?'

    >>> conversation.add_message({"role": "user", "content": "That's interesting, why do you like action movies?"})
    >>> conversation = chatbot(conversation)
    >>> conversation.messages[-1]["content"]
    ' I think it's just because they're so fast-paced and action-fantastic.'
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This conversational pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"conversational"`.

    This pipeline can be used with any model that has a [chat
    template](https://huggingface.co/docs/transformers/chat_templating) set.
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

    def __call__(self, conversations: Union[List[Dict], Conversation, List[Conversation]], num_workers=0, **kwargs):
        r"""
        Generate responses for the conversation(s) given as inputs.

        Args:
            conversations (a [`Conversation`] or a list of [`Conversation`]):
                Conversation to generate responses for. Inputs can also be passed as a list of dictionaries with `role`
                and `content` keys - in this case, they will be converted to `Conversation` objects automatically.
                Multiple conversations in either format may be passed as a list.
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
        if isinstance(conversations, list) and isinstance(conversations[0], dict):
            conversations = Conversation(conversations)
        elif isinstance(conversations, list) and isinstance(conversations[0], list):
            conversations = [Conversation(conv) for conv in conversations]
        outputs = super().__call__(conversations, num_workers=num_workers, **kwargs)
        if isinstance(outputs, list) and len(outputs) == 1:
            return outputs[0]
        return outputs

    def preprocess(self, conversation: Conversation, min_length_for_response=32) -> Dict[str, Any]:
        input_ids = self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True)

        if self.framework == "pt":
            input_ids = torch.LongTensor([input_ids])
        elif self.framework == "tf":
            input_ids = tf.constant([input_ids])
        return {"input_ids": input_ids, "conversation": conversation}

    def _forward(self, model_inputs, minimum_tokens=10, **generate_kwargs):
        n = model_inputs["input_ids"].shape[1]
        conversation = model_inputs.pop("conversation")
        if "max_length" not in generate_kwargs and "max_new_tokens" not in generate_kwargs:
            generate_kwargs["max_new_tokens"] = 256
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
