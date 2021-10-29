import uuid
from typing import Any, Dict, List, Optional, Union

from ..file_utils import add_end_docstrings, is_tf_available, is_torch_available
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, Pipeline


if is_tf_available():
    import tensorflow as tf

if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class Conversation:
    """
    Utility class containing a conversation and its history. This class is meant to be used as an input to the
    :class:`~transformers.ConversationalPipeline`. The conversation contains a number of utility function to manage the
    addition of new user input and generated model responses. A conversation needs to contain an unprocessed user input
    before being passed to the :class:`~transformers.ConversationalPipeline`. This user input is either created when
    the class is instantiated, or by calling :obj:`conversational_pipeline.append_response("input")` after a
    conversation turn.

    Arguments:
        text (:obj:`str`, `optional`):
            The initial user input to start the conversation. If not provided, a user input needs to be provided
            manually using the :meth:`~transformers.Conversation.add_user_input` method before the conversation can
            begin.
        conversation_id (:obj:`uuid.UUID`, `optional`):
            Unique identifier for the conversation. If not provided, a random UUID4 id will be assigned to the
            conversation.
        past_user_inputs (:obj:`List[str]`, `optional`):
            Eventual past history of the conversation of the user. You don't need to pass it manually if you use the
            pipeline interactively but if you want to recreate history you need to set both :obj:`past_user_inputs` and
            :obj:`generated_responses` with equal length lists of strings
        generated_responses (:obj:`List[str]`, `optional`):
            Eventual past history of the conversation of the model. You don't need to pass it manually if you use the
            pipeline interactively but if you want to recreate history you need to set both :obj:`past_user_inputs` and
            :obj:`generated_responses` with equal length lists of strings

    Usage::

        conversation = Conversation("Going to the movies tonight - any suggestions?")

        # Steps usually performed by the model when generating a response:
        # 1. Mark the user input as processed (moved to the history)
        conversation.mark_processed()
        # 2. Append a mode response
        conversation.append_response("The Big lebowski.")

        conversation.add_user_input("Is it good?")
    """

    def __init__(
        self, text: str = None, conversation_id: uuid.UUID = None, past_user_inputs=None, generated_responses=None
    ):
        if not conversation_id:
            conversation_id = uuid.uuid4()
        if past_user_inputs is None:
            past_user_inputs = []
        if generated_responses is None:
            generated_responses = []

        self.uuid: uuid.UUID = conversation_id
        self.past_user_inputs: List[str] = past_user_inputs
        self.generated_responses: List[str] = generated_responses
        self.new_user_input: Optional[str] = text

    def __eq__(self, other):
        if not isinstance(other, Conversation):
            return False
        if self.uuid == other.uuid:
            return True
        return (
            self.new_user_input == other.new_user_input
            and self.past_user_inputs == other.past_user_inputs
            and self.generated_responses == other.generated_responses
        )

    def add_user_input(self, text: str, overwrite: bool = False):
        """
        Add a user input to the conversation for the next round. This populates the internal :obj:`new_user_input`
        field.

        Args:
            text (:obj:`str`): The user input for the next conversation round.
            overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not existing and unprocessed user input should be overwritten when this function is called.
        """
        if self.new_user_input:
            if overwrite:
                logger.warning(
                    f'User input added while unprocessed input was existing: "{self.new_user_input}" was overwritten '
                    f'with: "{text}".'
                )
                self.new_user_input = text
            else:
                logger.warning(
                    f'User input added while unprocessed input was existing: "{self.new_user_input}" new input '
                    f'ignored: "{text}". Set `overwrite` to True to overwrite unprocessed user input'
                )
        else:
            self.new_user_input = text

    def mark_processed(self):
        """
        Mark the conversation as processed (moves the content of :obj:`new_user_input` to :obj:`past_user_inputs`) and
        empties the :obj:`new_user_input` field.
        """
        if self.new_user_input:
            self.past_user_inputs.append(self.new_user_input)
        self.new_user_input = None

    def append_response(self, response: str):
        """
        Append a response to the list of generated responses.

        Args:
            response (:obj:`str`): The model generated response.
        """
        self.generated_responses.append(response)

    def iter_texts(self):
        """
        Iterates over all blobs of the conversation.

        Returns: Iterator of (is_user, text_chunk) in chronological order of the conversation. ``is_user`` is a
        :obj:`bool`, ``text_chunks`` is a :obj:`str`.
        """
        for user_input, generated_response in zip(self.past_user_inputs, self.generated_responses):
            yield True, user_input
            yield False, generated_response
        if self.new_user_input:
            yield True, self.new_user_input

    def __repr__(self):
        """
        Generates a string representation of the conversation.

        Return:
            :obj:`str`:

            Example: Conversation id: 7d15686b-dc94-49f2-9c4b-c9eac6a1f114 user >> Going to the movies tonight - any
            suggestions? bot >> The Big Lebowski
        """
        output = f"Conversation id: {self.uuid} \n"
        for is_user, text in self.iter_texts():
            name = "user" if is_user else "bot"
            output += f"{name} >> {text} \n"
        return output


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        min_length_for_response (:obj:`int`, `optional`, defaults to 32):
            The minimum length (in number of tokens) for a response.
        minimum_tokens (:obj:`int`, `optional`, defaults to 10):
            The minimum length of tokens to leave for a response.
    """,
)
class ConversationalPipeline(Pipeline):
    """
    Multi-turn conversational pipeline.

    This conversational pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task
    identifier: :obj:`"conversational"`.

    The models that this pipeline can use are models that have been fine-tuned on a multi-turn conversational task,
    currently: `'microsoft/DialoGPT-small'`, `'microsoft/DialoGPT-medium'`, `'microsoft/DialoGPT-large'`. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=conversational>`__.

    Usage::

        conversational_pipeline = pipeline("conversational")

        conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
        conversation_2 = Conversation("What's the last book you have read?")

        conversational_pipeline([conversation_1, conversation_2])

        conversation_1.add_user_input("Is it an action movie?")
        conversation_2.add_user_input("What is the genre of this book?")

        conversational_pipeline([conversation_1, conversation_2])
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
            conversations (a :class:`~transformers.Conversation` or a list of :class:`~transformers.Conversation`):
                Conversations to generate responses for.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Returns:
            :class:`~transformers.Conversation` or a list of :class:`~transformers.Conversation`: Conversation(s) with
            updated generated responses for those containing a new user input.
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
            raise ValueError("ConversationalPipeline, expects Conversation as inputs")
        if conversation.new_user_input is None:
            raise ValueError(
                f"Conversation with UUID {type(conversation.uuid)} does not contain new user input to process. "
                "Add user inputs with the conversation's `add_user_input` method"
            )
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
        conversation.mark_processed()
        conversation.append_response(answer)
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
