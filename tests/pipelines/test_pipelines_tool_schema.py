import pytest
import torch

from transformers.pipelines.any_to_any import AnyToAnyPipeline
from transformers.pipelines.any_to_any import Chat as AnyChat
from transformers.pipelines.any_to_any import ReturnType as AnyReturnType
from transformers.pipelines.image_text_to_text import ImageTextToTextPipeline
from transformers.pipelines.image_text_to_text import ReturnType as ImageReturnType
from transformers.utils.chat_parsing import parse_response
from transformers.utils.chat_template_utils import Chat


PREFIX = "<|im_start|>assistant\n"

RESPONSE = (
    "<tool_call>\n<function=set_alarm>\n"
    "<parameter=hour>7</parameter>\n"
    "<parameter=enabled>false</parameter>\n"
    "</function>\n</tool_call>"
)

TEMPLATE = {
    "defaults": {"role": "assistant"},
    "start_anchor": PREFIX,
    "fields": {
        "tool_calls": {
            "open_pattern": r"<tool_call>\s*<function=(?P<name>\w+)>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "xml-inline",
            "content_args": {"tag_pattern": r"<parameter=(?P<key>\w+)>\s*(?P<value>.*?)\s*</parameter>"},
            "transform": {
                "type": "function",
                "function": {
                    "name": "{name}",
                    "arguments": "{content}",
                },
            },
        }
    },
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer"},
                    "enabled": {"type": "boolean"},
                },
            },
        },
    }
]


class Tokenizer:
    response_template = TEMPLATE

    def parse_response(self, value, *, prefix=None, tools=None):
        return parse_response(value, TEMPLATE, prefix=prefix, tools=tools)


class Processor:
    tokenizer = Tokenizer()

    def post_process_image_text_to_text(self, sequences, **kwargs):
        return [PREFIX + (RESPONSE if len(s) > 1 else "") for s in sequences]

    post_process_multimodal_output = post_process_image_text_to_text


@pytest.mark.parametrize(
    "cls,chat_cls,return_type",
    [
        (ImageTextToTextPipeline, Chat, ImageReturnType),
        (AnyToAnyPipeline, AnyChat, AnyReturnType),
    ],
)
@pytest.mark.parametrize("with_tools", [False, True])
def test_multimodal_response_tool_schema(cls, chat_cls, return_type, with_tools):
    pipe = cls.__new__(cls)
    pipe.processor = Processor()
    pipe.tokenizer = pipe.processor.tokenizer

    if cls is AnyToAnyPipeline:
        pipe.model = type("Model", (), {"output_modalities": ["text"]})()

    params = {"tools": TOOLS} if with_tools else {}
    _, _, post = pipe._sanitize_parameters(**params)

    if with_tools:
        assert post["tools"] == TOOLS

    result = pipe.postprocess(
        {
            "input_ids": torch.tensor([[1]]),
            "generated_sequence": torch.tensor([[1, 2]]),
            "prompt_text": chat_cls([{"role": "user", "content": "Set alarm"}]),
        },
        return_type=return_type.FULL_TEXT,
        **post,
    )

    actual = result[0]["generated_text"][-1]["tool_calls"][0]["function"]["arguments"]

    expected = {"hour": 7, "enabled": False} if with_tools else {"hour": "7", "enabled": "false"}
    assert actual == expected
