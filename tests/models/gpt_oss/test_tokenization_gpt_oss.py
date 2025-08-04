from datetime import datetime

from transformers.testing_utils import require_torch
from transformers.utils import DocstringParsingException, TypeHintParsingException, get_json_schema
from transformers import AutoTokenizer
from typing import Union, Optional
import json

if is_harmony_available():
    from openai_harmony import (
        Author,
        Conversation,
        DeveloperContent,
        HarmonyEncodingName,
        Message,
        Role,
        SystemContent,
        ToolDescription,
        load_harmony_encoding,
        ReasoningEffort,
        RenderConversationConfig
    )

def get_system_message(browser_tool=False, python_tool=False):
    system_message = (
        SystemContent.new()
        .with_model_identity(
            "You are ChatGPT, a large language model trained by OpenAI."
        )
        .with_reasoning_effort(ReasoningEffort.MEDIUM)
        .with_conversation_start_date(datetime.now().strftime("%Y-%m-%d"))
        .with_knowledge_cutoff("2024-06")
        .with_required_channels(["analysis", "commentary", "final"])
    )
    if browser_tool:
        system_message = system_message.with_browser_tool()
    if python_tool:
        system_message = system_message.with_python_tool()
    return system_message


def get_developer_message(tools: list | None):
    developer_message = (
        DeveloperContent.new()
        .with_instructions("Always respond in riddles")

    )
    if tools is not None:
        harmony_tools = [ToolDescription.new(
            tool["function"]["name"],
            tool["function"]["description"],
            parameters=tool["function"]["parameters"]
        ) for tool in tools]
        developer_message = developer_message.with_function_tools(harmony_tools)
    return developer_message


def get_harmony_chat(messages, tools=None, browser_tool=False, python_tool=False, for_completion=True):
    last_tool_name = None
    chat = [
               Message.from_role_and_content(Role.SYSTEM, get_system_message(browser_tool, python_tool)),
               Message.from_role_and_content(Role.DEVELOPER, get_developer_message(tools))
           ]
    for message in messages:
        if message["role"] == "user":
            chat.append(Message.from_role_and_content(Role.USER, message['content']))
        elif message["role"] == "assistant":
            if "tool_calls" in message:
                chat.append(
                    Message.from_role_and_content(Role.ASSISTANT, message['content'])
                    .with_channel("analysis")
                )
                chat.append(
                    Message.from_role_and_content(Role.ASSISTANT, json.dumps(message["tool_calls"][0]["arguments"]))
                    .with_channel("commentary")
                    .with_recipient(f"functions.{message['tool_calls'][0]['name']}")
                    .with_content_type("json")
                )
                last_tool_name = message["tool_calls"][0]["name"]
            elif "thinking" in message:
                chat.append(
                    Message.from_role_and_content(
                    Role.ASSISTANT,
                    message["thinking"],
                ).with_channel("analysis")
                )
                chat.append(
                    Message.from_role_and_content(
                        Role.ASSISTANT,
                        message['content']
                    ).with_channel("final")
                )
            else:
                chat.append(Message.from_role_and_content(Role.ASSISTANT, message['content']).with_channel("final"))
        elif message["role"] == "tool":
            chat.append(
                Message.from_author_and_content(Author.new(Role.TOOL, f"functions.{last_tool_name}"),
                                              json.dumps(message['content']))
                .with_recipient("assistant").with_channel("commentary")
            )
        else:
            raise ValueError(f"Unknown role: {message['role']}")

    convo = Conversation.from_messages(chat)
    if for_completion:
        tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    else:
        config = RenderConversationConfig(auto_drop_analysis=False)
        tokens = encoding.render_conversation(convo, config=config)
    return encoding.decode_utf8(tokens)


@require_torch
@require_harmony
class GptOssChatTemplateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        cls.tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    def test_simple_function(self):
        def get_current_weather(x: int):
            """
            Test function

            Args:
                 x: The input
            """
            return x

        schema = get_json_schema(get_current_weather)
        messages = [
            {"role": "system", "content": "Always respond in riddles"},
            {"role": "user", "content": "Hi there!"}
        ]
        harmony_formatted = get_harmony_chat(messages[1:], [schema], browser_tool=True, python_tool=True)
        hf_formatted = tokenizer.apply_chat_template(messages, tools=[schema], tokenize=False, builtin_tools=["python", "browser"], add_generation_prompt=True)
        self.assertEqual(harmony_formatted, hf_formatted)

    def test_simple_function_without_builtins(self):
        def get_current_weather(x: int):
            """
            Test function

            Args:
                 x: The input
            """
            return x

        schema = get_json_schema(get_current_weather)
        messages = [
            {"role": "system", "content": "Always respond in riddles"},
            {"role": "user", "content": "Hi there!"}
        ]
        harmony_formatted = get_harmony_chat(messages[1:], [schema], browser_tool=False, python_tool=False)
        hf_formatted = tokenizer.apply_chat_template(messages, tools=[schema], tokenize=False, add_generation_prompt=True)
        self.assertEqual(harmony_formatted, hf_formatted)


    def test_less_simple_function(self):
        def get_current_weather(x: int, y: str = "default") -> tuple[int, str]:
            """
            Test function

            Args:
                 x: The input
                 y: The optional input
            """
            return x, y

        schema = get_json_schema(get_current_weather)
        messages = [
            {"role": "system", "content": "Always respond in riddles"},
            {"role": "user", "content": "Hi there!"}
        ]
        harmony_formatted = get_harmony_chat(messages[1:], [schema], browser_tool=True, python_tool=True)
        hf_formatted = tokenizer.apply_chat_template(messages, tools=[schema], tokenize=False, builtin_tools=["python", "browser"], add_generation_prompt=True)
        self.assertEqual(harmony_formatted, hf_formatted)

    def test_complex_function(self):
        def get_current_weather(x: int, y: Union[dict, list], z: Optional[list[str]]):
            """
            Test function

            Args:
                 x: The input
                 y: The union input
                 z: The optional list of strings
            """
            return x, y, z
        schema = get_json_schema(get_current_weather)
        messages = [
            {"role": "system", "content": "Always respond in riddles"},
            {"role": "user", "content": "Hi there!"}
        ]
        harmony_formatted = get_harmony_chat(messages[1:], [schema], browser_tool=True, python_tool=True)
        hf_formatted = tokenizer.apply_chat_template(messages, tools=[schema], tokenize=False, builtin_tools=["python", "browser"], add_generation_prompt=True)
        self.assertEqual(harmony_formatted, hf_formatted)

    def test_tool_calls(self):
        def get_current_weather(x: int):
            """
            Test function

            Args:
                 x: The input
            """
            return x

        schema = get_json_schema(get_current_weather)
        messages = [
            {"role": "system", "content": "Always respond in riddles"},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Time to look up the weather!", "tool_calls": [{"name": "get_current_weather", "arguments": {"x": 42}}]},
            {"role": "tool", "content": "30"}
        ]
        harmony_formatted = get_harmony_chat(messages[1:], [schema], browser_tool=True, python_tool=True)
        hf_formatted = tokenizer.apply_chat_template(messages, tools=[schema], tokenize=False, builtin_tools=["python", "browser"], add_generation_prompt=True)
        self.assertEqual(harmony_formatted, hf_formatted)

    def test_tool_calls_in_longer_chat(self):
        def get_current_weather(x: int):
            """
            Test function

            Args:
                 x: The input
            """
            return x

        schema = get_json_schema(get_current_weather)
        messages = [
            {"role": "system", "content": "Always respond in riddles"},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Not calling a tool yet!"},
            {"role": "user", "content": "Now call a tool!"},
            {"role": "assistant", "content": "Time to look up the weather!", "tool_calls": [{"name": "get_current_weather", "arguments": {"x": 42}}]},
            {"role": "tool", "content": "30"}
        ]
        harmony_formatted = get_harmony_chat(messages[1:], [schema], browser_tool=True, python_tool=True)
        hf_formatted = tokenizer.apply_chat_template(messages, tools=[schema], tokenize=False, builtin_tools=["python", "browser"], add_generation_prompt=True)
        self.assertEqual(harmony_formatted, hf_formatted)

    def test_thinking(self):
        messages = [
            {"role": "system", "content": "Always respond in riddles"},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello!", "thinking": "Thinking about things..."},
            {"role": "user", "content": "Nice thinking!"},
        ]
        harmony_formatted = get_harmony_chat(messages[1:])
        hf_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertEqual(harmony_formatted, hf_formatted)


    def test_thinking_in_past_messages(self):
        messages = [
            {"role": "system", "content": "Always respond in riddles"},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello!", "thinking": "Thinking about things..."},
            {"role": "user", "content": "Nice thinking!"},
            {"role": "assistant", "content": "Yes it was!", "thinking": "I wonder if humans can detect sarcasm..."},
            {"role": "user", "content": "I can!"}
        ]
        harmony_formatted = get_harmony_chat(messages[1:])
        hf_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertEqual(harmony_formatted, hf_formatted)

    def test_for_training(self):
        # We end with an assistant message and use add_generation_prompt=False
        messages = [
            {"role": "system", "content": "Always respond in riddles"},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello!", "thinking": "Thinking about things..."},
        ]
        harmony_formatted = get_harmony_chat(messages[1:], for_completion=False)
        harmony_formatted = harmony_formatted.removesuffix("<|end|>") + "<|return|>"
        hf_formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        self.assertEqual(harmony_formatted, hf_formatted)

    def test_for_training_multi_turn(self):
        # We end with an assistant message and use add_generation_prompt=False
        messages = [
            {"role": "system", "content": "Always respond in riddles"},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello!", "thinking": "Thinking about things..."},
            {"role": "user", "content": "Nice thinking!"},
            {"role": "assistant", "content": "Yes it was!", "thinking": "I wonder if humans can detect sarcasm..."},
        ]
        harmony_formatted = get_harmony_chat(messages[1:], for_completion=False)
        harmony_formatted = harmony_formatted.removesuffix("<|end|>") + "<|return|>"
        hf_formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        self.assertEqual(harmony_formatted, hf_formatted)

    def test_assistant_turns_without_thinking(self):
        messages = [
            {"role": "system", "content": "Always respond in riddles"},
            {"role": "user", "content": "Hi there!"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Nice lack of thinking!"},
            {"role": "assistant", "content": "Yes it was!"},
            {"role": "user", "content": "Yeah!"}
        ]
        harmony_formatted = get_harmony_chat(messages[1:])
        hf_formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertEqual(harmony_formatted, hf_formatted)