import copy
import json

from encoding_dsv32 import encode_messages, parse_message_from_completion_text


with open("test_input.json", "r") as f:
    test_dict = json.load(f)
    messages = test_dict["messages"]
    messages[0]["tools"] = test_dict["tools"]

with open("test_output.txt", "r") as f:
    gold_prompt = f.read().strip()

print(messages)
print("=" * 60)

encode_config = dict(thinking_mode="thinking", drop_thinking=True, add_default_bos_token=True)
prompt = encode_messages(messages, **encode_config)
print(prompt)
assert prompt == gold_prompt
print("=" * 60)

tool_call_message = messages[4]
tool_call_prompt = encode_messages([tool_call_message], context=messages[:4], **encode_config)
tool_call_message_wo_id = copy.deepcopy(tool_call_message)
for tool_call in tool_call_message_wo_id["tool_calls"]:
    tool_call.pop("id")
parsed_tool_call_message = parse_message_from_completion_text(tool_call_prompt, thinking_mode="thinking")
parsed_tool_call_message.pop("content")
assert tool_call_message_wo_id == parsed_tool_call_message

thinking_message = messages[-6]
thinking_prompt = encode_messages([thinking_message], context=messages[:-6], **encode_config)
parsed_thinking_message = parse_message_from_completion_text(thinking_prompt, thinking_mode="thinking")
parsed_thinking_message.pop("tool_calls")
assert thinking_message == parsed_thinking_message

with open("test_input_search_wo_date.json", "r") as f:
    search_messages = json.load(f)["messages"]

with open("test_output_search_wo_date.txt", "r") as f:
    search_gold_prompt = f.read().strip()

search_prompt = encode_messages(search_messages, **encode_config)
assert search_prompt == search_gold_prompt

with open("test_input_search_w_date.json", "r") as f:
    search_messages_w_date = json.load(f)["messages"]

with open("test_output_search_w_date.txt", "r") as f:
    search_gold_prompt_w_date = f.read().strip()

search_prompt_w_date = encode_messages(search_messages_w_date, **encode_config)
with open("test_output_search_w_date_2.txt", "w") as f:
    f.write(search_prompt_w_date)
assert search_prompt_w_date == search_gold_prompt_w_date
