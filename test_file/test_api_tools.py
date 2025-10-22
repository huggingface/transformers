from openai import OpenAI
import json

# openai_api_base ="https://open.bigmodel.cn/api/paas/v4"
# openai_api_key="e2c883fb4bc243e28f62c99c85d89c01.9MVM2f666ywn803l"
openai_api_base = "http://127.0.0.1:8000/v1"
openai_api_key = "EMPTY"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country e.g. Bogotá, Colombia",
                    }
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "return_delivered_order_items",
            "description": "Return items from a delivered order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID, e.g. #W4794911",
                    },
                    "item_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of item IDs to return",
                    },
                    "payment_method_id": {
                        "type": "string",
                        "description": "Payment method ID for processing the return, e.g. paypal_7503218",
                    },
                },
                "required": ["order_id", "item_ids", "payment_method_id"],
                "additionalProperties": False,
            },
        },
    },
]
tools_messages = [
    {
        "role": "tool",
        "tool_call_id": "tool-call-bf208d1d-9b5f-407f-8c6e-c35e54aa2fef",
        "content": '{"city": "北京", "date": "2024-06-27", "weather": "晴", "temperature": "26C"}',
    },
]
messages = [
    {"role": "system", "content": "请你调用工具，用中文回答问题。"},
    {"role": "user", "content": "请帮我查询一下北京的天气。"},
]

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.chat.completions.create(
    model="GLM-4.5",
    messages=messages,
    tools=tools,
    max_tokens=4096,
    temperature=0.0,
)
tool_call = completion.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)
print("===== TOOL CALL =====")
print(tool_call)
messages.append(completion.choices[0].message)
messages.append(tools_messages[0])

completion_2 = client.chat.completions.create(
    model="GLM-4.5",
    messages=messages,
    tools=tools,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
print("===== RESPONSE =====")
print(completion_2.choices[0].message.content)
