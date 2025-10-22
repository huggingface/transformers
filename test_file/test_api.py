from openai import OpenAI

# openai_api_key = "sk-onzvhozfpuhmrdxwzuopnovtqadlhcaklppeftevqwicuivx"
# openai_api_base = "https://api.siliconflow.cn/v1/"

openai_api_key = "EMPTY"
openai_api_base = "http://172.18.64.252:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
response = client.chat.completions.create(
    model="glm-4.6",
    messages=[
        {"role": "user", "content": "你好，你是谁开发的"},
    ],
    max_tokens=256,
    temperature=0.0,
    extra_body={"enable_thinking": False}
)
print(response.choices[0].message.content.strip())
print("===========")
print(response.choices[0].message.reasoning_content.strip())
