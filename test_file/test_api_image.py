from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://172.18.64.252:8000/v1"
# openai_api_base = "http://127.0.0.1:8001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    timeout=900,
)
response = client.chat.completions.create(
    model="glyph",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    },
                },
                {"type": "text", "text": "中文讲述这个图"},
                # {
                #     "type": "video_url",
                #     "video_url": {
                #         "url": "file:///mnt/transformers/test_file/test.mp4",
                #     }
                # },
                # {
                #     "type": "text",
                #     "text": "Against a blue background, a man wearing black-framed glasses and a white short-sleeve shirt with a small bird pattern is explaining. Which of the following animals evolved hindgut fermentation?"
                # },
            ],
        }
    ],
    max_tokens=512,
    temperature=0.0,
    timeout=900,
    extra_body={
        "top_k": 50,
        "repetition_penalty": 1.1,
    },
)
print(response.choices[0].message.content.strip())
print("===========")
print(response.choices[0].message.reasoning_content.strip())