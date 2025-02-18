from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

# 设置模型名称
model_name = "meta-llama/Llama-2-7b-chat-hf"

# 加载 LLaMA 3.2-1B 的 Tokenizer 和 模型
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# 设置 pad_token 为 eos_token（LLaMA 默认没有 pad_token）
tokenizer.pad_token = tokenizer.eos_token

# 输入文本
input_text = "Hello, how are you?"

# 编码输入文本并添加 attention_mask
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# 移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 生成模型输出
with torch.no_grad():
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],  # 添加 attention_mask
        max_length=50
    )

# 解码输出
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)


