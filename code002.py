from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载模型和 tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

# 输入文本
input_text = "Hello, how are you?"

'''# 编码输入文本
inputs = tokenizer(input_text, return_tensors="pt")

# 生成模型输出
with torch.no_grad():
    outputs = model.generate(inputs['input_ids'], max_length=50)'''
# 设置 pad_token 为 eos_token
tokenizer.pad_token = tokenizer.eos_token
# 编码输入文本并添加 attention_mask
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

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