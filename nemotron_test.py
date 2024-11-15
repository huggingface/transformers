from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Hugging Face에서 Minitron 모델과 토크나이저 로드
model_name = "nvidia/Minitron-8B-Base"  # Minitron 모델 이름으로 바꿔주세요
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='/mnt/dongsoo-lee2/hf_models')
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir='/mnt/dongsoo-lee2/hf_models')

# 평가 모드로 설정
model.eval()

# 샘플 입력 텍스트
input_text = "Hello, how are you?"

# 입력 토큰화
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 첫 번째 호출 (prefill 단계)
with torch.no_grad():
    outputs = model(input_ids, use_cache=True)  # use_cache=True로 설정
    logits = outputs.logits
    past_key_values = outputs.past_key_values

# 출력 결과 확인
print("Logits shape:", logits.shape)
print("Number of layers in past_key_values:", len(past_key_values))
print("Shape of keys and values in the first layer:")
print("Key shape:", past_key_values[0][0].shape)
print("Value shape:", past_key_values[0][1].shape)

# 새로운 입력을 추가해 캐싱 활용 테스트
new_input_text = " What about you?"
new_input_ids = tokenizer(new_input_text, return_tensors="pt").input_ids

# 이전 키-값 캐시와 함께 새 입력 전달
with torch.no_grad():
    outputs_with_cache = model(new_input_ids, past_key_values=past_key_values, use_cache=True)

# 캐싱 후 결과 확인
new_logits = outputs_with_cache.logits
new_past_key_values = outputs_with_cache.past_key_values

print("New logits shape:", new_logits.shape)
print("Number of layers in new past_key_values:", len(new_past_key_values))
