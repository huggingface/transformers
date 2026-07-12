# Cohere[[cohere]]

## 개요[[overview]]

The Cohere Command-R 모델은 Cohere팀이 [Command-R: 프로덕션 규모의 검색 증강 생성](https://txt.cohere.com/command-r/)라는 블로그 포스트에서 소개 되었습니다.

논문 초록:

*Command-R은 기업의 프로덕션 규모 AI를 가능하게 하기 위해 RAG(검색 증강 생성)와 도구 사용을 목표로 하는 확장 가능한 생성 모델입니다. 오늘 우리는 대규모 프로덕션 워크로드를 목표로 하는 새로운 LLM인 Command-R을 소개합니다. Command-R은 높은 효율성과 강력한 정확성의 균형을 맞추는 '확장 가능한' 모델 카테고리를 대상으로 하여, 기업들이 개념 증명을 넘어 프로덕션 단계로 나아갈 수 있게 합니다.*

*Command-R은 검색 증강 생성(RAG)이나 외부 API 및 도구 사용과 같은 긴 문맥 작업에 최적화된 생성 모델입니다. 이 모델은 RAG 애플리케이션을 위한 최고 수준의 통합을 제공하고 기업 사용 사례에서 뛰어난 성능을 발휘하기 위해 우리의 업계 선도적인 Embed 및 Rerank 모델과 조화롭게 작동하도록 설계되었습니다. 기업이 대규모로 구현할 수 있도록 만들어진 모델로서, Command-R은 다음과 같은 특징을 자랑합니다:
- RAG 및 도구 사용에 대한 강력한 정확성
- 낮은 지연 시간과 높은 처리량
- 더 긴 128k 컨텍스트와 낮은 가격
- 10개의 주요 언어에 걸친 강력한 기능
- 연구 및 평가를 위해 HuggingFace에서 사용 가능한 모델 가중치

모델 체크포인트는 [이곳](https://huggingface.co/CohereForAI/c4ai-command-r-v01)에서 확인하세요.
이 모델은 [Saurabh Dash](https://huggingface.co/saurabhdash)과 [Ahmet Üstün](https://huggingface.co/ahmetustun)에 의해 기여 되었습니다. Hugging Face에서 이 코드의 구현은 [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)에 기반하였습니다.

## 사용 팁[[usage-tips]]

<Tip warning={true}>

Hub에 업로드된 체크포인트들은 `dtype = 'float16'`을 사용합니다.
이는 `AutoModel` API가 체크포인트를 `torch.float32`에서 `torch.float16`으로 변환하는 데 사용됩니다.

온라인 가중치의 `dtype`은 `model = AutoModelForCausalLM.from_pretrained("path", dtype = "auto")`를 사용하여 모델을 초기화할 때 `dtype="auto"`를 사용하지 않는 한 대부분 무관합니다. 그 이유는 모델이 먼저 다운로드되고(온라인 체크포인트의 `dtype` 사용), 그 다음 `torch`의 기본 `dtype`으로 변환되며(이때 `torch.float32`가 됨), 마지막으로 config에 `dtype`이 제공된 경우 이를 사용하기 때문입니다.

모델을 `float16`으로 훈련하는 것은 권장되지 않으며 `nan`을 생성하는 것으로 알려져 있습니다. 따라서 모델은 `bfloat16`으로 훈련해야 합니다.
</Tip>
모델과 토크나이저는 다음과 같이 로드할 수 있습니다:

```python
# pip install transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "CohereForAI/c4ai-command-r-v01"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Format message with the command-r chat template
messages = [{"role": "user", "content": "Hello, how are you?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
```

- Flash Attention 2를 `attn_implementation="flash_attention_2"`를 통해 사용할 때는, `from_pretrained` 클래스 메서드에 `dtype`을 전달하지 말고 자동 혼합 정밀도 훈련(Automatic Mixed-Precision training)을 사용하세요. `Trainer`를 사용할 때는 단순히 `fp16` 또는 `bf16`을 `True`로 지정하면 됩니다. 그렇지 않은 경우에는 `torch.autocast`를 사용하고 있는지 확인하세요. 이는 Flash Attention이 `fp16`와 `bf16` 데이터 타입만 지원하기 때문에 필요합니다.

## 리소스[[resources]]

Command-R을 시작하는 데 도움이 되는 Hugging Face와 community 자료 목록(🌎로 표시됨) 입니다. 여기에 포함될 자료를 제출하고 싶으시다면 PR(Pull Request)를 열어주세요. 리뷰 해드리겠습니다! 자료는 기존 자료를 복제하는 대신 새로운 내용을 담고 있어야 합니다.


<PipelineTag pipeline="text-generation"/>

FP16 모델 로딩
```python
# pip install transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "CohereForAI/c4ai-command-r-v01"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# command-r 챗 템플릿으로 메세지 형식을 정하세요
messages = [{"role": "user", "content": "Hello, how are you?"}]
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
## <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
```

bitsandbytes 라이브러리를 이용해서 4bit 양자화된 모델 로딩
```python
# pip install transformers bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

model_id = "CohereForAI/c4ai-command-r-v01"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)

gen_tokens = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.3,
    )

gen_text = tokenizer.decode(gen_tokens[0])
print(gen_text)
```


## CohereConfig[[transformers.CohereConfig]]

[[autodoc]] CohereConfig
ave_vocabulary

## CohereModel[[transformers.CohereModel]]

[[autodoc]] CohereModel
    - forward


## CohereForCausalLM[[transformers.CohereForCausalLM]]

[[autodoc]] CohereForCausalLM
    - forward


