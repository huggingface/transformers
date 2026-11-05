<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Llama3[[llama3]]

```py3
import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B"

pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"dtype": torch.bfloat16}, device_map="auto")
pipeline("Hey how are you doing today?")
```

## 개요[[overview]]

라마3 모델은 Meta AI 팀이 제안한 [메타 라마3 소개: 현재까지 가장 유능한 공개 가능 LLM](https://ai.meta.com/blog/meta-llama-3/)에서 소개되었습니다.

해당 블로그 포스트의 초록입니다:

*오늘, 광범위한 사용을 위해 이용 가능한 라마의 차세대 모델인 메타 라마3의 첫 두 모델을 공유하게 되어 기쁩니다. 이번 출시는 8B와 70B 매개변수를 가진 사전 훈련 및 지시 미세 조정된 언어 모델을 특징으로 하며, 광범위한 사용 사례를 지원할 수 있습니다. 라마의 이 차세대 모델은 다양한 산업 벤치마크에서 최첨단의 성능을 보여주며, 개선된 추론 능력을 포함한 새로운 기능을 제공합니다. 우리는 이것들이 단연코 해당 클래스에서 최고의 오픈 소스 모델이라고 믿습니다. 오랜 개방적 접근 방식을 지지하며, 우리는 라마3를 커뮤니티 기여자들에게 맡기고 있습니다. 애플리케이션에서 개발자 도구, 평가, 추론 최적화 등에 이르기까지 AI 스택 전반에 걸친 다음 혁신의 물결을 촉발하길 희망합니다. 여러분이 무엇을 만들지 기대하며 여러분의 피드백을 고대합니다.*

라마3 모델의 모든 체크포인트는 [이곳](https://huggingface.co/models?search=llama3)에서 확인하세요.
원본 코드는 [이곳](https://github.com/meta-llama/llama3)에서 확인할 수 있습니다.

## 사용 팁[[usage-tips]]

<Tip warning={true}>

`라마3` 모델들은 `bfloat16`를 사용하여 훈련되었지만, 원래의 추론은 `float16`을 사용합니다. Hub에 업로드된 체크포인트들은 `dtype = 'float16'`을 사용하는데, 이는 `AutoModel` API가 체크포인트를 `torch.float32`에서 `torch.float16`으로 변환하는데 이용됩니다. 

 `model = AutoModelForCausalLM.from_pretrained("path", dtype = "auto")`를 사용하여 모델을 초기화할 때, 온라인 가중치의 `dtype`는 `dtype="auto"`를 사용하지 않는 한 대부분 무관합니다. 그 이유는 모델이 먼저 다운로드되고(온라인 체크포인트의 `dtype`를 사용), 그 다음 `torch`의 `dtype`으로 변환되어(`torch.float32`가 됨), 마지막으로 config에 `dtype`이 제공된 경우 가중치가 사용되기 때문입니다.

`float16`으로 모델을 훈련하는 것은 권장되지 않으며 `nan`을 생성하는 것으로 알려져 있습니다. 따라서 모든 모델은 `bfloat16`으로 훈련되어야 합니다.

</Tip>

팁:

- 라마3 모델을 위한 가중치는 [이 폼](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)을 채우면서 얻어져야 합니다.
- 아키텍처는 라마2와 정확히 같습니다.
- 토크나이저는 [tiktoken](https://github.com/openai/tiktoken) (sentencepiece 구현에 기반한 라마2 와는 다르게)에 기반한 BPE 모델입니다. tiktoken 기반 토크나이저가 sebtencepiece 기반 방식과 다른점은 입력 토큰이 vocab에 이미 존재할 때 BPE 병합 룰을 무시하고 싱글 토큰으로 토크나이징한다는 점에서 가장 큰 차이를 보입니다. 자세히 말하면 `"hugging"`이 vocab에 존재하고 기존에 병합이 존재하지 않으면, `["hug","ging"]` 처럼 두 토큰으로 더 작은 단위의 단어를 가지는 것이 아니라, 하나의 토큰만을 자동으로 리턴하는 것을 의미합니다.
- 기본 모델은 패딩 토큰이 없다는 것을 의미하는 `pad_id = -1`을 사용합니다. 같은 로직을 사용할 수 없으니 `tokenizer.add_special_tokens({"pad_token":"<pad>"})`를 사용하여 토큰을 추가하고 임베딩 크기도 확실히 조정해야 합니다. `model.config.pad_token_id`도 설정이 필요합니다. 모델의 `embed_tokens` 레이어는 `self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.config.padding_idx)`로 초기화되며, 패딩 토큰을 인코딩하는 것이 0(zero)를 출력하게 할 것인지 그래서 초기화가 추천될때 이를 통화시킬 것인지를 정하게 합니다. 
- 원본 체크포인트는 이 [변환 스크립트](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)를 이용해서 변환 가능합니다. 스크립트는 다음 명령어로 호출할 수 있습니다:
    
    ```bash
    python src/transformers/models/llama/convert_llama_weights_to_hf.py \
        --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path --llama_version 3
    ```

- 변환 후, 모델과 토크나이저는 다음을 통해 로드된다.

    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("/output/path")
    model = AutoModelForCausalLM.from_pretrained("/output/path")
    ```

    이 스크립트를 실행시키려면 모델 전체를 float16 정밀도로 호스팅할 수 있는 충분한 메인메모리가 필요하다는 점을 유의하세요. 가장 큰 버전이 여러 체크포인트로 나뉘어 있더라도, 각 체크포인트가 모델의 가중치 일부를 포함하고 있기 때문에 이를 모두 RAM에 로드해야 합니다. 75B 모델을 예로 들면 대략 145GB의 RAM이 필요합니다. 

- `attn_implementation="flash_attention_2"`를 통해서 플래시 어텐션2를 사용할 때, `from_pretrained` 클래스 메서드에 `dtype`를 전달하지 말고 자동 혼합 정밀도(Automatic Mixed-Precision) 학습을 사용하세요. `Trainer`를 사용할 때는 단순히 `fp16` 또는 `bf16`을 `True`로 설정하면 됩니다. 그렇지 않으면 반드시 `torch.autocast`를 사용해야 합니다. 플래시 어텐션은 `fp16`과 `bf16` 데이터 유형만 지원하기 때문입니다.

## 자료[[resources]]

[라마2](./llama2) 문서 페이지에서는 이미 수 많은 멋지고 유익한 자료들을 제공하고 있습니다. 이곳에 라마3에 대한 새로운 자료를 더해주실 컨트리뷰터들을 초대합니다! 🤗
