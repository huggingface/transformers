<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*이 모델은 2020-07-28에 출시되었으며 2021-03-30에 Hugging Face Transformers에 추가되었습니다.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white" >
    </div>
</div>

# BigBird[[bigbird]]

[BigBird](https://huggingface.co/papers/2007.14062)는 [BERT](./bert)의 512토큰과 달리 최대 4096토큰까지의 시퀀스 길이를 처리하도록 설계된 트랜스포머 모델입니다. 기존 트랜스포머들은 시퀀스 길이가 늘어날수록 어텐션 계산 비용이 급격히 증가하여 긴 입력 처리에 어려움을 겪습니다. BigBird는 희소 어텐션 메커니즘으로 이 문제를 해결하는데, 모든 토큰을 동시에 살펴보는 대신 로컬 어텐션, 랜덤 어텐션, 그리고 몇 개의 전역 토큰을 조합하여 전체 입력을 효율적으로 처리합니다. 이런 방식을 통해 계산 효율성을 유지하면서도 시퀀스 전체를 충분히 이해할 수 있게 됩니다. 따라서 BigBird는 질의응답, 요약, 유전체학 응용처럼 긴 문서를 다루는 작업에 특히 우수한 성능을 보입니다.

모든 원본 BigBird 체크포인트는 [Google](https://huggingface.co/google?search_models=bigbird) 조직에서 찾아볼 수 있습니다.

> [!TIP]
> 오른쪽 사이드바의 BigBird 모델들을 클릭하여 다양한 언어 작업에 BigBird를 적용하는 더 많은 예시를 확인해보세요.

아래 예시는 [`Pipeline`], [`AutoModel`], 그리고 명령줄에서 `[MASK]` 토큰을 예측하는 방법을 보여줍니다.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(
    task="fill-mask",
    model="google/bigbird-roberta-base",
    dtype=torch.float16,
    device=0
)
pipeline("Plants create [MASK] through a process known as photosynthesis.")
```

</hfoption>
<hfoption id="AutoModel">

```py
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google/bigbird-roberta-base",
)
model = AutoModelForMaskedLM.from_pretrained(
    "google/bigbird-roberta-base",
    dtype=torch.float16,
    device_map="auto",
)
inputs = tokenizer("Plants create [MASK] through a process known as photosynthesis.", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

masked_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
predicted_token_id = predictions[0, masked_index].argmax(dim=-1)
predicted_token = tokenizer.decode(predicted_token_id)

print(f"The predicted token is: {predicted_token}")
```

</hfoption>
<hfoption id="transformers CLI">

```bash
!echo -e "Plants create [MASK] through a process known as photosynthesis." | transformers-cli run --task fill-mask --model google/bigbird-roberta-base --device 0
```

</hfoption>
</hfoptions>

## 참고사항[[notes]]

- BigBird는 절대 위치 임베딩을 사용하므로 입력을 오른쪽에 패딩해야 합니다.
- BigBird는 `original_full`과 `block_sparse` 어텐션을 지원합니다. 입력 시퀀스 길이가 1024 미만인 경우에는 희소 패턴의 이점이 크지 않으므로 `original_full` 사용을 권장합니다.
- 현재 구현은 3블록 윈도우 크기와 2개의 전역 블록을 사용하며, ITC 구현만 지원하고 `num_random_blocks=0`은 지원하지 않습니다.
- 시퀀스 길이는 블록 크기로 나누어떨어져야 합니다.

## 리소스[[resources]]

- BigBird 어텐션 메커니즘의 자세한 작동 원리는 [BigBird](https://huggingface.co/blog/big-bird) 블로그 포스트를 참고하세요.

## BigBirdConfig[[bigbirdconfig]]

[[autodoc]] BigBirdConfig

## BigBirdTokenizer[[bigbirdtokenizer]]

[[autodoc]] BigBirdTokenizer
    - get_special_tokens_mask
    - save_vocabulary

## BigBirdTokenizerFast[[bigbirdtokenizerfast]]

[[autodoc]] BigBirdTokenizerFast

## BigBird 특정 출력[[bigbird-specific-outputs]]

[[autodoc]] models.big_bird.modeling_big_bird.BigBirdForPreTrainingOutput

## BigBirdModel[[bigbirdmodel]]

[[autodoc]] BigBirdModel
    - forward

## BigBirdForPreTraining[[bigbirdforpretraining]]

[[autodoc]] BigBirdForPreTraining
    - forward

## BigBirdForCausalLM[[bigbirdforcausallm]]

[[autodoc]] BigBirdForCausalLM
    - forward

## BigBirdForMaskedLM[[bigbirdformaskedlm]]

[[autodoc]] BigBirdForMaskedLM
    - forward

## BigBirdForSequenceClassification[[bigbirdforsequenceclassification]]

[[autodoc]] BigBirdForSequenceClassification
    - forward

## BigBirdForMultipleChoice[[bigbirdformultiplechoice]]

[[autodoc]] BigBirdForMultipleChoice
    - forward

## BigBirdForTokenClassification[[bigbirdfortokenclassification]]

[[autodoc]] BigBirdForTokenClassification
    - forward

## BigBirdForQuestionAnswering[[bigbirdforquestionanswering]]

[[autodoc]] BigBirdForQuestionAnswering
    - forward