<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# BioGPT [[biogpt]]

## 개요 [[overview]]

BioGPT는 Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung Poon, Tie-Yan Liu에 의해 [BioGPT: generative pre-trained transformer for biomedical text generation and mining](https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bbac409/6713511?guestAccessKey=a66d9b5d-4f83-4017-bb52-405815c907b9) 에서 제안된 모델입니다. BioGPT는 생물의학 텍스트 생성과 마이닝을 위해 도메인에 특화된 생성형 사전 학습 트랜스포머 언어 모델입니다. BioGPT는 트랜스포머 언어 모델 구조를 따르며, 1,500만 개의 PubMed 초록을 이용해 처음부터 학습되었습니다.

논문의 초록은 다음과 같습니다:

*생물의학 분야에서 사전 학습된 언어 모델은 일반 자연어 처리 분야에서의 성공에 영감을 받아 점점 더 많은 주목을 받고 있습니다. 일반 언어 분야에서 사전 학습된 언어 모델의 두 가지 주요 계통인 BERT(및 그 변형)와 GPT(및 그 변형) 중 첫 번째는 생물의학 분야에서 BioBERT와 PubMedBERT와 같이 광범위하게 연구되었습니다. 이들은 다양한 분류 기반의 생물의학 작업에서 큰 성공을 거두었지만, 생성 능력의 부족은 그들의 적용 범위를 제한했습니다. 본 논문에서는 대규모 생물의학 문헌을 사전 학습한 도메인 특화 생성형 트랜스포머 언어 모델인 BioGPT를 제안합니다. 우리는 6개의 생물의학 자연어 처리 작업에서 BioGPT를 평가한 결과, 대부분의 작업에서 이전 모델보다 우수한 성능을 보였습니다. 특히, BC5CDR, KD-DTI, DDI 엔드-투-엔드 관계 추출 작업에서 각각 44.98%, 38.42%, 40.76%의 F1 점수를 기록하였으며, PubMedQA에서 78.2%의 정확도를 달성해 새로운 기록을 세웠습니다. 또한 텍스트 생성에 대한 사례 연구는 생물의학 용어에 대한 유창한 설명을 생성하는 데 있어 BioGPT의 장점을 더욱 입증했습니다.*

이 모델은 [kamalkraj](https://huggingface.co/kamalkraj)에 의해 기여되었습니다. 원본 코드는 [여기](https://github.com/microsoft/BioGPT)에서 찾을 수 있습니다.

## 사용 팁 [[usage-tips]]

- BioGPT는 절대적 위치 임베딩(absolute position embedding)을 사용하므로, 입력을 왼쪽이 아닌 오른쪽에서 패딩하는 것이 권장됩니다.
- BioGPT는 인과적 언어 모델링(Casual Langague Modeling, CLM) 목표로 학습되었기 때문에, 다음 토큰을 예측하는 데 강력한 성능을 보입니다. 이 기능을 활용하여 BioGPT는 구문적으로 일관된 텍스트를 생성할 수 있으며, 예시 스크립트 `run_generation.py`에서 이를 확인할 수 있습니다.
- 이 모델은 `past_key_values`(PyTorch 용)를 입력으로 받을 수 있는데, 이는 이전에 계산된 키/값 어텐션 쌍입니다. 이 값을 사용하면 텍스트 생성 중 이미 계산된 값을 다시 계산하지 않도록 할 수 있습니다. PyTorch에서 `past_key_values` 인수는 BioGptForCausalLM.forward() 메소드에서 자세히 설명되어 있습니다.

### Scaled Dot Product Attention(SDPA) 사용 [[using-scaled-dot-product-attention-sdpa]]

PyTorch는 `torch.nn.functional`의 일부로 스케일된 점곱 어텐션(SDPA) 연산자를 기본적으로 포함합니다. 이 함수는 입력과 사용 중인 하드웨어에 따라 여러 구현을 적용할 수 있습니다. 자세한 내용은 [공식 문서](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 또는 [GPU 추론](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention) 페이지를 참조하세요.

`torch>=2.1.1`에서 구현이 가능한 경우 SDPA는 기본적으로 사용되며, `attn_implementation="sdpa"`를 `from_pretrained()`에서 설정하여 SDPA 사용을 명시적으로 요청할 수 있습니다.

```
from transformers import BioGptForCausalLM
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt", attn_implementation="sdpa", torch_dtype=torch.float16)
```

NVIDIA GeForce RTX 2060-8GB, PyTorch 2.3.1, Ubuntu 20.04 환경에서 `float16` 및 CausalLM 헤드가 있는 `microsoft/biogpt` 모델로 로컬 벤치마크를 수행한 결과, 훈련 중 다음과 같은 속도 향상을 확인했습니다.

최적의 속도 향상을 위해 모델을 반정밀도(예: `torch.float16` 또는 `torch.bfloat16`)로 로드하는 것이 좋습니다.

| num_training_steps | batch_size | seq_len | is cuda | Time per batch (eager - s) | Time per batch (sdpa - s) | Speedup (%) | Eager peak mem (MB) | sdpa peak mem (MB) | Mem saving (%) |
|--------------------|------------|---------|---------|----------------------------|---------------------------|-------------|---------------------|--------------------|----------------|
| 100                | 1          | 128     | False   | 0.038                      | 0.031                     | 21.301      | 1601.862            | 1601.497           | 0.023          |
| 100                | 1          | 256     | False   | 0.039                      | 0.034                     | 15.084      | 1624.944            | 1625.296           | -0.022         |
| 100                | 2          | 128     | False   | 0.039                      | 0.033                     | 16.820      | 1624.567            | 1625.296           | -0.045         |
| 100                | 2          | 256     | False   | 0.065                      | 0.059                     | 10.255      | 1672.164            | 1672.164           | 0.000          |
| 100                | 4          | 128     | False   | 0.062                      | 0.058                     | 6.998       | 1671.435            | 1672.164           | -0.044         |
| 100                | 4          | 256     | False   | 0.113                      | 0.100                     | 13.316      | 2350.179            | 1848.435           | 27.144         |
| 100                | 8          | 128     | False   | 0.107                      | 0.098                     | 9.883       | 2098.521            | 1848.435           | 13.530         |
| 100                | 8          | 256     | False   | 0.222                      | 0.196                     | 13.413      | 3989.980            | 2986.492           | 33.601         |

NVIDIA GeForce RTX 2060-8GB, PyTorch 2.3.1, Ubuntu 20.04 환경에서 `float16` 및 AutoModel 헤드가 있는 `microsoft/biogpt` 모델로 추론 중 다음과 같은 속도 향상을 확인했습니다.

| num_batches | batch_size | seq_len | is cuda | is half | use mask | Per token latency eager (ms) | Per token latency SDPA (ms) | Speedup (%) | Mem eager (MB) | Mem BT (MB) | Mem saved (%) |
|-------------|------------|---------|---------|---------|----------|------------------------------|-----------------------------|-------------|----------------|--------------|---------------|
| 50          | 1          | 64      | True    | True    | True     | 0.115                        | 0.098                       | 17.392      | 716.998        | 716.998      | 0.000         |
| 50          | 1          | 128     | True    | True    | True     | 0.115                        | 0.093                       | 24.640      | 730.916        | 730.916      | 0.000         |
| 50          | 2          | 64      | True    | True    | True     | 0.114                        | 0.096                       | 19.204      | 730.900        | 730.900      | 0.000         |
| 50          | 2          | 128     | True    | True    | True     | 0.117                        | 0.095                       | 23.529      | 759.262        | 759.262      | 0.000         |
| 50          | 4          | 64      | True    | True    | True     | 0.113                        | 0.096                       | 18.325      | 759.229        | 759.229      | 0.000         |
| 50          | 4          | 128     | True    | True    | True     | 0.186                        | 0.178                       | 4.289       | 816.478        | 816.478      | 0.000         |


## 리소스 [[resources]]

- [인과적 언어 모델링 작업 가이드](../tasks/language_modeling)

## BioGptConfig [[transformers.BioGptConfig]]

[[autodoc]] BioGptConfig


## BioGptTokenizer [[transformers.BioGptTokenizer]]

[[autodoc]] BioGptTokenizer
    - save_vocabulary


## BioGptModel [[transformers.BioGptModel]]

[[autodoc]] BioGptModel
    - forward


## BioGptForCausalLM [[transformers.BioGptForCausalLM]]

[[autodoc]] BioGptForCausalLM
    - forward


## BioGptForTokenClassification [[transformers.BioGptForTokenClassification]]

[[autodoc]] BioGptForTokenClassification
    - forward


## BioGptForSequenceClassification [[transformers.BioGptForSequenceClassification]]

[[autodoc]] BioGptForSequenceClassification
    - forward