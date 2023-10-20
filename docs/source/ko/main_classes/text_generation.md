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

# 생성!

각 프레임워크는 텍스트 생성을 위한 생성 메서드(Generate Method)를 가지고 있으며, 각 프레임워크의 GenerationMixin Class에 구현되어 있습니다. :


* PyTorch의 [`generate()`](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.generate)는 [`GenerationMixin`](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin)에 구현되어 있습니다. 
* TensorFlow의 [`generate()`](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.TFGenerationMixin.generate)는 [`TFGenerationMixin`](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.TFGenerationMixin)에 구현되어 있습니다. 
* Flax/JAX의 [`generate()`](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.FlaxGenerationMixin.generate)는 [`FlaxGenerationMixin`](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.FlaxGenerationMixin)에 구현되어 있습니다. 

선택한 프레임워크와 관계없이 [`GenerationConfig`](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationConfig) 클래스 인스턴스로 생성 메서드를 매개변수화 할 수 있습니다. 
생성 매개변수의 전체 목록을 보려면 [`GenerationConfig`](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationConfig)를 확인해주세요. 
나열된 매개변수들을 통해 생성 메서드의 동작을 제어할 수 있습니다.

모델의 생성 구성을 어떻게 검사하고, 기본값이 무엇인지, 매개변수를 즉석에서 어떻게 변경하며, 사용자 정의 생성 구성을 어떻게 생성하고 저장하는지 알아보려면 [텍스트 생성 전략 가이드](../generation_strategies)를 참조 부탁드립니다. 

이 가이드에는 토큰 스트리밍과 같 관련 기능을 사용하는 방법도 설명되어 있습니다.

# GenerationConfig

## 출력의 길이를 제어하는 매개변수들

**max_length** (int, Optional, 기본값 20) — 생성된 토큰의 최대 길이입니다. 입력 프롬프트의 길이 + ```max_new_tokens```에 해당합니다. ```max_new_tokens```가 설정되어 있으면 ```max_length```의 설정이 무시됩니다.
- **max_new_tokens** (int, Optional) — 프롬프트의 토큰 수를 무시하고 생성할 **최대** 토큰 수입니다.
- **min_length** (int, Optional, 기본값 0) — 생성될 시퀀스의 최소 길이입니다. 입력 프롬프트의 길이 + ```min_new_tokens```에 해당합니다. ```min_new_tokens```가 설정되어 있으면 ```min_length```가 무시됩니다.
- **min_new_tokens** (int, Optional) — 프롬프트의 토큰 수를 무시하고 생성할 **최소** 토큰 수입니다.
- **early_stopping** (bool 또는 문자열, Optional, 기본값 False) — ```beam-search```와 같 beam 기반 메서드의 정지 조건을 제어합니다. 다음 값을 수용합니다: True는 ```num_beams``` 완전 후보자가 생길 때 생성이 중지됩니다; False는 휴리스틱이 적용되며 더 나 후보자를 찾을 가능성이 매우 낮을 때 생성이 중지됩니다; "never"는 더 나 후보가 없을 때만 beam 검색 절차가 중지됩니다. (정규 beam 검색 알고리즘).
- **max_time** (float, Optional) — 계산이 초 단위로 실행되도록 허용하는 최대 시간입니다. 할당된 시간이 지나도 현재 패스의 생성 계속 진행됩니다.

## 사용되는 생성 전략을 제어하는 매개변수들

- **do_sample** (bool, Optional, 기본값 False) — 샘플링 사용에 대한 옵션; 사용하지 않는다면 그리디 디코딩을 사용합니다.
- **num_beams** (int, Optional, 기본값 1) — beam 검색을 위한 beam의 수입니다. 1 beam 검색이 없음을 의미합니다.
- **num_beam_groups** (int, Optional, 기본값 1) — beam 간의 다양성을 보장하기 위해 num_beams를 나눌 그룹의 수입니다. 자세한 내용 [이 논문](https://arxiv.org/pdf/1610.02424.pdf)을 참조하세요.
- **penalty_alpha** (float, Optional) — 대조적인 검색 디코딩에서 모델의 신뢰 수준과 퇴화 페널티를 균형있게 조절하는 값입니다.
- **use_cache** (bool, Optional, 기본값 True) — 모델이 디코딩 속도를 높이기 위해 과거의 마지막 key/values attentions(모델에 적용 가능한 경우)을 사용할지 여부입니다.

## 모델 출력 로짓의 조작을 위한 매개변수

 **temperature** (float, Optional, 기본값 1.0) — 다음 토큰 확률을 조절하는 데 사용되는 값입니다.
- **top_k** (int, Optional, 기본값 50) — top-k 필터링을 위해 유지할 확률이 가장 높 어휘 토큰의 수입니다.
- **top_p** (float, Optional, 기본값 1.0) — 1보다 작 실수로 설정하면, 확률의 합이 top_p 이상인 가장 확률이 높 토큰의 작 집합만 생성에 사용됩니다.
- **typical_p** (float, Optional, 기본값 1.0) — 지역적 일반성 이미 생성된 부분 텍스트를 기반으로 다음 대상 토큰의 예측 조건부 확률이 무작위 토큰의 예측 조건부 확률과 얼마나 유사한지를 측정합니다. 1보다 작 실수로 설정하면, 확률의 합이 typical_p 이상인 가장 지역적으로 일반적인 토큰의 작 집합만 생성에 사용됩니다. 자세한 내용 이 논문을 참조하세요.
- **epsilon_cutoff** (float, Optional, 기본값 0.0) — 0과 1 사이의 실수로 설정할 수 있으며, 조건부 확률이 설정값보다 큰 토큰만 샘플링됩니다. 논문에서는 모델의 크기에 따라 제안되는 값이 3e-4에서 9e-4의 범위에 있습니다. 자세한 내용 다음 논문 [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191)을 참조하세요.
- **eta_cutoff** (float, Optional, 기본값 0.0) — 에타 샘플링 지역적으로 일반적인 샘플링과 엡실론 샘플링의 혼합입니다. 0과 1 사이의 실수로 설정하면, 토큰 eta_cutoff보다 크거나 $sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits)))$보다 큰 경우에만 고려됩니다. 뒷 부분의 지수항($exp$) 직관적으로 eta_cutoff로 스케일링된 다음 토큰의 예 확률입니다. 논문에서는 모델의 크기에 따라 값을 범위로 제안하고 있는데요. 제안하는 범위는 3e-4에서 2e-3 입니다. 자세한 내용 다음 논문 [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191)을 참조하세요.
- **diversity_penalty** (float, Optional, 기본값 0.0) —  중복 생성으로 인한 bias를 관리하기 위해, beam이 특정 시간에 다른 그룹의 beam과 동일한 토큰을 생성하면 beam의 점수에서 이 설정값을 뺍니다. diversity_penalty는 그룹 beam 검색이 활성화되어 있을 때만 효과를 적용합니다.
- **repetition_penalty** (float, Optional, 기본값 1.0) — 반복 패널티의 매개변수입니다. 1.0 패널티가 없음을 의미합니다. 자세한 내용 [이 논문](https://arxiv.org/pdf/1909.05858.pdf)을 참조하세요.
- **encoder_repetition_penalty** (float, Optional, 기본값 1.0) — encoder_repetition_penalty의 매개변수입니다. 원본 입력에 없는 시퀀스에 대한 지수 패널티입니다. 1.0 패널티가 없음을 의미합니다.
- **length_penalty** (float, Optional, 기본값 1.0) — beam 기반 생성과 함께 사용되는 길이에 대한 지수 패널티입니다. 시퀀스 길이의 지수로 적용되며, 그 결과로 시퀀스의 점수를 나눕니다. 점수가 시퀀스의 로그 가능성(즉, 부정적)인 경우, length_penalty > 0.0 더 긴 시퀀스를 만들도록 촉진시키며, length_penalty < 0.0 더 짧 시퀀스를 만들도록 촉진시킵니다.
- **no_repeat_ngram_size** (int, Optional, 기본값 0) — 해당 변수를 0보다 큰 정수로 설정하면, 해당 설정값과 동일한 크기의 모든 ngram 한 번만 나타날 수 있습니다.
- **bad_words_ids**(List[List[int]], Optional) — 생성하면 안되는 토큰 아이디의 목록입니다. 자세한 문서와 예제는 [NoBadWordsLogitsProcessor](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.NoBadWordsLogitsProcessor)를 참조하세요.
- **force_words_ids**(List[List[int]] 또는 List[List[List[int]]], Optional) — 생성해야 하는 토큰 아이디의 목록입니다. bad_words_ids와 반대의 개념이지만, 취급되는 형태는 동일합니다. List[List[int]]로 입력할 수 있고, 생성에 포함되어야 할 단순한 단어 목록이 형태로 사용됩니다. 차원이 1단계 더 깊 List[List[List[int]]]의 형태로 주어지면, 각 단어의 다른 형태를 허용할 수 있는 분리 제약을 트리거합니다.
- **renormalize_logits** (bool, Optional, 기본값 False) — 모든 로짓 프로세서나 워퍼(사용자 정의 포함)를 적용한 후 로짓을 재정규화할지 여부입니다. 로짓의 점수가 정규화되었다고 가정하는 검색 알고리즘을 사용하기 때문에 이 플래그를 True로 설정하는 것을 매우 권장합니다. 그러나 일부 로짓 프로세서나 워퍼는 해당 정규화를 깨뜨릴 수 있습니다.
- **constraints** (List[Constraint], Optional) — 생성 결과가 가능한한 합리적인 방법으로 특정 토큰의 사용을 포함하도록 보장하기 위해 생성에 추가할 수 있는 사용자 정의를 설정합니다.
- **forced_bos_token_id** (int, Optional, 기본값 model.config.forced_bos_token_id) — decoder_start_token_id 직후 생성된 토큰을 강제로 설정할 토큰의 아이디입니다. 타겟 언어 토큰이어야 하는 mBART와 같 다국어 모델에 유용합니다.
- **forced_eos_token_id** (Union[int, List[int]], Optional, 기본값 model.config.forced_eos_token_id) — max_length에 도달할 때 마지막으로 생성될 토큰의 아이디를 강제로 설정합니다. Optional으로 여러 종료 토큰을 설정하기 위해 List를 사용합니다.
- **remove_invalid_values** (bool, Optional, 기본값 model.config.remove_invalid_values) — 모델의 nan 및 inf 출력을 최대한 제거하여 생성 방법이 충돌하는 것을 방지 하는 것을 선택합니다. remove_invalid_values를 사용하면 생성이 느려질 수 있습니다.
- **exponential_decay_length_penalty** (tuple(int, float), Optional) — 이 튜플 일정량의 토큰이 생성된 후 지수함수의 증가 폭 만큼의 길이 패널티를 추가합니다. 튜플 (start_index, decay_factor)로 구성되며, start_index는 패널티가 시작되는 지점을 나타내고 decay_factor는 지수 감소의 인자를 나타냅니다.
- **suppress_tokens** (List[int], Optional) — 생성 시 억제될 토큰의 목록입니다. SupressTokens 로짓 프로세서는 그들의 로그 확률을 $-inf$로 설정하여 선택되지 않도록 합니다.
- **begin_suppress_tokens** (List[int], Optional) — 생성 시작 시 억제될 토큰의 목록입니다. SupressBeginTokens 로짓 프로세서는 그들의 로그 확률을 $-inf$로 설정하여 선택되지 않도록 합니다.
- **forced_decoder_ids** (List[List[int]], Optional) — 샘플링 전에 강제로 설정될 토큰 인덱스에 대한 생성 인덱스의 정수 쌍 목록입니다. 예를 들어, [[1, 123]] 두 번째 생성된 토큰이 항상 인덱스 123의 토큰이 될 것임을 의미합니다.
- **sequence_bias** (Dict[Tuple[int], float], Optional) — 입력된 Dictionary안 토큰을 해당 토큰과 매핑된 편향성에 의해 시퀀스 선택의 확률을 결정합니다. 긍정적인 편향 시퀀스가 선택될 확률을 높이고, 부정적인 편향 반대 효과를 냅니다. 자세한 내용 [SequenceBiasLogitsProcessor](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.SequenceBiasLogitsProcessor)를 참조하십시오.
- **guidance_scale** (float, Optional) — 분류의 기준이 없는 지침(CFG)을 위한 지침 스케일입니다. guidance_scale > 1로 설정하면 CFG가 활성화됩니다. 더 높 지침 스케일 모델이 입력 프롬프트와 더 밀접하게 연결된 샘플을 생성하도록 권장하며, 보통 좋지 않 품질에 영향을 줍니다.
- **low_memory** (bool, Optional) — 대조 검색을 사용할 때 최대 메모리를 줄이기 위해 순차적인 topk로 전환합니다.

## ```generate```의 출력 변수를 정의하는 매개변수들

- **num_return_sequences**(int, Optional, 기본값 1) — 배치의 각 요소에 대해 독립적으로 계산된 반환 시퀀스의 수를 나타냅니다.
- **output_attentions** (bool, Optional, 기본값 False) — 모든 attention 레이어의 attention 텐서를 출력 할지를 설정할 수 있습니다.
- **output_hidden_states** (bool, Optional, 기본값 False) — 모든 레이어의 숨겨진 상태를 반환할지 를 설정할 수 있습니다.
- **output_scores** (bool, Optional, 기본값 False) — 예측 점수를 반환할지 설정할 수 있습니다.
- **return_dict_in_generate** (bool, Optional, 기본값 False) — 일반 튜플 대신 ModelOutput을 반환할지 설정할 수 있습니.

## 생성 시 사용할 수 있는 특수 토큰들

- **pad_token_id** (int, Optional) — 패딩 토큰의 ID입니다.
- **bos_token_id** (int, Optional) — 시퀀스 시작 토큰의 ID입니다.
- **eos_token_id** (Union[int, List[int]], Optional) — 시퀀스 종료 토큰의 ID입니다. Optional으로 여러 종료 토큰을 설정하기 위해 List를 사용합니다.

## 인코더-디코더 모델에만 해당되는 생성 매개변수

- **encoder_no_repeat_ngram_size** (int, Optional, 기본값 0) — 기본값인 0보다 크게 설정되면, encoder_input_ids에 나타나는 해당 크기의 모든 ngram decoder_input_ids에서 나타나지 않게 설정합니다.
- **decoder_start_token_id** (int, Optional) — 인코더-디코더 모델이 bos와 다른 토큰으로 디코딩을 시작하면 그 토큰의 ID를 나타냅니다.

## Wild card

생성 작업을 위한 설정을 할 수 있는 Class입니다. generate 호출 텍스트 디코더, 텍스트 대 텍스트, 음성 대 텍스트 및 비전 대 텍스트 모델에 대한 다음 생성 방법을 설정 및 지원합니다:

- num_beams=1 및 do_sample=False인 경우 [greedy_search()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.greedy_search)를 호출하여 그리디 디코딩 지원.
- penalty_alpha>0. 및 top_k>1인 경우 [contrastive_search()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.contrastive_search)를 호출하여 대조 검색(contrastive search) 지원.
- num_beams=1 및 do_sample=True인 경우 [sample()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.sample)를 호출하여 다항 샘플링(multinomial sampling) 지원.
- num_beams>1 및 do_sample=False인 경우 [beam_search()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_search)를 호출하여 beam 검색 디코딩 지원.
- num_beams>1 및 do_sample=True인 경우 [beam_sample()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_sample)를 호출하여 beam 검색 다항 샘플링 지원
- num_beams>1 및 num_beam_groups>1인 경우 [group_beam_search()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.group_beam_search)를 호출하여 다양한(diverse) beam 검색 디코딩 지원.
- constraints!=None 또는 force_words_ids!=None인 경우 [constrained_beam_search()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.constrained_beam_search)를 호출하여 제약 조건이 있는 beam 검색 디코딩 지원.
- assistant_model이 .generate()에 전달되면 [assisted_decoding()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.assisted_decoding)를 호출하여 어시스트 디코딩 지원.

위의 나열된 메서드를 직접 호출할 필요는 없습니다! 맞춤 매개변수 값을 ‘.generate()‘에 입력하여 사용할 수 있습니다. 디코딩 전략에 대해 자세히 알아보려면 [텍스트 생성 전략 가이드](https://huggingface.co/docs/transformers/v4.34.0/en/generation_strategies)를 참조하세요.

> ## from_pretrained
> 
> ``` python 
> from_pretrained ( pretrained_model_name: typing.Union[str, os.PathLike], 
>                   config_file_name: typing.Union[str, os.PathLike, NoneType] = None,
>                   cache_dir: typing.Union[str, os.PathLike, NoneType] = None,
>                   force_download: bool = False,
>                   local_files_only: bool = False,
>                   token: typing.Union[bool, str, NoneType] = None,
>                   revision: str = 'main',
>                   **kwargs )
> ```
> ### Parameters ---
> **pretrained_model_name** (str or os.PathLike) - 두 가지 파라매터 모두 사용 가능.
> * 문자열, huggingface.co의 모델 저장소에 호스팅된 사전 훈련된 모델 구성의 모델 ID입니다. 유효한 모델 ID는 bert-base-uncased와 같이 최상위에 위치할 수 있거나 dbmdz/bert-base-german-cased와 같이 사용자 또는 조직 이름 아래에 네임스페이스될 수 있습니다.
> * [save_pretrained()](#save_pretrained) 메서드를 사용하여 저장된 구성 파일이 포함된 디렉토리의 경로,  예: ./my_model_directory/.
> 
> **config_file_name** (문자열 또는 os.PathLike, Optional, 기본값 "generation_config.json") — pretrained_model_name에서 로드할 생성 구성 JSON 파일의 이름입니다.
> 
> **cache_dir** (문자열 또는 os.PathLike, Optional) — 표준 캐시를 사용하지 않아야 하는 경우 다운로드된 사전 훈련된 모델 구성이 캐시될 디렉토리의 경로입니다
> 
> **force_download** (부울, Optional, 기본값 False) — 구성 파일을 (다시) 다운로드하고 존재하는 경우 캐시된 버전을 덮어쓸지 여부입니다.
> 
> **resume_download** (부울, Optional, 기본값 False) — 불완전하게 수신된 파일을 삭제할지 여부입니다. 해당 파일이 존재하면 다운로드를 재개합니다.
> 
> **proxies** (Dict[str, str], Optional) — 프로토콜 또는 엔드포인트별로 사용할 프록시 서버의 사전, 예: {'http': 'foo.bar:3128', '[http://hostname](http://hostname/)': 'foo.bar:4012'}. 프록시는 각 요청에 사용됩니다.
> 
> **token** (문자열 또는 부울, Optional) — 원격 파일에 대한 HTTP 베어러 인증으로 사용할 토큰입니다. True 또는 지정되지 않 경우 huggingface-cli login을 실행할 때 생성된 토큰을 사용합니다(~/.huggingface에 저장됨).
> 
> **revision** (문자열, Optional, 기본값 "main") — 사용할 특정 모델 버전입니다. huggingface.co에서 모델 및 기타 아티팩트를 저장하기 위해 git 기반 시스템을 사용하므로 revision git에서 허용하는 모든 식별자가 될 수 있습니다.
> 
>> ``` Hub에서 pull request를 테스트하려면 `revision=“refs/pr/“`을 전달할 수 있습니다.```
> 
> **return_unused_kwargs** (부울, Optional, 기본값 False) — False인 경우 이 함수는 최종 구성 객체만 반환합니다. True인 경우 이 함수는 Tuple(config, unused_kwargs)를 반환하며, unused_kwargs는 구성 속성이 아닌 키의 키/값 쌍으로 구성된 사전입니다. 즉, config를 업데이트하는 데 사용되지 않고 그렇지 않으면 무시되는 kwargs의 부분입니다.
> 
> **subfolder** (문자열, Optional, 기본값 "") — 관련 파일이 huggingface.co의 모델 저장소의 하위 폴더 안에 있을 경우 여기에 폴더 이름을 지정할 수 있습니다.
> 
> **kwargs** (Dict[str, Any], Optional) — kwargs의 키가 구성 속성인 모든 값 로드된 값에 대한 오버라이드로 사용됩니다. 구성 속성이 아닌 키에 대한 키/값 쌍의 동작 return_unused_kwargs 키워드 매개변수에 의해 제어됩니다.
>
> ### Returns -> *GenerationConfig* ---
>
> 사전 훈련된 모델로 부터 인스턴스화된 Configuration 객체 입니다.
>
> Configuration 구성 파일에허 *GenerationConfig*를 인스턴스화 하는 예제는 아래에서 확인하실 수 있습니다.
>
> Examples:
> ```python
> from transformers import GenerationConfig
>
> # Download configuration from huggingface.co and cache.
> generation_config = GenerationConfig.from_pretrained("gpt2")
>
> # E.g. config was saved using *save_pretrained('./test/saved_model/')*
> generation_config.save_pretrained("./test/saved_model/")
> generation_config = GenerationConfig.from_pretrained("./test/saved_model/")
>
> # You can also specify configuration names to your generation configuration file
> generation_config.save_pretrained("./test/saved_model/",config_file_name="my_configuration.json")
> generation_config = GenerationConfig.from_pretrained("./test/saved_model/", "my_configuration.json")
>
> # If you'd like to try a minor variation to an existing configuration, you can also pass generation
> # arguments to `.from_pretrained()`. Be mindful that typos and unused arguments will be ignored
>generation_config, unused_kwargs = GenerationConfig.from_pretrained(
>    "gpt2", top_k=1, foo=False, do_sample=True, return_unused_kwargs=True
>)
>generation_config.top_k
>
>unused_kwargs
>{'foo' : False}
> ```

> ## from_model_config
> ```python 
> from_model_config ( model_config: PretrainedConfig )
> ```
> ### Parameters ---
> * **model_config** (PretrainedConfig) — 
생성 구성을 인스턴스화하는 데 사용되는 모델 구성입니다.
> ### Returns *GenerationConfig* ---
> 이러한 매개변수에서 인스턴스화된 구성 객체입니다.
>
> PretrainedConfig에서 [GenerationConfig](#generationconfig)를 인스턴스화합니다. 이 함수는 생성 매개변수를 포함할 수 있는 레거시 PretrainedConfig 객체를 독립 실행형 GenerationConfig로 변환하는 데 유용합니다.

> ## save_pretrained
>
> ``` python
> save_pretrained ( save_directory: typing.Union[str, os.PathLike], 
>                   config_file_name: typing.Union[str, os.PathLike, NoneType] = None,
>                   push_to_hub: bool = False**kwargs ) 
> ```
>
>  ### Parameters ---
> * **save_directory** (str or os.PathLike) — 구성 JSON 파일이 저장될 디렉터리(존재하지 않으면 생성)를 지정합니다.
> 
> * **config_file_name** (str or os.PathLike, Optional, 기본값 "generation_config.json") — save_directory에 저장될 JSON 파일의 이름입니다.
> 
> * **push_to_hub** (bool, Optional, 기본값 False) — 저장한 후에 모델을 Hugging Face 모델 허브에 푸시할지 여부입니다. repo_id로 푸시하려는 저장소를 지정할 수 있습니다(네임스페이스에서 save_directory의 이름으로 기본 설정됩니다).
> 
> * **kwargs** (Dict[str, Any], Optional) — push_to_hub() 메소드로 전달되는 추가 키워드 인수입니다.
> 
> save_directory 디렉터리에 생성 구성 객체를 저장하여, [from_pretrained()](#from_pretrained) 클래스 메소드를 사용하여 다시 로드할 수 있게 합니다.



# GenerationMixin  
PreTrainedModel에 혼합하여 사용될 자동-회귀 텍스트 생성을 위한 모든 함수를 포함한 클래스입니다.

이 클래스는 'generate()'를 제공하며, 이를 사용하여 아래의 함수를 사용할 수 있습니다:

- num_beams=1 및 do_sample=False인 경우 [greedy_search()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.greedy_search)를 호출하여 그리디 디코딩 지원.
- penalty_alpha>0. 및 top_k>1인 경우 [contrastive_search()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.contrastive_search)를 호출하여 대조 검색(contrastive search) 지원.
- num_beams=1 및 do_sample=True인 경우 [sample()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.sample)를 호출하여 다항 샘플링(multinomial sampling) 지원.
- num_beams>1 및 do_sample=False인 경우 [beam_search()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_search)를 호출하여 beam 검색 디코딩 지원.
- num_beams>1 및 do_sample=True인 경우 [beam_sample()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.beam_sample)를 호출하여 beam 검색 다항 샘플링 지원
- num_beams>1 및 num_beam_groups>1인 경우 [group_beam_search()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.group_beam_search)를 호출하여 다양한(diverse) beam 검색 디코딩 지원.
- constraints!=None 또는 force_words_ids!=None인 경우 [constrained_beam_search()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.constrained_beam_search)를 호출하여 제약 조건이 있는 beam 검색 디코딩 지원.
- assistant_model이 .generate()에 전달되면 [assisted_decoding()](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/text_generation#transformers.GenerationMixin.assisted_decoding)를 호출하여 어시스트 디코딩 지원.

직접 위의 메서드를 호출할 필요는 없습니다. 대신 ‘generate’에 사용자 정의 매개변수 값을 전달하는 것으로 충분합니다. 디코딩 전략에 대해 더 알고 싶다면 [텍스트 생성 전략 가이드](https://huggingface.co/docs/transformers/v4.34.0/en/generation_strategies)를 참조하세요.

> ## generate
> ``` python
> generate ( inputs: typing.Optional[torch.Tensor] = None,
>            generation_config: typing.Optional[transformers.generation.configuration_utils.GenerationConfig] = None,
>            logits_processor: typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
>            stopping_criteria: typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None,
>            prefix_allowed_tokens_fn: typing.Union[typing.Callable[[int, torch.Tensor], typing.List[int]], NoneType] = None,
>            synced_gpus: typing.Optional[bool] = None,
>            assistant_model: typing.Optional[ForwardRef('PreTrainedModel')] = None,
>            streamer: typing.Optional[ForwardRef('BaseStreamer')] = None,
>            negative_prompt_ids: typing.Optional[torch.Tensor] = None,
>            negative_prompt_attention_mask: typing.Optional[torch.Tensor] = None,
>            **kwargs )
> ```
> ### Parameters ---
>
> * **inputs** (torch.Tensor): 생성 또는 인코더의 모델 입력으로 사용되는 시퀀스. 제공되지 않으면 bos_token_id와 배치 크기 1로 초기화합니다. 디코더 전용 모델의 경우 input_ids 형식이어야 합니다.
> 
> * **generation_config** (~generation.GenerationConfig, Optional) — 생성 호출에 사용될 기본 매개변수로의 생성 구성입니다. **kwargs가 제공되고 generation_config의 속성과 일치하면 덮어씁니다.
> 
> * **logits_processor** (LogitsProcessorList, Optional) — 기본 logits 프로세서를 보완하는 사용자 정의 logits 프로세서.
> 
> * **stopping_criteria** (StoppingCriteriaList, Optional) — 기본 중지 기준을 보완하는 사용자 정의 중지 기준.
> 
> * **prefix_allowed_tokens_fn** (Callable[[int, torch.Tensor], List[int]], Optional) — 제공되면, 이 함수는 각 단계에서 beam 검색을 허용된 토큰에만 제한합니다.
> 
> * **synced_gpus** (bool, Optional) — max_length까지 while 루프를 계속 실행할지 여부.
> 
> * **assistant_model** (PreTrainedModel, Optional) — 생성을 가속화하는 데 사용할 수 있는 보조 모델.
> 
> * **streamer** (BaseStreamer, Optional) — 생성된 시퀀스를 스트리밍하는 데 사용될 스트리머 객체.
> 
> * **negative_prompt_ids** (torch.LongTensor, Optional) — 일부 프로세서에 필요한 부정적인 프롬프트.
> 
> * **negative_prompt_attention_mask** (torch.LongTensor, Optional) — negative_prompt_ids에 대한 주의 마스크.
> 
> * **kwargs** (Dict[str, Any], Optional) — generate_config의 적절한 매개변수화 및/또는 모델 전용 kwargs.
> 
>
> ### Returns -> [ModelOutput](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) or **torch.LongTensor**
>
> 모델 출력([ModelOutput](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput)) (if return_dict_in_generate=True or when config.return_dict_in_generate=True) or a torch.FloatTensor.
>
> 해당 모델이 인코더-디코더 모델(model.config.is_encode_decode=False)이 아닐 경우, 가능한 모델 출력 유형 아래와 같습니다.
> * [GreedySearchDecodeOnlyOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.GreedySearchDecoderOnlyOutput),
> * [SampleDecodeOnlyOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.SampleDecoderOnlyOutput),
> * [BeamSearchDecodeOnlyOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.BeamSearchDecoderOnlyOutput),
> * [BeamSampleDecodeOnlyOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.BeamSampleDecoderOnlyOutput)
>
> 해당 모델이 인코더-디코더 모델(model.config.is_encode_decode=True)일 경우, 가능한 모델 출력 유형 아래와 같습니다.
> 
> * [GreedySearchEncoderDecoderOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.GreedySearchEncoderDecoderOutput),
> * [SampleEncoderDecoderOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.SampleEncoderDecoderOutput),
> * [BeamSearchEncoderDecoderOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.BeamSearchEncoderDecoderOutput),
> * [BeamSampleEncoderDecoderOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.BeamSampleEncoderDecoderOutput)
>
> 언어 모델링 헤드를 가진 모델을 위한 토큰 ID 시퀀스를 생성합니다.
>
> ## compute_transition_scores
> ``` python
> compute_transition_scores ( sequences: Tensorscores: typing.Tuple[torch.Tensor],
>                             beam_indices: typing.Optional[torch.Tensor] = None,
>                             normalize_logits: bool = False )
> ```
> ### Parameters ---
>
> * **sequences** (torch.LongTensor)  — 생성된 시퀀스입니다. 두 번째 차원 (시퀀스 길이) max_length와 동일하거나 eos_token_id로 인해 모든 배치가 일찍 완료되었을 경우 더 짧습니다.
> 
> * **scores** (tuple(torch.FloatTensor)) — 각 생성 단계에서 어휘 토큰에 대한 전환 점수입니다. Beam 전환 점수는 사전에 생성된 토큰의 Log Softmax를 기반한 토큰의 로그 확률로 구성되며, torch.FloatTensor의 튜플에는 최대 max_new_tokens개의 요소를 포함합니다. 각 텐서는 (batch_sizenum_beams, config.vocab_size)의 형태를 가집니다.
>
> * **beam_indices** (torch.LongTensor, Optional) — 각 생성 단계에서 생성된 토큰 id의 beam 인덱스입니다. 구조 형태가 (batch_sizenum_return_sequences, sequence_length)인 torch.LongTensor입니다. generate-time안에 num_beams>1 인 경우에만 필요합니다.
>
> * **normalize_logits** (bool, Optional, 기본값 False) — 로짓을 정규화할지를 선택할 수 있습니다.(과거 구조 상의 이유로 로짓이 정규화되지 않을 수 있습니다).
>
> 
> ### Returns -> **torch.Tensor**
> (batch_size*num_return_sequences, sequence_length) 형태를 가지는 torch.Tensor를 반환하며, 전환 점수(logits)를 포함합니다.
>
> 생성 점수(및 beam 검색이 사용된 경우 beam 인덱스)를 기반으로 시퀀스의 전환 점수를 계산합니다. 이와 같 방법 생성 시간에 선택된 토큰의 점수를 빠르게 얻기 위해 사용할 수 있는 편리한 방법 중 하나입니다.
>
> Examples:
> ```python
> from transformers import GPT2Tokenizer, AutoModelForCausalLM
> import numpy as np
>
> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
> model = AutoModelForCausalLM.from_pretrained("gpt2")
> tokenizer.pad_token_id = tokenizer.eos_token_id
> inputs = tokenizer(["Today is"], return_tensors="pt")
>
> # Example 1: Print the scores for each token generated with Greedy Search
> outputs = model.generate(**inputs, max_new_tokens=5, > return_dict_in_generate=True, output_scores=True)
> transition_scores = model.compute_transition_scores(
>     outputs.sequences, outputs.scores, normalize_logits=True
> )
> # input_length is the length of the input prompt for decoder-only models, like > the GPT family, and 1 for
> # encoder-decoder models, like BART or T5.
> input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
> generated_tokens = outputs.sequences[:, input_length:]
> for tok, score in zip(generated_tokens[0], transition_scores[0]):
>     # | token | token string | logits | probability
>     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
> 
> # Example 2: Reconstruct the sequence scores from Beam Search
> outputs = model.generate(
>     **inputs,
>     max_new_tokens=5,
>     num_beams=4,
>     num_return_sequences=4,
>     return_dict_in_generate=True,
>     output_scores=True,
> )
> transition_scores = model.compute_transition_scores(
>     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
> )
> # If you sum the generated tokens' scores and apply the length penalty, you'll > get the sequence scores.
> # Tip: recomputing the scores is only guaranteed to match with > `normalize_logits=False`. Depending on the
> # use case, you might want to recompute it with `normalize_logits=True`.
> output_length = input_length + np.sum(transition_scores.numpy() < 0, axis=1)
> length_penalty = model.generation_config.length_penalty
> reconstructed_scores = transition_scores.sum(axis=1) / > (output_length**length_penalty)
> print(np.allclose(outputs.sequences_scores, reconstructed_scores))
> ```
>
>
> ## greedy_search
>
> ```python
> greedy_search ( input_ids: LongTensor,
>                 logits_processor: typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
>                 stopping_criteria: typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None, 
>                 max_length: typing.Optional[int] = None, 
>                 pad_token_id: typing.Optional[int] = None, 
>                 eos_token_id: typing.Union[int, typing.List[int], NoneType] = None, 
>                 output_attentions: typing.Optional[bool] = None, 
>                 output_hidden_states: typing.Optional[bool] = None, 
>                 output_scores: typing.Optional[bool] = None, 
>                 return_dict_in_generate: typing.Optional[bool] = None, 
>                 synced_gpus: bool = False, 
>                 streamer: typing.Optional[ForwardRef('BaseStreamer')] = None, 
>                 **model_kwargs )
> ```
>
> ### Parameters ---
> 
> * **input_ids** (torch.LongTensor 형태의 (batch_size, sequence_length)) — 생성을 위한 프롬프트로 사용되는 시퀀스 입니다.
> 
> * **logits_processor** (LogitsProcessorList, Optional) — [LogitsProcessorList](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessorList)의 인스턴스 입니다. LogitsProcessorList는 [LogitsProcessor](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor)에서 파생된 클래스의 인스턴스 목록으로 각 생성 단계에서 언어 모델링 헤드의 예측 점수를 수정하는 데 사용됩니다.
> 
> * **stopping_criteria** (StoppingCriteriaList, Optional) — StoppingCriteriaList의 인스턴스 입니다. [StoppingCriteriaList](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteriaList)는 [StoppingCriteria](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteria)에서 파생된 클래스의 인스턴스 목록으로 생성 루프가 중단되어야 하는지 알리기 위해 사용됩니다.
> 
> * **max_length** (int, Optional, 기본값 20) — **현재는 사용되지 않습니다.** 생성된 토큰의 수를 제한하기 위해 logits_processor 또는 stopping_criteria를 직접 사용해야 합니다.
> 
> * **pad_token_id** (int, Optional) — 패딩 토큰의 ID 입니다.
> 
> * **eos_token_id** (Union[int, List[int]], Optional) — 시퀀스 종료 토큰의 ID 입니다. 여러 시퀀스 종료 토큰을 설정하기 위해 목록을 Optional로 사용합니다.
> 
> * **output_attentions** (bool, Optional, 기본값 False) — 모든 주의 계층의 주의 텐서를 반환할지 여부를 지정합니다. 자세한 내용은 ```attentions```를 참고하세요.
> 
> * **output_hidden_states** (bool, Optional, 기본값 False) — 모든 계층의 숨겨진 상태를 반환할지 여부를 지정합니다. 자세한 내용은 ```hidden_states```를 참고하세요.
> 
> * **output_scores** (bool, Optional, 기본값 False) — 예측 점수를 반환할지 여부를 지정합니다. 자세한 내용은 ```scores```를 참고하세요.
> 
> * **return_dict_in_generate** (bool, Optional, 기본값 False) — 일반 튜플 대신 [ModelOutput](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput)을 반환할지 여부를 지정합니다.
> 
> * **synced_gpus** (bool, Optional, 기본값 False) — max_length까지 while 루프를 계속 실행할지 여부를 지정합니다. (ZeRO 단계 3에 필요함)
> 
> * **streamer** (BaseStreamer, Optional) — 생성된 시퀀스를 스트림할때 사용될 Streamer 객체를 설정합니다. 생성된 토큰인 streamer.put(token_ids)를 통해 전달되며, 스트리머는 추가 처리를 담당합니다.
> 
> * **model_kwargs** — 추가 모델 특정 키워드 인수는 모델의 forward 함수로 전달됩니다. 모델이 인코더-디코더 모델이면 kwargs는 encoder_outputs를 포함해야 합니다.
> 
> greedy 디코딩을 사용하여 언어 모델링 헤드가 있는 모델의 토큰 ID 시퀀스를 생성하며, 텍스트 디코더, 텍스트-텍스트, 음성-텍스트 및 시각-텍스트 모델에 사용할 수 있습니다.
>
> > 대부분의 경우에, 당신 greedy_search()를 직접 호출할 필요가 없습니다. 주로 generate()를 사용하게 됩니다. 생성 전략과 코드 예제에 대한 개요는 [이 가이드](https://huggingface.co/docs/transformers/v4.34.0/en/generation_strategies)를 확인하세요.
>
> Examples:
> ```python
> from transformers import (
>     AutoTokenizer,
>     AutoModelForCausalLM,
>     LogitsProcessorList,
>     MinLengthLogitsProcessor,
>     StoppingCriteriaList,
>     MaxLengthCriteria,
> )
> 
> tokenizer = AutoTokenizer.from_pretrained("gpt2")
> model = AutoModelForCausalLM.from_pretrained("gpt2")
> 
> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
> model.generation_config.pad_token_id = model.generation_config.eos_token_id
> 
> input_prompt = "It might be possible to"
> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
> 
> # instantiate logits processors
> logits_processor = LogitsProcessorList(
>     [
>         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
>     ]
> )
> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
> 
> outputs = model.greedy_search(
>     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
> )
> 
> tokenizer.batch_decode(outputs, skip_special_tokens=True)> 
> ```
>
> ## sample
> ```python
> sample ( input_ids: LongTensor,
>          logits_processor: typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None, 
>          stopping_criteria: typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None, 
>          logits_warper: typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None, 
>          max_length: typing.Optional[int] = None, 
>          pad_token_id: typing.Optional[int] = None, 
>          eos_token_id: typing.Union[int, typing.List[int], NoneType] = None, 
>          output_attentions: typing.Optional[bool] = None, 
>          output_hidden_states: typing.Optional[bool] = None, 
>          output_scores: typing.Optional[bool] = None, 
>          return_dict_in_generate: typing.Optional[bool] = None, 
>          synced_gpus: bool = Falsestreamer: typing.Optional[ForwardRef('BaseStreamer')] = None, 
>          **model_kwargs)
> ```
>
> ### Parameters
>
> * **input_ids** (torch.LongTensor의 모양 (batch_size, sequence_length)) — 생성을 위한 프롬프트(prompt)로 사용되는 시퀀스입니다.
> 
> * **logits_processor** (LogitsProcessorList, Optional) — [LogitsProcessorList](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessorList)의 인스턴스 입니다. LogitsProcessorList는 [LogitsProcessor](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor)에서 파생된 클래스의 인스턴스 목록으로 각 생성 단계에서 언어 모델링 헤드의 예측 점수를 수정하는 데 사용됩니다.
> 
> * **stopping_criteria** (StoppingCriteriaList, Optional) — StoppingCriteriaList의 인스턴스 입니다. [StoppingCriteriaList](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteriaList)는 [StoppingCriteria](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteria)에서 파생된 클래스의 인스턴스 목록으로 생성 루프가 중단되어야 하는지 알리기 위해 사용됩니다.
>
> * **logits_warper** (LogitsProcessorList, Optional) — LogitsProcessorList의 인스턴스입니다. LogitsProcessorList는 [LogitsWarper](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsWarper)에서 파생된 클래스의 인스턴스 목록으로 각 생성 단계에서 다항 샘플링 전에 언어 모델링 헤드의 예측 점수 분포를 왜곡하는 데 사용됩니다.
> 
> * **max_length** (int, Optional, 기본값 20) — **현재는 사용되지 않습니다.** 생성된 토큰의 수를 제한하기 위해 logits_processor 또는 stopping_criteria를 직접 사용해야 합니다.
> 
> * **pad_token_id** (int, Optional) — 패딩 토큰의 ID 입니다.
> 
> * **eos_token_id** (Union[int, List[int]], Optional) — 시퀀스 종료 토큰의 ID 입니다. 여러 시퀀스 종료 토큰을 설정하기 위해 목록을 Optional로 사용합니다.
> 
> * **output_attentions** (bool, Optional, 기본값 False) — 모든 주의 계층의 주의 텐서를 반환할지 여부를 지정합니다. 자세한 내용은 ```attentions```를 참고하세요.
> 
> * **output_hidden_states** (bool, Optional, 기본값 False) — 모든 계층의 숨겨진 상태를 반환할지 여부를 지정합니다. 자세한 내용은 ```hidden_states```를 참고하세요.
> 
> * **output_scores** (bool, Optional, 기본값 False) — 예측 점수를 반환할지 여부를 지정합니다. 자세한 내용은 ```scores```를 참고하세요.
> 
> * **return_dict_in_generate** (bool, Optional, 기본값 False) — 일반 튜플 대신 [ModelOutput](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput)을 반환할지 여부를 지정합니다.
> 
> * **synced_gpus** (bool, Optional, 기본값 False) — max_length까지 while 루프를 계속 실행할지 여부를 지정합니다. (ZeRO 단계 3에 필요함)
> 
> * **streamer** (BaseStreamer, Optional) — 생성된 시퀀스를 스트림할때 사용될 Streamer 객체를 설정합니다. 생성된 토큰인 streamer.put(token_ids)를 통해 전달되며, 스트리머는 추가 처리를 담당합니다.
> 
> * **model_kwargs** — 추가 모델 특정 키워드 인수는 모델의 forward 함수로 전달됩니다. 모델이 인코더-디코더 모델이면 kwargs는 encoder_outputs를 포함해야 합니다.
>
> ### Returns -> [SampleDecodeOnlyOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.SampleDecoderOnlyOutput), [SampleEncoderDecoderOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.SampleEncoderDecoderOutput) or *torch.LongTensor*
> 
> 언어 모델링 헤드를 가진 모델에 대한 토큰 ID의 시퀀스를 **다항 샘플링**을 사용하여 생성하며 텍스트-디코더, 텍스트-텍스트, 음성-텍스트 및 비전-텍스트 모델에 사용할 수 있습니다.
>
> > 대부분의 경우에, 당신 sample()를 직접 호출할 필요가 없습니다. 주로 generate()를 사용하게 됩니다. 생성 전략과 코드 예제에 대한 개요는 [이 가이드](https://huggingface.co/docs/transformers/v4.34.0/en/generation_strategies)를 확인하세요.
>
> Examples:
> ```python
> from transformers import (
>     AutoTokenizer,
>     AutoModelForCausalLM,
>     LogitsProcessorList,
>     MinLengthLogitsProcessor,
>     TopKLogitsWarper,
>     TemperatureLogitsWarper,
>     StoppingCriteriaList,
>     MaxLengthCriteria,
> )
> import torch
> 
> tokenizer = AutoTokenizer.from_pretrained("gpt2")
> model = AutoModelForCausalLM.from_pretrained("gpt2")
> 
> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
> model.config.pad_token_id = model.config.eos_token_id
> model.generation_config.pad_token_id = model.config.eos_token_id
> 
> input_prompt = "Today is a beautiful day, and"
> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
> 
> # instantiate logits processors
> logits_processor = LogitsProcessorList(
>     [
>         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
>     ]
> )
> # instantiate logits processors
> logits_warper = LogitsProcessorList(
>     [
>         TopKLogitsWarper(50),
>         TemperatureLogitsWarper(0.7),
>     ]
> )
> 
> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
> 
> torch.manual_seed(0)
> outputs = model.sample(
>     input_ids,
>     logits_processor=logits_processor,
>     logits_warper=logits_warper,
>     stopping_criteria=stopping_criteria,
> )
> 
> tokenizer.batch_decode(outputs, skip_special_tokens=True)
> ```
>
> ## beam_search
> ```python
> beam_search (input_ids: LongTensor, 
>              beam_scorer: BeamScorer,
>              logits_processor: typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None, 
>              stopping_criteria: typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None, 
>              max_length: typing.Optional[int] = None, 
>              pad_token_id: typing.Optional[int] = None, 
>              eos_token_id: typing.Union[int, typing.List[int], NoneType] = None, 
>              output_attentions: typing.Optional[bool] = None, 
>              output_hidden_states: typing.Optional[bool] = None, 
>              output_scores: typing.Optional[bool] = None, 
>              return_dict_in_generate: typing.Optional[bool] = None, 
>              synced_gpus: bool = False, 
>              **model_kwargs)
> ```
>
> ### Parameters
>
> * **input_ids** (torch.LongTensor의 모양 (batch_size, sequence_length)) — 생성을 위한 프롬프트(prompt)로 사용되는 시퀀스입니다.
>
> * **beam_scorer** (BeamScorer) — [BeamScorer](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.BeamScorer)의 파생 인스턴스로서 생성 중에 beam 가설이 어떻게 구성되고 저장되며 정렬되는지 정의합니다. 자세한 내용은 [BeamScorer](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.BeamScorer)의 문서를 참고하세요.
> 
> * **logits_processor** (LogitsProcessorList, Optional) — [LogitsProcessorList](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessorList)의 인스턴스 입니다. LogitsProcessorList는 [LogitsProcessor](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor)에서 파생된 클래스의 인스턴스 목록으로 각 생성 단계에서 언어 모델링 헤드의 예측 점수를 수정하는 데 사용됩니다.
> 
> * **stopping_criteria** (StoppingCriteriaList, Optional) — StoppingCriteriaList의 인스턴스 입니다. [StoppingCriteriaList](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteriaList)는 [StoppingCriteria](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteria)에서 파생된 클래스의 인스턴스 목록으로 생성 루프가 중단되어야 하는지 알리기 위해 사용됩니다.
>
> * **max_length** (int, Optional, 기본값 20) — **현재는 사용되지 않습니다.** 생성된 토큰의 수를 제한하기 위해 logits_processor 또는 stopping_criteria를 직접 사용해야 합니다.
> 
> * **pad_token_id** (int, Optional) — 패딩 토큰의 ID 입니다.
> 
> * **eos_token_id** (Union[int, List[int]], Optional) — 시퀀스 종료 토큰의 ID 입니다. 여러 시퀀스 종료 토큰을 설정하기 위해 목록을 Optional로 사용합니다.
> 
> * **output_attentions** (bool, Optional, 기본값 False) — 모든 주의 계층의 주의 텐서를 반환할지 여부를 지정합니다. 자세한 내용은 ```attentions```를 참고하세요.
> 
> * **output_hidden_states** (bool, Optional, 기본값 False) — 모든 계층의 숨겨진 상태를 반환할지 여부를 지정합니다. 자세한 내용은 ```hidden_states```를 참고하세요.
> 
> * **output_scores** (bool, Optional, 기본값 False) — 예측 점수를 반환할지 여부를 지정합니다. 자세한 내용은 ```scores```를 참고하세요.
> 
> * **return_dict_in_generate** (bool, Optional, 기본값 False) — 일반 튜플 대신 [ModelOutput](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput)을 반환할지 여부를 지정합니다.
> 
> * **synced_gpus** (bool, Optional, 기본값 False) — max_length까지 while 루프를 계속 실행할지 여부를 지정합니다. (ZeRO 단계 3에 필요함)
> 
> * **streamer** (BaseStreamer, Optional) — 생성된 시퀀스를 스트림할때 사용될 Streamer 객체를 설정합니다. 생성된 토큰인 streamer.put(token_ids)를 통해 전달되며, 스트리머는 추가 처리를 담당합니다.
> 
> * **model_kwargs** — 추가 모델 특정 키워드 인수는 모델의 forward 함수로 전달됩니다. 모델이 인코더-디코더 모델이면 kwargs는 encoder_outputs를 포함해야 합니다.
>
> 언어 모델링 헤드를 가진 모델에 대한 토큰 ID 시퀀스를 beam_search 디코딩을 사용하여 생성하며, 텍스트 디코더, 텍스트-텍스트, 음성-텍스트 및 비전-텍스트 모델에 사용할 수 있습니다.
>
> > 대부분의 경우에, 당신 beam_search()를 직접 호출할 필요가 없습니다. 주로 generate()를 사용하게 됩니다. 생성 전략과 코드 예제에 대한 개요는 [이 가이드](https://huggingface.co/docs/transformers/v4.34.0/en/generation_strategies)를 확인하세요.
>
> Examples:
> ```python
> from transformers import (
>     AutoTokenizer,
>     AutoModelForSeq2SeqLM,
>     LogitsProcessorList,
>     MinLengthLogitsProcessor,
>     BeamSearchScorer,
> )
> import torch
> 
> tokenizer = AutoTokenizer.from_pretrained("t5-base")
> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
> 
> encoder_input_str = "translate English to German: How old are you?"
> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
> 
> 
> # lets run beam search using 3 beams
> num_beams = 3
> # define decoder start token ids
> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
> input_ids = input_ids * model.config.decoder_start_token_id
> 
> # add encoder_outputs to model keyword arguments
> model_kwargs = {
>     "encoder_outputs": model.get_encoder()(
>         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
>     )
> }
> 
> # instantiate beam scorer
> beam_scorer = BeamSearchScorer(
>     batch_size=1,
>     num_beams=num_beams,
>     device=model.device,
> )
> 
> # instantiate logits processors
> logits_processor = LogitsProcessorList(
>     [
>         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
>     ]
> )
> 
> outputs = model.beam_search(input_ids, beam_scorer, > logits_processor=logits_processor, **model_kwargs)
> 
> tokenizer.batch_decode(outputs, skip_special_tokens=True)
> ```
>
> ## beam_sample
> ```python
> beam_sample (input_ids: LongTensor, 
>              beam_scorer: BeamScorer, 
>              logits_processor: typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None, 
>              stopping_criteria: typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None, 
>              logits_warper: typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None, 
>              max_length: typing.Optional[int] = None, 
>              pad_token_id: typing.Optional[int] = None, 
>              eos_token_id: typing.Union[int, typing.List[int], NoneType] = None, 
>              output_attentions: typing.Optional[bool] = None, 
>              output_hidden_states: typing.Optional[bool] = None, 
>              output_scores: typing.Optional[bool] = None, 
>              return_dict_in_generate: typing.Optional[bool] = None, 
>              synced_gpus: bool = False, 
>              **model_kwargs)
> ```
>
> ### Parameters
>
> * **input_ids** (torch.LongTensor의 모양 (batch_size, sequence_length)) — 생성을 위한 프롬프트(prompt)로 사용되는 시퀀스입니다.
>
> * **beam_scorer** (BeamScorer) — [BeamScorer](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.BeamScorer)의 파생 인스턴스로서 생성 중에 beam 가설이 어떻게 구성되고 저장되며 정렬되는지 정의합니다. 자세한 내용은 [BeamScorer](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.BeamScorer)의 문서를 참고하세요.
> 
> * **logits_processor** (LogitsProcessorList, Optional) — [LogitsProcessorList](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessorList)의 인스턴스 입니다. LogitsProcessorList는 [LogitsProcessor](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor)에서 파생된 클래스의 인스턴스 목록으로 각 생성 단계에서 언어 모델링 헤드의 예측 점수를 수정하는 데 사용됩니다.
> 
> * **stopping_criteria** (StoppingCriteriaList, Optional) — [StoppingCriteriaList](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteriaList)의 인스턴스 입니다. StoppingCriteriaList는 [StoppingCriteria](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteria)에서 파생된 클래스의 인스턴스 목록으로 생성 루프가 중단되어야 하는지 알리기 위해 사용됩니다.
>
> * **logits_warper** (LogitsProcessorList, Optional) — LogitsProcessorList의 인스턴스입니다. LogitsProcessorList는 [LogitsWarper](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsWarper)에서 파생된 클래스의 인스턴스 목록으로 각 생성 단계에서 다항 샘플링 전에 언어 모델링 헤드의 예측 점수 분포를 왜곡하는 데 사용됩니다.
> 
> * **max_length** (int, Optional, 기본값 20) — **현재는 사용되지 않습니다.** 생성된 토큰의 수를 제한하기 위해 logits_processor 또는 stopping_criteria를 직접 사용해야 합니다.
> 
> * **pad_token_id** (int, Optional) — 패딩 토큰의 ID 입니다.
> 
> * **eos_token_id** (Union[int, List[int]], Optional) — 시퀀스 종료 토큰의 ID 입니다. 여러 시퀀스 종료 토큰을 설정하기 위해 목록을 Optional로 사용합니다.
> 
> * **output_attentions** (bool, Optional, 기본값 False) — 모든 주의 계층의 주의 텐서를 반환할지 여부를 지정합니다. 자세한 내용은 ```attentions```를 참고하세요.
> 
> * **output_hidden_states** (bool, Optional, 기본값 False) — 모든 계층의 숨겨진 상태를 반환할지 여부를 지정합니다. 자세한 내용은 ```hidden_states```를 참고하세요.
> 
> * **output_scores** (bool, Optional, 기본값 False) — 예측 점수를 반환할지 여부를 지정합니다. 자세한 내용은 ```scores```를 참고하세요.
> 
> * **return_dict_in_generate** (bool, Optional, 기본값 False) — 일반 튜플 대신 [ModelOutput](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput)을 반환할지 여부를 지정합니다.
> 
> * **synced_gpus** (bool, Optional, 기본값 False) — max_length까지 while 루프를 계속 실행할지 여부를 지정합니다. (ZeRO 단계 3에 필요함)
>  
> * **model_kwargs** — 추가 모델 특정 키워드 인수는 모델의 forward 함수로 전달됩니다. 모델이 인코더-디코더 모델이면 kwargs는 encoder_outputs를 포함해야 합니다.
>
> 언어 모델링 헤드를 가진 모델에 대한 토큰 ID 시퀀스를 beam_search multinomial sampling 디코딩을 사용하여 생성하며, 텍스트 디코더, 텍스트-텍스트, 음성-텍스트 및 비전-텍스트 모델에 사용할 수 있습니다.
>
> > 대부분의 경우에, 당신 beam_sample()를 직접 호출할 필요가 없습니다. 주로 generate()를 사용하게 됩니다. 생성 전략과 코드 예제에 대한 개요는 [이 가이드](https://huggingface.co/docs/transformers/v4.34.0/en/generation_strategies)를 확인하세요.
>
> Example:
> ```python
> from transformers import (
>     AutoTokenizer,
>     AutoModelForSeq2SeqLM,
>     LogitsProcessorList,
>     MinLengthLogitsProcessor,
>     TopKLogitsWarper,
>     TemperatureLogitsWarper,
>     BeamSearchScorer,
> )
> import torch
> 
> tokenizer = AutoTokenizer.from_pretrained("t5-base")
> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
> 
> encoder_input_str = "translate English to German: How old are you?"
> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
> 
> # lets run beam search using 3 beams
> num_beams = 3
> # define decoder start token ids
> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
> input_ids = input_ids * model.config.decoder_start_token_id
> 
> # add encoder_outputs to model keyword arguments
> model_kwargs = {
>     "encoder_outputs": model.get_encoder()(
>         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
>     )
> }
> 
> # instantiate beam scorer
> beam_scorer = BeamSearchScorer(
>     batch_size=1,
>     max_length=model.config.max_length,
>     num_beams=num_beams,
>     device=model.device,
> )
> 
> # instantiate logits processors
> logits_processor = LogitsProcessorList(
>     [MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)]
> )
> # instantiate logits processors
> logits_warper = LogitsProcessorList(
>     [
>         TopKLogitsWarper(50),
>         TemperatureLogitsWarper(0.7),
>     ]
> )
> 
> outputs = model.beam_sample(
>     input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
> )
> 
> tokenizer.batch_decode(outputs, skip_special_tokens=True)
> ```
>
>
> ## contrastive_search
> ```python
> contrastive_search (input_ids: LongTensor, 
>                     top_k: typing.Optional[int] = 1, 
>                     penalty_alpha: typing.Optional[float] = 0, 
>                     logits_processor: typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None, 
>                     logits_warper: typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None, 
>                     stopping_criteria: typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None, 
>                     pad_token_id: typing.Optional[int] = None, 
>                     eos_token_id: typing.Union[int, typing.List[int], NoneType] = None, 
>                     output_attentions: typing.Optional[bool] = None, 
>                     output_hidden_states: typing.Optional[bool] = None, 
>                     output_scores: typing.Optional[bool] = None, 
>                     return_dict_in_generate: typing.Optional[bool] = None, 
>                     synced_gpus: bool = False, 
>                     streamer: typing.Optional[ForwardRef('BaseStreamer')] = None, 
>                     sequential: typing.Optional[bool] = None, 
>                     **model_kwargs)
> ```
> ### Parameters
>
> * **top_k** (int, Optional, 기본값 1) — 대조적 검색을 위해 재순위를 지정하는 데 가용되는 후보 세트의 크기를 설정합니다.
>
> * **penalty_alpha** (float, Optional, 기본값 0)  — 대조적 검색에 대한 퇴화 페널티를 나타냅니다. 0보다 클 때 활성화 됩니다.
>
> * **input_ids** (torch.LongTensor의 모양 (batch_size, sequence_length)) — 생성을 위한 프롬프트(prompt)로 사용되는 시퀀스입니다.
> 
> * **logits_processor** (LogitsProcessorList, Optional) — [LogitsProcessorList](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessorList)의 인스턴스 입니다. LogitsProcessorList는 [LogitsProcessor](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor)에서 파생된 클래스의 인스턴스 목록으로 각 생성 단계에서 언어 모델링 헤드의 예측 점수를 수정하는 데 사용됩니다.
> 
> * **stopping_criteria** (StoppingCriteriaList, Optional) — [StoppingCriteriaList](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteriaList)의 인스턴스 입니다. StoppingCriteriaList는 [StoppingCriteria](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteria)에서 파생된 클래스의 인스턴스 목록으로 생성 루프가 중단되어야 하는지 알리기 위해 사용됩니다.
>
> * **logits_warper** (LogitsProcessorList, Optional) — LogitsProcessorList의 인스턴스입니다. LogitsProcessorList는 [LogitsWarper](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsWarper)에서 파생된 클래스의 인스턴스 목록으로 각 생성 단계에서 다항 샘플링 전에 언어 모델링 헤드의 예측 점수 분포를 왜곡하는 데 사용됩니다.
> 
> * **pad_token_id** (int, Optional) — 패딩 토큰의 ID 입니다.
> 
> * **eos_token_id** (Union[int, List[int]], Optional) — 시퀀스 종료 토큰의 ID 입니다. 여러 시퀀스 종료 토큰을 설정하기 위해 목록을 Optional로 사용합니다.
> 
> * **output_attentions** (bool, Optional, 기본값 False) — 모든 주의 계층의 주의 텐서를 반환할지 여부를 지정합니다. 자세한 내용은 ```attentions```를 참고하세요.
> 
> * **output_hidden_states** (bool, Optional, 기본값 False) — 모든 계층의 숨겨진 상태를 반환할지 여부를 지정합니다. 자세한 내용은 ```hidden_states```를 참고하세요.
> 
> * **output_scores** (bool, Optional, 기본값 False) — 예측 점수를 반환할지 여부를 지정합니다. 자세한 내용은 ```scores```를 참고하세요.
> 
> * **return_dict_in_generate** (bool, Optional, 기본값 False) — 일반 튜플 대신 [ModelOutput](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput)을 반환할지 여부를 지정합니다.
> 
> * **synced_gpus** (bool, Optional, 기본값 False) — max_length까지 while 루프를 계속 실행할지 여부를 지정합니다. (ZeRO 단계 3에 필요함)
> 
> * **streamer** (BaseStreamer, Optional) — 생성된 시퀀스를 스트림할때 사용될 Streamer 객체를 설정합니다. 생성된 토큰인 streamer.put(token_ids)를 통해 전달되며, 스트리머는 추가 처리를 담당합니다.
> 
> * **model_kwargs** — 추가 모델 특정 키워드 인수는 모델의 forward 함수로 전달됩니다. 모델이 인코더-디코더 모델이면 kwargs는 encoder_outputs를 포함해야 합니다.
>
> 언어 모델링 헤드를 가진 모델에 대한 토큰 ID 시퀀스를 contrastive search 디코딩을 사용하여 생성하며, 텍스트 디코더, 텍스트-텍스트, 음성-텍스트 및 비전-텍스트 모델에 사용할 수 있습니다.
>
> > 대부분의 경우에, 당신 contrastive_search()를 직접 호출할 필요가 없습니다. 주로 generate()를 사용하게 됩니다. 생성 전략과 코드 예제에 대한 개요는 [이 가이드](https://huggingface.co/docs/transformers/v4.34.0/en/generation_strategies)를 확인하세요.
>
> Example:
> ```python
>from transformers import (
>     AutoTokenizer,
>     AutoModelForCausalLM,
>     StoppingCriteriaList,
>     MaxLengthCriteria,
> )
>
> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
> model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
> # set pad_token_id to eos_token_id because OPT does not have a PAD token
> model.config.pad_token_id = model.config.eos_token_id
> input_prompt = "DeepMind Company is"
> input_ids = tokenizer(input_prompt, return_tensors="pt")
> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=64)])
> outputs = model.contrastive_search(
>     **input_ids, penalty_alpha=0.6, top_k=4, stopping_criteria=stopping_criteria
> )
> tokenizer.batch_decode(outputs, skip_special_tokens=True)
> ```
>
> ## group_beam_search
> ```python
> group_beam_search(input_ids: LongTensor, 
>                   beam_scorer: BeamScorer, 
>                   logits_processor: typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None, 
>                   stopping_criteria: typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None, 
>                   max_length: typing.Optional[int] = None, 
>                   pad_token_id: typing.Optional[int] = None, 
>                   eos_token_id: typing.Union[int, typing.List[int], NoneType] = None, 
>                   output_attentions: typing.Optional[bool] = None, 
>                   output_hidden_states: typing.Optional[bool] = None, 
>                   output_scores: typing.Optional[bool] = None, 
>                   return_dict_in_generate: typing.Optional[bool] = None, 
>                   synced_gpus: bool = False, 
>                   **model_kwargs, 
>                   )
> ```
>
>
> ### Parameters ---
>
> * **input_ids** (torch.LongTensor의 모양 (batch_size, sequence_length)) — 생성을 위한 프롬프트(prompt)로 사용되는 시퀀스입니다.
>
> * **beam_scorer** (BeamScorer) — [BeamScorer](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.BeamScorer)의 파생 인스턴스로서 생성 중에 beam 가설이 어떻게 구성되고 저장되며 정렬되는지 정의합니다. 자세한 내용은 [BeamScorer](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.BeamScorer)의 문서를 참고하세요.
> 
> * **logits_processor** (LogitsProcessorList, Optional) — [LogitsProcessorList](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessorList)의 인스턴스 입니다. LogitsProcessorList는 [LogitsProcessor](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor)에서 파생된 클래스의 인스턴스 목록으로 각 생성 단계에서 언어 모델링 헤드의 예측 점수를 수정하는 데 사용됩니다.
> 
> * **stopping_criteria** (StoppingCriteriaList, Optional) — [StoppingCriteriaList](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteriaList)의 인스턴스 입니다. StoppingCriteriaList는 [StoppingCriteria](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteria)에서 파생된 클래스의 인스턴스 목록으로 생성 루프가 중단되어야 하는지 알리기 위해 사용됩니다.
>
> * **max_length** (int, Optional, 기본값 20) — **현재는 사용되지 않습니다.** 생성된 토큰의 수를 제한하기 위해 logits_processor 또는 stopping_criteria를 직접 사용해야 합니다.
> 
> * **pad_token_id** (int, Optional) — 패딩 토큰의 ID 입니다.
> 
> * **eos_token_id** (Union[int, List[int]], Optional) — 시퀀스 종료 토큰의 ID 입니다. 여러 시퀀스 종료 토큰을 설정하기 위해 목록을 Optional로 사용합니다.
> 
> * **output_attentions** (bool, Optional, 기본값 False) — 모든 주의 계층의 주의 텐서를 반환할지 여부를 지정합니다. 자세한 내용은 ```attentions```를 참고하세요.
> 
> * **output_hidden_states** (bool, Optional, 기본값 False) — 모든 계층의 숨겨진 상태를 반환할지 여부를 지정합니다. 자세한 내용은 ```hidden_states```를 참고하세요.
> 
> * **output_scores** (bool, Optional, 기본값 False) — 예측 점수를 반환할지 여부를 지정합니다. 자세한 내용은 ```scores```를 참고하세요.
> 
> * **return_dict_in_generate** (bool, Optional, 기본값 False) — 일반 튜플 대신 [ModelOutput](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput)을 반환할지 여부를 지정합니다.
> 
> * **synced_gpus** (bool, Optional, 기본값 False) — max_length까지 while 루프를 계속 실행할지 여부를 지정합니다. (ZeRO 단계 3에 필요함)
> 
> * **streamer** (BaseStreamer, Optional) — 생성된 시퀀스를 스트림할때 사용될 Streamer 객체를 설정합니다. 생성된 토큰인 streamer.put(token_ids)를 통해 전달되며, 스트리머는 추가 처리를 담당합니다.
> 
> * **model_kwargs** — 추가 모델 특정 키워드 인수는 모델의 forward 함수로 전달됩니다. 모델이 인코더-디코더 모델이면 kwargs는 encoder_outputs를 포함해야 합니다.
>
> 언어 모델링 헤드를 가진 모델에 대한 토큰 ID 시퀀스를 diverse beam search decoding 디코딩을 사용하여 생성하며, 텍스트 디코더, 텍스트-텍스트, 음성-텍스트 및 비전-텍스트 모델에 사용할 수 있습니다.
>
> > 대부분의 경우에, 당신 group_beam_search()를 직접 호출할 필요가 없습니다. 주로 generate()를 사용하게 됩니다. 생성 전략과 코드 예제에 대한 개요는 [이 가이드](https://huggingface.co/docs/transformers/v4.34.0/en/generation_strategies)를 확인하세요.
>
> Examples:
> ```python
>from transformers import (
>    AutoTokenizer,
>    AutoModelForSeq2SeqLM,
>    LogitsProcessorList,
>    MinLengthLogitsProcessor,
>    HammingDiversityLogitsProcessor,
>    BeamSearchScorer,
>)
>import torch
>
>tokenizer = AutoTokenizer.from_pretrained("t5-base")
>model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
>
>encoder_input_str = "translate English to German: How old are you?"
>encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
>
>
># lets run diverse beam search using 6 beams
>num_beams = 6
># define decoder start token ids
>input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
>input_ids = input_ids * model.config.decoder_start_token_id
>
># add encoder_outputs to model keyword arguments
>model_kwargs = {
>    "encoder_outputs": model.get_encoder()(
>        encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
>    )
>}
>
># instantiate beam scorer
>beam_scorer = BeamSearchScorer(
>    batch_size=1,
>    max_length=model.config.max_length,
>    num_beams=num_beams,
>    device=model.device,
>    num_beam_groups=3,
>)
>
># instantiate logits processors
>logits_processor = LogitsProcessorList(
>    [
>        HammingDiversityLogitsProcessor(5.5, num_beams=6, num_beam_groups=3),
>        MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
>    ]
>)
>
>outputs = model.group_beam_search(
>    input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs
>)
>
>tokenizer.batch_decode(outputs, skip_special_tokens=True)
> ```
>
> ## constrained_beam_search
> ```python
> constrained_beam_search(input_ids: LongTensor, 
>                         constrained_beam_scorer: ConstrainedBeamSearchScorer, 
>                         logits_processor: typing.Optional[transformers.generation.logits_process.LogitsProcessorList] = None, 
>                         stopping_criteria: typing.Optional[transformers.generation.stopping_criteria.StoppingCriteriaList] = None, 
>                         max_length: typing.Optional[int] = None, 
>                         pad_token_id: typing.Optional[int] = None, 
>                         eos_token_id: typing.Union[int, typing.List[int], NoneType] = None, 
>                         output_attentions: typing.Optional[bool] = None, 
>                         output_hidden_states: typing.Optional[bool] = None, 
>                         output_scores: typing.Optional[bool] = None, 
>                         return_dict_in_generate: typing.Optional[bool] = None, 
>                         synced_gpus: typing.Optional[bool] = None, 
>                         **model_kwargs)
> ```
>
>
> ### Parameters ---
>
> * **input_ids** (torch.LongTensor의 모양 (batch_size, sequence_length)) — 생성을 위한 프롬프트(prompt)로 사용되는 시퀀스입니다.
>
> * **constrained_beam_scorer** (ConstrainedBeamSearchScorer) — 생성 동안 beam 가설이 어떻게 구성되고 저장되며 정렬되는지 정의하는 BeamScorer의 파생 인스턴스로, 긍정적 제약 조건 목록을 만족시킵니다. 자세한 내용은 ```ConstrainedBeamSearchScorer```를 참고하세요.
> 
> * **logits_processor** (LogitsProcessorList, Optional) — [LogitsProcessorList](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessorList)의 인스턴스 입니다. LogitsProcessorList는 [LogitsProcessor](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsProcessor)에서 파생된 클래스의 인스턴스 목록으로 각 생성 단계에서 언어 모델링 헤드의 예측 점수를 수정하는 데 사용됩니다.
> 
> * **stopping_criteria** (StoppingCriteriaList, Optional) — [StoppingCriteriaList](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteriaList)의 인스턴스 입니다. StoppingCriteriaList는 [StoppingCriteria](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.StoppingCriteria)에서 파생된 클래스의 인스턴스 목록으로 생성 루프가 중단되어야 하는지 알리기 위해 사용됩니다.
>
> * **logits_warper** (LogitsProcessorList, Optional) — LogitsProcessorList의 인스턴스입니다. LogitsProcessorList는 [LogitsWarper](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.LogitsWarper)에서 파생된 클래스의 인스턴스 목록으로 각 생성 단계에서 다항 샘플링 전에 언어 모델링 헤드의 예측 점수 분포를 왜곡하는 데 사용됩니다.
> 
> * **max_length** (int, Optional, 기본값 20) — **현재는 사용되지 않습니다.** 생성된 토큰의 수를 제한하기 위해 logits_processor 또는 stopping_criteria를 직접 사용해야 합니다.
> 
> * **pad_token_id** (int, Optional) — 패딩 토큰의 ID 입니다.
> 
> * **eos_token_id** (Union[int, List[int]], Optional) — 시퀀스 종료 토큰의 ID 입니다. 여러 시퀀스 종료 토큰을 설정하기 위해 목록을 Optional로 사용합니다.
> 
> * **output_attentions** (bool, Optional, 기본값 False) — 모든 주의 계층의 주의 텐서를 반환할지 여부를 지정합니다. 자세한 내용은 ```attentions```를 참고하세요.
> 
> * **output_hidden_states** (bool, Optional, 기본값 False) — 모든 계층의 숨겨진 상태를 반환할지 여부를 지정합니다. 자세한 내용은 ```hidden_states```를 참고하세요.
> 
> * **output_scores** (bool, Optional, 기본값 False) — 예측 점수를 반환할지 여부를 지정합니다. 자세한 내용은 ```scores```를 참고하세요.
> 
> * **return_dict_in_generate** (bool, Optional, 기본값 False) — 일반 튜플 대신 [ModelOutput](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput)을 반환할지 여부를 지정합니다.
> 
> * **synced_gpus** (bool, Optional, 기본값 False) — max_length까지 while 루프를 계속 실행할지 여부를 지정합니다. (ZeRO 단계 3에 필요함)
> 
> * **streamer** (BaseStreamer, Optional) — 생성된 시퀀스를 스트림할때 사용될 Streamer 객체를 설정합니다. 생성된 토큰인 streamer.put(token_ids)를 통해 전달되며, 스트리머는 추가 처리를 담당합니다.
> 
> * **model_kwargs** — 추가 모델 특정 키워드 인수는 모델의 forward 함수로 전달됩니다. 모델이 인코더-디코더 모델이면 kwargs는 encoder_outputs를 포함해야 합니다.
>
> 언어 모델링 헤드를 가진 모델에 대한 토큰 ID 시퀀스를 diverse beam search decoding 디코딩을 사용하여 생성하며, 텍스트 디코더, 텍스트-텍스트, 음성-텍스트 및 비전-텍스트 모델에 사용할 수 있습니다.
>
> > 대부분의 경우에, 당신 group_beam_search()를 직접 호출할 필요가 없습니다. 주로 generate()를 사용하게 됩니다. 생성 전략과 코드 예제에 대한 개요는 [이 가이드](https://huggingface.co/docs/transformers/v4.34.0/en/generation_strategies)를 확인하세요.
>
> Examples:
>```python
>from transformers import (
>    AutoTokenizer,
>    AutoModelForSeq2SeqLM,
>    LogitsProcessorList,
>    MinLengthLogitsProcessor,
>    ConstrainedBeamSearchScorer,
>    PhrasalConstraint,
>)
>import torch
>
>tokenizer = AutoTokenizer.from_pretrained("t5-base")
>model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
>
>encoder_input_str = "translate English to German: How old are you?"
>encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
>
>
># lets run beam search using 3 beams
>num_beams = 3
># define decoder start token ids
>input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
>input_ids = input_ids * model.config.decoder_start_token_id
>
># add encoder_outputs to model keyword arguments
>model_kwargs = {
>    "encoder_outputs": model.get_encoder()(
>        encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
>    )
>}
>
>constraint_str = "Sie"
>constraint_token_ids = tokenizer.encode(constraint_str)[:-1]  # slice to remove >eos token
>constraints = [PhrasalConstraint(token_ids=constraint_token_ids)]
>
>
># instantiate beam scorer
>beam_scorer = ConstrainedBeamSearchScorer(
>    batch_size=1, num_beams=num_beams, device=model.device, constraints=constraints
>)
>
># instantiate logits processors
>logits_processor = LogitsProcessorList(
>    [
>        MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
>    ]
>)
>
>outputs = model.constrained_beam_search(
>    input_ids, beam_scorer, constraints=constraints, logits_processor=logits_processor, **model_kwargs
>)
>
>tokenizer.batch_decode(outputs, skip_special_tokens=True)
>```

# TFGenerationMixin

[TFPreTrainedModel](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/model#transformers.TFPreTrainedModel)에 혼합으로 사용할 수 있는 모든 생성 지원 함수를 포함하는 클래스입니다.

아래에 나열된 상황에서 TFGenerationMixin의 [generate()](#generate-1)를 사용할 수 있습니다:

* num_beams=1 및 do_sample=False 인 greedy_search()를 호출하여 Greedy 디코딩
* penalty_alpha>0 및 top_k>1 인 contrastive_search()를 호출하여 대조적 검색
* num_beams=1 및 do_sample=True 인 sample()를 호출하여 다항 샘플링
* num_beams>1 인 beam_search()를 호출하여 beam 검색 디코딩
  
위의 메소드 중 어느 것도 직접 호출할 필요가 없습니다. 대신 ‘generate’에 사용자 정의 매개변수 값을 전달하면 모두 설정 및 사용할 수 있습니다. 디코딩 전략에 대해 자세히 알아보려면 [텍스트 생성 전략 가이드](https://huggingface.co/docs/transformers/v4.34.0/en/generation_strategies)를 참조하십시오.


> ## generate
> ```python
> generate(inputs: typing.Optional[tensorflow.python.framework.ops.Tensor] = None, 
>          generation_config: typing.Optional[transformers.generation.configuration_utils.GenerationConfig] = None, 
>          logits_processor: typing.Optional[transformers.generation.tf_logits_process.TFLogitsProcessorList] = None, 
>          seed = None, 
>          **kwargs )
> ```
>
> ### Parameters ---
>
>* **inputs** (tf.Tensor 모델에 따라 다양한 형태, Optional) — 생성을 위한 선제적으로 사용되는 시퀀스 또는 인코더에 대한 모델 입력입니다. None인 경우 메서드는 bos_token_id와 배치 크기 1로 초기화합니다. 디코더 전용 모델의 경우 inputs는 input_ids 형식이어야 합니다. 인코더-디코더 모델의 경우 inputs는 input_ids, input_values, input_features, 또는 pixel_values 중 어떤 것을 나타낼 수 있습니다.
>
>* **generation_config** (~generation.GenerationConfig, Optional) — 생성 호출을 위한 기본 매개변수화로 사용될 생성 구성입니다. generate에 전달되는 generation_config의 속성과 일치하는 **kwargs는 이를 오버라이드합니다. generation_config가 제공되지 않는 경우, 기본값이 사용됩니다. 이 기본값은 다음 나타난 우선순위로 사용됩니다: 1) generation_config.json 모델 파일이 있는 경우; 2) 모델 구성에서. 생성을 매개변수화하려면 지정되지 않은 매개변수는 [GenerationConfig](#generationconfig)의 기본값을 상속받을 것이므로 해당 문서를 확인해야 합니다.
>
>* **logits_processor** (LogitsProcessorList, Optional) — 인수와 생성 구성에서 구축된 기본 logits 프로세서를 보완하는 사용자 정의 logits 프로세서입니다. 인수나 생성 구성으로 이미 생성된 로짓 프로세서가 전달되면 오류가 발생합니다. 이 기능은 고급 사용자를 위한 것입니다.
>
>* **seed** (List[int], Optional) — 만약 do_sample이 True이라면, 샘플링을 제어하기 위한 랜덤 시드로, 두 개의 정수가 포함됩니다. tf.random의 상태가 없는 함수의 시드 인수를 참조하세요.
>
>* **kwargs** (Dict[str, Any], Optional) — generate_config의 ad hoc 매개변수화 및/또는 모델에 전달될 추가 모델 특정 kwargs입니다. 모델이 인코더-디코더 모델인 경우 인코더 특정 kwargs는 prefix가 없어야 하며 디코더 특정 kwargs는 decoder_로 prefix가 있어야 합니다.
>
> ### Returns -> ModelOutput or *tf.Tensor*
>
>모델이 인코더-디코더 모델이 아닌 경우 (model.config.is_encoder_decoder=False), 가능한 [ModelOutput](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) 유형은 다음과 같습니다:
>
>* [TFGreedySearchDecoderOnlyOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.TFGreedySearchDecoderOnlyOutput),
>* [TFSampleDecoderOnlyOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.TFSampleDecoderOnlyOutput),
>* [TFBeamSearchDecoderOnlyOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.TFBeamSearchDecoderOnlyOutput),
>* [TFBeamSampleDecoderOnlyOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.TFBeamSampleDecoderOnlyOutput)
>
>모델이 인코더-디코더 모델인 경우 (model.config.is_encoder_decoder=True), 가능한 [ModelOutput](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/output#transformers.utils.ModelOutput) 유형은 다음과 같습니다:
>
>* [TFGreedySearchEncoderDecoderOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.TFGreedySearchEncoderDecoderOutput),
>* [TFSampleEncoderDecoderOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.TFSampleEncoderDecoderOutput),
>* [TFBeamSearchEncoderDecoderOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.TFBeamSearchEncoderDecoderOutput),
>* [TFBeamSampleEncoderDecoderOutput](https://huggingface.co/docs/transformers/v4.34.0/en/internal/generation_utils#transformers.generation.TFBeamSampleEncoderDecoderOutput)
>
>언어 모델링 헤드를 가진 모델에 대한 토큰 ID 시퀀스를 생성합니다.
>
>>대부분의 생성 제어 매개변수는 generation_config에 설정되며, 전달되지 않는 경우 모델의 기본 생성 구성으로 설정됩니다. .generate(inputs, num_beams=4, do_sample=True)와 같이 해당 매개변수를 generate에 전달하여 generation_config의 어떤 것이든 덮어쓸 수 있습니다.
>>
>>생성 전략과 코드 예제에 대한 개요는 [이 가이드](https://huggingface.co/docs/transformers/v4.34.0/en/generation_strategies)를 확인하세요.
>
>
> ## compute_transition_scores
> ```python
> compute_transition_scores(sequences: Tensor, 
>                           scores: typing.Tuple[tensorflow.python.framework.ops.Tensor], 
>                           beam_indices: typing.Optional[tensorflow.python.framework.ops.Tensor] = None, 
>                           normalize_logits: bool = False)
> ```
>
> ### Parameters ---
>
> * **sequences** (torch.LongTensor)  — 생성된 시퀀스입니다. 두 번째 차원 (시퀀스 길이) max_length와 동일하거나 eos_token_id로 인해 모든 배치가 일찍 완료되었을 경우 더 짧습니다.
> 
> * **scores** (tuple(torch.FloatTensor)) — 각 생성 단계에서 어휘 토큰에 대한 전환 점수입니다. Beam 전환 점수는 사전에 생성된 토큰의 Log Softmax를 기반한 토큰의 로그 확률로 구성되며, torch.FloatTensor의 튜플에는 최대 max_new_tokens개의 요소를 포함합니다. 각 텐서는 (batch_sizenum_beams, config.vocab_size)의 형태를 가집니다.
>
> * **beam_indices** (torch.LongTensor, Optional) — 각 생성 단계에서 생성된 토큰 id의 beam 인덱스입니다. 구조 형태가 (batch_sizenum_return_sequences, sequence_length)인 torch.LongTensor입니다. generate-time안에 num_beams>1 인 경우에만 필요합니다.
>
> * **normalize_logits** (bool, Optional, 기본값 False) — 로짓을 정규화할지를 선택할 수 있습니다.(과거 구조 상의 이유로 로짓이 정규화되지 않을 수 있습니다).
>
> 
> ### Returns -> **tf.Tensor**
> (batch_size*num_return_sequences, sequence_length) 형태를 가지는 tf.Tensor를 반환하며, 전환 점수(logits)를 포함합니다.
>
> 생성 점수(및 beam 검색이 사용된 경우 beam 인덱스)를 기반으로 시퀀스의 전환 점수를 계산합니다. 이와 같 방법 생성 시간에 선택된 토큰의 점수를 빠르게 얻기 위해 사용할 수 있는 편리한 방법 중 하나입니다.
>
> Examples:
> ```python
>from transformers import GPT2Tokenizer, TFAutoModelForCausalLM
>import numpy as np
>
>tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
>model = TFAutoModelForCausalLM.from_pretrained("gpt2")
>tokenizer.pad_token_id = tokenizer.eos_token_id
>inputs = tokenizer(["Today is"], return_tensors="tf")
>
># Example 1: Print the scores for each token generated with Greedy Search
>outputs = model.generate(**inputs, max_new_tokens=5, >return_dict_in_generate=True, output_scores=True)
>transition_scores = model.compute_transition_scores(
>    outputs.sequences, outputs.scores, normalize_logits=True
>)
># input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
># encoder-decoder models, like BART or T5.
>input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape>[1]
>generated_tokens = outputs.sequences[:, input_length:]
>for tok, score in zip(generated_tokens[0], transition_scores[0]):
>    # | token | token string | logits | probability
>    print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
>
># Example 2: Reconstruct the sequence scores from Beam Search
>outputs = model.generate(
>    **inputs,
>    max_new_tokens=5,
>    num_beams=4,
>    num_return_sequences=4,
>    return_dict_in_generate=True,
>    output_scores=True,
>)
>transition_scores = model.compute_transition_scores(
>    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
>)
># If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
># Tip: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
># use case, you might want to recompute it with `normalize_logits=True`.
>output_length = input_length + np.sum(transition_scores.numpy() < 0, axis=1)
>length_penalty = model.generation_config.length_penalty
>reconstructed_scores = np.sum(transition_scores, axis=1) / (output_length**length_penalty)
>print(np.allclose(outputs.sequences_scores, reconstructed_scores))
> ```
>

# FlaxGenerationMixin

[FlaxPreTrainedModel](https://huggingface.co/docs/transformers/v4.34.0/en/main_classes/model#transformers.FlaxPreTrainedModel)에 혼합하여 사용하기 위한 자동 회귀 텍스트 생성을 위한 모든 기능을 포함하는 클래스입니다.

이 클래스는 [generate()](#generate-2)를 공개하며, 다음에 사용할 수 있습니다:

* num_beams=1 및 do_sample=False인 greedy_search()를 호출하여 Greedy 디코딩.
* num_beams=1 및 do_sample=True인 sample()를 호출하여 다항 샘플링.
* num_beams>1 및 do_sample=False인 beam_search()를 호출하여 빔 검색 디코딩.
  
위의 메소드 중 어느 것도 직접 호출할 필요가 없습니다. 대신 ‘generate’에 사용자 정의 매개변수 값을 전달하면 모두 설정 및 사용할 수 있습니다. 디코딩 전략에 대해 자세히 알아보려면 [텍스트 생성 전략 가이드](https://huggingface.co/docs/transformers/v4.34.0/en/generation_strategies)를 참조하십시오.

> ## generate
> ```python
> generate(input_ids: Array, 
>          generation_config: typing.Optional[transformers.generation.configuration_utils.GenerationConfig] = None, 
>          prng_key: typing.Optional[jax.Array] = None, 
>          trace: bool = True, 
>          params: typing.Union[typing.Dict[str, jax.Array], NoneType] = None, 
>          logits_processor: typing.Optional[transformers.generation.flax_logits_process.FlaxLogitsProcessorList] = None, 
>          **kwargs)
> ```
>
> ### Parameters ---
>
>* **inputs** (tf.Tensor 모델에 따라 다양한 형태, Optional) — 생성을 위한 선제적으로 사용되는 시퀀스 또는 인코더에 대한 모델 입력입니다. None인 경우 메서드는 bos_token_id와 배치 크기 1로 초기화합니다. 디코더 전용 모델의 경우 inputs는 input_ids 형식이어야 합니다. 인코더-디코더 모델의 경우 inputs는 input_ids, input_values, input_features, 또는 pixel_values 중 어떤 것을 나타낼 수 있습니다.
>
>* **generation_config** (~generation.GenerationConfig, Optional) — 생성 호출을 위한 기본 매개변수화로 사용될 생성 구성입니다. generate에 전달되는 generation_config의 속성과 일치하는 **kwargs는 이를 오버라이드합니다. generation_config가 제공되지 않는 경우, 기본값이 사용됩니다. 이 기본값은 다음 나타난 우선순위로 사용됩니다: 1) generation_config.json 모델 파일이 있는 경우; 2) 모델 구성에서. 생성을 매개변수화하려면 지정되지 않은 매개변수는 [GenerationConfig](#generationconfig)의 기본값을 상속받을 것이므로 해당 문서를 확인해야 합니다.
>
>* **logits_processor** (LogitsProcessorList, Optional) — 인수와 생성 구성에서 구축된 기본 logits 프로세서를 보완하는 사용자 정의 logits 프로세서입니다. 인수나 생성 구성으로 이미 생성된 로짓 프로세서가 전달되면 오류가 발생합니다. 이 기능은 고급 사용자를 위한 것입니다.
>* **trace** (bool, Optional, 기본값 True) — 생성을 추적할지 선택할 수 있습니다. 디버깅을 목적으로 trace값을 False로 설정할 수 있지만, 실행 시간이 상당히 느려질겁니다.
>
>* **params** (Dict[str, jnp.ndarray], Optional) — 모델 매개변수를 선택적으로 전달할 수 있습니다. 병렬 생성에 유용할 수 있습니다.
>
>* **kwargs** (Dict[str, Any], Optional) — generate_config의 ad hoc 매개변수화 및/또는 모델에 전달될 추가 모델 특정 kwargs입니다. 모델이 인코더-디코더 모델인 경우 인코더 특정 kwargs는 prefix가 없어야 하며 디코더 특정 kwargs는 decoder_로 prefix가 있어야 합니다.
>
>
>언어 모델링 헤드가 있는 모델에 대한 토큰 ID 시퀀스를 생성합니다.
