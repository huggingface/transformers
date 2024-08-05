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

# 큰 모델 인스턴스화 [[instantiating-a-big-model]]

매우 큰 사전훈련된 모델을 사용하려면, RAM 사용을 최소화해야 하는 과제가 있습니다. 일반적인 PyTorch 워크플로우는 다음과 같습니다:

1. 무작위 가중치로 모델을 생성합니다.
2. 사전훈련된 가중치를 불러옵니다.
3. 사전훈련된 가중치를 무작위 모델에 적용합니다.

1단계와 2단계 모두 모델의 전체 버전을 메모리에 적재해야 하며, 대부분 문제가 없지만 모델이 기가바이트급의 용량을 차지하기 시작하면 복사본 2개가 RAM을 초과하여 메모리 부족 이슈를 야기할 수 있습니다. 더 심각한 문제는 분산 학습을 위해 `torch.distributed`를 사용하는 경우, 프로세스마다 사전훈련된 모델을 로드하고 복사본을 2개씩 RAM에 저장한다는 것입니다.

<Tip>

무작위로 생성된 모델은 "비어 있는" (즉 그때 메모리에 있던 것으로 이뤄진) 텐서로 초기화되며 메모리 공간을 차지합니다. 초기화된 모델/파라미터의 종류에 적합한 분포(예: 정규 분포)에 따른 무작위 초기화는 가능한 한 빠르게 하기 위해 초기화되지 않은 가중치에 대해 3단계 이후에만 수행됩니다!

</Tip>

이 안내서에서는 Transformers가 이 문제를 해결하기 위해 제공하는 솔루션을 살펴봅니다. 주의할 점은 아직 활발히 개발 중인 분야이므로 여기서 설명하는 API가 앞으로 약간 변경될 수 있다는 것입니다.

## 샤딩된 체크포인트 [[sharded-checkpoints]]

4.18.0 버전 이후, 10GB 이상의 공간을 차지하는 모델 체크포인트는 자동으로 작은 조각들로 샤딩됩니다. `model.save_pretrained(save_dir)`를 실행할 때 하나의 단일 체크포인트를 가지게 될 대신, 여러 부분 체크포인트(각각의 크기는 10GB 미만)와 매개변수 이름을 해당 파일에 매핑하는 인덱스가 생성됩니다.

`max_shard_size` 매개변수로 샤딩 전 최대 크기를 제어할 수 있으므로, 이 예제를 위해 샤드 크기가 작은 일반 크기의 모델을 사용하겠습니다: 전통적인 BERT 모델을 사용해 봅시다.

```py
from transformers import AutoModel

model = AutoModel.from_pretrained("google-bert/bert-base-cased")
```

[`~PreTrainedModel.save_pretrained`]을 사용하여 모델을 저장하면, 모델의 구성과 가중치가 들어있는 두 개의 파일이 있는 새 폴더가 생성됩니다:

```py
>>> import os
>>> import tempfile

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir)
...     print(sorted(os.listdir(tmp_dir)))
['config.json', 'pytorch_model.bin']
```

이제 최대 샤드 크기를 200MB로 사용해 봅시다:

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     print(sorted(os.listdir(tmp_dir)))
['config.json', 'pytorch_model-00001-of-00003.bin', 'pytorch_model-00002-of-00003.bin', 'pytorch_model-00003-of-00003.bin', 'pytorch_model.bin.index.json']
```

모델의 구성에 더해, 세 개의 다른 가중치 파일과 파라미터 이름과 해당 파일의 매핑이 포함된 `index.json` 파일을 볼 수 있습니다. 이러한 체크포인트는 [`~PreTrainedModel.from_pretrained`] 메서드를 사용하여 완전히 다시 로드할 수 있습니다:

```py
>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     new_model = AutoModel.from_pretrained(tmp_dir)
```

큰 모델의 경우 이러한 방식으로 처리하는 주된 장점은 위에서 보여준 흐름의 2단계에서, 각 샤드가 이전 샤드 다음에 로드되므로 메모리 사용량이 모델 크기와 가장 큰 샤드의 크기를 초과하지 않는다는 점입니다.

이 인덱스 파일은 키가 체크포인트에 있는지, 그리고 해당 가중치가 어디에 저장되어 있는지를 결정하는 데 사용됩니다. 이 인덱스를 json과 같이 로드하고 딕셔너리를 얻을 수 있습니다:

```py
>>> import json

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     with open(os.path.join(tmp_dir, "pytorch_model.bin.index.json"), "r") as f:
...         index = json.load(f)

>>> print(index.keys())
dict_keys(['metadata', 'weight_map'])
```

메타데이터는 현재 모델의 총 크기만 포함됩니다. 앞으로 다른 정보를 추가할 계획입니다:

```py
>>> index["metadata"]
{'total_size': 433245184}
```

가중치 맵은 이 인덱스의 주요 부분으로, 각 매개변수 이름(PyTorch 모델 `state_dict`에서 보통 찾을 수 있는)을 해당 파일에 매핑합니다:

```py
>>> index["weight_map"]
{'embeddings.LayerNorm.bias': 'pytorch_model-00001-of-00003.bin',
 'embeddings.LayerNorm.weight': 'pytorch_model-00001-of-00003.bin',
 ...
```

만약 [`~PreTrainedModel.from_pretrained`]를 사용하지 않고 모델 내에서 이러한 샤딩된 체크포인트를 직접 가져오려면 (전체 체크포인트를 위해 `model.load_state_dict()`를 수행하는 것처럼), [`~modeling_utils.load_sharded_checkpoint`]를 사용해야 합니다.

```py
>>> from transformers.modeling_utils import load_sharded_checkpoint

>>> with tempfile.TemporaryDirectory() as tmp_dir:
...     model.save_pretrained(tmp_dir, max_shard_size="200MB")
...     load_sharded_checkpoint(model, tmp_dir)
```

## 저(低)메모리 로딩 [[low-memory-loading]]

샤딩된 체크포인트는 위에서 언급한 작업 흐름의 2단계에서 메모리 사용량을 줄이지만, 저(低)메모리 설정에서 모델을 사용하기 위해 우리의 Accelerate 라이브러리를 기반으로 한 도구를 활용하는 것이 좋습니다.

자세한 사항은 다음 가이드를 참조해주세요: [Accelerate로 대규모 모델 가져오기 (영문)](../en/main_classes/model#large-model-loading)