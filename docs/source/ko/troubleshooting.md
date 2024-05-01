<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 문제 해결[[troubleshoot]]

때때로 오류가 발생할 수 있지만, 저희가 도와드리겠습니다! 이 가이드는 현재까지 확인된 가장 일반적인 문제 몇 가지와 그것들을 해결하는 방법에 대해 다룹니다. 그러나 이 가이드는 모든 🤗 Transformers 문제를 포괄적으로 다루고 있지 않습니다. 문제 해결에 더 많은 도움을 받으려면 다음을 시도해보세요:

<Youtube id="S2EEG3JIt2A"/>

1. [포럼](https://discuss.huggingface.co/)에서 도움을 요청하세요. [Beginners](https://discuss.huggingface.co/c/beginners/5) 또는 [🤗 Transformers](https://discuss.huggingface.co/c/transformers/9)와 같은 특정 카테고리에 질문을 게시할 수 있습니다. 재현 가능한 코드와 함께 잘 서술된 포럼 게시물을 작성하여 여러분의 문제가 해결될 가능성을 극대화하세요!

<Youtube id="_PAli-V4wj0"/>

2. 라이브러리와 관련된 버그이면 🤗 Transformers 저장소에서 [이슈](https://github.com/huggingface/transformers/issues/new/choose)를 생성하세요. 버그에 대해 설명하는 정보를 가능한 많이 포함하려고 노력하여, 무엇이 잘못 되었는지와 어떻게 수정할 수 있는지 더 잘 파악할 수 있도록 도와주세요.

3. 이전 버전의 🤗 Transformers을 사용하는 경우 중요한 변경 사항이 버전 사이에 도입되었기 때문에 [마이그레이션](migration) 가이드를 확인하세요.

문제 해결 및 도움 매뉴얼에 대한 자세한 내용은 Hugging Face 강좌의 [8장](https://huggingface.co/course/chapter8/1?fw=pt)을 참조하세요.


## 방화벽 환경[[firewalled-environments]]

클라우드 및 내부망(intranet) 설정의 일부 GPU 인스턴스는 외부 연결에 대한 방화벽으로 차단되어 연결 오류가 발생할 수 있습니다. 스크립트가 모델 가중치나 데이터를 다운로드하려고 할 때, 다운로드가 중단되고 다음 메시지와 함께 시간 초과됩니다: 

```
ValueError: Connection error, and we cannot find the requested files in the cached path.
Please try again or make sure your Internet connection is on.
```

이 경우에는 연결 오류를 피하기 위해 🤗 Transformers를 [오프라인 모드](installation#offline-mode)로 실행해야 합니다.

## CUDA 메모리 부족(CUDA out of memory)[[cuda-out-of-memory]]

수백만 개의 매개변수로 대규모 모델을 훈련하는 것은 적절한 하드웨어 없이 어려울 수 있습니다. GPU 메모리가 부족한 경우 발생할 수 있는 일반적인 오류는 다음과 같습니다:

```
CUDA out of memory. Tried to allocate 256.00 MiB (GPU 0; 11.17 GiB total capacity; 9.70 GiB already allocated; 179.81 MiB free; 9.85 GiB reserved in total by PyTorch)
```

다음은 메모리 사용을 줄이기 위해 시도해 볼 수 있는 몇 가지 잠재적인 해결책입니다:

- [`TrainingArguments`]의 [`per_device_train_batch_size`](main_classes/trainer#transformers.TrainingArguments.per_device_train_batch_size) 값을 줄이세요.
- [`TrainingArguments`]의 [`gradient_accumulation_steps`](main_classes/trainer#transformers.TrainingArguments.gradient_accumulation_steps)은 전체 배치 크기를 효과적으로 늘리세요.

<Tip>

메모리 절약 기술에 대한 자세한 내용은 성능 [가이드](performance)를 참조하세요.

</Tip>

## 저장된 TensorFlow 모델을 가져올 수 없습니다(Unable to load a saved TensorFlow model)[[unable-to-load-a-saved-uensorFlow-model]]

TensorFlow의 [model.save](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model) 메소드는 아키텍처, 가중치, 훈련 구성 등 전체 모델을 단일 파일에 저장합니다. 그러나 모델 파일을 다시 가져올 때 🤗 Transformers는 모델 파일에 있는 모든 TensorFlow 관련 객체를 가져오지 않을 수 있기 때문에 오류가 발생할 수 있습니다. TensorFlow 모델 저장 및 가져오기 문제를 피하려면 다음을 권장합니다:

- 모델 가중치를 `h5` 파일 확장자로 [`model.save_weights`](https://www.tensorflow.org/tutorials/keras/save_and_load#save_the_entire_model)로 저장한 다음 [`~TFPreTrainedModel.from_pretrained`]로 모델을 다시 가져옵니다:

```py
>>> from transformers import TFPreTrainedModel
>>> from tensorflow import keras

>>> model.save_weights("some_folder/tf_model.h5")
>>> model = TFPreTrainedModel.from_pretrained("some_folder")
```

- 모델을 [`~TFPretrainedModel.save_pretrained`]로 저장하고 [`~TFPreTrainedModel.from_pretrained`]로 다시 가져옵니다:

```py
>>> from transformers import TFPreTrainedModel

>>> model.save_pretrained("path_to/model")
>>> model = TFPreTrainedModel.from_pretrained("path_to/model")
```

## ImportError[[importerror]]

특히 최신 모델인 경우 만날 수 있는 다른 일반적인 오류는 `ImportError`입니다:

```
ImportError: cannot import name 'ImageGPTImageProcessor' from 'transformers' (unknown location)
```

이러한 오류 유형의 경우 최신 모델에 액세스할 수 있도록 최신 버전의 🤗 Transformers가 설치되어 있는지 확인하세요:

```bash
pip install transformers --upgrade
```

## CUDA error: device-side assert triggered[[cuda-error-deviceside-assert-triggered]]

때때로 장치 코드 오류에 대한 일반적인 CUDA 오류가 발생할 수 있습니다.

```
RuntimeError: CUDA error: device-side assert triggered
```

더 자세한 오류 메시지를 얻으려면 우선 코드를 CPU에서 실행합니다. 다음 환경 변수를 코드의 시작 부분에 추가하여 CPU로 전환하세요:

```py
>>> import os

>>> os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

또 다른 옵션은 GPU에서 더 나은 역추적(traceback)을 얻는 것입니다. 다음 환경 변수를 코드의 시작 부분에 추가하여 역추적이 오류가 발생한 소스를 가리키도록 하세요:

```py
>>> import os

>>> os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

## 패딩 토큰이 마스킹되지 않은 경우 잘못된 출력(Incorrect output when padding tokens aren't masked)[[incorrect-output-when-padding-tokens-arent-masked]]

경우에 따라 `input_ids`에 패딩 토큰이 포함된 경우 `hidden_state` 출력이 올바르지 않을 수 있습니다. 데모를 위해 모델과 토크나이저를 가져오세요. 모델의 `pad_token_id`에 액세스하여 해당 값을 확인할 수 있습니다. 일부 모델의 경우 `pad_token_id`가 `None`일 수 있지만 언제든지 수동으로 설정할 수 있습니다.

```py
>>> from transformers import AutoModelForSequenceClassification
>>> import torch

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")
>>> model.config.pad_token_id
0
```

다음 예제는 패딩 토큰을 마스킹하지 않은 출력을 보여줍니다:

```py
>>> input_ids = torch.tensor([[7592, 2057, 2097, 2393, 9611, 2115], [7592, 0, 0, 0, 0, 0]])
>>> output = model(input_ids)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
        [ 0.1317, -0.1683]], grad_fn=<AddmmBackward0>)
```

다음은 두 번째 시퀀스의 실제 출력입니다:

```py
>>> input_ids = torch.tensor([[7592]])
>>> output = model(input_ids)
>>> print(output.logits)
tensor([[-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

대부분의 경우 모델에 `attention_mask`를 제공하여 패딩 토큰을 무시해야 이러한 조용한 오류를 방지할 수 있습니다. 이제 두 번째 시퀀스의 출력이 실제 출력과 일치합니다:

<Tip>

일반적으로 토크나이저는 특정 토크나이저의 기본 값을 기준으로 사용자에 대한 'attention_mask'를 만듭니다.

</Tip>

```py
>>> attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]])
>>> output = model(input_ids, attention_mask=attention_mask)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
        [-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

🤗 Transformers는 패딩 토큰이 제공된 경우 패딩 토큰을 마스킹하기 위한 `attention_mask`를 자동으로 생성하지 않습니다. 그 이유는 다음과 같습니다:

- 일부 모델에는 패딩 토큰이 없습니다.
- 일부 사용 사례의 경우 사용자가 모델이 패딩 토큰을 관리하기를 원합니다.

## ValueError: 이 유형의 AutoModel에 대해 인식할 수 없는 XYZ 구성 클래스(ValueError: Unrecognized configuration class XYZ for this kind of AutoModel)[[valueerror-unrecognized-configuration-class-xyz-for-this-kind-of-automodel]]

일반적으로, 사전 학습된 모델의 인스턴스를 가져오기 위해 [`AutoModel`] 클래스를 사용하는 것이 좋습니다.
이 클래스는 구성에 따라 주어진 체크포인트에서 올바른 아키텍처를 자동으로 추론하고 가져올 수 있습니다.
모델을 체크포인트에서 가져올 때 이 `ValueError`가 발생하면, 이는 Auto 클래스가 주어진 체크포인트의 구성에서 
가져오려는 모델 유형과 매핑을 찾을 수 없다는 것을 의미합니다. 가장 흔하게 발생하는 경우는 
체크포인트가 주어진 태스크를 지원하지 않을 때입니다.
예를 들어, 다음 예제에서 질의응답에 대한 GPT2가 없기 때문에 오류가 발생합니다:

```py
>>> from transformers import AutoProcessor, AutoModelForQuestionAnswering

>>> processor = AutoProcessor.from_pretrained("openai-community/gpt2-medium")
>>> model = AutoModelForQuestionAnswering.from_pretrained("openai-community/gpt2-medium")
ValueError: Unrecognized configuration class <class 'transformers.models.gpt2.configuration_gpt2.GPT2Config'> for this kind of AutoModel: AutoModelForQuestionAnswering.
Model type should be one of AlbertConfig, BartConfig, BertConfig, BigBirdConfig, BigBirdPegasusConfig, BloomConfig, ...
```
