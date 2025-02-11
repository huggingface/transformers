<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# TensorFlow로 TPU에서 훈련하기[[training-on-tpu-with-tensorflow]]

<Tip>

자세한 설명이 필요하지 않고 바로 TPU 샘플 코드를 시작하고 싶다면 [우리의 TPU 예제 노트북!](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)을 확인하세요.

</Tip>

### TPU가 무엇인가요?[[what-is-a-tpu]]

TPU는 **텐서 처리 장치**입니다. Google에서 설계한 하드웨어로, GPU처럼 신경망 내에서 텐서 연산을 더욱 빠르게 처리하기 위해 사용됩니다. 네트워크 훈련과 추론 모두에 사용할 수 있습니다. 일반적으로 Google의 클라우드 서비스를 통해 이용할 수 있지만, Google Colab과 Kaggle Kernel을 통해 소규모 TPU를 무료로 직접 이용할 수도 있습니다.

[🤗 Transformers의 모든 Tensorflow 모델은 Keras 모델](https://huggingface.co/blog/tensorflow-philosophy)이기 때문에, 이 문서에서 다루는 대부분의 메소드는 대체로 모든 Keras 모델을 위한 TPU 훈련에 적용할 수 있습니다! 하지만 Transformer와 데이터 세트의 HuggingFace 생태계(hug-o-system?)에 특화된 몇 가지 사항이 있으며, 해당 사항에 대해 설명할 때 반드시 언급하도록 하겠습니다.

### 어떤 종류의 TPU가 있나요?[[what-kinds-of-tpu-are-available]]

신규 사용자는 TPU의 범위와 다양한 이용 방법에 대해 매우 혼란스러워하는 경우가 많습니다. **TPU 노드**와 **TPU VM**의 차이점은 가장 먼저 이해해야 할 핵심적인 구분 사항입니다.

**TPU 노드**를 사용한다면, 실제로는 원격 TPU를 간접적으로 이용하는 것입니다. 네트워크와 데이터 파이프라인을 초기화한 다음, 이를 원격 노드로 전달할 별도의 VM이 필요합니다. Google Colab에서 TPU를 사용하는 경우, **TPU 노드** 방식으로 이용하게 됩니다.

TPU 노드를 사용하는 것은 이를 사용하지 않는 사용자에게 예기치 않은 현상이 발생하기도 합니다! 특히, TPU는 파이썬 코드를 실행하는 기기(machine)와 물리적으로 다른 시스템에 있기 때문에 로컬 기기에 데이터를 저장할 수 없습니다. 즉, 컴퓨터의 내부 저장소에서 가져오는 데이터 파이프라인은 절대 작동하지 않습니다! 로컬 기기에 데이터를 저장하는 대신에, 데이터 파이프라인이 원격 TPU 노드에서 실행 중일 때에도 데이터 파이프라인이 계속 이용할 수 있는 Google Cloud Storage에 데이터를 저장해야 합니다.

<Tip>

메모리에 있는 모든 데이터를 `np.ndarray` 또는 `tf.Tensor`로 맞출 수 있다면, Google Cloud Storage에 업로드할 필요 없이, Colab 또는 TPU 노드를 사용해서 해당 데이터에 `fit()` 할 수 있습니다.

</Tip>

<Tip>

**🤗특수한 Hugging Face 팁🤗:** TF 코드 예제에서 볼 수 있는 `Dataset.to_tf_dataset()` 메소드와 그 상위 래퍼(wrapper)인 `model.prepare_tf_dataset()`는 모두 TPU 노드에서 작동하지 않습니다. 그 이유는 `tf.data.Dataset`을 생성하더라도 “순수한” `tf.data` 파이프라인이 아니며 `tf.numpy_function` 또는 `Dataset.from_generator()`를 사용하여 기본 HuggingFace `Dataset`에서 데이터를 전송하기 때문입니다. 이 HuggingFace `Dataset`는 로컬 디스크에 있는 데이터로 지원되며 원격 TPU 노드가 읽을 수 없습니다.

</Tip>

TPU를 이용하는 두 번째 방법은 **TPU VM**을 사용하는 것입니다. TPU VM을 사용할 때, GPU VM에서 훈련하는 것과 같이 TPU가 장착된 기기에 직접 연결합니다. 특히 데이터 파이프라인과 관련하여, TPU VM은 대체로 작업하기 더 쉽습니다. 위의 모든 경고는 TPU VM에는 해당되지 않습니다!

이 문서는 의견이 포함된 문서이며, 저희의 의견이 여기에 있습니다: **가능하면 TPU 노드를 사용하지 마세요.** TPU 노드는 TPU VM보다 더 복잡하고 디버깅하기가 더 어렵습니다. 또한 향후에는 지원되지 않을 가능성이 높습니다. Google의 최신 TPU인 TPUv4는 TPU VM으로만 이용할 수 있으므로, TPU 노드는 점점 더 "구식" 이용 방법이 될 것으로 전망됩니다. 그러나 TPU 노드를 사용하는 Colab과 Kaggle Kernel에서만 무료 TPU 이용이 가능한 것으로 확인되어, 필요한 경우 이를 다루는 방법을 설명해 드리겠습니다! 이에 대한 자세한 설명이 담긴 코드 샘플은 [TPU 예제 노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)에서 확인하시기 바랍니다.

### 어떤 크기의 TPU를 사용할 수 있나요?[[what-sizes-of-tpu-are-available]]

단일 TPU(v2-8/v3-8/v4-8)는 8개의 복제본(replicas)을 실행합니다. TPU는 수백 또는 수천 개의 복제본을 동시에 실행할 수 있는 **pod**로 존재합니다. 단일 TPU를 하나 이상 사용하지만 전체 Pod보다 적게 사용하는 경우(예를 들면, v3-32), TPU 구성을 **pod 슬라이스**라고 합니다.

Colab을 통해 무료 TPU에 이용하는 경우, 기본적으로 단일 v2-8 TPU를 제공받습니다.

### XLA에 대해 들어본 적이 있습니다. XLA란 무엇이고 TPU와 어떤 관련이 있나요?[[i-keep-hearing-about-this-xla-thing-whats-xla-and-how-does-it-relate-to-tpus]]

XLA는 최적화 컴파일러로, TensorFlow와 JAX에서 모두 사용됩니다. JAX에서는 유일한 컴파일러이지만, TensorFlow에서는 선택 사항입니다(하지만 TPU에서는 필수입니다!). Keras 모델을 훈련할 때 이를 활성화하는 가장 쉬운 방법은 `jit_compile=True` 인수를 `model.compile()`에 전달하는 것입니다. 오류가 없고 성능이 양호하다면, TPU로 전환할 준비가 되었다는 좋은 신호입니다!

TPU에서 디버깅하는 것은 대개 CPU/GPU보다 조금 더 어렵기 때문에, TPU에서 시도하기 전에 먼저 XLA로 CPU/GPU에서 코드를 실행하는 것을 권장합니다. 물론 오래 학습할 필요는 없습니다. 즉, 모델과 데이터 파이프라인이 예상대로 작동하는지 확인하기 위해 몇 단계만 거치면 됩니다.

<Tip>

XLA로 컴파일된 코드는 대체로 더 빠릅니다. 따라서 TPU에서 실행할 계획이 없더라도, `jit_compile=True`를 추가하면 성능이 향상될 수 있습니다. 하지만 XLA 호환성에 대한 아래 주의 사항을 반드시 확인하세요!

</Tip>

<Tip warning={true}>

**뼈아픈 경험에서 얻은 팁:** `jit_compile=True`를 사용하면 속도를 높이고 CPU/GPU 코드가 XLA와 호환되는지 검증할 수 있는 좋은 방법이지만, 실제 TPU에서 훈련할 때 그대로 남겨두면 많은 문제를 초래할 수 있습니다. XLA 컴파일은 TPU에서 암시적으로 이뤄지므로, 실제 TPU에서 코드를 실행하기 전에 해당 줄을 제거하는 것을 잊지 마세요!

</Tip>

### 제 XLA 모델과 호환하려면 어떻게 해야 하나요?[[how-do-i-make-my-model-xla-compatible]]

대부분의 경우, 여러분의 코드는 이미 XLA와 호환될 것입니다! 그러나 표준 TensorFlow에서 작동하지만, XLA에서는 작동하지 않는 몇 가지 사항이 있습니다. 이를 아래 세 가지 핵심 규칙으로 간추렸습니다:

<Tip>

**특수한 HuggingFace 팁🤗:** 저희는 TensorFlow 모델과 손실 함수를 XLA와 호환되도록 재작성하는 데 많은 노력을 기울였습니다. 저희의 모델과 손실 함수는 대개 기본적으로 규칙 #1과 #2를 따르므로 `transformers` 모델을 사용하는 경우, 이를 건너뛸 수 있습니다. 하지만 자체 모델과 손실 함수를 작성할 때는 이러한 규칙을 잊지 마세요!

</Tip>

#### XLA 규칙 #1: 코드에서 “데이터 종속 조건문”을 사용할 수 없습니다[[xla-rule-1-your-code-cannot-have-datadependent-conditionals]]

어떤 `if`문도 `tf.Tensor` 내부의 값에 종속될 수 없다는 것을 의미합니다. 예를 들어, 이 코드 블록은 XLA로 컴파일할 수 없습니다!

```python
if tf.reduce_sum(tensor) > 10:
    tensor = tensor / 2.0
```

처음에는 매우 제한적으로 보일 수 있지만, 대부분의 신경망 코드에서는 이를 수행할 필요가 없습니다. `tf.cond`를 사용하거나([여기](https://www.tensorflow.org/api_docs/python/tf/cond) 문서를 참조), 다음과 같이 조건문을 제거하고 대신 지표 변수를 사용하는 영리한 수학 트릭을 찾아내어 이 제한을 우회할 수 있습니다:

```python
sum_over_10 = tf.cast(tf.reduce_sum(tensor) > 10, tf.float32)
tensor = tensor / (1.0 + sum_over_10)
```

이 코드는 위의 코드와 정확히 동일한 효과를 구현하지만, 조건문을 제거하여 문제 없이 XLA로 컴파일되도록 합니다!

#### XLA 규칙 #2: 코드에서 "데이터 종속 크기"를 가질 수 없습니다[[xla-rule-2-your-code-cannot-have-datadependent-shapes]]

코드에서 모든 `tf.Tensor` 객체의 크기가 해당 값에 종속될 수 없다는 것을 의미합니다. 예를 들어, `tf.unique` 함수는 입력에서 각 고유 값의 인스턴스 하나를 포함하는 `tensor`를 반환하기 때문에 XLA로 컴파일할 수 없습니다. 이 출력의 크기는 입력 `Tensor`가 얼마나 반복적인지에 따라 분명히 달라질 것이므로, XLA는 이를 처리하지 못합니다!

일반적으로, 대부분의 신경망 코드는 기본값으로 규칙 2를 따릅니다. 그러나 문제가 되는 몇 가지 대표적인 사례가 있습니다. 가장 흔한 사례 중 하나는 **레이블 마스킹**을 사용하여 손실(loss)을 계산할 때, 해당 위치를 무시하도록 나타내기 위해 레이블을 음수 값으로 설정하는 경우입니다. 레이블 마스킹을 지원하는 NumPy나 PyTorch 손실 함수를 보면 [불 인덱싱](https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing)을 사용하는 다음과 같은 코드를 자주 접할 수 있습니다:

```python
label_mask = labels >= 0
masked_outputs = outputs[label_mask]
masked_labels = labels[label_mask]
loss = compute_loss(masked_outputs, masked_labels)
mean_loss = torch.mean(loss)
```

이 코드는 NumPy나 PyTorch에서는 문제 없이 작동하지만, XLA에서는 손상됩니다! 왜 그럴까요? 얼마나 많은 위치가 마스킹되는지에 따라 `masked_outputs`와 `masked_labels`의 크기가 달라져서, **데이터 종속 크기**가 되기 때문입니다. 그러나 규칙 #1과 마찬가지로, 이 코드를 다시 작성하면 데이터 종속적 모양 크기가 정확히 동일한 출력을 산출할 수 있습니다.

```python
label_mask = tf.cast(labels >= 0, tf.float32)
loss = compute_loss(outputs, labels)
loss = loss * label_mask  # Set negative label positions to 0
mean_loss = tf.reduce_sum(loss) / tf.reduce_sum(label_mask)
```

여기서, 모든 위치에 대한 손실을 계산하지만, 평균을 계산할 때 분자와 분모 모두에서 마스크된 위치를 0으로 처리합니다. 이는 데이터 종속 크기를 방지하고 XLA 호환성을 유지하면서 첫 번째 블록과 정확히 동일한 결과를 산출합니다. 규칙 #1에서와 동일한 트릭을 사용하여 `tf.bool`을 `tf.float32`로 변환하고 이를 지표 변수로 사용합니다. 해당 트릭은 매우 유용하며, 자체 코드를 XLA로 변환해야 할 경우 기억해 두세요!

#### XLA 규칙 #3: XLA는 각기 다른 입력 크기가 나타날 때마다 모델을 다시 컴파일해야 합니다[[xla-rule-3-xla-will-need-to-recompile-your-model-for-every-different-input-shape-it-sees]]

이것은 가장 큰 문제입니다. 입력 크기가 매우 가변적인 경우, XLA는 모델을 반복해서 다시 컴파일해야 하므로 성능에 큰 문제가 발생할 수 있습니다. 이 문제는 토큰화 후 입력 텍스트의 길이가 가변적인 NLP 모델에서 주로 발생합니다. 다른 모달리티에서는 정적 크기가 더 흔하며, 해당 규칙이 훨씬 덜 문제시 됩니다.

규칙 #3을 어떻게 우회할 수 있을까요? 핵심은 **패딩**입니다. 모든 입력을 동일한 길이로 패딩한 다음, `attention_mask`를 사용하면 어떤 XLA 문제도 없이 가변 크기에서 가져온 것과 동일한 결과를 가져올 수 있습니다. 그러나 과도한 패딩은 심각한 속도 저하를 야기할 수도 있습니다. 모든 샘플을 전체 데이터 세트의 최대 길이로 패딩하면, 무한한 패딩 토큰으로 구성된 배치가 생성되어 많은 연산과 메모리가 낭비될 수 있습니다!

이 문제에 대한 완벽한 해결책은 없습니다. 하지만, 몇 가지 트릭을 시도해볼 수 있습니다. 한 가지 유용한 트릭은 **샘플 배치를 32 또는 64 토큰과 같은 숫자의 배수까지 패딩하는 것입니다.** 이는 토큰 수가 소폭 증가하지만, 모든 입력 크기가 32 또는 64의 배수여야 하기 때문에 고유한 입력 크기의 수가 대폭 줄어듭니다. 고유한 입력 크기가 적다는 것은 XLA 컴파일 횟수가 적어진다는 것을 의미합니다!

<Tip>

**🤗특수한 HuggingFace 팁🤗:** 토크나이저와 데이터 콜레이터에 도움이 될 수 있는 메소드가 있습니다. 토크나이저를 불러올 때 `padding="max_length"` 또는 `padding="longest"`를 사용하여 패딩된 데이터를 출력하도록 할 수 있습니다. 토크나이저와 데이터 콜레이터는 나타나는 고유한 입력 크기의 수를 줄이기 위해 사용할 수 있는 `pad_to_multiple_of` 인수도 있습니다!

</Tip>

### 실제 TPU로 모델을 훈련하려면 어떻게 해야 하나요?[[how-do-i-actually-train-my-model-on-tpu]]

훈련이 XLA와 호환되고 (TPU 노드/Colab을 사용하는 경우) 데이터 세트가 적절하게 준비되었다면, TPU에서 실행하는 것은 놀랍도록 쉽습니다! 코드에서 몇 줄만 추가하여, TPU를 초기화하고 모델과 데이터 세트가 `TPUStrategy` 범위 내에 생성되도록 변경하면 됩니다. [우리의 TPU 예제 노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb)을 참조하여 실제로 작동하는 모습을 확인해 보세요!

### 요약[[summary]]

여기에 많은 내용이 포함되어 있으므로, TPU 훈련을 위한 모델을 준비할 때 따를 수 있는 간략한 체크리스트로 요약해 보겠습니다:

- 코드가 XLA의 세 가지 규칙을 따르는지 확인합니다.
- CPU/GPU에서 `jit_compile=True`로 모델을 컴파일하고 XLA로 훈련할 수 있는지 확인합니다.
- 데이터 세트를 메모리에 가져오거나 TPU 호환 데이터 세트를 가져오는 방식을 사용합니다([노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb) 참조)
- 코드를 Colab(accelerator가 “TPU”로 설정됨) 또는 Google Cloud의 TPU VM으로 마이그레이션합니다.
- TPU 초기화 코드를 추가합니다([노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb) 참조)
- `TPUStrategy`를 생성하고 데이터 세트를 가져오는 것과 모델 생성이 `strategy.scope()` 내에 있는지 확인합니다([노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/tpu_training-tf.ipynb) 참조)
- TPU로 이동할 때 `jit_compile=True`를 다시 설정하는 것을 잊지 마세요!
- 🙏🙏🙏🥺🥺🥺
- model.fit()을 불러옵니다.
- 여러분이 해냈습니다!