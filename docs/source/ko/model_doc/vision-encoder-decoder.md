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

# Vision Encoder Decoder Models

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">
<img alt="Flax" src="https://img.shields.io/badge/Flax-29a79b.svg?style=flat&logo=data:image/png;base64,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
">
<img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

## 개요[[overview]]

[`VisionEncoderDecoderModel`]은 사전 훈련된 Transformer 기반 비전 모델을 인코더로 (*예:* [ViT](vit), [BEiT](beit), [DeiT](deit), [Swin](swin)),
사전 훈련된 언어 모델을 디코더로 (*예:* [RoBERTa](roberta), [GPT2](gpt2), [BERT](bert), [DistilBERT](distilbert)) 사용하여 이미지-텍스트 모델을 초기화하는 데 사용할 수 있습니다.

사전 훈련된 체크포인트로 이미지-텍스트 시퀀스 모델을 초기화하는 효과는 (예를 들어) Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang,
Zhoujun Li, Furu Wei의 [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://huggingface.co/papers/2109.10282)에서 입증되었습니다.

이러한 [`VisionEncoderDecoderModel`]이 훈련/파인튜닝된 후에는 다른 모델들과 마찬가지로 저장/로드할 수 있습니다 (자세한 내용은 아래 예제를 참조하세요).

응용 예시로는 이미지 캡셔닝이 있습니다. 여기서는 인코더가 이미지를 인코딩하고, 이후 자기회귀 언어 모델이 캡션을 생성합니다. 다른 예시로는 광학 문자 인식이 있습니다. [`VisionEncoderDecoderModel`]의 인스턴스인 [TrOCR](trocr)을 참조하세요.

## 모델 설정에서 `VisionEncoderDecoderModel` 랜덤 초기화[[randomly-initializing-visionencoderdecodermodel-from-model-configurations]]

[`VisionEncoderDecoderModel`]은 인코더와 디코더 설정에서 랜덤하게 초기화할 수 있습니다. 다음 예제에서는 인코더에 기본 [`ViTModel`] 설정을,
디코더에 기본 [`BertForCausalLM`] 설정을 사용하여 이를 수행하는 방법을 보여줍니다.

```python
>>> from transformers import BertConfig, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

>>> config_encoder = ViTConfig()
>>> config_decoder = BertConfig()

>>> config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
>>> model = VisionEncoderDecoderModel(config=config)
```

## 사전 훈련된 인코더와 디코더에서 `VisionEncoderDecoderModel` 초기화[[initialising-visionencoderdecodermodel-from-a-pretrained-encoder-and-a-pretrained-decoder]]

[`VisionEncoderDecoderModel`]은 사전 훈련된 인코더 체크포인트와 사전 훈련된 디코더 체크포인트에서 초기화할 수 있습니다. 사전 훈련된 Transformer 기반 비전 모델(*예:* [Swin](swin))은 인코더 역할을 할 수 있으며, 사전 훈련된 자동 인코딩 모델(*예:* BERT), 사전 훈련된 인과적 언어 모델(*예:* GPT2), 그리고 시퀀스-투-시퀀스 모델의 사전 훈련된 디코더 부분(*예:* BART의 디코더)은 모두 디코더로 사용할 수 있습니다.
디코더로 선택하는 아키텍처에 따라 교차 주의(cross-attention) 레이어가 랜덤하게 초기화될 수 있습니다.
사전 훈련된 인코더와 디코더 체크포인트에서 [`VisionEncoderDecoderModel`]을 초기화하려면 [*Warm-starting-encoder-decoder blog post*](https://huggingface.co/blog/warm-starting-encoder-decoder)에서 보여준 바와 같이 다운스트림 작업에서 모델을 파인튜닝해야 합니다.
이를 위해 `VisionEncoderDecoderModel` 클래스는 [`VisionEncoderDecoderModel.from_encoder_decoder_pretrained`] 메서드를 제공합니다.

```python
>>> from transformers import VisionEncoderDecoderModel

>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "microsoft/swin-base-patch4-window7-224-in22k", "google-bert/bert-base-uncased"
... )
```

## 기존 `VisionEncoderDecoderModel` 체크포인트 로드 및 추론 수행[[loading-an-existing-visionencoderdecodermodel-checkpoint-and-perform-inference]]

`VisionEncoderDecoderModel` 클래스의 파인튜닝된 체크포인트를 로드하기 위해, [`VisionEncoderDecoderModel`]은 Transformers의 다른 모델 아키텍처와 마찬가지로 `from_pretrained(...)` 메서드를 제공합니다.

추론을 수행하려면 [`generate`] 메서드를 사용합니다. 이 메서드는 자기회귀적으로 텍스트를 생성할 수 있습니다. 이 메서드는 그리디(greedy), 빔 서치(beam search), 다항 샘플링(multinomial sampling) 등 다양한 형태의 디코딩을 지원합니다.

```python
>>> import requests
>>> from PIL import Image

>>> from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel

>>> # 파인튜닝된 이미지 캡셔닝 모델과 해당 토크나이저 및 이미지 프로세서를 로드합니다
>>> model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
>>> tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
>>> image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

>>> # 이미지에 대해 추론을 수행해보겠습니다
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values

>>> # 자기회귀적으로 캡션 생성 (기본적으로 그리디 디코딩 사용)
>>> generated_ids = model.generate(pixel_values)
>>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
>>> print(generated_text)
a cat laying on a blanket next to a cat laying on a bed
```

## PyTorch 체크포인트를 `TFVisionEncoderDecoderModel`로 로드[[loading-a-pytorch-checkpoint-into-tfvisionencoderdecodermodel]]

[`TFVisionEncoderDecoderModel.from_pretrained`]는 현재 PyTorch 체크포인트에서 모델을 초기화하는 것을 지원하지 않습니다. 이 메서드에 `from_pt=True`를 전달하면 예외가 발생합니다. 특정 비전 인코더-디코더 모델에 PyTorch 체크포인트만 있는 경우, 해결 방법은 다음과 같습니다:

```python
>>> from transformers import VisionEncoderDecoderModel, TFVisionEncoderDecoderModel

>>> _model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

>>> _model.encoder.save_pretrained("./encoder")
>>> _model.decoder.save_pretrained("./decoder")

>>> model = TFVisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "./encoder", "./decoder", encoder_from_pt=True, decoder_from_pt=True
... )
>>> # 이것은 이 특정 모델의 몇 가지 특정 속성을 복사하기 위한 것입니다.
>>> model.config = _model.config
```

## 훈련[[training]]

모델이 생성된 후에는 (이미지, 텍스트) 쌍의 데이터셋에서 BART, T5 또는 다른 인코더-디코더 모델과 유사하게 파인튜닝할 수 있습니다.
보시다시피, 손실을 계산하기 위해 모델에 필요한 입력은 단 2개입니다: `pixel_values` (이미지)와 `labels` (인코딩된 대상 시퀀스의 `input_ids`)입니다.

```python
>>> from transformers import ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel
>>> from datasets import load_dataset

>>> image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
>>> tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
>>> model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
...     "google/vit-base-patch16-224-in21k", "google-bert/bert-base-uncased"
... )

>>> model.config.decoder_start_token_id = tokenizer.cls_token_id
>>> model.config.pad_token_id = tokenizer.pad_token_id

>>> dataset = load_dataset("huggingface/cats-image")
>>> image = dataset["test"]["image"][0]
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values

>>> labels = tokenizer(
...     "an image of two cats chilling on a couch",
...     return_tensors="pt",
... ).input_ids

>>> # forward 함수가 자동으로 올바른 decoder_input_ids를 생성합니다
>>> loss = model(pixel_values=pixel_values, labels=labels).loss
```

이 모델은 [nielsr](https://github.com/nielsrogge)이 기여했습니다. 이 모델의 TensorFlow와 Flax 버전은
[ydshieh](https://github.com/ydshieh)가 기여했습니다.

## VisionEncoderDecoderConfig

[[autodoc]] VisionEncoderDecoderConfig

<frameworkcontent>
<pt>

## VisionEncoderDecoderModel

[[autodoc]] VisionEncoderDecoderModel
    - forward
    - from_encoder_decoder_pretrained

</pt>
<tf>

## TFVisionEncoderDecoderModel

[[autodoc]] TFVisionEncoderDecoderModel
    - call
    - from_encoder_decoder_pretrained

</tf>
<jax>

## FlaxVisionEncoderDecoderModel

[[autodoc]] FlaxVisionEncoderDecoderModel
    - __call__
    - from_encoder_decoder_pretrained

</jax>
</frameworkcontent>
