<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ONNX로 내보내기 [[export-to-onnx]]

🤗 Transformers 모델을 제품 환경에서 배포하기 위해서는 모델을 직렬화된 형식으로 내보내고 특정 런타임과 하드웨어에서 로드하고 실행할 수 있으면 유용합니다.

🤗 Optimum은 Transformers의 확장으로, PyTorch 또는 TensorFlow에서 모델을 ONNX와 TFLite와 같은 직렬화된 형식으로 내보낼 수 있도록 하는 `exporters` 모듈을 통해 제공됩니다. 🤗 Optimum은 또한 성능 최적화 도구 세트를 제공하여 특정 하드웨어에서 모델을 훈련하고 실행할 때 최대 효율성을 달성할 수 있습니다.

이 안내서는 🤗 Optimum을 사용하여 🤗 Transformers 모델을 ONNX로 내보내는 방법을 보여줍니다. TFLite로 모델을 내보내는 안내서는 [TFLite로 내보내기 페이지](tflite)를 참조하세요.

## ONNX로 내보내기 [[export-to-onnx]]

[ONNX (Open Neural Network eXchange)](http://onnx.ai)는 PyTorch와 TensorFlow를 포함한 다양한 프레임워크에서 심층 학습 모델을 나타내는 데 사용되는 공통 연산자 세트와 공통 파일 형식을 정의하는 오픈 표준입니다. 모델이 ONNX 형식으로 내보내지면 이러한 연산자를 사용하여 신경망을 통해 데이터가 흐르는 흐름을 나타내는 계산 그래프(일반적으로 _중간 표현_이라고 함)가 구성됩니다.

표준화된 연산자와 데이터 유형을 가진 그래프를 노출함으로써, ONNX는 프레임워크 간에 쉽게 전환할 수 있습니다. 예를 들어, PyTorch에서 훈련된 모델을 ONNX 형식으로 내보내고 TensorFlow에서 가져올 수 있습니다(그 반대도 가능합니다).

ONNX 형식으로 내보낸 모델은 다음과 같이 사용할 수 있습니다:
- [그래프 최적화](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization) 및 [양자화](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization)와 같은 기법을 사용하여 추론을 위해 최적화됩니다.
- ONNX Runtime을 통해 실행할 수 있습니다. [`ORTModelForXXX` 클래스들](https://huggingface.co/docs/optimum/onnxruntime/package_reference/modeling_ort)을 통해 동일한 `AutoModel` API를 따릅니다. 이 API는 🤗 Transformers에서 사용하는 것과 동일합니다.
- [최적화된 추론 파이프라인](https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/pipelines)을 사용할 수 있습니다. 이는 🤗 Transformers의 [`pipeline`] 함수와 동일한 API를 가지고 있습니다.

🤗 Optimum은 구성 객체를 활용하여 ONNX 내보내기를 지원합니다. 이러한 구성 객체는 여러 모델 아키텍처에 대해 미리 준비되어 있으며 다른 아키텍처에 쉽게 확장할 수 있도록 설계되었습니다.

미리 준비된 구성 목록은 [🤗 Optimum 문서](https://huggingface.co/docs/optimum/exporters/onnx/overview)를 참조하세요.

🤗 Transformers 모델을 ONNX로 내보내는 두 가지 방법이 있습니다. 여기에서 두 가지 방법을 모두 보여줍니다:

- 🤗 Optimum을 사용하여 CLI로 내보내기
- `optimum.onnxruntime`을 사용하여 🤗 Optimum으로 ONNX로 내보내기

### CLI를 사용하여 🤗 Transformers 모델을 ONNX로 내보내기 [[exporting-a-transformers-model-to-onnx-with-cli]]

🤗 Transformers 모델을 ONNX로 내보내려면 먼저 추가 종속성을 설치하세요:

```bash
pip install optimum[exporters]
```

사용 가능한 모든 인수를 확인하려면 [🤗 Optimum 문서](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli)를 참조하거나 명령줄에서 도움말을 보세요.

```bash
optimum-cli export onnx --help
```

예를 들어, 🤗 Hub에서 `distilbert-base-uncased-distilled-squad`와 같은 모델의 체크포인트를 내보내려면 다음 명령을 실행하세요:

```bash
optimum-cli export onnx --model distilbert-base-uncased-distilled-squad distilbert_base_uncased_squad_onnx/
```

위와 같이 진행 상황을 나타내는 로그가 표시되고 결과인 `model.onnx`가 저장된 위치가 표시됩니다.

```bash
Validating ONNX model distilbert_base_uncased_squad_onnx/model.onnx...
	-[✓] ONNX model output names match reference model (start_logits, end_logits)
	- Validating ONNX Model output "start_logits":
		-[✓] (2, 16) matches (2, 16)
		-[✓] all values close (atol: 0.0001)
	- Validating ONNX Model output "end_logits":
		-[✓] (2, 16) matches (2, 16)
		-[✓] all values close (atol: 0.0001)
The ONNX export succeeded and the exported model was saved at: distilbert_base_uncased_squad_onnx
```

위의 예제는 🤗 Hub에서 체크포인트를 내보내는 것을 설명합니다. 로컬 모델을 내보낼 때에는 모델의 가중치와 토크나이저 파일을 동일한 디렉토리(`local_path`)에 저장했는지 확인하세요. CLI를 사용할 때에는 🤗 Hub의 체크포인트 이름 대신 `model` 인수에 `local_path`를 전달하고 `--task` 인수를 제공하세요. 지원되는 작업의 목록은 [🤗 Optimum 문서](https://huggingface.co/docs/optimum/exporters/task_manager)를 참조하세요. `task` 인수가 제공되지 않으면 작업에 특화된 헤드 없이 모델 아키텍처로 기본 설정됩니다.

```bash
optimum-cli export onnx --model local_path --task question-answering distilbert_base_uncased_squad_onnx/
```

그 결과로 생성된 `model.onnx` 파일은 ONNX 표준을 지원하는 많은 [가속기](https://onnx.ai/supported-tools.html#deployModel) 중 하나에서 실행할 수 있습니다. 예를 들어, [ONNX Runtime](https://onnxruntime.ai/)을 사용하여 모델을 로드하고 실행할 수 있습니다:

```python
>>> from transformers import AutoTokenizer
>>> from optimum.onnxruntime import ORTModelForQuestionAnswering

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> model = ORTModelForQuestionAnswering.from_pretrained("distilbert_base_uncased_squad_onnx")
>>> inputs = tokenizer("What am I using?", "Using DistilBERT with ONNX Runtime!", return_tensors="pt")
>>> outputs = model(**inputs)
```

Hub의 TensorFlow 체크포인트에 대해서도 동일한 프로세스가 적용됩니다. 예를 들어, [Keras organization](https://huggingface.co/keras-io)에서 순수한 TensorFlow 체크포인트를 내보내는 방법은 다음과 같습니다:

```bash
optimum-cli export onnx --model keras-io/transformers-qa distilbert_base_cased_squad_onnx/
```

### `optimum.onnxruntime`을 사용하여 🤗 Transformers 모델을 ONNX로 내보내기 [[exporting-a-transformers-model-to-onnx-with-optimumonnxruntime]]

CLI 대신에 `optimum.onnxruntime`을 사용하여 프로그래밍 방식으로 🤗 Transformers 모델을 ONNX로 내보낼 수도 있습니다. 다음과 같이 진행하세요:

```python
>>> from optimum.onnxruntime import ORTModelForSequenceClassification
>>> from transformers import AutoTokenizer

>>> model_checkpoint = "distilbert_base_uncased_squad"
>>> save_directory = "onnx/"

>>> # Load a model from transformers and export it to ONNX
>>> ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, export=True)
>>> tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

>>> # Save the onnx model and tokenizer
>>> ort_model.save_pretrained(save_directory)
>>> tokenizer.save_pretrained(save_directory)
```

### 지원되지 않는 아키텍처의 모델 내보내기 [[exporting-a-model-for-an-unsupported-architecture]]

현재 내보낼 수 없는 모델을 지원하기 위해 기여하려면, 먼저 [`optimum.exporters.onnx`](https://huggingface.co/docs/optimum/exporters/onnx/overview)에서 지원되는지 확인한 후 지원되지 않는 경우에는 [🤗 Optimum에 기여](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/contribute)하세요.

### `transformers.onnx`를 사용하여 모델 내보내기 [[exporting-a-model-with-transformersonnx]]

<Tip warning={true}>

`tranformers.onnx`는 더 이상 유지되지 않습니다. 위에서 설명한 대로 🤗 Optimum을 사용하여 모델을 내보내세요. 이 섹션은 향후 버전에서 제거될 예정입니다.

</Tip>

🤗 Transformers 모델을 ONNX로 내보내려면 추가 종속성을 설치하세요:

```bash
pip install transformers[onnx]
```

`transformers.onnx` 패키지를 Python 모듈로 사용하여 준비된 구성을 사용하여 체크포인트를 내보냅니다:

```bash
python -m transformers.onnx --model=distilbert-base-uncased onnx/
```

이렇게 하면 `--model` 인수에 정의된 체크포인트의 ONNX 그래프가 내보내집니다. 🤗 Hub에서 제공하는 체크포인트나 로컬에 저장된 체크포인트를 전달할 수 있습니다. 결과로 생성된 `model.onnx` 파일은 ONNX 표준을 지원하는 많은 가속기 중 하나에서 실행할 수 있습니다. 예를 들어, 다음과 같이 ONNX Runtime을 사용하여 모델을 로드하고 실행할 수 있습니다:

```python
>>> from transformers import AutoTokenizer
>>> from onnxruntime import InferenceSession

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
>>> session = InferenceSession("onnx/model.onnx")
>>> # ONNX Runtime expects NumPy arrays as input
>>> inputs = tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="np")
>>> outputs = session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))
```

필요한 출력 이름(예: `["last_hidden_state"]`)은 각 모델의 ONNX 구성을 확인하여 얻을 수 있습니다. 예를 들어, DistilBERT의 경우 다음과 같습니다:

```python
>>> from transformers.models.distilbert import DistilBertConfig, DistilBertOnnxConfig

>>> config = DistilBertConfig()
>>> onnx_config = DistilBertOnnxConfig(config)
>>> print(list(onnx_config.outputs.keys()))
["last_hidden_state"]
```

Hub의 TensorFlow 체크포인트에 대해서도 동일한 프로세스가 적용됩니다. 예를 들어, 다음과 같이 순수한 TensorFlow 체크포인트를 내보냅니다:

```bash
python -m transformers.onnx --model=keras-io/transformers-qa onnx/
```

로컬에 저장된 모델을 내보내려면 모델의 가중치 파일과 토크나이저 파일을 동일한 디렉토리에 저장한 다음, transformers.onnx 패키지의 --model 인수를 원하는 디렉토리로 지정하여 ONNX로 내보냅니다:

```bash
python -m transformers.onnx --model=local-pt-checkpoint onnx/
```