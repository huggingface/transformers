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

# 모델 공유하기[[share-a-model]]

지난 두 튜토리얼에서 분산 설정을 위해 PyTorch, Keras 및 🤗 Accelerate를 사용하여 모델을 미세 조정하는 방법을 보았습니다. 다음 단계는 모델을 커뮤니티와 공유하는 것입니다! Hugging Face는 인공지능의 민주화를 위해 모두에게 지식과 자원을 공개적으로 공유해야 한다고 믿습니다. 다른 사람들이 시간과 자원을 절약할 수 있도록 커뮤니티에 모델을 공유하는 것을 고려해 보세요.

이 튜토리얼에서 [Model Hub](https://huggingface.co/models)에서 훈련되거나 미세 조정 모델을 공유하는 두 가지 방법에 대해 알아봅시다:

- API를 통해 파일을 Hub에 푸시합니다.
- 웹사이트를 통해 파일을 Hub로 끌어다 놓습니다.

<iframe width="560" height="315" src="https://www.youtube.com/embed/XvSGPZFEjDY" title="YouTube video player"
frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope;
picture-in-picture" allowfullscreen></iframe>

<Tip>

커뮤니티에 모델을 공유하려면, [huggingface.co](https://huggingface.co/join)에 계정이 필요합니다. 기존 조직에 가입하거나 새로 만들 수도 있습니다.

</Tip>

## 저장소 특징[[repository-features]]

모델 허브의 각 저장소는 일반적인 GitHub 저장소처럼 작동합니다. 저장소는 버전 관리, 커밋 기록, 차이점 시각화 기능을 제공합니다.

모델 허브에 내장된 버전 관리는 git 및 [git-lfs](https://git-lfs.github.com/)를 기반으로 합니다. 즉, 하나의 모델을 하나의 저장소로 취급하여 접근 제어 및 확장성이 향상됩니다. 버전 제어는 커밋 해시, 태그 또는 브랜치로 모델의 특정 버전을 고정하는 방법인 *revision*을 허용합니다.

따라서 `revision` 매개변수를 사용하여 특정 모델 버전을 가져올 수 있습니다:

```py
>>> model = AutoModel.from_pretrained(
...     "julien-c/EsperBERTo-small", revision="v2.0.1"  # tag name, or branch name, or commit hash
... )
```

또한 저장소에서 파일을 쉽게 편집할 수 있으며, 커밋 기록과 차이를 볼 수 있습니다:

![vis_diff](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png)

## 설정[[setup]]

모델을 허브에 공유하기 전에 Hugging Face 자격 증명이 필요합니다. 터미널에 액세스할 수 있는 경우, 🤗 Transformers가 설치된 가상 환경에서 다음 명령을 실행합니다. 그러면 Hugging Face 캐시 폴더(기본적으로 `~/.cache/`)에 액세스 토큰을 저장합니다:

```bash
huggingface-cli login
```

Jupyter 또는 Colaboratory와 같은 노트북을 사용 중인 경우, [`huggingface_hub`](https://huggingface.co/docs/hub/adding-a-library) 라이브러리가 설치되었는지 확인하세요. 이 라이브러리를 사용하면 API로 허브와 상호 작용할 수 있습니다.

```bash
pip install huggingface_hub
```

그런 다음 `notebook_login`로 허브에 로그인하고, [여기](https://huggingface.co/settings/token) 링크에서 로그인할 토큰을 생성합니다:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## 프레임워크 간 모델 변환하기[[convert-a-model-for-all-frameworks]]

다른 프레임워크로 작업하는 사용자가 모델을 사용할 수 있도록 하려면, PyTorch 및 TensorFlow 체크포인트를 모두 사용하여 모델을 변환하고 업로드하는 것이 좋습니다. 이 단계를 건너뛰어도 사용자는 다른 프레임워크에서 모델을 가져올 수 있지만, 🤗 Transformers가 체크포인트를 즉석에서 변환해야 하므로 속도가 느려질 수 있습니다.

체크포인트를 다른 프레임워크로 변환하는 것은 쉽습니다. PyTorch 및 TensorFlow가 설치되어 있는지 확인한 다음(설치 지침은 [여기](installation) 참조) 다른 프레임워크에서 작업에 대한 특정 모델을 찾습니다.

<frameworkcontent>
<pt>
체크포인트를 TensorFlow에서 PyTorch로 변환하려면 `from_tf=True`를 지정하세요:

```py
>>> pt_model = DistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_tf=True)
>>> pt_model.save_pretrained("path/to/awesome-name-you-picked")
```
</pt>
<tf>
체크포인트를 PyTorch에서 TensorFlow로 변환하려면 `from_pt=True`를 지정하세요:

```py
>>> tf_model = TFDistilBertForSequenceClassification.from_pretrained("path/to/awesome-name-you-picked", from_pt=True)
```

그런 다음 새로운 체크포인트와 함께 새로운 TensorFlow 모델을 저장할 수 있습니다:

```py
>>> tf_model.save_pretrained("path/to/awesome-name-you-picked")
```
</tf>
<jax>
Flax에서 모델을 사용하는 경우, PyTorch에서 Flax로 체크포인트를 변환할 수도 있습니다:

```py
>>> flax_model = FlaxDistilBertForSequenceClassification.from_pretrained(
...     "path/to/awesome-name-you-picked", from_pt=True
... )
```
</jax>
</frameworkcontent>

## 훈련 중 모델 푸시하기[[push-a-model-during-training]]

<frameworkcontent>
<pt>
<Youtube id="Z1-XMy-GNLQ"/>

모델을 허브에 공유하는 것은 추가 매개변수나 콜백을 추가하는 것만큼 간단합니다. [미세 조정 튜토리얼](training)에서 [`TrainingArguments`] 클래스는 하이퍼파라미터와 추가 훈련 옵션을 지정하는 곳이라는 것을 기억하세요. 이러한 훈련 옵션 중 하나는 모델을 허브로 직접 푸시하는 기능을 포함합니다. [`TrainingArguments`]에서 `push_to_hub=True`를 설정하세요:

```py
>>> training_args = TrainingArguments(output_dir="my-awesome-model", push_to_hub=True)
```

평소와 같이 훈련 인수를 [`Trainer`]에 전달합니다:

```py
>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=small_train_dataset,
...     eval_dataset=small_eval_dataset,
...     compute_metrics=compute_metrics,
... )
```

모델을 미세 조정한 후, [`Trainer`]에서 [`~transformers.Trainer.push_to_hub`]를 호출하여 훈련된 모델을 허브로 푸시하세요. 🤗 Transformers는 훈련 하이퍼파라미터, 훈련 결과 및 프레임워크 버전을 모델 카드에 자동으로 추가합니다!

```py
>>> trainer.push_to_hub()
```
</pt>
<tf>
[`PushToHubCallback`]을 사용하여 모델을 허브에 공유하려면, [`PushToHubCallback`]에 다음 인수를 정의하세요:

- 출력된 모델의 파일 경로
- 토크나이저
- `{Hub 사용자 이름}/{모델 이름}` 형식의 `hub_model_id`

```py
>>> from transformers import PushToHubCallback

>>> push_to_hub_callback = PushToHubCallback(
...     output_dir="./your_model_save_path", tokenizer=tokenizer, hub_model_id="your-username/my-awesome-model"
... )
```

[`fit`](https://keras.io/api/models/model_training_apis/)에 콜백을 추가하면, 🤗 Transformers가 훈련된 모델을 허브로 푸시합니다:

```py
>>> model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=3, callbacks=push_to_hub_callback)
```
</tf>
</frameworkcontent>

## `push_to_hub` 함수 사용하기[[use-the-pushtohub-function]]

모델에서 직접 `push_to_hub`를 호출하여 허브에 업로드할 수도 있습니다.

`push_to_hub`에 모델 이름을 지정하세요:

```py
>>> pt_model.push_to_hub("my-awesome-model")
```

이렇게 하면 사용자 이름 아래에 모델 이름 `my-awesome-model`로 저장소가 생성됩니다. 이제 사용자는 `from_pretrained` 함수를 사용하여 모델을 가져올 수 있습니다:

```py
>>> from transformers import AutoModel

>>> model = AutoModel.from_pretrained("your_username/my-awesome-model")
```

조직에 속하고 모델을 조직 이름으로 대신 푸시하려면 `repo_id`에 추가하세요:

```py
>>> pt_model.push_to_hub("my-awesome-org/my-awesome-model")
```

`push_to_hub` 함수는 모델 저장소에 다른 파일을 추가하는 데에도 사용할 수 있습니다. 예를 들어 모델 저장소에 토크나이저를 추가할 수 있습니다:

```py
>>> tokenizer.push_to_hub("my-awesome-model")
```

또는 미세 조정된 PyTorch 모델의 TensorFlow 버전을 추가할 수도 있습니다:

```py
>>> tf_model.push_to_hub("my-awesome-model")
```

이제 Hugging Face 프로필로 이동하면, 새로 생성한 모델 저장소가 표시됩니다. **Files** 탭을 클릭하면 저장소에 업로드한 모든 파일이 표시됩니다.

저장소에 파일을 만들고 업로드하는 방법에 대한 자세한 내용은 허브 설명서 [여기](https://huggingface.co/docs/hub/how-to-upstream)를 참조하세요.

## 웹 인터페이스로 업로드하기[[upload-with-the-web-interface]]

코드 없는 접근 방식을 선호하는 사용자는 허브의 웹 인터페이스를 통해 모델을 업로드할 수 있습니다. [huggingface.co/new](https://huggingface.co/new)를 방문하여 새로운 저장소를 생성하세요:

![new_model_repo](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png)

여기서 모델에 대한 몇 가지 정보를 추가하세요:

- 저장소의 **소유자**를 선택합니다. 이는 사용자 또는 사용자가 속한 조직일 수 있습니다.
- 저장소 이름이 될 모델의 이름을 선택합니다.
- 모델이 공개인지 비공개인지 선택합니다.
- 모델의 라이센스 사용을 지정합니다.

이제 **Files** 탭을 클릭하고 **Add file** 버튼을 클릭하여 새로운 파일을 저장소에 업로드합니다. 그런 다음 업로드할 파일을 끌어다 놓고 커밋 메시지를 추가하세요.

![upload_file](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png)

## 모델 카드 추가하기[[add-a-model-card]]

사용자가 모델의 기능, 제한, 잠재적 편향 및 윤리적 고려 사항을 이해할 수 있도록 저장소에 모델 카드를 추가하세요. 모델 카드는 `README.md` 파일에 정의되어 있습니다. 다음 방법으로 모델 카드를 추가할 수 있습니다:

* `README.md` 파일을 수동으로 생성하여 업로드합니다.
* 모델 저장소에서 **Edit model card** 버튼을 클릭합니다.

모델 카드에 포함할 정보 유형에 대한 좋은 예는 DistilBert [모델 카드](https://huggingface.co/distilbert-base-uncased)를 참조하세요. 모델의 탄소 발자국이나 위젯 예시 등 `README.md` 파일에서 제어할 수 있는 다른 옵션에 대한 자세한 내용은 [여기](https://huggingface.co/docs/hub/models-cards) 문서를 참조하세요.
