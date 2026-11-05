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

# 머신러닝 앱 [[machine-learning-apps]]

머신러닝 앱을 빠르고 쉽게 구축하고 공유할 수 있는 라이브러리인 [Gradio](https://www.gradio.app/)는 [`Pipeline`]과 통합되어 추론을 위한 간단한 인터페이스를 빠르게 생성할 수 있습니다.

시작하기 전에 Gradio가 설치되어 있는지 확인하세요.

```py
!pip install gradio
```

원하는 작업에 맞는 pipeline을 생성한 다음, Gradio의 [Interface.from_pipeline](https://www.gradio.app/docs/gradio/interface#interface-from_pipeline) 함수에 전달하여 인터페이스를 만드세요. Gradio는 [`Pipeline`]에 맞는 입력 및 출력 컴포넌트를 자동으로 결정합니다.

[launch](https://www.gradio.app/main/docs/gradio/blocks#blocks-launch)를 추가하여 웹 서버를 생성하고 앱을 시작하세요.

```py
from transformers import pipeline
import gradio as gr

pipeline = pipeline("image-classification", model="google/vit-base-patch16-224")
gr.Interface.from_pipeline(pipeline).launch()
```

웹 앱은 기본적으로 로컬 서버에서 실행됩니다. 다른 사용자와 앱을 공유하려면 [launch](https://www.gradio.app/main/docs/gradio/blocks#blocks-launch)에서 `share=True`로 설정하여 임시 공개 링크를 생성하세요. 더 지속적인 솔루션을 원한다면 Hugging Face [Spaces](https://hf.co/spaces)에서 앱을 호스팅하세요.

```py
gr.Interface.from_pipeline(pipeline).launch(share=True)
```

아래 Space는 위 코드를 사용하여 생성되었으며, Spaces에서 호스팅됩니다.

<iframe
	src="https://stevhliu-gradio-pipeline-demo.hf.space"
	frameborder="0"
	width="850"
	height="850"
></iframe>
