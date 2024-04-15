<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# 다중 GPU에서 효율적인 추론 [[efficient-inference-on-a-multiple-gpus]]

이 문서에는 다중 GPU에서 효율적으로 추론하는 방법에 대한 정보가 포함되어 있습니다.
<Tip>

참고: 다중 GPU 설정은 [단일 GPU 섹션](./perf_infer_gpu_one)에서 설명된 대부분의 전략을 사용할 수 있습니다. 그러나 더 나은 활용을 위해 간단한 기법들을 알아야 합니다.

</Tip>

## 더 빠른 추론을 위한 `BetterTransformer` [[bettertransformer-for-faster-inference]]

우리는 최근 텍스트, 이미지 및 오디오 모델에 대한 다중 GPU에서 더 빠른 추론을 위해 `BetterTransformer`를 통합했습니다. 자세한 내용은 이 통합에 대한 [문서](https://huggingface.co/docs/optimum/bettertransformer/overview)를 확인하십시오.