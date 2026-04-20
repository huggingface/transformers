<!--Copyright 2026 The LG AI Research and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2026-04-09 and added to Hugging Face Transformers on 2026-04-20.*

# EXAONE 4.5

## 개요

[EXAONE 4.5](https://github.com/LG-AI-EXAONE/EXAONE-4.5) 모델은 LG AI연구원에서 공개한 최초의 오픈 웨이트(open-weight) 비전-자연어 모델(vision-language model)입니다. 
전용 비전 인코더를 기존 개발된 EXAONE 4.0 프레임워크에 통합하여 모델의 능력을 비전과 자연어를 고려한 멀티모달리티로 확장했습니다. EXAONE 4.5는 1.2B 크기의 비전 인코더를 포함해 총 33B 크기의 모델로 구성됩니다. 
EXAONE 4.5는 이전 EXAONE 모델군으로부터 이어져 온 강력한 언어 처리 능력 덕분에 범용 벤치마크에서 경쟁력 있는 성능을 달성함과 동시에, 동등 규모의 최신 SOTA 모델을 능가하는 문서 이해 능력과 한국 문화적 추론 능력을 갖추고 있습니다.

더 자세한 정보는 [기술 보고서](http://arxiv.org/abs/2604.08644)나 [블로그](https://www.lgresearch.ai/blog/view?seq=641), 그리고 [공식 GitHub](https://github.com/LG-AI-EXAONE/EXAONE-4.5) 페이지를 참고해주시길 바랍니다.

양자화된 버전을 포함한 공개된 모든 체크포인트는 [Huggingface 콜렉션](https://huggingface.co/collections/LGAI-EXAONE/exaone-45)에서 확인할 수 있습니다.

## 모델 세부 정보

### Model Configuration of EXAONE 4.5

- Model Type: Causal Language Model + Vision Encoder
- Number of Parameters (Language Model): 31.7B
- Number of Parameters (Vision Encoder): 1.29B
- Hidden Dimension: 5,120
- Intermediate size: 27,392
- Number of Layers: 64 Main layers + 1 MTP layers
  - Hybrid Attention Pattern: 16 x (3 Sliding window attention + 1 Global attention)
  - Reordered Norm: Apply normalization after Attention/MLP, and before residual connection
- Sliding Window Attention
  - Number of Attention Heads: 40 Q-heads and 8 KV-heads
  - Head Dimension: 128 for both Q/KV
  - Sliding Window Size: 4,096
- Global Attention
  - Number of Attention Heads: 40 Q-heads and 8 KV-heads
  - Head Dimension: 128 for both Q/KV
  - No Rotary Positional Embedding Used (NoPE)
- Vision Encoder
  - Grouped Query Attention (GQA) with 32 Q-heads and 8 KV-heads
  - 2D RoPE for vision embeddings
- Vocab Size: 153,600
- Context Length: 262,144 tokens
- Knowledge Cutoff: Dec 2024 (2024/12)


## Exaone4_5_Config

[[autodoc]] Exaone4_5_Config

## Exaone4_5_VisionConfig

[[autodoc]] Exaone4_5_VisionConfig

## Exaone4_5_Processor

[[autodoc]] Exaone4_5_Processor

## Exaone4_5_Model

[[autodoc]] Exaone4_5_Model
    - forward

## Exaone4_5_ForConditionalGeneration

[[autodoc]] Exaone4_5_ForConditionalGeneration
    - forward