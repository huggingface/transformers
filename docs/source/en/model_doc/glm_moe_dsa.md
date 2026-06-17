<!--Copyright 2026 the HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be rendered properly in your Markdown viewer.

-->
*This model was published in HF papers on 2026-02-17 and contributed to Hugging Face Transformers on 2026-06-17.*

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="Tensor parallelism" src="https://img.shields.io/badge/Tensor%20parallelism-06b6d4?style=flat&logoColor=white">
    </div>
</div>

# GLM-5, GLM-5.1, GLM-5.2

### GLM-5.2

GLM-5.2, our latest flagship model for long-horizon tasks. It marks a substantial leap in long-horizon task capability over its predecessor GLM-5.1 and, for the first time, delivers that capability on a **solid 1M-token context**. 

GLM-5.2's new capabilities include:
- **Solid 1M Context:** A solid 1M-token context that stably sustains long-horizon work
- **Advanced Coding with Flexible Effort**: Stronger coding capabilities with multiple thinking effort levels to balance performance and latency
- **Improved Architecture**: We propose [IndexShare](https://arxiv.org/abs/2603.12201), which reuses the same indexer across every four sparse attention layers, reducing per-token FLOPs by 2.9× at a 1M context length. We also improve GLM-5.2’s MTP layer for speculative decoding, increasing the acceptance length by up to 20%

![bench_52](https://raw.githubusercontent.com/zai-org/GLM-5/refs/heads/main/resources/bench_52.png)

On standard coding benchmarks, GLM-5.2 is the strongest open-source model, improving on GLM-5.1 by a wide margin: 81.0 vs. 62.0 on Terminal-Bench 2.1 and 62.1 vs. 58.4 on SWE-bench Pro. It also closes much of the gap to the closed-source frontier — on Terminal-Bench 2.1 (81.0) it lands within a few points of Claude Opus 4.8 (85.0) — while staying ahead of Gemini 3.1 Pro.

For more detail, check our [blog](https://z.ai/blog/glm-5.2).

### GLM-5.1

GLM-5.1 is our next-generation flagship model for agentic engineering, with significantly stronger coding capabilities than its predecessor. It achieves state-of-the-art performance on SWE-Bench Pro and leads GLM-5 by a wide margin on NL2Repo (repo generation) and Terminal-Bench 2.0 (real-world terminal tasks).

![bench_51](https://raw.githubusercontent.com/zai-org/GLM-5/refs/heads/main/resources/bench_51.png)

But the most meaningful leap goes beyond first-pass performance. Previous models—including GLM-5—tend to exhaust their repertoire early: they apply familiar techniques for quick initial gains, then plateau. Giving them more time doesn't help.

GLM-5.1, by contrast, is built to stay effective on agentic tasks over much longer horizons. We've found that the model handles ambiguous problems with better judgment and stays productive over longer sessions. It breaks complex problems down, runs experiments, reads results, and identifies blockers with real precision. By revisiting its reasoning and revising its strategy through repeated iteration, GLM-5.1 sustains optimization over hundreds of rounds and thousands of tool calls. The longer it runs, the better the result.


### GLM-5

GLM-5 ([GlmMoeDsa](https://huggingface.co/papers/2602.15763)) is a 744B-parameter mixture-of-experts model with 40B active parameters per token, using DeepSeek Sparse Attention (DSA) for efficient 200K-token context handling. It was trained entirely on Huawei Ascend chips and matches frontier-level performance on reasoning and long-context benchmarks.


With advances in both pre-training and post-training, GLM-5 delivers significant improvement compared to GLM-4.7 across a wide range of academic benchmarks and achieves best-in-class performance among all open-source models in the world on reasoning, coding, and agentic tasks, closing the gap with frontier models.

![bench](https://raw.githubusercontent.com/zai-org/GLM-5/refs/heads/main/resources/bench.png)


## GlmMoeDsaConfig

[[autodoc]] GlmMoeDsaConfig

## GlmMoeDsaModel

[[autodoc]] GlmMoeDsaModel
    - forward

## GlmMoeDsaForCausalLM

[[autodoc]] GlmMoeDsaForCausalLM
    - forward
