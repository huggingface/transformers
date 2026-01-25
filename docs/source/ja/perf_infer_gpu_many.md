<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Efficient Inference on a Multiple GPUs

この文書には、複数のGPUで効率的に推論を行う方法に関する情報が含まれています。
<Tip>

注意: 複数のGPUセットアップは、[単一のGPUセクション](./perf_infer_gpu_one)で説明されているほとんどの戦略を使用できます。ただし、より良い使用法のために使用できる簡単なテクニックについても認識しておく必要があります。

</Tip>

## Flash Attention 2

Flash Attention 2の統合は、複数のGPUセットアップでも機能します。詳細については、[単一のGPUセクション](./perf_infer_gpu_one#Flash-Attention-2)の適切なセクションをご覧ください。
