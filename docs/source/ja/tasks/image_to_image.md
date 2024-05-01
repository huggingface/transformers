<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Image-to-Image Task Guide

[[open-in-colab]]

Image-to-Image タスクは、アプリケーションが画像を受信し、別の画像を出力するタスクです。これには、画像強化 (超解像度、低光量強化、ディレインなど)、画像修復などを含むさまざまなサブタスクがあります。

このガイドでは、次の方法を説明します。
- 超解像度タスクに画像間のパイプラインを使用します。
- パイプラインを使用せずに、同じタスクに対してイメージ間モデルを実行します。

このガイドがリリースされた時点では、`image-to-image`パイプラインは超解像度タスクのみをサポートしていることに注意してください。

必要なライブラリをインストールすることから始めましょう。

```bash
pip install transformers
```

[Swin2SR モデル](https://huggingface.co/caidas/swin2SR-lightweight-x2-64) を使用してパイプラインを初期化できるようになりました。次に、イメージを使用してパイプラインを呼び出すことで、パイプラインを推論できます。現時点では、[Swin2SR モデル](https://huggingface.co/models?sort=trending&search=swin2sr) のみがこのパイプラインでサポートされています。

```python
from transformers import pipeline

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipe = pipeline(task="image-to-image", model="caidas/swin2SR-lightweight-x2-64", device=device)
```

では、画像を読み込みましょう。

```python
from PIL import Image
import requests

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg"
image = Image.open(requests.get(url, stream=True).raw)

print(image.size)
```
```bash
# (532, 432)
```
<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat.jpg" alt="Photo of a cat"/>
</div>


これで、パイプラインを使用して推論を実行できるようになりました。猫の画像の拡大バージョンを取得します。

```python
upscaled = pipe(image)
print(upscaled.size)
```
```bash
# (1072, 880)
```

パイプラインを使用せずに自分で推論を実行したい場合は、トランスフォーマーの `Swin2SRForImageSuperResolution` クラスと `Swin2SRImageProcessor` クラスを使用できます。これには同じモデルのチェックポイントを使用します。モデルとプロセッサを初期化しましょう。

```python
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor 

model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-lightweight-x2-64").to(device)
processor = Swin2SRImageProcessor("caidas/swin2SR-lightweight-x2-64")
```

`pipeline`」は、自分で行う必要がある前処理と後処理のステップを抽象化するので、画像を前処理しましょう。画像をプロセッサに渡してから、ピクセル値を GPU に移動します。

```python
pixel_values = processor(image, return_tensors="pt").pixel_values
print(pixel_values.shape)

pixel_values = pixel_values.to(device)
```

これで、ピクセル値をモデルに渡すことで画像を推測できるようになりました。

```python
import torch

with torch.no_grad():
  outputs = model(pixel_values)
```

出力は、以下のような `ImageSuperResolutionOutput` タイプのオブジェクトです 👇

```
(loss=None, reconstruction=tensor([[[[0.8270, 0.8269, 0.8275,  ..., 0.7463, 0.7446, 0.7453],
          [0.8287, 0.8278, 0.8283,  ..., 0.7451, 0.7448, 0.7457],
          [0.8280, 0.8273, 0.8269,  ..., 0.7447, 0.7446, 0.7452],
          ...,
          [0.5923, 0.5933, 0.5924,  ..., 0.0697, 0.0695, 0.0706],
          [0.5926, 0.5932, 0.5926,  ..., 0.0673, 0.0687, 0.0705],
          [0.5927, 0.5914, 0.5922,  ..., 0.0664, 0.0694, 0.0718]]]],
       device='cuda:0'), hidden_states=None, attentions=None)
```

`reconstruction`を取得し、それを視覚化するために後処理する必要があります。どのように見えるか見てみましょう。

```python
outputs.reconstruction.data.shape
# torch.Size([1, 3, 880, 1072])
```

出力を圧縮して軸 0 を削除し、値をクリップしてから、それを numpy float に変換する必要があります。次に、軸を [1072, 880] の形状になるように配置し、最後に出力を範囲 [0, 255] に戻します。

```python
import numpy as np

# squeeze, take to CPU and clip the values
output = outputs.reconstruction.data.squeeze().cpu().clamp_(0, 1).numpy()
# rearrange the axes
output = np.moveaxis(output, source=0, destination=-1)
# bring values back to pixel values range
output = (output * 255.0).round().astype(np.uint8)
Image.fromarray(output)
```
<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat_upscaled.png" alt="Upscaled photo of a cat"/>
</div>