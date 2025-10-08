<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
*This model was released on 2023-05-11 and added to Hugging Face Transformers on 2024-06-25 and contributed by [RaushanTurganbay](https://huggingface.co/RaushanTurganbay).*

# InstructBlipVideo

[InstructBLIPVideo](https://huggingface.co/papers/2305.06500) extends InstructBLIP to handle video inputs while maintaining the same architecture and checkpoints. It leverages instruction tuning on a variety of datasets, introducing instruction-aware visual feature extraction to enhance performance. This results in state-of-the-art zero-shot performance across multiple datasets and superior accuracy on fine-tuned tasks compared to BLIP-2 and Flamingo.

<hfoptions id="usage">
<hfoption id="InstructBlipVideoForConditionalGeneration">

```py
import torch
import av
import numpy as np
from transformers import InstructBlipVideoProcessor, InstructBlipVideoForConditionalGeneration
from huggingface_hub import hf_hub_download

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

model = InstructBlipVideoForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", device_map="auto")
processor = InstructBlipVideoProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

file_path = hf_hub_download(
      repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
)
container = av.open(file_path)

total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 4).astype(int)
clip = read_video_pyav(container, indices)

prompt = "What is happening in the video?"
inputs = processor(text=prompt, images=clip, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    do_sample=False,
    num_beams=5,
    max_length=256,
    repetition_penalty=1.5,
    length_penalty=1.0,
)
generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)
```

</hfoption>
</hfoptions>

## InstructBlipVideoConfig

[[autodoc]] InstructBlipVideoConfig

## InstructBlipVideoVisionConfig

[[autodoc]] InstructBlipVideoVisionConfig

## InstructBlipVideoQFormerConfig

[[autodoc]] InstructBlipVideoQFormerConfig

## InstructBlipVideoProcessor

[[autodoc]] InstructBlipVideoProcessor

## InstructBlipVideoImageProcessor

[[autodoc]] InstructBlipVideoImageProcessor
    - preprocess

## InstructBlipVideoVisionModel

[[autodoc]] InstructBlipVideoVisionModel
    - forward

## InstructBlipVideoQFormerModel

[[autodoc]] InstructBlipVideoQFormerModel
    - forward

## InstructBlipVideoForConditionalGeneration

[[autodoc]] InstructBlipVideoForConditionalGeneration
    - forward
    - generate

## InstructBlipVideoModel

[[autodoc]] InstructBlipVideoModel
    - forward

## InstructBlipVideoVideoProcessor

[[autodoc]] InstructBlipVideoVideoProcessor

