<!--Copyright 2025 The NVIDIA NeMo Team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on {release_date} and added to Hugging Face Transformers on 2025-09-25.*

<div class="flex flex-wrap space-x-1">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
<img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
</div>

# Parakeet

## Overview

Parakeet models, [introduced by NVIDIA NeMo](https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/), are models that combine a [Fast Conformer](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#fast-conformer) encoder with connectionist temporal classification (CTC), recurrent neural network transducer (RNNT) or token and duration transducer (TDT) decoder for automatic speech recognition.

**Model Architecture**

- **Fast Conformer Encoder**: A linearly scalable Conformer architecture that processes mel-spectrogram features and reduces sequence length through subsampling. This is more efficient version of the Conformer Encoder found in [FastSpeech2Conformer](./fastspeech2_conformer.md) (see [`ParakeetEncoder`] for the encoder implementation and details).
- [**ParakeetForCTC**](#parakeetforctc): a Fast Conformer Encoder + a CTC decoder
  - **CTC Decoder**: Simple but effective decoder consisting of:
    - 1D convolution projection from encoder hidden size to vocabulary size (for optimal NeMo compatibility).
    - CTC loss computation for training.
    - Greedy CTC decoding for inference.

The original implementation can be found in [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).
Model checkpoints are to be found under [the NVIDIA organization](https://huggingface.co/nvidia/models?search=parakeet).

This model was contributed by [Nithin Rao Koluguri](https://huggingface.co/nithinraok), [Eustache Le Bihan](https://huggingface.co/eustlb) and [Eric Bezzam](https://huggingface.co/bezzam).

## Usage

### Basic usage

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="nvidia/parakeet-ctc-1.1b")
out = pipe("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")
print(out)
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoModelForCTC, AutoProcessor
from datasets import load_dataset, Audio
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-1.1b")
model = AutoModelForCTC.from_pretrained("nvidia/parakeet-ctc-1.1b", dtype="auto", device_map=device)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el['array'] for el in ds["audio"][:5]]

inputs = processor(speech_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs)
print(processor.batch_decode(outputs))
```

</hfoption>
</hfoptions>

### Making The Model Go Brrr

Parakeet supports full-graph compilation with CUDA graphs! This optimization is most effective when you know the maximum audio length you want to transcribe. The key idea is using static input shapes to avoid recompilation. For example, if you know your audio will be under 30 seconds, you can use the processor to pad all inputs to 30 seconds, preparing consistent input features and attention masks. See the example below!

```python
from transformers import AutoModelForCTC, AutoProcessor
from datasets import load_dataset, Audio
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-1.1b")
model = AutoModelForCTC.from_pretrained("nvidia/parakeet-ctc-1.1b", dtype="auto", device_map=device)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el['array'] for el in ds["audio"][:5]]

# Compile the generate method with fullgraph and CUDA graphs
model.generate = torch.compile(model.generate, fullgraph=True, mode="reduce-overhead")

# let's define processor kwargs to pad to 30 seconds
processor_kwargs = {
    "padding": "max_length",
    "max_length": 30 * processor.feature_extractor.sampling_rate,
}

# Define a timing context using CUDA events
class TimerContext:
    def __init__(self, name="Execution"):
        self.name = name
        self.start_event = None
        self.end_event = None
        
    def __enter__(self):
        # Use CUDA events for more accurate GPU timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()
        elapsed_time = self.start_event.elapsed_time(self.end_event) / 1000.0
        print(f"{self.name} time: {elapsed_time:.4f} seconds")


inputs = processor(speech_samples[0], **processor_kwargs)
inputs.to(device, dtype=model.dtype)
print("\n" + "="*50)
print("First generation - compiling...")
# Generate with the compiled model
with TimerContext("First generation"):
    outputs = model.generate(**inputs)
print(processor.batch_decode(outputs))

inputs = processor(speech_samples[1], **processor_kwargs)
inputs.to(device, dtype=model.dtype)
print("\n" + "="*50)
print("Second generation - recording CUDA graphs...")
with TimerContext("Second generation"):
    outputs = model.generate(**inputs)
print(processor.batch_decode(outputs))

inputs = processor(speech_samples[2], **processor_kwargs)
inputs.to(device, dtype=model.dtype)
print("\n" + "="*50)
print("Third generation - fast !!!")
with TimerContext("Third generation"):
    outputs = model.generate(**inputs)
print(processor.batch_decode(outputs))

inputs = processor(speech_samples[3], **processor_kwargs)
inputs.to(device, dtype=model.dtype)
print("\n" + "="*50)
print("Fourth generation - still fast !!!")
with TimerContext("Fourth generation"):
    outputs = model.generate(**inputs)
print(processor.batch_decode(outputs))
```

### Training

```python
from transformers import AutoModelForCTC, AutoProcessor
from datasets import load_dataset, Audio
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("nvidia/parakeet-ctc-1.1b")
model = AutoModelForCTC.from_pretrained("nvidia/parakeet-ctc-1.1b", dtype="auto", device_map=device)

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el['array'] for el in ds["audio"][:5]]
text_samples = [el for el in ds["text"][:5]]

# passing `text` to the processor will prepare inputs' `labels` key
inputs = processor(audio=speech_samples, text=text_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(device, dtype=model.dtype)

outputs = model(**inputs)
outputs.loss.backward()
```

## ParakeetTokenizer

[[autodoc]] ParakeetTokenizer

## ParakeetFeatureExtractor

[[autodoc]] ParakeetFeatureExtractor
    - __call__

## ParakeetProcessor

[[autodoc]] ParakeetProcessor
    - __call__
    - batch_decode
    - decode

## ParakeetEncoderConfig

[[autodoc]] ParakeetEncoderConfig

## ParakeetCTCConfig

[[autodoc]] ParakeetCTCConfig

## ParakeetEncoder

[[autodoc]] ParakeetEncoder

## ParakeetForCTC

[[autodoc]] ParakeetForCTC
