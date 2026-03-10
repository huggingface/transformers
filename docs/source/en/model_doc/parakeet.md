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
- [**ParakeetForTDT**](#parakeetfortdt): a Fast Conformer Encoder + a TDT (Token Duration Transducer) decoder
  - **TDT Decoder**: Jointly predicts tokens and their durations, enabling efficient decoding:
    - LSTM prediction network maintains language context across token predictions.
    - Joint network combines encoder and decoder outputs.
    - Duration head predicts how many frames to skip, enabling fast inference.

The original implementation can be found in [NVIDIA NeMo](https://github.com/NVIDIA/NeMo).
Model checkpoints are to be found under [the NVIDIA organization](https://huggingface.co/nvidia/models?search=parakeet).

This model was contributed by [Nithin Rao Koluguri](https://huggingface.co/nithinraok), [Eustache Le Bihan](https://huggingface.co/eustlb), [Eric Bezzam](https://huggingface.co/bezzam), [Maksym Lypivskyi](https://huggingface.co/MaksL), and [Hainan Xu](https://huggingface.co/hainanx).

## Usage

### `ParakeetForCTC` usage

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

model_id = "nvidia/parakeet-ctc-1.1b"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id, dtype="auto", device_map="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el['array'] for el in ds["audio"][:5]]

inputs = processor(speech_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs)
print(processor.decode(outputs))
```

</hfoption>
</hfoptions>

### `ParakeetForTDT` usage

<hfoptions id="tdt-usage">
<hfoption id="Pipeline">

Parakeet TDT transcripts include casing, and the model can also performk token timestamping.

```py
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-0.6b-v3")
out = pipe("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")
print(out)
```

</hfoption>
<hfoption id="AutoModel">

```py
from transformers import AutoModelForTDT, AutoProcessor
from datasets import load_dataset, Audio

model_id = "nvidia/parakeet-tdt-0.6b-v3"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTDT.from_pretrained(model_id, dtype="auto", device_map="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el['array'] for el in ds["audio"][:5]]

inputs = processor(speech_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)
output = model.generate(**inputs, return_dict_in_generate=True)
print(processor.decode(output.sequences, skip_special_tokens=True))
```

</hfoption>
<hfoption id="Timestamping">

```py
from datasets import Audio, load_dataset
from transformers import AutoModelForTDT, AutoProcessor

model_id = "nvidia/parakeet-tdt-0.6b-v3"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTDT.from_pretrained(model_id, dtype="auto", device_map="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el['array'] for el in ds["audio"][:1]]

inputs = processor(speech_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)
output = model.generate(**inputs, return_dict_in_generate=True, return_timestamps=True)
decoded_output, decoded_timestamps = processor.decode(
    output.sequences,
    token_timestamps=output.token_timestamps,
    token_durations=output.token_durations,
    skip_special_tokens=True
)
print("Transcription:", decoded_output)
print("\nTimestamped tokens:", decoded_timestamps)

"""
Transcription: ['mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.']

Timestamped tokens: [[{'token': 'm', 'start': 0.24, 'end': 0.48}, {'token': 'ister', 'start': 0.48, 'end': 0.64}, {'token': 'Qu', 'start': 0.64, 'end': 0.88}, {'token': 'il', 'start': 0.88, 'end': 1.12}, {'token': 'ter', 'start': 1.12, 'end': 1.36}, {'token': 'is', 'start': 1.36, 'end': 1.44}, {'token': 'the', 'start': 1.44, 'end': 1.6}, {'token': 'ap', 'start': 1.6, 'end': 1.76}, {'token': 'ost', 'start': 1.76, 'end': 1.92}, {'token': 'le', 'start': 2.0, 'end': 2.16}, {'token': 'of', 'start': 2.16, 'end': 2.24}, {'token': 'the', 'start': 2.24, 'end': 2.4}, {'token': 'mid', 'start': 2.4, 'end': 2.48}, {'token': 'd', 'start': 2.48, 'end': 2.56}, {'token': 'le', 'start': 2.56, 'end': 2.64}, {'token': 'clas', 'start': 2.72, 'end': 2.88}, {'token': 's', 'start': 2.88, 'end': 3.04}, {'token': 'es', 'start': 3.04, 'end': 3.12}, {'token': ',', 'start': 3.12, 'end': 3.12}, {'token': 'and', 'start': 3.2800000000000002, 'end': 3.44}, {'token': 'we', 'start': 3.44, 'end': 3.6}, {'token': 'are', 'start': 3.6, 'end': 3.7600000000000002}, {'token': 'gl', 'start': 3.7600000000000002, 'end': 3.92}, {'token': 'ad', 'start': 3.92, 'end': 4.08}, {'token': 'to', 'start': 4.08, 'end': 4.24}, {'token': 'wel', 'start': 4.24, 'end': 4.4}, {'token': 'c', 'start': 4.4, 'end': 4.48}, {'token': 'ome', 'start': 4.48, 'end': 4.72}, {'token': 'his', 'start': 4.72, 'end': 4.96}, {'token': 'gos', 'start': 4.96, 'end': 5.12}, {'token': 'pel', 'start': 5.36, 'end': 5.6000000000000005}, {'token': '.', 'start': 5.6000000000000005, 'end': 5.6000000000000005}]]
"""
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
print(processor.decode(outputs))

inputs = processor(speech_samples[1], **processor_kwargs)
inputs.to(device, dtype=model.dtype)
print("\n" + "="*50)
print("Second generation - recording CUDA graphs...")
with TimerContext("Second generation"):
    outputs = model.generate(**inputs)
print(processor.decode(outputs))

inputs = processor(speech_samples[2], **processor_kwargs)
inputs.to(device, dtype=model.dtype)
print("\n" + "="*50)
print("Third generation - fast !!!")
with TimerContext("Third generation"):
    outputs = model.generate(**inputs)
print(processor.decode(outputs))

inputs = processor(speech_samples[3], **processor_kwargs)
inputs.to(device, dtype=model.dtype)
print("\n" + "="*50)
print("Fourth generation - still fast !!!")
with TimerContext("Fourth generation"):
    outputs = model.generate(**inputs)
print(processor.decode(outputs))
```

### CTC Training

```python
import torch
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, AutoProcessor

model_id = "nvidia/parakeet-ctc-1.1b"
NUM_SAMPLES = 5

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
model.train()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el['array'] for el in ds["audio"][:NUM_SAMPLES]]
text_samples = ds["text"][:NUM_SAMPLES]

# passing `text` to the processor will prepare inputs' `labels` key
inputs = processor(audio=speech_samples, text=text_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(device=model.device, dtype=model.dtype)

outputs = model(**inputs)
print("Loss:", outputs.loss.item())
outputs.loss.backward()
```

### TDT Training

The TDT loss has been implemented within Transformers to enable training. For faster training (around 10-50x depending on batch size), consider using NeMo's `TDTLossNumba`. Note that this requires installing the NeMo toolkit with `pip install nemo_toolkit[asr]`.  

<hfoptions id="usage">
<hfoption id="Transformers-only">

```py
from datasets import Audio, load_dataset
import torch
from transformers import AutoModelForTDT, AutoProcessor

model_id = "nvidia/parakeet-tdt-0.6b-v3-hf"
NUM_SAMPLES = 4

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTDT.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
model.train()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el['array'] for el in ds["audio"][:NUM_SAMPLES]]
text_samples = ds["text"][:NUM_SAMPLES]

# passing `text` to the processor will prepare inputs' `labels` key
inputs = processor(audio=speech_samples, text=text_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(device=model.device, dtype=model.dtype)

outputs = model(**inputs)
print("Loss:", outputs.loss.item())
outputs.loss.backward()
```

</hfoption>
<hfoption id="With TDTLossNumba">

```py
import torch
from datasets import Audio, load_dataset
from nemo.collections.asr.losses.rnnt import TDTLossNumba
from transformers import AutoModelForTDT, AutoProcessor


model_id = "nvidia/parakeet-tdt-0.6b-v3-hf"
NUM_SAMPLES = 4

# Load model and processor
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForTDT.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
model.train()

# Initialize NeMo TDT loss
# NOTE: NeMo's TDTLossNumba doesn't seem to do normalization with target lengths as suggested by its docstring so doing manually:
# - Docstring: https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/numba/rnnt_loss/rnnt_pytorch.py#L373
# - Normalization: https://github.com/NVIDIA-NeMo/NeMo/blob/main/nemo/collections/asr/parts/numba/rnnt_loss/rnnt_pytorch.py#L247-L253
loss_fn = TDTLossNumba(
    blank=model.config.blank_token_id,
    durations=model.config.durations,
    reduction="none",   
)

# Load dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el["array"] for el in ds["audio"][:NUM_SAMPLES]]
text_samples = ds["text"][:NUM_SAMPLES]

# Prepare inputs
inputs = processor(audio=speech_samples, text=text_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(device=model.device, dtype=model.dtype)

# Forward pass without computing loss
outputs = model(**inputs, compute_loss=False)

# Prepare inputs for NeMo TDT loss
# -- NOTE: convert to float32 for NeMo loss since Numba doesn't support float16/bfloat16, but keep labels as integers
encoder_lengths = torch.full((outputs.last_hidden_state.shape[0],), outputs.last_hidden_state.shape[1], dtype=torch.long, device=model.device)
labels = inputs["labels"]
target_lengths = (labels != model.config.pad_token_id).sum(-1)
losses = loss_fn(
    acts=outputs.logits.float(),
    labels=labels.long(),
    act_lens=encoder_lengths.long(),
    label_lens=target_lengths.long(),
)

# Normalize by target lengths
loss = (losses / target_lengths.float()).mean()
print(f"Loss (NeMo TDTLossNumba): {loss.item():.6f}")

# Backward pass
loss.backward()
print("\n✓ Successfully computed loss and gradients using NeMo's fast TDT loss!")
```

</hfoption>
</hfoptions>


## ParakeetTokenizer

[[autodoc]] ParakeetTokenizer

## ParakeetFeatureExtractor

[[autodoc]] ParakeetFeatureExtractor
    - __call__

## ParakeetProcessor

[[autodoc]] ParakeetProcessor
    - __call__
    - decode

## ParakeetEncoderConfig

[[autodoc]] ParakeetEncoderConfig

## ParakeetCTCConfig

[[autodoc]] ParakeetCTCConfig

## ParakeetTDTConfig

[[autodoc]] ParakeetTDTConfig

## ParakeetEncoder

[[autodoc]] ParakeetEncoder

## ParakeetForCTC

[[autodoc]] ParakeetForCTC

## ParakeetForTDT

[[autodoc]] ParakeetForTDT
