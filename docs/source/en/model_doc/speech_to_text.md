<!--Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2020-10-11 and added to Hugging Face Transformers on 2021-03-10 and contributed by [valhalla](https://huggingface.co/valhalla).*

# Speech2Text

[Speech2Text](https://huggingface.co/papers/2010.05171) is an extension of the Fairseq framework designed for speech-to-text tasks like automatic speech recognition and speech translation. It offers a full pipeline for data preprocessing, model training, and both offline and online inference. The toolkit supports state-of-the-art RNN, Transformer, and Conformer architectures, along with open-source training recipes. It also enables integration with Fairseq’s machine translation and language models for multi-task or transfer learning applications.

<hfoptions id="usage">
<hfoption id="Pipeline">

```py
import torch
from transformers import pipeline

pipeline = pipeline(task="automatic-speech-recognition", model="facebook/s2t-small-librispeech-asr", dtype="auto")
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
```

</hfoption>
<hfoption id="Speech2TextForConditionalGeneration">

```py
import torch
from datasets import load_dataset
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr", dtype="auto")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
inputs = processor(ds[0]["audio"]["array"], sampling_rate=ds[0]["audio"]["sampling_rate"], return_tensors="pt")
generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])
print(f"Transcription: {processor.batch_decode(generated_ids, skip_special_tokens=True)}")
```

</hfoption>
</hfoptions>

## Speech2TextConfig

[[autodoc]] Speech2TextConfig

## Speech2TextTokenizer

[[autodoc]] Speech2TextTokenizer
    - build_inputs_with_special_tokens
    - get_special_tokens_mask
    - create_token_type_ids_from_sequences
    - save_vocabulary

## Speech2TextFeatureExtractor

[[autodoc]] Speech2TextFeatureExtractor
    - __call__

## Speech2TextProcessor

[[autodoc]] Speech2TextProcessor
    - __call__
    - from_pretrained
    - save_pretrained
    - batch_decode
    - decode

## Speech2TextModel

[[autodoc]] Speech2TextModel
    - forward

## Speech2TextForConditionalGeneration

[[autodoc]] Speech2TextForConditionalGeneration
    - forward

