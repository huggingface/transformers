<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Processors

Multimodal models require a preprocessor capable of handling inputs that combine more than one modality. Depending on the input modality, a processor needs to convert text into an array of tensors, images into pixel values, and audio into an array of tensors with the correct sampling rate.

For example, [PaliGemma](./model_doc/paligemma) is a vision-language model that uses the [SigLIP](./model_doc/siglip) image processor and the [Llama](./model_doc/llama) tokenizer. A [`ProcessorMixin`] class wraps both of these preprocessor types, providing a single and unified processor class for a multimodal model.

Call [`~ProcessorMixin.from_pretrained`] to load a processor. Pass the input type to the processor to generate the expected model inputs, input ids and pixel values.

```py
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests

processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

prompt = "answer en Where is the cat standing?"
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt")
inputs
```

This guide describes the processor class and how to preprocess multimodal inputs.

## Processor classes

All processors inherit from the [`ProcessorMixin`] class which provides methods like [`~ProcessorMixin.from_pretrained`], [`~ProcessorMixin.save_pretrained`], and [`~ProcessorMixin.push_to_hub`] for loading, saving, and sharing processors to the Hub.

There are two ways to load a processor, with an [`AutoProcessor`] and with a model-specific processor class.

<hfoptions id="processor-class">
<hfoption id="AutoProcessor">

The [AutoClass](./model_doc/auto) API provides a simple interface to load processors without directly specifying the specific model class it belongs to.

Use [`~AutoProcessor.from_pretrained`] to load a processor.

```py
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
```

</hfoption>
<hfoption id="model-specific processor">

Processors are also associated with a specific pretrained multimodal model class. You can load a processor directly from the model class with [`~ProcessorMixin.from_pretrained`].

```py
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
```

You could also separately load the two preprocessor types, [`WhisperTokenizerFast`] and [`WhisperFeatureExtractor`].

```py
from transformers import WhisperTokenizerFast, WhisperFeatureExtractor, WhisperProcessor

tokenizer = WhisperTokenizerFast.from_pretrained("openai/whisper-tiny")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
```

</hfoption>
</hfoptions>

## Save and load a processor

[`~ProcessorMixin.save_pretrained`] writes the processor's configuration to a single `processor_config.json` file. The tokenizer and chat template are saved in their own files.

```py
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
processor.save_pretrained("./llava-onevision")
```

```text
llava-onevision/
├── chat_template.jinja
├── processor_config.json
├── tokenizer.json
└── tokenizer_config.json
```

Every sub-processor except the tokenizer is nested inside `processor_config.json` under its attribute name, such as `image_processor`, `video_processor`, or `feature_extractor`. Settings that belong to the processor itself, like `image_token`, sit at the top level. The example below is shortened.

```json
{
  "image_processor": {
    "image_processor_type": "LlavaOnevisionImageProcessor",
    "do_resize": true,
    "size": {"height": 384, "width": 384},
    "image_mean": [0.5, 0.5, 0.5]
  },
  "image_token": "<image>",
  "processor_class": "LlavaOnevisionProcessor",
  "video_processor": {
    "video_processor_type": "LlavaOnevisionVideoProcessor",
    "do_resize": true,
    "size": {"height": 384, "width": 384},
    "image_mean": [0.5, 0.5, 0.5]
  },
  "video_token": "<video>"
}
```

The other files are saved separately.

- The tokenizer writes its usual files, like `tokenizer.json` and `tokenizer_config.json`. If a processor has a second tokenizer, it's saved in a subfolder named after its attribute, like `decoder_tokenizer/`.
- The chat template is saved to `chat_template.jinja`. If a processor has several chat templates, the `default` template goes in `chat_template.jinja` and the others go in a `chat_templates/` folder.

When you load a processor with [`~ProcessorMixin.from_pretrained`], each sub-processor looks for its nested key in `processor_config.json` first. If the key isn't there, it falls back to a standalone file. Image processors and feature extractors read `preprocessor_config.json`, and video processors read `video_preprocessor_config.json` and then `preprocessor_config.json`. Older checkpoints that store each sub-processor in its own file still load this way.

> [!NOTE]
> A nested config takes priority over a standalone file. If a checkpoint has both, editing `preprocessor_config.json` or `video_preprocessor_config.json` has no effect. Edit the nested entry in `processor_config.json` instead.

## Preprocess

Processors preprocess multimodal inputs into the expected Transformers format. There are a couple combinations of input modalities that a processor can handle such as text and audio or text and image.

Automatic speech recognition (ASR) tasks require a processor that can handle text and audio inputs. Load a dataset and take a look at the `audio` and `text` columns (you can remove the other columns which aren't needed).

```py
from datasets import load_dataset

dataset = load_dataset("keithito/lj_speech", split="train")
dataset = dataset.map(remove_columns=["file", "id", "normalized_text"])
dataset[0]["audio"]
{'array': array([-7.3242188e-04, -7.6293945e-04, -6.4086914e-04, ...,
         7.3242188e-04,  2.1362305e-04,  6.1035156e-05], dtype=float32),
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/917ece08c95cf0c4115e45294e3cd0dee724a1165b7fc11798369308a465bd26/LJSpeech-1.1/wavs/LJ001-0001.wav',
 'sampling_rate': 22050}

dataset[0]["text"]
'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition'
```

Remember to resample the sampling rate to match the pretrained models required sampling rate.

```py
from datasets import Audio

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

Load a processor and pass the audio `array` and `text` columns to it.

```py
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("openai/whisper-tiny")

def prepare_dataset(example):
    audio = example["audio"]
    example.update(processor(audio=audio["array"], text=example["text"], sampling_rate=16000))
    return example
```

Apply the `prepare_dataset` function to preprocess the dataset. The processor returns `input_features` for the `audio` column and `labels` for the text column.

```py
prepare_dataset(dataset[0])
```
