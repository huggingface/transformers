<!--Copyright 2025 The OpenBMB Team and The HuggingFace Inc. team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.
--->

# MiniCPM-o 2.6

<h2>A GPT-4o Level MLLM for Vision, Speech and Multimodal Live Streaming on Your Phone</h2>

[GitHub](https://github.com/OpenBMB/MiniCPM-o) | [Online Demo](https://minicpm-omni-webdemo-us.modelbest.cn) | [Technical Blog](https://openbmb.notion.site/MiniCPM-o-2-6-A-GPT-4o-Level-MLLM-for-Vision-Speech-and-Multimodal-Live-Streaming-on-Your-Phone-185ede1b7a558042b5d5e45e6b237da9)

## Overview

The [MiniCPM-o 2.6](https://github.com/OpenBMB/MiniCPM-o) model is an end-to-end omni-modal large multimodal model proposed by the OpenBMB Team. MiniCPM-o 2.6 is built based on SigLip-400M, Whisper-medium-300M, ChatTTS-200M, and Qwen2.5-7B with a total of 8B parameters.

The model features:

_MiniCPM-o 2.6 is the latest and most capable model in the MiniCPM-o series, featuring leading visual capability with an average score of 70.2 on OpenCompass. With only 8B parameters, it surpasses widely used proprietary models like GPT-4o-202405, Gemini 1.5 Pro, and Claude 3.5 Sonnet for single image understanding. It supports state-of-the-art speech capability with bilingual real-time speech conversation and configurable voices in English and Chinese, outperforming GPT-4o-realtime on audio understanding tasks. The model introduces strong multimodal live streaming capability, accepting continuous video and audio streams independent of user queries with real-time speech interaction. It features superior efficiency with state-of-the-art token density, producing only 640 tokens when processing a 1.8M pixel image, which is 75% fewer than most models. The architecture employs an end-to-end omni-modal design with time-division multiplexing (TDM) mechanism for omni-modality streaming processing and configurable speech modeling design with multimodal system prompts._

## Usage

Inference using Huggingface transformers on NVIDIA GPUs. Requirements tested on python 3.10：

```
transformers
Pillow
torch
torchaudio
torchvision
librosa
soundfile
vector-quantize-pytorch
vocos
decord
moviepy
```

### Model initialization

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    'openbmb/MiniCPM-o-2_6',
    attn_implementation='sdpa', # sdpa or flash_attention_2, no eager
    dtype=torch.bfloat16
)
model = model.eval().cuda()
model.init_tts()

processor = AutoProcessor.from_pretrained('openbmb/MiniCPM-o-2_6')
```

If you are using an older version of PyTorch, you might encounter this issue `"weight_norm_fwd_first_dim_kernel" not implemented for 'BFloat16'`, Please convert the TTS to float32 type.

```python
model.tts.float()
```

### Omni mode

We provide two inference modes: normal generate and streaming

#### Normal generate inference

```python
image = Image.open('/assets/single.png').convert('RGB')
question = "Imitate the timbre and intonation of the input voice, describe this picture."
audio_input, _ = librosa.load('assets/female_example.wav', sr=16000, mono=True)
msgs = [{'role': 'user', 'content': [question, audio_input, image]}]
msgs = [msgs, msgs]
inputs = processor.apply_chat_template(msgs=msgs).to(model.device)
res = model.generate(
    **inputs,
    processor=processor,
    sampling=True,
    max_new_tokens=4096,
    use_tts_template=True,
    temperature=0.3,
    generate_audio=True,
    output_audio_path="./tts_desc.wav",
)
res = processor.decode(res.outputs.sequences)[0]
print(res)
```

#### Streaming inference

```python
import math
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
import tempfile
import librosa
import soundfile as sf

def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path)
    print('video_duration:', video.duration)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)

    # 1 frame + 1s audio chunk
    contents= []
    for i in range(num_units):
        frame = video.get_frame(i+1)
        image = Image.fromarray((frame).astype(np.uint8))
        audio = audio_np[sr*i:sr*(i+1)]
        if flatten:
            contents.extend(["<unit>", image, audio])
        else:
            contents.append(["<unit>", image, audio])

    return contents

# a new conversation need reset session first, it will reset the kv-cache
model.reset_session()
contents = get_video_chunk_content(video_path, flatten=False)
session_id = '123'
use_tts = True

# 1. prefill system prompt
res = model.streaming_prefill(
    session_id=session_id,
    msgs=[sys_msg],
    processor=processor
)

# 2. prefill video/audio chunks
for content in contents:
    msgs = [{"role":"user", "content": content}]
    res = model.streaming_prefill(
        session_id=session_id,
        msgs=msgs,
        processor=processor
    )
# 3. generate
res = model.streaming_generate(
    session_id=session_id,
    processor=processor,
    use_tts=use_tts,
    tts_output_chunk_size=25
)
audios = []
text = ""
if use_tts:
    for r in res:
        audio_wav = r.audio_wav
        sampling_rate = r.sampling_rate
        txt = r.text
        audios.append(audio_wav)
        text += txt

    res = np.concatenate(audios)
    sf.write("output.wav", res, samplerate=sampling_rate)
    print("text:", text)
    print("audio saved to output.wav")
else:
    for r in res:
        text += r['text']
    print("text:", text)
```

<hr/>

### Audio mode

#### Mimick

`Mimick` task reflects a model's end-to-end speech modeling capability. The model takes audio input, and outputs an ASR transcription and subsequently reconstructs the original audio with high similarity.

```python
mimick_prompt = "Please repeat each user's speech, including voice style and speech content."
audio_input, _ = librosa.load('./assets/input_examples/Trump_WEF_2018_10s.mp3', sr=16000, mono=True)
msgs = [{'role': 'user', 'content': [mimick_prompt, audio_input]}]
inputs = processor.apply_chat_template(msgs=msgs).to(model.device)

res = model.generate(
    **inputs,
    processor=processor,
    sampling=True,
    max_new_tokens=128,
    use_tts_template=True,
    temperature=0.3,
    generate_audio=True,
    output_audio_path='output_mimick.wav',
)
res = processor.decode(res.outputs.sequences)[0]
print(res)
```

<hr/>


#### Instruction-to-Speech (Voice Creation)

You can describe a voice in detail, and the model will generate a voice that matches the description.

```python
instruction = 'Speak like a male charming superstar, radiating confidence and style in every word.'
msgs = [{'role': 'user', 'content': [instruction]}]
inputs = processor.apply_chat_template(msgs=msgs).to(model.device)

res = model.generate(
    **inputs,
    processor=processor,
    sampling=True,
    max_new_tokens=128,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.3,
    output_audio_path='result_voice_creation.wav',
)
res = processor.decode(res.outputs.sequences)[0]
print(res)
```

<hr/>

#### Audio Understanding Tasks

Various audio understanding tasks such as ASR, speaker analysis, audio captioning, and sound scene tagging.

Available prompts:

- ASR (Chinese): `请仔细听这段音频片段，并将其内容逐字记录。`
- ASR (English): `Please listen to the audio snippet carefully and transcribe the content.`
- Speaker Analysis: `Based on the speaker's content, speculate on their gender, condition, age range, and health status.`
- Audio Caption: `Summarize the main content of the audio.`
- Scene Tagging: `Utilize one keyword to convey the audio's content or the associated scene.`

```python
task_prompt = "Please listen to the audio snippet carefully and transcribe the content.\n"
audio_input, _ = librosa.load('./assets/input_examples/audio_understanding.mp3', sr=16000, mono=True)
msgs = [{'role': 'user', 'content': [task_prompt, audio_input]}]
inputs = processor.apply_chat_template(msgs=msgs).to(model.device)

res = model.generate(
    **inputs,
    processor=processor,
    sampling=True,
    max_new_tokens=128,
    use_tts_template=True,
    generate_audio=True,
    temperature=0.3,
    output_audio_path='result_audio_understanding.wav',
)
res = processor.decode(res.outputs.sequences)[0]
print(res)
```

### Vision-Only mode

`MiniCPM-o-2_6` has the same inference methods as `MiniCPM-V-2_6`

#### Chat with single image

```python
image = Image.open('xx.jpg').convert('RGB')
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': [image, question]}]
inputs = processor.apply_chat_template(msgs=msgs).to(model.device)

res = model.generate(
    **inputs,
    processor=processor,
    sampling=True,
    max_new_tokens=1024,
)
res = processor.decode(res.sequences)[0]
print(res)

## for streaming generation
res = model.generate(
    **inputs,
    processor=processor,
    sampling=True,
    stream=True,
    max_new_tokens=1024,
)
generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')
```

#### Chat with multiple images

```python
image1 = Image.open('image1.jpg').convert('RGB')
image2 = Image.open('image2.jpg').convert('RGB')
question = 'Compare image 1 and image 2, tell me about the differences between image 1 and image 2.'
msgs = [{'role': 'user', 'content': [image1, image2, question]}]
inputs = processor.apply_chat_template(msgs=msgs).to(model.device)

res = model.generate(
    **inputs,
    processor=processor,
    sampling=True,
    max_new_tokens=1024,
)
res = processor.decode(res.sequences)[0]
print(res)
```

#### In-context few-shot learning

```python
question = "production date"
image1 = Image.open('example1.jpg').convert('RGB')
answer1 = "2023.08.04"
image2 = Image.open('example2.jpg').convert('RGB')
answer2 = "2007.04.24"
image_test = Image.open('test.jpg').convert('RGB')
msgs = [
    {'role': 'user', 'content': [image1, question]}, {'role': 'assistant', 'content': [answer1]},
    {'role': 'user', 'content': [image2, question]}, {'role': 'assistant', 'content': [answer2]},
    {'role': 'user', 'content': [image_test, question]}
]
inputs = processor.apply_chat_template(msgs=msgs).to(model.device)

res = model.generate(
    **inputs,
    processor=processor,
    sampling=True,
    max_new_tokens=1024,
)
res = processor.decode(res.sequences)[0]
print(res)
```

#### Chat with video

```python
from decord import VideoReader, cpu
import numpy as np

MAX_NUM_FRAMES=64 # if cuda OOM set a smaller number
def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]
    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

video_path ="video_test.mp4"
frames = encode_video(video_path)
question = "Describe the video"
msgs = [{'role': 'user', 'content': frames + [question]}]
inputs = processor.apply_chat_template(msgs=msgs).to(model.device)

# Set decode params for video
res = model.generate(
    **inputs,
    processor=processor,
    sampling=True,
    max_new_tokens=1024,
    use_image_id=False,
    max_slice_nums=2,  # use 1 if cuda OOM and video resolution > 448*448
)
res = processor.decode(res.sequences)[0]
print(res)
```

Please look at [GitHub](https://github.com/OpenBMB/MiniCPM-o) for more detail about usage.

## Usage Tips

### system prompts

We provide some system prompts for different tasks, you can use it as this:

```python
def get_sys_prompt(ref_audio=None, mode="default", language="zh"):
    """
    Choose different system prompts according to different tasks
    Args:
        ref_audio: if ref_audio is not None, will use the voice cloning prompts, and the voice
                   generated by the model will refer to the timbre of ref audio
        mode:
            "default": default system prompt and not refer to any task
            "omni": input video and audio simultaneously
            "audio_assistant": Default voice-only mode, the model will use the ref_audio's voice to reply user's question as a helpful assistant.
            "audio_roleplay": Roleplay voice-only mode, the model will use the ref_audio's voice to reply, and also role-play the character based on the audio prompt.
            "voice_cloning": TTS mode, the model will clone the voice of ref_audio.
        language: prompts language, the model has the ability to automatically select the response language based on the question language
    """
    if ref_audio is not None:
        if not isinstance(ref_audio, np.ndarray):
            raise TypeError("ref_audio error, should be np.ndarray, but got {}".format(type(ref_audio)))
    if mode == "omni":
        if language == "zh":
            sys_prompt = "你是一个AI助手。你能接受视频，音频和文本输入并输出语音和文本。"
            vc_prompt_prefix = sys_prompt + "模仿输入音频中的声音特征。"
            vc_prompt_suffix = "作为助手，你将使用这种声音风格说话。"
        else:
            sys_prompt = "You are a helpful assistant. You can accept video, audio and text input and output voice and text. "
            vc_prompt_prefix = sys_prompt + "Clone the voice in the provided audio prompt."
            vc_prompt_suffix = "As an assistant, you will speak using this voice style."

        if ref_audio is not None:
            sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}

        else:
            sys_msgs = {"role": "user", "content": [sys_prompt]}

        return sys_msgs
    elif mode == "audio_assistant":
        if language == "zh":
            vc_prompt_prefix = "模仿输入音频中的声音特征。"
            vc_prompt_suffix = "作为助手，你将使用这种声音风格说话。"
        else:
            vc_prompt_prefix = "Clone the voice in the provided audio prompt."
            vc_prompt_suffix = "As an assistant, you will speak using this voice style."

        if ref_audio is not None:
            sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}

        else:
            logger.warning(
                "Warning: ref_audio is None, speech generation will be performed based on the default voice."
            )
            sys_msgs = {"role": "user", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

        return sys_msgs
    elif mode == "audio_roleplay":
        if language == "zh":
            vc_prompt_prefix = "模仿输入音频中的声音特征。"
            vc_prompt_suffix = "假装你是上述音频中的人物，与我进行对话。"
        else:
            vc_prompt_prefix = "Clone the voice in the provided audio prompt."
            vc_prompt_suffix = "Try to role-play the character based on the audio prompt above."

        if ref_audio is not None:
            sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio, vc_prompt_suffix]}
        else:
            print("Warning: ref_audio is None, speech generation will be performed based on the default voice.")
            sys_msgs = {"role": "user", "content": ["Use the <reserved_53> voice.", vc_prompt_suffix]}

        return sys_msgs
    elif mode == "voice_cloning":
        if language == "zh":
            vc_prompt_prefix = "模仿输入音频中的声音特征。"
        else:
            vc_prompt_prefix = "Clone the voice in the provided audio prompt."

        if ref_audio is not None:
            sys_msgs = {"role": "user", "content": [vc_prompt_prefix, ref_audio]}
        else:
            raise ValueError("ref_audio con't be None in voice_cloning mode.")

        return sys_msgs
    else:
        sys_prompt = "You are a helpful assistant. You can accept audio and text input and output voice and text."
        sys_msgs = {"role": "user", "content": [sys_prompt]}

        return sys_msgs
```

### Inference with llama.cpp<a id="llamacpp"></a>

MiniCPM-o 2.6 (vision-only mode) can run with llama.cpp. See our fork of [llama.cpp](https://github.com/OpenBMB/llama.cpp/tree/minicpm-omni) and [readme](https://github.com/OpenBMB/llama.cpp/blob/minicpm-omni/examples/llava/README-minicpmo2.6.md) for more detail.

### Int4 quantized version

Download the int4 quantized version for lower GPU memory (7GB) usage: [MiniCPM-o-2_6-int4](https://huggingface.co/openbmb/MiniCPM-o-2_6-int4).

## License

#### Model License

- The code in this repo is released under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License.
- The usage of MiniCPM-o and MiniCPM-V series model weights must strictly follow [MiniCPM Model License.md](https://github.com/OpenBMB/MiniCPM/blob/main/MiniCPM%20Model%20License.md).
- The models and weights of MiniCPM are completely free for academic research. After filling out a ["questionnaire"](https://modelbest.feishu.cn/share/base/form/shrcnpV5ZT9EJ6xYjh3Kx0J6v8g) for registration, MiniCPM-o 2.6 weights are also available for free commercial use.

#### Statement

- As an LMM, MiniCPM-o 2.6 generates contents by learning a large mount of multimodal corpora, but it cannot comprehend, express personal opinions or make value judgement. Anything generated by MiniCPM-o 2.6 does not represent the views and positions of the model developers
- We will not be liable for any problems arising from the use of the MinCPM-V models, including but not limited to data security issues, risk of public opinion, or any risks and problems arising from the misdirection, misuse, dissemination or misuse of the model.

## Key Techniques and Other Multimodal Projects

👏 Welcome to explore key techniques of MiniCPM-o 2.6 and other multimodal projects of our team:

[VisCPM](https://github.com/OpenBMB/VisCPM/tree/main) | [RLHF-V](https://github.com/RLHF-V/RLHF-V) | [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD) | [RLAIF-V](https://github.com/RLHF-V/RLAIF-V)

## Citation

If you find our work helpful, please consider citing our papers 📝 and liking this project ❤️！

```bib
@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and others},
  journal={arXiv preprint arXiv:2408.01800},
  year={2024}
}
```
