# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regenerate integration expectations using ONLY Liquid Audio's original implementation.

Use an isolated Python 3.12+ environment; do not install the Transformers checkout in it::

    pip install liquid-audio==1.3.0 torch==2.8.0 torchaudio==2.8.0 librosa==0.11.0
    pip install git+https://github.com/huggingface/transformers.git@89a727319005f512aff3714c3838c8c37630b4fa
    python tests/models/lfm2_audio/reproduce_integration_outputs.py --output /tmp/lfm2_audio_expected.json

Reference source: https://github.com/Liquid4All/liquid-audio/tree/19e65845923a7f136442c95137884ec61eb386aa
The pinned Transformers revision is the PR base, which has no native Lfm2Audio implementation.
Liquid Audio itself imports Lfm2Model from Transformers: using this shared backbone revision isolates audio
integration changes from unrelated backbone/cache changes in older Transformers releases.
The script never imports the native Lfm2Audio classes and never uploads artifacts.
"""

import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
from urllib.request import urlopen

import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor


REFERENCE_REVISION = "19e65845923a7f136442c95137884ec61eb386aa"
CHECKPOINT = "LiquidAI/LFM2.5-Audio-1.5B"
TRANSFORMERS_REVISION = "89a727319005f512aff3714c3838c8c37630b4fa"


def main(output, checkpoint_revision=None):
    torch.set_float32_matmul_precision("highest")
    model_path = Path(snapshot_download(CHECKPOINT, revision=checkpoint_revision))
    processor = LFM2AudioProcessor.from_pretrained(model_path, device="cuda").eval()
    model = LFM2AudioModel.from_pretrained(model_path, dtype=torch.bfloat16, device="cuda").eval()
    cases = [
        {"name": "asr", "prompt": "Perform ASR.", "audio": "asr.wav", "mode": "sequential", "max_new_tokens": 128},
        *[
            {
                "name": "tts_" + voice.replace(" ", "_").lower(),
                "prompt": f"Perform TTS. Use the {voice} voice.",
                "text": "Hello, how are you today?",
                "mode": "sequential",
                "max_new_tokens": 128,
            }
            for voice in ("US male", "US female", "UK male", "UK female")
        ],
        {
            "name": "interleaved",
            "prompt": "Respond with interleaved text and audio.",
            "audio": "question.wav",
            "mode": "interleaved",
            "max_new_tokens": 96,
        },
    ]
    results = {
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": Path(model_path).name,
        "reference_revision": REFERENCE_REVISION,
        "transformers_revision": TRANSFORMERS_REVISION,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("liquid-audio", "transformers", "torch", "librosa")
        },
        "device": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "cases": [],
    }
    for case in cases:
        chat = ChatState(processor)
        chat.new_turn("system")
        chat.add_text(case["prompt"])
        chat.end_turn()
        chat.new_turn("user")
        if "audio" in case:
            url = f"https://raw.githubusercontent.com/Liquid4All/liquid-audio/{REFERENCE_REVISION}/assets/{case['audio']}"
            with urlopen(url) as response:
                audio_bytes = response.read()
            case["audio_url"] = url
            case["audio_sha256"] = hashlib.sha256(audio_bytes).hexdigest()
            waveform, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            chat.add_audio(torch.from_numpy(waveform).unsqueeze(0), sampling_rate)
        else:
            chat.add_text(case["text"])
        chat.end_turn()
        chat.new_turn("assistant")
        generate = model.generate_sequential if case["mode"] == "sequential" else model.generate_interleaved
        events = list(generate(**chat, max_new_tokens=case["max_new_tokens"], text_top_k=1, audio_top_k=1))
        text_tokens = [event.item() for event in events if event.numel() == 1]
        audio_tokens = [event.reshape(-1).tolist() for event in events if event.numel() > 1]
        case["sequences"] = text_tokens
        case["audio_codes"] = audio_tokens
        case["modalities"] = [1 if event.numel() == 1 else 3 for event in events]
        case["decoded_text"] = processor.text.decode(text_tokens, skip_special_tokens=True)
        if audio_tokens:
            codes = torch.tensor(audio_tokens, device="cuda").T.unsqueeze(0)
            if torch.all(codes[..., -1] == 2048):
                codes = codes[..., :-1]
            waveform = processor.decode(codes).float().cpu()
            case["waveform_shape"] = list(waveform.shape)
            case["waveform_slice"] = waveform[0, 1000:1032].tolist()
            case["waveform_mean"] = waveform.mean().item()
            case["waveform_std"] = waveform.std().item()
        results["cases"].append(case)
        print(case["name"], len(events), case["decoded_text"], flush=True)
    Path(output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint_revision")
    main(**vars(parser.parse_args()))
