# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""
Agentic CLI for Transformers — single-command access to all major use-cases.

This package adds ~30 CLI commands to ``transformers``, covering inference
(text, vision, audio, video, multimodal), training, quantization, export,
and model inspection. Every command is designed to be invoked by an AI agent
or a human with no Python scripting required.

Integration with the main CLI is minimal: ``app.py`` exposes a single
``register_agentic_commands(app)`` function that is called from
``transformers.cli.transformers``. Removing that one call disables the
entire module.

Quick reference — run ``transformers <command> --help`` for any command::

    # Inference
    transformers classify --text "Great movie!"
    transformers generate --model meta-llama/Llama-3.2-1B-Instruct --prompt "Hello" --stream
    transformers transcribe --model openai/whisper-small --audio recording.wav

    # Training
    transformers train text-classification --model bert-base-uncased --dataset glue/sst2 --output ./out

    # Quantization & export
    transformers quantize --model meta-llama/Llama-3.1-8B --method bnb-4bit --output ./out
    transformers export onnx --model bert-base-uncased --output ./bert-onnx/

    # Utilities
    transformers inspect meta-llama/Llama-3.2-1B-Instruct
    transformers tokenize --model meta-llama/Llama-3.2-1B-Instruct --text "Hello, world!"
"""
