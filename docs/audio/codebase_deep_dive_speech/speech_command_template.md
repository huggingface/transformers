# Speech Command Template

Load your API key first:

```zsh
source /Users/mnuno/PycharmProjects/.env.openai
```

Generate one chunk with the reusable helper:

```zsh
/Users/mnuno/PycharmProjects/huggingface_transformers/docs/audio/codebase_deep_dive_speech/generate_speech_chunk.sh 01_introduction_chunk_01
```

If the base MP3 already exists, the helper keeps it and writes a comparison file such as:

```text
01_introduction_chunk_01_compare_20260505_143012.mp3
```

Overwrite an existing MP3:

```zsh
/Users/mnuno/PycharmProjects/huggingface_transformers/docs/audio/codebase_deep_dive_speech/generate_speech_chunk.sh --force 01_introduction_chunk_01
```

Or run the full command directly:

```zsh
python3 /Users/mnuno/.codex/skills/speech/scripts/text_to_speech.py speak \
  --model gpt-4o-mini-tts-2025-12-15 \
  --voice cedar \
  --instructions "Tone: clear and instructional. Pacing: steady. Delivery: technical narration." \
  --input-file /Users/mnuno/PycharmProjects/huggingface_transformers/docs/audio/codebase_deep_dive_speech/tts_chunks/01_introduction_chunk_01.txt \
  --out /Users/mnuno/PycharmProjects/huggingface_transformers/docs/audio/codebase_deep_dive_speech/01_introduction_chunk_01.mp3
```

Change the input and output filenames for each chunk listed in `manifest.md`.
