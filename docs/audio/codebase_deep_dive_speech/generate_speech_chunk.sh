#!/usr/bin/env zsh
set -euo pipefail

FORCE_ARGS=()

if [[ "${1:-}" == "--force" ]]; then
  FORCE_ARGS=(--force)
  shift
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 [--force] <chunk-name-or-path>"
  echo "Example: $0 01_introduction_chunk_01"
  echo "Example: $0 --force 01_introduction_chunk_01"
  exit 2
fi

source /Users/mnuno/PycharmProjects/.env.openai

ROOT="/Users/mnuno/PycharmProjects/huggingface_transformers/docs/audio/codebase_deep_dive_speech"
TTS="/Users/mnuno/.codex/skills/speech/scripts/text_to_speech.py"
MODEL="gpt-4o-mini-tts-2025-12-15"
VOICE="cedar"
INSTRUCTIONS="Tone: clear and instructional. Pacing: steady. Delivery: technical narration."

INPUT="$1"
if [[ "$INPUT" != /* ]]; then
  INPUT="${ROOT}/tts_chunks/${INPUT:r}.txt"
fi

if [[ ! -f "$INPUT" ]]; then
  echo "Input chunk not found: $INPUT" >&2
  exit 1
fi

BASENAME="${INPUT:t:r}"
OUT="${ROOT}/${BASENAME}.mp3"

if [[ -e "$OUT" && ${#FORCE_ARGS[@]} -eq 0 ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  OUT="${ROOT}/${BASENAME}_compare_${STAMP}.mp3"
  echo "Base MP3 already exists; writing comparison file instead:"
  echo "$OUT"
fi

python3 "$TTS" speak \
  --model "$MODEL" \
  --voice "$VOICE" \
  --instructions "$INSTRUCTIONS" \
  --input-file "$INPUT" \
  --out "$OUT" \
  "${FORCE_ARGS[@]}"

echo "Wrote $OUT"
