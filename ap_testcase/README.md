# Apertus 1.5 integration: hands-on test scripts

Standalone scripts to exercise the transformers integration against a local composite checkpoint.
Each script is self-contained and prints `[OK]`-style checks.

## Setup (from scratch)

### 1. Clone and install

```bash
git clone https://github.com/RaphaelKreft/transformers.git
cd transformers
git checkout feature/apertus_1p5_pipeline

uv venv .my-env
source .my-env/bin/activate
uv pip install -e ".[testing]"
uv pip install torchvision librosa   # torchvision: image processor backend; librosa: audio file/URL loading
```

Sanity check: `python -c "import transformers, torch; print(transformers.__version__)"` should report a
`5.x.dev0` version served from `src/`.

### 2. Prepare the composite checkpoint

The scripts read the checkpoint path from the `APERTUS1P5_CHECKPOINT` environment variable
(default: `/Users/rkre/swissai_repos/material/Apertus-1.5-8B-composite-hf`). If you already have a composite
checkpoint, export the variable and skip ahead. Otherwise assemble one from its three weight sources
(roughly 35 GB of disk during the build, about 17 GB afterwards):

```bash
MATERIAL=~/apertus-material && mkdir -p $MATERIAL

# a) Apertus 1.5 text backbone with the PRUNED output layer (config carries `output_vocab_size: 131072`)
hf download apertus-ai/Apertus-v1.5-8B-integration --revision refs/pr/1 \
  --local-dir $MATERIAL/Apertus-1.5-8B-pruned
# (alternatively skip this download: the converter in step d also accepts hub ids directly, e.g.
#  --apertus_checkpoint apertus-ai/Apertus-v1.5-8B-integration@refs/pr/1, cached in the HF cache)

# b) vision tokenizer: downloads BAAI/Emu3.5-VisionTokenizer, runs the bit-exact parity suite against the
#    original code, and saves the converted encode-only weights (fp32, ~0.9 GB)
python scripts/check_apertus1p5_vision_tokenizer_parity.py --save_converted $MATERIAL/apertus1p5-visionvq-hf

# c) audio codec: download the original WavTokenizer checkpoint and convert it
CKPT=$(python -c "from huggingface_hub import hf_hub_download; \
print(hf_hub_download('novateur/WavTokenizer-large-unify-40token','wavtokenizer_large_unify_600_24k.ckpt'))")
python src/transformers/models/wavtokenizer/convert_wavtokenizer_checkpoint.py \
  --checkpoint_path "$CKPT" --output_dir $MATERIAL/wavtokenizer-large-unify-40token-hf

# d) assemble the composite (writes weights + tokenizer + processor + patched chat template) and verify it
python src/transformers/models/apertus1p5/convert_apertus1p5_weights_to_hf.py \
  --apertus_checkpoint $MATERIAL/Apertus-1.5-8B-pruned \
  --vision_tokenizer_checkpoint $MATERIAL/apertus1p5-visionvq-hf \
  --audio_tokenizer_checkpoint $MATERIAL/wavtokenizer-large-unify-40token-hf \
  --output_dir $MATERIAL/Apertus-1.5-8B-composite-hf --verify

export APERTUS1P5_CHECKPOINT=$MATERIAL/Apertus-1.5-8B-composite-hf
```

Optional cross-check if you pruned a backbone yourself instead of downloading the pruned one:
`python scripts/check_apertus1p5_pruning_crosscheck.py --local <your-pruned-backbone-dir>` compares your
`lm_head.weight` bit-exactly against the reference pruning on the hub.

### 3. Run the scripts

```bash
source .my-env/bin/activate
python ap_testcase/01_processor_single_inputs.py
```

## Scripts

| Script | What it tests | Loads the 8B model? |
|---|---|---|
| `01_processor_single_inputs.py` | Processor on single (non-batched) inputs: text-only, +image, +audio, +both; URL fetching | no (fast) |
| `02_processor_batched_inputs.py` | Batched processing: flat + nested media, uneven counts, empty samples, strict errors | no (fast) |
| `03_image_tokenization.py` | Image tokenization from processor output vs. fully manual preprocessing | yes |
| `04_audio_tokenization.py` | Audio tokenization from processor output vs. fully manual preprocessing | yes |
| `05_generation_chat_messages.py` | Full generation from chat messages (text-only + multimodal, incl. media auto-loading) | yes |
| `06_generation_raw_text.py` | Full generation from raw text with placeholders (base-model style), incl. a batch | yes |
| `07_language_model_from_composite.py` | All text-only classes from the composite: `Apertus1p5TextConfig` extraction, `Apertus1p5TextForCausalLM` (pruned output layer, greedy + beam generation), bare `Apertus1p5TextModel` hidden states | yes |

The model-loading scripts run on CPU and take a few minutes each (bf16 8B load is about 1 minute).
The URL-fetching section in 01 and the media auto-loading in 05 need network access.
