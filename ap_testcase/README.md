# Apertus 1.5 integration: hands-on test scripts

Standalone scripts to exercise the transformers integration against the composite checkpoint
(the official hub release by default, or a locally assembled copy).
The scripts share bootstrap and result-reporting helpers from `_common.py`, but each remains directly
executable and keeps its model-specific setup local.

## Setup (from scratch)

### 1. Clone and install

```bash
# one-time: install uv if missing (https://docs.astral.sh/uv)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/RaphaelKreft/transformers.git
cd transformers
git checkout feature/apertus_1p5_pipeline

uv venv .my-env
source .my-env/bin/activate
# testing: pytest tooling; vision: torchvision image-processing backend;
# audio: librosa & co. for audio loading (files, URLs, base64) and resampling to 24 kHz
uv pip install -e ".[testing,vision,audio]"
```

Sanity checks:

```bash
python -c "import transformers, torch; print(transformers.__version__)"   # 5.x.dev0, served from src/
python -c "import torchvision, librosa; print('media deps OK')"
```

The shared bootstrap enforces that `transformers` is imported from this checkout's `src/transformers`
directory and reports a setup failure if another installation takes precedence.

All referenced hub repos are currently public, so no `hf auth login` is needed.

### 2. Prepare the composite checkpoint

The scripts read the checkpoint from the `APERTUS1P5_CHECKPOINT` environment variable, which accepts a
local directory or a hub repo id (optionally `repo_id@revision`).

**Fast path (the default):** with the variable unset, the scripts use the official release
`swiss-ai/Apertus-v1.5-8B` directly, downloading it into the HF cache on first
use (the two processor-only scripts skip the 17 GB weight shards and fetch only the small files). To keep
a persistent local copy instead:

```bash
hf download swiss-ai/Apertus-v1.5-8B \
  --local-dir ~/Apertus-1.5-8B-composite-hf
export APERTUS1P5_CHECKPOINT=~/Apertus-1.5-8B-composite-hf
```

**Build path:** alternatively assemble the composite from its three weight sources
(roughly 35 GB of disk during the build incl. the HF cache, about 17 GB afterwards):

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

# c) assemble the composite (writes weights + tokenizer + processor + chat template) and verify it;
#    each source may be a local dir or a hub repo id (optionally `repo_id@revision`)
python src/transformers/models/apertus1p5/convert_apertus1p5_weights_to_hf.py \
  --apertus_checkpoint $MATERIAL/Apertus-1.5-8B-pruned \
  --vision_tokenizer_checkpoint $MATERIAL/apertus1p5-visionvq-hf \
  --audio_tokenizer_checkpoint swiss-ai/wavtokenizer-large-unify-40token \
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

Each script runs numbered case functions and prints one final `PASS` / `FAIL` / `SKIP` table. Independent
cases continue after a failure. Missing optional network, CUDA, or `torchrun` capabilities are skipped;
setup or case failures produce a nonzero exit code.

## Scripts

| Script | What it tests | Loads the 8B model? |
|---|---|---|
| `01_processor_single_inputs.py` | Processor on single (non-batched) inputs: text-only, +image, +audio, +both; URL fetching | no (fast) |
| `02_processor_batched_inputs.py` | Batched processing: flat + nested media, uneven counts, empty samples, strict errors | no (fast) |
| `03_image_tokenization.py` | Image tokenization from processor output vs. fully manual preprocessing | yes |
| `04_audio_tokenization.py` | Audio tokenization from processor output vs. fully manual preprocessing | yes |
| `05_generation_chat_messages.py` | Full generation from chat messages (text-only + multimodal, incl. media auto-loading), thinking activation via `enable_thinking`, seeded sampling parameters | yes |
| `06_generation_raw_text.py` | Full generation from raw text with image and audio placeholders (base-model style), incl. a batch | yes |
| `07_language_model_from_composite.py` | All text-only classes from the composite: `Apertus1p5TextConfig` extraction, `Apertus1p5TextForCausalLM` (pruned output layer, padded-logits contract with finite logits and a zero-probability tail, greedy + beam generation), bare `Apertus1p5TextModel` hidden states | yes |
| `08_multi_device_inference.py` | Multi-device placement (needs >= 2 GPUs, else skips): `device_map="auto"` sharding (fp32-keep of media tokenizers, padded-logits contract, generation parity vs single device) and `DistributedConfig(tp_size=2)` tensor parallelism over the text backbone via torchrun | yes |
| `09_training_loop.py` | Training smoke test (needs a GPU, else skips): pruned-head label contract (physical-width loss logits, `-100` masking enforced), an overfit loop on the last layer + lm_head, and a DDP variant via torchrun with >= 2 GPUs | yes |

The model-loading scripts 03-07 run on CPU and take a few minutes each (bf16 8B load is about
1 minute); 08 and 09 need CUDA devices and skip themselves otherwise. The URL-fetching case in 01
needs network access (the media auto-loading in 05 works from a local temporary file).
