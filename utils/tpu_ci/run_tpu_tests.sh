#!/usr/bin/env bash
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
#
# Run the daily model tests on the machine this is launched from and turn the reports into the
# `model_results.json` the CI dashboard reads. Meant to be run by hand from the repo root, on a
# device whose wheels are not in the shared CI images (currently TPU).
#
#   MACHINE_TYPE=multi-gpu    key the results land under: `multi-gpu` or `single-gpu`. The "gpu" is
#                             the CI's own spelling and only ever appears in report directory names.
#   ONLY_IN=IMPORTANT_MODELS  which models to run, `IMPORTANT_MODELS` or a space-separated list.
#   RUN_SLOW=1                also run the tests marked slow.
#   TMP_CACHE=<prefix>        run with a throwaway hub cache under that prefix.
#   UPLOAD=1                  upload the results to the dataset repo as well as writing them out.
#
# Only one process at a time can hold a given TPU chip, and torch_tpu aborts the process rather than
# raising when it cannot acquire one, so do not run anything else that runs a TPU op alongside.
#
# The environment needs `transformers[testing]` plus the media extras, otherwise the vision, audio and
# video tests fail on missing backends instead of telling you anything about the device. Install
# torchvision and torchaudio with the installed torch pinned, or the resolver silently replaces it:
#   uv pip install --index-url https://download.pytorch.org/whl/cpu \
#       torchvision torchaudio "torch==$(python3 -c 'import torch; print(torch.__version__.split("+")[0])')"
# `pyctcdecode` is best left out: it pins numpy < 2 and only gates a handful of CTC decoding tests.
set -euo pipefail

export TRANSFORMERS_IS_CI=yes NO_COLOR=1 OMP_NUM_THREADS=8

# `TPU_VISIBLE_DEVICES` is torch_tpu's `CUDA_VISIBLE_DEVICES` and does restrict which chips the
# process opens, but `torch.tpu.device_count()` keeps reporting every chip on the host (see the
# `visible_devices` probe in backend_gaps.py). Every test that gates on the device count would
# therefore still run, and still reach for chips the run is supposed to have given up, so a
# `single-gpu` run here would be a `multi-gpu` run reported under the other name. Leave that column
# empty instead: the dashboard reads a missing key as no data.
if [ "${MACHINE_TYPE:-multi-gpu}" = "single-gpu" ]; then
    echo "single-gpu runs are not available on this device: restricting the visible chips does not" >&2
    echo "restrict what the tests see, so the results would be a multi-gpu run under another name." >&2
    exit 1
fi

python3 -m utils.get_test_reports tests/ --suite models \
    --only-in ${ONLY_IN:-IMPORTANT_MODELS} \
    --machine-type "${MACHINE_TYPE:-multi-gpu}" \
    ${RUN_SLOW:+--run-slow} \
    ${TMP_CACHE:+--tmp-cache "$TMP_CACHE"}

python3 utils/tpu_ci/make_model_results.py reports/ --summary model_results.md ${UPLOAD:+--upload}
