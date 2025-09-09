#!/bin/bash
# Copyright (c) 2024 Your Name or Company
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex
set -o pipefail

# This script runs inference for both normal mode and with tensor parallelism
# for comparison.
#
# Usage:
#   bash compare_run_llama.sh
#
# Any arguments provided to this script will be passed to run_llama.py.

# Configurable number of GPUs
NGPU=${NGPU:-"2"}

run_normal() {
    echo "###################################"
    echo "### Running in Normal Mode      ###"
    echo "###################################"
    python run_llama.py "$@"
}

run_tp() {
    echo "#############################################"
    echo "### Running with Tensor Parallelism (TP)  ###"
    echo "#############################################"
    torchrun --nproc_per_node=${NGPU} run_llama.py --tp_size=${NGPU} --pp_size=1 "$@"
}

NORMAL_LOG="normal_run.log"
TP_LOG="tp_run.log"
DIFF_LOG="run_diff.log"

run_normal "$@" 2>&1 | tee ${NORMAL_LOG}
run_tp "$@" 2>&1 | tee ${TP_LOG}

# Filter logs to remove noisy differences
NORMAL_LOG_FILTERED="${NORMAL_LOG}.filtered"
TP_LOG_FILTERED="${TP_LOG}.filtered"

# This sed command removes ANSI color escape codes, which are noisy
# and cause false differences in the logs.
sed -E "s/\x1b\[[0-9;]*m//g" < "${NORMAL_LOG}" > "${NORMAL_LOG_FILTERED}"
sed -E "s/\x1b\[[0-9;]*m//g" < "${TP_LOG}" > "${TP_LOG_FILTERED}"


echo "############################################"
echo "### Diff between Normal and TP run logs  ###"
echo "############################################"
echo "### Log diff is being saved to ${DIFF_LOG}"
echo "############################################"
git diff --no-index --color=always --word-diff=color "${NORMAL_LOG_FILTERED}" "${TP_LOG_FILTERED}" | tee "${DIFF_LOG}" || true
