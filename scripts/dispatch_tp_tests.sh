#!/usr/bin/env bash
# dispatch_tp_tests.sh — round-robin dispatch of tensor-parallel-related tests.
#
# Usage:
#   ./scripts/dispatch_tp_tests.sh <model_type> [options] [-- pytest extra args...]
#   ./scripts/dispatch_tp_tests.sh --continuous-batching [options]
#
# Options:
#   -j N                  Max parallel workers for CPU suites (default: auto)
#   --continuous-batching Include continuous-batching + TP tests (GPU multi-accel)
#   --megamoe             Include DeepGEMM megamoe integration tests (deepseek_v4 only)
#   --mixin-only          Only run TP mixin tests (default when model is given)
#   --all                 Mixin + continuous-batching + megamoe (megamoe if deepseek_v4)
#   --dry-run             Print round-robin plan without executing
#   -h, --help            Show help
#
# Examples:
#   ./scripts/dispatch_tp_tests.sh qwen2
#   ./scripts/dispatch_tp_tests.sh deepseek_v4 --all -j 1
#   ./scripts/dispatch_tp_tests.sh --continuous-batching -j 2
#
# Notes:
# - Uses env_tp in the repo root by default (override with ENV_TP=...).
# - Host-aware scheduling (auto):
#     * CPU unit/plan: batched pytest, parallel across -j workers
#     * CPU mixin (gloo): parallel; on GPU hosts pytest runs with CUDA hidden so tests execute
#     * GPU CB: parallel NCCL jobs (2 GPUs each); CUDA_VISIBLE_DEVICES=0,1 / 2,3 / …
#       Up to GPU_COUNT/2 CB workers (4 on 8 GPUs); overlaps with CPU mixin when both scheduled
#     * GPU megamoe: serial after CPU+CB (torchrun EP=8, all GPUs)
# - Mixin tests require RUN_TENSOR_PARALLEL_TESTS=1 (CPU-only inside pytest; dispatch hides GPUs).
# - Continuous-batching tests require 2+ accelerators; megamoe requires 8 GPUs.
# - Dispatch logs: tp_dispatch_logs/<model>_<timestamp>/ (override with TP_DISPATCH_LOG_DIR=...).
#     summary.txt   — pass/fail/skip overview (start here)
#     passed.txt    — passed test names (one per line)
#     failed.txt    — failed test names (one per line)
#     skipped.txt   — skipped test names with reason (name|reason)
#     logs/         — one pytest log per test
#     workers/      — internal batch worker logs (debug only)
# - Logs are kept after the run (not deleted on exit).
# - Disable colors with NO_COLOR=1.
# - Dispatch excludes flaky CB tests (cuda_graph + async); see DISPATCH_SKIP_TESTS below.

set -euo pipefail

# Short test names excluded from dispatch (still runnable via pytest directly).
DISPATCH_SKIP_TESTS=(
  test_continuous_batching_tp_cancellation_realistic
  test_continuous_batching_tp_with_cuda_graph_and_async
)

MODEL=""
JOBS=0
DRY_RUN=0
INCLUDE_MIXIN=0
INCLUDE_CB=0
INCLUDE_MEGAMOE=0
PYTEST_EXTRA=()
# -rs: skip reasons; -rfE: recap failed/error; --tb=long: full tracebacks in worker logs
PYTEST_REPORT=( -rs -rfE --tb=long )
PYTEST=()
PYTHON=()
START_EPOCH=""
COLLECTION_DONE=0
COLLECTED_ALL_TESTS=()
HAS_CUDA=0
GPU_COUNT=0
CPU_COUNT=0
CPU_JOBS=1
GPU_CB_JOBS=1
SKIPPED_HOST_COUNT=0

model_test_class_name() {
  local model="$1"
  local part class=""
  local -a parts
  IFS='_' read -r -a parts <<< "$model"
  for part in "${parts[@]}"; do
    class+="$(tr '[:lower:]' '[:upper:]' <<< "${part:0:1}")${part:1}"
  done
  echo "${class}ModelTest"
}

tp_plan_nodeid() {
  local test_file="tests/models/${MODEL}/test_modeling_${MODEL}.py"
  [[ -n "$MODEL" && -f "$test_file" ]] || return 0
  echo "${test_file}::$(model_test_class_name "$MODEL")::test_tp_plan_matches_params"
}

# ── Colors (respect NO_COLOR) ───────────────────────────────────────────────
if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  C_RESET='\033[0m'
  C_BOLD='\033[1m'
  C_DIM='\033[2m'
  C_RED='\033[31m'
  C_GREEN='\033[32m'
  C_YELLOW='\033[33m'
  C_BLUE='\033[34m'
  C_MAGENTA='\033[35m'
  C_CYAN='\033[36m'
else
  C_RESET='' C_BOLD='' C_DIM='' C_RED='' C_GREEN='' C_YELLOW='' C_BLUE='' C_MAGENTA='' C_CYAN=''
fi

c_header() { printf '%b%s%b\n' "${C_BOLD}${C_CYAN}" "$*" "${C_RESET}"; }
c_info()   { printf '%b%s%b\n' "${C_BLUE}" "$*" "${C_RESET}"; }
c_ok()     { printf '%b%s%b\n' "${C_GREEN}" "$*" "${C_RESET}"; }
c_warn()   { printf '%b%s%b\n' "${C_YELLOW}" "$*" "${C_RESET}"; }
c_err()    { printf '%b%s%b\n' "${C_RED}" "$*" "${C_RESET}"; }
c_dim()    { printf '%b%s%b\n' "${C_DIM}" "$*" "${C_RESET}"; }
c_worker() { printf '%b%s%b\n' "${C_MAGENTA}" "$*" "${C_RESET}"; }

short_test_name() {
  local nodeid="$1"
  echo "${nodeid##*::}"
}

test_source_file() {
  echo "${1%%::*}"
}

log_basename() {
  basename "$1"
}

c_skip() { printf '%b%s%b\n' "${C_YELLOW}" "$*" "${C_RESET}"; }

parse_pytest_outcome() {
  local log="$1"
  local nodeid="$2"
  local short outcome reason line

  short="$(short_test_name "$nodeid")"

  # Match same-line results, rich-plugin [PASSED], and PASSED/FAIL on lines shortly
  # after the nodeid (batch runs with live-log output split across lines).
  outcome="$(awk -v nodeid="$nodeid" -v short="$short" '
    function outcome_on_line(s,    m) {
      if (match(s, /\[(PASSED|FAILED|SKIPPED|ERROR)\]/, m)) {
        return tolower(m[1])
      }
      if (match(s, / (PASSED|FAILED|SKIPPED|ERROR)/, m)) {
        return tolower(m[1])
      }
      if (match(s, /^(PASSED|FAILED|SKIPPED|ERROR)/, m)) {
        return tolower(m[1])
      }
      return ""
    }
    function line_matches_test(s) {
      return index(s, nodeid) || index(s, "::" short)
    }
    {
      o = outcome_on_line($0)
      if (o != "" && line_matches_test($0)) {
        print o
        exit
      }
      if (line_matches_test($0)) {
        pending = 1
        pending_start = NR
        next
      }
      if (pending && NR <= pending_start + 12) {
        o = outcome_on_line($0)
        if (o != "") {
          print o
          exit
        }
      }
      if (pending && NR > pending_start + 12) {
        pending = 0
      }
    }
  ' "$log")"

  if [[ -z "$outcome" ]]; then
    echo "unknown|"
    return
  fi

  reason=""
  if [[ "$outcome" == "skipped" ]]; then
    line="$(grep -F "SKIPPED [1] ${nodeid}:" "$log" 2>/dev/null | tail -1 || true)"
    if [[ -z "$line" ]]; then
      line="$(grep -F "SKIPPED [1] " "$log" 2>/dev/null | grep -F "::${short}" | tail -1 || true)"
    fi
    if [[ -n "$line" ]]; then
      reason="${line#*: }"
    fi
  fi
  echo "${outcome}|${reason}"
}

test_log_path() {
  local nodeid="$1"
  echo "$TMPDIR/logs/$(short_test_name "$nodeid").log"
}

save_test_log() {
  local worker_log="$1"
  local nodeid="$2"
  local dest excerpt

  dest="$(test_log_path "$nodeid")"
  mkdir -p "$(dirname "$dest")"

  if [[ "$worker_log" == "$dest" ]]; then
    return 0
  fi

  excerpt="$(extract_test_run_section "$worker_log" "$nodeid")"
  if [[ -z "$excerpt" ]]; then
    excerpt="$(extract_failures_section_for_test "$worker_log" "$nodeid")"
  fi
  if [[ -z "$excerpt" ]]; then
    excerpt="$(awk '/^={3,} FAILURES /{p=1} p{print} /^={3,} warnings summary/{exit}' "$worker_log")"
  fi
  if [[ -n "$excerpt" ]]; then
    {
      echo "# nodeid: $nodeid"
      echo "# ---"
      printf '%s\n' "$excerpt"
    } > "$dest"
  else
    short="$(short_test_name "$nodeid")"
    awk -v nodeid="$nodeid" -v short="$short" '
      index($0, nodeid) || index($0, short) { print }
    ' "$worker_log" > "$dest"
  fi
  if [[ ! -s "$dest" ]]; then
    cp "$worker_log" "$dest"
  fi
}

extract_test_run_section() {
  local log="$1"
  local nodeid="$2"

  awk -v nodeid="$nodeid" '
    function run_start(line) {
      return index(line, "===== RUN ") && index(line, nodeid) && index(line, " =====")
    }
    function any_run_start(line) {
      return index(line, "===== RUN ") && index(line, " =====")
    }
    run_start($0) { capture=1; print; next }
    capture && any_run_start($0) && !run_start($0) { exit }
    capture { print }
  ' "$log"
}

extract_failures_section_for_test() {
  local log="$1"
  local nodeid="$2"
  local short

  short="$(short_test_name "$nodeid")"
  awk -v short="$short" -v nodeid="$nodeid" '
    /^={3,} FAILURES / { inf=1; next }
    inf && /^={3,}/ { exit }
    inf { print }
  ' "$log" | awk -v short="$short" -v nodeid="$nodeid" '
    index($0, short) || index($0, nodeid) { capture=1 }
    capture && /^_{5,}/ && !index($0, short) && !index($0, nodeid) { exit }
    capture { print }
  '
}

record_test_result() {
  local outcome="$1"
  local nodeid="$2"
  local reason="${3:-}"
  local short

  short="$(short_test_name "$nodeid")"

  case "$outcome" in
    passed)
      echo "$short" >> "$TMPDIR/passed.txt"
      ;;
    skipped)
      echo "${short}|${reason}" >> "$TMPDIR/skipped.txt"
      ;;
    failed|error)
      echo "$short" >> "$TMPDIR/failed.txt"
      ;;
    unknown)
      ;;
  esac
}

finalize_test_result() {
  local worker_log="$1"
  local nodeid="$2"
  local outcome="$3"
  local reason="${4:-}"

  save_test_log "$worker_log" "$nodeid"
  record_test_result "$outcome" "$nodeid" "$reason"
}

classify_and_record_pytest_results() {
  local log="$1"
  shift
  local -a nodeids=("$@")
  local nodeid outcome reason any_failed=0

  for nodeid in "${nodeids[@]}"; do
    IFS='|' read -r outcome reason <<< "$(parse_pytest_outcome "$log" "$nodeid")"
    finalize_test_result "$log" "$nodeid" "$outcome" "$reason"
    case "$outcome" in
      passed|skipped) ;;
      failed|error) any_failed=1 ;;
      unknown) any_failed=1 ;;
    esac
  done
  return "$any_failed"
}

print_failure_excerpt_to_console() {
  local log_file="$1"
  local line printed=0

  [[ -f "$log_file" ]] || return 0

  while IFS= read -r line; do
    [[ "$line" =~ ^# ]] && continue
    if [[ "$line" =~ ^E[[:space:]]+(AssertionError|ProcessRaisedException|.*Error:) ]]; then
      c_err "      ${line#E       }"
      printed=1
    elif [[ "$line" =~ ^E[[:space:]]+ ]]; then
      c_dim "      ${line#E       }"
      printed=1
    elif [[ "$line" =~ ^FAILED[[:space:]] || "$line" =~ ^ERROR[[:space:]] ]]; then
      c_dim "      ${line}"
      printed=1
    fi
  done < "$log_file"

  if [[ "$printed" -eq 0 ]]; then
    c_dim "      (see logs/$(basename "$log_file") for full output)"
  fi
}

write_summary() {
  local elapsed="$1"
  local out="$TMPDIR/summary.txt"
  local -a passed_tests=() failed_tests=() skipped_tests=()
  local passed=0 failed=0 skipped=0 short entry reason

  [[ -f "$TMPDIR/passed.txt" ]] && mapfile -t passed_tests < "$TMPDIR/passed.txt"
  [[ -f "$TMPDIR/failed.txt" ]] && mapfile -t failed_tests < "$TMPDIR/failed.txt"
  [[ -f "$TMPDIR/skipped.txt" ]] && mapfile -t skipped_tests < "$TMPDIR/skipped.txt"
  passed=${#passed_tests[@]}
  failed=${#failed_tests[@]}
  skipped=${#skipped_tests[@]}

  {
    echo "TP test dispatch — ${MODEL:-(none)}"
    echo "Duration: $(format_duration "$elapsed")"
    echo "Passed: $passed  Failed: $failed  Skipped: $skipped"
    echo "Logs: $TMPDIR/logs/"
    echo

    if [[ "$failed" -gt 0 ]]; then
      echo "FAILED (open logs/<name>.log for output):"
      for short in "${failed_tests[@]}"; do
        echo "  ✗ $short  →  logs/${short}.log"
      done
      echo
    fi

    if [[ "$skipped" -gt 0 ]]; then
      echo "SKIPPED:"
      for entry in "${skipped_tests[@]}"; do
        short="${entry%%|*}"
        reason="${entry#*|}"
        if [[ -n "$reason" ]]; then
          echo "  - $short  ($reason)"
        else
          echo "  - $short"
        fi
      done
      echo
    fi

    if [[ "$passed" -gt 0 ]]; then
      echo "PASSED:"
      for short in "${passed_tests[@]}"; do
        echo "  ✓ $short  →  logs/${short}.log"
      done
    fi
  } > "$out"
}

announce_test_result() {
  local worker_label="$1"
  local nodeid="$2"
  local outcome="$3"
  local reason="${4:-}"
  local short test_log

  short="$(short_test_name "$nodeid")"
  test_log="logs/${short}.log"

  case "$outcome" in
    passed)
      c_ok "${worker_label} PASS $short"
      ;;
    skipped)
      c_skip "${worker_label} SKIP $short"
      [[ -n "$reason" ]] && c_dim "      ${reason}"
      ;;
    *)
      c_err "${worker_label} FAIL $short"
      print_failure_excerpt_to_console "$TMPDIR/$test_log"
      ;;
  esac
  c_dim "      log: ${test_log}  source: $(test_source_file "$nodeid")"
}

run_and_record_pytest() {
  local worker_label="$1"
  local worker_log="$2"
  local cpu_isolate="$3"
  local nodeid="$4"
  local gpu_devices="${5:-}"
  local -a pytest_cmd=()
  local outcome reason failed=0 test_log

  test_log="$(test_log_path "$nodeid")"
  mkdir -p "$(dirname "$test_log")"

  if [[ "$cpu_isolate" -eq 1 ]]; then
    pytest_cmd=(pytest_cpu_isolated)
  elif [[ -n "$gpu_devices" ]]; then
    pytest_cmd=(pytest_gpu_pinned "$gpu_devices")
  else
    pytest_cmd=("${PYTEST[@]}")
  fi

  {
    echo "# nodeid: $nodeid"
    echo "# run: $(date -Iseconds)"
    echo "# ---"
  } > "$test_log"

  "${pytest_cmd[@]}" "$nodeid" -v "${PYTEST_REPORT[@]}" "${PYTEST_EXTRA[@]}" >>"$test_log" 2>&1 || true

  IFS='|' read -r outcome reason <<< "$(parse_pytest_outcome "$test_log" "$nodeid")"
  finalize_test_result "$test_log" "$nodeid" "$outcome" "$reason"
  announce_test_result "$worker_label" "$nodeid" "$outcome" "$reason"

  case "$outcome" in
    passed|skipped) ;;
    *) failed=1 ;;
  esac
  return "$failed"
}

format_duration() {
  local secs="$1"
  local h=$((secs / 3600))
  local m=$(((secs % 3600) / 60))
  local s=$((secs % 60))
  if [[ "$h" -gt 0 ]]; then
    printf '%dh %dm %ds' "$h" "$m" "$s"
  elif [[ "$m" -gt 0 ]]; then
    printf '%dm %ds' "$m" "$s"
  else
    printf '%ds' "$s"
  fi
}

activate_env_tp() {
  local repo_root="$1"
  local default_env="${repo_root}/env_tp"
  local env_dir="${ENV_TP:-$default_env}"

  if [[ ! -f "${env_dir}/bin/activate" ]]; then
    c_err "env_tp not found at: ${env_dir}"
    c_dim "Set ENV_TP to the venv path, or create env_tp in the repo root."
    exit 1
  fi

  # shellcheck disable=SC1091
  source "${env_dir}/bin/activate"
  PYTEST=( "${env_dir}/bin/pytest" )
  PYTHON=( "${env_dir}/bin/python" )
}

detect_host_capabilities() {
  local detected
  detected="$("${PYTHON[@]}" -c "
import os
try:
    import torch
    cuda = torch.cuda.is_available()
    gpus = torch.cuda.device_count() if cuda else 0
    accel = cuda or (hasattr(torch, 'xpu') and torch.xpu.is_available())
except Exception:
    cuda = False
    gpus = 0
    accel = False
cpus = os.cpu_count() or 1
print(f'{int(accel)} {gpus} {cpus}')
")"
  read -r HAS_CUDA GPU_COUNT CPU_COUNT <<< "$detected"
}

compute_cpu_jobs() {
  local requested="${1:-0}"
  local cap=$((CPU_COUNT / 2))
  if [[ "$cap" -lt 1 ]]; then
    cap=1
  fi

  if [[ "$requested" -gt 0 ]]; then
    CPU_JOBS="$requested"
  elif [[ "$HAS_CUDA" -eq 1 ]]; then
    # GPU hosts: use CPUs while GPUs run CB jobs; cap per-mixin rank cost (2 procs each).
    CPU_JOBS=$(( CPU_COUNT / 2 ))
    if [[ "$CPU_JOBS" -lt 4 ]]; then CPU_JOBS=4; fi
    if [[ "$CPU_JOBS" -gt 24 ]]; then CPU_JOBS=24; fi
  else
    CPU_JOBS="$cap"
    if [[ "$CPU_JOBS" -gt 8 ]]; then CPU_JOBS=8; fi
  fi

  if [[ "$CPU_JOBS" -gt "$cap" ]]; then
    c_warn "Capping CPU workers to ${cap} (each mixin test spawns 2 CPU processes; ${CPU_COUNT} CPUs detected)."
    CPU_JOBS="$cap"
  fi
  if [[ "$CPU_JOBS" -lt 1 ]]; then
    CPU_JOBS=1
  fi
}

compute_gpu_cb_jobs() {
  if [[ "$HAS_CUDA" -eq 0 || "$GPU_COUNT" -lt 2 ]]; then
    GPU_CB_JOBS=0
    return
  fi
  GPU_CB_JOBS=$((GPU_COUNT / 2))
  if [[ "$GPU_CB_JOBS" -lt 1 ]]; then
    GPU_CB_JOBS=1
  fi
}

gpu_pair_for_worker() {
  local worker_id="$1"
  local start=$((worker_id * 2))
  echo "${start},$((start + 1))"
}

test_tier() {
  local nodeid="$1"
  local short
  short="$(short_test_name "$nodeid")"
  if [[ "$short" == *megamoe* ]]; then
    echo "gpu_megamoe"
  elif [[ "$short" == test_continuous_batching_tp_* ]]; then
    echo "gpu_cb"
  elif [[ "$short" =~ ^test_(ep|tp)_(forward|backward|generation) ]]; then
    echo "cpu_mixin"
  else
    echo "cpu_unit"
  fi
}

tier_label() {
  case "$1" in
    cpu_unit) echo "CPU unit/plan (batched)" ;;
    cpu_mixin) echo "CPU mixin (gloo, 2 ranks each)" ;;
    gpu_cb) echo "GPU continuous batching (NCCL, 2 GPUs/worker)" ;;
    gpu_megamoe) echo "GPU megamoe (torchrun EP=8, serial)" ;;
    gpu) echo "GPU integration (serial)" ;;
    *) echo "$1" ;;
  esac
}

host_skip_reason() {
  local tier="$1"
  case "$tier" in
    gpu_cb)
      if [[ "$HAS_CUDA" -eq 0 ]]; then
        echo "no CUDA device"
      elif [[ "$GPU_COUNT" -lt 2 ]]; then
        echo "need 2+ GPUs (have ${GPU_COUNT})"
      fi
      ;;
    gpu_megamoe)
      if [[ "$HAS_CUDA" -eq 0 ]]; then
        echo "no CUDA device"
      elif [[ "$GPU_COUNT" -lt 8 ]]; then
        echo "need 8 GPUs (have ${GPU_COUNT})"
      fi
      ;;
  esac
}

dispatch_skip_reason() {
  local nodeid="$1"
  local short skip
  short="$(short_test_name "$nodeid")"
  for skip in "${DISPATCH_SKIP_TESTS[@]}"; do
    if [[ "$short" == "$skip" ]]; then
      echo "excluded by dispatch script (flaky cuda_graph+async CB tests)"
      return 0
    fi
  done
}

should_run_on_host() {
  local tier="$1"
  local reason
  reason="$(host_skip_reason "$tier")"
  [[ -z "$reason" ]]
}

partition_tests_for_host() {
  local -n _unit=$1
  local -n _mixin=$2
  local -n _gpu=$3
  local -n _skipped=$4
  local t tier reason

  _unit=()
  _mixin=()
  _gpu=()
  _skipped=()

  for t in "${COLLECTED_ALL_TESTS[@]}"; do
    reason="$(dispatch_skip_reason "$t")"
    if [[ -n "$reason" ]]; then
      _skipped+=("${t}|${reason}")
      continue
    fi
    tier="$(test_tier "$t")"
    if ! should_run_on_host "$tier"; then
      reason="$(host_skip_reason "$tier")"
      _skipped+=("${t}|${reason}")
      continue
    fi
    case "$tier" in
      cpu_unit) _unit+=("$t") ;;
      cpu_mixin) _mixin+=("$t") ;;
      gpu_cb|gpu_megamoe) _gpu+=("$t") ;;
    esac
  done
}

# Hide accelerators so CPU-only mixin tests pass on GPU machines.
pytest_cpu_isolated() {
  env CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" ROCR_VISIBLE_DEVICES="" \
    "${PYTEST[@]}" "$@"
}

pytest_gpu_pinned() {
  local devices="$1"
  shift
  env CUDA_VISIBLE_DEVICES="$devices" "${PYTEST[@]}" "$@"
}

split_gpu_tests() {
  local -n _all=$1
  local -n _cb=$2
  local -n _megamoe=$3
  local t
  _cb=()
  _megamoe=()
  for t in "${_all[@]}"; do
    if [[ "$(test_tier "$t")" == "gpu_megamoe" ]]; then
      _megamoe+=("$t")
    else
      _cb+=("$t")
    fi
  done
}

usage() {
  sed -n '2,30p' "$0"
  exit "${1:-1}"
}

collect_tests() {
  local collect_err output
  collect_err="$(mktemp)"
  output="$("${PYTEST[@]}" "$@" --collect-only -q 2>"$collect_err" | grep -E '^tests/[^:]+::' || true)"
  if [[ -z "$output" && -s "$collect_err" ]]; then
    c_err "pytest collection failed:"
    cat "$collect_err" >&2
    rm -f "$collect_err"
    return 1
  fi
  rm -f "$collect_err"
  echo "$output"
}

collect_tests_with_progress() {
  local label="$1"
  shift
  c_dim "  • collecting ${label}..." >&2
  collect_tests "$@"
}

collect_megamoe_tests() {
  if [[ "$MODEL" != "deepseek_v4" ]]; then
    c_warn "Warning: --megamoe only applies to deepseek_v4; skipping megamoe tests."
    return 0
  fi

  local test_file="tests/models/deepseek_v4/test_modeling_deepseek_v4.py"
  local class_name="DeepseekV4FlashIntegrationTest"
  local tests=(
    test_v4_flash_fp4_generation_megamoe_distributed
    test_v4_flash_fp4_forward_compile_fullgraph_megamoe_distributed
  )

  local test_name
  for test_name in "${tests[@]}"; do
    echo "${test_file}::${class_name}::${test_name}"
  done
}

append_unique_test() {
  local -n _dest=$1
  local line="$2"
  [[ -z "$line" ]] && return 0
  local existing
  for existing in "${_dest[@]:-}"; do
    [[ "$existing" == "$line" ]] && return 0
  done
  _dest+=("$line")
}

ensure_tests_collected() {
  if [[ "$COLLECTION_DONE" -eq 1 ]]; then
    return 0
  fi

  local -a tests=()
  local line
  local collect_start
  collect_start="$(date +%s)"

  c_info "Collecting tests..." >&2

  if [[ "$INCLUDE_MIXIN" -eq 1 ]]; then
    local test_file="tests/models/${MODEL}/test_modeling_${MODEL}.py"
    if [[ ! -f "$test_file" ]]; then
      c_err "Test file not found: $test_file"
      c_dim "Expected model folder to match model_type, e.g. qwen2 -> tests/models/qwen2/..."
      exit 1
    fi
    while IFS= read -r line; do append_unique_test tests "$line"; done < <(
      collect_tests_with_progress "mixin tests" "$test_file" -m is_tensor_parallel_test
    )
  fi

  if [[ "$INCLUDE_CB" -eq 1 ]]; then
    while IFS= read -r line; do append_unique_test tests "$line"; done < <(
      collect_tests_with_progress "unit + continuous batching tests" \
        tests/tensor_parallel/test_tensor_parallel.py \
        tests/generation/test_continuous_batching.py::ContinuousBatchingTensorParallelTest
    )
  else
    while IFS= read -r line; do append_unique_test tests "$line"; done < <(
      collect_tests_with_progress "tensor_parallel unit tests" \
        tests/tensor_parallel/test_tensor_parallel.py -m is_tensor_parallel_test
    )
  fi

  if [[ "$INCLUDE_MEGAMOE" -eq 1 ]]; then
    c_dim "  • megamoe tests (static list)" >&2
    while IFS= read -r line; do append_unique_test tests "$line"; done < <(collect_megamoe_tests)
  fi

  line="$(tp_plan_nodeid)"
  if [[ -n "$line" ]]; then
    c_dim "  • tp_plan validation (static nodeid)" >&2
    append_unique_test tests "$line"
  fi

  if [[ ${#tests[@]} -eq 0 ]]; then
    c_err "No tests collected."
    exit 1
  fi

  COLLECTED_ALL_TESTS=("${tests[@]}")
  COLLECTION_DONE=1
  c_ok "  Collected ${#COLLECTED_ALL_TESTS[@]} tests in $(format_duration "$(( $(date +%s) - collect_start ))")" >&2
  echo >&2
}

print_recap() {
  local end_epoch="$1"
  local elapsed=$((end_epoch - START_EPOCH))
  local passed=0 failed=0 skipped=0
  local -a failed_tests=() passed_tests=() skipped_tests=()
  local short entry reason

  write_summary "$elapsed"

  [[ -f "$TMPDIR/passed.txt" ]] && passed=$(wc -l < "$TMPDIR/passed.txt")
  [[ -f "$TMPDIR/failed.txt" ]] && mapfile -t failed_tests < "$TMPDIR/failed.txt"
  [[ -f "$TMPDIR/skipped.txt" ]] && mapfile -t skipped_tests < "$TMPDIR/skipped.txt"
  failed=${#failed_tests[@]}
  skipped=${#skipped_tests[@]}

  local total=$((passed + failed + skipped))

  echo
  c_header "══════════════════════════════════════════════════════════════"
  c_header "  RECAP"
  c_header "══════════════════════════════════════════════════════════════"
  printf '  %-14s %s\n' "Model:" "${MODEL:-(none)}"
  printf '  %-14s mixin=%s  cb=%s  megamoe=%s\n' "Suites:" "$INCLUDE_MIXIN" "$INCLUDE_CB" "$INCLUDE_MEGAMOE"
  printf '  %-14s %s\n' "Duration:" "$(format_duration "$elapsed")"
  echo
  printf '  %-14s %b%d%b\n' "Passed:" "$C_GREEN" "$passed" "$C_RESET"
  if [[ "$failed" -gt 0 ]]; then
    printf '  %-14s %b%d%b\n' "Failed:" "$C_RED" "$failed" "$C_RESET"
  else
    printf '  %-14s %b%d%b\n' "Failed:" "$C_DIM" "$failed" "$C_RESET"
  fi
  if [[ "$skipped" -gt 0 ]]; then
    printf '  %-14s %b%d%b\n' "Skipped:" "$C_YELLOW" "$skipped" "$C_RESET"
  fi
  echo

  if [[ "$failed" -gt 0 ]]; then
    c_err "  Failed tests (see logs/<name>.log):"
    for short in "${failed_tests[@]}"; do
      printf '    %b✗%b %s\n' "$C_RED" "$C_RESET" "$short"
      print_failure_excerpt_to_console "$TMPDIR/logs/${short}.log"
    done
    echo
  fi

  if [[ "$skipped" -gt 0 ]]; then
    c_warn "  Skipped tests:"
    for entry in "${skipped_tests[@]}"; do
      short="${entry%%|*}"
      reason="${entry#*|}"
      printf '    %b-%b %s\n' "$C_YELLOW" "$C_RESET" "$short"
      [[ -n "$reason" ]] && c_dim "      ${reason}"
    done
    echo
  fi

  c_dim "  Summary: $TMPDIR/summary.txt"
  c_dim "  Logs:    $TMPDIR/logs/"

  if [[ "$failed" -gt 0 ]]; then
    c_header "══════════════════════════════════════════════════════════════"
    c_err "  RESULT: FAILED ($failed failed, $passed passed, $skipped skipped)"
    c_header "══════════════════════════════════════════════════════════════"
    return 1
  fi

  c_header "══════════════════════════════════════════════════════════════"
  if [[ "$skipped" -gt 0 ]]; then
    c_ok "  RESULT: PASSED ($passed passed, $skipped skipped)"
  else
    c_ok "  RESULT: ALL PASSED ($passed/$total)"
  fi
  c_header "══════════════════════════════════════════════════════════════"
  return 0
}

assign_round_robin() {
  local -n _tests=$1
  local -n _buckets=$2
  local workers="$3"
  local i b

  _buckets=()
  for ((b = 0; b < workers; b++)); do
    _buckets+=("")
  done

  for i in "${!_tests[@]}"; do
    b=$((i % workers))
    if [[ -z "${_buckets[$b]}" ]]; then
      _buckets[$b]="${_tests[$i]}"
    else
      _buckets[$b]+=$'\n'"${_tests[$i]}"
    fi
  done
}

print_phase_plan() {
  local phase="$1"
  local workers="$2"
  local batched="$3"
  shift 3
  local -a tests=("$@")
  local -a buckets=()

  [[ ${#tests[@]} -eq 0 ]] && return 0

  c_worker "── Phase: $(tier_label "$phase") ──"
  printf '  %-14s %s\n' "Tests:" "${#tests[@]}"
  printf '  %-14s %s\n' "Parallelism:" "${workers} worker(s)$([[ "$batched" -eq 1 ]] && echo ', batched pytest' || echo ', one pytest/test')"
  assign_round_robin tests buckets "$workers"
  local b local_idx line
  for ((b = 0; b < workers; b++)); do
    [[ -z "${buckets[$b]}" ]] && continue
    if [[ "$phase" == "gpu_cb" ]]; then
      c_worker "  Worker $b (CUDA_VISIBLE_DEVICES=$(gpu_pair_for_worker "$b"))"
    else
      c_worker "  Worker $b"
    fi
    local_idx=0
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      ((local_idx++)) || true
      if [[ "$batched" -eq 1 && "$local_idx" -eq 1 && $(echo "${buckets[$b]}" | wc -l) -gt 1 ]]; then
        printf '    %b%2d.%b %s (+ %d more in one pytest)\n' "$C_DIM" "$local_idx" "$C_RESET" \
          "$(short_test_name "$line")" "$(( $(echo "${buckets[$b]}" | wc -l) - 1 ))"
        break
      fi
      printf '    %b%2d.%b %s\n' "$C_DIM" "$local_idx" "$C_RESET" "$(short_test_name "$line")"
    done <<< "${buckets[$b]}"
  done
  echo
}

run_pytest_nodeids() {
  local worker_id="$1"
  local phase="$2"
  local log="$3"
  local cpu_isolate="$4"
  shift 4
  local -a nodeids=("$@")
  local failed=0 nodeid short worker_label
  local -a pytest_cmd=()
  local batch_rc=0

  worker_label="[cpu $worker_id]"

  if [[ ${#nodeids[@]} -eq 0 ]]; then
    return 0
  fi

  if [[ "$cpu_isolate" -eq 1 ]]; then
    pytest_cmd=(pytest_cpu_isolated)
  else
    pytest_cmd=("${PYTEST[@]}")
  fi

  if [[ ${#nodeids[@]} -eq 1 ]]; then
    short="$(short_test_name "${nodeids[0]}")"
    c_worker "${worker_label} RUN  $short"
    run_and_record_pytest "$worker_label" "$log" "$cpu_isolate" "${nodeids[0]}"
    return $?
  fi

  c_worker "${worker_label} RUN  batch/${phase} (${#nodeids[@]} tests)"
  {
    echo ""
    echo "===== RUN batch/$(date -Iseconds) ${phase} (${#nodeids[@]} tests) ====="
  } >> "$log"

  if "${pytest_cmd[@]}" "${nodeids[@]}" -v "${PYTEST_REPORT[@]}" "${PYTEST_EXTRA[@]}" >>"$log" 2>&1; then
    batch_rc=0
  else
    batch_rc=1
  fi

  if [[ "$batch_rc" -eq 0 ]]; then
    if classify_and_record_pytest_results "$log" "${nodeids[@]}"; then
      c_ok "${worker_label} PASS batch/${phase} (${#nodeids[@]} tests)"
      return 0
    fi
    c_err "${worker_label} FAIL batch/${phase} (see $(log_basename "$log"))"
    return 1
  fi

  c_warn "${worker_label} batch/${phase} failed; re-running individually..."
  for nodeid in "${nodeids[@]}"; do
    short="$(short_test_name "$nodeid")"
    c_worker "${worker_label} RUN  $short"
    run_and_record_pytest "$worker_label" "$log" "$cpu_isolate" "$nodeid" || failed=1
  done
  return "$failed"
}

run_cpu_phase() {
  local phase="$1"
  local batched="$2"
  local cpu_isolate="$3"
  shift 3
  local -a tests=("$@")
  local -a buckets=()
  local -a pids=()
  local b failed=0 pid

  [[ ${#tests[@]} -eq 0 ]] && return 0

  assign_round_robin tests buckets "$CPU_JOBS"

  run_cpu_worker() {
    local worker_id="$1"
    local log="$TMPDIR/workers/worker_cpu_${worker_id}_${phase}.log"
    local -a bucket_tests=()
    local line
    : > "$log"
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      bucket_tests+=("$line")
    done <<< "${buckets[$worker_id]}"
    if [[ ${#bucket_tests[@]} -eq 0 ]]; then
      return 0
    fi
    if [[ "$batched" -eq 1 ]]; then
      run_pytest_nodeids "$worker_id" "$phase" "$log" "$cpu_isolate" "${bucket_tests[@]}"
    else
      local nodeid
      local worker_failed=0
      for nodeid in "${bucket_tests[@]}"; do
        run_pytest_nodeids "$worker_id" "$phase" "$log" "$cpu_isolate" "$nodeid" || worker_failed=1
      done
      return "$worker_failed"
    fi
  }

  for ((b = 0; b < CPU_JOBS; b++)); do
    [[ -z "${buckets[$b]}" ]] && continue
    run_cpu_worker "$b" &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  return "$failed"
}

run_gpu_cb_phase() {
  local -a tests=("$@")
  local -a buckets=()
  local -a pids=()
  local failed=0 b devices

  [[ ${#tests[@]} -eq 0 ]] && return 0
  if [[ "$GPU_CB_JOBS" -lt 1 ]]; then
    GPU_CB_JOBS=1
  fi

  assign_round_robin tests buckets "$GPU_CB_JOBS"

  run_gpu_cb_worker() {
    local worker_id="$1"
    local devices="$2"
    local log="$TMPDIR/workers/worker_gpu_cb_${worker_id}.log"
    local -a bucket_tests=()
    local line nodeid short worker_failed=0
    local worker_label="[gpu cb $worker_id dev=$devices]"

    : > "$log"
    while IFS= read -r line; do
      [[ -z "$line" ]] && continue
      bucket_tests+=("$line")
    done <<< "${buckets[$worker_id]}"

    for nodeid in "${bucket_tests[@]}"; do
      short="$(short_test_name "$nodeid")"
      c_worker "${worker_label} RUN  $short"
      run_and_record_pytest "$worker_label" "$log" 0 "$nodeid" "$devices" || worker_failed=1
    done
    return "$worker_failed"
  }

  for ((b = 0; b < GPU_CB_JOBS; b++)); do
    [[ -z "${buckets[$b]}" ]] && continue
    devices="$(gpu_pair_for_worker "$b")"
    run_gpu_cb_worker "$b" "$devices" &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  return "$failed"
}

run_gpu_megamoe_phase() {
  local -a tests=("$@")
  local log="$TMPDIR/workers/worker_gpu_megamoe.log"
  local failed=0 nodeid short worker_label="[gpu megamoe serial]"

  [[ ${#tests[@]} -eq 0 ]] && return 0

  : > "$log"
  for nodeid in "${tests[@]}"; do
    short="$(short_test_name "$nodeid")"
    c_worker "${worker_label} RUN  $short"
    run_and_record_pytest "$worker_label" "$log" 0 "$nodeid" || failed=1
  done
  return "$failed"
}

run_mixin_and_gpu_cb_parallel() {
  local cpu_isolate="$1"
  local -n _mixin=$2
  local -n _gpu_cb=$3
  local mixin_fail=0 gpu_fail=0 mixin_pid

  c_info "Overlapping CPU mixin (${CPU_JOBS} workers) with GPU CB (${GPU_CB_JOBS} workers, 2 GPUs each)..."
  run_cpu_phase "cpu_mixin" 0 "$cpu_isolate" "${_mixin[@]}" &
  mixin_pid=$!

  if [[ ${#_gpu_cb[@]} -gt 0 ]]; then
    if ! run_gpu_cb_phase "${_gpu_cb[@]}"; then
      gpu_fail=1
    fi
  fi

  if ! wait "$mixin_pid"; then
    mixin_fail=1
  fi

  if [[ "$mixin_fail" -ne 0 || "$gpu_fail" -ne 0 ]]; then
    return 1
  fi
  return 0
}

run_dispatch() {
  local -a tests_unit=() tests_mixin=() tests_gpu=() tests_gpu_cb=() tests_gpu_megamoe=() tests_skipped=()
  local total_scheduled=0
  local cpu_isolate=0

  if [[ "$HAS_CUDA" -eq 1 ]]; then
    cpu_isolate=1
  fi

  mkdir -p "$TMPDIR/logs" "$TMPDIR/workers"
  : > "$TMPDIR/passed.txt"
  : > "$TMPDIR/failed.txt"
  : > "$TMPDIR/skipped.txt"

  ensure_tests_collected
  partition_tests_for_host tests_unit tests_mixin tests_gpu tests_skipped
  split_gpu_tests tests_gpu tests_gpu_cb tests_gpu_megamoe

  total_scheduled=$((${#tests_unit[@]} + ${#tests_mixin[@]} + ${#tests_gpu[@]}))
  if [[ ${#tests_skipped[@]} -gt 0 ]]; then
    local entry nodeid reason
    for entry in "${tests_skipped[@]}"; do
      nodeid="${entry%%|*}"
      reason="${entry#*|}"
      echo "$(short_test_name "$nodeid")|host: ${reason}" >> "$TMPDIR/skipped.txt"
    done
  fi

  if [[ "$total_scheduled" -eq 0 ]]; then
    c_err "No tests scheduled on this host."
    if [[ ${#tests_skipped[@]} -gt 0 ]]; then
      c_warn "All ${#tests_skipped[@]} collected tests were host-skipped."
    fi
    return 1
  fi

  c_header "══════════════════════════════════════════════════════════════"
  c_header "  TP TEST DISPATCH"
  c_header "══════════════════════════════════════════════════════════════"
  printf '  %-14s %s\n' "Env:" "${VIRTUAL_ENV:-${ENV_TP:-env_tp}}"
  printf '  %-14s %s\n' "Model:" "${MODEL:-(none)}"
  printf '  %-14s mixin=%s  cb=%s  megamoe=%s\n' "Suites:" "$INCLUDE_MIXIN" "$INCLUDE_CB" "$INCLUDE_MEGAMOE"
  printf '  %-14s CUDA=%s  GPUs=%s  CPUs=%s\n' "Host:" "$([[ "$HAS_CUDA" -eq 1 ]] && echo yes || echo no)" "$GPU_COUNT" "$CPU_COUNT"
  printf '  %-14s %s\n' "CPU workers:" "$CPU_JOBS"
  if [[ ${#tests_gpu_cb[@]} -gt 0 ]]; then
    printf '  %-14s %s\n' "GPU CB workers:" "$GPU_CB_JOBS (2 GPUs each)"
  fi
  if [[ "$cpu_isolate" -eq 1 ]]; then
    printf '  %-14s %s\n' "CPU tests:" "CUDA hidden per subprocess"
  fi
  printf '  %-14s %b%d scheduled%b' "Tests:" "$C_BOLD" "$total_scheduled" "$C_RESET"
  if [[ ${#tests_skipped[@]} -gt 0 ]]; then
    printf ' (%b%d host-skipped%b)' "$C_YELLOW" "${#tests_skipped[@]}" "$C_RESET"
  fi
  echo
  echo

  print_phase_plan "cpu_unit" "$CPU_JOBS" 1 "${tests_unit[@]}"
  print_phase_plan "cpu_mixin" "$CPU_JOBS" 0 "${tests_mixin[@]}"
  if [[ ${#tests_gpu_cb[@]} -gt 0 ]]; then
    print_phase_plan "gpu_cb" "$GPU_CB_JOBS" 0 "${tests_gpu_cb[@]}"
  fi
  if [[ ${#tests_gpu_megamoe[@]} -gt 0 ]]; then
    print_phase_plan "gpu_megamoe" 1 0 "${tests_gpu_megamoe[@]}"
  fi
  if [[ ${#tests_mixin[@]} -gt 0 && ${#tests_gpu_cb[@]} -gt 0 ]]; then
    c_dim "  Note: CPU mixin and GPU CB phases overlap during execution."
  fi

  if [[ ${#tests_skipped[@]} -gt 0 ]]; then
    c_warn "── Host-skipped (${#tests_skipped[@]}) ──"
    local entry nodeid reason
    for entry in "${tests_skipped[@]}"; do
      nodeid="${entry%%|*}"
      reason="${entry#*|}"
      printf '  %b-%b %s — %s\n' "$C_YELLOW" "$C_RESET" "$(short_test_name "$nodeid")" "$reason"
    done
    echo
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    c_warn "Dry run only; not executing tests."
    return 0
  fi

  START_EPOCH=$(date +%s)
  local FAIL=0

  c_header "── Running ──"
  echo

  local phase_idx=0 phase_total=0
  [[ ${#tests_unit[@]} -gt 0 ]] && ((phase_total++)) || true
  if [[ ${#tests_mixin[@]} -gt 0 && ${#tests_gpu_cb[@]} -gt 0 ]]; then
    ((phase_total++)) || true
  else
    [[ ${#tests_mixin[@]} -gt 0 ]] && ((phase_total++)) || true
    [[ ${#tests_gpu_cb[@]} -gt 0 ]] && ((phase_total++)) || true
  fi
  [[ ${#tests_gpu_megamoe[@]} -gt 0 ]] && ((phase_total++)) || true

  if [[ ${#tests_unit[@]} -gt 0 ]]; then
    ((phase_idx++)) || true
    c_info "Phase ${phase_idx}/${phase_total}: CPU unit/plan (batched, ${CPU_JOBS} workers)"
    if ! run_cpu_phase "cpu_unit" 1 "$cpu_isolate" "${tests_unit[@]}"; then
      FAIL=1
    fi
    echo
  fi

  if [[ ${#tests_mixin[@]} -gt 0 && ${#tests_gpu_cb[@]} -gt 0 ]]; then
    ((phase_idx++)) || true
    c_info "Phase ${phase_idx}/${phase_total}: CPU mixin + GPU CB (overlapped)"
    if ! run_mixin_and_gpu_cb_parallel "$cpu_isolate" tests_mixin tests_gpu_cb; then
      FAIL=1
    fi
    echo
  else
    if [[ ${#tests_mixin[@]} -gt 0 ]]; then
      ((phase_idx++)) || true
      c_info "Phase ${phase_idx}/${phase_total}: CPU mixin (gloo, ${CPU_JOBS} workers)"
      if ! run_cpu_phase "cpu_mixin" 0 "$cpu_isolate" "${tests_mixin[@]}"; then
        FAIL=1
      fi
      echo
    fi

    if [[ ${#tests_gpu_cb[@]} -gt 0 ]]; then
      ((phase_idx++)) || true
      c_info "Phase ${phase_idx}/${phase_total}: GPU continuous batching (${GPU_CB_JOBS} workers, 2 GPUs each)"
      if ! run_gpu_cb_phase "${tests_gpu_cb[@]}"; then
        FAIL=1
      fi
      echo
    fi
  fi

  if [[ ${#tests_gpu_megamoe[@]} -gt 0 ]]; then
    ((phase_idx++)) || true
    c_info "Phase ${phase_idx}/${phase_total}: GPU megamoe (serial, EP=8)"
    if ! run_gpu_megamoe_phase "${tests_gpu_megamoe[@]}"; then
      FAIL=1
    fi
    echo
  fi

  if ! print_recap "$(date +%s)"; then
    FAIL=1
  fi

  return "$FAIL"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    -j)
      [[ $# -ge 2 ]] || usage
      JOBS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --continuous-batching|--cb)
      INCLUDE_CB=1
      shift
      ;;
    --megamoe)
      INCLUDE_MEGAMOE=1
      shift
      ;;
    --mixin-only)
      INCLUDE_MIXIN=1
      shift
      ;;
    --all)
      INCLUDE_MIXIN=1
      INCLUDE_CB=1
      INCLUDE_MEGAMOE=1
      shift
      ;;
    --)
      shift
      PYTEST_EXTRA=("$@")
      break
      ;;
    -*)
      c_err "Unknown option: $1"
      usage
      ;;
    *)
      if [[ -z "$MODEL" ]]; then
        MODEL="$1"
        shift
      else
        c_err "Unexpected argument: $1"
        usage
      fi
      ;;
  esac
done

[[ "$JOBS" =~ ^[0-9]+$ ]] || { c_err "Invalid -j value: $JOBS"; exit 1; }

if [[ -n "$MODEL" && "$INCLUDE_MIXIN" -eq 0 && "$INCLUDE_CB" -eq 0 && "$INCLUDE_MEGAMOE" -eq 0 ]]; then
  INCLUDE_MIXIN=1
fi

if [[ -z "$MODEL" && "$INCLUDE_MIXIN" -eq 1 ]]; then
  c_err "Model type is required for mixin tests."
  usage
fi

if [[ "$INCLUDE_MIXIN" -eq 0 && "$INCLUDE_CB" -eq 0 && "$INCLUDE_MEGAMOE" -eq 0 ]]; then
  c_err "No test suite selected. Pass a model and/or --continuous-batching / --megamoe / --all."
  usage
fi

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

activate_env_tp "$REPO_ROOT"
export RUN_TENSOR_PARALLEL_TESTS=1

detect_host_capabilities
compute_cpu_jobs "$JOBS"
compute_gpu_cb_jobs

c_info "Host: CUDA=$([[ "$HAS_CUDA" -eq 1 ]] && echo yes || echo no)  GPUs=$GPU_COUNT  CPUs=$CPU_COUNT  CPU_workers=$CPU_JOBS  GPU_CB_workers=$GPU_CB_JOBS"
if [[ "$HAS_CUDA" -eq 1 ]]; then
  c_dim "  CPU suites run with CUDA hidden; CB jobs pin CUDA_VISIBLE_DEVICES to GPU pairs; mixin+CB overlap when both scheduled."
fi
echo

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
TMPDIR="${TP_DISPATCH_LOG_DIR:-${REPO_ROOT}/tp_dispatch_logs}/${MODEL:-cb}_${RUN_STAMP}_$$"
mkdir -p "$TMPDIR"
c_info "Dispatch dir: $TMPDIR"

if ! run_dispatch; then
  exit 1
fi
