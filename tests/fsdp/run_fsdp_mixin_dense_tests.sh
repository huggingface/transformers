#!/bin/bash

# Script to run all FSDP mixin tests for dense models in parallel.
# Work in tandem with a special test_fsdp_mixin.py that batches all 11 distributed tests in a single mp.spawn. (will not be committed)
# Uses concurrency-limited dispatch: multiple models share GPU pairs since test models are tiny (~7 MiB).
# Each model runs test_fsdp2_all which batches all 11 distributed tests in a single mp.spawn.

# Usage: ./run_fsdp_mixin_dense_tests.sh [/path/to/results]
#        ./run_fsdp_mixin_dense_tests.sh --model <model_name> [/path/to/results]
#        ./run_fsdp_mixin_dense_tests.sh --test <test_method> [/path/to/results]
#        ./run_fsdp_mixin_dense_tests.sh --model <model_name> --test <test_method> [/path/to/results]
#        ./run_fsdp_mixin_dense_tests.sh --report /path/to/results
#        ./run_fsdp_mixin_dense_tests.sh --debug --model <model_name>
#        ./run_fsdp_mixin_dense_tests.sh --rerun-failed /path/to/results

# Rerunning 11 failed model(s):
#   apertus
#   blt
#   exaone4
#   gemma2
#   gemma3
#   gemma3n
#   glm4
#   modernbert_decoder
#   olmo3
#   phi
#   qwen3_5
# Using 8 GPUs (4 GPU pairs, 2 GPUs each)
# Min free memory per GPU to launch a new test: 1000 MiB
# Results directory: /fsx/ferdinandmom/ferdinand-hf/transformers_pr/work/v5-distributed-training-ci/results_fsdp_dense
# ==========================================
#   FSDP Mixin Dense Tests
#   (11 models, 12 tests each)
#   (Memory-gated parallel execution across 4 GPU pairs)
# ==========================================

# [GPUs 0,1] Starting: apertus
# [GPUs 2,3] Starting: blt
# [GPUs 4,5] Starting: exaone4
# [GPUs 6,7] Starting: gemma2
# [GPUs 0,1] Starting: gemma3
# [GPUs 2,3] Starting: gemma3n
# [GPUs 4,5] Starting: glm4
# [GPUs 6,7] Starting: modernbert_decoder
# [GPUs 0,1] Starting: olmo3
# [GPUs 2,3] Starting: phi

# Waiting for all tests to complete...
# [GPUs 4,5] Starting: qwen3_5


SCRIPT_START=$(date +%s)

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREY='\033[0;90m'
DIM='\033[0;90m'
NC='\033[0m'

GPUS_PER_TEST=2

# Batched test method that runs all 11 distributed FSDP tests in a single mp.spawn.
# Individual methods are still available for debugging via --test:
#   test_get_transformer_block_classes, test_fsdp2_sharding_structure_untied,
#   test_fsdp2_sharding_structure_tied, test_fsdp2_auto_plan_vs_ddp_float32_untied,
#   test_fsdp2_auto_plan_vs_ddp_bfloat16_untied, test_fsdp2_auto_plan_vs_ddp_float32_tied,
#   test_fsdp2_auto_plan_vs_ddp_bfloat16_tied, test_fsdp2_manual_plan_vs_ddp_float32_untied,
#   test_fsdp2_manual_plan_vs_ddp_bfloat16_untied, test_fsdp2_manual_plan_vs_ddp_float32_tied,
#   test_fsdp2_manual_plan_vs_ddp_bfloat16_tied, test_fsdp2_save_load
TEST_METHODS=(
    "test_fsdp2_all"
)

# Dense models that inherit from CausalLMModelTest
TEST_ENTRIES=(
    "tests/models/apertus/test_modeling_apertus.py::ApertusModelTest"
    "tests/models/arcee/test_modeling_arcee.py::ArceeModelTest"
    "tests/models/bloom/test_modeling_bloom.py::BloomModelTest"
    "tests/models/blt/test_modeling_blt.py::BltModelTest"
    "tests/models/cwm/test_modeling_cwm.py::CwmModelTest"
    "tests/models/ernie4_5/test_modeling_ernie4_5.py::Ernie4_5ModelTest"
    "tests/models/exaone4/test_modeling_exaone4.py::Exaone4ModelTest"
    "tests/models/falcon/test_modeling_falcon.py::FalconModelTest"
    "tests/models/gemma/test_modeling_gemma.py::GemmaModelTest"
    "tests/models/gemma2/test_modeling_gemma2.py::Gemma2ModelTest"
    "tests/models/gemma3/test_modeling_gemma3.py::Gemma3TextModelTest"
    "tests/models/gemma3n/test_modeling_gemma3n.py::Gemma3nTextModelTest"
    "tests/models/glm/test_modeling_glm.py::GlmModelTest"
    "tests/models/glm4/test_modeling_glm4.py::Glm4ModelTest"
    "tests/models/gpt2/test_modeling_gpt2.py::GPT2ModelTest"
    "tests/models/helium/test_modeling_helium.py::HeliumModelTest"
    "tests/models/hunyuan_v1_dense/test_modeling_hunyuan_v1_dense.py::HunYuanDenseV1ModelTest"
    "tests/models/jais2/test_modeling_jais2.py::Jais2ModelTest"
    "tests/models/lfm2/test_modeling_lfm2.py::Lfm2ModelTest"
    "tests/models/llama/test_modeling_llama.py::LlamaModelTest"
    "tests/models/ministral/test_modeling_ministral.py::MinistralModelTest"
    "tests/models/ministral3/test_modeling_ministral3.py::Ministral3ModelTest"
    "tests/models/mistral/test_modeling_mistral.py::MistralModelTest"
    "tests/models/modernbert_decoder/test_modeling_modernbert_decoder.py::ModernBertDecoderModelTest"
    "tests/models/nanochat/test_modeling_nanochat.py::NanoChatModelTest"
    "tests/models/nemotron/test_modeling_nemotron.py::NemotronModelTest"
    "tests/models/olmo3/test_modeling_olmo3.py::Olmo3ModelTest"
    "tests/models/persimmon/test_modeling_persimmon.py::PersimmonModelTest"
    "tests/models/phi/test_modeling_phi.py::PhiModelTest"
    "tests/models/phi3/test_modeling_phi3.py::Phi3ModelTest"
    "tests/models/qwen2/test_modeling_qwen2.py::Qwen2ModelTest"
    "tests/models/qwen3/test_modeling_qwen3.py::Qwen3ModelTest"
    "tests/models/qwen3_5/test_modeling_qwen3_5.py::Qwen3_5TextModelTest"
    "tests/models/recurrent_gemma/test_modeling_recurrent_gemma.py::RecurrentGemmaModelTest"
    "tests/models/seed_oss/test_modeling_seed_oss.py::SeedOssModelTest"
    "tests/models/smollm3/test_modeling_smollm3.py::SmolLM3ModelTest"
    "tests/models/stablelm/test_modeling_stablelm.py::StableLmModelTest"
    "tests/models/starcoder2/test_modeling_starcoder2.py::Starcoder2ModelTest"
    "tests/models/vaultgemma/test_modeling_vaultgemma.py::VaultGemmaModelTest"
    "tests/models/youtu/test_modeling_youtu.py::YoutuModelTest"
)

# ── Helpers ─────────────────────────────────────────────────────────────────
extract_model_name() {
    echo "$1" | sed 's|tests/models/\([^/]*\)/.*|\1|'
}

# ── Report ──────────────────────────────────────────────────────────────────
print_report() {
    local results_dir=$1
    results_dir=$(cd "$results_dir" && pwd)

    if [ ! -d "$results_dir" ]; then
        echo "Error: Results directory '$results_dir' does not exist"
        exit 1
    fi

    echo "=========================================="
    echo "  FSDP Mixin Dense Test Report"
    echo "  Results directory: $results_dir"
    echo "=========================================="
    echo ""

    local success_count=0
    local fail_count=0
    local skip_count=0
    local missing_count=0

    for entry in "${TEST_ENTRIES[@]}"; do
        local model_name
        model_name=$(extract_model_name "$entry")
        local result_file="$results_dir/${model_name}.result"
        if [ -f "$result_file" ]; then
            local result=$(cat "$result_file")
            if [[ "$result" == "SUCCESS" ]]; then
                echo -e "${GREEN}✓ ${model_name}: ${result}${NC}"
                ((success_count++))
            elif [[ "$result" == "SKIPPED" ]]; then
                echo -e "${GREY}○ ${model_name}: ${result}${NC}"
                ((skip_count++))
            else
                echo -e "${RED}✗ ${model_name}: ${result}${NC}"
                if [ -f "$results_dir/${model_name}.log" ]; then
                    echo -e "${DIM}  Error snippet:"
                    tail -n 5 "$results_dir/${model_name}.log" | while read -r line; do echo -e "    ${DIM}${line}${NC}"; done
                fi
                ((fail_count++))
            fi
        else
            echo -e "${YELLOW}? ${model_name}: NOT RUN${NC}"
            ((missing_count++))
        fi
    done

    echo ""
    echo "-------------------------------------------"
    echo -e "Total: ${GREEN}${success_count} passed${NC}, ${GREY}${skip_count} skipped${NC}, ${RED}${fail_count} failed${NC}, ${YELLOW}${missing_count} not run${NC}"
    echo "=========================================="

    if [ $fail_count -gt 0 ]; then
        echo ""
        echo "Failed test logs (full paths):"
        for entry in "${TEST_ENTRIES[@]}"; do
            local model_name
            model_name=$(extract_model_name "$entry")
            result_file="$results_dir/${model_name}.result"
            if [ -f "$result_file" ] && [ "$(cat "$result_file")" != "SUCCESS" ] && [ "$(cat "$result_file")" != "SKIPPED" ]; then
                echo "  $results_dir/${model_name}.log"
            fi
        done
        exit 1
    fi
}

# ── Argument parsing ────────────────────────────────────────────────────────
if [ "$1" == "--report" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 --report /path/to/results"
        exit 1
    fi
    print_report "$2"
    exit 0
fi

DEBUG_MODE=""
if [ "$1" == "--debug" ]; then
    DEBUG_MODE=1
    shift
fi

if [ "$1" == "--model" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 --model <model_name> [/path/to/results]"
        echo "Available models:"
        for entry in "${TEST_ENTRIES[@]}"; do
            printf '  %s\n' "$(extract_model_name "$entry")"
        done
        exit 1
    fi
    SINGLE_MODEL="$2"
    shift 2
    MATCHED_ENTRY=""
    for entry in "${TEST_ENTRIES[@]}"; do
        if [ "$(extract_model_name "$entry")" == "$SINGLE_MODEL" ]; then
            MATCHED_ENTRY="$entry"
            break
        fi
    done
    if [ -z "$MATCHED_ENTRY" ]; then
        echo "Error: Unknown model '$SINGLE_MODEL'"
        echo "Available models:"
        for entry in "${TEST_ENTRIES[@]}"; do
            printf '  %s\n' "$(extract_model_name "$entry")"
        done
        exit 1
    fi
    TEST_ENTRIES=("$MATCHED_ENTRY")
fi

if [ "$1" == "--test" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 --test <test_method> [/path/to/results]"
        echo "Available test methods:"
        for method in "${TEST_METHODS[@]}"; do
            printf '  %s\n' "$method"
        done
        exit 1
    fi
    SINGLE_TEST="$2"
    shift 2
    MATCHED_METHOD=""
    for method in "${TEST_METHODS[@]}"; do
        if [ "$method" == "$SINGLE_TEST" ]; then
            MATCHED_METHOD="$method"
            break
        fi
    done
    if [ -z "$MATCHED_METHOD" ]; then
        echo "Error: Unknown test method '$SINGLE_TEST'"
        echo "Available test methods:"
        for method in "${TEST_METHODS[@]}"; do
            printf '  %s\n' "$method"
        done
        exit 1
    fi
    TEST_METHODS=("$MATCHED_METHOD")
fi

RERUN_FAILED=""
if [ "$1" == "--rerun-failed" ]; then
    if [ -z "$2" ]; then
        echo "Usage: $0 --rerun-failed /path/to/results"
        exit 1
    fi
    RERUN_FAILED=1
    RESULTS_DIR="$2"
    shift 2
    if [ ! -d "$RESULTS_DIR" ]; then
        echo "Error: Results directory '$RESULTS_DIR' does not exist"
        exit 1
    fi
    RESULTS_DIR=$(cd "$RESULTS_DIR" && pwd)
    FAILED_ENTRIES=()
    for entry in "${TEST_ENTRIES[@]}"; do
        model_name=$(extract_model_name "$entry")
        result_file="$RESULTS_DIR/${model_name}.result"
        if [ -f "$result_file" ]; then
            result=$(cat "$result_file")
            if [[ "$result" != "SUCCESS" ]] && [[ "$result" != "SKIPPED" ]]; then
                FAILED_ENTRIES+=("$entry")
            fi
        fi
    done
    if [ ${#FAILED_ENTRIES[@]} -eq 0 ]; then
        echo "No failed tests to rerun in $RESULTS_DIR"
        exit 0
    fi
    TEST_ENTRIES=("${FAILED_ENTRIES[@]}")
    echo "Rerunning ${#TEST_ENTRIES[@]} failed model(s):"
    for entry in "${TEST_ENTRIES[@]}"; do
        echo "  $(extract_model_name "$entry")"
    done
fi

# ── GPU detection ───────────────────────────────────────────────────────────
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$AVAILABLE_GPUS" -lt "$GPUS_PER_TEST" ]; then
    echo "Need at least $GPUS_PER_TEST GPUs, but only $AVAILABLE_GPUS detected!"
    exit 1
fi
NUM_GPU_PAIRS=$((AVAILABLE_GPUS / GPUS_PER_TEST))
MAX_CONCURRENT=${MAX_CONCURRENT:-$((NUM_GPU_PAIRS * 2))}
echo "Using $AVAILABLE_GPUS GPUs ($NUM_GPU_PAIRS GPU pairs, max $MAX_CONCURRENT concurrent models)"

if [ -n "$DEBUG_MODE" ] && [ -z "$SINGLE_MODEL" ]; then
    echo "Error: --debug requires --model <model_name>"
    echo "Usage: $0 --debug --model <model_name>"
    exit 1
fi

# ── Results directory ───────────────────────────────────────────────────────
if [ -n "$RERUN_FAILED" ]; then
    mkdir -p "$RESULTS_DIR"
    CLEANUP_RESULTS=false
elif [ -n "$1" ]; then
    RESULTS_DIR="$1"
    mkdir -p "$RESULTS_DIR"
    CLEANUP_RESULTS=false
elif [ -n "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
    CLEANUP_RESULTS=false
else
    RESULTS_DIR=$(mktemp -d)
    CLEANUP_RESULTS=true
fi
RESULTS_DIR=$(cd "$RESULTS_DIR" && pwd)

if [ "$CLEANUP_RESULTS" = true ]; then
    trap "rm -rf $RESULTS_DIR" EXIT
fi

echo "Results directory: $RESULTS_DIR"

echo "=========================================="
echo "  FSDP Mixin Dense Tests"
echo "  (${#TEST_ENTRIES[@]} models, ${#TEST_METHODS[@]} test method(s) each)"
echo "  (Max $MAX_CONCURRENT concurrent models across $NUM_GPU_PAIRS GPU pairs)"
echo "=========================================="
echo ""

# ── Run a single model's tests on a GPU pair ───────────────────────────────
run_test() {
    local entry=$1
    local gpu_pair_idx=$2
    local model_name
    model_name=$(extract_model_name "$entry")

    local gpu_start=$((gpu_pair_idx * GPUS_PER_TEST))
    local gpu_end=$((gpu_start + GPUS_PER_TEST - 1))
    local gpu_list=""
    for ((g=gpu_start; g<=gpu_end; g++)); do
        [ -n "$gpu_list" ] && gpu_list+=","
        gpu_list+="$g"
    done

    local test_ids=()
    for test_method in "${TEST_METHODS[@]}"; do
        test_ids+=("${entry}::${test_method}")
    done

    echo -e "${YELLOW}[GPUs ${gpu_list}] Starting: ${model_name}${NC}"

    local log_file="$RESULTS_DIR/${model_name}.log"
    local result_file="$RESULTS_DIR/${model_name}.result"

    CUDA_VISIBLE_DEVICES=$gpu_list \
        python -m pytest -v -rs "${test_ids[@]}" \
        > "$log_file" 2>&1

    local exit_code=$?

    local skipped_only=false
    if [ "$exit_code" -eq 5 ]; then
        skipped_only=true
    elif [ "$exit_code" -eq 0 ] && [ -f "$log_file" ]; then
        if grep -q "passed" "$log_file"; then
            skipped_only=false
        elif grep -q "skipped" "$log_file"; then
            skipped_only=true
        elif grep -q "deselected" "$log_file" && ! grep -q "passed" "$log_file"; then
            skipped_only=true
        fi
    fi

    if [ "$skipped_only" = true ]; then
        echo "SKIPPED" > "$result_file"
    elif [ "$exit_code" -eq 0 ]; then
        echo "SUCCESS" > "$result_file"
    else
        echo "FAILED (exit code: $exit_code)" > "$result_file"
    fi

    if [ "$skipped_only" = true ]; then
        echo -e "${GREY}[GPUs ${gpu_list}] Skipped: ${model_name}${NC}"
    elif [ "$exit_code" -eq 0 ]; then
        echo -e "${GREEN}[GPUs ${gpu_list}] Success: ${model_name}${NC}"
    else
        echo -e "${RED}[GPUs ${gpu_list}] Failed: ${model_name}${NC}"
    fi
}

# ── Debug mode (single model with debugpy) ──────────────────────────────────
if [ -n "$DEBUG_MODE" ]; then
    entry="${TEST_ENTRIES[0]}"
    model_name=$(extract_model_name "$entry")
    test_ids=()
    for test_method in "${TEST_METHODS[@]}"; do
        test_ids+=("${entry}::${test_method}")
    done
    DEBUGPY_PORT=${DEBUGPY_PORT:-5678}
    echo -e "${YELLOW}Debug mode: launching ${model_name} with debugpy on port ${DEBUGPY_PORT}${NC}"
    echo -e "${YELLOW}Attach your debugger (VS Code / Cursor) to localhost:${DEBUGPY_PORT}, then the test will proceed.${NC}"
    echo ""
    CUDA_VISIBLE_DEVICES=0,1 \
        python -m debugpy --listen 0.0.0.0:${DEBUGPY_PORT} --wait-for-client \
        -m pytest -v -rs -s "${test_ids[@]}"
    exit $?
fi

# ── Concurrency-limited dispatch ─────────────────────────────────────────────
# Multiple models can share GPU pairs since test models are tiny (~7 MiB each).
# GPU pairs are assigned round-robin; concurrency is throttled by MAX_CONCURRENT.

active_jobs() {
    jobs -r 2>/dev/null | wc -l
}

model_idx=0
for entry in "${TEST_ENTRIES[@]}"; do
    while [ "$(active_jobs)" -ge "$MAX_CONCURRENT" ]; do
        sleep 1
    done
    gpu_pair_idx=$((model_idx % NUM_GPU_PAIRS))
    run_test "$entry" "$gpu_pair_idx" &
    ((model_idx++))
done

echo ""
echo "Waiting for all tests to complete..."
wait

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
echo ""

success_count=0
fail_count=0
skip_count=0

for entry in "${TEST_ENTRIES[@]}"; do
    model_name=$(extract_model_name "$entry")
    result_file="$RESULTS_DIR/${model_name}.result"
    if [ -f "$result_file" ]; then
        result=$(cat "$result_file")
        if [[ "$result" == "SUCCESS" ]]; then
            echo -e "${GREEN}✓ ${model_name}: ${result}${NC}"
            ((success_count++))
        elif [[ "$result" == "SKIPPED" ]]; then
            echo -e "${GREY}○ ${model_name}: ${result}${NC}"
            ((skip_count++))
        else
            echo -e "${RED}✗ ${model_name}: ${result}${NC}"
            echo -e "${DIM}  Error snippet:"
            tail -n 5 "$RESULTS_DIR/${model_name}.log" | while read -r line; do echo -e "    ${DIM}${line}${NC}"; done
            ((fail_count++))
        fi
    else
        echo -e "${RED}✗ ${model_name}: NO RESULT (test may have crashed)${NC}"
        ((fail_count++))
    fi
done

echo ""
echo "-------------------------------------------"
echo -e "Total: ${GREEN}${success_count} passed${NC}, ${GREY}${skip_count} skipped${NC}, ${RED}${fail_count} failed${NC}"
echo "=========================================="

if [ $fail_count -gt 0 ]; then
    echo ""
    echo "Failed test logs (full paths):"
    for entry in "${TEST_ENTRIES[@]}"; do
        model_name=$(extract_model_name "$entry")
        result_file="$RESULTS_DIR/${model_name}.result"
        if [ -f "$result_file" ] && [ "$(cat "$result_file")" != "SUCCESS" ] && [ "$(cat "$result_file")" != "SKIPPED" ]; then
            echo "  $RESULTS_DIR/${model_name}.log"
        fi
    done
fi

SCRIPT_END=$(date +%s)
ELAPSED=$((SCRIPT_END - SCRIPT_START))
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))
echo ""
echo "Total time: ${MINS}m ${SECS}s"

if [ $fail_count -gt 0 ]; then
    exit 1
fi
