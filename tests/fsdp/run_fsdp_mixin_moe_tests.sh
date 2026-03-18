#!/bin/bash

# Script to run all FSDP mixin tests for MoE models in parallel.
# Work in tandem with a special test_fsdp_mixin.py that batches all distributed tests in a single mp.spawn. (will not be committed)
# Uses concurrency-limited dispatch: multiple models share GPU pairs since test models are tiny (~7 MiB).
# Each model runs test_fsdp2_all which batches all distributed tests in a single mp.spawn.

# Usage: ./run_fsdp_mixin_moe_tests.sh [/path/to/results]
#        ./run_fsdp_mixin_moe_tests.sh --model <model_name> [/path/to/results]
#        ./run_fsdp_mixin_moe_tests.sh --test <test_method> [/path/to/results]
#        ./run_fsdp_mixin_moe_tests.sh --model <model_name> --test <test_method> [/path/to/results]
#        ./run_fsdp_mixin_moe_tests.sh --report /path/to/results
#        ./run_fsdp_mixin_moe_tests.sh --debug --model <model_name>
#        ./run_fsdp_mixin_moe_tests.sh --rerun-failed /path/to/results

SCRIPT_START=$(date +%s)

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREY='\033[0;90m'
DIM='\033[0;90m'
NC='\033[0m'

GPUS_PER_TEST=2

# Batched test method that runs all distributed FSDP tests in a single mp.spawn.
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

# MoE models that inherit from CausalLMModelTest
# Ranked by HuggingFace Hub downloads (30-day, as of 2026-03-10):
#   1. gpt_oss              7.4M   openai/gpt-oss-20b
#   2. glm_moe_dsa          4.0M   zai-org/GLM-5-FP8
#   3. qwen3_moe            2.2M   Qwen/Qwen3-30B-A3B-Instruct-2507
#   4. glm4_moe_lite        1.7M   zai-org/GLM-4.7-Flash
#   5. qwen3_5_moe          1.5M   Qwen/Qwen3.5-397B-A17B
#   6. deepseek_v2          1.3M   deepseek-ai/DeepSeek-V3
#   7. qwen3_next           1.2M   Qwen/Qwen3-Coder-Next
#   8. mixtral              764K   mistralai/Mixtral-8x7B-Instruct-v0.1
#   9. qwen2_moe            103K   Qwen/Qwen1.5-MoE-A2.7B
#  10. phimoe                89K   microsoft/Phi-3.5-MoE-instruct
#  11. glm4_moe              73K   zai-org/GLM-4.6
#  12. minimax_m2            63K   MiniMaxAI/MiniMax-M1-80k
#  13. lfm2_moe              50K   LiquidAI/LFM2-8B-A1B
#  14. longcat_flash         32K   meituan-longcat/LongCat-Flash-Chat
#  15. exaone_moe            26K   LGAI-EXAONE/K-EXAONE-236B-A23B
#  16. ernie4_5_moe          23K   baidu/ERNIE-4.5-21B-A3B-PT
#  17. hunyuan_v1_moe        16K   tencent/Hunyuan-A13B-Instruct
#  18. solar_open           6.4K   upstage/Solar-Open-100B
#  19. dots1                5.3K   rednote-hilab/dots.llm1.inst
#  20. dbrx                   —    databricks/dbrx-instruct (gated)
#  21. afmoe                2.7K   arcee-ai/Trinity-Mini
#  22. jetmoe               2.6K   jetmoe/jetmoe-8b
#  23. flex_olmo              —    allenai/FlexOlmo (gated)
#  24. minimax              1.2K   MiniMaxAI/MiniMax-Text-01

# Entries ordered by download rank. Top 7 are active; rest are commented out.
TEST_ENTRIES=(
    # --- Top 7 (active) ---
    "tests/models/gpt_oss/test_modeling_gpt_oss.py::GptOssModelTest"                                                #  1. gpt_oss              7.4M
    "tests/models/glm_moe_dsa/test_modeling_glm_moe_dsa.py::GlmMoeDsaModelTest"                                     #  2. glm_moe_dsa          4.0M
    "tests/models/qwen3_moe/test_modeling_qwen3_moe.py::Qwen3MoeModelTest"                                          #  3. qwen3_moe            2.2M
    "tests/models/glm4_moe_lite/test_modeling_glm4_moe_lite.py::Glm4MoeModelTest"                                   #  4. glm4_moe_lite        1.7M
    "tests/models/qwen3_5_moe/test_modeling_qwen3_5_moe.py::Qwen3_5MoeTextModelTest"                                #  5. qwen3_5_moe          1.5M
    "tests/models/deepseek_v2/test_modeling_deepseek_v2.py::DeepseekV2ModelTest"                                     #  6. deepseek_v2          1.3M
    "tests/models/qwen3_next/test_modeling_qwen3_next.py::Qwen3NextModelTest"                                        #  7. qwen3_next           1.2M
    # --- Remaining (commented out, by download rank) ---
    "tests/models/mixtral/test_modeling_mixtral.py::MixtralModelTest"                                                #  8. mixtral              764K
    "tests/models/qwen2_moe/test_modeling_qwen2_moe.py::Qwen2MoeModelTest"                                          #  9. qwen2_moe            103K
    "tests/models/phimoe/test_modeling_phimoe.py::PhimoeModelTest"                                                   # 10. phimoe                89K
    # "tests/models/glm4_moe/test_modeling_glm4_moe.py::Glm4MoeModelTest"                                           # 11. glm4_moe              73K
    # "tests/models/minimax_m2/test_modeling_minimax_m2.py::MiniMaxM2ModelTest"                                      # 12. minimax_m2            63K
    # "tests/models/lfm2_moe/test_modeling_lfm2_moe.py::Lfm2MoeModelTest"                                           # 13. lfm2_moe              50K
    # "tests/models/longcat_flash/test_modeling_longcat_flash.py::LongcatFlashModelTest"                             # 14. longcat_flash         32K
    # "tests/models/exaone_moe/test_modeling_exaone_moe.py::ExaoneMoeModelTest"                                      # 15. exaone_moe            26K
    # "tests/models/ernie4_5_moe/test_modeling_ernie4_5_moe.py::Ernie4_5_MoeModelTest"                              # 16. ernie4_5_moe          23K
    # "tests/models/hunyuan_v1_moe/test_modeling_hunyuan_v1_moe.py::HunYuanMoEV1ModelTest"                          # 17. hunyuan_v1_moe        16K
    # "tests/models/solar_open/test_modeling_solar_open.py::SolarOpenModelTest"                                      # 18. solar_open           6.4K
    # "tests/models/dots1/test_modeling_dots1.py::Dots1ModelTest"                                                    # 19. dots1                5.3K
    # "tests/models/dbrx/test_modeling_dbrx.py::DbrxModelTest"                                                       # 20. dbrx                   —
    # "tests/models/afmoe/test_modeling_afmoe.py::AfmoeModelTest"                                                    # 21. afmoe                2.7K
    # "tests/models/jetmoe/test_modeling_jetmoe.py::JetMoeModelTest"                                                 # 22. jetmoe               2.6K
    # "tests/models/flex_olmo/test_modeling_flex_olmo.py::FlexOlmoModelTest"                                         # 23. flex_olmo              —
    # "tests/models/minimax/test_modeling_minimax.py::MiniMaxModelTest"                                              # 24. minimax              1.2K
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
    echo "  FSDP Mixin MoE Test Report"
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
echo "  FSDP Mixin MoE Tests"
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
        python -m pytest -v -rs -o log_cli=true -o log_cli_level=INFO \
        --log-disable=httpx --log-disable=httpcore --log-disable=huggingface_hub --log-disable=urllib3 \
        "${test_ids[@]}" \
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
        -m pytest -v -rs -s -o log_cli=true -o log_cli_level=INFO \
        --log-disable=httpx --log-disable=httpcore --log-disable=huggingface_hub --log-disable=urllib3 \
        "${test_ids[@]}"
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
