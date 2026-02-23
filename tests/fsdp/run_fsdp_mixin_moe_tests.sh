#!/bin/bash

# Script to run all FSDP mixin tests for MoE models.
# Usage: ./run_fsdp_mixin_moe_tests.sh [/path/to/results]
#        ./run_fsdp_mixin_moe_tests.sh --model <model_name> [/path/to/results]
#        ./run_fsdp_mixin_moe_tests.sh --report /path/to/results

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREY='\033[0;90m'
DIM='\033[0;90m'
NC='\033[0m'

# All FSDP mixin test methods
TEST_METHODS=(
    "test_get_transformer_block_classes"
    "test_fsdp2_sharding_structure_untied"
    "test_fsdp2_sharding_structure_tied"
    "test_fsdp2_auto_plan_vs_ddp_float32_untied"
    "test_fsdp2_auto_plan_vs_ddp_bfloat16_untied"
    "test_fsdp2_auto_plan_vs_ddp_float32_tied"
    "test_fsdp2_auto_plan_vs_ddp_bfloat16_tied"
    "test_fsdp2_manual_plan_vs_ddp_float32_untied"
    "test_fsdp2_manual_plan_vs_ddp_bfloat16_untied"
    "test_fsdp2_manual_plan_vs_ddp_float32_tied"
    "test_fsdp2_manual_plan_vs_ddp_bfloat16_tied"
    "test_fsdp2_save_load"
)

# MoE models that inherit from CausalLMModelTest
TEST_ENTRIES=(
    "tests/models/afmoe/test_modeling_afmoe.py::AfmoeModelTest"
    "tests/models/dbrx/test_modeling_dbrx.py::DbrxModelTest"
    "tests/models/deepseek_v2/test_modeling_deepseek_v2.py::DeepseekV2ModelTest"
    "tests/models/dots1/test_modeling_dots1.py::Dots1ModelTest"
    "tests/models/ernie4_5_moe/test_modeling_ernie4_5_moe.py::Ernie4_5_MoeModelTest"
    "tests/models/exaone_moe/test_modeling_exaone_moe.py::ExaoneMoeModelTest"
    "tests/models/flex_olmo/test_modeling_flex_olmo.py::FlexOlmoModelTest"
    "tests/models/glm4_moe/test_modeling_glm4_moe.py::Glm4MoeModelTest"
    "tests/models/glm4_moe_lite/test_modeling_glm4_moe_lite.py::Glm4MoeModelTest"
    "tests/models/glm_moe_dsa/test_modeling_glm_moe_dsa.py::GlmMoeDsaModelTest"
    "tests/models/gpt_oss/test_modeling_gpt_oss.py::GptOssModelTest"
    "tests/models/hunyuan_v1_moe/test_modeling_hunyuan_v1_moe.py::HunYuanMoEV1ModelTest"
    "tests/models/jetmoe/test_modeling_jetmoe.py::JetMoeModelTest"
    "tests/models/lfm2_moe/test_modeling_lfm2_moe.py::Lfm2MoeModelTest"
    "tests/models/longcat_flash/test_modeling_longcat_flash.py::LongcatFlashModelTest"
    "tests/models/minimax/test_modeling_minimax.py::MiniMaxModelTest"
    "tests/models/minimax_m2/test_modeling_minimax_m2.py::MiniMaxM2ModelTest"
    "tests/models/mixtral/test_modeling_mixtral.py::MixtralModelTest"
    "tests/models/phimoe/test_modeling_phimoe.py::PhimoeModelTest"
    "tests/models/qwen2_moe/test_modeling_qwen2_moe.py::Qwen2MoeModelTest"
    "tests/models/qwen3_moe/test_modeling_qwen3_moe.py::Qwen3MoeModelTest"
    "tests/models/qwen3_5_moe/test_modeling_qwen3_5_moe.py::Qwen3_5MoeTextModelTest"
    "tests/models/qwen3_next/test_modeling_qwen3_next.py::Qwen3NextModelTest"
    "tests/models/solar_open/test_modeling_solar_open.py::SolarOpenModelTest"
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
        fi
    done

    echo ""
    echo "-------------------------------------------"
    echo -e "Total: ${GREEN}${success_count} passed${NC}, ${GREY}${skip_count} skipped${NC}, ${RED}${fail_count} failed${NC}"
    echo "=========================================="

    if [ $fail_count -gt 0 ]; then
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

# ── Results directory ───────────────────────────────────────────────────────
if [ -n "$1" ]; then
    RESULTS_DIR="$1"
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

echo "=========================================="
echo "  FSDP Mixin MoE Tests"
echo "  (${#TEST_ENTRIES[@]} models, ${#TEST_METHODS[@]} tests each)"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo ""

# ── Run tests ───────────────────────────────────────────────────────────────
success_count=0
fail_count=0
skip_count=0

for entry in "${TEST_ENTRIES[@]}"; do
    model_name=$(extract_model_name "$entry")

    # Build pytest node IDs for all test methods
    test_ids=()
    for test_method in "${TEST_METHODS[@]}"; do
        test_ids+=("${entry}::${test_method}")
    done

    # Single pytest invocation per model -> one log file
    python -m pytest -vs "${test_ids[@]}" \
        > "$RESULTS_DIR/${model_name}.log" 2>&1
    exit_code=$?

    if [ $exit_code -eq 5 ]; then
        echo "SKIPPED" > "$RESULTS_DIR/${model_name}.result"
        echo -e "${GREY}○ ${model_name}: SKIPPED${NC}"
        ((skip_count++))
    elif [ $exit_code -eq 0 ]; then
        if grep -q "passed" "$RESULTS_DIR/${model_name}.log"; then
            echo "SUCCESS" > "$RESULTS_DIR/${model_name}.result"
            echo -e "${GREEN}✓ ${model_name}${NC}"
            ((success_count++))
        else
            echo "SKIPPED" > "$RESULTS_DIR/${model_name}.result"
            echo -e "${GREY}○ ${model_name}: SKIPPED${NC}"
            ((skip_count++))
        fi
    else
        echo "FAILED (exit code: $exit_code)" > "$RESULTS_DIR/${model_name}.result"
        echo -e "${RED}✗ ${model_name}: FAILED${NC}"
        echo -e "${DIM}  $(tail -n 3 "$RESULTS_DIR/${model_name}.log")${NC}"
        ((fail_count++))
    fi
done

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
echo ""
echo -e "Total: ${GREEN}${success_count} passed${NC}, ${GREY}${skip_count} skipped${NC}, ${RED}${fail_count} failed${NC}"
echo "=========================================="

if [ $fail_count -gt 0 ]; then
    echo ""
    echo "Failed test logs:"
    for entry in "${TEST_ENTRIES[@]}"; do
        model_name=$(extract_model_name "$entry")
        result_file="$RESULTS_DIR/${model_name}.result"
        if [ -f "$result_file" ] && [[ "$(cat "$result_file")" != "SUCCESS" ]] && [[ "$(cat "$result_file")" != "SKIPPED" ]]; then
            echo "  $RESULTS_DIR/${model_name}.log"
        fi
    done
    exit 1
fi
